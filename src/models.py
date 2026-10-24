"""
models.py

Input: integer tokens, shape (B, 3) - each row is [a, b, =]
Output: logit (B, 113)

"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src import config

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, causal):
        # Mandatory first line to start nn.Module's bookkeeping to find every learned tensor inside the obeject
        # d_model: length of token vector
        # num_heads: parallel attention computation per layer
        super().__init__()

        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.causal = causal

        self.W_Q = nn.Linear(d_model, d_model, bias=False) 
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        # d_model = the input, d_model = the output, bias = additive component
        # nn.Linear creates a linear transformation layer that takes d_model in and outputs d_model after multiplied by the weighs before splitting into heads.

        mask = torch.tril(torch.ones(config.SEQ_LEN, config.SEQ_LEN))
        self.register_buffer("mask", mask)
        # torch.ones makes a 3x3 of ones and torch.tril makes it a bottom triangle.
        # For the 3 x 3 attention matrix the zeros at the top represent future relationships which is "cheating"
        # The causal mask:
        #   [[1, 0, 0],
        #    [1, 1, 0],
        #    [1, 1, 1]]

        self.last_attention = None

    def forward(self, x):
        B, n, d_model = x.shape

        q = self.W_Q(x).reshape(B, n, self.num_heads, self.d_head).transpose(1, 2)
        k = self.W_K(x).reshape(B, n, self.num_heads, self.d_head).transpose(1, 2)
        v = self.W_V(x).reshape(B, n, self.num_heads, self.d_head).transpose(1, 2)
        # the reshape().transpose(1,2) switches axes 1 and 2:
        # (b, num_heads, n, d_head)
        # pyTorch matrix multiplication treats all but last two as batch dimensions, and we want to multiply the (n, d_head) matrix

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)

        if self.causal:
            scores = scores.masked_fill(self.mask[:n, :n] == 0, float("-inf"))
            # the indexes of the scores that = 0 turn into "-inf"
            
        attn = F.softmax(scores, dim=-1)
        # softmax the last axis "dim=-1" (B, H, n, n)

        self.last_attention = attn.detach()
        # .detach() keeps a copy that autograd ignores, costs no gradient memory

        out = attn @ v 
        # (B, H, n, n) @ (B, H, n, dh) -> (B, H, n, dh)

        out = out.transpose(1, 2).reshape(B, n, self.num_heads * self.d_head)
        # (B, H, n, dh) -> transpose(1, 2) -> (B, n, H, dh) -> reshape -> (B, n, d)
        # We want to multiply the number of heads with the d_head to get the total d(width)

        return self.W_O(out)
        
class MLP(nn.Module):

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.W_1 = nn.Linear(d_model, d_ff)
        self.W_2 = nn.Linear(d_ff, d_model)
        # W1 turns d_model into d_ff in order to give the model more space to computer more complex non-linear patterns
        # W2 turns d_ff back into d_model
    
    def forward(self, x):
        return self.W_2(F.relu(self.W_1(x)))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, causal):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads, causal)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        # Adding to the residual stream
        return x
        # (B, n , d)
    
class ModularAdditionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embed = nn.Embedding(config.VOCAB_SIZE, config.EMBEDDING_DIM)
        self.pos_embed = nn.Embedding(config.SEQ_LEN, config.EMBEDDING_DIM)
        # Embeds each position in SEQ_LEN as a length EMBEDDING_DIM vector.
        # config.EMBEDDING_DIM = d_model

        self.blocks = nn.ModuleList([
            TransformerBlock(config.EMBEDDING_DIM, config.NUM_HEADS, config.D_FF, config.USE_CAUSAL_MASK)
            for _ in range(config.NUM_LAYERS)
        ])
        # nn.ModuleList stores a list of TransformerBlocks, one for each layer

        self.unembed = nn.Linear(config.EMBEDDING_DIM, config.NUM_CLASSES)
    
    def forward(self, tokens):
        B, n = tokens.size()

        positions = torch.arange(n, device=tokens.device)
        # computations must be on the same device

        x = self.pos_embed(positions) + self.token_embed(tokens)
        # creating the x vector that contains position and token data


        for block in self.blocks:
            x = block(x)
            # We are passing the x(which was passed through the previous layer) tensor through the block(which is a TransformerBlock object).
            # We don't add to the previous beacuse we add after all the layers run.
        
        x = x[:, -1, :]
        # Since x is (B, n, d), it keeps B and d, but only takes the last n.
        # Since we are doing modular addition, the = sign, i.e the final token carries all the context of the math problem

        return self.unembed(x)
        # (b, d) -> (B, 113)
        # Cross-Entropy loss is calculated internally.

if __name__ == "__main__":
    # dataset.py defines get_dataloaders(), which returns (train_loader, test_loader):
    # objects that hand out batches of (tokens, labels) with shapes (512, 3) and (512,).
    from src.dataset import get_dataloaders
 
    model = ModularAdditionTransformer().to(config.DEVICE)
    print(model)
 
    # 1. Parameter count must match the formula we derived by hand.
    d, d_ff, L = config.EMBEDDING_DIM, config.D_FF, config.NUM_LAYERS
    expected = (
        config.VOCAB_SIZE * d                                   # token embedding
        + config.SEQ_LEN * d                                    # position embedding
        + L * (4 * d * d + 2 * d * d_ff + d_ff + d)             # per layer: attention + MLP
        + d * config.NUM_CLASSES + config.NUM_CLASSES           # unembedding + its bias
    )
    total = sum(p.numel() for p in model.parameters())
    print(f"\nParameters: {total:,}  (hand-computed: {expected:,})")
    assert total == expected, "Parameter count disagrees with the hand calculation!"
 
    # 2. Dummy forward pass: random token indices, correct shape.
    dummy = torch.randint(0, config.VOCAB_SIZE, (8, config.SEQ_LEN)).to(config.DEVICE)
    logits = model(dummy)
    print(f"Dummy input {tuple(dummy.shape)} -> logits {tuple(logits.shape)}")
    assert logits.shape == (8, config.NUM_CLASSES)
 
    # 3. Attention patterns: right shape, rows sum to 1, and (if causal)
    # nothing above the diagonal.
    attn = model.blocks[0].attn.last_attention
    print(f"Attention pattern shape: {tuple(attn.shape)}  (expected (8, {config.NUM_HEADS}, 3, 3))")
    row_sums = attn.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
    if config.USE_CAUSAL_MASK:
        assert torch.triu(attn, diagonal=1).abs().max().item() == 0.0
        print("Causal mask check passed: no attention to later positions.")
 
    # 4. Real batch: the starting loss should be near ln(113) = 4.73 (random guessing).
    train_loader, _ = get_dataloaders()
    tokens, labels = next(iter(train_loader))
    tokens, labels = tokens.to(config.DEVICE), labels.to(config.DEVICE)
    loss = F.cross_entropy(model(tokens), labels)
    print(f"\nStarting loss: {loss.item():.3f}  (random guessing = {math.log(config.NUM_CLASSES):.3f})")
    assert 4.0 < loss.item() < 7.0
 
    # 5. Every learned tensor must receive a gradient.
    loss.backward()
    print("\nGradient norm for every learned tensor:")
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} received no gradient!"
        print(f"  {name:32s} {str(tuple(p.shape)):16s} {p.grad.norm().item():.4f}")
 
    print("\nAll checks passed.") 













