import torch
from torch.utils.data import Dataset, DataLoader
 
from src import config
 
 
class ModularAdditionDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs # Pairs is a list of (a, b) tuples

    def __len__(self):
        return len(self.pairs)

    def __get_item__(self, idx):
        a, b = self.pairs[idx]
        input_token = torch.tensor([a, b, config.EQUALS_TOKEN], dtype=torch.long)
        label = torch.tensor([a + b % config.P], dtype=torch.long)
        return input_token, label

def get_dataloaders():
    all_pairs = [(a, b) for a in range(config.P) for b in range(config.P)]
    generator = torch.Generator().manual_seed(config.RANDOM_SEED)
    shuffled_indicies = torch.randperm(len(all_pairs), generator=generator).tolist()

    split_point = int(len(all_pairs) * config.TRAIN_FRACTION)

    train_indicies = shuffled_indicies[:split_point]
    test_indicies = shuffled_indicies[split_point:]

    train_pairs = [all_pairs[i] for i in train_indicies]
    test_pairs = [all_pairs[i] for i in test_indicies]

    train_dataset = ModularAdditionDataset(train_pairs)
    test_dataset = ModularAdditionDataset(test_pairs)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
 
    return train_loader, test_loader



if __name__ == "__main__":
    train_loader, test_loader = get_dataloaders()
 
    print(f"Total possible pairs: {config.P * config.P}")
    print(f"Train examples: {len(train_loader.dataset)}  |  Test examples: {len(test_loader.dataset)}")
    print(f"Train batches: {len(train_loader)}  |  Test batches: {len(test_loader)}")
 
    # Pull one batch and verify shapes match what we derived by hand in B1/B2.
    input_tokens, labels = next(iter(train_loader))
    print(f"\nInput tokens batch shape: {input_tokens.shape}   (expected: [batch_size, 3])")
    print(f"Labels batch shape:       {labels.shape}          (expected: [batch_size])")
    print(f"Input dtype: {input_tokens.dtype}  |  Label dtype: {labels.dtype}")
 
    # Inspect one real example directly, and verify the arithmetic by hand.
    example_input = input_tokens[0].tolist()
    example_label = labels[0].item()
    a, b, eq = example_input
    print(f"\nExample: a={a}, b={b}, equals_token={eq} (expected {config.EQUALS_TOKEN})")
    print(f"Label: {example_label}  |  Manually computed (a+b)%P: {(a + b) % config.P}")
    assert example_label == (a + b) % config.P, "Label doesn't match manual computation!"
 
    # Verify the HELD-OUT property: confirm zero overlap between train and
    # test pairs -- this is the property Phase B3 was entirely about, and
    # we should never just assume it worked, same discipline as verifying
    # the label-9 gap in the CNN project.
    train_set = set(train_loader.dataset.pairs)
    test_set = set(test_loader.dataset.pairs)
    overlap = train_set & test_set  # set intersection
    print(f"\nOverlap between train and test pairs: {len(overlap)}  (expected: 0)")
    assert len(overlap) == 0, "Train/test leakage detected!"
 
    print("\n✅ All checks passed.")

