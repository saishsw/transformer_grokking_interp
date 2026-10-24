from pathlib import Path
import torch

# Directories

BASE_DIR = Path(__file__).resolve().parent.parent

CHECKPOINT_DIR = BASE_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Computing Device Heirarchy

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

DEVICE = get_device()

# Task Definition

P = 113
EQUALS_TOKEN = P
VOCAB_SIZE = P + 1
SEQ_LEN = 3
NUM_CLASSES = P

# Hyperperameters

EMBEDDING_DIM = 128 # The orginal grokking paper had 128, but lowering for speed
NUM_HEADS = 4 # Number of parallel attention heads 
NUM_LAYERS = 1 # Number of transformer blocks
D_FF = 4 * EMBEDDING_DIM
USE_CAUSAL_MASK = True

# Training Hyperperameters

RANDOM_SEED = 42
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
NUM_EPOCHS = 500  # grokking typically needs us to train much further than when the model looks "done"
TRAIN_FRACTION = 0.4 # Fraction of all pairs for training


# Sanity check

if __name__ == "__main__":
    print(f"Project root:    {BASE_DIR}")
    print(f"Selected device: {DEVICE}")
    print(f"P (modulus):     {P}")
    print(f"Vocab size:      {VOCAB_SIZE}  (= {P} numbers + 1 equals-token)")
    print(f"Total possible (a,b) pairs: {P * P}")
    print(f"Training on ~{int(P * P * TRAIN_FRACTION)} pairs, "
          f"holding out ~{int(P * P * (1 - TRAIN_FRACTION))} pairs for testing")