import os
import torch
from torchvision import datasets

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "EuroSAT_RGB")
OUT_FILE = os.path.join(BASE_DIR, "eurosat_split_indices.pt")
SEED = 42

ds = datasets.ImageFolder(DATA_DIR)
n_total = len(ds)
n_train = int(0.8 * n_total)
n_val = int(0.1 * n_total)
n_test = n_total - n_train - n_val

g = torch.Generator().manual_seed(SEED)
perm = torch.randperm(n_total, generator=g).tolist()

train_idx = perm[:n_train]
val_idx = perm[n_train:n_train + n_val]
test_idx = perm[n_train + n_val:]

torch.save(
    {"train": train_idx, "val": val_idx, "test": test_idx, "classes": ds.classes},
    OUT_FILE
)

print("Saved split file:", OUT_FILE)
print("Counts:", len(train_idx), len(val_idx), len(test_idx))
print("Classes:", ds.classes)
