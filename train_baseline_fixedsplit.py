import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from torch.optim import AdamW

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "EuroSAT_RGB")
SPLIT_FILE = os.path.join(BASE_DIR, "eurosat_split_indices.pt")

BATCH_SIZE = 64
SEED = 42

EPOCHS_HEAD = 3
LR_HEAD = 1e-3

EPOCHS_FT = 5
LR_FT = 1e-4

torch.manual_seed(SEED)

device = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

split = torch.load(SPLIT_FILE)
train_idx, val_idx, test_idx = split["train"], split["val"], split["test"]

train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

test_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

full_train = datasets.ImageFolder(DATA_DIR, transform=train_tf)
full_eval = datasets.ImageFolder(DATA_DIR, transform=test_tf)

train_ds = Subset(full_train, train_idx)
val_ds = Subset(full_eval, val_idx)
test_ds = Subset(full_eval, test_idx)

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 10)
model.to(device)

criterion = nn.CrossEntropyLoss()


@torch.no_grad()
def eval_acc(loader):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(1)
        correct += (preds == y).sum().item()
        total += x.size(0)
    return correct / total


def train_epochs(epochs, optimizer):
    best_val = 0.0
    best_state = None

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        val_acc = eval_acc(val_loader)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        print(
            f"Epoch {ep:02d}/{epochs} | val acc {val_acc:.4f} | time {time.time()-t0:.1f}s")

    model.load_state_dict(best_state)
    model.to(device)
    return best_val


for p in model.parameters():
    p.requires_grad = False
for p in model.fc.parameters():
    p.requires_grad = True

opt_head = AdamW(model.fc.parameters(), lr=LR_HEAD, weight_decay=1e-4)
print("\nStage A: training FC only")
best_val_head = train_epochs(EPOCHS_HEAD, opt_head)

if EPOCHS_FT > 0:
    for p in model.parameters():
        p.requires_grad = True

    opt_ft = AdamW(filter(lambda p: p.requires_grad,
                   model.parameters()), lr=LR_FT, weight_decay=1e-4)
    print("\nStage B: fine-tuning (unfrozen)")
    best_val_ft = train_epochs(EPOCHS_FT, opt_ft)
else:
    best_val_ft = best_val_head

test_acc = eval_acc(test_loader)
print("\nFinal best val acc:", best_val_ft)
print("Test acc:", test_acc)

save_path = os.path.join(BASE_DIR, "eurosat_resnet18_baseline_fixedsplit.pth")
torch.save({k: v.cpu() for k, v in model.state_dict().items()}, save_path)
print("Saved:", save_path)
