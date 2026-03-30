import os
import torch
import torch.nn as nn
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# =========================
# CONFIG
# =========================
BASE_PATH    = "/content/drive/MyDrive/Thesis/data/raw"
MODEL_PATH   = "/content/Thesis/src/training/best_model_resnet.pt"

BATCH_SIZE   = 32
LR           = 5e-4
EPOCHS       = 30
TARGET_SR    = 2000
SEGMENT_SEC  = 3
PATIENCE     = 7
NUM_WORKERS  = 2
SEED         = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# =========================
# REPRODUCIBILITY
# =========================
torch.manual_seed(SEED)
np.random.seed(SEED)
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(SEED)

# =========================
# LOAD CSV
# =========================
csv_path = os.path.join(BASE_PATH, "REFERENCES.csv")
df = pd.read_csv(csv_path)

file_paths, labels = [], []

for _, row in df.iterrows():
    path = os.path.join(BASE_PATH, row["folder"], row["record_id"] + ".wav")
    if os.path.exists(path):
        file_paths.append(path)
        labels.append(int(row["label"]))

print(f"Total samples found : {len(file_paths)}")
print(f"  Positive (abnormal): {sum(labels)}")
print(f"  Negative (normal)  : {len(labels) - sum(labels)}")

assert len(file_paths) > 0, "No audio files found. Check BASE_PATH and CSV."
assert sum(labels) > 0,     "No positive samples found. Check label column."

# =========================
# SPLIT  (70 / 15 / 15)
# =========================
temp_paths, test_paths, temp_labels, test_labels = train_test_split(
    file_paths, labels,
    test_size=0.15,
    stratify=labels,
    random_state=SEED
)

train_paths, val_paths, train_labels, val_labels = train_test_split(
    temp_paths, temp_labels,
    test_size=0.1765,
    stratify=temp_labels,
    random_state=SEED
)

print(f"\nSplit — Train: {len(train_paths)} | Val: {len(val_paths)} | Test: {len(test_paths)}")
print(f"  Train pos/neg : {sum(train_labels)} / {len(train_labels)-sum(train_labels)}")
print(f"  Val   pos/neg : {sum(val_labels)}   / {len(val_labels)-sum(val_labels)}")
print(f"  Test  pos/neg : {sum(test_labels)}  / {len(test_labels)-sum(test_labels)}")

# =========================
# DATASET
# =========================
class PCGDataset(Dataset):
    def __init__(self, paths, labels, augment=False):
        self.paths   = paths
        self.labels  = labels
        self.augment = augment
        self.segment_samples = TARGET_SR * SEGMENT_SEC
        self._resamplers = {}

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        waveform, sr = torchaudio.load(self.paths[idx])

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != TARGET_SR:
            if sr not in self._resamplers:
                self._resamplers[sr] = torchaudio.transforms.Resample(sr, TARGET_SR)
            waveform = self._resamplers[sr](waveform)

        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform / peak

        waveform = self._fix_length(waveform)

        if self.augment:
            waveform = self._augment(waveform)

        return waveform, torch.tensor(self.labels[idx], dtype=torch.float32)

    def _fix_length(self, x):
        length = x.shape[1]
        if length > self.segment_samples:
            start = torch.randint(0, length - self.segment_samples, (1,)).item()
            x = x[:, start : start + self.segment_samples]
        elif length < self.segment_samples:
            x = torch.nn.functional.pad(x, (0, self.segment_samples - length))
        return x

    def _augment(self, x):
        if torch.rand(1).item() < 0.5:
            x = x + torch.randn_like(x) * 0.005
        if torch.rand(1).item() < 0.5:
            scale = torch.empty(1).uniform_(0.8, 1.2).item()
            x = (x * scale).clamp(-1, 1)
        return x

# =========================
# DATALOADERS
# =========================
train_ds = PCGDataset(train_paths, train_labels, augment=True)
val_ds   = PCGDataset(val_paths,   val_labels,   augment=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"))
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"))

# =========================
# MODEL  (1D ResNet)
# =========================

class ResBlock(nn.Module):
    """
    Two-layer residual block with optional downsampling.

    Input → Conv → BN → ReLU → Conv → BN → (+skip) → ReLU

    If in_channels != out_channels or stride > 1, the skip connection
    uses a 1×1 conv to match dimensions.
    """
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=7,
                               stride=stride, padding=3, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=7,
                               padding=3, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)

        self.relu  = nn.ReLU(inplace=True)

        # skip connection — only needed when shape changes
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(out_ch)
            )

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.relu(out + identity)   # ← residual addition
        return out


class ResNet1D(nn.Module):
    """
    Lightweight 1D ResNet:
      Stem → Stage1 → Stage2 → Stage3 → Stage4 → GAP → FC
    """
    def __init__(self):
        super().__init__()

        # stem — initial feature extraction
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        # residual stages (each doubles channels, halves time via stride=2)
        self.stage1 = ResBlock(32,  64,  stride=2)
        self.stage2 = ResBlock(64,  128, stride=2)
        self.stage3 = ResBlock(128, 256, stride=2)
        self.stage4 = ResBlock(256, 256, stride=2)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        x = self.fc(x)
        return x


model = ResNet1D().to(DEVICE)
print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

# =========================
# LOSS + OPTIMIZER + SCHEDULER
# =========================
n_pos = sum(train_labels)
n_neg = len(train_labels) - n_pos
pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(DEVICE)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3
)

# =========================
# HELPERS
# =========================
def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE).unsqueeze(1)

            if train:
                optimizer.zero_grad()

            out  = model(x)
            loss = criterion(out, y)

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            probs = torch.sigmoid(out).detach().cpu().squeeze().numpy()
            labs  = y.detach().cpu().squeeze().numpy()
            all_preds.extend(np.atleast_1d(probs).tolist())
            all_labels.extend(np.atleast_1d(labs).tolist())

    avg_loss  = total_loss / len(loader)
    preds_bin = [1 if p > 0.5 else 0 for p in all_preds]
    acc       = sum(p == l for p, l in zip(preds_bin, all_labels)) / len(all_labels)

    try:
        auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        auc = float("nan")

    return avg_loss, acc, auc

# =========================
# TRAIN LOOP
# =========================
best_val_loss  = float("inf")
patience_count = 0
history        = []

print("\n" + "="*60)
for epoch in range(1, EPOCHS + 1):
    print(f"\nEpoch {epoch}/{EPOCHS}")

    train_loss, train_acc, train_auc = run_epoch(train_loader, train=True)
    val_loss,   val_acc,   val_auc   = run_epoch(val_loader,   train=False)

    scheduler.step(val_loss)

    history.append({
        "epoch":      epoch,
        "train_loss": train_loss, "train_acc": train_acc, "train_auc": train_auc,
        "val_loss":   val_loss,   "val_acc":   val_acc,   "val_auc":   val_auc,
    })

    print(f"  Train — Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  AUC: {train_auc:.4f}")
    print(f"  Val   — Loss: {val_loss:.4f}  Acc: {val_acc:.4f}  AUC: {val_auc:.4f}  LR: {optimizer.param_groups[0]['lr']:.2e}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_count = 0
        torch.save(model.state_dict(), MODEL_PATH)
        print("  ✅ Best model saved.")
    else:
        patience_count += 1
        print(f"  ⚠️  No improvement ({patience_count}/{PATIENCE})")
        if patience_count >= PATIENCE:
            print("\n🛑 Early stopping triggered.")
            break

# =========================
# DONE
# =========================
print("\n" + "="*60)
print("Training finished.")
print(f"Best validation loss : {best_val_loss:.4f}")
print(f"Best weights saved to: {MODEL_PATH}")