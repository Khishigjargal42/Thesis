# ╔══════════════════════════════════════════════════════════════╗
# ║  ResNet2D + SE Attention + Mel-Spectrogram                  ║
# ║  Training — Colab Notebook Version                          ║
# ║  Зохиогч: Г.Хишигжаргал, 2026                              ║
# ╚══════════════════════════════════════════════════════════════╝

# ── CELL 1: Drive mount ──────────────────────────────────────────
from google.colab import drive
drive.mount("/content/drive")

# ── CELL 2: Install ──────────────────────────────────────────────
# !pip install scikit-learn -q  # ихэвчлэн суугаад байдаг

# ── CELL 3: Imports + Config ─────────────────────────────────────
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, roc_auc_score, accuracy_score,
    precision_score, recall_score, confusion_matrix
)

# ── PATHS ────────────────────────────────────────────────────────
DATA_DIR   = "/content/drive/MyDrive/Thesis/data/features"
MODEL_DIR  = "/content/drive/MyDrive/Thesis/models"
SAVE_PATH  = os.path.join(MODEL_DIR, "resnet_mel_attention_v3.pth")
STATS_PATH = os.path.join(MODEL_DIR, "resnet_mel_attention_v3_stats.npz")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── HYPERPARAMETERS ──────────────────────────────────────────────
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED         = 42
BATCH_SIZE   = 64
EPOCHS       = 50
LR           = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE     = 8

torch.manual_seed(SEED)
np.random.seed(SEED)
print(f"Device : {DEVICE}")
print(f"DATA_DIR exists: {os.path.exists(DATA_DIR)}")

# ── CELL 4: Model ────────────────────────────────────────────────
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1    = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1      = nn.BatchNorm2d(out_ch)
        self.conv2    = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2      = nn.BatchNorm2d(out_ch)
        self.relu     = nn.ReLU(inplace=True)
        self.se       = SEBlock(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += identity
        return self.relu(out)


class ResNet2D_SE(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.layer1 = BasicBlock(32, 32)
        self.layer2 = BasicBlock(32, 64, stride=2)
        self.layer3 = BasicBlock(64, 128, stride=2)
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.drop   = nn.Dropout(dropout)
        self.fc     = nn.Linear(128, 1)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.drop(x)
        return self.fc(x)

print("Model тодорхойлогдлоо.")

# ── CELL 5: Data ─────────────────────────────────────────────────
print("Дата ачаалж байна...")
mel = np.load(os.path.join(DATA_DIR, "mel_spectrogram.npy"))
y   = np.load(os.path.join(DATA_DIR, "labels.npy"))

print(f"mel shape : {mel.shape}")
print(f"Normal    : {(y==0).sum()} | Abnormal: {(y==1).sum()}")

# Stratified split 70 / 15 / 15
idx = np.arange(len(y))
idx_tr, idx_tmp = train_test_split(
    idx, test_size=0.30, stratify=y, random_state=SEED
)
idx_val, idx_te = train_test_split(
    idx_tmp, test_size=0.50, stratify=y[idx_tmp], random_state=SEED
)

print(f"\nSplit:")
print(f"  Train : {len(idx_tr)}  (N={(y[idx_tr]==0).sum()}  A={(y[idx_tr]==1).sum()})")
print(f"  Val   : {len(idx_val)}  (N={(y[idx_val]==0).sum()}  A={(y[idx_val]==1).sum()})")
print(f"  Test  : {len(idx_te)}  (N={(y[idx_te]==0).sum()}  A={(y[idx_te]==1).sum()})")

# Normalize — TRAIN stats (data leak үгүй)
TR_MEAN = float(mel[idx_tr].mean())
TR_STD  = float(mel[idx_tr].std())
print(f"\nNorm stats: mean={TR_MEAN:.4f}  std={TR_STD:.4f}")

mel_tr  = (mel[idx_tr]  - TR_MEAN) / (TR_STD + 1e-8)
mel_val = (mel[idx_val] - TR_MEAN) / (TR_STD + 1e-8)
mel_te  = (mel[idx_te]  - TR_MEAN) / (TR_STD + 1e-8)

y_tr  = y[idx_tr]
y_val = y[idx_val]
y_te  = y[idx_te]

# DataLoader
def to_loader(X, y, shuffle=False):
    Xt = torch.FloatTensor(X[:, np.newaxis])
    yt = torch.FloatTensor(y)
    return DataLoader(
        TensorDataset(Xt, yt),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True
    )

train_loader = to_loader(mel_tr,  y_tr,  shuffle=True)
val_loader   = to_loader(mel_val, y_val)
test_loader  = to_loader(mel_te,  y_te)

# pos_weight
n_neg      = int((y_tr == 0).sum())
n_pos      = int((y_tr == 1).sum())
pos_weight = torch.tensor([n_neg / n_pos]).to(DEVICE)
print(f"pos_weight: {pos_weight.item():.4f}")

# Norm stats хадгалах
np.savez(STATS_PATH, mean=TR_MEAN, std=TR_STD)
print(f"Norm stats saved: {STATS_PATH}")

# ── CELL 6: Training ─────────────────────────────────────────────
model     = ResNet2D_SE(dropout=0.3).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(
    model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", patience=3, factor=0.5, verbose=True
)

best_f1      = 0.0
patience_cnt = 0

print(f"Training эхэлж байна... (epochs={EPOCHS}  patience={PATIENCE})")
print("-" * 72)

for epoch in range(1, EPOCHS + 1):

    # Train
    model.train()
    tr_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(xb).squeeze(), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        tr_loss += loss.item()
    tr_loss /= len(train_loader)

    # Validation
    model.eval()
    val_loss   = 0.0
    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits   = model(xb).squeeze()
            val_loss += criterion(logits, yb).item()
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(yb.cpu().numpy())

    val_loss  /= len(val_loader)
    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds      = (all_probs >= 0.5).astype(int)

    f1  = f1_score(all_labels, preds)
    auc = roc_auc_score(all_labels, all_probs)

    print(f"Epoch {epoch:02d} | "
          f"tr_loss {tr_loss:.4f} | "
          f"val_loss {val_loss:.4f} | "
          f"F1 {f1:.4f} | "
          f"AUC {auc:.4f}")

    scheduler.step(f1)

    if f1 > best_f1:
        best_f1      = f1
        patience_cnt = 0
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  [SAVED] best F1: {best_f1:.4f}")
    else:
        patience_cnt += 1
        if patience_cnt >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            break

print(f"\nDone. Best F1: {best_f1:.4f}")
print(f"Model: {SAVE_PATH}")

# ── CELL 7: Test Evaluation ──────────────────────────────────────
print("\n" + "=" * 72)
print("TEST SET EVALUATION")
print("=" * 72)

model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
model.eval()

all_probs  = []
all_labels = []

with torch.no_grad():
    for xb, yb in test_loader:
        xb    = xb.to(DEVICE)
        logits = model(xb).squeeze()
        probs  = torch.sigmoid(logits).cpu().numpy()
        if probs.ndim == 0:
            probs = [float(probs)]
        all_probs.extend(probs)
        all_labels.extend(yb.numpy())

all_probs  = np.array(all_probs)
all_labels = np.array(all_labels)
preds      = (all_probs >= 0.5).astype(int)

acc  = accuracy_score(all_labels, preds)
prec = precision_score(all_labels, preds)
rec  = recall_score(all_labels, preds)
f1   = f1_score(all_labels, preds)
auc  = roc_auc_score(all_labels, all_probs)
cm   = confusion_matrix(all_labels, preds)

print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1-Score  : {f1:.4f}")
print(f"AUC-ROC   : {auc:.4f}")
print(f"\nConfusion Matrix:")
print(f"  TN={cm[0,0]:5d}  FP={cm[0,1]:5d}")
print(f"  FN={cm[1,0]:5d}  TP={cm[1,1]:5d}")
print(f"\napp.py-д хэрэглэх Norm stats:")
print(f"  NORM_MEAN = {TR_MEAN:.4f}")
print(f"  NORM_STD  = {TR_STD:.4f}")