"""
train_models.py  —  Step 6–9
==============================
Оролт: /content/drive/MyDrive/Thesis/data/ready/ дотор байгаа .npy файлууд
       (prepare_data.py ажилласны дараа)

Гаралт:
  models/   ← 9 .pth файл
  results/  ← summary_9experiments.csv
             confusion_matrix_grid.png
             f1_comparison.png
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

# ── PATHS ─────────────────────────────────────────────────────────
DRIVE_ROOT  = "/content/drive/MyDrive/Thesis"
READY_DIR   = os.path.join(DRIVE_ROOT, "data", "ready")
MODEL_DIR   = os.path.join(DRIVE_ROOT, "models")
RESULTS_DIR = os.path.join(DRIVE_ROOT, "results")

os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── CONFIG ────────────────────────────────────────────────────────
BATCH_SIZE = 32
EPOCHS     = 50
LR         = 1e-3
PATIENCE   = 10
SEED       = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ══════════════════════════════════════════════════════════════════
# STEP 6a — LOAD PREPARED DATA
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("STEP 6a: Бэлтгэгдсэн өгөгдөл ачаалж байна...")
print("=" * 55)

def load(name):
    return np.load(os.path.join(READY_DIR, f"{name}.npy"))

# Normalized splits
rX_tr = load("raw_train");  rX_va = load("raw_val");  rX_te = load("raw_test")
mX_tr = load("mfcc_train"); mX_va = load("mfcc_val"); mX_te = load("mfcc_test")
lX_tr = load("lmel_train"); lX_va = load("lmel_val"); lX_te = load("lmel_test")
y_tr  = load("y_train");    y_va  = load("y_val");    y_te  = load("y_test")

# Oversampled
rX_sm = load("raw_sm");  ry_sm = load("ry_sm")
mX_sm = load("mfcc_sm"); my_sm = load("my_sm")
lX_sm = load("lmel_sm"); ly_sm = load("ly_sm")

print(f"  Train : {len(y_tr)}  "
      f"(хэвийн {np.sum(y_tr==0)}, хэвийн бус {np.sum(y_tr==1)})")
print(f"  Val   : {len(y_va)}")
print(f"  Test  : {len(y_te)}")
print(f"  Raw_sm: {rX_sm.shape}  MFCC_sm: {mX_sm.shape}")

# ══════════════════════════════════════════════════════════════════
# DATALOADER HELPER
# ══════════════════════════════════════════════════════════════════

def make_loader(X, y, shuffle=False, is_1d=False):
    if is_1d:
        X_t = torch.tensor(X[:, np.newaxis, :],      dtype=torch.float32)
    else:
        X_t = torch.tensor(X[:, np.newaxis, :, :],   dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(X_t, y_t),
                      batch_size=BATCH_SIZE, shuffle=shuffle)

def cw_tensor(y):
    cw = compute_class_weight("balanced",
                               classes=np.array([0,1]), y=y)
    return torch.tensor(cw, dtype=torch.float32).to(device)

# ── Pre-build loaders ─────────────────────────────────────────────
# Val / Test (shared)
r_va = make_loader(rX_va, y_va, is_1d=True)
r_te = make_loader(rX_te, y_te, is_1d=True)
m_va = make_loader(mX_va, y_va)
m_te = make_loader(mX_te, y_te)
l_va = make_loader(lX_va, y_va)
l_te = make_loader(lX_te, y_te)

# Train loaders
r_tr_plain  = make_loader(rX_tr, y_tr,  shuffle=True, is_1d=True)
r_tr_sm     = make_loader(rX_sm, ry_sm, shuffle=True, is_1d=True)
m_tr_plain  = make_loader(mX_tr, y_tr,  shuffle=True)
m_tr_sm     = make_loader(mX_sm, my_sm, shuffle=True)
l_tr_plain  = make_loader(lX_tr, y_tr,  shuffle=True)
l_tr_sm     = make_loader(lX_sm, ly_sm, shuffle=True)

# Class weights
cw_r = cw_tensor(y_tr)
cw_m = cw_tensor(y_tr)
cw_l = cw_tensor(y_tr)

# ══════════════════════════════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════════════════════════════

class CNN1D(nn.Module):
    """Input: (B, 1, 4000)"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1,  32, 7, padding=3), nn.BatchNorm1d(32),  nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64),  nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(64,128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.classifier(self.features(x))


class CNN2D(nn.Module):
    """Input: (B, 1, H, W)"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4,4))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*4*4, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 2)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

# ══════════════════════════════════════════════════════════════════
# TRAIN / EVAL HELPERS
# ══════════════════════════════════════════════════════════════════

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total = 0.0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(Xb), yb)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)

def eval_loss(model, loader, criterion):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            total += criterion(model(Xb), yb).item()
    return total / len(loader)

def predict(model, loader):
    model.eval()
    yt, yp = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            preds = torch.argmax(model(Xb.to(device)), dim=1)
            yt.extend(yb.numpy())
            yp.extend(preds.cpu().numpy())
    return np.array(yt), np.array(yp)

def metrics(yt, yp):
    return {
        "Accuracy":  round(accuracy_score(yt, yp),                   3),
        "Precision": round(precision_score(yt, yp, zero_division=0), 3),
        "Recall":    round(recall_score(yt, yp,    zero_division=0), 3),
        "F1":        round(f1_score(yt, yp,         zero_division=0),3),
        "cm":        confusion_matrix(yt, yp),
    }

def run(tag, model, tr_loader, va_loader, te_loader, cw=None):
    print(f"\n  [{tag}]")
    t0 = time.time()

    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, verbose=False)

    best_val, best_state, no_imp = float("inf"), None, 0

    for epoch in range(1, EPOCHS+1):
        tr_loss  = train_epoch(model, tr_loader, criterion, optimizer)
        val_loss = eval_loss(model, va_loader, criterion)
        scheduler.step(val_loss)

        if epoch % 10 == 0:
            print(f"    Epoch {epoch:3d}/{EPOCHS} | "
                  f"train {tr_loss:.4f} | val {val_loss:.4f}")

        if val_loss < best_val:
            best_val  = val_loss
            best_state = {k: v.clone() for k,v in model.state_dict().items()}
            no_imp    = 0
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                print(f"    Early stop — epoch {epoch}")
                break

    model.load_state_dict(best_state)
    torch.save(model.state_dict(),
               os.path.join(MODEL_DIR, f"{tag}.pth"))

    yt, yp = predict(model, te_loader)
    m = metrics(yt, yp)
    print(f"    Acc {m['Accuracy']}  Prec {m['Precision']}  "
          f"Rec {m['Recall']}  F1 {m['F1']}  "
          f"({time.time()-t0:.0f}s)")
    return m

# ══════════════════════════════════════════════════════════════════
# STEP 6b — 9 ТУРШИЛТ
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("STEP 6b: 9 туршилт эхэлж байна...")
print("=" * 55)

results = []

# ── RAW ───────────────────────────────────────────────────────────
print("\n── RAW (1D CNN) ──")
m = run("raw_baseline",    CNN1D().to(device), r_tr_plain, r_va, r_te)
results.append({"Feature":"Raw", "Method":"Baseline", **m})

m = run("raw_classweight", CNN1D().to(device), r_tr_plain, r_va, r_te, cw=cw_r)
results.append({"Feature":"Raw", "Method":"Class Weight", **m})

m = run("raw_smote",       CNN1D().to(device), r_tr_sm,    r_va, r_te)
results.append({"Feature":"Raw", "Method":"SMOTE", **m})

# ── MFCC ──────────────────────────────────────────────────────────
print("\n── MFCC (2D CNN) ──")
m = run("mfcc_baseline",    CNN2D().to(device), m_tr_plain, m_va, m_te)
results.append({"Feature":"MFCC", "Method":"Baseline", **m})

m = run("mfcc_classweight", CNN2D().to(device), m_tr_plain, m_va, m_te, cw=cw_m)
results.append({"Feature":"MFCC", "Method":"Class Weight", **m})

m = run("mfcc_smote",       CNN2D().to(device), m_tr_sm,    m_va, m_te)
results.append({"Feature":"MFCC", "Method":"SMOTE", **m})

# ── LOG-MEL ───────────────────────────────────────────────────────
print("\n── LOG-MEL (2D CNN) ──")
m = run("lmel_baseline",    CNN2D().to(device), l_tr_plain, l_va, l_te)
results.append({"Feature":"Log-Mel", "Method":"Baseline", **m})

m = run("lmel_classweight", CNN2D().to(device), l_tr_plain, l_va, l_te, cw=cw_l)
results.append({"Feature":"Log-Mel", "Method":"Class Weight", **m})

m = run("lmel_smote",       CNN2D().to(device), l_tr_sm,    l_va, l_te)
results.append({"Feature":"Log-Mel", "Method":"SMOTE", **m})

# ══════════════════════════════════════════════════════════════════
# STEP 7 — ХҮСНЭГТ
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("STEP 7: Үр дүн нэгтгэж байна...")
print("=" * 55)

df = pd.DataFrame([{k:v for k,v in r.items() if k!="cm"}
                   for r in results])
df = df.set_index(["Feature","Method"])
print("\n" + df.to_string())

csv_path = os.path.join(RESULTS_DIR, "summary_9experiments.csv")
df.to_csv(csv_path)
print(f"\nCSV → {csv_path}")

# ══════════════════════════════════════════════════════════════════
# STEP 8 — CONFUSION MATRIX GRID
# ══════════════════════════════════════════════════════════════════
print("\nSTEP 8: Confusion matrix grid...")

features = ["Raw", "MFCC", "Log-Mel"]
methods  = ["Baseline", "Class Weight", "SMOTE"]
labels   = ["Хэвийн", "Хэвийн бус"]

fig, axes = plt.subplots(3, 3, figsize=(13, 11))
fig.suptitle("Confusion Matrices — 9 Туршилт",
             fontsize=13, y=1.01)

for i, feat in enumerate(features):
    for j, method in enumerate(methods):
        ax  = axes[i][j]
        row = next(r for r in results
                   if r["Feature"]==feat and r["Method"]==method)
        sns.heatmap(row["cm"], annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels,
                    ax=ax, cbar=False)
        ax.set_title(f"{feat} + {method}\n"
                     f"Acc {row['Accuracy']}  F1 {row['F1']}", fontsize=9)
        ax.set_xlabel("Таамаглал", fontsize=8)
        ax.set_ylabel("Бодит",     fontsize=8)

plt.tight_layout()
fig_path = os.path.join(RESULTS_DIR, "confusion_matrix_grid.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Confusion matrix → {fig_path}")

# ══════════════════════════════════════════════════════════════════
# STEP 9 — F1 BAR CHART
# ══════════════════════════════════════════════════════════════════
print("\nSTEP 9: F1 chart...")

fig, ax = plt.subplots(figsize=(10, 5))
x      = np.arange(len(features))
width  = 0.25
colors = ["#4C72B0", "#DD8452", "#55A868"]

for j, (method, color) in enumerate(zip(methods, colors)):
    vals = [next(r["F1"] for r in results
                 if r["Feature"]==feat and r["Method"]==method)
            for feat in features]
    bars = ax.bar(x + j*width, vals, width, label=method,
                  color=color, alpha=0.85)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                str(val), ha="center", va="bottom", fontsize=9)

ax.set_xlabel("Оролтын өгөгдлийн төлөөлөл", fontsize=11)
ax.set_ylabel("F1-score",                    fontsize=11)
ax.set_title("F1-score харьцуулалт: Feature × Balancing Method",
             fontsize=12)
ax.set_xticks(x + width)
ax.set_xticklabels(features, fontsize=11)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
bar_path = os.path.join(RESULTS_DIR, "f1_comparison.png")
plt.savefig(bar_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"F1 chart → {bar_path}")

print("\n" + "=" * 55)
print("  БҮГД ДУУСЛАА")
print(f"  Models  → {MODEL_DIR}")
print(f"  Results → {RESULTS_DIR}")
print("=" * 55)