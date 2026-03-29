"""
9-Experiment Runner  ─  3 Features × 3 Balancing Methods
=========================================================
Thesis: Detection of Abnormal Heart Rhythm Based on Heart Sound Signals

Experiments
-----------
  Feature         │ Arch   │ Baseline │ Class Weight │ SMOTE
  ────────────────┼────────┼──────────┼──────────────┼──────
  Raw signal      │ 1D CNN │    ✓     │      ✓       │   ✓
  MFCC            │ 2D CNN │    ✓     │      ✓       │   ✓
  Log-Mel Spec    │ 2D CNN │    ✓     │      ✓       │   ✓

Input
-----
  data/processed/X.npy   ← preprocessing.py гаралт (raw waveform)
  data/processed/y.npy

Output
------
  models/   ← 9 .pth файл
  results/  ← summary_9experiments.csv
             confusion_matrix_*.png

Colab usage
-----------
  # 1. Drive mount + repo clone
  from google.colab import drive; drive.mount('/content/drive')
  !git clone https://github.com/<user>/<repo> /content/project
  %cd /content/project
  !pip install imbalanced-learn seaborn -q

  # 2. Run
  !python src/training/experiment_9x.py
"""

# ──────────────────────────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────────────────────────
import os, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
from imblearn.over_sampling import SMOTE
import librosa
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────
# PATHS  ←  Google Drive хэрэглэж байвал DRIVE_ROOT өөрчил
# ──────────────────────────────────────────────────────────────────
DRIVE_ROOT   = "/content/drive/MyDrive/Thesis"   # Drive дахь root

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR     = os.path.join(BASE_DIR, "data", "processed")
FEAT_DIR     = os.path.join(BASE_DIR, "data", "features")
MODEL_DIR    = os.path.join(BASE_DIR, "models")
RESULTS_DIR  = os.path.join(BASE_DIR, "results")

# data/processed нь Drive дээр байвал замыг override хий
# PROC_DIR = os.path.join(DRIVE_ROOT, "data", "processed")
# FEAT_DIR = os.path.join(DRIVE_ROOT, "data", "features")

os.makedirs(FEAT_DIR,    exist_ok=True)
os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────
# HYPERPARAMETERS
# ──────────────────────────────────────────────────────────────────
SR           = 2000
SEGMENT_LEN  = 4000    # 2 сек × 2000 Hz
N_MFCC       = 40
N_MELS       = 128
N_FFT        = 512
HOP_LENGTH   = 256

BATCH_SIZE   = 32
EPOCHS       = 50
LR           = 1e-3
PATIENCE     = 10
SEED         = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# ══════════════════════════════════════════════════════════════════
# STEP 1 ─ RAW SEGMENTS АЧААЛАХ
# ══════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1: Raw segments ачаалж байна...")
print("=" * 60)

X_raw = np.load(os.path.join(PROC_DIR, "X.npy"))   # (N, 4000)
y_all = np.load(os.path.join(PROC_DIR, "y.npy"))   # (N,)

print(f"  X shape : {X_raw.shape}")
print(f"  Хэвийн     : {np.sum(y_all == 0)}")
print(f"  Хэвийн бус : {np.sum(y_all == 1)}")

# ══════════════════════════════════════════════════════════════════
# STEP 2 ─ FEATURE EXTRACTION  (MFCC, Log-Mel)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: Feature extraction...")
print("=" * 60)

def extract_features(X_segments, sr=SR):
    """
    Raw waveform-аас MFCC болон Log-Mel Spectrogram гаргана.
    Cache файл байвал дахин тооцохгүй.
    """
    mfcc_path   = os.path.join(FEAT_DIR, "mfcc.npy")
    logmel_path = os.path.join(FEAT_DIR, "logmel.npy")

    if os.path.exists(mfcc_path) and os.path.exists(logmel_path):
        print("  Cache олдлоо — дахин тооцохгүй.")
        mfcc_arr   = np.load(mfcc_path)
        logmel_arr = np.load(logmel_path)
        return mfcc_arr, logmel_arr

    print(f"  {len(X_segments)} сегментээс feature гаргаж байна...")
    mfcc_list, logmel_list = [], []

    for seg in tqdm(X_segments, desc="  Extracting"):
        # MFCC  →  (40, T)
        mfcc = librosa.feature.mfcc(
            y=seg, sr=sr, n_mfcc=N_MFCC,
            n_fft=N_FFT, hop_length=HOP_LENGTH
        )
        mfcc_list.append(mfcc)

        # Log-Mel  →  (128, T)
        mel    = librosa.feature.melspectrogram(
            y=seg, sr=sr, n_fft=N_FFT,
            hop_length=HOP_LENGTH, n_mels=N_MELS
        )
        logmel = librosa.power_to_db(mel, ref=np.max)
        logmel_list.append(logmel)

    mfcc_arr   = np.array(mfcc_list)    # (N, 40, T)
    logmel_arr = np.array(logmel_list)  # (N, 128, T)

    np.save(mfcc_path,   mfcc_arr)
    np.save(logmel_path, logmel_arr)
    np.save(os.path.join(FEAT_DIR, "labels.npy"), y_all)

    print(f"  MFCC    : {mfcc_arr.shape}")
    print(f"  Log-Mel : {logmel_arr.shape}")
    return mfcc_arr, logmel_arr

X_mfcc, X_logmel = extract_features(X_raw)

# ══════════════════════════════════════════════════════════════════
# STEP 3 ─ TRAIN / VAL / TEST  SPLIT  (70 / 15 / 15)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3: Dataset split (70 / 15 / 15)...")
print("=" * 60)

def split_data(X, y, seed=SEED):
    idx = np.arange(len(y))
    idx_tr, idx_tmp = train_test_split(
        idx, test_size=0.30, stratify=y, random_state=seed)
    idx_va, idx_te  = train_test_split(
        idx_tmp, test_size=0.50, stratify=y[idx_tmp], random_state=seed)
    return (X[idx_tr], y[idx_tr],
            X[idx_va], y[idx_va],
            X[idx_te], y[idx_te])

# Raw
rX_tr, ry_tr, rX_va, ry_va, rX_te, ry_te = split_data(X_raw,   y_all)
# MFCC
mX_tr, my_tr, mX_va, my_va, mX_te, my_te = split_data(X_mfcc,  y_all)
# Log-Mel
lX_tr, ly_tr, lX_va, ly_va, lX_te, ly_te = split_data(X_logmel, y_all)

for name, tr, va, te in [("Raw    ", ry_tr, ry_va, ry_te),
                          ("MFCC   ", my_tr, my_va, my_te),
                          ("Log-Mel", ly_tr, ly_va, ly_te)]:
    print(f"  {name} | train {len(tr)} "
          f"(n:{np.sum(tr==0)} ab:{np.sum(tr==1)}) "
          f"| val {len(va)} | test {len(te)}")

# ══════════════════════════════════════════════════════════════════
# STEP 4 ─ NORMALIZATION  (train statistics only)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4: Normalization...")
print("=" * 60)

def normalize(X_tr, X_va, X_te):
    mu, sigma = X_tr.mean(), X_tr.std() + 1e-8
    return (X_tr-mu)/sigma, (X_va-mu)/sigma, (X_te-mu)/sigma

rX_tr, rX_va, rX_te = normalize(rX_tr, rX_va, rX_te)
mX_tr, mX_va, mX_te = normalize(mX_tr, mX_va, mX_te)
lX_tr, lX_va, lX_te = normalize(lX_tr, lX_va, lX_te)
print("  Done.")

# ══════════════════════════════════════════════════════════════════
# STEP 5 ─ SMOTE  (train set only, per feature type)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5: SMOTE...")
print("=" * 60)

def apply_smote(X_tr, y_tr, tag, raw_downsample_to=None):
    """
    SMOTE with optional downsample→SMOTE→upsample for raw waveform.

    Parameters
    ----------
    X_tr               : ndarray  (N, L) raw  |  (N, H, W) 2-D feature
    y_tr               : ndarray  (N,)
    tag                : str      logging label
    raw_downsample_to  : int|None
        Raw waveform дээр хэрэглэх бол энд downsample target length өгнө.
        None бол flatten хийгээд шууд SMOTE хэрэглэнэ (MFCC/LogMel).

    Strategy (raw only)
    -------------------
      1. 4000 → raw_downsample_to  (scipy.signal.resample)
      2. SMOTE on (N, raw_downsample_to)
      3. raw_downsample_to → 4000  (upsample back)

    Яагаад ажиллах вэ?
      SMOTE-ийн цаг нь O(n_minority × k × d) — d (feature dim) багасах тусам
      экспоненциал хурдасна.  4000→200 бол 20× бага memory, ~400× хурдан.
      Уpsample хийсний дараа дохионы хэлбэр хадгалагдах ч нарийн дэлгэрэнгүй
      мэдээлэл бага зэрэг алдагдана — энэ нь raw 1D CNN-д тийм ч чухал биш.
    """
    from scipy.signal import resample as scipy_resample

    orig_shape = X_tr.shape[1:]   # (4000,) эсвэл (H, W)
    n_before   = len(X_tr)

    if raw_downsample_to is not None:
        # ── Raw waveform: downsample → SMOTE → upsample ──────────────
        orig_len = X_tr.shape[1]         # 4000

        print(f"  {tag}: downsample {orig_len}→{raw_downsample_to} ...")
        X_down = scipy_resample(X_tr, raw_downsample_to, axis=1)
        # shape: (N, raw_downsample_to)

        smote = SMOTE(k_neighbors=5, random_state=SEED)
        X_res_down, y_res = smote.fit_resample(X_down, y_tr)
        # shape: (N_res, raw_downsample_to)

        print(f"  {tag}: upsample {raw_downsample_to}→{orig_len} ...")
        X_res = scipy_resample(X_res_down, orig_len, axis=1)
        # shape: (N_res, 4000)

    else:
        # ── 2-D feature (MFCC / Log-Mel): flatten → SMOTE → reshape ──
        X_flat = X_tr.reshape(n_before, -1)
        smote  = SMOTE(k_neighbors=5, random_state=SEED)
        X_res_flat, y_res = smote.fit_resample(X_flat, y_tr)
        X_res = X_res_flat.reshape(-1, *orig_shape)

    print(f"  {tag}: {np.sum(y_tr==0)}/{np.sum(y_tr==1)} "
          f"→ {np.sum(y_res==0)}/{np.sum(y_res==1)}")
    return X_res, y_res

# Raw: downsample 4000→200, SMOTE, upsample 200→4000
rX_sm, ry_sm = apply_smote(rX_tr, ry_tr, "Raw    ", raw_downsample_to=200)
# MFCC / Log-Mel: flatten→SMOTE→reshape (хэмжээ бага тул шууд)
mX_sm, my_sm = apply_smote(mX_tr, my_tr, "MFCC   ")
lX_sm, ly_sm = apply_smote(lX_tr, ly_tr, "Log-Mel")

# ══════════════════════════════════════════════════════════════════
# DATALOADER HELPER
# ══════════════════════════════════════════════════════════════════

def make_loader(X, y, shuffle=False, is_1d=False):
    """
    numpy → TensorDataset → DataLoader
    is_1d=True  : (N, L)      → (N, 1, L)
    is_1d=False : (N, H, W)   → (N, 1, H, W)
    """
    if is_1d:
        X_t = torch.tensor(X[:, np.newaxis, :],       dtype=torch.float32)
    else:
        X_t = torch.tensor(X[:, np.newaxis, :, :],    dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(X_t, y_t),
                      batch_size=BATCH_SIZE, shuffle=shuffle)

def class_weight_tensor(y_tr):
    cw = compute_class_weight("balanced",
                               classes=np.array([0, 1]), y=y_tr)
    return torch.tensor(cw, dtype=torch.float32).to(device)

# ══════════════════════════════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════════════════════════════

class CNN1D(nn.Module):
    """Raw waveform  →  input: (B, 1, 4000)"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1,  32, 7, padding=3), nn.BatchNorm1d(32),  nn.ReLU(),
            nn.MaxPool1d(4),                                    # → 1000
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64),  nn.ReLU(),
            nn.MaxPool1d(4),                                    # → 250
            nn.Conv1d(64,128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),                            # → 1
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.classifier(self.features(x))


class CNN2D(nn.Module):
    """MFCC / Log-Mel  →  input: (B, 1, H, W)"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*4*4, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 2)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

# ══════════════════════════════════════════════════════════════════
# TRAINING / EVALUATION HELPERS
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

def compute_metrics(yt, yp):
    return {
        "Accuracy":  round(accuracy_score(yt, yp),                    3),
        "Precision": round(precision_score(yt, yp, zero_division=0),  3),
        "Recall":    round(recall_score(yt, yp,    zero_division=0),  3),
        "F1":        round(f1_score(yt, yp,         zero_division=0), 3),
        "cm":        confusion_matrix(yt, yp),
    }

# ══════════════════════════════════════════════════════════════════
# ONE-EXPERIMENT RUNNER
# ══════════════════════════════════════════════════════════════════

def run(tag, model, tr_loader, va_loader, te_loader, cw=None):
    """
    tag        : str  e.g. "raw_baseline"
    model      : nn.Module (шинэ instantiation)
    tr_loader  : train DataLoader
    va_loader  : val DataLoader
    te_loader  : test DataLoader
    cw         : class weight tensor | None
    """
    print(f"\n  ── {tag} ──")
    t0 = time.time()

    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, verbose=False)

    best_val, best_state, no_improve = float("inf"), None, 0

    for epoch in range(1, EPOCHS + 1):
        tr_loss  = train_epoch(model, tr_loader, criterion, optimizer)
        val_loss = eval_loss(model, va_loader, criterion)
        scheduler.step(val_loss)

        if epoch % 10 == 0:
            print(f"    Epoch {epoch:3d}/{EPOCHS} | "
                  f"train {tr_loss:.4f} | val {val_loss:.4f}")

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"    Early stop — epoch {epoch}")
                break

    model.load_state_dict(best_state)
    torch.save(model.state_dict(),
               os.path.join(MODEL_DIR, f"{tag}.pth"))

    yt, yp  = predict(model, te_loader)
    metrics = compute_metrics(yt, yp)

    elapsed = time.time() - t0
    print(f"    Acc {metrics['Accuracy']}  "
          f"Prec {metrics['Precision']}  "
          f"Rec {metrics['Recall']}  "
          f"F1 {metrics['F1']}  "
          f"({elapsed:.0f}s)")
    return metrics

# ══════════════════════════════════════════════════════════════════
# STEP 6 ─ 9 ТУРШИЛТ ГҮЙЦЭТГЭХ
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 6: 9 туршилт эхэлж байна...")
print("=" * 60)

results = []

# ── DataLoader-ууд ────────────────────────────────────────────────
# Raw (1D)
r_va = make_loader(rX_va, ry_va, is_1d=True)
r_te = make_loader(rX_te, ry_te, is_1d=True)

# MFCC (2D)
m_va = make_loader(mX_va, my_va)
m_te = make_loader(mX_te, my_te)

# Log-Mel (2D)
l_va = make_loader(lX_va, ly_va)
l_te = make_loader(lX_te, ly_te)

# Class weight tensors
cw_raw = class_weight_tensor(ry_tr)
cw_mfc = class_weight_tensor(my_tr)
cw_lml = class_weight_tensor(ly_tr)

# ─────────────────────────────────────────────────────────────────
# 1 · RAW — BASELINE
# ─────────────────────────────────────────────────────────────────
m = run("raw_baseline",
        CNN1D().to(device),
        make_loader(rX_tr, ry_tr, shuffle=True, is_1d=True),
        r_va, r_te)
results.append({"Feature": "Raw", "Method": "Baseline", **m})

# 2 · RAW — CLASS WEIGHT
m = run("raw_classweight",
        CNN1D().to(device),
        make_loader(rX_tr, ry_tr, shuffle=True, is_1d=True),
        r_va, r_te, cw=cw_raw)
results.append({"Feature": "Raw", "Method": "Class Weight", **m})

# 3 · RAW — SMOTE
m = run("raw_smote",
        CNN1D().to(device),
        make_loader(rX_sm, ry_sm, shuffle=True, is_1d=True),
        r_va, r_te)
results.append({"Feature": "Raw", "Method": "SMOTE", **m})

# ─────────────────────────────────────────────────────────────────
# 4 · MFCC — BASELINE
# ─────────────────────────────────────────────────────────────────
m = run("mfcc_baseline",
        CNN2D().to(device),
        make_loader(mX_tr, my_tr, shuffle=True),
        m_va, m_te)
results.append({"Feature": "MFCC", "Method": "Baseline", **m})

# 5 · MFCC — CLASS WEIGHT
m = run("mfcc_classweight",
        CNN2D().to(device),
        make_loader(mX_tr, my_tr, shuffle=True),
        m_va, m_te, cw=cw_mfc)
results.append({"Feature": "MFCC", "Method": "Class Weight", **m})

# 6 · MFCC — SMOTE
m = run("mfcc_smote",
        CNN2D().to(device),
        make_loader(mX_sm, my_sm, shuffle=True),
        m_va, m_te)
results.append({"Feature": "MFCC", "Method": "SMOTE", **m})

# ─────────────────────────────────────────────────────────────────
# 7 · LOG-MEL — BASELINE
# ─────────────────────────────────────────────────────────────────
m = run("logmel_baseline",
        CNN2D().to(device),
        make_loader(lX_tr, ly_tr, shuffle=True),
        l_va, l_te)
results.append({"Feature": "Log-Mel", "Method": "Baseline", **m})

# 8 · LOG-MEL — CLASS WEIGHT
m = run("logmel_classweight",
        CNN2D().to(device),
        make_loader(lX_tr, ly_tr, shuffle=True),
        l_va, l_te, cw=cw_lml)
results.append({"Feature": "Log-Mel", "Method": "Class Weight", **m})

# 9 · LOG-MEL — SMOTE
m = run("logmel_smote",
        CNN2D().to(device),
        make_loader(lX_sm, ly_sm, shuffle=True),
        l_va, l_te)
results.append({"Feature": "Log-Mel", "Method": "SMOTE", **m})

# ══════════════════════════════════════════════════════════════════
# STEP 7 ─ ХҮСНЭГТ ХЭВЛЭХ + CSV ХАДГАЛАХ
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 7: Үр дүн нэгтгэж байна...")
print("=" * 60)

df = pd.DataFrame([
    {k: v for k, v in r.items() if k != "cm"}
    for r in results
])
df = df.set_index(["Feature", "Method"])

print("\n" + df.to_string())

csv_path = os.path.join(RESULTS_DIR, "summary_9experiments.csv")
df.to_csv(csv_path)
print(f"\nCSV хадгалагдлаа → {csv_path}")

# ══════════════════════════════════════════════════════════════════
# STEP 8 ─ CONFUSION MATRIX  (3×3 grid)
# ══════════════════════════════════════════════════════════════════
print("\nStep 8: Confusion matrix зурж байна...")

features = ["Raw", "MFCC", "Log-Mel"]
methods  = ["Baseline", "Class Weight", "SMOTE"]
labels   = ["Хэвийн", "Хэвийн бус"]

fig, axes = plt.subplots(3, 3, figsize=(13, 11))
fig.suptitle("Confusion Matrices — 9 Туршилт (Feature × Balancing Method)",
             fontsize=13, y=1.01)

for i, feat in enumerate(features):
    for j, method in enumerate(methods):
        ax  = axes[i][j]
        row = next(r for r in results
                   if r["Feature"] == feat and r["Method"] == method)
        sns.heatmap(row["cm"], annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels,
                    ax=ax, cbar=False)
        ax.set_title(f"{feat} + {method}\n"
                     f"Acc {row['Accuracy']}  F1 {row['F1']}",
                     fontsize=9)
        ax.set_xlabel("Таамаглал", fontsize=8)
        ax.set_ylabel("Бодит", fontsize=8)

plt.tight_layout()
fig_path = os.path.join(RESULTS_DIR, "confusion_matrix_grid.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Confusion matrix → {fig_path}")

# ══════════════════════════════════════════════════════════════════
# STEP 9 ─ F1 SCORE ХАРЬЦУУЛАЛТЫН BAR CHART
# ══════════════════════════════════════════════════════════════════
print("\nStep 9: F1 харьцуулалтын график зурж байна...")

fig, ax = plt.subplots(figsize=(10, 5))

x      = np.arange(len(features))
width  = 0.25
colors = ["#4C72B0", "#DD8452", "#55A868"]

for j, (method, color) in enumerate(zip(methods, colors)):
    f1_vals = [
        next(r["F1"] for r in results
             if r["Feature"] == feat and r["Method"] == method)
        for feat in features
    ]
    bars = ax.bar(x + j * width, f1_vals, width,
                  label=method, color=color, alpha=0.85)
    for bar, val in zip(bars, f1_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                str(val), ha="center", va="bottom", fontsize=9)

ax.set_xlabel("Оролтын өгөгдлийн төлөөлөл", fontsize=11)
ax.set_ylabel("F1-score",                   fontsize=11)
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

print("\n" + "=" * 60)
print("  БҮГД ДУУСЛАА")
print(f"  Models  → {MODEL_DIR}")
print(f"  Results → {RESULTS_DIR}")
print("=" * 60)