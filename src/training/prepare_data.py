"""
prepare_data.py  —  Step 1–5
==============================
Гаралт: /content/drive/MyDrive/Thesis/data/ready/ дотор
  raw_train.npy  raw_val.npy  raw_test.npy
  mfcc_train.npy mfcc_val.npy mfcc_test.npy
  lmel_train.npy lmel_val.npy lmel_test.npy
  y_train.npy    y_val.npy    y_test.npy

  raw_sm.npy   ry_sm.npy    ← Raw  + oversample
  mfcc_sm.npy  my_sm.npy    ← MFCC + oversample
  lmel_sm.npy  ly_sm.npy    ← LogMel + oversample
"""

import os, sys
import numpy as np
import librosa
from scipy.signal import resample as scipy_resample
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, RandomOverSampler
from tqdm import tqdm

# ── PATHS ─────────────────────────────────────────────────────────
DRIVE_ROOT = "/content/drive/MyDrive/Thesis"
PROC_DIR   = os.path.join(DRIVE_ROOT, "data", "processed")
FEAT_DIR   = os.path.join(DRIVE_ROOT, "data", "features")
READY_DIR  = os.path.join(DRIVE_ROOT, "data", "ready")

os.makedirs(FEAT_DIR,  exist_ok=True)
os.makedirs(READY_DIR, exist_ok=True)

# ── CONFIG ────────────────────────────────────────────────────────
SR         = 2000
N_MFCC     = 40
N_MELS     = 128
N_FFT      = 512
HOP_LENGTH = 256
SEED       = 42
np.random.seed(SEED)

# ══════════════════════════════════════════════════════════════════
# STEP 1 — RAW SEGMENTS
# ══════════════════════════════════════════════════════════════════
print("=" * 55)
print("STEP 1: Raw segments ачаалж байна...")
print("=" * 55)

X_raw = np.load(os.path.join(PROC_DIR, "X.npy"))
y_all = np.load(os.path.join(PROC_DIR, "y.npy"))

print(f"  X shape : {X_raw.shape}")
print(f"  Хэвийн     : {np.sum(y_all==0)}")
print(f"  Хэвийн бус : {np.sum(y_all==1)}")

# ══════════════════════════════════════════════════════════════════
# STEP 2 — FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("STEP 2: Feature extraction...")
print("=" * 55)

mfcc_path = os.path.join(FEAT_DIR, "mfcc.npy")
lmel_path = os.path.join(FEAT_DIR, "logmel.npy")

if os.path.exists(mfcc_path) and os.path.exists(lmel_path):
    print("  Cache олдлоо — дахин тооцохгүй.")
    X_mfcc  = np.load(mfcc_path)
    X_logmel = np.load(lmel_path)
else:
    print(f"  {len(X_raw)} сегментээс feature гаргаж байна...")
    mfcc_list, lmel_list = [], []
    for seg in tqdm(X_raw, desc="  Extracting"):
        mfcc = librosa.feature.mfcc(
            y=seg, sr=SR, n_mfcc=N_MFCC,
            n_fft=N_FFT, hop_length=HOP_LENGTH)
        mfcc_list.append(mfcc)

        mel  = librosa.feature.melspectrogram(
            y=seg, sr=SR, n_fft=N_FFT,
            hop_length=HOP_LENGTH, n_mels=N_MELS)
        lmel_list.append(librosa.power_to_db(mel, ref=np.max))

    X_mfcc   = np.array(mfcc_list)
    X_logmel = np.array(lmel_list)
    np.save(mfcc_path, X_mfcc)
    np.save(lmel_path, X_logmel)
    np.save(os.path.join(FEAT_DIR, "labels.npy"), y_all)

print(f"  MFCC    : {X_mfcc.shape}")
print(f"  Log-Mel : {X_logmel.shape}")

# ══════════════════════════════════════════════════════════════════
# STEP 3 — SPLIT  70 / 15 / 15
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("STEP 3: Dataset split (70 / 15 / 15)...")
print("=" * 55)

# Ижил index-ээр хуваана — бүх feature адил split авна
idx = np.arange(len(y_all))
idx_tr, idx_tmp = train_test_split(
    idx, test_size=0.30, stratify=y_all, random_state=SEED)
idx_va, idx_te  = train_test_split(
    idx_tmp, test_size=0.50,
    stratify=y_all[idx_tmp], random_state=SEED)

def split_by_idx(X):
    return X[idx_tr], X[idx_va], X[idx_te]

rX_tr, rX_va, rX_te   = split_by_idx(X_raw)
mX_tr, mX_va, mX_te   = split_by_idx(X_mfcc)
lX_tr, lX_va, lX_te   = split_by_idx(X_logmel)
y_tr,  y_va,  y_te    = (y_all[idx_tr],
                          y_all[idx_va],
                          y_all[idx_te])

print(f"  Train : {len(y_tr)}  "
      f"(хэвийн {np.sum(y_tr==0)}, хэвийн бус {np.sum(y_tr==1)})")
print(f"  Val   : {len(y_va)}")
print(f"  Test  : {len(y_te)}")

# ══════════════════════════════════════════════════════════════════
# STEP 4 — NORMALIZATION
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("STEP 4: Normalization (train stats only)...")
print("=" * 55)

def normalize(tr, va, te):
    mu, sigma = tr.mean(), tr.std() + 1e-8
    return (tr-mu)/sigma, (va-mu)/sigma, (te-mu)/sigma

rX_tr, rX_va, rX_te = normalize(rX_tr, rX_va, rX_te)
mX_tr, mX_va, mX_te = normalize(mX_tr, mX_va, mX_te)
lX_tr, lX_va, lX_te = normalize(lX_tr, lX_va, lX_te)
print("  Done.")

# ── Normalized split хадгална (sургалтын файлд хэрэгтэй)
for name, arr in [
    ("raw_train", rX_tr), ("raw_val",  rX_va), ("raw_test",  rX_te),
    ("mfcc_train",mX_tr), ("mfcc_val", mX_va), ("mfcc_test", mX_te),
    ("lmel_train",lX_tr), ("lmel_val", lX_va), ("lmel_test", lX_te),
    ("y_train",   y_tr),  ("y_val",    y_va),  ("y_test",    y_te),
]:
    np.save(os.path.join(READY_DIR, f"{name}.npy"), arr)
print(f"  Normalized splits → {READY_DIR}")

# ══════════════════════════════════════════════════════════════════
# STEP 5 — OVERSAMPLE  (train set only)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("STEP 5: Oversample (train set only)...")
print("=" * 55)

# ── 5a. Raw: downsample→SMOTE→upsample ───────────────────────────
print("  [5a] Raw: downsample 4000→200 ...")
X_down = scipy_resample(rX_tr, 200, axis=1)       # (N, 200)

smote  = SMOTE(k_neighbors=3, random_state=SEED)
X_res_down, ry_sm = smote.fit_resample(X_down, y_tr)

print("  [5a] Raw: upsample 200→4000 ...")
rX_sm = scipy_resample(X_res_down, 4000, axis=1)  # (N_res, 4000)
print(f"  Raw: {np.sum(y_tr==0)}/{np.sum(y_tr==1)} "
      f"→ {np.sum(ry_sm==0)}/{np.sum(ry_sm==1)}")

# ── 5b. MFCC / Log-Mel: RandomOverSampler ────────────────────────
def oversample_2d(X, y, tag):
    orig_shape = X.shape[1:]
    ros = RandomOverSampler(random_state=SEED)
    X_res, y_res = ros.fit_resample(X.reshape(len(X), -1), y)
    X_res = X_res.reshape(-1, *orig_shape)
    print(f"  {tag}: {np.sum(y==0)}/{np.sum(y==1)} "
          f"→ {np.sum(y_res==0)}/{np.sum(y_res==1)}")
    return X_res, y_res

print("  [5b] MFCC oversampling ...")
mX_sm, my_sm = oversample_2d(mX_tr, y_tr, "MFCC   ")

print("  [5c] Log-Mel oversampling ...")
lX_sm, ly_sm = oversample_2d(lX_tr, y_tr, "Log-Mel")

# ── Oversampled arrays хадгална ───────────────────────────────────
for name, arr in [
    ("raw_sm",  rX_sm), ("ry_sm",  ry_sm),
    ("mfcc_sm", mX_sm), ("my_sm",  my_sm),
    ("lmel_sm", lX_sm), ("ly_sm",  ly_sm),
]:
    np.save(os.path.join(READY_DIR, f"{name}.npy"), arr)

print(f"\n  Oversampled arrays → {READY_DIR}")
print("\n" + "=" * 55)
print("  БЭЛТГЭЛ ДУУСЛАА. train_models.py ажиллуулна уу.")
print("=" * 55)