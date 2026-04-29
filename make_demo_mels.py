"""
make_demo_mels.py
Wav файл бүрийг step=2000 (50% overlap)-оор сегментчилж
X.npy-тай тулгаад mel_db хэлбэрээр хадгална.
Нэг удаа ажиллуулна.
"""

import os
import numpy as np
import librosa
import pandas as pd
from tqdm import tqdm

# ── PATHS ────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR  = os.path.join(BASE_DIR, "data", "raw")
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
FEAT_DIR = os.path.join(BASE_DIR, "data", "features")

# ── CONFIG ───────────────────────────────────────────────────────
SR         = 2000
SEG        = 4000
STEP       = 2000   # 50% overlap — X.npy-тай таарсан
N_FFT      = 512
HOP_LENGTH = 256
N_MELS     = 128
NORM_MEAN  = -60.8221
NORM_STD   = 21.9991

# ── LOAD ─────────────────────────────────────────────────────────
print("X.npy ачаалж байна...")
X   = np.load(os.path.join(PROC_DIR, "X.npy"))
y_x = np.load(os.path.join(PROC_DIR, "y.npy"))
print(f"X shape: {X.shape}")

ref = pd.read_csv(
    os.path.join(RAW_DIR, "REFERENCES.csv"),
    header=None, skiprows=1
)
ref.columns = ["record_id", "folder", "label"]
ref["label"] = ref["label"].astype(int)
print(f"Wav файл: {len(ref)}")

# ── BUILD INDEX: wav файл бүрийн X.npy start index ──────────────
# X.npy нь wav файлуудыг дарааллаар нь step=2000-аар сегментчилсэн
print("\nIndex тооцоож байна...")
wav_to_xidx = {}   # record_id -> (start_xi, end_xi)
xi = 0

for _, row in ref.iterrows():
    path = os.path.join(RAW_DIR, row.folder, f"{row.record_id}.wav")
    if not os.path.exists(path):
        wav_to_xidx[row.record_id] = None
        continue

    signal, _ = librosa.load(path, sr=SR, mono=True)
    n_segs = max(1, (len(signal) - SEG) // STEP + 1)
    wav_to_xidx[row.record_id] = (xi, xi + n_segs)
    xi += n_segs

print(f"Index дуусав. Нийт segments: {xi}  (X.npy: {len(X)})")

# ── PRECOMPUTE MEL ───────────────────────────────────────────────
demo_mels   = []
demo_labels = []
demo_ids    = []
failed      = []

for _, row in tqdm(ref.iterrows(), total=len(ref), desc="Mel гаргаж байна"):
    path = os.path.join(RAW_DIR, row.folder, f"{row.record_id}.wav")

    if not os.path.exists(path):
        failed.append(row.record_id)
        continue

    xrange = wav_to_xidx.get(row.record_id)
    if xrange is None:
        failed.append(row.record_id)
        continue

    start_xi, end_xi = xrange
    end_xi = min(end_xi, len(X))

    if start_xi >= end_xi:
        failed.append(row.record_id)
        continue

    # Тухайн wav-ийн X.npy segments-ийн дундаас
    # хамгийн өндөр RMS энерги бүхий segment авна
    best_rms = -1
    best_seg = None

    for xi_i in range(start_xi, end_xi):
        seg = X[xi_i].astype(np.float32)
        rms = np.sqrt(np.mean(seg ** 2))
        if rms > best_rms:
            best_rms = rms
            best_seg = seg

    if best_seg is None:
        failed.append(row.record_id)
        continue

    # Mel-spectrogram
    mel    = librosa.feature.melspectrogram(
        y=best_seg, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    demo_mels.append(mel_db)
    demo_labels.append(row.label)
    demo_ids.append(row.record_id)

# ── SAVE ─────────────────────────────────────────────────────────
demo_mels   = np.array(demo_mels)
demo_labels = np.array(demo_labels)
demo_ids    = np.array(demo_ids)

print(f"\nНийт   : {len(demo_mels)}")
print(f"Normal  : {(demo_labels==0).sum()}")
print(f"Abnormal: {(demo_labels==1).sum()}")
print(f"Failed  : {len(failed)}")
if failed:
    print(f"Failed files: {failed[:5]}")

np.save(os.path.join(FEAT_DIR, "demo_wav_mels.npy"),   demo_mels)
np.save(os.path.join(FEAT_DIR, "demo_wav_labels.npy"),  demo_labels)
np.save(os.path.join(FEAT_DIR, "demo_wav_ids.npy"),     demo_ids)

print(f"\nSaved:")
print(f"  {os.path.join(FEAT_DIR, 'demo_wav_mels.npy')}")
print(f"  {os.path.join(FEAT_DIR, 'demo_wav_labels.npy')}")
print(f"  {os.path.join(FEAT_DIR, 'demo_wav_ids.npy')}")

# ── VERIFY ───────────────────────────────────────────────────────
print("\n=== Шалгалт ===")
print(f"Mel mean (Normal)  : {demo_mels[demo_labels==0].mean():.2f}  (expected: ~-60.8)")
print(f"Mel mean (Abnormal): {demo_mels[demo_labels==1].mean():.2f}  (expected: ~-60.8)")