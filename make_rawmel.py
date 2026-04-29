"""
make_rawmel.py
Raw wav файлуудаас шууд mel-spectrogram гаргаж хадгална.
Сургалтанд ашиглаагүй гаднаас ирсэн wav файл дээр
ажиллах model сургахад хэрэгтэй.
"""

import os
import numpy as np
import librosa
import pandas as pd
from tqdm import tqdm

# ── CONFIG ───────────────────────────────────────────────────────
SR         = 2000
SEG        = 4000
STEP       = 2000   # 50% overlap
N_FFT      = 512
HOP_LENGTH = 256
N_MELS     = 128

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR  = os.path.join(BASE_DIR, "data", "raw")
FEAT_DIR = os.path.join(BASE_DIR, "data", "features")
os.makedirs(FEAT_DIR, exist_ok=True)

# ── REFERENCES ───────────────────────────────────────────────────
ref = pd.read_csv(
    os.path.join(RAW_DIR, "REFERENCES.csv"),
    header=None, skiprows=1
)
ref.columns = ["record_id", "folder", "label"]
ref["label"] = ref["label"].astype(int)

print(f"Нийт wav файл : {len(ref)}")
print(f"Normal        : {(ref.label==0).sum()}")
print(f"Abnormal      : {(ref.label==1).sum()}")

# ── FEATURE EXTRACTION ───────────────────────────────────────────
all_mels   = []
all_labels = []
all_ids    = []
failed     = []

for _, row in tqdm(ref.iterrows(), total=len(ref), desc="Mel гаргаж байна"):
    wav_path = os.path.join(RAW_DIR, row.folder, f"{row.record_id}.wav")

    if not os.path.exists(wav_path):
        failed.append(row.record_id)
        continue

    signal, _ = librosa.load(wav_path, sr=SR, mono=True)

    # Step=2000 overlap segments
    segments = []
    for start in range(0, len(signal) - SEG + 1, STEP):
        segments.append(signal[start:start + SEG])

    # Богино файл бол pad хийж нэг segment авна
    if not segments:
        segments = [np.pad(signal, (0, SEG - len(signal)))]

    # Сегмент бүрээс mel гаргана
    for seg in segments:
        mel    = librosa.feature.melspectrogram(
            y=seg, sr=SR,
            n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        all_mels.append(mel_db)
        all_labels.append(row.label)
        all_ids.append(row.record_id)

all_mels   = np.array(all_mels,   dtype=np.float32)
all_labels = np.array(all_labels, dtype=np.int32)
all_ids    = np.array(all_ids)

print(f"\nНийт segments : {len(all_mels)}")
print(f"Normal        : {(all_labels==0).sum()}")
print(f"Abnormal      : {(all_labels==1).sum()}")
print(f"Failed        : {len(failed)}")
print(f"Mel mean      : {all_mels.mean():.4f}")
print(f"Mel std       : {all_mels.std():.4f}")
print(f"Mel shape     : {all_mels.shape}")

# Хадгалах
np.save(os.path.join(FEAT_DIR, "rawmel_features.npy"), all_mels)
np.save(os.path.join(FEAT_DIR, "rawmel_labels.npy"),   all_labels)
np.save(os.path.join(FEAT_DIR, "rawmel_ids.npy"),      all_ids)

print(f"\nSaved:")
print(f"  {os.path.join(FEAT_DIR, 'rawmel_features.npy')}")
print(f"  {os.path.join(FEAT_DIR, 'rawmel_labels.npy')}")
print(f"  {os.path.join(FEAT_DIR, 'rawmel_ids.npy')}")