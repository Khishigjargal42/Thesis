import os
import librosa
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import butter, filtfilt

# =========================
# CONFIG
# =========================
MODEL_PATH = "parallelcnn_mfcc_attention.pth"
FILE_PATH  = "circor/training_data/36327_AV.wav"

MU    = -4.220855349111927
SIGMA = 66.5658466625049

N_MFCC   = 40
MAX_LEN  = 16
HOP_LENGTH = 125   # 2000 samples / 125 = 16 frames → (40,16) ✅

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# MODEL
# =========================
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


class ParallelCNN_SE(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), SEBlock(16), nn.MaxPool2d(2)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2), nn.ReLU(), SEBlock(16), nn.MaxPool2d(2)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(1, 16, 7, padding=3), nn.ReLU(), SEBlock(16), nn.MaxPool2d(2)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(48, 32, 3, padding=1), nn.ReLU(), SEBlock(32), nn.MaxPool2d(2)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(1280, 128),  # 32 × 10 × 4 = 1280
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        x  = torch.cat([b1, b2, b3], dim=1)
        x  = self.conv(x)
        x  = self.flatten(x)
        x  = self.fc(x)
        return x


# =========================
# PREPROCESSING
# =========================
def load_audio(path):
    y, sr = librosa.load(path, sr=None)
    if sr != 2000:
        y  = librosa.resample(y, orig_sr=sr, target_sr=2000)
        sr = 2000
    return y, sr


def bandpass_filter(y, sr):
    b, a = butter(4, [20 / (sr / 2), 400 / (sr / 2)], btype='band')
    return filtfilt(b, a, y)


def normalize_signal(y):
    max_val = np.max(np.abs(y))
    return y / max_val if max_val > 0 else y


def segment_audio(y, sr):
    size = int(2 * sr)          # 4000 samples
    step = int(size * 0.5)      # 2000 samples
    segments = []
    for i in range(0, len(y) - size + 1, step):
        seg = y[i:i + size]
        if np.mean(seg ** 2) > 1e-8:
            segments.append(seg)
    return segments


def extract_mfcc(seg, sr):
    mfcc = librosa.feature.mfcc(
        y=seg, sr=sr,
        n_mfcc=N_MFCC,
        hop_length=HOP_LENGTH   # 2000/125 = 16 frames
    )
    # shape guarantee: (40, 16)
    if mfcc.shape[1] < MAX_LEN:
        pad  = MAX_LEN - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad)), mode='constant')
    else:
        mfcc = mfcc[:, :MAX_LEN]
    return mfcc


def normalize_mfcc(mfcc):
    return (mfcc - MU) / SIGMA


# =========================
# MAIN
# =========================
def main():
    print("=" * 45)
    print(" Heart Sound Abnormality Detector")
    print("=" * 45)
    print(f" Model : {MODEL_PATH}")
    print(f" Audio : {FILE_PATH}")
    print("=" * 45)

    if not os.path.exists(MODEL_PATH):
        print("❌ Model file not found:", MODEL_PATH)
        return
    if not os.path.exists(FILE_PATH):
        print("❌ Audio file not found:", FILE_PATH)
        return

    # --- Load model ---
    print("\n🔹 Loading model...")
    model = ParallelCNN_SE().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("   ✅ Done")

    # --- Load audio ---
    print("🔹 Loading audio...")
    y, sr = load_audio(FILE_PATH)
    print(f"   Sample rate : {sr} Hz")
    print(f"   Duration    : {len(y)/sr:.1f} sec  ({len(y)} samples)")

    print("🔹 Bandpass filtering (20–400 Hz)...")
    y = bandpass_filter(y, sr)

    print("🔹 Normalizing amplitude...")
    y = normalize_signal(y)

    print("🔹 Segmenting (2s window, 50% overlap)...")
    segments = segment_audio(y, sr)
    print(f"   Segments : {len(segments)}")

    if not segments:
        print("❌ No valid segments found.")
        return

    # --- Inference ---
    print("🔹 Running inference...")
    preds = []
    for seg in segments:
        mfcc = extract_mfcc(seg, sr)      # (40, 16)
        mfcc = normalize_mfcc(mfcc)
        x    = torch.tensor(mfcc, dtype=torch.float32)\
                     .unsqueeze(0).unsqueeze(0).to(device)  # (1,1,40,16)
        with torch.no_grad():
            logit = model(x)
            prob  = torch.sigmoid(logit).item()
        preds.append(prob)

    final_score = float(np.mean(preds))
    std_score   = float(np.std(preds))

    # --- Result ---
    print("\n" + "=" * 45)
    print("               RESULT")
    print("=" * 45)
    print(f"  Segments analysed : {len(preds)}")
    print(f"  Mean score        : {final_score:.4f}")
    print(f"  Std               : {std_score:.4f}")
    print(f"  Threshold         : 0.50")
    print("-" * 45)
    if final_score > 0.5:
        print("  Prediction  →  ❗ ABNORMAL")
    else:
        print("  Prediction  →  ✅ NORMAL")
    print("=" * 45)


if __name__ == "__main__":
    main()