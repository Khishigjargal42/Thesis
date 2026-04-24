"""
=============================================================
CirCor DigiScope 2022 — Cross-Dataset Evaluation
Model  : Parallel CNN + MFCC + SE Attention
Trained: PhysioNet 2016
Tested : PhysioNet 2022 (CirCor DigiScope)
=============================================================

Dataset structure expected:
  circor2022/
    training_data/
      XXXXX_AV.wav
      XXXXX_AV.hea        ← label is inside .hea or .tsv
      ...
    training_data.csv     ← OR labels from this CSV

Label mapping (CirCor 2022):
  "Present"  → 1 (Abnormal)
  "Absent"   → 0 (Normal)
  "Unknown"  → skip

Run:
  py evaluate_circor2022.py
=============================================================
"""

import os
import csv
import glob
import warnings
import numpy as np
import librosa
import torch
import torch.nn as nn
from scipy.signal import butter, filtfilt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# =============================================================
# CONFIG  — шаардлагатай бол замуудаа өөрчилнө үү
# =============================================================
MODEL_PATH   = "parallelcnn_mfcc_attention.pth"
DATA_DIR     = "circor/training_data"      # .wav + .hea файлууд
CSV_PATH     = "circor/training_data.csv"  # CirCor-ийн label CSV
OUTPUT_DIR   = "circor2022_results"

# PhysioNet 2016-ийн train статистик (загвар сургалтын үед ашигласан)
MU    = -4.220855349111927
SIGMA = 66.5658466625049

N_MFCC     = 40
MAX_LEN    = 16
HOP_LENGTH = 125        # 2000 samples / 125 = 16 frames

THRESHOLD  = 0.3
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================
# MODEL  (сургасантай яг ижил архитектур)
# =============================================================
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
            nn.Linear(1280, 128),
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


# =============================================================
# LABEL LOADER
# Strategy 1: CSV файлаас уншина (CirCor-ийн training_data.csv)
# Strategy 2: .hea файлаас уншина
# =============================================================
def load_labels_from_csv(csv_path):
    """
    CirCor 2022 CSV хэлбэр:
      Patient ID, ..., Murmur
      Murmur column: "Present" | "Absent" | "Unknown"
    Returns: dict { patient_id (str) -> label (int) }
    """
    labels = {}
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid    = str(row.get("Patient ID", "")).strip()
            murmur = str(row.get("Murmur", "")).strip()
            if murmur == "Present":
                labels[pid] = 1
            elif murmur == "Absent":
                labels[pid] = 0
            # Unknown → skip
    return labels


def load_label_from_hea(hea_path):
    """
    .hea файлаас label унших (fallback)
    CirCor .hea: # Murmur: Present / Absent
    """
    with open(hea_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                low = line.lower()
                if "present" in low:
                    return 1
                elif "absent" in low:
                    return 0
    return None


# =============================================================
# AUDIO PREPROCESSING
# =============================================================
def load_audio(path):
    y, sr = librosa.load(path, sr=None)
    # CirCor: 4000 Hz → 2000 Hz
    if sr != 2000:
        y  = librosa.resample(y, orig_sr=sr, target_sr=2000)
        sr = 2000
    return y, sr


def bandpass_filter(y, sr):
    b, a = butter(4, [20 / (sr / 2), 400 / (sr / 2)], btype='band')
    return filtfilt(b, a, y)


def normalize_signal(y):
    m = np.max(np.abs(y))
    return y / m if m > 0 else y


def segment_audio(y, sr):
    size = int(2 * sr)       # 4000 samples
    step = int(size * 0.5)   # 2000 samples
    segs = []
    for i in range(0, len(y) - size + 1, step):
        seg = y[i:i + size]
        if np.mean(seg ** 2) > 1e-8:
            segs.append(seg)
    return segs


def extract_mfcc(seg, sr):
    mfcc = librosa.feature.mfcc(
        y=seg, sr=sr,
        n_mfcc=N_MFCC,
        hop_length=HOP_LENGTH
    )
    if mfcc.shape[1] < MAX_LEN:
        pad  = MAX_LEN - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad)), mode='constant')
    else:
        mfcc = mfcc[:, :MAX_LEN]
    return mfcc  # (40, 16)


def predict_file(model, wav_path, sr_source=4000):
    """Нэг .wav файлын prediction score буцаана."""
    try:
        y, sr = load_audio(wav_path)
        y     = bandpass_filter(y, sr)
        y     = normalize_signal(y)
        segs  = segment_audio(y, sr)

        if not segs:
            return None

        probs = []
        for seg in segs:
            mfcc = extract_mfcc(seg, sr)
            mfcc = (mfcc - MU) / SIGMA
            x    = torch.tensor(mfcc, dtype=torch.float32)\
                        .unsqueeze(0).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                prob = torch.sigmoid(model(x)).item()
            probs.append(prob)

        return float(np.mean(probs))

    except Exception as e:
        print(f"   ⚠ Error processing {os.path.basename(wav_path)}: {e}")
        return None


# =============================================================
# METRICS & PLOTS
# =============================================================
def print_metrics(y_true, y_pred, y_score):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, y_score)
    cm   = confusion_matrix(y_true, y_pred)

    print("\n" + "=" * 50)
    print("   CirCor 2022 — Cross-Dataset Evaluation")
    print("=" * 50)
    print(f"  Files evaluated : {len(y_true)}")
    print(f"  Normal  (0)     : {sum(1 for l in y_true if l==0)}")
    print(f"  Abnormal(1)     : {sum(1 for l in y_true if l==1)}")
    print("-" * 50)
    print(f"  Accuracy        : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision       : {prec:.4f}")
    print(f"  Recall          : {rec:.4f}")
    print(f"  F1 Score        : {f1:.4f}")
    print(f"  AUC-ROC         : {auc:.4f}")
    print("-" * 50)
    print("  Confusion Matrix:")
    print(f"           Pred N   Pred Ab")
    print(f"  True N   {cm[0,0]:5d}    {cm[0,1]:5d}")
    print(f"  True Ab  {cm[1,0]:5d}    {cm[1,1]:5d}")
    print("=" * 50)

    return {"accuracy": acc, "precision": prec, "recall": rec,
            "f1": f1, "auc": auc, "cm": cm}


def save_plots(y_true, y_score, metrics):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- ROC Curve ---
    fpr, tpr, _ = roc_curve(y_true, y_score)
    axes[0].plot(fpr, tpr, color='steelblue', lw=2,
                 label=f"AUC = {metrics['auc']:.4f}")
    axes[0].plot([0,1],[0,1],'k--', lw=1)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve — CirCor 2022")
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.3)

    # --- Confusion Matrix ---
    cm = metrics['cm']
    im = axes[1].imshow(cm, interpolation='nearest', cmap='Blues')
    axes[1].figure.colorbar(im, ax=axes[1])
    classes = ["Normal", "Abnormal"]
    axes[1].set(xticks=[0,1], yticks=[0,1],
                xticklabels=classes, yticklabels=classes,
                ylabel="True label", xlabel="Predicted label",
                title="Confusion Matrix — CirCor 2022")
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, str(cm[i,j]),
                        ha="center", va="center",
                        color="white" if cm[i,j] > cm.max()/2 else "black",
                        fontsize=14, fontweight='bold')

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "circor2022_results.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\n  📊 Plot saved → {out}")


def save_per_file_csv(records):
    out = os.path.join(OUTPUT_DIR, "per_file_scores.csv")
    with open(out, "w", newline='') as f:
        w = csv.writer(f)
        w.writerow(["filename", "true_label", "score", "predicted_label", "correct"])
        for r in records:
            w.writerow(r)
    print(f"  📄 Per-file scores → {out}")


# =============================================================
# MAIN
# =============================================================
def main():
    print("=" * 50)
    print("  CirCor 2022 Cross-Dataset Evaluation")
    print(f"  Device : {DEVICE}")
    print("=" * 50)

    # 1. Load model
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        return

    print("\n🔹 Loading model...")
    model = ParallelCNN_SE().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("   ✅ Model loaded")

    # 2. Load labels
    print("🔹 Loading labels...")
    label_dict = {}

    if os.path.exists(CSV_PATH):
        label_dict = load_labels_from_csv(CSV_PATH)
        print(f"   ✅ Labels from CSV: {len(label_dict)} patients")
    else:
        print(f"   ⚠ CSV not found ({CSV_PATH}), trying .hea files...")

    # 3. Find wav files
    wav_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.wav")))
    print(f"🔹 Found {len(wav_files)} .wav files in {DATA_DIR}")

    if not wav_files:
        print("❌ No .wav files found. Check DATA_DIR path.")
        return

    # 4. Run inference
    print("🔹 Running inference...\n")

    y_true, y_score = [], []
    records = []
    skipped = 0

    for i, wav_path in enumerate(wav_files):
        fname  = os.path.basename(wav_path)
        # Patient ID: "12345_AV.wav" → "12345"
        pid    = fname.split("_")[0]

        # Label lookup
        label = label_dict.get(pid)

        if label is None:
            # Fallback: .hea файлаас
            hea_path = wav_path.replace(".wav", ".hea")
            if os.path.exists(hea_path):
                label = load_label_from_hea(hea_path)

        if label is None:
            skipped += 1
            continue

        # Predict
        score = predict_file(model, wav_path)
        if score is None:
            skipped += 1
            continue

        pred    = 1 if score > THRESHOLD else 0
        correct = (pred == label)

        y_true.append(label)
        y_score.append(score)
        records.append([fname, label, round(score, 4), pred, int(correct)])

        # Progress
        status = "✅" if correct else "❌"
        tag    = "Ab" if label == 1 else "No"
        if (i + 1) % 50 == 0 or i < 5:
            print(f"  [{i+1:4d}] {fname:<25s} true={tag}  score={score:.3f}  {status}")

    print(f"\n  Processed : {len(y_true)}  |  Skipped: {skipped}")

    if len(y_true) < 2:
        print("❌ Not enough labeled files to compute metrics.")
        return

    # 5. Metrics
    y_true  = np.array(y_true)
    y_score = np.array(y_score)
    y_pred  = (y_score > THRESHOLD).astype(int)

    metrics = print_metrics(y_true, y_pred, y_score)

    # 6. Save outputs
    save_plots(y_true, y_score, metrics)
    save_per_file_csv(records)

    # 7. Compare with PhysioNet 2016 results
    print("\n  📌 PhysioNet 2016 (train domain) vs CirCor 2022 (cross-domain):")
    print(f"  {'Metric':<12} {'2016':>8} {'2022':>8}")
    print(f"  {'-'*30}")
    ref = {"accuracy": 0.904, "precision": 0.7144,
           "recall": 0.986, "f1": 0.8285, "auc": 0.9808}
    for k in ["accuracy", "precision", "recall", "f1", "auc"]:
        arrow = "↑" if metrics[k] >= ref[k] else "↓"
        print(f"  {k:<12} {ref[k]:>8.4f} {metrics[k]:>8.4f}  {arrow}")
    print()


if __name__ == "__main__":
    main()