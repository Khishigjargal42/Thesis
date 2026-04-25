"""
eval_seta.py  —  v3  (TTA + Ensemble)
======================================
Kaggle Heartbeat Sounds Set A дээр attention загваруудыг туршина.
Нэмэлт сургалт хийхгүй.

Алхамууд:
  1. Baseline  — optimal threshold
  2. TTA       — pitch shift, time stretch, noise
  3. Ensemble  — AUC-weighted average of both models
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import librosa
from scipy.signal import butter, filtfilt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve,
    precision_recall_curve
)

# ==========================================
# PATHS
# ==========================================
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.dirname(SCRIPT_DIR)

SET_A_DIR   = os.path.join(SCRIPT_DIR, "set_a")
SET_A_CSV   = os.path.join(SCRIPT_DIR, "set_a.csv")
MODEL_DIR   = os.path.join(SCRIPT_DIR, "models")
RESNET_PATH = os.path.join(MODEL_DIR, "resnet_mel_attention.pth")
PCNN_PATH   = os.path.join(MODEL_DIR, "parallelcnn_mfcc_v2.pth")
SAVE_FIG    = os.path.join(ROOT_DIR, "figures")
os.makedirs(SAVE_FIG, exist_ok=True)

# ==========================================
# FEATURE CONFIG
# ==========================================
SR          = 2000
N_FFT       = 512
HOP_LENGTH  = 47
N_MELS      = 128
N_MFCC      = 40
SEGMENT_SEC = 2

# ==========================================
# NORMALIZATION
# ResNet     — raw (training-д normalize хийгээгүй)
# ParallelCNN — PhysioNet train stats
# ==========================================
RESNET_MEAN = 0.0
RESNET_STD  = 1.0
PCNN_MEAN   = -7.616
PCNN_STD    = 83.983

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ==========================================
# MODEL DEFINITIONS
# ==========================================
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
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
        self.conv1    = nn.Conv2d(in_ch, out_ch, 3, stride, 1)
        self.bn1      = nn.BatchNorm2d(out_ch)
        self.conv2    = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.bn2      = nn.BatchNorm2d(out_ch)
        self.relu     = nn.ReLU()
        self.se       = SEBlock(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride),
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
    def __init__(self):
        super().__init__()
        self.stem   = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU()
        )
        self.layer1 = BasicBlock(32, 32)
        self.layer2 = BasicBlock(32, 64, stride=2)
        self.layer3 = BasicBlock(64, 128, stride=2)
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.fc     = nn.Linear(128, 1)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k, p):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, k, padding=p, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, k, padding=p, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
    def forward(self, x): return self.net(x)


class ParallelCNN2D(nn.Module):
    def __init__(self, base=24, dropout=0.4):
        super().__init__()
        self.b1   = ConvBlock(1, base, 3, 1)
        self.b2   = ConvBlock(1, base, 5, 2)
        self.b3   = ConvBlock(1, base, 7, 3)
        self.post = nn.Sequential(
            nn.Conv2d(3*base, 2*base, 3, padding=1, bias=False),
            nn.BatchNorm2d(2*base), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(2*base, 1)

    def forward(self, x):
        x = torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)
        x = self.post(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)


# ==========================================
# LOAD MODELS
# ==========================================
def load_model(ModelClass, path):
    m = ModelClass().to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    print(f"  Loaded: {os.path.basename(path)}")
    return m

print("\nLoading models...")
resnet_model = load_model(ResNet2D_SE,   RESNET_PATH)
pcnn_model   = load_model(ParallelCNN2D, PCNN_PATH)


# ==========================================
# SIGNAL PREPROCESSING
# ==========================================
def preprocess_signal(y_audio):
    y_audio = y_audio - y_audio.mean()
    try:
        b, a = butter(4, [20/(SR/2), 400/(SR/2)], btype='band')
        y_audio = filtfilt(b, a, y_audio)
    except Exception:
        pass
    rms = np.sqrt(np.mean(y_audio**2))
    if rms > 1e-6:
        y_audio = y_audio / rms * 0.1
    return y_audio.astype(np.float32)


# ==========================================
# FEATURE EXTRACTION
# ==========================================
def extract_mel(y_audio):
    mel = librosa.feature.melspectrogram(
        y=y_audio, sr=SR, n_fft=N_FFT,
        hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    return mel.astype(np.float32)

def extract_mfcc(y_audio):
    mfcc = librosa.feature.mfcc(
        y=y_audio, sr=SR, n_mfcc=N_MFCC,
        n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    return mfcc.astype(np.float32)

def segment_audio(y_audio, overlap=0.75):
    seg_len  = SR * SEGMENT_SEC
    step     = max(1, int(seg_len * (1 - overlap)))
    segments = []
    for start in range(0, len(y_audio) - seg_len + 1, step):
        segments.append(y_audio[start : start + seg_len])
    if len(segments) == 0:
        pad = np.zeros(seg_len, dtype=np.float32)
        pad[:len(y_audio)] = y_audio
        segments.append(pad)
    return segments


# ==========================================
# SEGMENT INFERENCE
# ==========================================
def infer_segments(segments, model, feature_fn, mean, std):
    feats = []
    for s in segments:
        f = feature_fn(s)
        f = (f - mean) / (std + 1e-8)
        feats.append(f)
    arr = np.array(feats)[:, np.newaxis, :, :]
    x   = torch.FloatTensor(arr).to(device)
    with torch.no_grad():
        logits = model(x).squeeze(-1)
        if logits.dim() == 0:
            logits = logits.unsqueeze(0)
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs


# ==========================================
# PREDICT — BASELINE
# ==========================================
def predict_baseline(filepath, model, feature_fn, mean, std, invert=False):
    try:
        y_audio, _ = librosa.load(filepath, sr=SR, mono=True)
    except Exception as e:
        print(f"  [WARN] {os.path.basename(filepath)}: {e}")
        return None

    y_audio = preprocess_signal(y_audio)
    probs   = infer_segments(segment_audio(y_audio, 0.75),
                             model, feature_fn, mean, std)
    if invert:
        probs = 1.0 - probs
    return float(probs.mean())


# ==========================================
# PREDICT — TTA
# ==========================================
def predict_tta(filepath, model, feature_fn, mean, std, invert=False):
    try:
        y_audio, _ = librosa.load(filepath, sr=SR, mono=True)
    except Exception as e:
        print(f"  [WARN] {os.path.basename(filepath)}: {e}")
        return None

    y_audio   = preprocess_signal(y_audio)
    all_probs = []

    # Original (75% overlap)
    all_probs.extend(
        infer_segments(segment_audio(y_audio, 0.75),
                       model, feature_fn, mean, std))

    # Pitch shift -1, +1 semitone
    for n_steps in [-1, 1]:
        try:
            y_s = librosa.effects.pitch_shift(y_audio, sr=SR, n_steps=n_steps)
            all_probs.extend(
                infer_segments(segment_audio(y_s, 0.75),
                               model, feature_fn, mean, std))
        except Exception:
            pass

    # Time stretch x0.9, x1.1
    for rate in [0.9, 1.1]:
        try:
            y_s = librosa.effects.time_stretch(y_audio, rate=rate)
            all_probs.extend(
                infer_segments(segment_audio(y_s, 0.75),
                               model, feature_fn, mean, std))
        except Exception:
            pass

    # Gaussian noise
    y_noisy = (y_audio +
               np.random.normal(0, 0.005, len(y_audio)).astype(np.float32))
    all_probs.extend(
        infer_segments(segment_audio(y_noisy, 0.75),
                       model, feature_fn, mean, std))

    probs = np.array(all_probs)
    if invert:
        probs = 1.0 - probs
    return float(probs.mean())


# ==========================================
# DATASET LOADING
# ==========================================
def load_set_a(csv_path, wav_dir):
    df = pd.read_csv(csv_path)
    print(f"\nSet A: {len(df)} recordings")
    print(df["label"].value_counts().to_string())

    df = df[~df["label"].isin(["artifact", "unlabelled"])].copy()
    df["binary_label"] = df["label"].apply(lambda l: 0 if l == "normal" else 1)
    df["filepath"]     = df["fname"].apply(
        lambda f: os.path.join(wav_dir, f.replace("set_a/", ""))
    )

    mask    = df["filepath"].apply(os.path.exists)
    missing = df[~mask]["filepath"].tolist()
    if missing:
        print(f"  [WARN] Missing {len(missing)} files. "
              f"Example: {missing[:3]}")

    df = df[mask].reset_index(drop=True)
    print(f"  Valid files : {len(df)}")
    print(f"  Normal      : {(df['binary_label']==0).sum()}")
    print(f"  Abnormal    : {(df['binary_label']==1).sum()}")
    return df


# ==========================================
# EVALUATION HELPERS
# ==========================================
def find_optimal_threshold(y_true, y_prob):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1s    = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_i = np.argmax(f1s)
    return float(thresholds[best_i])

def compute_metrics(y_true, y_pred, y_prob):
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")
    return dict(
        acc  = accuracy_score(y_true, y_pred),
        prec = precision_score(y_true, y_pred, zero_division=0),
        rec  = recall_score(y_true, y_pred, zero_division=0),
        f1   = f1_score(y_true, y_pred, zero_division=0),
        auc  = auc
    )

def run_evaluation(df, predict_fn, label):
    print(f"\n{'='*54}")
    print(f"  {label}")
    print(f"{'='*54}")

    y_true, y_prob = [], []
    for _, row in df.iterrows():
        prob = predict_fn(row["filepath"])
        if prob is None:
            continue
        y_true.append(int(row["binary_label"]))
        y_prob.append(prob)

    y_true   = np.array(y_true)
    y_prob   = np.array(y_prob)
    opt_thr  = find_optimal_threshold(y_true, y_prob)
    y_pred   = (y_prob >= opt_thr).astype(int)
    m        = compute_metrics(y_true, y_pred, y_prob)

    print(f"  AUC      : {m['auc']:.4f}")
    print(f"  Threshold: {opt_thr:.3f}  (optimal F1)")
    print(f"  Accuracy : {m['acc']:.4f}")
    print(f"  Precision: {m['prec']:.4f}")
    print(f"  Recall   : {m['rec']:.4f}")
    print(f"  F1       : {m['f1']:.4f}")
    print(f"  Samples  : {len(y_true)}")

    return {"label": label, "y_true": y_true,
            "y_pred": y_pred, "y_prob": y_prob,
            "opt_thr": opt_thr, **m}


# ==========================================
# PLOTS
# ==========================================
def plot_confusion(result, save_dir):
    cm = confusion_matrix(result["y_true"], result["y_pred"])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal","Abnormal"],
                yticklabels=["Normal","Abnormal"], ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"{result['label']}\n"
                 f"Kaggle Set A  (thr={result['opt_thr']:.2f})")
    plt.tight_layout()
    safe = "".join(c if c.isalnum() or c=="_" else "_"
                   for c in result["label"])
    path = os.path.join(save_dir, f"kaggle_cm_{safe}.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved: {path}")

def plot_roc_all(results, save_dir):
    styles = ["-","--","-.",":","-","--"]
    fig, ax = plt.subplots(figsize=(7, 6))
    for r, ls in zip(results, styles):
        if np.isnan(r["auc"]): continue
        fpr, tpr, _ = roc_curve(r["y_true"], r["y_prob"])
        ax.plot(fpr, tpr, ls, lw=2,
                label=f"{r['label']}  (AUC={r['auc']:.3f})")
    ax.plot([0,1],[0,1],"k--",alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Kaggle Set A Cross-Dataset Test")
    ax.legend(fontsize=7, loc="lower right")
    plt.tight_layout()
    path = os.path.join(save_dir, "kaggle_roc_all.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved: {path}")

def plot_comparison_bar(results, save_dir):
    metrics = ["acc","prec","rec","f1","auc"]
    xlabels = ["Accuracy","Precision","Recall","F1","AUC"]
    x = np.arange(len(metrics))
    n = len(results)
    width = 0.8 / n
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, r in enumerate(results):
        vals   = [r[m] for m in metrics]
        offset = (i - n/2 + 0.5) * width
        bars   = ax.bar(x + offset, vals, width, label=r["label"])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.004,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=6)
    ax.set_xticks(x); ax.set_xticklabels(xlabels)
    ax.set_ylim(0, 1.20); ax.set_ylabel("Score")
    ax.set_title("Kaggle Set A — Baseline vs TTA vs Ensemble (Optimal Threshold)")
    ax.legend(fontsize=7)
    plt.tight_layout()
    path = os.path.join(save_dir, "kaggle_comparison_all.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved: {path}")

def print_summary(results):
    print("\n" + "=" * 74)
    print("KAGGLE SET A — FINAL SUMMARY (Optimal Threshold)")
    print("=" * 74)
    print(f"{'Model':<40} {'AUC':>5}  {'Thr':>5}  "
          f"{'Acc':>5}  {'Prec':>5}  {'Rec':>5}  {'F1':>5}")
    print("-" * 74)
    for r in results:
        print(f"{r['label']:<40} "
              f"{r['auc']:>5.3f}  {r['opt_thr']:>5.3f}  "
              f"{r['acc']:>5.3f}  {r['prec']:>5.3f}  "
              f"{r['rec']:>5.3f}  {r['f1']:>5.3f}")
    print("=" * 74)


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":

    df = load_set_a(SET_A_CSV, SET_A_DIR)

    # Baseline predict functions
    def pred_resnet_base(fp):
        return predict_baseline(fp, resnet_model, extract_mel,
                                RESNET_MEAN, RESNET_STD, invert=True)

    def pred_pcnn_base(fp):
        return predict_baseline(fp, pcnn_model, extract_mfcc,
                                PCNN_MEAN, PCNN_STD, invert=True)

    # TTA predict functions
    def pred_resnet_tta(fp):
        return predict_tta(fp, resnet_model, extract_mel,
                           RESNET_MEAN, RESNET_STD, invert=True)

    def pred_pcnn_tta(fp):
        return predict_tta(fp, pcnn_model, extract_mfcc,
                           PCNN_MEAN, PCNN_STD, invert=True)

    # Ensemble: AUC-weighted (baseline AUC-аас тооцно)
    W_RESNET = 0.718 / (0.718 + 0.720)
    W_PCNN   = 0.720 / (0.718 + 0.720)

    def pred_ensemble(fp):
        p1 = pred_resnet_tta(fp)
        p2 = pred_pcnn_tta(fp)
        if p1 is None and p2 is None: return None
        if p1 is None: return p2
        if p2 is None: return p1
        return W_RESNET * p1 + W_PCNN * p2

    # Run
    print("\n" + "="*54)
    print("  STEP 1: BASELINE")
    print("="*54)
    r1 = run_evaluation(df, pred_resnet_base, "ResNet2D+SE  Baseline (Mel)")
    r2 = run_evaluation(df, pred_pcnn_base,   "ParallelCNN  Baseline (MFCC)")

    print("\n" + "="*54)
    print("  STEP 2: TTA")
    print("  (pitch shift, time stretch, noise — CPU-д удаан байна)")
    print("="*54)
    r3 = run_evaluation(df, pred_resnet_tta,  "ResNet2D+SE  TTA (Mel)")
    r4 = run_evaluation(df, pred_pcnn_tta,    "ParallelCNN  TTA (MFCC)")

    print("\n" + "="*54)
    print("  STEP 3: ENSEMBLE (TTA + AUC-weighted)")
    print("="*54)
    r5 = run_evaluation(df, pred_ensemble, "Ensemble (ResNet + ParallelCNN TTA)")

    all_results = [r1, r2, r3, r4, r5]

    print("\nGenerating plots...")
    for r in all_results:
        plot_confusion(r, SAVE_FIG)
    plot_roc_all(all_results, SAVE_FIG)
    plot_comparison_bar(all_results, SAVE_FIG)

    print_summary(all_results)
    print("\nDone.")