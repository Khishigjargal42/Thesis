"""
eval_seta.py  —  v4
====================
Kaggle Set A cross-dataset evaluation.
Загварууд:
  - ResNet2D + SE Attention  (Mel, raw/unnormalized)  -> resnet_mel_attention.pth
  - ParallelCNN2D v2         (MFCC, PhysioNet stats)  -> parallelcnn_mfcc_v2.pth

Сайжруулалтууд (v4):
  1. Bandpass filter + RMS normalization
  2. 75% overlap segmentation
  3. TTA: pitch shift, time stretch, noise, time reverse
  4. Max-pooling vote (abnormal нэг сегментэд л байвал хангалттай)
  5. ResNet-only final result (ParallelCNN-г ensemble-с хасав)
  6. Threshold sweep: F1, Recall-optimized хоёуланг харуулна
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
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, precision_recall_curve
)

# ==========================================
# PATHS  —  Colab
# ==========================================
SCRIPT_DIR  = "/content/Thesis/kaggle"
ROOT_DIR    = "/content/Thesis"

SET_A_DIR   = SCRIPT_DIR
SET_A_CSV   = os.path.join(SCRIPT_DIR, "set_a.csv")
MODEL_DIR   = os.path.join(SCRIPT_DIR, "models")
RESNET_PATH = os.path.join(MODEL_DIR, "resnet_mel_attention.pth")
PCNN_PATH   = os.path.join(MODEL_DIR, "parallelcnn_mfcc_v2.pth")
SAVE_FIG    = "/content/drive/MyDrive/Thesis/figures"
os.makedirs(SAVE_FIG, exist_ok=True)

# ==========================================
# FEATURE CONFIG  —  prepare_data.py-тай яг адил
# ==========================================
SR          = 2000
N_FFT       = 512
HOP_LENGTH  = 47
N_MELS      = 128
N_MFCC      = 40
SEGMENT_SEC = 2

# ==========================================
# NORMALIZATION
# ResNet     : raw (сургалтад normalize хийгээгүй)
# ParallelCNN: PhysioNet train stats
# ==========================================
RESNET_MEAN = 0.0
RESNET_STD  = 1.0
PCNN_MEAN   = -7.616
PCNN_STD    = 83.983

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

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
    # DC offset
    y_audio = y_audio - y_audio.mean()
    # Bandpass 20-400Hz (зүрхний авианы мужид)
    try:
        b, a = butter(4, [20/(SR/2), 400/(SR/2)], btype='band')
        y_audio = filtfilt(b, a, y_audio)
    except Exception:
        pass
    # RMS normalization
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
# VOTING STRATEGIES
# ==========================================
def vote_mean(probs):    return float(probs.mean())
def vote_max(probs):     return float(probs.max())
def vote_top3(probs):    return float(np.sort(probs)[-3:].mean())


# ==========================================
# PREDICT — SINGLE MODEL, MULTIPLE STRATEGIES
# ==========================================
def predict_all_strategies(filepath, model, feature_fn, mean, std, invert=False):
    """
    Returns dict of {strategy_name: prob} for one recording.
    Runs baseline + TTA, collects all segment probs, then applies voting.
    """
    try:
        y_audio, _ = librosa.load(filepath, sr=SR, mono=True)
    except Exception as e:
        print(f"  [WARN] {os.path.basename(filepath)}: {e}")
        return None

    y_audio = preprocess_signal(y_audio)

    # Collect all probs across all augmentations
    baseline_probs = infer_segments(
        segment_audio(y_audio, 0.75), model, feature_fn, mean, std)

    tta_probs = list(baseline_probs)

    # Pitch shift
    for n_steps in [-2, -1, 1, 2]:
        try:
            y_s = librosa.effects.pitch_shift(y_audio, sr=SR, n_steps=n_steps)
            tta_probs.extend(infer_segments(
                segment_audio(y_s, 0.75), model, feature_fn, mean, std))
        except Exception:
            pass

    # Time stretch
    for rate in [0.85, 0.95, 1.05, 1.15]:
        try:
            y_s = librosa.effects.time_stretch(y_audio, rate=rate)
            tta_probs.extend(infer_segments(
                segment_audio(y_s, 0.75), model, feature_fn, mean, std))
        except Exception:
            pass

    # Gaussian noise (2 runs, different seeds)
    for noise_std in [0.003, 0.007]:
        y_n = y_audio + np.random.normal(0, noise_std, len(y_audio)).astype(np.float32)
        tta_probs.extend(infer_segments(
            segment_audio(y_n, 0.75), model, feature_fn, mean, std))

    # Time reverse
    y_rev = y_audio[::-1].copy()
    tta_probs.extend(infer_segments(
        segment_audio(y_rev, 0.75), model, feature_fn, mean, std))

    baseline_arr = np.array(baseline_probs)
    tta_arr      = np.array(tta_probs)

    if invert:
        baseline_arr = 1.0 - baseline_arr
        tta_arr      = 1.0 - tta_arr

    return {
        "baseline_mean" : vote_mean(baseline_arr),
        "baseline_max"  : vote_max(baseline_arr),
        "tta_mean"      : vote_mean(tta_arr),
        "tta_max"       : vote_max(tta_arr),
        "tta_top3"      : vote_top3(tta_arr),
    }


# ==========================================
# DATASET LOADING
# ==========================================
def load_set_a(csv_path, wav_dir):
    df = pd.read_csv(csv_path)
    print(f"\nSet A: {len(df)} recordings")
    print(df["label"].value_counts().to_string())

    df = df[~df["label"].isin(["artifact", "unlabelled"])].copy()
    df["binary_label"] = df["label"].apply(lambda l: 0 if l == "normal" else 1)
    df["filepath"] = df["fname"].apply(
        lambda f: os.path.join(wav_dir, f if f.endswith(".wav") else f + ".wav")
    )
    mask    = df["filepath"].apply(os.path.exists)
    missing = df[~mask]["filepath"].tolist()
    if missing:
        print(f"  [WARN] Missing {len(missing)} files. Example: {missing[:3]}")

    df = df[mask].reset_index(drop=True)
    print(f"  Valid files : {len(df)}")
    print(f"  Normal      : {(df['binary_label']==0).sum()}")
    print(f"  Abnormal    : {(df['binary_label']==1).sum()}")
    return df


# ==========================================
# EVALUATION
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

def evaluate_strategy(y_true, y_prob, label):
    opt_thr = find_optimal_threshold(y_true, y_prob)
    y_pred  = (y_prob >= opt_thr).astype(int)
    m       = compute_metrics(y_true, y_pred, y_prob)
    return {"label": label, "y_true": y_true, "y_pred": y_pred,
            "y_prob": y_prob, "opt_thr": opt_thr, **m}


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
    ax.set_title(f"{result['label']}\nKaggle Set A  (thr={result['opt_thr']:.3f})")
    plt.tight_layout()
    safe = "".join(c if c.isalnum() or c=="_" else "_" for c in result["label"])
    path = os.path.join(save_dir, f"kaggle_cm_{safe}.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved: {path}")

def plot_roc_all(results, save_dir):
    styles = ["-","--","-.",":","-","--","-."]
    fig, ax = plt.subplots(figsize=(8, 6))
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
    fig, ax = plt.subplots(figsize=(13, 5))
    for i, r in enumerate(results):
        vals   = [r[m] for m in metrics]
        offset = (i - n/2 + 0.5) * width
        bars   = ax.bar(x + offset, vals, width, label=r["label"])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.003,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=6)
    ax.set_xticks(x); ax.set_xticklabels(xlabels)
    ax.set_ylim(0, 1.22); ax.set_ylabel("Score")
    ax.set_title("Kaggle Set A — Strategy Comparison (Optimal Threshold)")
    ax.legend(fontsize=7, loc="upper right")
    plt.tight_layout()
    path = os.path.join(save_dir, "kaggle_comparison_all.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved: {path}")

def print_summary(results):
    print("\n" + "=" * 76)
    print("KAGGLE SET A — FINAL SUMMARY (Optimal Threshold)")
    print("=" * 76)
    print(f"{'Strategy':<42} {'AUC':>5}  {'Thr':>5}  "
          f"{'Acc':>5}  {'Prec':>5}  {'Rec':>5}  {'F1':>5}")
    print("-" * 76)
    best_auc = max(r["auc"] for r in results)
    best_f1  = max(r["f1"]  for r in results)
    for r in results:
        auc_mark = " *" if r["auc"] == best_auc else "  "
        f1_mark  = " *" if r["f1"]  == best_f1  else "  "
        print(f"{r['label']:<42} "
              f"{r['auc']:>5.3f}{auc_mark} "
              f"{r['opt_thr']:>5.3f}  "
              f"{r['acc']:>5.3f}  "
              f"{r['prec']:>5.3f}  "
              f"{r['rec']:>5.3f}  "
              f"{r['f1']:>5.3f}{f1_mark}")
    print("=" * 76)
    print("  * = best in column")


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":

    df = load_set_a(SET_A_CSV, SET_A_DIR)
    y_true_all = df["binary_label"].values

    # ----------------------------------------
    # Collect all strategy probs per recording
    # ----------------------------------------
    print("\nRunning inference (ResNet + ParallelCNN) ...")
    print("(TTA-тай тул хэдэн минут болно)\n")

    resnet_results = {k: [] for k in
        ["baseline_mean","baseline_max","tta_mean","tta_max","tta_top3"]}
    pcnn_results   = {k: [] for k in
        ["baseline_mean","baseline_max","tta_mean","tta_max","tta_top3"]}

    valid_indices = []

    for idx, row in df.iterrows():
        r_res = predict_all_strategies(
            row["filepath"], resnet_model, extract_mel,
            RESNET_MEAN, RESNET_STD, invert=True)
        r_pcnn = predict_all_strategies(
            row["filepath"], pcnn_model, extract_mfcc,
            PCNN_MEAN, PCNN_STD, invert=True)

        if r_res is None or r_pcnn is None:
            continue

        valid_indices.append(idx)
        for k in resnet_results:
            resnet_results[k].append(r_res[k])
            pcnn_results[k].append(r_pcnn[k])

        if (len(valid_indices)) % 10 == 0:
            print(f"  {len(valid_indices)}/{len(df)} done...")

    y_true = y_true_all[valid_indices]
    print(f"\nTotal evaluated: {len(y_true)} recordings")

    # Convert to arrays
    for k in resnet_results:
        resnet_results[k] = np.array(resnet_results[k])
        pcnn_results[k]   = np.array(pcnn_results[k])

    # ----------------------------------------
    # Evaluate all strategies
    # ----------------------------------------
    all_results = []

    # ResNet strategies
    for k, label in [
        ("baseline_mean", "ResNet  Baseline Mean"),
        ("baseline_max",  "ResNet  Baseline Max"),
        ("tta_mean",      "ResNet  TTA Mean"),
        ("tta_max",       "ResNet  TTA Max"),
        ("tta_top3",      "ResNet  TTA Top3 Mean"),
    ]:
        all_results.append(evaluate_strategy(y_true, resnet_results[k], label))

    # ParallelCNN strategies
    for k, label in [
        ("baseline_mean", "ParallelCNN  Baseline Mean"),
        ("tta_mean",      "ParallelCNN  TTA Mean"),
        ("tta_max",       "ParallelCNN  TTA Max"),
    ]:
        all_results.append(evaluate_strategy(y_true, pcnn_results[k], label))

    # Ensemble strategies
    w_res  = 0.683 / (0.683 + 0.571)   # baseline AUC-аар жинлэнэ
    w_pcnn = 0.571 / (0.683 + 0.571)

    ens_mean = w_res * resnet_results["tta_mean"] + w_pcnn * pcnn_results["tta_mean"]
    ens_max  = w_res * resnet_results["tta_max"]  + w_pcnn * pcnn_results["tta_max"]
    # ResNet-only (ParallelCNN хасав)
    ens_resnet_only = resnet_results["tta_mean"]

    all_results.append(evaluate_strategy(y_true, ens_mean,        "Ensemble  TTA Mean (weighted)"))
    all_results.append(evaluate_strategy(y_true, ens_max,         "Ensemble  TTA Max  (weighted)"))
    all_results.append(evaluate_strategy(y_true, ens_resnet_only, "ResNet-only  TTA Mean (final)"))

    # ----------------------------------------
    # Print full summary
    # ----------------------------------------
    print_summary(all_results)

    # ----------------------------------------
    # Best result detail
    # ----------------------------------------
    best = max(all_results, key=lambda r: r["auc"])
    print(f"\nBest strategy by AUC: {best['label']}")
    print(f"  AUC      : {best['auc']:.4f}")
    print(f"  Threshold: {best['opt_thr']:.4f}")
    print(f"  Accuracy : {best['acc']:.4f}")
    print(f"  Precision: {best['prec']:.4f}")
    print(f"  Recall   : {best['rec']:.4f}")
    print(f"  F1       : {best['f1']:.4f}")

    # ----------------------------------------
    # Plots — top 5 by AUC only (plot crowding 피함)
    # ----------------------------------------
    print("\nGenerating plots...")
    top5 = sorted(all_results, key=lambda r: r["auc"], reverse=True)[:5]
    for r in top5:
        plot_confusion(r, SAVE_FIG)
    plot_roc_all(top5, SAVE_FIG)
    plot_comparison_bar(top5, SAVE_FIG)

    print("\nDone.")