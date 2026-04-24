"""
eval_seta.py
============
Kaggle Heartbeat Sounds - Set A дээр attention загваруудыг туршина.
Нэмэлт сургалт хийхгүй. Зөвхөн inference + recording-level majority vote.

Загварууд:
  1. ResNet2D + SE Attention  (Mel-spectrogram)  -> resnet_mel_attention.pth
  2. ParallelCNN2D v2         (MFCC)              -> parallelcnn_mfcc_v2.pth
"""

import os
import numpy as np
import torch
import torch.nn as nn
import librosa
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
# prepare_data.py / Table 2.3-тай яг адил
# ==========================================
SR          = 2000
N_FFT       = 512
HOP_LENGTH  = 47
N_MELS      = 128
N_MFCC      = 40
SEGMENT_SEC = 2

# ==========================================
# NORMALIZATION
# ESTIMATE_STATS=True  -> Set A-аас тооцно  (approximation)
# ESTIMATE_STATS=False -> PhysioNet train stats ашиглана (ideal)
# ==========================================
ESTIMATE_STATS = True
MEL_MEAN   = 0.0
MEL_STD    = 1.0
MFCC_MEAN  = 0.0
MFCC_STD   = 1.0

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
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, k, padding=p, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
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
            nn.BatchNorm2d(2*base),
            nn.ReLU(inplace=True),
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

def segment_audio(y_audio):
    seg_len  = SR * SEGMENT_SEC
    step     = seg_len // 2
    segments = []
    for start in range(0, len(y_audio) - seg_len + 1, step):
        segments.append(y_audio[start : start + seg_len])
    if len(segments) == 0:
        pad = np.zeros(seg_len, dtype=np.float32)
        pad[:len(y_audio)] = y_audio
        segments.append(pad)
    return segments


# ==========================================
# NORMALIZATION STATS
# ==========================================
def compute_stats_from_files(file_list, feature="mel"):
    print(f"  Computing normalization stats from {len(file_list)} files...")
    feats = []
    for fp in file_list[:200]:
        try:
            y_a, _ = librosa.load(fp, sr=SR, mono=True)
            for s in segment_audio(y_a):
                feats.append(extract_mel(s) if feature == "mel" else extract_mfcc(s))
        except Exception:
            continue

    if len(feats) == 0:
        print("  [WARN] No segments extracted for normalization.")
        return 0.0, 1.0

    arr = np.array(feats)
    return float(arr.mean()), float(arr.std() + 1e-8)


def make_filepath(wav_dir, fname):
    fname = fname.replace("\\", "/")
    if fname.startswith("set_a/"):
        fname = fname[len("set_a/"):]
    if not fname.endswith(".wav"):
        fname = fname + ".wav"
    return os.path.join(wav_dir, *fname.split("/"))


# ==========================================
# DATASET LOADING
# ==========================================
def load_set_a(csv_path, wav_dir):
    df = pd.read_csv(csv_path)
    print(f"\nSet A: {len(df)} recordings")
    print(df["label"].value_counts().to_string())

    df = df[~df["label"].isin(["artifact", "unlabelled"])].copy()
    df["binary_label"] = df["label"].apply(lambda l: 0 if l == "normal" else 1)
    df["filepath"]     = df["fname"].apply(lambda f: make_filepath(wav_dir, f))

    missing = df[~df["filepath"].apply(os.path.exists)]["filepath"].tolist()
    if missing:
        sample = missing[:5]
        print(f"  [WARN] Missing {len(missing)} files. Example: {sample}")

    df = df[df["filepath"].apply(os.path.exists)].reset_index(drop=True)

    print(f"  Valid files : {len(df)}")
    print(f"  Normal      : {(df['binary_label']==0).sum()}")
    print(f"  Abnormal    : {(df['binary_label']==1).sum()}")
    if len(df) == 0:
        raise RuntimeError(
            "No valid Set A wave files found. Verify set_a.csv file paths and the set_a directory contents."
        )
    return df


# ==========================================
# INFERENCE
# ==========================================
def predict_file(filepath, model, feature_fn, mean, std, invert=False):
    try:
        y_audio, _ = librosa.load(filepath, sr=SR, mono=True)
    except Exception as e:
        print(f"  [WARN] Cannot load {os.path.basename(filepath)}: {e}")
        return None, None

    feats = []
    for s in segment_audio(y_audio):
        f      = feature_fn(s)
        f_norm = (f - mean) / std
        feats.append(f_norm)

    arr    = np.array(feats)[:, np.newaxis, :, :]
    x      = torch.FloatTensor(arr).to(device)

    with torch.no_grad():
        logits = model(x).squeeze(-1)
        if logits.dim() == 0:
            logits = logits.unsqueeze(0)
        probs = torch.sigmoid(logits).cpu().numpy()

    if invert:
        probs = 1.0 - probs

    return float(probs.mean()), probs


# ==========================================
# EVALUATE
# ==========================================
def find_optimal_threshold(y_true, y_prob):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1s     = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_i  = np.argmax(f1s)
    return float(thresholds[best_i]), float(f1s[best_i])


def evaluate_model(df, model, feature_fn, mean, std, model_name, invert=False):
    print(f"\n{'='*52}")
    print(f"  {model_name}")
    print(f"{'='*52}")
    if invert:
        print("  [INFO] Probability inverted (domain shift correction)")

    y_true, y_prob = [], []
    failed = 0

    for _, row in df.iterrows():
        prob, _ = predict_file(
            row["filepath"], model, feature_fn, mean, std, invert=invert
        )
        if prob is None:
            failed += 1
            continue
        y_true.append(int(row["binary_label"]))
        y_prob.append(prob)

    if failed:
        print(f"  [WARN] {failed} files skipped")

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    # Default threshold = 0.5
    y_pred_05 = (y_prob >= 0.5).astype(int)

    # Optimal threshold (F1 maximize)
    opt_thr, _ = find_optimal_threshold(y_true, y_prob)
    y_pred_opt = (y_prob >= opt_thr).astype(int)

    def metrics(y_p):
        return dict(
            acc  = accuracy_score(y_true, y_p),
            prec = precision_score(y_true, y_p, zero_division=0),
            rec  = recall_score(y_true, y_p, zero_division=0),
            f1   = f1_score(y_true, y_p, zero_division=0),
        )

    m05  = metrics(y_pred_05)
    mopt = metrics(y_pred_opt)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")

    print(f"\n  AUC                    : {auc:.4f}")
    print(f"\n  Threshold = 0.50")
    print(f"    Accuracy   : {m05['acc']:.4f}")
    print(f"    Precision  : {m05['prec']:.4f}")
    print(f"    Recall     : {m05['rec']:.4f}")
    print(f"    F1         : {m05['f1']:.4f}")
    print(f"\n  Optimal threshold = {opt_thr:.3f}  (max F1)")
    print(f"    Accuracy   : {mopt['acc']:.4f}")
    print(f"    Precision  : {mopt['prec']:.4f}")
    print(f"    Recall     : {mopt['rec']:.4f}")
    print(f"    F1         : {mopt['f1']:.4f}")
    print(f"\n  Samples   : {len(y_true)}")

    return {
        "model"     : model_name,
        "auc"       : auc,
        "thr_05"    : m05,
        "thr_opt"   : mopt,
        "opt_thr"   : opt_thr,
        "y_true"    : y_true,
        "y_pred"    : y_pred_opt,   # plot-д optimal ашиглана
        "y_prob"    : y_prob,
    }


# ==========================================
# PLOTS
# ==========================================
def plot_confusion(result, save_dir):
    cm  = confusion_matrix(result["y_true"], result["y_pred"])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal","Abnormal"],
                yticklabels=["Normal","Abnormal"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    thr = result["opt_thr"]
    ax.set_title(f"{result['model']}\nKaggle Set A  (thr={thr:.2f})")
    plt.tight_layout()
    safe = result["model"].replace(" ","_").replace("+","").replace("/","")
    path = os.path.join(save_dir, f"kaggle_cm_{safe}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_roc(results, save_dir):
    fig, ax = plt.subplots(figsize=(6, 5))
    for r in results:
        if np.isnan(r["auc"]):
            continue
        fpr, tpr, _ = roc_curve(r["y_true"], r["y_prob"])
        ax.plot(fpr, tpr, lw=2, label=f"{r['model']}  (AUC={r['auc']:.3f})")
    ax.plot([0,1],[0,1], "k--", alpha=0.4, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Kaggle Set A Cross-Dataset Test")
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(save_dir, "kaggle_roc_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_bar_comparison(results, save_dir):
    metrics = ["acc","prec","rec","f1"]
    labels  = ["Accuracy","Precision","Recall","F1"]
    x       = np.arange(len(metrics))
    width   = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, r in enumerate(results):
        vals   = [r["thr_opt"][m] for m in metrics]
        offset = (i - len(results)/2 + 0.5) * width
        bars   = ax.bar(x + offset, vals, width, label=r["model"])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Attention Models — Kaggle Set A (Optimal Threshold)")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(save_dir, "kaggle_bar_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def print_summary(results):
    print("\n" + "=" * 68)
    print("KAGGLE SET A — CROSS-DATASET EVALUATION SUMMARY")
    print("=" * 68)
    header = f"{'Model':<36} {'AUC':>5}  {'Thr':>5}  {'Acc':>5}  {'Prec':>5}  {'Rec':>5}  {'F1':>5}"
    print(header)
    print("-" * 68)
    for r in results:
        m = r["thr_opt"]
        print(f"{r['model']:<36} "
              f"{r['auc']:>5.3f}  "
              f"{r['opt_thr']:>5.3f}  "
              f"{m['acc']:>5.3f}  "
              f"{m['prec']:>5.3f}  "
              f"{m['rec']:>5.3f}  "
              f"{m['f1']:>5.3f}")
    print("=" * 68)


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":

    df = load_set_a(SET_A_CSV, SET_A_DIR)

    all_files = df["filepath"].tolist()

    if ESTIMATE_STATS:
        print("\nEstimating normalization stats from Set A...")
        mel_mean,  mel_std  = compute_stats_from_files(all_files, "mel")
        mfcc_mean, mfcc_std = compute_stats_from_files(all_files, "mfcc")
        print(f"  Mel  — mean: {mel_mean:.4f}  std: {mel_std:.4f}")
        print(f"  MFCC — mean: {mfcc_mean:.4f}  std: {mfcc_std:.4f}")
        print("  [NOTE] Ideal: PhysioNet train set statistics.")
    else:
        mel_mean,  mel_std  = MEL_MEAN,  MEL_STD
        mfcc_mean, mfcc_std = MFCC_MEAN, MFCC_STD

    # ResNet: invert=True (domain shift урвуу таамаглал засна)
    r1 = evaluate_model(
        df, resnet_model, extract_mel,
        mel_mean, mel_std,
        "ResNet2D + SE Attention (Mel)",
        invert=True
    )

    # ParallelCNN: invert=False
    r2 = evaluate_model(
        df, pcnn_model, extract_mfcc,
        mfcc_mean, mfcc_std,
        "ParallelCNN2D v2 (MFCC)",
        invert=True
    )

    results = [r1, r2]

    print("\nGenerating plots...")
    for r in results:
        plot_confusion(r, SAVE_FIG)
    plot_roc(results, SAVE_FIG)
    plot_bar_comparison(results, SAVE_FIG)

    print_summary(results)
    print("\nDone.")