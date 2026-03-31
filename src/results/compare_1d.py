import os
import torch
import torch.nn as nn
import torchaudio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, roc_curve
)

# =========================
# CONFIG
# =========================
BASE_PATH    = "/content/drive/MyDrive/Thesis/data/raw"
FIGURES_PATH = "/content/drive/MyDrive/Thesis/figures"

MODEL_PATHS = {
    "1D CNN"    : "/content/Thesis/src/training/best_model.pt",
    "1D ResNet" : "/content/Thesis/src/training/best_model_resnet.pt",
    "CNN + LSTM": "/content/Thesis/src/training/best_model_cnnlstm.pt",
}

BATCH_SIZE  = 32
TARGET_SR   = 2000
SEGMENT_SEC = 3
NUM_WORKERS = 2
SEED        = 42

os.makedirs(FIGURES_PATH, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# =========================
# LOAD CSV
# =========================
csv_path = os.path.join(BASE_PATH, "REFERENCES.csv")
df = pd.read_csv(csv_path)

file_paths, labels = [], []
for _, row in df.iterrows():
    path = os.path.join(BASE_PATH, row["folder"], row["record_id"] + ".wav")
    if os.path.exists(path):
        file_paths.append(path)
        labels.append(int(row["label"]))

# =========================
# REPRODUCE TEST SPLIT
# =========================
_, test_paths, _, test_labels = train_test_split(
    file_paths, labels,
    test_size=0.15,
    stratify=labels,
    random_state=SEED
)

print(f"Test samples : {len(test_paths)}")
print(f"  Positive   : {sum(test_labels)}")
print(f"  Negative   : {len(test_labels) - sum(test_labels)}")

# =========================
# DATASET
# =========================
class PCGDataset(Dataset):
    def __init__(self, paths, labels):
        self.paths   = paths
        self.labels  = labels
        self.segment_samples = TARGET_SR * SEGMENT_SEC
        self._resamplers = {}

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        waveform, sr = torchaudio.load(self.paths[idx])
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != TARGET_SR:
            if sr not in self._resamplers:
                self._resamplers[sr] = torchaudio.transforms.Resample(sr, TARGET_SR)
            waveform = self._resamplers[sr](waveform)
        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform / peak
        waveform = self._fix_length(waveform)
        return waveform, torch.tensor(self.labels[idx], dtype=torch.float32)

    def _fix_length(self, x):
        length = x.shape[1]
        if length > self.segment_samples:
            start = (length - self.segment_samples) // 2
            x = x[:, start : start + self.segment_samples]
        elif length < self.segment_samples:
            x = torch.nn.functional.pad(x, (0, self.segment_samples - length))
        return x

test_ds     = PCGDataset(test_paths, test_labels)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=(str(DEVICE) == "cuda"))

# =========================
# MODEL DEFINITIONS
# =========================
class Model1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.4), nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.fc(self.conv(x))


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 7, stride=stride, padding=3, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 7, padding=3, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch)
            )
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)

class ResNet1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1)
        )
        self.stage1 = ResBlock(32,  64,  stride=2)
        self.stage2 = ResBlock(64,  128, stride=2)
        self.stage3 = ResBlock(128, 256, stride=2)
        self.stage4 = ResBlock(256, 256, stride=2)
        self.pool   = nn.AdaptiveAvgPool1d(1)
        self.fc     = nn.Sequential(
            nn.Flatten(), nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.4), nn.Linear(128, 1)
        )
    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x); x = self.stage2(x)
        x = self.stage3(x); x = self.stage4(x)
        return self.fc(self.pool(x))


class CNNLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(inplace=True), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(128, 128, num_layers=2, batch_first=True,
                            bidirectional=True, dropout=0.3)
        self.fc = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(inplace=True), nn.Dropout(0.4), nn.Linear(64, 1)
        )
    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        return self.fc(x[:, -1, :])


MODEL_CLASSES = {
    "1D CNN"    : Model1D,
    "1D ResNet" : ResNet1D,
    "CNN + LSTM": CNNLSTM,
}

# =========================
# EVALUATE ALL MODELS
# =========================
def evaluate(model):
    model.eval()
    y_true, y_pred, y_probs = [], [], []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(DEVICE)
            logits  = model(x_batch)
            probs   = torch.sigmoid(logits).cpu().squeeze().numpy()
            preds   = (probs > 0.5).astype(int)
            y_probs.extend(np.atleast_1d(probs).tolist())
            y_pred.extend(np.atleast_1d(preds).tolist())
            y_true.extend(y_batch.numpy().astype(int).tolist())

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "y_true"     : y_true,
        "y_pred"     : y_pred,
        "y_probs"    : y_probs,
        "Accuracy"   : accuracy_score(y_true, y_pred),
        "Precision"  : precision_score(y_true, y_pred, zero_division=0),
        "Recall"     : recall_score(y_true, y_pred, zero_division=0),
        "F1 Score"   : f1_score(y_true, y_pred, zero_division=0),
        "AUC-ROC"    : roc_auc_score(y_true, y_probs),
        "Sensitivity": tp / (tp + fn + 1e-8),
        "Specificity": tn / (tn + fp + 1e-8),
    }


results = {}

for name, path in MODEL_PATHS.items():
    print(f"\nEvaluating {name} ...")
    model = MODEL_CLASSES[name]().to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    results[name] = evaluate(model)
    print(f"  AUC: {results[name]['AUC-ROC']:.4f}  Acc: {results[name]['Accuracy']:.4f}")

# =========================
# COMPARISON TABLE
# =========================
metrics = ["Accuracy", "Precision", "Recall", "F1 Score",
           "AUC-ROC", "Sensitivity", "Specificity"]

table = pd.DataFrame(
    {name: {m: round(results[name][m], 4) for m in metrics}
     for name in MODEL_PATHS}
).T

print("\n" + "="*60)
print("Model Comparison")
print("="*60)
print(table.to_string())

# save as CSV
csv_out = os.path.join(FIGURES_PATH, "comparison_table.csv")
table.to_csv(csv_out)
print(f"\nSaved: {csv_out}")

# =========================
# COMPARISON BAR CHART
# =========================
fig, ax = plt.subplots(figsize=(10, 5))

x      = np.arange(len(metrics))
width  = 0.25
colors = ["#4C72B0", "#DD8452", "#55A868"]

for i, (name, color) in enumerate(zip(MODEL_PATHS.keys(), colors)):
    vals = [results[name][m] for m in metrics]
    bars = ax.bar(x + i * width, vals, width, label=name, color=color)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=7)

ax.set_xticks(x + width)
ax.set_xticklabels(metrics, rotation=15, ha="right")
ax.set_ylim(0, 1.12)
ax.set_ylabel("Score")
ax.set_title("Model Comparison — All Metrics")
ax.legend(loc="lower right")
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()

bar_out = os.path.join(FIGURES_PATH, "comparison_bar.png")
plt.savefig(bar_out, dpi=300)
plt.show()
print(f"Saved: {bar_out}")

# =========================
# COMBINED ROC CURVE
# =========================
fig, ax = plt.subplots(figsize=(6, 5))

colors     = ["#4C72B0", "#DD8452", "#55A868"]
linestyles = ["-", "--", "-."]

for (name, color, ls) in zip(MODEL_PATHS.keys(), colors, linestyles):
    fpr, tpr, _ = roc_curve(results[name]["y_true"], results[name]["y_probs"])
    auc_val     = results[name]["AUC-ROC"]
    ax.plot(fpr, tpr, color=color, lw=2, linestyle=ls,
            label=f"{name}  (AUC = {auc_val:.4f})")

ax.plot([0, 1], [0, 1], color="gray", linestyle=":", lw=1)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve Comparison")
ax.legend(loc="lower right")
ax.grid(alpha=0.3)
plt.tight_layout()

roc_out = os.path.join(FIGURES_PATH, "comparison_roc.png")
plt.savefig(roc_out, dpi=300)
plt.show()
print(f"Saved: {roc_out}")

# =========================
# DONE
# =========================
print("\n" + "="*60)
print("All outputs saved to:", FIGURES_PATH)