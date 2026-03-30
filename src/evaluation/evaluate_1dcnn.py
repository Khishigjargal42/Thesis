import os
import torch
import torch.nn as nn
import torchaudio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, roc_curve
)

# =========================
# CONFIG  (must match train)
# =========================
BASE_PATH    = "/content/drive/MyDrive/Thesis/data/raw"
MODEL_PATH   = "best_model.pt"
FIGURES_PATH = "figures"

BATCH_SIZE   = 32
TARGET_SR    = 2000
SEGMENT_SEC  = 3
NUM_WORKERS  = 2
SEED         = 42

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
# Step 1: carve out 15% test (same as training script)
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

        # mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # resample (cached)
        if sr != TARGET_SR:
            if sr not in self._resamplers:
                self._resamplers[sr] = torchaudio.transforms.Resample(sr, TARGET_SR)
            waveform = self._resamplers[sr](waveform)

        # normalize
        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform / peak

        # fix length (center crop / pad — no randomness at eval)
        waveform = self._fix_length(waveform)

        return waveform, torch.tensor(self.labels[idx], dtype=torch.float32)

    def _fix_length(self, x):
        length = x.shape[1]
        if length > self.segment_samples:
            start = (length - self.segment_samples) // 2   # center crop
            x = x[:, start : start + self.segment_samples]
        elif length < self.segment_samples:
            x = torch.nn.functional.pad(x, (0, self.segment_samples - length))
        return x

# =========================
# DATALOADER
# =========================
test_ds     = PCGDataset(test_paths, test_labels)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=(str(DEVICE) == "cuda"))

# =========================
# MODEL  (must match train)
# =========================
class Model1D(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, 5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.fc(self.conv(x))

# =========================
# LOAD MODEL
# =========================
model = Model1D().to(DEVICE)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

model.eval()
print(f"\nLoaded weights from: {MODEL_PATH}")

# =========================
# PREDICTION
# =========================
y_true  = []
y_pred  = []
y_probs = []

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(DEVICE)

        logits = model(x_batch)
        probs  = torch.sigmoid(logits).cpu().squeeze().numpy()
        preds  = (probs > 0.5).astype(int)

        y_probs.extend(np.atleast_1d(probs).tolist())
        y_pred.extend(np.atleast_1d(preds).tolist())
        y_true.extend(y_batch.numpy().astype(int).tolist())

# =========================
# METRICS
# =========================
accuracy    = accuracy_score(y_true, y_pred)
precision   = precision_score(y_true, y_pred, zero_division=0)
recall      = recall_score(y_true, y_pred, zero_division=0)      # = Sensitivity
f1          = f1_score(y_true, y_pred, zero_division=0)
auc         = roc_auc_score(y_true, y_probs)

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
sensitivity = tp / (tp + fn + 1e-8)
specificity = tn / (tn + fp + 1e-8)

print("\nEvaluation Results")
print("-" * 30)
print(f"Accuracy    : {accuracy:.4f}")
print(f"Precision   : {precision:.4f}")
print(f"Recall      : {recall:.4f}")
print(f"F1 Score    : {f1:.4f}")
print(f"AUC-ROC     : {auc:.4f}")
print(f"Sensitivity : {sensitivity:.4f}")
print(f"Specificity : {specificity:.4f}")

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(5, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Normal", "Abnormal"],
    yticklabels=["Normal", "Abnormal"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix — 1D CNN")
plt.tight_layout()

plt.savefig(os.path.join(FIGURES_PATH, "confusion_1dcnn.png"), dpi=300)
plt.show()
print("Saved: figures/confusion_1dcnn.png")

# =========================
# ROC CURVE
# =========================
fpr, tpr, _ = roc_curve(y_true, y_probs)

plt.figure(figsize=(5, 4))

plt.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {auc:.4f}")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — 1D CNN")
plt.legend(loc="lower right")
plt.tight_layout()

plt.savefig(os.path.join(FIGURES_PATH, "roc_1dcnn.png"), dpi=300)
plt.show()
print("Saved: figures/roc_1dcnn.png")