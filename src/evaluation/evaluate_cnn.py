import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_auc_score, roc_curve)
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import os

# =========================
# CONFIG
# =========================
FEATURE = "mel_spectrogram"

data_dir = "data/features"
model_dir = "models"
fig_dir = "figures"
os.makedirs(fig_dir, exist_ok=True)

# =========================
# LOAD DATA
# =========================
X = np.load(f"{data_dir}/{FEATURE}.npy")
y = np.load(f"{data_dir}/labels.npy")

print("Loaded:", X.shape, y.shape)

X = X[:, np.newaxis, :, :]

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

loader = DataLoader(TensorDataset(X, y), batch_size=128)

# =========================
# EXACT MODEL 🔥
# =========================
class HeartSoundCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Linear(64*16*2,128),   # 🔥 EXACT MATCH
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128,2)
        )

    def forward(self,x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# =========================
# LOAD MODEL
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = HeartSoundCNN().to(device)

model_path = f"{model_dir}/cnn_{FEATURE}.pth"
print("Loading:", model_path)

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# =========================
# PREDICT
# =========================
y_true, y_pred, y_probs = [], [], []

with torch.no_grad():
    for xb, yb in loader:
        xb = xb.to(device)

        out = model(xb)

        probs = torch.softmax(out, dim=1)[:,1]
        preds = torch.argmax(out, dim=1)

        y_probs.extend(probs.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        y_true.extend(yb.numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_probs = np.array(y_probs)

# =========================
# METRICS
# =========================
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
auc = roc_auc_score(y_true, y_probs)

print("\n=== BASELINE CNN RESULT ===")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1       : {f1:.4f}")
print(f"AUC      : {auc:.4f}")

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_true, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal","Abnormal"],
            yticklabels=["Normal","Abnormal"])

plt.title(f"Confusion Matrix ({FEATURE})")
plt.savefig(f"{fig_dir}/cm_{FEATURE}.png", dpi=300)
plt.show()

# =========================
# ROC CURVE
# =========================
fpr, tpr, _ = roc_curve(y_true, y_probs)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
plt.plot([0,1],[0,1],'--')
plt.legend()
plt.title(f"ROC ({FEATURE})")
plt.savefig(f"{fig_dir}/roc_{FEATURE}.png", dpi=300)
plt.show()