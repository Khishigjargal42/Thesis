import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
import os
fig_dir = "/content/drive/MyDrive/Thesis/figures"
os.makedirs(fig_dir, exist_ok=True)

# =========================
# CONFIG
# =========================
FEATURE = "mfcc"

data_dir = "/content/drive/MyDrive/Thesis/data/features"
model_path = "/content/drive/MyDrive/Thesis/models/parallelcnn_mfcc_attention.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LOAD DATA
# =========================
X = np.load(f"{data_dir}/{FEATURE}.npy")
y = np.load(f"{data_dir}/labels.npy")

X = X[:, np.newaxis, :, :]

X = torch.FloatTensor(X)
y = torch.FloatTensor(y)

loader = DataLoader(
    TensorDataset(X, y),
    batch_size=256
)

print("Loaded:", X.shape)

# =========================
# SE BLOCK
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

# =========================
# MODEL
# =========================
class ParallelCNN_SE(nn.Module):
    def __init__(self):
        super().__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1),
            nn.ReLU(),
            SEBlock(16),
            nn.MaxPool2d(2)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(1,16,5,padding=2),
            nn.ReLU(),
            SEBlock(16),
            nn.MaxPool2d(2)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(1,16,7,padding=3),
            nn.ReLU(),
            SEBlock(16),
            nn.MaxPool2d(2)
        )

        self.conv = nn.Sequential(
            nn.Conv2d(48,32,3,padding=1),
            nn.ReLU(),
            SEBlock(32),
            nn.MaxPool2d(2)
        )

        self.flatten = nn.Flatten()

        self.fc = nn.Sequential(
            nn.Linear(32*10*4,128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128,1)
        )

    def forward(self,x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)

        x = torch.cat([b1,b2,b3], dim=1)

        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x

# =========================
# LOAD MODEL
# =========================
model = ParallelCNN_SE().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# =========================
# EVALUATION
# =========================
y_true, y_pred, y_probs = [], [], []

with torch.no_grad():
    for xb, yb in loader:
        xb = xb.to(device)

        logits = model(xb).view(-1)   # 🔥 FIX
        probs = torch.sigmoid(logits)

        preds = (probs >= 0.5).int()

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
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_probs)

print("\n=== ATTENTION MODEL RESULT ===")
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
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.title("Confusion Matrix (Attention)")

plt.savefig(f"{fig_dir}/cm_attention.png", dpi=300)
plt.show()

# =========================
# ROC CURVE
# =========================
fpr, tpr, _ = roc_curve(y_true, y_probs)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
plt.plot([0,1],[0,1],'--')

plt.legend()
plt.title("ROC Curve (Attention)")

plt.savefig(f"{fig_dir}/roc_attention.png", dpi=300)
plt.show()