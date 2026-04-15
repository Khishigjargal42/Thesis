import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
import os

# =========================
# CONFIG
# =========================
FEATURE = "mel_spectrogram"

data_dir = "/content/drive/MyDrive/Thesis/data/features"
model_path = "/content/drive/MyDrive/Thesis/models/resnet_mel_attention.pth"

fig_dir = "/content/drive/MyDrive/Thesis/figures"
os.makedirs(fig_dir, exist_ok=True)

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
# RESNET BLOCK
# =========================
class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_ch)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.relu = nn.ReLU()

        self.se = SEBlock(out_ch)

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
        out = self.relu(out)

        return out

# =========================
# RESNET MODEL
# =========================
class ResNet2D_SE(nn.Module):
    def __init__(self):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.layer1 = BasicBlock(32, 32)
        self.layer2 = BasicBlock(32, 64, stride=2)
        self.layer3 = BasicBlock(64, 128, stride=2)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.pool(x).view(x.size(0), -1)

        x = self.fc(x)

        return x

# =========================
# LOAD MODEL
# =========================
model = ResNet2D_SE().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# =========================
# EVALUATION
# =========================
y_true, y_pred, y_probs = [], [], []

with torch.no_grad():
    for xb, yb in loader:
        xb = xb.to(device)

        logits = model(xb).view(-1)
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

print("\n=== RESNET + ATTENTION RESULT ===")
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
plt.title("Confusion Matrix (ResNet + Attention)")

plt.savefig(f"{fig_dir}/cm_resnet_attention.png", dpi=300)
plt.show()

# =========================
# ROC CURVE
# =========================
fpr, tpr, _ = roc_curve(y_true, y_probs)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
plt.plot([0,1],[0,1],'--')

plt.legend()
plt.title("ROC Curve (ResNet + Attention)")

plt.savefig(f"{fig_dir}/roc_resnet_attention.png", dpi=300)
plt.show()