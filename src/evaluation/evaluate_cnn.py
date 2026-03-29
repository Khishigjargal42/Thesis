import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# SELECT FEATURE
# =========================

FEATURE = "logmel"     #  mfcc / spec / logmel

X = np.load(f"data/features/{FEATURE}.npy")
y = np.load(f"data/features/{FEATURE}_labels.npy")

X = X[:, np.newaxis, :, :]

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

dataset = TensorDataset(X, y)

loader = DataLoader(dataset, batch_size=32)

# =========================
# MODEL
# =========================

class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((4,4))
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*4*4,128),
            nn.ReLU(),
            nn.Linear(128,2)
        )

    def forward(self,x):

        x = self.conv(x)
        x = self.fc(x)

        return x


# =========================
# LOAD MODEL
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN().to(device)

model.load_state_dict(
    torch.load(f"models/cnn_{FEATURE}.pth", map_location=device)
)

model.eval()

# =========================
# PREDICTION
# =========================

y_true = []
y_pred = []

with torch.no_grad():

    for X_batch, y_batch in loader:

        X_batch = X_batch.to(device)

        outputs = model(X_batch)

        preds = torch.argmax(outputs, dim=1).cpu().numpy()

        y_pred.extend(preds)
        y_true.extend(y_batch.numpy())

# =========================
# METRICS
# =========================

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("\nEvaluation Results")
print("-----------------------")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

# =========================
# CONFUSION MATRIX
# =========================

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(5,4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Normal","Abnormal"],
    yticklabels=["Normal","Abnormal"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.title("Confusion Matrix")

plt.savefig(f"figures/confusion_{FEATURE}.png", dpi=300)

plt.show()