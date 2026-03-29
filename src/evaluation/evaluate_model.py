import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
DATA_DIR = "/content/drive/MyDrive/Thesis/data/normalized"
MODEL_PATH = "/content/Thesis/models/cnn_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# ------------------------
# Load test dataset
# ------------------------

X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

# Add channel dimension
X_test = X_test[:, np.newaxis, :, :]

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

test_loader = DataLoader(
    TensorDataset(X_test, y_test),
    batch_size=32
)

# ------------------------
# CNN Architecture
# ------------------------

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

            nn.Linear(64*16*2,128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128,2)
        )

    def forward(self,x):

        x = self.conv(x)

        x = x.view(x.size(0),-1)

        x = self.fc(x)

        return x


# ------------------------
# Load trained model
# ------------------------

model = HeartSoundCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# ------------------------
# Run inference
# ------------------------

y_true = []
y_pred = []

with torch.no_grad():

    for X_batch, y_batch in test_loader:

        X_batch = X_batch.to(device)

        outputs = model(X_batch)

        _, predicted = torch.max(outputs,1)

        y_true.extend(y_batch.numpy())
        y_pred.extend(predicted.cpu().numpy())

# ------------------------
# Metrics
# ------------------------

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("\nTest Results")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

# ------------------------
# Confusion Matrix
# ------------------------

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(5,4))

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal","Abnormal"],
            yticklabels=["Normal","Abnormal"])

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.savefig("confusion_matrix.png", dpi=300)
plt.show()