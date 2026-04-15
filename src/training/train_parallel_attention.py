import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# =========================
# CONFIG (🔥 UPDATED)
# =========================
FEATURE = "mfcc"

data_dir = "/content/drive/MyDrive/Thesis/data/features"
save_dir = "/content/drive/MyDrive/Thesis/models"

os.makedirs(save_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

# =========================
# LOAD DATA
# =========================
X = np.load(f"{data_dir}/{FEATURE}.npy")
y = np.load(f"{data_dir}/labels.npy")

print("Loaded:", X.shape, y.shape)

# add channel dim
X = X[:, np.newaxis, :, :]

# split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# to tensor
X_train = torch.FloatTensor(X_train)
X_val   = torch.FloatTensor(X_val)
y_train = torch.FloatTensor(y_train)
y_val   = torch.FloatTensor(y_val)

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=128,
    shuffle=True
)

val_loader = DataLoader(
    TensorDataset(X_val, y_val),
    batch_size=128
)

# =========================
# CLASS WEIGHT
# =========================
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
pos_weight = torch.tensor([pos_weight]).to(device)

print("pos_weight:", pos_weight.item())

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
            nn.Linear(32*10*4,128),  # MFCC shape
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

model = ParallelCNN_SE().to(device)

# =========================
# TRAIN SETUP
# =========================
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=3, factor=0.5
)

# =========================
# TRAIN LOOP
# =========================
epochs = 20
best_val = float("inf")
patience = 10
counter = 0

save_path = os.path.join(save_dir, "parallelcnn_mfcc_attention.pth")

for epoch in range(epochs):

    model.train()
    train_loss = 0

    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()

        outputs = model(xb).squeeze()
        loss = criterion(outputs, yb)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # VALIDATION
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            outputs = model(xb).squeeze()
            val_loss += criterion(outputs, yb).item()

    val_loss /= len(val_loader)

    scheduler.step(val_loss)

    print(f"Epoch {epoch+1} | train {train_loss:.4f} | val {val_loss:.4f}")

    if val_loss < best_val:
        best_val = val_loss
        counter = 0
        torch.save(model.state_dict(), save_path)
        print("✓ saved")

    else:
        counter += 1

    if counter >= patience:
        print("⛔ early stopping")
        break

print("\nSaved:", save_path)