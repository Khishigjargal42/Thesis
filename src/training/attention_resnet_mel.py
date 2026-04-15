import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# =========================
# CONFIG
# =========================
FEATURE = "mel_spectrogram"

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

X = X[:, np.newaxis, :, :]

# split (simple)
split = int(0.8 * len(X))

X_train = torch.FloatTensor(X[:split])
X_val   = torch.FloatTensor(X[split:])

y_train = torch.FloatTensor(y[:split])
y_val   = torch.FloatTensor(y[split:])

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=64,
    shuffle=True
)

val_loader = DataLoader(
    TensorDataset(X_val, y_val),
    batch_size=64
)

# =========================
# CLASS WEIGHT
# =========================
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
pos_weight = torch.tensor([pos_weight]).to(device)

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
# RESNET BLOCK + ATTENTION
# =========================
class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_ch)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.relu = nn.ReLU()

        # 🔥 ATTENTION
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

        out = self.se(out)  # 🔥 ATTENTION

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

model = ResNet2D_SE().to(device)

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

save_path = os.path.join(save_dir, "resnet_mel_attention.pth")

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