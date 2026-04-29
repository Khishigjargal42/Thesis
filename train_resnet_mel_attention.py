import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score

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

# FIX 1: Stratified split — shuffle хийгдсэн, class balance хадгална
from sklearn.model_selection import train_test_split
idx = np.arange(len(y))
idx_tr, idx_val = train_test_split(idx, test_size=0.2, stratify=y, random_state=42)

# FIX 2: Train stats-аар normalize (data leak үгүй)
TR_MEAN = X[idx_tr].mean()
TR_STD  = X[idx_tr].std()
print(f"Norm — mean: {TR_MEAN:.4f}  std: {TR_STD:.4f}")

X_norm = (X - TR_MEAN) / (TR_STD + 1e-8)

X_train = torch.FloatTensor(X_norm[idx_tr][:, np.newaxis])
X_val   = torch.FloatTensor(X_norm[idx_val][:, np.newaxis])
y_train = torch.FloatTensor(y[idx_tr])
y_val   = torch.FloatTensor(y[idx_val])

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val,   y_val),   batch_size=64)

# =========================
# CLASS WEIGHT
# =========================
# FIX 3: Train set-ийн харьцаагаар pos_weight тооцно
pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()]).to(device)
print(f"pos_weight: {pos_weight.item():.4f}")

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
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()
        self.se    = SEBlock(out_ch)
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
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.fc     = nn.Linear(128, 1)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)

model = ResNet2D_SE().to(device)

# =========================
# TRAIN SETUP
# =========================
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# FIX 4: Scheduler mode="max" + F1 дээр хянана
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", patience=3, factor=0.5
)

# =========================
# TRAIN LOOP
# =========================
epochs   = 30
best_f1  = 0.0
patience = 8
counter  = 0

save_path = os.path.join(save_dir, "resnet_mel_attention_v3.pth")

for epoch in range(epochs):

    # Train
    model.train()
    train_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb).squeeze(), yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # Validation
    model.eval()
    val_loss   = 0
    all_probs  = []
    all_labels = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits   = model(xb).squeeze()
            val_loss += criterion(logits, yb).item()
            all_probs.extend(torch.sigmoid(logits).cpu().numpy())
            all_labels.extend(yb.cpu().numpy())
    val_loss /= len(val_loader)

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds      = (all_probs >= 0.5).astype(int)
    f1         = f1_score(all_labels, preds)
    auc        = roc_auc_score(all_labels, all_probs)

    print(f"Epoch {epoch+1:02d} | train {train_loss:.4f} | val {val_loss:.4f} | F1 {f1:.4f} | AUC {auc:.4f}")

    # FIX 5: F1 дээр early stopping + checkpoint
    scheduler.step(f1)

    if f1 > best_f1:
        best_f1 = f1
        counter = 0
        torch.save(model.state_dict(), save_path)
        print("  saved")
    else:
        counter += 1

    if counter >= patience:
        print("early stopping")
        break

print(f"\nBest F1: {best_f1:.4f}")
print(f"Saved  : {save_path}")
print(f"\napp.py-д хэрэглэх:")
print(f"  NORM_MEAN = {TR_MEAN:.4f}")
print(f"  NORM_STD  = {TR_STD:.4f}")