import sys
sys.path.append("/content/Thesis/src")

import os, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# =========================
# MODEL (same)
# =========================
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k, p):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, k, padding=p, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, k, padding=p, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
    def forward(self, x): return self.net(x)

class ParallelCNN2D(nn.Module):
    def __init__(self, base=24, dropout=0.4):  # 🔥 бага болгосон
        super().__init__()
        self.b1 = ConvBlock(1, base, 3, 1)
        self.b2 = ConvBlock(1, base, 5, 2)
        self.b3 = ConvBlock(1, base, 7, 3)

        self.post = nn.Sequential(
            nn.Conv2d(3*base, 2*base, 3, padding=1, bias=False),
            nn.BatchNorm2d(2*base),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2*base, 1)

    def forward(self, x):
        x = torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)
        x = self.post(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)

# =========================
# MAIN
# =========================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = "/content/drive/MyDrive/Thesis/data/features"
    save_dir = "/content/drive/MyDrive/Thesis/models"
    os.makedirs(save_dir, exist_ok=True)

    print("\n=== Parallel CNN v2 + MFCC ===")

    # 🔥 MFCC load
    X = np.load(os.path.join(data_dir, "mfcc.npy"))
    y = np.load(os.path.join(data_dir, "labels.npy"))

    print("Loaded:", X.shape)

    # SPLIT
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42)

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

    # NORMALIZE
    mean, std = X_train.mean(), X_train.std()
    X_train = (X_train - mean) / (std + 1e-8)
    X_val   = (X_val - mean) / (std + 1e-8)
    X_test  = (X_test - mean) / (std + 1e-8)

    # TENSOR
    X_train = torch.FloatTensor(X_train).unsqueeze(1)
    X_val   = torch.FloatTensor(X_val).unsqueeze(1)
    X_test  = torch.FloatTensor(X_test).unsqueeze(1)

    y_train = torch.FloatTensor(y_train)
    y_val   = torch.FloatTensor(y_val)

    # DATALOADER
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size=64)

    # MODEL
    model = ParallelCNN2D().to(device)

    # imbalance
    n_neg = (y_train == 0).sum().item()
    n_pos = (y_train == 1).sum().item()
    pos_weight = torch.tensor([n_neg / n_pos]).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

    best_val = float("inf")
    patience = 10
    early = 0

    save_path = os.path.join(save_dir, "parallelcnn_mfcc_v2.pth")

    for epoch in range(25):

        model.train()
        train_loss = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            out = model(xb).squeeze()
            loss = criterion(out, yb)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # VALID
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb).squeeze()
                val_loss += criterion(out, yb).item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            early = 0
            torch.save(model.state_dict(), save_path)
            print(f"Epoch {epoch+1} | train {train_loss:.4f} | val {val_loss:.4f} ✓")
        else:
            early += 1
            print(f"Epoch {epoch+1} | train {train_loss:.4f} | val {val_loss:.4f} | no improve ({early}/10)")

        if early >= patience:
            print("⛔ Early stopping")
            break

    print("\nSaved best model")

    # TEST
    test_loader = DataLoader(TensorDataset(X_test, torch.FloatTensor(y_test)), batch_size=256)

    model.load_state_dict(torch.load(save_path))
    model.eval()

    probs_all = []

    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device)
            out = model(xb).squeeze()
            probs = torch.sigmoid(out)
            probs_all.extend(probs.cpu().numpy())

    probs = np.array(probs_all)
    preds = (probs >= 0.5).astype(int)

    print("\n=== TEST RESULT ===")
    print("Accuracy:", accuracy_score(y_test, preds))
    print("Precision:", precision_score(y_test, preds))
    print("Recall:", recall_score(y_test, preds))
    print("F1:", f1_score(y_test, preds))
    print("AUC:", roc_auc_score(y_test, probs))


if __name__ == "__main__":
    main()