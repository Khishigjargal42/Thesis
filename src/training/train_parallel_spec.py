import sys
sys.path.append("/content/Thesis/src")

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# =========================
# MODEL
# =========================
class ParallelCNN2D(nn.Module):
    def __init__(self):
        super().__init__()

        self.b1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.b2 = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.b3 = nn.Sequential(
            nn.Conv2d(1, 16, 7, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv = nn.Conv2d(48, 64, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        b1 = self.b1(x)
        b2 = self.b2(x)
        b3 = self.b3(x)

        x = torch.cat([b1, b2, b3], dim=1)

        x = F.relu(self.conv(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        return self.fc(x)


# =========================
# MAIN
# =========================
def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = "/content/drive/MyDrive/Thesis/data/features"
    save_dir = "/content/drive/MyDrive/Thesis/models"
    os.makedirs(save_dir, exist_ok=True)

    print("\n=== Parallel CNN + Spectrogram ===")

    # LOAD
    X = np.load(os.path.join(data_dir, "spec.npy"))
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
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # TRAIN
    best_val = float("inf")

    for epoch in range(20):

        # TRAIN
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

        print(f"Epoch {epoch+1} | train {train_loss:.4f} | val {val_loss:.4f}")

        # SAVE BEST
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(),
                       os.path.join(save_dir, "parallelcnn_spec.pth"))

    print("\nSaved model")

    # =========================
    # TEST
    # =========================
    loader = DataLoader(TensorDataset(X_test, torch.FloatTensor(y_test)),
                        batch_size=256)

    model.load_state_dict(torch.load(os.path.join(save_dir, "parallelcnn_spec.pth")))
    model.eval()

    probs_all = []

    with torch.no_grad():
        for xb, _ in loader:
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