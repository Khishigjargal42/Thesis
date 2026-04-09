# quick_train_local.py
from xml.parsers.expat import model

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from src.models.resnet2d import ResNet2D


def quick_train(feature_file, label_file,
                epochs=12, batch_size=32, lr=1e-3,
                max_samples=10000):   # ← хурдан болгох

    device = torch.device('cpu')
    print(f"Device: {device}")

    # =========================
    # LOAD
    # =========================
    X = np.load(feature_file)
    y = np.load(label_file)

    print("Original:", X.shape)

    # 🔥 DEBUG MODE (subset)
    if max_samples:
        X = X[:max_samples]
        y = y[:max_samples]

    print("Used:", X.shape)

    # =========================
    # SPLIT
    # =========================
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # =========================
    # NORMALIZE
    # =========================
    mean, std = X_train.mean(), X_train.std()

    X_train = (X_train - mean) / (std + 1e-8)
    X_val   = (X_val   - mean) / (std + 1e-8)

    # =========================
    # TENSOR
    # =========================
    X_train = torch.FloatTensor(X_train).unsqueeze(1)
    X_val   = torch.FloatTensor(X_val).unsqueeze(1)

    y_train = torch.FloatTensor(y_train)
    y_val   = torch.FloatTensor(y_val)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=batch_size,
        num_workers=2
    )

    # =========================
    # MODEL
    # =========================
    model = ResNet2D().to(device)

    n_neg = (y_train == 0).sum().item()
    n_pos = (y_train == 1).sum().item()
    pos_weight = torch.tensor([n_neg / n_pos])

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # =========================
    # TRAIN
    # =========================
    for epoch in range(epochs):

        model.train()
        train_loss = 0

        for xb, yb in train_loader:
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
                outputs = model(xb).squeeze()
                val_loss += criterion(outputs, yb).item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1:2d} | train {train_loss:.4f} | val {val_loss:.4f}")

    print("\n✅ DONE")


# =========================
# RUN
# =========================
if __name__ == "__main__":

    quick_train(
        feature_file="data/features/spec.npy",
        label_file="data/features/spec_labels.npy",
        epochs=12,
        batch_size=32,
        max_samples=10000  # ← хурдан туршилт
    )
torch.save(model.state_dict(), "models/resnet2d_spec.pth")