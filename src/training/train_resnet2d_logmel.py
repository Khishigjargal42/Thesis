import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from models.resnet2d import ResNet2D


def train_resnet2d_logmel(data_dir, save_dir,
                          epochs=30, batch_size=64, lr=1e-3,
                          patience=10):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*50}")
    print(f"ResNet2D + Log-Mel | Device: {device}")
    print(f"{'='*50}")

    feat_dir = os.path.join(data_dir, 'features')

    X = np.load(os.path.join(feat_dir, 'logmel.npy'))
    y = np.load(os.path.join(feat_dir, 'labels.npy'))

    print(f"Loaded: {X.shape}")

    # SPLIT
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42)

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

    # NORMALIZE
    mean, std = X_train.mean(), X_train.std()
    X_train = (X_train - mean) / (std + 1e-8)
    X_val   = (X_val   - mean) / (std + 1e-8)

    # TENSOR
    X_train = torch.FloatTensor(X_train).unsqueeze(1)
    X_val   = torch.FloatTensor(X_val).unsqueeze(1)

    y_train = torch.FloatTensor(y_train)
    y_val   = torch.FloatTensor(y_val)

    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True, num_workers=0)

    val_loader = DataLoader(TensorDataset(X_val, y_val),
                            batch_size=batch_size, num_workers=0)

    # MODEL
    model = ResNet2D().to(device)

    n_neg = (y_train == 0).sum().item()
    n_pos = (y_train == 1).sum().item()
    pos_weight = torch.tensor([n_neg / n_pos]).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )

    best_val_loss = float('inf')
    counter = 0

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "resnet2d_logmel.pth")

    for epoch in range(epochs):

        model.train()
        train_loss = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            outputs = model(xb).view(-1)
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
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb).view(-1)
                val_loss += criterion(outputs, yb).item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), save_path)

            print(f"Epoch {epoch+1:2d} | train {train_loss:.4f} | val {val_loss:.4f} ✓")

        else:
            counter += 1
            print(f"Epoch {epoch+1:2d} | train {train_loss:.4f} | val {val_loss:.4f} "
                  f"| no improve ({counter}/{patience})")

        if counter >= patience:
            print(f"\n⛔ Early stopping at epoch {epoch+1}")
            break

    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    train_resnet2d_logmel(
        data_dir="/content/drive/MyDrive/Thesis/data",
        save_dir="/content/drive/MyDrive/Thesis/models",
        epochs=30,
        batch_size=64
    )