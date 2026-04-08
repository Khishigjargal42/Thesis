import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from models.resnet2d import ResNet2D


def train_resnet2d(feature_name, data_dir, save_dir,
                   epochs=30, batch_size=64, lr=1e-3,
                   patience=10):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*50}")
    print(f"Feature: {feature_name} | Device: {device}")
    print(f"{'='*50}")

    # =========================
    # LOAD DATA
    # =========================
    norm_dir = os.path.join(data_dir, 'normalized')

    if not os.path.exists(os.path.join(norm_dir, 'X_train.npy')):
        raise FileNotFoundError(f"{norm_dir} олдсонгүй")

    X_train = np.load(os.path.join(norm_dir, 'X_train.npy'))
    X_val   = np.load(os.path.join(norm_dir, 'X_val.npy'))
    y_train = np.load(os.path.join(norm_dir, 'y_train.npy'))
    y_val   = np.load(os.path.join(norm_dir, 'y_val.npy'))

    print(f"Loaded normalized data: {X_train.shape}")

    # shape → (B, 1, 128, 16)
    X_train = torch.FloatTensor(X_train).unsqueeze(1)
    X_val   = torch.FloatTensor(X_val).unsqueeze(1)
    y_train = torch.FloatTensor(y_train)
    y_val   = torch.FloatTensor(y_val)

    train_loader = DataLoader(TensorDataset(X_train, y_train),
                             batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val, y_val),
                             batch_size=batch_size)

    # =========================
    # MODEL
    # =========================
    model = ResNet2D().to(device)

    # class imbalance
    n_neg = (y_train == 0).sum().item()
    n_pos = (y_train == 1).sum().item()
    pos_weight = torch.tensor([n_neg / n_pos]).to(device)

    print(f"Class weight (pos): {pos_weight.item():.2f}")

    # loss + optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=5e-4)  # ↑ regularization

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )

    # =========================
    # TRAIN LOOP
    # =========================
    best_val_loss = float('inf')
    early_counter = 0

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'resnet2d_{feature_name}.pth')

    for epoch in range(epochs):

        # ---- TRAIN ----
        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            outputs = model(xb).squeeze()
            loss = criterion(outputs, yb)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ---- VALIDATION ----
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb).squeeze()
                val_loss += criterion(outputs, yb).item()

        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        # =========================
        # EARLY STOPPING + SAVE
        # =========================
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_counter = 0
            torch.save(model.state_dict(), save_path)

            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"train: {train_loss:.4f} | val: {val_loss:.4f} ✓ (saved)")

        else:
            early_counter += 1
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"train: {train_loss:.4f} | val: {val_loss:.4f} "
                  f"| no improve ({early_counter}/{patience})")

        if early_counter >= patience:
            print(f"\n⛔ Early stopping triggered at epoch {epoch+1}")
            break

    print(f"\nBest model saved at: {save_path}")
    return save_path


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--feature', default='mel_spectrogram',
                        choices=['mel_spectrogram', 'mfcc', 'spec', 'logmel'])
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)

    args = parser.parse_args()

    train_resnet2d(args.feature,
                   args.data_dir,
                   args.save_dir,
                   args.epochs,
                   args.batch_size)