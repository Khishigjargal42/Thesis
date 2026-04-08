# src/training/train_resnet2d.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from models.resnet2d import ResNet2D

# Feature нэр → файлын нэрийн харгалзуулалт
FEATURE_MAP = {
    'mel_spectrogram': 'mel_spectrogram',
    'mfcc':            'mfcc',
    'spec':            'spec',
    'logmel':          'logmel',
}

def load_feature_data(feature_name, data_dir, split):
    """
    data/features/ дотроос тухайн feature-г ачаалаад
    data/splits/ дотроос split индексийг ашиглана.
    Эсвэл data/normalized/ байгаа бол тэрийг ашиглана.
    """
    feat_file  = os.path.join(data_dir, 'features', f'{FEATURE_MAP[feature_name]}.npy')
    label_file = os.path.join(data_dir, 'features', f'{FEATURE_MAP[feature_name]}_labels.npy')
    
    # Split индексүүд
    X_split = np.load(os.path.join(data_dir, 'splits', f'X_{split}.npy'))
    y_split = np.load(os.path.join(data_dir, 'splits', f'y_{split}.npy'))
    
    return X_split, y_split


def train_resnet2d(feature_name, data_dir, save_dir,
                   epochs=50, batch_size=64, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*50}")
    print(f"Feature: {feature_name} | Device: {device}")
    print(f"{'='*50}")

    # --- Өгөгдөл ачаалах ---
    # normalized/ дотор аль хэдийн хуваагдсан бол тэрийг ашигла
    norm_dir = os.path.join(data_dir, 'normalized')
    
    if os.path.exists(os.path.join(norm_dir, 'X_train.npy')):
        # Baseline CNN-тэй ижил normalized өгөгдөл ашиглана
        X_train = np.load(os.path.join(norm_dir, 'X_train.npy'))
        X_val   = np.load(os.path.join(norm_dir, 'X_val.npy'))
        y_train = np.load(os.path.join(norm_dir, 'y_train.npy'))
        y_val   = np.load(os.path.join(norm_dir, 'y_val.npy'))
        print(f"normalized/ дотроос ачааллаа: {X_train.shape}")
    else:
        raise FileNotFoundError("data/normalized/ олдсонгүй")

    # (B, 128, 16) → (B, 1, 128, 16)
    X_train = torch.FloatTensor(X_train).unsqueeze(1)
    X_val   = torch.FloatTensor(X_val).unsqueeze(1)
    y_train = torch.FloatTensor(y_train)
    y_val   = torch.FloatTensor(y_val)

    print(f"Train: {X_train.shape}, Val: {X_val.shape}")

    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(TensorDataset(X_val,   y_val),
                              batch_size=batch_size, num_workers=2)

    # --- Загвар ---
    model = ResNet2D().to(device)

    n_neg = (y_train == 0).sum().item()
    n_pos = (y_train == 1).sum().item()
    pos_weight = torch.tensor([n_neg / n_pos]).to(device)
    print(f"Class weight (pos): {pos_weight.item():.2f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, patience=5, factor=0.5)

    best_val_loss = float('inf')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'resnet2d_{feature_name}.pth')

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb).squeeze(), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb).squeeze(), yb).item()

        train_loss /= len(train_loader)
        val_loss   /= len(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"train: {train_loss:.4f} | val: {val_loss:.4f} ✓")
        elif (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"train: {train_loss:.4f} | val: {val_loss:.4f}")

    print(f"Saved: {save_path}\n")
    return save_path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature', default='mel_spectrogram',
                        choices=['mel_spectrogram', 'mfcc', 'spec', 'logmel'])
    parser.add_argument('--data_dir',  default='../data')
    parser.add_argument('--save_dir',  default='../models')
    parser.add_argument('--epochs',    type=int, default=50)
    parser.add_argument('--batch_size',type=int, default=64)
    args = parser.parse_args()

    train_resnet2d(args.feature, args.data_dir,
                   args.save_dir, args.epochs, args.batch_size)