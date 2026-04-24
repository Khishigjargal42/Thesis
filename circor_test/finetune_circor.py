# ============================================================
# FINAL STABLE FINE-TUNE PIPELINE
# (Architecture MATCHED with pretrained model)
# ============================================================

import os
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score

# ============================================================
# CONFIG
# ============================================================

class CFG:
    DATA_DIR = "circor/training_data"
    CSV_PATH = "circor/training_data.csv"
    MODEL_PATH = "parallelcnn_mfcc_attention.pth"

    SR = 2000
    N_MFCC = 40
    MAX_LEN = 16

    BATCH_SIZE = 16
    EPOCHS = 6
    LR = 1e-4

    THRESHOLD = 0.4

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = CFG()

# ============================================================
# DATASET
# ============================================================

class CirCorDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.files = os.listdir(cfg.DATA_DIR)

    def __len__(self):
        return len(self.df)

    def _get_file(self, pid):
        wavs = [f for f in self.files if f.startswith(str(pid)) and f.endswith(".wav")]
        return os.path.join(cfg.DATA_DIR, wavs[0]) if wavs else None

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        path = self._get_file(row["Patient ID"])
        if path is None:
            return self.__getitem__((idx+1)%len(self.df))

        y, sr = librosa.load(path, sr=cfg.SR)

        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=cfg.SR, n_mfcc=cfg.N_MFCC)

        # normalize
        mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-6)

        # pad
        if mfcc.shape[1] < cfg.MAX_LEN:
            pad = cfg.MAX_LEN - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0,0),(0,pad)))
        else:
            mfcc = mfcc[:, :cfg.MAX_LEN]

        x = torch.tensor(mfcc).unsqueeze(0).float()
        y = torch.tensor(row["label"]).float()

        return x, y

# ============================================================
# MODEL (EXACT MATCH)
# ============================================================

class SEBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 16),
            nn.ReLU(),
            nn.Linear(channels // 16, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class Model(nn.Module):
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
            nn.Linear(1280,128),
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

# ============================================================
# LOSS (weighted for recall)
# ============================================================

def get_loss():
    pos_weight = torch.tensor([2.0]).to(cfg.DEVICE)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# ============================================================
# TRAIN / EVAL
# ============================================================

def train_epoch(model, loader, opt, loss_fn):
    model.train()
    total = 0

    for x,y in loader:
        x,y = x.to(cfg.DEVICE), y.to(cfg.DEVICE)

        opt.zero_grad()
        out = model(x).squeeze()

        loss = loss_fn(out, y)
        loss.backward()
        opt.step()

        total += loss.item()

    return total

def evaluate(model, loader):
    model.eval()

    y_true, y_pred, y_score = [], [], []

    with torch.no_grad():
        for x,y in loader:
            x = x.to(cfg.DEVICE)

            logits = model(x).squeeze()
            prob = torch.sigmoid(logits)

            pred = (prob > cfg.THRESHOLD).int()

            y_true.extend(y.numpy())
            y_pred.extend(pred.cpu().numpy())
            y_score.extend(prob.cpu().numpy())

    return {
        "acc": accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_score)
    }

# ============================================================
# MAIN
# ============================================================

def main():
    print("Loading data...")

    df = pd.read_csv(cfg.CSV_PATH)
    df = df[df["Murmur"] != "Unknown"]
    df["label"] = df["Murmur"].map({"Present":1,"Absent":0})

    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"])

    train_loader = DataLoader(CirCorDataset(train_df), batch_size=cfg.BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(CirCorDataset(test_df), batch_size=cfg.BATCH_SIZE)

    print("Loading model...")

    model = Model().to(cfg.DEVICE)
    model.load_state_dict(torch.load(cfg.MODEL_PATH, map_location=cfg.DEVICE))

    # freeze
    for p in model.parameters():
        p.requires_grad = False

    # unfreeze last layers
    for p in model.fc.parameters():
        p.requires_grad = True

    for p in model.conv.parameters():
        p.requires_grad = True

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.LR
    )

    loss_fn = get_loss()

    print("Training...")

    for epoch in range(cfg.EPOCHS):
        loss = train_epoch(model, train_loader, optimizer, loss_fn)
        metrics = evaluate(model, test_loader)

        print(f"\nEpoch {epoch+1}")
        print(f"Loss: {loss:.4f}")
        print(metrics)

    print("\nFINAL RESULTS:", evaluate(model, test_loader))


if __name__ == "__main__":
    main()