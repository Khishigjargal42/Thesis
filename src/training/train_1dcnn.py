import os
import torch
import torch.nn as nn
import torchaudio
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score    
import librosa
torchaudio.set_audio_backend("soundfile")
# =========================
# CONFIG
# =========================
BASE_PATH = "data/raw"
CSV_PATH = os.path.join(BASE_PATH, "REFERENCES.csv")

BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 10
TARGET_SR = 2000
SEGMENT_SEC = 3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# LOAD CSV
# =========================
df = pd.read_csv(CSV_PATH)

file_paths = []
labels = []

for _, row in df.iterrows():
    path = os.path.join(
        BASE_PATH,
        row["folder"],
        row["record_id"] + ".wav"
    )
    if os.path.exists(path):
        file_paths.append(path)
        labels.append(int(row["label"]))

print(f"Total samples: {len(file_paths)}")

# =========================
# SPLIT
# =========================
train_paths, val_paths, train_labels, val_labels = train_test_split(
    file_paths,
    labels,
    test_size=0.2,
    stratify=labels,
    random_state=42
)

# =========================
# DATASET
# =========================
class PCGDataset(Dataset):
    def __init__(self, paths, labels):
        self.paths = paths
        self.labels = labels
        self.segment_samples = TARGET_SR * SEGMENT_SEC

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]

        waveform, sr = torchaudio.load(path)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != TARGET_SR:
            waveform = torchaudio.transforms.Resample(sr, TARGET_SR)(waveform)

        waveform = waveform / (waveform.abs().max() + 1e-8)

        waveform = self.fix_length(waveform)

        return waveform, torch.tensor(label, dtype=torch.float32)

    def fix_length(self, x):
        length = x.shape[1]

        if length > self.segment_samples:
            start = torch.randint(0, length - self.segment_samples, (1,)).item()
            x = x[:, start:start+self.segment_samples]
        else:
            pad = self.segment_samples - length
            x = torch.nn.functional.pad(x, (0, pad))

        return x

# =========================
# DATALOADER
# =========================
train_ds = PCGDataset(train_paths, train_labels)
val_ds = PCGDataset(val_paths, val_labels)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# =========================
# MODEL
# =========================
class Model1D(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Conv1d(16, 32, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, 5, padding=2),
            nn.ReLU(),

            nn.Conv1d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv(x)
        x = x.squeeze(-1)
        return x

model = Model1D().to(DEVICE)

# =========================
# LOSS + OPTIMIZER
# =========================
pos_weight = torch.tensor([
    (len(train_labels) - sum(train_labels)) / (sum(train_labels) + 1e-8)
]).to(DEVICE)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# =========================
# TRAIN FUNCTION
# =========================
def train_epoch():
    model.train()
    total_loss = 0

    for x, y in train_loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE).unsqueeze(1)

        optimizer.zero_grad()
        out = model(x)

        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

# =========================
# EVAL FUNCTION
# =========================
def evaluate():
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(DEVICE)
            out = torch.sigmoid(model(x)).cpu()

            preds.extend((out.numpy() > 0.5).astype(int))
            targets.extend(y.numpy())

    return f1_score(targets, preds)

# =========================
# TRAIN LOOP
# =========================
best_f1 = 0

for epoch in range(EPOCHS):
    loss = train_epoch()
    f1 = evaluate()

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss:.4f} | F1: {f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), "best_1dcnn.pth")

print("Training complete!")