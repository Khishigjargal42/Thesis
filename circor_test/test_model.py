# test_model.py
import numpy as np
import torch
import torch.nn as nn

# infer.py-с model-оо import хийнэ
from infer import ParallelCNN_SE

device = torch.device("cpu")
model = ParallelCNN_SE().to(device)
model.load_state_dict(torch.load("parallelcnn_mfcc_attention.pth", map_location=device))
model.eval()

X = np.load("mfcc.npy")   # shape: (N, 40, 16)
y = np.load("labels.npy")

MU    = -4.220855349111927
SIGMA = 66.5658466625049

X_norm = (X - MU) / SIGMA
X_t    = torch.tensor(X_norm, dtype=torch.float32).unsqueeze(1)  # (N,1,40,16)

with torch.no_grad():
    logits = model(X_t).squeeze()
    probs  = torch.sigmoid(logits).numpy()

preds = (probs > 0.5).astype(int)
acc   = (preds == y).mean()

from collections import Counter
print("Label distribution:", Counter(y.tolist()))
print(f"Accuracy : {acc*100:.1f}%")
print(f"Prob range: {probs.min():.4f} – {probs.max():.4f}")
print(f"Prob mean : {probs.mean():.4f}")

# Abnormal (label=1) дээрх дундаж score
ab_mask = y == 1
no_mask = y == 0
print(f"\nAbnormal avg score: {probs[ab_mask].mean():.4f}")
print(f"Normal avg score  : {probs[no_mask].mean():.4f}")