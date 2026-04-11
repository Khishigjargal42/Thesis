import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             roc_auc_score, confusion_matrix)

from models.resnet2d import ResNet2D


def evaluate_resnet2d_spec(data_dir, model_path, batch_size=256):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\nEvaluating on: {device}")

    # =========================
    # LOAD DATA
    # =========================
    feat_dir = os.path.join(data_dir, 'features')

    X = np.load(os.path.join(feat_dir, 'spec.npy'))
    y = np.load(os.path.join(feat_dir, 'spec_labels.npy'))

    print("Loaded:", X.shape)

    # =========================
    # SPLIT (same as training!)
    # =========================
    from sklearn.model_selection import train_test_split

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42)

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

    # =========================
    # NORMALIZE (IMPORTANT)
    # =========================
    mean, std = X_train.mean(), X_train.std()

    X_test = (X_test - mean) / (std + 1e-8)

    X_test = torch.FloatTensor(X_test).unsqueeze(1)
    y_test = y_test

    loader = DataLoader(
        TensorDataset(X_test, torch.FloatTensor(y_test)),
        batch_size=batch_size
    )

    # =========================
    # LOAD MODEL
    # =========================
    model = ResNet2D().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # =========================
    # INFERENCE
    # =========================
    all_probs = []

    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            logits = model(xb).view(-1)
            probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().numpy())

    probs = np.array(all_probs)
    preds = (probs >= 0.5).astype(int)

    # =========================
    # METRICS
    # =========================
    print("\n=== SPEC RESULT ===")
    print(f"Accuracy:  {accuracy_score(y_test, preds):.4f}")
    print(f"Precision: {precision_score(y_test, preds, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, preds):.4f}")
    print(f"F1-score:  {f1_score(y_test, preds):.4f}")
    print(f"AUC-ROC:   {roc_auc_score(y_test, probs):.4f}")
    print(f"Confusion:\n{confusion_matrix(y_test, preds)}")


if __name__ == "__main__":
    evaluate_resnet2d_spec(
        data_dir="/content/drive/MyDrive/Thesis/data",
        model_path="/content/drive/MyDrive/Thesis/models/resnet2d_spec.pth"
    )