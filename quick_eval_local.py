import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from src.models.resnet2d import ResNet2D


def quick_evaluate(feature_file, label_file, model_path,
                   max_samples=10000):

    device = torch.device('cpu')

    # =========================
    # LOAD DATA
    # =========================
    X = np.load(feature_file)
    y = np.load(label_file)

    if max_samples:
        X = X[:max_samples]
        y = y[:max_samples]

    # =========================
    # NORMALIZE (IMPORTANT ⚠️)
    # =========================
    mean, std = X.mean(), X.std()
    X = (X - mean) / (std + 1e-8)

    X = torch.FloatTensor(X).unsqueeze(1)

    # =========================
    # LOAD MODEL
    # =========================
    model = ResNet2D()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # =========================
    # INFERENCE
    # =========================
    with torch.no_grad():
        logits = model(X).squeeze()
        probs = torch.sigmoid(logits).numpy()
        preds = (probs >= 0.5).astype(int)

    # =========================
    # METRICS
    # =========================
    print("\n=== RESULT ===")
    print(f"Accuracy:  {accuracy_score(y, preds):.4f}")
    print(f"Precision: {precision_score(y, preds, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y, preds):.4f}")
    print(f"F1-score:  {f1_score(y, preds):.4f}")
    print(f"AUC:       {roc_auc_score(y, probs):.4f}")
    print("Confusion:")
    print(confusion_matrix(y, preds))


# =========================
# RUN
# =========================
if __name__ == "__main__":

    quick_evaluate(
        feature_file="data/features/spec.npy",
        label_file="data/features/spec_labels.npy",
        model_path="models/resnet2d_spec.pth",  # ← train хийсэн model
        max_samples=10000
    )