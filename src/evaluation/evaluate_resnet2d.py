import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (accuracy_score, precision_score, 
                             recall_score, f1_score, roc_auc_score,
                             confusion_matrix, roc_curve)
import matplotlib.pyplot as plt
from models.resnet2d import ResNet2D


def evaluate_resnet2d(feature_name: str, data_dir: str, model_dir: str, batch_size=256):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # =========================
    # LOAD DATA
    # =========================
    feat_dir = os.path.join(data_dir, 'normalized')
    X_test = np.load(os.path.join(feat_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(feat_dir, 'y_test.npy'))

    X_test = torch.FloatTensor(X_test).unsqueeze(1)
    y_test_tensor = torch.FloatTensor(y_test)

    loader = DataLoader(TensorDataset(X_test, y_test_tensor),
                        batch_size=batch_size)

    # =========================
    # LOAD MODEL
    # =========================
    model = ResNet2D().to(device)
    model_path = os.path.join(model_dir, f'resnet2d_{feature_name}.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # =========================
    # INFERENCE (BATCH)
    # =========================
    all_probs = []

    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            logits = model(xb).squeeze()
            probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().numpy())

    probs = np.array(all_probs)

    # =========================
    # THRESHOLD
    # =========================
    preds = (probs >= 0.5).astype(int)

    # =========================
    # METRICS
    # =========================
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    cm = confusion_matrix(y_test, preds)

    print(f"\n=== ResNet2D | {feature_name} ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"AUC-ROC:   {auc:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    # =========================
    # ROC CURVE
    # =========================
    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {feature_name}")
    plt.show()

    return {
        'feature': feature_name,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auc': auc
    }


if __name__ == '__main__':
    features = ['mel_spectrogram']

    DATA_DIR = "/content/drive/MyDrive/Thesis/data"
    MODEL_DIR = "/content/drive/MyDrive/Thesis/models"

    results = []

    for feat in features:
        r = evaluate_resnet2d(feat, DATA_DIR, MODEL_DIR)
        results.append(r)

    print("\n=== Нэгтгэл ===")
    for r in results:
        print(f"{r['feature']:20s} | Acc: {r['accuracy']:.4f} | "
              f"F1: {r['f1']:.4f} | AUC: {r['auc']:.4f}")