# src/evaluation/evaluate_resnet2d.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, precision_score, 
                             recall_score, f1_score, roc_auc_score,
                             confusion_matrix)
from models.resnet2d import ResNet2D

def evaluate_resnet2d(feature_name: str, data_dir: str, model_dir: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    feat_dir = os.path.join(data_dir, 'normalized')
    X_test = np.load(os.path.join(feat_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(feat_dir, 'y_test.npy'))

    X_test = torch.FloatTensor(X_test).unsqueeze(1)
    
    model = ResNet2D().to(device)
    model_path = os.path.join(model_dir, f'resnet2d_{feature_name}.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        logits = model(X_test.to(device)).squeeze().cpu()
        probs  = torch.sigmoid(logits).numpy()
        preds  = (probs >= 0.5).astype(int)

    print(f"\n=== ResNet2D | {feature_name} ===")
    print(f"Accuracy:  {accuracy_score(y_test, preds):.4f}")
    print(f"Precision: {precision_score(y_test, preds):.4f}")
    print(f"Recall:    {recall_score(y_test, preds):.4f}")
    print(f"F1-score:  {f1_score(y_test, preds):.4f}")
    print(f"AUC-ROC:   {roc_auc_score(y_test, probs):.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, preds)}")
    
    return {
        'feature': feature_name,
        'accuracy': accuracy_score(y_test, preds),
        'precision': precision_score(y_test, preds),
        'recall': recall_score(y_test, preds),
        'f1': f1_score(y_test, preds),
        'auc': roc_auc_score(y_test, probs)
    }

if __name__ == '__main__':
    features = ['mel_spectrogram', 'mfcc', 'spec', 'logmel']
    results = []
    for feat in features:
        r = evaluate_resnet2d(feat, '../../data', '../../models')
        results.append(r)
    
    print("\n=== Нэгтгэл ===")
    for r in results:
        print(f"{r['feature']:20s} | Acc: {r['accuracy']:.4f} | "
              f"F1: {r['f1']:.4f} | AUC: {r['auc']:.4f}")