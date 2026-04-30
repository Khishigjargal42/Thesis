import os
import numpy as np
import pandas as pd
import torch
import librosa
from scipy.signal import butter, filtfilt
from sklearn.metrics import *
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
CSV_PATH = "circor_subsets/raw_subset.csv"
DATA_DIR = "training_data"
MODEL_PATH = "resnet_rawmel_v1.pth"

TARGET_SR = 2000
LOWCUT = 20
HIGHCUT = 400
SEGMENT_DURATION = 2.0
OVERLAP = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LOAD MODEL
# =========================
model = torch.load(MODEL_PATH, map_location=DEVICE)
model.eval()

# =========================
# FILTER
# =========================
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    return butter(order, [lowcut/nyq, highcut/nyq], btype="band")

def apply_bandpass(signal, sr):
    b, a = butter_bandpass(LOWCUT, HIGHCUT, sr)
    return filtfilt(b, a, signal)

# =========================
# SEGMENT
# =========================
def segment_signal(signal, sr):
    seg_len = int(SEGMENT_DURATION * sr)
    step = int(seg_len * (1 - OVERLAP))
    segments = []

    for start in range(0, len(signal) - seg_len + 1, step):
        seg = signal[start:start+seg_len]
        if np.mean(seg**2) > 1e-8:
            segments.append(seg)

    return segments

# =========================
# FEATURE
# =========================
def extract_logmel(signal, sr):
    mel = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=128)
    return librosa.power_to_db(mel)

# =========================
# PREDICT ONE FILE
# =========================
def predict_wav(wav_path):
    signal, sr = librosa.load(wav_path, sr=None)

    if sr != TARGET_SR:
        signal = librosa.resample(signal, orig_sr=sr, target_sr=TARGET_SR)

    signal = apply_bandpass(signal, TARGET_SR)
    segments = segment_signal(signal, TARGET_SR)

    if len(segments) == 0:
        return None

    preds = []

    for seg in segments:
        feat = extract_logmel(seg, TARGET_SR)
        feat = torch.tensor(feat).unsqueeze(0).unsqueeze(0).float().to(DEVICE)

        with torch.no_grad():
            out = model(feat)
            prob = torch.sigmoid(out).item()

        preds.append(prob)

    return np.mean(preds)

# =========================
# PREDICT PATIENT (🔥 CORE FIX)
# =========================
def predict_patient(patient_id):
    preds = []

    for file in os.listdir(DATA_DIR):
        if file.startswith(str(patient_id) + "_") and file.endswith(".wav"):
            wav_path = os.path.join(DATA_DIR, file)
            p = predict_wav(wav_path)
            if p is not None:
                preds.append(p)

    if len(preds) == 0:
        return None

    return np.mean(preds)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(CSV_PATH)

y_true = []
y_pred = []

print("Running evaluation...")

for _, row in df.iterrows():
    pid = row["Patient ID"]
    label = 1 if row["Outcome"] == "Abnormal" else 0

    prob = predict_patient(pid)

    if prob is None:
        continue

    y_true.append(label)
    y_pred.append(prob)

# =========================
# METRICS
# =========================
y_pred_label = [1 if p >= 0.5 else 0 for p in y_pred]

print("\n===== RESULTS =====")
print("Accuracy :", accuracy_score(y_true, y_pred_label))
print("Precision:", precision_score(y_true, y_pred_label))
print("Recall   :", recall_score(y_true, y_pred_label))
print("F1-score :", f1_score(y_true, y_pred_label))
print("AUC      :", roc_auc_score(y_true, y_pred))

cm = confusion_matrix(y_true, y_pred_label)
print("\nConfusion Matrix:\n", cm)

# =========================
# ROC
# =========================
fpr, tpr, _ = roc_curve(y_true, y_pred)

plt.figure()
plt.plot(fpr, tpr, label="ROC")
plt.plot([0,1],[0,1],'--')
plt.legend()
plt.title("ROC Curve")
plt.savefig("roc.png")

# =========================
# CONFUSION MATRIX
# =========================
import seaborn as sns

plt.figure()
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.savefig("cm.png")