import os
import numpy as np
import librosa
from tqdm import tqdm

# segmented signals
X = np.load("data/processed/X.npy")
y = np.load("data/processed/y.npy")

logmel_features = []

for signal in tqdm(X):

    mel = librosa.feature.melspectrogram(
        y=signal,
        sr=2000,
        n_fft=512,
        hop_length=256,
        n_mels=128
    )

    logmel = librosa.power_to_db(mel, ref=np.max)

    logmel_features.append(logmel)

logmel_features = np.array(logmel_features)

print("Log-Mel shape:", logmel_features.shape)

os.makedirs("data/features", exist_ok=True)

np.save("data/features/logmel.npy", logmel_features)
np.save("data/features/logmel_labels.npy", y)