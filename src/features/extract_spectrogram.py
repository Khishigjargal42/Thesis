import os
import numpy as np
import librosa
from tqdm import tqdm

# segmented signals
X = np.load("data/processed/X.npy")
y = np.load("data/processed/y.npy")

spec_features = []

for signal in tqdm(X):

    spec = librosa.stft(
        y=signal,
        n_fft=512,
        hop_length=256
    )

    spec = np.abs(spec)

    spec_features.append(spec)

spec_features = np.array(spec_features)

print("Spectrogram shape:", spec_features.shape)

os.makedirs("data/features", exist_ok=True)

np.save("data/features/spec.npy", spec_features)
np.save("data/features/spec_labels.npy", y)