import numpy as np
import librosa
from tqdm import tqdm
import os

# segmented signals
X = np.load("data/processed/X.npy")
y = np.load("data/processed/y.npy")

mfcc_features = []

for signal in tqdm(X):

    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=2000,
        n_mfcc=40
    )

    mfcc_features.append(mfcc)

mfcc_features = np.array(mfcc_features)

print("MFCC shape:", mfcc_features.shape)

os.makedirs("data/features", exist_ok=True)

np.save("data/features/mfcc.npy", mfcc_features)
np.save("data/features/mfcc_labels.npy", y)