import os
import numpy as np
import librosa
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
FEATURE_DIR = os.path.join(BASE_DIR, "data", "features")

os.makedirs(FEATURE_DIR, exist_ok=True)

X_PATH = os.path.join(PROCESSED_DIR, "X.npy")
Y_PATH = os.path.join(PROCESSED_DIR, "y.npy")

TARGET_SR = 2000

N_MELS = 128
N_MFCC = 40

N_FFT = 512
HOP_LENGTH = 256

print("\nLoading segmented dataset...")

X = np.load(X_PATH)
y = np.load(Y_PATH)

print("Segments:", X.shape)
print("Labels:", y.shape)

mel_features = []
mfcc_features = []
spectrogram_features = []

print("\nExtracting features...")

for segment in tqdm(X):

    # MEL SPECTROGRAM
    mel = librosa.feature.melspectrogram(
        y=segment,
        sr=TARGET_SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    mel_features.append(mel_db)

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=segment,
        sr=TARGET_SR,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    mfcc_features.append(mfcc)

    # SPECTROGRAM (STFT magnitude)
    stft = np.abs(librosa.stft(
        segment,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    ))

    spectrogram_features.append(stft)

mel_features = np.array(mel_features)
mfcc_features = np.array(mfcc_features)
spectrogram_features = np.array(spectrogram_features)

print("\nFeature shapes:")

print("Mel Spectrogram:", mel_features.shape)
print("MFCC:", mfcc_features.shape)
print("Spectrogram:", spectrogram_features.shape)

np.save(os.path.join(FEATURE_DIR, "mel_spectrogram.npy"), mel_features)
np.save(os.path.join(FEATURE_DIR, "mfcc.npy"), mfcc_features)
np.save(os.path.join(FEATURE_DIR, "spectrogram.npy"), spectrogram_features)

np.save(os.path.join(FEATURE_DIR, "labels.npy"), y)

print("\nSaved features to data/features/")