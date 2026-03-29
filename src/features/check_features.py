import numpy as np

print("\nLoading feature files...\n")

mel = np.load("data/features/mel_spectrogram.npy")
mfcc = np.load("data/features/mfcc.npy")
spec = np.load("data/features/spectrogram.npy")
labels = np.load("data/features/labels.npy")

print("Mel spectrogram shape:", mel.shape)
print("MFCC shape:", mfcc.shape)
print("Spectrogram shape:", spec.shape)
print("Labels shape:", labels.shape)

print("\nChecking label distribution...")

print("Normal:", np.sum(labels == 0))
print("Abnormal:", np.sum(labels == 1))