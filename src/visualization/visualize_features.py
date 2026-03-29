import numpy as np
import matplotlib.pyplot as plt
import librosa.display

# Load data
mel = np.load("data/features/mel_spectrogram.npy")
mfcc = np.load("data/features/mfcc.npy")
labels = np.load("data/features/labels.npy")
segments = np.load("data/processed/X.npy")

# Find one normal and one abnormal sample
normal_idx = np.where(labels == 0)[0][0]
abnormal_idx = np.where(labels == 1)[0][0]

normal_wave = segments[normal_idx]
abnormal_wave = segments[abnormal_idx]

normal_mel = mel[normal_idx]
abnormal_mel = mel[abnormal_idx]

normal_mfcc = mfcc[normal_idx]
abnormal_mfcc = mfcc[abnormal_idx]

# Plot waveforms
plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(normal_wave)
plt.title("Normal Heart Sound Waveform")

plt.subplot(1,2,2)
plt.plot(abnormal_wave)
plt.title("Abnormal Heart Sound Waveform")

plt.tight_layout()
plt.show()

# Plot Mel Spectrograms
plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
librosa.display.specshow(normal_mel, x_axis="time", y_axis="mel")
plt.title("Normal Mel Spectrogram")

plt.subplot(1,2,2)
librosa.display.specshow(abnormal_mel, x_axis="time", y_axis="mel")
plt.title("Abnormal Mel Spectrogram")

plt.tight_layout()
plt.show()

# Plot MFCC
plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
librosa.display.specshow(normal_mfcc, x_axis="time")
plt.title("Normal MFCC")

plt.subplot(1,2,2)
librosa.display.specshow(abnormal_mfcc, x_axis="time")
plt.title("Abnormal MFCC")

plt.tight_layout()
plt.show()