import numpy as np
import matplotlib.pyplot as plt
import librosa
from torch import signal
X = np.load("data/processed/X.npy")
signal = X[0]


mel = librosa.feature.melspectrogram(
    y=signal,
    sr=2000,
    n_mels=128
)

logmel = librosa.power_to_db(mel)

plt.figure(figsize=(10,4))

librosa.display.specshow(
    logmel,
    sr=2000,
    x_axis='time',
    y_axis='mel'
)

plt.title("Log-Mel Spectrogram")
plt.colorbar()

plt.savefig("figures/logmel.png", dpi=300)
plt.show()