import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

X = np.load("data/processed/X.npy")
y = np.load("data/processed/y.npy")

normal_signal = X[y == 0][0]
abnormal_signal = X[y == 1][0]

mel_normal = librosa.feature.melspectrogram(
    y=normal_signal,
    sr=2000,
    n_mels=128
)

mel_abnormal = librosa.feature.melspectrogram(
    y=abnormal_signal,
    sr=2000,
    n_mels=128
)

logmel_normal = librosa.power_to_db(mel_normal)
logmel_abnormal = librosa.power_to_db(mel_abnormal)

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
librosa.display.specshow(
    logmel_normal,
    sr=2000,
    x_axis='time',
    y_axis='mel'
)
plt.title("Normal Heart Sound - Log-Mel Spectrogram")
plt.colorbar()

plt.subplot(1,2,2)
librosa.display.specshow(
    logmel_abnormal,
    sr=2000,
    x_axis='time',
    y_axis='mel'
)
plt.title("Abnormal Heart Sound - Log-Mel Spectrogram")
plt.colorbar()

plt.tight_layout()
plt.savefig("figures/logmel_comparison.png", dpi=300)
plt.show()