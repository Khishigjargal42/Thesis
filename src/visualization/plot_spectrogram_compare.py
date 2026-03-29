import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# load dataset
X = np.load("data/processed/X.npy")
y = np.load("data/processed/y.npy")

# find normal and abnormal sample
normal_signal = X[y == 0][0]
abnormal_signal = X[y == 1][0]

# STFT
spec_normal = librosa.stft(normal_signal)
spec_abnormal = librosa.stft(abnormal_signal)

spec_normal_db = librosa.amplitude_to_db(abs(spec_normal))
spec_abnormal_db = librosa.amplitude_to_db(abs(spec_abnormal))

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
librosa.display.specshow(
    spec_normal_db,
    sr=2000,
    x_axis='time',
    y_axis='hz'
)
plt.title("Normal Heart Sound - Spectrogram")
plt.colorbar()

plt.subplot(1,2,2)
librosa.display.specshow(
    spec_abnormal_db,
    sr=2000,
    x_axis='time',
    y_axis='hz'
)
plt.title("Abnormal Heart Sound - Spectrogram")
plt.colorbar()

plt.tight_layout()
plt.savefig("figures/spectrogram_comparison.png", dpi=300)
plt.show()