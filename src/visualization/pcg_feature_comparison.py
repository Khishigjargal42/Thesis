import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

X = np.load("data/processed/X.npy")
y = np.load("data/processed/y.npy")

normal = X[y == 0][0]
abnormal = X[y == 1][0]

sr = 2000

# spectrogram
spec_n = librosa.amplitude_to_db(abs(librosa.stft(normal)))
spec_a = librosa.amplitude_to_db(abs(librosa.stft(abnormal)))

# mel spectrogram
mel_n = librosa.power_to_db(
    librosa.feature.melspectrogram(y=normal, sr=sr)
)

mel_a = librosa.power_to_db(
    librosa.feature.melspectrogram(y=abnormal, sr=sr)
)

# mfcc
mfcc_n = librosa.feature.mfcc(y=normal, sr=sr, n_mfcc=40)
mfcc_a = librosa.feature.mfcc(y=abnormal, sr=sr, n_mfcc=40)

# waveform
plt.subplot(4,2,1)
plt.plot(normal)
plt.title("Normal - Waveform")

plt.subplot(4,2,2)
plt.plot(abnormal)
plt.title("Abnormal - Waveform")

# spectrogram
plt.subplot(4,2,3)
librosa.display.specshow(spec_n, sr=sr)
plt.title("Normal - Spectrogram")

plt.subplot(4,2,4)
librosa.display.specshow(spec_a, sr=sr)
plt.title("Abnormal - Spectrogram")

# mel
plt.subplot(4,2,5)
librosa.display.specshow(mel_n, sr=sr, y_axis='mel')
plt.title("Normal - Mel Spectrogram")

plt.subplot(4,2,6)
librosa.display.specshow(mel_a, sr=sr, y_axis='mel')
plt.title("Abnormal - Mel Spectrogram")

# mfcc
plt.subplot(4,2,7)
librosa.display.specshow(mfcc_n, x_axis='time')
plt.title("Normal - MFCC")

plt.subplot(4,2,8)
librosa.display.specshow(mfcc_a, x_axis='time')
plt.title("Abnormal - MFCC")

plt.tight_layout()

plt.savefig("figures/pcg_feature_comparison.png", dpi=300)

plt.show()