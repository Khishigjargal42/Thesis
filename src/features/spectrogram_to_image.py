import numpy as np
import matplotlib.pyplot as plt
import librosa.display

mel = np.load("data/features/mel_spectrogram.npy")

sample = mel[0]

plt.figure(figsize=(6,4))
librosa.display.specshow(sample, x_axis='time', y_axis='mel')
plt.colorbar()
plt.title("Mel Spectrogram Example")
plt.show()