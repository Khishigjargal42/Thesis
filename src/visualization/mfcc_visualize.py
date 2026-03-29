import numpy as np
import matplotlib.pyplot as plt
import librosa



X = np.load("data/processed/X.npy")
signal = X[0]


mfcc = librosa.feature.mfcc(
    y=signal,
    sr=2000,
    n_mfcc=40
)

plt.figure(figsize=(10,4))

librosa.display.specshow(
    mfcc,
    x_axis='time'
)

plt.title("MFCC Feature Map")
plt.colorbar()

plt.savefig("figures/mfcc.png", dpi=300)
plt.show()