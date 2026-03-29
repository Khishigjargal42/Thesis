import numpy as np
import matplotlib.pyplot as plt

X = np.load("data/processed/X.npy")

signal = X[0]

plt.figure(figsize=(10,4))
plt.plot(signal)
plt.title("Heart Sound Signal (PCG Waveform)")
plt.xlabel("Samples")
plt.ylabel("Amplitude")

plt.savefig("figures/waveform.png", dpi=300)
plt.show()