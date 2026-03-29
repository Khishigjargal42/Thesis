import numpy as np

X = np.load("data/processed/X.npy")
y = np.load("data/processed/y.npy")

print("Dataset shape:", X.shape)

print("Normal segments:", np.sum(y == 0))
print("Abnormal segments:", np.sum(y == 1))