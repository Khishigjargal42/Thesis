import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

SPLIT_DIR = os.path.join(BASE_DIR, "data", "splits")
NORM_DIR = os.path.join(BASE_DIR, "data", "normalized")

os.makedirs(NORM_DIR, exist_ok=True)

print("\nLoading dataset splits...\n")

X_train = np.load(os.path.join(SPLIT_DIR, "X_train.npy"))
X_val = np.load(os.path.join(SPLIT_DIR, "X_val.npy"))
X_test = np.load(os.path.join(SPLIT_DIR, "X_test.npy"))

y_train = np.load(os.path.join(SPLIT_DIR, "y_train.npy"))
y_val = np.load(os.path.join(SPLIT_DIR, "y_val.npy"))
y_test = np.load(os.path.join(SPLIT_DIR, "y_test.npy"))

print("Train shape:", X_train.shape)
print("Validation shape:", X_val.shape)
print("Test shape:", X_test.shape)

# Compute normalization statistics from TRAIN SET ONLY
mean = X_train.mean()
std = X_train.std()

print("\nNormalization stats:")
print("Mean:", mean)
print("Std:", std)

# Normalize
X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std

# Save normalized data
np.save(os.path.join(NORM_DIR, "X_train.npy"), X_train)
np.save(os.path.join(NORM_DIR, "y_train.npy"), y_train)

np.save(os.path.join(NORM_DIR, "X_val.npy"), X_val)
np.save(os.path.join(NORM_DIR, "y_val.npy"), y_val)

np.save(os.path.join(NORM_DIR, "X_test.npy"), X_test)
np.save(os.path.join(NORM_DIR, "y_test.npy"), y_test)

print("\nNormalized datasets saved to data/normalized/")