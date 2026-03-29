import os
import numpy as np
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

FEATURE_DIR = os.path.join(BASE_DIR, "data", "features")
SPLIT_DIR = os.path.join(BASE_DIR, "data", "splits")

os.makedirs(SPLIT_DIR, exist_ok=True)

print("\nLoading features...")

X = np.load(os.path.join(FEATURE_DIR, "mel_spectrogram.npy"))
y = np.load(os.path.join(FEATURE_DIR, "labels.npy"))

print("Dataset:", X.shape)

# First split: train vs temp (30%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=42
)

# Second split: validation vs test (15% each)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=42
)

print("\nSplit sizes:")

print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

# Save splits
np.save(os.path.join(SPLIT_DIR, "X_train.npy"), X_train)
np.save(os.path.join(SPLIT_DIR, "y_train.npy"), y_train)

np.save(os.path.join(SPLIT_DIR, "X_val.npy"), X_val)
np.save(os.path.join(SPLIT_DIR, "y_val.npy"), y_val)

np.save(os.path.join(SPLIT_DIR, "X_test.npy"), X_test)
np.save(os.path.join(SPLIT_DIR, "y_test.npy"), y_test)

print("\nSaved splits to data/splits/")