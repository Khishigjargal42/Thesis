import numpy as np
from sklearn.model_selection import train_test_split

mel = np.load("/content/drive/MyDrive/Thesis/data/features/mel_spectrogram.npy")
y   = np.load("/content/drive/MyDrive/Thesis/data/features/labels.npy")

idx = np.arange(len(y))
_, idx_test = train_test_split(idx, test_size=0.2, stratify=y, random_state=42)

normal_idx   = idx_test[y[idx_test] == 0][:15]
abnormal_idx = idx_test[y[idx_test] == 1][:15]

np.save("/content/drive/MyDrive/Thesis/data/features/demo_normal_mel.npy",   mel[normal_idx])
np.save("/content/drive/MyDrive/Thesis/data/features/demo_abnormal_mel.npy", mel[abnormal_idx])

print(f"Normal   : {len(normal_idx)}")
print(f"Abnormal : {len(abnormal_idx)}")
print("Saved!")