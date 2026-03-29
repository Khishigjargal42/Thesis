import os
import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")

durations = []
sampling_rates = []

normal_count = 0
abnormal_count = 0

folder_stats = {}

SEGMENT_DURATION = 2.0
OVERLAP = 0.5

print("\nExploring dataset...\n")

for folder in os.listdir(DATA_DIR):

    folder_path = os.path.join(DATA_DIR, folder)

    if not os.path.isdir(folder_path):
        continue

    reference_path = os.path.join(folder_path, "REFERENCE.csv")

    if not os.path.exists(reference_path):
        continue

    df = pd.read_csv(reference_path, header=None)

    folder_normal = 0
    folder_abnormal = 0

    for _, row in df.iterrows():

        filename = row[0] + ".wav"
        label = str(row[1]).strip().lower()

        file_path = os.path.join(folder_path, filename)

        if not os.path.exists(file_path):
            continue

        y, sr = librosa.load(file_path, sr=None)

        duration = len(y) / sr

        durations.append(duration)
        sampling_rates.append(sr)

        if label in ["1", "normal"]:
            normal_count += 1
            folder_normal += 1
        else:
            abnormal_count += 1
            folder_abnormal += 1

    folder_stats[folder] = {
        "normal": folder_normal,
        "abnormal": folder_abnormal
    }

durations = np.array(durations)

total_recordings = len(durations)

# Estimate segments after preprocessing
segment_length = SEGMENT_DURATION
step = SEGMENT_DURATION * (1 - OVERLAP)

estimated_segments = np.sum(
    np.maximum(0, np.floor((durations - segment_length) / step) + 1)
)

print("===== DATASET OVERVIEW =====")

print("Total recordings:", total_recordings)
print("Normal recordings:", normal_count)
print("Abnormal recordings:", abnormal_count)

print("\nClass ratio:")

print("Normal %:", round(normal_count / total_recordings * 100, 2))
print("Abnormal %:", round(abnormal_count / total_recordings * 100, 2))

print("\nSampling rate unique:", set(sampling_rates))

print("\n===== DURATION STATISTICS =====")

print("Average duration:", round(np.mean(durations), 2), "seconds")
print("Median duration:", round(np.median(durations), 2), "seconds")
print("Min duration:", round(np.min(durations), 2), "seconds")
print("Max duration:", round(np.max(durations), 2), "seconds")

print("\nEstimated segments after preprocessing:", int(estimated_segments))

print("\n===== FOLDER DISTRIBUTION =====")

for folder, stats in folder_stats.items():
    print(
        f"{folder} → Normal: {stats['normal']} | Abnormal: {stats['abnormal']}"
    )

# Plot duration histogram
plt.figure(figsize=(8,4))
plt.hist(durations, bins=30)
plt.title("Recording Duration Distribution")
plt.xlabel("Seconds")
plt.ylabel("Count")
plt.tight_layout()
plt.show()