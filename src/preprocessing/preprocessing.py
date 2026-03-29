"""
Preprocessing Pipeline for PhysioNet 2016 Heart Sound Dataset

Thesis:
Detection of Abnormal Heart Rhythm Based on Heart Sound Signals

Pipeline Steps:
1. Load PCG .wav files
2. Verify sampling rate (2000 Hz)
3. Apply Butterworth bandpass filter (20–400 Hz)
4. Normalize amplitude to [-1, 1]
5. Segment into 2-second windows with 50% overlap
6. Remove extremely low-energy segments
7. Save dataset as NumPy arrays (X.npy, y.npy)
"""

import os
import numpy as np
import librosa
from scipy.signal import butter, filtfilt
from tqdm import tqdm

# ==========================================================
# PARAMETERS
# ==========================================================

DATASET_DIR = "data/raw"
OUTPUT_DIR = "data/processed"

TARGET_SR = 2000

LOWCUT = 20
HIGHCUT = 400

SEGMENT_DURATION = 2.0
OVERLAP = 0.5

ENERGY_THRESHOLD = 1e-8


# ==========================================================
# BUTTERWORTH BANDPASS FILTER
# ==========================================================

def butter_bandpass(lowcut, highcut, fs, order=4):

    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(order, [low, high], btype="band")

    return b, a


def apply_bandpass_filter(signal, sr):

    b, a = butter_bandpass(LOWCUT, HIGHCUT, sr)

    filtered_signal = filtfilt(b, a, signal)

    return filtered_signal


# ==========================================================
# NORMALIZATION
# ==========================================================

def normalize_signal(signal):

    max_val = np.max(np.abs(signal))

    if max_val == 0:
        return signal

    return signal / max_val


# ==========================================================
# SEGMENTATION
# ==========================================================

def segment_signal(signal, sr):

    segment_length = int(SEGMENT_DURATION * sr)

    step = int(segment_length * (1 - OVERLAP))

    segments = []

    for start in range(0, len(signal) - segment_length + 1, step):

        end = start + segment_length

        segment = signal[start:end]

        energy = np.mean(segment ** 2)

        if energy > ENERGY_THRESHOLD:
            segments.append(segment)

    return segments


# ==========================================================
# LABEL PARSER
# ==========================================================
def parse_label_from_header(header_path):

    with open(header_path, "r") as f:
        for line in f:

            line = line.strip().lower()

            if line.startswith("#"):

                if "abnormal" in line:
                    return 1

                elif "normal" in line:
                    return 0

    return None
# ==========================================================
# FILE PROCESSING
# ==========================================================

def process_file(wav_path, header_path):

    label = parse_label_from_header(header_path)

    if label is None:
        return [], []

    signal, sr = librosa.load(wav_path, sr=None)

    if sr != TARGET_SR:

        signal = librosa.resample(signal, orig_sr=sr, target_sr=TARGET_SR)

        sr = TARGET_SR

    signal = apply_bandpass_filter(signal, sr)

    signal = normalize_signal(signal)

    segments = segment_signal(signal, sr)

    labels = [label] * len(segments)

    return segments, labels


# ==========================================================
# DATASET BUILDER
# ==========================================================

def build_dataset(dataset_dir):

    X = []
    y = []

    folders = sorted(os.listdir(dataset_dir))

    for folder in folders:

        folder_path = os.path.join(dataset_dir, folder)

        if not os.path.isdir(folder_path):
            continue

        print(f"\nProcessing folder: {folder}")

        wav_files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]

        for wav_file in tqdm(wav_files):

            wav_path = os.path.join(folder_path, wav_file)

            header_file = wav_file.replace(".wav", ".hea")

            header_path = os.path.join(folder_path, header_file)

            segments, labels = process_file(wav_path, header_path)

            X.extend(segments)

            y.extend(labels)

    X = np.array(X)
    y = np.array(y)

    return X, y


# ==========================================================
# MAIN FUNCTION
# ==========================================================

def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Building dataset...")

    X, y = build_dataset(DATASET_DIR)

    print("\nDataset Created")
    print("Segments shape:", X.shape)
    print("Labels shape:", y.shape)

    np.save(os.path.join(OUTPUT_DIR, "X.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "y.npy"), y)

    print("\nSaved files:")
    print("data/processed/X.npy")
    print("data/processed/y.npy")


# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":
    main()