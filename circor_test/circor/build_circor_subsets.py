import pandas as pd
import numpy as np

# =========================
# CONFIG
# =========================
CSV_PATH = "training_data.csv"   # metadata file
OUTPUT_DIR = "circor_subsets"

RAW_SIZE = 1000
FILTERED_SIZE = 1000
BALANCED_SIZE = 600   # 300 normal + 300 abnormal

RANDOM_SEED = 42

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(CSV_PATH)

print("Original dataset:", df.shape)

# =========================
# CLEAN BASIC
# =========================
df = df[df["Outcome"].isin(["Normal", "Abnormal"])].copy()

print("After Outcome filter:", df.shape)

# =========================
# HELPER: PATIENT-LEVEL SAMPLING
# =========================
def patient_level_sample(df, n):
    patients = df["Patient ID"].unique()
    np.random.seed(RANDOM_SEED)
    selected = np.random.choice(patients, size=min(len(patients), n), replace=False)
    return df[df["Patient ID"].isin(selected)]

# =========================
# 1. RAW SUBSET
# =========================
raw_subset = df.sample(n=min(RAW_SIZE, len(df)), random_state=RANDOM_SEED)

print("Raw subset:", raw_subset.shape)

# =========================
# 2. FILTERED SUBSET
# =========================

filtered = df.copy()

# --- Age filter
filtered = filtered[filtered["Age"].isin(["Child", "Adolescent"])]

# --- Valid recording locations
valid_locations = ["AV", "MV", "PV", "TV"]

def check_location(x):
    if pd.isna(x):
        return False
    return any(loc in x for loc in valid_locations)

filtered = filtered[filtered["Recording locations:"].apply(check_location)]

print("After filtering:", filtered.shape)

filtered_subset = filtered.sample(
    n=min(FILTERED_SIZE, len(filtered)),
    random_state=RANDOM_SEED
)

print("Filtered subset:", filtered_subset.shape)

# =========================
# 3. BALANCED SUBSET
# =========================

normal_df = filtered[filtered["Outcome"] == "Normal"]
abnormal_df = filtered[filtered["Outcome"] == "Abnormal"]

n_each = BALANCED_SIZE // 2

normal_sample = normal_df.sample(n=min(n_each, len(normal_df)), random_state=RANDOM_SEED)
abnormal_sample = abnormal_df.sample(n=min(n_each, len(abnormal_df)), random_state=RANDOM_SEED)

balanced_subset = pd.concat([normal_sample, abnormal_sample]).sample(frac=1, random_state=RANDOM_SEED)

print("Balanced subset:", balanced_subset.shape)
print("Class distribution:")
print(balanced_subset["Outcome"].value_counts())

# =========================
# SAVE
# =========================
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

raw_subset.to_csv(f"{OUTPUT_DIR}/raw_subset.csv", index=False)
filtered_subset.to_csv(f"{OUTPUT_DIR}/filtered_subset.csv", index=False)
balanced_subset.to_csv(f"{OUTPUT_DIR}/balanced_subset.csv", index=False)

print("\nSaved to:", OUTPUT_DIR)