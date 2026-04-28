import os, pandas as pd
import random

base = 'Thesis/circor_test'

df = pd.read_csv(f"{base}/training_data.csv")
print("Columns:", df.columns.tolist())
print("\nOutcome:", df["Outcome"].value_counts().to_string())
print("\nSample rows:")
print(df[["Patient ID","Murmur","Outcome"]].head(5).to_string())

# WAV файл нэрийг шалгах
wavs = [f for f in os.listdir(base) if f.endswith(".wav")]
print(f"\nTotal WAV: {len(wavs)}")
print("Example WAV names:", wavs[:5])