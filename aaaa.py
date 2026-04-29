import numpy as np
import librosa
import torch

SR=2000; SEG=4000; STEP=2000; N_FFT=512; HOP_LENGTH=256; N_MELS=128
NORM_MEAN=-60.8221; NORM_STD=21.9991

def preprocess_wav(wav_path):
    signal, _ = librosa.load(wav_path, sr=SR, mono=True)

    # Step=2000 overlap segments — X.npy-тай адил
    segments = []
    for start in range(0, len(signal) - SEG + 1, STEP):
        segments.append(signal[start:start+SEG])

    if not segments:
        segments = [np.pad(signal, (0, SEG - len(signal)))]

    # Хамгийн өндөр RMS segment авна
    rms_vals = [np.sqrt(np.mean(s**2)) for s in segments]
    best_seg = segments[np.argmax(rms_vals)]

    # Mel
    mel    = librosa.feature.melspectrogram(
        y=best_seg, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    print(f"  mel mean: {mel_db.mean():.2f}  (expected: ~-60.8)")

    mel_norm = (mel_db - NORM_MEAN) / (NORM_STD + 1e-8)
    return mel_db, mel_norm

# Test — training-д ашигласан файл
for path, true in [
    ("data/raw/training-a/a0001.wav", "ABN"),
    ("data/raw/training-a/a0007.wav", "NOR"),
    ("data/raw/training-b/b0001.wav", "ABN"),
]:
    print(f"\n{path} (true={true})")
    mel_db, mel_norm = preprocess_wav(path)
    t    = torch.FloatTensor(mel_norm).unsqueeze(0).unsqueeze(0)
    prob = torch.sigmoid(model(t)).item()
    print(f"  prob_abn={prob:.3f}  pred={'ABN' if prob>=0.5 else 'NOR'}")