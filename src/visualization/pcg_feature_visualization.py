import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────
X_PATH = "data/processed/X.npy"
Y_PATH = "data/processed/y.npy"
SAVE_PATH = "figures/pcg_feature_visualization.png"

# ── Parameters ────────────────────────────────────────────────────────────────
SR = 2000
N_FFT = 256
HOP_LENGTH = 64
WIN_LENGTH = 128
N_MELS = 64
N_MFCC = 40
FMAX = 1000
CMAP = "magma"

# ── Load data ─────────────────────────────────────────────────────────────────
X = np.load(X_PATH, allow_pickle=True)
y = np.load(Y_PATH, allow_pickle=True)

normal = np.squeeze(X[y == 0][0]).astype(np.float32)
abnormal = np.squeeze(X[y == 1][0]).astype(np.float32)

# ── Normalize helper ──────────────────────────────────────────────────────────
def normalize_signal(signal: np.ndarray) -> np.ndarray:
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        return signal / max_val
    return signal

normal = normalize_signal(normal)
abnormal = normalize_signal(abnormal)

# ── Feature extraction ────────────────────────────────────────────────────────
def extract_features(signal: np.ndarray, sr: int):
    # STFT magnitude -> dB
    stft = np.abs(
        librosa.stft(
            signal,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH
        )
    )
    stft_db = librosa.amplitude_to_db(stft, ref=np.max)

    # Mel spectrogram (power)
    mel = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_mels=N_MELS,
        fmax=FMAX,
        power=2.0
    )

    # Log-Mel spectrogram
    logmel_db = librosa.power_to_db(mel, ref=np.max)

    # MFCC from log-mel for consistency
    mfcc = librosa.feature.mfcc(
        S=logmel_db,
        sr=sr,
        n_mfcc=N_MFCC
    )

    return stft_db, mel, logmel_db, mfcc


stft_n, mel_n, logmel_n, mfcc_n = extract_features(normal, SR)
stft_ab, mel_ab, logmel_ab, mfcc_ab = extract_features(abnormal, SR)

# ── Shared color limits for fair comparison ──────────────────────────────────
stft_vmin = min(stft_n.min(), stft_ab.min())
stft_vmax = max(stft_n.max(), stft_ab.max())

mel_vmin = min(mel_n.min(), mel_ab.min())
mel_vmax = max(mel_n.max(), mel_ab.max())

logmel_vmin = min(logmel_n.min(), logmel_ab.min())
logmel_vmax = max(logmel_n.max(), logmel_ab.max())

mfcc_vmin = min(mfcc_n.min(), mfcc_ab.min())
mfcc_vmax = max(mfcc_n.max(), mfcc_ab.max())

# ── Time axes ─────────────────────────────────────────────────────────────────
time_n = np.arange(len(normal)) / SR
time_ab = np.arange(len(abnormal)) / SR

# ── Plot style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8
})

COLORS = {
    "normal": "#1f77b4",
    "abnormal": "#d62728"
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def add_colorbar(fig, img, ax, label="dB"):
    cb = fig.colorbar(img, ax=ax, pad=0.02, fraction=0.046)
    cb.set_label(label, fontsize=8)

def show_spec(
    fig,
    data,
    sr,
    ax,
    y_axis,
    title,
    vmin=None,
    vmax=None,
    label="dB",
    fmax=None
):
    img = librosa.display.specshow(
        data,
        sr=sr,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis=y_axis,
        ax=ax,
        cmap=CMAP,
        vmin=vmin,
        vmax=vmax,
        fmax=fmax
    )
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Time (s)", fontsize=8)
    add_colorbar(fig, img, ax, label=label)
    return img

# ── Create figure ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(5, 2, figsize=(16, 22))
fig.suptitle(
    "PCG Feature Visualization — Хэвийн vs Хэвийн бус",
    fontsize=14,
    fontweight="bold",
    y=0.995
)

# Row 1 — Waveform
for ax, sig, t, label, color in [
    (axes[0, 0], normal, time_n, "Хэвийн", COLORS["normal"]),
    (axes[0, 1], abnormal, time_ab, "Хэвийн бус", COLORS["abnormal"]),
]:
    ax.plot(t, sig, color=color, linewidth=0.8)
    ax.set_title(f"{label} PCG — Waveform", fontsize=10, fontweight="bold")
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel("Amplitude", fontsize=8)
    ax.grid(alpha=0.15)
    ax.margins(x=0)

# Row 2 — Spectrogram (STFT)
show_spec(
    fig, stft_n, SR, axes[1, 0], "hz",
    "Хэвийн — Spectrogram (STFT)",
    vmin=stft_vmin, vmax=stft_vmax, label="dB"
)
show_spec(
    fig, stft_ab, SR, axes[1, 1], "hz",
    "Хэвийн бус — Spectrogram (STFT)",
    vmin=stft_vmin, vmax=stft_vmax, label="dB"
)

# Row 3 — Mel Spectrogram
show_spec(
    fig, mel_n, SR, axes[2, 0], "mel",
    "Хэвийн — Mel Spectrogram (Power)",
    vmin=mel_vmin, vmax=mel_vmax, label="Power", fmax=FMAX
)
show_spec(
    fig, mel_ab, SR, axes[2, 1], "mel",
    "Хэвийн бус — Mel Spectrogram (Power)",
    vmin=mel_vmin, vmax=mel_vmax, label="Power", fmax=FMAX
)

# Row 4 — Log-Mel Spectrogram
show_spec(
    fig, logmel_n, SR, axes[3, 0], "mel",
    "Хэвийн — Log-Mel Spectrogram",
    vmin=logmel_vmin, vmax=logmel_vmax, label="dB", fmax=FMAX
)
show_spec(
    fig, logmel_ab, SR, axes[3, 1], "mel",
    "Хэвийн бус — Log-Mel Spectrogram",
    vmin=logmel_vmin, vmax=logmel_vmax, label="dB", fmax=FMAX
)

# Row 5 — MFCC
show_spec(
    fig, mfcc_n, SR, axes[4, 0], None,
    f"Хэвийн — MFCC ({N_MFCC} coefficients)",
    vmin=mfcc_vmin, vmax=mfcc_vmax, label="Coefficient value"
)
show_spec(
    fig, mfcc_ab, SR, axes[4, 1], None,
    f"Хэвийн бус — MFCC ({N_MFCC} coefficients)",
    vmin=mfcc_vmin, vmax=mfcc_vmax, label="Coefficient value"
)

for ax in axes[4]:
    ax.set_ylabel("MFCC index", fontsize=8)

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
plt.tight_layout(rect=[0, 0, 1, 0.985], h_pad=2.0, w_pad=1.5)
plt.savefig(SAVE_PATH, dpi=300, bbox_inches="tight", facecolor="white")
plt.show()