"""
PCG Heart Sound Classifier - Gradio Demo
ResNet2D + SE Attention + Mel-Spectrogram

Features:
  - Raw wav inference (unseen recordings supported)
  - Input quality validation before inference
  - SE Attention channel visualization
  - Grad-CAM temporal-frequency heatmap
  - Optimized classification threshold (0.35)

Author: Г.Хишигжаргал
Bachelor Thesis, 2026
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

# ══════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "resnet_rawmel_v1.pth")

SR         = 2000
SEG        = 4000
STEP       = 2000
N_FFT      = 512
HOP_LENGTH = 256
N_MELS     = 128

NORM_MEAN  = -48.6446
NORM_STD   = 16.0562
THRESHOLD  = 0.35   # Optimized for recall in clinical screening

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1    = nn.Conv2d(in_ch, out_ch, 3, stride, 1)
        self.bn1      = nn.BatchNorm2d(out_ch)
        self.conv2    = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.bn2      = nn.BatchNorm2d(out_ch)
        self.relu     = nn.ReLU()
        self.se       = SEBlock(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += identity
        return self.relu(out)


class ResNet2D_SE(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.layer1 = BasicBlock(32, 32)
        self.layer2 = BasicBlock(32, 64, stride=2)
        self.layer3 = BasicBlock(64, 128, stride=2)
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.fc     = nn.Linear(128, 1)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)


# ══════════════════════════════════════════════════════════════════
# LOAD MODEL
# ══════════════════════════════════════════════════════════════════
print(f"Device: {DEVICE}")

model = ResNet2D_SE()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()
print(f"Model loaded: {MODEL_PATH}")

# SE Attention hook — captures layer3 channel weights
_se_weights = {}

def _se_hook(module, input, output):
    b, c, _, _ = input[0].size()
    y = module.pool(input[0]).view(b, c)
    w = module.fc(y).squeeze().detach().cpu().numpy()
    _se_weights["layer3"] = w

model.layer3.se.register_forward_hook(_se_hook)


# ══════════════════════════════════════════════════════════════════
# AUDIO QUALITY CHECK  — must be defined before predict()
# ══════════════════════════════════════════════════════════════════
def check_audio_quality(signal: np.ndarray) -> tuple:
    """
    Validates raw signal quality before any preprocessing.
    Runs 5 checks in order of increasing cost.
    Returns: (is_valid: bool, message: str)
    """
    # 1. Minimum duration
    if len(signal) < SR * 1:
        return False, "⚠️ Recording too short. Minimum 2 seconds required."

    # 2. RMS energy — catches silent files
    rms = np.sqrt(np.mean(signal ** 2))
    if rms < 0.0005:
        return False, "⚠️ Signal too quiet. Check stethoscope placement or microphone gain."

    # 3. Clipping — catches gain overload
    clip_ratio = (np.abs(signal) > 0.99).mean()
    if clip_ratio > 0.005:
        return False, "⚠️ Audio clipping detected. Reduce recording level."

    # 4. Near-constant signal — catches DC offset or silent files
    if np.std(signal) < 0.0005:
        return False, "⚠️ Signal is near-constant. This does not appear to be a PCG recording."

    # 5. Frequency content — heart sounds concentrate in 20-200 Hz
    seg          = signal[:min(len(signal), SR * 5)]
    fft          = np.abs(np.fft.rfft(seg))
    freqs        = np.fft.rfftfreq(len(seg), 1 / SR)
    heart_energy = fft[(freqs >= 20) & (freqs <= 200)].sum()
    total_energy = fft.sum() + 1e-8

    if (heart_energy / total_energy) < 0.05:
        return False, "⚠️ No cardiac frequency content detected (20-200 Hz). Please use a PCG/stethoscope recording."

    return True, "✅ Audio quality acceptable."


# ══════════════════════════════════════════════════════════════════
# PREPROCESSING
# ══════════════════════════════════════════════════════════════════
def load_signal(audio_input) -> np.ndarray:
    """Loads audio from filepath or Gradio microphone tuple."""
    if isinstance(audio_input, tuple):
        native_sr, signal = audio_input
        signal = signal.astype(np.float32)
        if signal.ndim == 2:
            signal = signal.mean(axis=1)
        if np.abs(signal).max() > 1.0:
            signal = signal / 32768.0
        if native_sr != SR:
            signal = librosa.resample(signal, orig_sr=native_sr, target_sr=SR)
    else:
        signal, _ = librosa.load(audio_input, sr=SR, mono=True)
    return signal


def preprocess(signal: np.ndarray) -> tuple:
    """
    Converts raw signal to normalized mel-spectrogram tensor.

    Pipeline matches training exactly:
      1. Step=2000 overlap segmentation
      2. Highest-RMS segment selection
      3. Mel-spectrogram (128x16)
      4. z-score normalization with train stats

    Returns: (segment, mel_db, tensor)
    """
    # Overlap segments
    segments = []
    for start in range(0, len(signal) - SEG + 1, STEP):
        segments.append(signal[start:start + SEG])
    if not segments:
        segments = [np.pad(signal, (0, SEG - len(signal)))]

    # Best segment by RMS
    rms_vals = [np.sqrt(np.mean(s ** 2)) for s in segments]
    segment  = segments[int(np.argmax(rms_vals))]

    # Mel-spectrogram
    mel    = librosa.feature.melspectrogram(
        y=segment, sr=SR,
        n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    mel_db   = librosa.power_to_db(mel, ref=np.max)
    mel_norm = (mel_db - NORM_MEAN) / (NORM_STD + 1e-8)
    tensor   = torch.FloatTensor(mel_norm).unsqueeze(0).unsqueeze(0).to(DEVICE)

    return segment, mel_db, tensor


# ══════════════════════════════════════════════════════════════════
# GRAD-CAM
# ══════════════════════════════════════════════════════════════════
def compute_gradcam(tensor: torch.Tensor) -> np.ndarray:
    """
    Computes Grad-CAM heatmap over the mel-spectrogram input.

    Targets layer3 feature maps — the last spatial layer before
    global average pooling. Gradient of the abnormality logit
    w.r.t. layer3 activations gives per-channel importance weights.
    The weighted sum of activations is upsampled to (128, 16) and
    overlaid on the mel-spectrogram.

    Returns: heatmap np.ndarray (128, 16), values in [0, 1]
    """
    tensor = tensor.clone().requires_grad_(True)

    # Forward pass with gradient tracking
    activations = {}
    gradients   = {}

    def fwd_hook(module, input, output):
        activations["layer3"] = output

    def bwd_hook(module, grad_in, grad_out):
        gradients["layer3"] = grad_out[0]

    fwd_handle = model.layer3.register_forward_hook(fwd_hook)
    bwd_handle = model.layer3.register_full_backward_hook(bwd_hook)

    model.zero_grad()
    logit = model(tensor)
    logit.backward()

    fwd_handle.remove()
    bwd_handle.remove()

    # Grad-CAM computation
    acts  = activations["layer3"].squeeze(0)   # (128, H, W)
    grads = gradients["layer3"].squeeze(0)     # (128, H, W)

    # Global average pool gradients to get channel weights
    weights = grads.mean(dim=(1, 2))           # (128,)

    # Weighted sum of activation maps
    cam = torch.zeros(acts.shape[1:], device=DEVICE)
    for i, w in enumerate(weights):
        cam += w * acts[i]

    # ReLU — keep only positive contributions
    cam = F.relu(cam)

    # Upsample to input mel size (128, 16)
    cam = cam.unsqueeze(0).unsqueeze(0)
    cam = F.interpolate(cam, size=(N_MELS, 16), mode="bilinear", align_corners=False)
    cam = cam.squeeze().detach().cpu().numpy()

    # Normalize to [0, 1]
    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    else:
        cam = np.zeros_like(cam)

    return cam


# ══════════════════════════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════════════════════════
def plot_waveform(segment: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 2.5))
    t = np.arange(len(segment)) / SR
    ax.plot(t, segment, color="#1565C0", linewidth=0.8, alpha=0.9)
    ax.fill_between(t, segment, alpha=0.12, color="#1565C0")
    ax.axhline(0, color="gray", linewidth=0.4, linestyle="--")
    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_ylabel("Amplitude", fontsize=11)
    ax.set_title("PCG Waveform — Highest RMS Segment", fontsize=12, fontweight="bold")
    ax.set_xlim(0, len(segment) / SR)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def plot_mel(mel_db: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    img = ax.imshow(
        mel_db, aspect="auto", origin="lower",
        cmap="magma", interpolation="nearest"
    )
    plt.colorbar(img, ax=ax, label="dB")
    ax.set_xlabel("Time Frame", fontsize=11)
    ax.set_ylabel("Mel Filter Bank", fontsize=11)
    ax.set_title("Mel-Spectrogram (128 × 16) — Model Input", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_gradcam(mel_db: np.ndarray, cam: np.ndarray) -> plt.Figure:
    """
    Overlays Grad-CAM heatmap on mel-spectrogram.
    Highlights regions (frequency x time) that drove the classification.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: raw mel
    axes[0].imshow(mel_db, aspect="auto", origin="lower", cmap="magma")
    axes[0].set_title("Mel-Spectrogram", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Time Frame")
    axes[0].set_ylabel("Mel Filter Bank")

    # Right: mel + Grad-CAM overlay
    axes[1].imshow(mel_db, aspect="auto", origin="lower", cmap="magma", alpha=0.6)
    axes[1].imshow(cam, aspect="auto", origin="lower", cmap="jet", alpha=0.5,
                   vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM Overlay\n(Red = most discriminative region)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Time Frame")
    axes[1].set_ylabel("Mel Filter Bank")

    fig.suptitle("Temporal-Frequency Attention (Grad-CAM)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_se_attention(weights: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 3))
    top_k   = 20
    top_idx = set(np.argsort(weights)[-top_k:])
    colors  = ["#E53935" if i in top_idx else "#90CAF9"
               for i in range(len(weights))]
    ax.bar(range(len(weights)), weights, color=colors, width=1.0, edgecolor="none")
    ax.set_xlabel("Channel Index (128 channels)", fontsize=11)
    ax.set_ylabel("SE Attention Weight", fontsize=11)
    ax.set_title(
        "SE Attention Weights — Layer 3\nRed: Top-20 most active frequency channels",
        fontsize=12, fontweight="bold"
    )
    ax.set_xlim(-1, len(weights))
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def plot_metrics() -> plt.Figure:
    metrics = {
        "Accuracy":  0.9524,
        "Precision": 0.8898,
        "Recall":    0.9105,
        "F1-Score":  0.9000,
        "AUC-ROC":   0.9892,
    }
    fig, ax = plt.subplots(figsize=(9, 2.2))
    ax.axis("off")
    table = ax.table(
        cellText=[[f"{v:.4f}" for v in metrics.values()]],
        colLabels=list(metrics.keys()),
        loc="center",
        cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1.2, 2.4)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1565C0")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#E3F2FD")
            cell.set_text_props(fontweight="bold")
    ax.set_title(
        "ResNet2D + SE Attention + Mel-Spectrogram — Test Set Performance",
        fontsize=12, fontweight="bold", pad=14
    )
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════
# PREDICT
# ══════════════════════════════════════════════════════════════════
def predict(audio_input):
    EMPTY = (None, None, None, None, None, {}, "N/A", "N/A")

    if audio_input is None:
        return EMPTY

    try:
        # 1. Load raw signal
        signal = load_signal(audio_input)

        # 2. Quality check on raw signal — BEFORE any preprocessing
        is_valid, quality_msg = check_audio_quality(signal)
        if not is_valid:
            return (
                None, None, None, None, None,
                {"⚠️ Quality Issue": 1.0},
                "N/A",
                quality_msg
            )

        # 3. Preprocessing
        segment, mel_db, tensor = preprocess(signal)

        # 4. Inference
        with torch.no_grad():
            logit = model(tensor).squeeze().item()

        prob_abn = float(torch.sigmoid(torch.tensor(logit)))
        prob_nor = 1.0 - prob_abn
        abnormal = prob_abn >= THRESHOLD

        # 5. Grad-CAM — requires gradient, run separately
        cam = compute_gradcam(tensor)

        # 6. Plots
        fig_wave    = plot_waveform(segment)
        fig_mel     = plot_mel(mel_db)
        fig_gradcam = plot_gradcam(mel_db, cam)
        fig_attn    = plot_se_attention(_se_weights.get("layer3", np.zeros(128)))
        fig_met     = plot_metrics()

        # 7. Result
        label_dict = {
            "🔴 ABNORMAL (Хэвийн бус)": round(prob_abn, 4),
            "🟢 NORMAL (Хэвийн)":       round(prob_nor, 4),
        }

        conf_pct  = prob_abn * 100 if abnormal else prob_nor * 100
        conf_str  = f"{conf_pct:.1f}%"
        score_str = f"logit: {logit:.4f}  sigmoid: {prob_abn:.4f}  threshold: {THRESHOLD}"

        return fig_wave, fig_mel, fig_gradcam, fig_attn, fig_met, label_dict, conf_str, score_str

    except Exception as e:
        import traceback
        traceback.print_exc()
        return (
            None, None, None, None, None,
            {"❌ Error": 1.0}, "N/A", str(e)[:120]
        )


# ══════════════════════════════════════════════════════════════════
# GRADIO UI
# ══════════════════════════════════════════════════════════════════
with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="indigo",
        font=gr.themes.GoogleFont("IBM Plex Sans")
    ),
    title="PCG Heart Sound Classifier"
) as demo:

    gr.Markdown("""
    # 🫀 Abnormal Heart Sound Detection System
    **ResNet2D + SE Attention + Mel-Spectrogram** with Grad-CAM interpretability

    > PhysioNet/CinC 2016 PCG Dataset — 3,240 recordings | SR: 2,000 Hz
    > Bachelor Thesis — Г.Хишигжаргал, 2026
    ---
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📂 Input")
            audio_input = gr.Audio(
                label="Upload PCG Recording (.wav)",
                type="filepath",
                sources=["upload", "microphone"]
            )
            predict_btn = gr.Button("🔍 Analyze", variant="primary", size="lg")
            gr.Markdown("""
            **Requirements:**
            - `.wav` format, minimum 2 seconds
            - Any sample rate (auto-resampled to 2,000 Hz)
            - PCG / digital stethoscope recording recommended
            - Live microphone recording also supported
            """)

        with gr.Column(scale=1):
            gr.Markdown("### 📊 Classification Result")
            result_label = gr.Label(label="Prediction", num_top_classes=2)
            with gr.Row():
                confidence_out = gr.Textbox(
                    label="Confidence", interactive=False, scale=1
                )
                score_out = gr.Textbox(
                    label="Raw Score", interactive=False, scale=2
                )

    gr.Markdown("---\n### 📈 Signal Visualization")
    with gr.Row():
        fig_wave_out = gr.Plot(label="Waveform — Highest RMS Segment")
        fig_mel_out  = gr.Plot(label="Mel-Spectrogram — Model Input (128×16)")

    gr.Markdown("""
    ---
    ### 🔥 Grad-CAM — Temporal-Frequency Interpretability
    Shows **where in time and frequency** the model found evidence of abnormality.
    Red regions indicate the most discriminative areas of the mel-spectrogram.
    This corresponds to specific cardiac phases (e.g., systolic murmur, S3/S4 sounds).
    """)
    fig_gradcam_out = gr.Plot(label="Grad-CAM Heatmap Overlay")

    gr.Markdown("""
    ---
    ### 🧠 SE Attention — Channel Importance
    Shows which mel frequency channels Layer 3 attended to.
    **Red bars** = Top-20 most active channels across the entire segment.
    """)
    fig_attn_out = gr.Plot(label="SE Channel Attention Weights — Layer 3")

    gr.Markdown("---\n### 📋 Model Performance (Test Set)")
    fig_metrics_out = gr.Plot(label="Evaluation Metrics")

    predict_btn.click(
        fn=predict,
        inputs=[audio_input],
        outputs=[
            fig_wave_out, fig_mel_out, fig_gradcam_out,
            fig_attn_out, fig_metrics_out,
            result_label, confidence_out, score_out
        ]
    )

    gr.Markdown("""
    ---
    <div style='text-align:center; color:#888; font-size:12px;'>
    Department of Computer Science | MUST | 2026
    &nbsp;|&nbsp; Accuracy: 95.24%
    &nbsp;|&nbsp; F1: 90.00%
    &nbsp;|&nbsp; AUC: 98.92%
    &nbsp;|&nbsp; Recall: 91.05%
    &nbsp;|&nbsp; Threshold: 0.35
    </div>
    """)


# ══════════════════════════════════════════════════════════════════
# LAUNCH
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    demo.launch(share=True, server_port=7860, show_error=True)