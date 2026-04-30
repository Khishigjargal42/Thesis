"""
PCG Heart Sound Classifier - Gradio Demo
ResNet2D + SE Attention + Mel-Spectrogram

Features:
  - Multi-segment aggregation (all segments averaged)
  - 5-stage input quality validation with detailed report
  - Grad-CAM on highest-prob_abn segment
  - SE Attention channel visualization
  - Tabbed UI with pre-upload guidance popup
  - Classification threshold optimized at 0.35

Author: Г.Хишигжаргал, Bachelor Thesis 2026
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
THRESHOLD  = 0.35

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

_se_weights = {}

def _se_hook(module, input, output):
    b, c, _, _ = input[0].size()
    y = module.pool(input[0]).view(b, c)
    w = module.fc(y).squeeze().detach().cpu().numpy()
    _se_weights["layer3"] = w

model.layer3.se.register_forward_hook(_se_hook)


# ══════════════════════════════════════════════════════════════════
# AUDIO QUALITY CHECK
# ══════════════════════════════════════════════════════════════════
def check_audio_quality(signal: np.ndarray) -> tuple:
    """
    5-stage quality validation on raw signal before any preprocessing.
    Returns: (is_valid: bool, report: list of (passed, check_name, detail))
    """
    report = []

    # 1. Duration
    duration = len(signal) / SR
    if duration < 2.0:
        report.append((False, "Duration", f"{duration:.1f}s — minimum 2.0s required"))
        return False, report
    else:
        report.append((True, "Duration", f"{duration:.1f}s"))

    # 2. RMS energy
    rms = float(np.sqrt(np.mean(signal ** 2)))
    if rms < 0.0005:
        report.append((False, "Signal Energy", f"RMS={rms:.5f} — too quiet, minimum 0.0005"))
        return False, report
    else:
        report.append((True, "Signal Energy", f"RMS={rms:.5f}"))

    # 3. Clipping
    clip_ratio = float((np.abs(signal) > 0.99).mean())
    if clip_ratio > 0.005:
        report.append((False, "Clipping", f"{clip_ratio*100:.2f}% samples clipped — maximum 0.5%"))
        return False, report
    else:
        report.append((True, "Clipping", f"{clip_ratio*100:.3f}% clipped"))

    # 4. Signal variance
    std = float(np.std(signal))
    if std < 0.0005:
        report.append((False, "Variance", f"std={std:.5f} — signal is near-constant"))
        return False, report
    else:
        report.append((True, "Variance", f"std={std:.5f}"))

    # 5. Cardiac frequency content (20-200 Hz)
    seg          = signal[:min(len(signal), SR * 5)]
    fft          = np.abs(np.fft.rfft(seg))
    freqs        = np.fft.rfftfreq(len(seg), 1.0 / SR)
    heart_energy = float(fft[(freqs >= 20) & (freqs <= 200)].sum())
    total_energy = float(fft.sum()) + 1e-8
    heart_ratio  = heart_energy / total_energy
    if heart_ratio < 0.05:
        report.append((False, "Cardiac Frequency (20-200 Hz)",
                       f"{heart_ratio*100:.1f}% — minimum 5% required. Not a PCG recording?"))
        return False, report
    else:
        report.append((True, "Cardiac Frequency (20-200 Hz)",
                       f"{heart_ratio*100:.1f}% energy in cardiac band"))

    return True, report


def format_quality_report(report: list) -> str:
    """Formats quality check results as readable text."""
    lines = ["**Input Quality Report**\n"]
    for passed, name, detail in report:
        icon = "✅" if passed else "❌"
        lines.append(f"{icon} **{name}**: {detail}")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# SIGNAL LOADING
# ══════════════════════════════════════════════════════════════════
def load_signal(audio_input) -> np.ndarray:
    """Loads and normalizes audio from filepath or Gradio tuple."""
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


# ══════════════════════════════════════════════════════════════════
# SEGMENTATION + FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════
def get_all_segments(signal: np.ndarray) -> list:
    """Returns all 50%-overlap 2-second segments from the signal."""
    segments = []
    for start in range(0, len(signal) - SEG + 1, STEP):
        segments.append(signal[start:start + SEG])
    if not segments:
        segments = [np.pad(signal, (0, SEG - len(signal)))]
    return segments


def segment_to_tensor(seg: np.ndarray) -> tuple:
    """Converts a segment to mel_db and normalized tensor."""
    mel    = librosa.feature.melspectrogram(
        y=seg, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    mel_db   = librosa.power_to_db(mel, ref=np.max)
    mel_norm = (mel_db - NORM_MEAN) / (NORM_STD + 1e-8)
    tensor   = torch.FloatTensor(mel_norm).unsqueeze(0).unsqueeze(0).to(DEVICE)
    return mel_db, tensor


# ══════════════════════════════════════════════════════════════════
# GRAD-CAM
# ══════════════════════════════════════════════════════════════════
def compute_gradcam(tensor: torch.Tensor) -> np.ndarray:
    """
    Grad-CAM on layer3. Returns heatmap (128, 16) normalized to [0,1].
    Targets the abnormality logit — red regions indicate where the
    model found evidence of cardiac abnormality.
    """
    tensor = tensor.clone().requires_grad_(True)
    activations, gradients = {}, {}

    def fwd_hook(m, i, o):
        activations["l3"] = o

    def bwd_hook(m, gi, go):
        gradients["l3"] = go[0]

    fh = model.layer3.register_forward_hook(fwd_hook)
    bh = model.layer3.register_full_backward_hook(bwd_hook)

    model.zero_grad()
    model(tensor).backward()

    fh.remove()
    bh.remove()

    acts  = activations["l3"].squeeze(0)
    grads = gradients["l3"].squeeze(0)
    weights = grads.mean(dim=(1, 2))

    cam = torch.zeros(acts.shape[1:], device=DEVICE)
    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = F.relu(cam).unsqueeze(0).unsqueeze(0)
    cam = F.interpolate(cam, size=(N_MELS, 16), mode="bilinear", align_corners=False)
    cam = cam.squeeze().detach().cpu().numpy()

    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    else:
        cam = np.zeros_like(cam)

    return cam


# ══════════════════════════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════════════════════════
def plot_waveform(segment: np.ndarray, title: str = "") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 2.8))
    t = np.arange(len(segment)) / SR
    ax.plot(t, segment, color="#2563EB", linewidth=0.9, alpha=0.9)
    ax.fill_between(t, segment, alpha=0.10, color="#2563EB")
    ax.axhline(0, color="#94A3B8", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_ylabel("Amplitude", fontsize=11)
    ax.set_title(title or "PCG Waveform — Highest RMS Segment", fontsize=12, fontweight="bold")
    ax.set_xlim(0, len(segment) / SR)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_mel(mel_db: np.ndarray, title: str = "") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 4))
    img = ax.imshow(
        mel_db, aspect="auto", origin="lower",
        cmap="magma", interpolation="nearest"
    )
    plt.colorbar(img, ax=ax, label="dB")
    ax.set_xlabel("Time Frame", fontsize=11)
    ax.set_ylabel("Mel Filter Bank", fontsize=11)
    ax.set_title(title or "Mel-Spectrogram (128 × 16) — Model Input", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_gradcam(mel_db: np.ndarray, cam: np.ndarray, prob: float) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].imshow(mel_db, aspect="auto", origin="lower", cmap="magma")
    axes[0].set_title("Mel-Spectrogram\n(Highest Abnormality Segment)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Time Frame")
    axes[0].set_ylabel("Mel Filter Bank")

    axes[1].imshow(mel_db, aspect="auto", origin="lower", cmap="magma", alpha=0.55)
    axes[1].imshow(cam,    aspect="auto", origin="lower", cmap="jet",   alpha=0.5,
                   vmin=0, vmax=1)
    axes[1].set_title(
        f"Grad-CAM Overlay  (prob_abn={prob:.3f})\nRed = most discriminative region",
        fontsize=11, fontweight="bold"
    )
    axes[1].set_xlabel("Time Frame")
    axes[1].set_ylabel("Mel Filter Bank")

    fig.suptitle("Temporal-Frequency Interpretability (Grad-CAM)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_se_attention(weights: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 3))
    top_k   = 20
    top_idx = set(np.argsort(weights)[-top_k:])
    colors  = ["#DC2626" if i in top_idx else "#93C5FD" for i in range(len(weights))]
    ax.bar(range(len(weights)), weights, color=colors, width=1.0, edgecolor="none")
    ax.set_xlabel("Channel Index (128 mel channels)", fontsize=11)
    ax.set_ylabel("Attention Weight", fontsize=11)
    ax.set_title(
        "SE Channel Attention — Layer 3\nRed: Top-20 most active frequency channels",
        fontsize=12, fontweight="bold"
    )
    ax.set_xlim(-1, len(weights))
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
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
        loc="center", cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1.2, 2.5)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1D4ED8")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#EFF6FF")
            cell.set_text_props(fontweight="bold")
    ax.set_title(
        "ResNet2D + SE Attention — Test Set Performance  (threshold=0.35)",
        fontsize=12, fontweight="bold", pad=14
    )
    fig.tight_layout()
    return fig


def plot_quality_fail(report: list) -> plt.Figure:
    """Visual quality report shown when input fails validation."""
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.axis("off")

    lines = []
    for passed, name, detail in report:
        icon = "✅" if passed else "❌"
        lines.append(f"{icon}  {name}: {detail}")

    text = "\n".join(lines)
    ax.text(
        0.05, 0.95, text,
        transform=ax.transAxes,
        fontsize=12, verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#FEF2F2", edgecolor="#DC2626", linewidth=2)
    )
    ax.set_title("Input Quality Validation Report", fontsize=13, fontweight="bold", color="#DC2626")
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════
# PREDICT
# ══════════════════════════════════════════════════════════════════
def predict(audio_input):
    """
    Full inference pipeline:
      1. Load signal
      2. Quality check (5 stages) — stop if failed
      3. Segment all overlapping windows
      4. Inference on every segment → average probability
      5. Grad-CAM on highest-prob_abn segment
      6. Visualization on highest-RMS segment

    Returns 10 outputs for Gradio components.
    """
    EMPTY = (None, None, None, None, None, None, {}, "N/A", "N/A", "")

    if audio_input is None:
        return EMPTY

    try:
        # 1. Load
        signal = load_signal(audio_input)

        # 2. Quality check
        is_valid, report = check_audio_quality(signal)
        quality_text = format_quality_report(report)

        if not is_valid:
            fail_fig = plot_quality_fail(report)
            return (
                fail_fig,   # quality report figure in Tab 1
                None, None, None, None, None,
                {"❌ Invalid Input — Check Quality Report": 1.0},
                "N/A", "N/A",
                quality_text
            )

        # 3. All segments
        segments = get_all_segments(signal)
        n_segs   = len(segments)

        # 4. Inference on all segments
        all_probs  = []
        all_mel_db = []
        all_rms    = []

        for seg in segments:
            mel_db, tensor = segment_to_tensor(seg)
            with torch.no_grad():
                logit = model(tensor).squeeze().item()
            prob = float(torch.sigmoid(torch.tensor(logit)))
            all_probs.append(prob)
            all_mel_db.append(mel_db)
            all_rms.append(float(np.sqrt(np.mean(seg ** 2))))

        # Final probability = mean of all segments
        final_prob = float(np.mean(all_probs))
        abnormal   = final_prob >= THRESHOLD

        # 5. Grad-CAM — on highest prob_abn segment
        best_prob_idx    = int(np.argmax(all_probs))
        _, gradcam_tensor = segment_to_tensor(segments[best_prob_idx])
        cam = compute_gradcam(gradcam_tensor)

        # 6. Visualization — on highest RMS segment
        best_rms_idx = int(np.argmax(all_rms))
        viz_segment  = segments[best_rms_idx]
        viz_mel_db   = all_mel_db[best_rms_idx]

        # Plots
        fig_wave    = plot_waveform(viz_segment,
                                    f"PCG Waveform — Best RMS Segment ({best_rms_idx+1}/{n_segs})")
        fig_mel     = plot_mel(viz_mel_db,
                               f"Mel-Spectrogram — Best RMS Segment ({best_rms_idx+1}/{n_segs})")
        fig_gradcam = plot_gradcam(all_mel_db[best_prob_idx], cam, all_probs[best_prob_idx])
        fig_attn    = plot_se_attention(_se_weights.get("layer3", np.zeros(128)))
        fig_met     = plot_metrics()

        # Result label
        label_dict = {
            "🔴 ABNORMAL": round(final_prob, 4),
            "🟢 NORMAL":   round(1.0 - final_prob, 4),
        }

        conf_pct  = final_prob * 100 if abnormal else (1.0 - final_prob) * 100
        conf_str  = f"{conf_pct:.1f}%"
        score_str = (
            f"mean prob: {final_prob:.4f}  |  "
            f"segments: {n_segs}  |  "
            f"threshold: {THRESHOLD}"
        )

        return (
            None,          # quality fail fig (None = no error)
            fig_wave,
            fig_mel,
            fig_gradcam,
            fig_attn,
            fig_met,
            label_dict,
            conf_str,
            score_str,
            quality_text
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return (
            None, None, None, None, None, None,
            {"❌ Error": 1.0}, "N/A", str(e)[:120], ""
        )


# ══════════════════════════════════════════════════════════════════
# POPUP HTML + JS
# ══════════════════════════════════════════════════════════════════
POPUP_HTML = """
<div id="pcg-popup-overlay" style="
    display:flex; position:fixed; inset:0; z-index:9999;
    background:rgba(0,0,0,0.55); align-items:center; justify-content:center;">
  <div style="
      background:#fff; border-radius:14px; padding:32px 36px;
      max-width:520px; width:90%; box-shadow:0 8px 40px rgba(0,0,0,0.25);
      font-family:'IBM Plex Sans',sans-serif;">
    <h2 style="margin:0 0 16px; color:#1D4ED8; font-size:1.25rem;">
      🫀 PCG Recording Requirements
    </h2>
    <table style="width:100%; border-collapse:collapse; font-size:0.95rem;">
      <tr><td style="padding:6px 8px;">✅</td><td>Digital stethoscope or electronic auscultation device</td></tr>
      <tr><td style="padding:6px 8px;">✅</td><td>Minimum <strong>2 seconds</strong> duration</td></tr>
      <tr><td style="padding:6px 8px;">✅</td><td>Any sample rate — auto-resampled to 2,000 Hz</td></tr>
      <tr><td style="padding:6px 8px;">✅</td><td><strong>.wav</strong> format recommended</td></tr>
      <tr><td style="padding:6px 8px; color:#DC2626;">❌</td><td>Ambient noise, music, or voice recordings</td></tr>
      <tr><td style="padding:6px 8px; color:#DC2626;">❌</td><td>Silent, clipped, or corrupted files</td></tr>
      <tr><td style="padding:6px 8px; color:#DC2626;">❌</td><td>Recordings shorter than 2 seconds</td></tr>
    </table>
    <p style="margin:16px 0 0; font-size:0.85rem; color:#64748B;">
      The system performs 5-stage quality validation before analysis.
      Invalid recordings will be rejected with a detailed report.
    </p>
    <button onclick="document.getElementById('pcg-popup-overlay').style.display='none'"
      style="
        margin-top:20px; width:100%; padding:12px;
        background:#1D4ED8; color:#fff; border:none;
        border-radius:8px; font-size:1rem; font-weight:600;
        cursor:pointer; letter-spacing:0.02em;">
      I Understand — Proceed to Upload
    </button>
  </div>
</div>
"""


# ══════════════════════════════════════════════════════════════════
# GRADIO UI
# ══════════════════════════════════════════════════════════════════
with gr.Blocks(
    theme=gr.themes.Default(
        primary_hue="blue",
        secondary_hue="slate",
        font=gr.themes.GoogleFont("IBM Plex Sans"),
        font_mono=gr.themes.GoogleFont("IBM Plex Mono"),
    ),
    title="PCG Heart Sound Classifier",
    css="""
        .gradio-container { max-width: 1200px !important; }
        .tab-nav button { font-weight: 600; font-size: 0.95rem; }
        footer { display: none !important; }
    """
) as demo:

    # Popup shown on load
    gr.HTML(POPUP_HTML)

    # Header
    gr.Markdown("""
    # 🫀 Abnormal Heart Sound Detection
    **ResNet2D + SE Attention + Mel-Spectrogram** &nbsp;|&nbsp;
    PhysioNet/CinC 2016 · 3,240 recordings · SR 2,000 Hz &nbsp;|&nbsp;
    Bachelor Thesis — Г.Хишигжаргал, 2026
    """)

    # Input row
    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                label="PCG Recording (.wav)",
                type="filepath",
                sources=["upload", "microphone"]
            )
            analyze_btn = gr.Button("Analyze Recording", variant="primary", size="lg")

        with gr.Column(scale=1):
            result_label = gr.Label(label="Classification", num_top_classes=2)
            with gr.Row():
                confidence_out = gr.Textbox(label="Confidence",  interactive=False, scale=1)
                score_out      = gr.Textbox(label="Aggregation", interactive=False, scale=2)

    # Tabs
    with gr.Tabs():

        with gr.Tab("📋 Diagnosis"):
            quality_fail_plot = gr.Plot(label="Quality Report", visible=True)
            with gr.Row():
                fig_wave_out = gr.Plot(label="Waveform — Highest RMS Segment")
                fig_mel_out  = gr.Plot(label="Mel-Spectrogram — Model Input (128×16)")
            quality_text_out = gr.Markdown(label="Quality Details")

        with gr.Tab("🔬 Explainability"):
            gr.Markdown("""
            **Grad-CAM** shows *where in time and frequency* the model detected abnormality.
            Computed on the segment with the highest abnormality probability.

            **SE Attention** shows *which mel frequency channels* Layer 3 weighted most heavily.
            These correspond to specific cardiac sound bands (S1, S2, murmurs).
            """)
            fig_gradcam_out = gr.Plot(label="Grad-CAM — Temporal-Frequency Heatmap")
            fig_attn_out    = gr.Plot(label="SE Channel Attention Weights — Layer 3")

        with gr.Tab("📊 Model Info"):
            gr.Markdown("""
            ### Architecture
            **ResNet2D + Squeeze-and-Excitation Attention**
            - Input: Mel-spectrogram (1 × 128 × 16)
            - Stem: Conv2D(1→32) + BN + ReLU
            - Layer 1: BasicBlock(32→32) + SEBlock
            - Layer 2: BasicBlock(32→64, stride=2) + SEBlock
            - Layer 3: BasicBlock(64→128, stride=2) + SEBlock
            - Output: AdaptiveAvgPool → Dropout → FC(128→1)

            ### Training
            - Dataset: PhysioNet/CinC 2016, 3,240 recordings, 68,104 segments
            - Split: 70% train / 15% val / 15% test (stratified)
            - Loss: BCEWithLogitsLoss (pos_weight=3.25)
            - Optimizer: Adam (lr=1e-3, weight_decay=1e-4)
            - Threshold: **0.35** (optimized for recall on validation set)

            ### Inference
            - All overlapping 2-second segments are classified
            - Final probability = mean across all segments
            - Grad-CAM computed on highest-probability segment
            """)
            fig_metrics_out = gr.Plot(label="Test Set Performance")

    # Footer
    gr.Markdown("""
    <div style='text-align:center; color:#94A3B8; font-size:0.8rem; margin-top:8px;'>
    Department of Computer Science · MUST · 2026 &nbsp;·&nbsp;
    Accuracy 95.24% &nbsp;·&nbsp; F1 90.00% &nbsp;·&nbsp;
    AUC 98.92% &nbsp;·&nbsp; Recall 91.05%
    </div>
    """)

    # Event
    analyze_btn.click(
        fn=predict,
        inputs=[audio_input],
        outputs=[
            quality_fail_plot,
            fig_wave_out,
            fig_mel_out,
            fig_gradcam_out,
            fig_attn_out,
            fig_metrics_out,
            result_label,
            confidence_out,
            score_out,
            quality_text_out,
        ]
    )


# ══════════════════════════════════════════════════════════════════
# LAUNCH
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    demo.launch(share=True, server_port=7860, show_error=True)