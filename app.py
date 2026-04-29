"""
PCG Heart Sound Classifier - Gradio Demo
ResNet2D + SE Attention + Mel-Spectrogram

Raw wav файлаас шууд inference хийнэ.
Сургалтанд ашиглаагүй гаднаас ирсэн аудио файл дэмжинэ.

Зохиогч: Г.Хишигжаргал
Бакалаврын судалгааны ажил, 2026
"""

import os
import numpy as np
import torch
import torch.nn as nn
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

# Preprocessing — raw wav pipeline
SR         = 2000
SEG        = 4000
STEP       = 2000
N_FFT      = 512
HOP_LENGTH = 256
N_MELS     = 128

# Train set normalization stats
NORM_MEAN  = -48.6446
NORM_STD   = 16.0562

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

# SE Attention hook
_se_weights = {}

def _se_hook(module, input, output):
    b, c, _, _ = input[0].size()
    y = module.pool(input[0]).view(b, c)
    w = module.fc(y).squeeze().detach().cpu().numpy()
    _se_weights["layer3"] = w

model.layer3.se.register_forward_hook(_se_hook)


# ══════════════════════════════════════════════════════════════════
# PREPROCESSING
# ══════════════════════════════════════════════════════════════════
def preprocess(audio_input):
    """
    Raw wav файлаас mel-spectrogram гаргана.

    Pipeline:
      1. SR=2000 руу resample
      2. Step=2000 overlap segments үүсгэх
      3. Хамгийн өндөр RMS сегментийг авах
      4. Mel-spectrogram (128x16)
      5. Train stats-аар normalize

    Returns:
        segment  : np.ndarray (4000,)
        mel_db   : np.ndarray (128, 16)
        tensor   : torch.Tensor (1, 1, 128, 16)
    """
    # 1. Аудио унших
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

    # 2. Step=2000 overlap segments
    segments = []
    for start in range(0, len(signal) - SEG + 1, STEP):
        segments.append(signal[start:start + SEG])

    # Богино файл бол pad
    if not segments:
        segments = [np.pad(signal, (0, SEG - len(signal)))]

    # 3. Хамгийн өндөр RMS сегмент
    rms_vals = [np.sqrt(np.mean(s ** 2)) for s in segments]
    segment  = segments[int(np.argmax(rms_vals))]

    # 4. Mel-spectrogram
    mel    = librosa.feature.melspectrogram(
        y=segment, sr=SR,
        n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # 5. Normalize
    mel_norm = (mel_db - NORM_MEAN) / (NORM_STD + 1e-8)
    tensor   = torch.FloatTensor(mel_norm).unsqueeze(0).unsqueeze(0).to(DEVICE)

    return segment, mel_db, tensor


# ══════════════════════════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════════════════════════
def plot_waveform(segment: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 2.5))
    t = np.arange(len(segment)) / SR
    ax.plot(t, segment, color="#1565C0", linewidth=0.8, alpha=0.9)
    ax.fill_between(t, segment, alpha=0.12, color="#1565C0")
    ax.axhline(0, color="gray", linewidth=0.4, linestyle="--")
    ax.set_xlabel("Хугацаа (секунд)", fontsize=11)
    ax.set_ylabel("Далайц", fontsize=11)
    ax.set_title("PCG Дохионы Хэлбэр — Хамгийн Өндөр RMS Сегмент", fontsize=12, fontweight="bold")
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
    plt.colorbar(img, ax=ax, label="дБ (dB)")
    ax.set_xlabel("Хугацааны Frame", fontsize=11)
    ax.set_ylabel("Mel Filter Bank", fontsize=11)
    ax.set_title("Mel-Spectrogram (128 × 16) — Model Оролт", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_se_attention(weights: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 3))
    top_k   = 20
    top_idx = set(np.argsort(weights)[-top_k:])
    colors  = ["#E53935" if i in top_idx else "#90CAF9"
               for i in range(len(weights))]
    ax.bar(range(len(weights)), weights, color=colors, width=1.0, edgecolor="none")
    ax.set_xlabel("Channel Индекс (128 суваг)", fontsize=11)
    ax.set_ylabel("SE Attention Жин", fontsize=11)
    ax.set_title(
        "SE Attention Weights — Layer 3\n"
        "Улаан: хамгийн идэвхтэй Top-20 давтамжийн суваг",
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
        "ResNet2D + SE Attention + Mel-Spectrogram — Test Set Гүйцэтгэл",
        fontsize=12, fontweight="bold", pad=14
    )
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════
# PREDICT
# ══════════════════════════════════════════════════════════════════
def predict(audio_input):
    EMPTY = (None, None, None, None, {}, "N/A", "N/A")

    if audio_input is None:
        return EMPTY

    try:
        # 1. Raw signal уншиж quality check хийх
        if isinstance(audio_input, tuple):
            native_sr, raw_signal = audio_input
            raw_signal = raw_signal.astype(np.float32)
            if raw_signal.ndim == 2:
                raw_signal = raw_signal.mean(axis=1)
            if np.abs(raw_signal).max() > 1.0:
                raw_signal = raw_signal / 32768.0
            if native_sr != SR:
                raw_signal = librosa.resample(raw_signal, orig_sr=native_sr, target_sr=SR)
        else:
            raw_signal, _ = librosa.load(audio_input, sr=SR, mono=True)

        is_valid, quality_msg = check_audio_quality(raw_signal)
        if not is_valid:
            return None, None, None, None, {"⚠️ Чанарын асуудал": 1.0}, "N/A", quality_msg

        # 2. Preprocessing
        segment, mel_db, tensor = preprocess(audio_input)

        # Inference
        with torch.no_grad():
            logit = model(tensor).squeeze().item()

        prob_abn = float(torch.sigmoid(torch.tensor(logit)))
        prob_nor = 1.0 - prob_abn
        abnormal = prob_abn >= 0.35

        # Plots
        fig_wave = plot_waveform(segment)
        fig_mel  = plot_mel(mel_db)
        fig_attn = plot_se_attention(_se_weights.get("layer3", np.zeros(128)))
        fig_met  = plot_metrics()

        # Result
        label_dict = {
            "🔴 ХЭВИЙН БУС (Abnormal)": round(prob_abn, 4),
            "🟢 ХЭВИЙН (Normal)":        round(prob_nor, 4),
        }

        conf_pct  = prob_abn * 100 if abnormal else prob_nor * 100
        conf_str  = f"{conf_pct:.1f}%"
        score_str = f"logit: {logit:.4f}  sigmoid: {prob_abn:.4f}"

        return fig_wave, fig_mel, fig_attn, fig_met, label_dict, conf_str, score_str

    except Exception as e:
        print(f"[ERROR] {e}")
        return None, None, None, None, {"❌ Алдаа": 1.0}, "N/A", str(e)[:100]
# ══════════════════════════════════════════════════════════════════
# AUDIO QUALITY CHECK
# ══════════════════════════════════════════════════════════════════
def check_audio_quality(signal: np.ndarray) -> tuple:
    """
    Raw signal дээр аудио чанар шалгана — preprocessing-аас өмнө.
    Returns: (is_valid, warning_message)
    """
    # 1. Хамгийн доод урт шалгах
    if len(signal) < SR * 1:
        return False, "⚠️ Бичлэг хэт богино байна. Дор хаяж 2 секунд байх хэрэгтэй."

    # 2. RMS энерги шалгах — бүтэн signal дээр
    rms = np.sqrt(np.mean(signal ** 2))
    if rms < 0.0005:
        return False, "⚠️ Дохио хэт чимээгүй байна. Стетоскоп зөв байрлуулсан эсэхийг шалгана уу."

    # 3. Clipping шалгах
    clip_ratio = (np.abs(signal) > 0.99).mean()
    if clip_ratio > 0.005:
        return False, "⚠️ Аудио clipping илэрлээ. Бичлэгийн түвшинг бууруулна уу."

    # 4. Дохионы хэлбэлзэл шалгах — хэт тогтмол бол дуугүй файл
    std = np.std(signal)
    if std < 0.0005:
        return False, "⚠️ Дохио хэт тогтмол байна. Зүрхний бичлэг биш байж болзошгүй."

    # 5. Зүрхний давтамжийн агуулга шалгах (20-200 Hz)
    analysis_seg = signal[:min(len(signal), SR * 5)]
    fft          = np.abs(np.fft.rfft(analysis_seg))
    freqs        = np.fft.rfftfreq(len(analysis_seg), 1 / SR)
    heart_energy = fft[(freqs >= 20) & (freqs <= 200)].sum()
    total_energy = fft.sum() + 1e-8
    heart_ratio  = heart_energy / total_energy

    if heart_ratio < 0.05:
        return False, "⚠️ Зүрхний давтамжийн дуу илэрсэнгүй. Стетоскопын PCG бичлэг оруулна уу."

    return True, "✅ Аудио чанар хэвийн."

# ══════════════════════════════════════════════════════════════════
# GRADIO UI
# ══════════════════════════════════════════════════════════════════
with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="indigo",
        font=gr.themes.GoogleFont("IBM Plex Sans")
    ),
    title="Зүрхний Авиа Шинжилгээ"
) as demo:

    gr.Markdown("""
    # 🫀 Зүрхний Авианы Хэвийн Бус Байдлыг Илрүүлэх Систем
    **ResNet2D + SE Attention + Mel-Spectrogram** архитектур

    > PhysioNet/CinC 2016 PCG өгөгдлийн сан — 3,240 бичлэг | SR: 2000 Hz
    > Бакалаврын судалгааны ажил — Г.Хишигжаргал, 2026
    ---
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📂 PCG Бичлэг Оруулах")
            audio_input = gr.Audio(
                label="Аудио файл оруулна уу (.wav)",
                type="filepath",
                sources=["upload", "microphone"]
            )
            predict_btn = gr.Button("🔍 Шинжлэх", variant="primary", size="lg")
            gr.Markdown("""
            **Зөвлөмж:**
            - `.wav` формат, дор хаяж 2 секунд
            - SR: 2000 Hz (автоматаар resample хийгдэнэ)
            - PhysioNet эсвэл өөр эх сурвалжийн бичлэг
            - Microphone-оор шууд бичих боломжтой
            """)

        with gr.Column(scale=1):
            gr.Markdown("### 📊 Оношлогооны Үр Дүн")
            result_label = gr.Label(
                label="Ангиллын Үр Дүн",
                num_top_classes=2
            )
            with gr.Row():
                confidence_out = gr.Textbox(
                    label="Итгэлцэл",
                    interactive=False,
                    scale=1
                )
                score_out = gr.Textbox(
                    label="Raw Score",
                    interactive=False,
                    scale=2
                )

    gr.Markdown("---\n### 📈 Дохионы Дүрслэл")
    with gr.Row():
        fig_wave_out = gr.Plot(label="Waveform — Хамгийн Өндөр RMS Сегмент")
        fig_mel_out  = gr.Plot(label="Mel-Spectrogram — Model Оролт (128×16)")

    gr.Markdown(
        "---\n### 🧠 SE Attention Visualization\n"
        "Загвар ямар давтамжийн сувгуудад анхаарлаа хандуулж байгааг харуулна. "
        "**Улаан** = Top-20 идэвхтэй суваг."
    )
    fig_attn_out = gr.Plot(label="SE Channel Attention Weights — Layer 3")

    gr.Markdown("---\n### 📋 Загварын Гүйцэтгэл (Test Set)")
    fig_metrics_out = gr.Plot(label="Evaluation Metrics")

    predict_btn.click(
        fn=predict,
        inputs=[audio_input],
        outputs=[
            fig_wave_out, fig_mel_out, fig_attn_out, fig_metrics_out,
            result_label, confidence_out, score_out
        ]
    )

    gr.Markdown("""
    ---
    <div style='text-align:center; color:#888; font-size:12px;'>
    Компьютрын Ухааны Тэнхим | МКУТ | 2026
    &nbsp;|&nbsp; Accuracy: 95.24%
    &nbsp;|&nbsp; F1: 90.00%
    &nbsp;|&nbsp; AUC: 98.92%
    &nbsp;|&nbsp; Recall: 91.05%
    </div>
    """)


# ══════════════════════════════════════════════════════════════════
# LAUNCH
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    demo.launch(share=True, server_port=7860, show_error=True)