"""
PCG Heart Sound Classifier - Gradio Demo
ResNet2D + SE Attention + Mel-Spectrogram
Зүрхний авианы хэвийн бус байдлыг илрүүлэх систем

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
# CONFIGURATION — training-тай яг адилхан байх ёстой
# ══════════════════════════════════════════════════════════════════
SR           = 2000
SEGMENT_LEN  = 4000       # 2 секунд x 2000 Hz
N_FFT        = 512
HOP_LENGTH   = 256
N_MELS       = 128
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training set normalization stats — thesis Table 2.3-аас
NORM_MEAN    = -60.84
NORM_STD     = 21.99

# ══════════════════════════════════════════════════════════════════
# MODEL PATHS — өөрийн Drive замаар солино уу
# ══════════════════════════════════════════════════════════════════
MODEL_PATHS = {
    "ResNet2D + Mel + SE Attention (Шилдэг)": "models/resnet_mel_attention.pth",
    "ResNet2D + Mel-Spectrogram":             "models/resnet2d_mel_spectrogram.pth",
    "ResNet2D + Mel (v2)":                    "models/resnet2d_mel.pth",
}

# Colab Drive дээр байвал замуудыг override хийнэ үү:
# DRIVE = "/content/drive/MyDrive/Thesis/models"
# MODEL_PATHS = {k: os.path.join(DRIVE, os.path.basename(v)) for k, v in MODEL_PATHS.items()}

# Model cache — дахин ачаалахгүй
_model_cache: dict = {}

# SE attention weights cache
_se_cache: dict = {}


# ══════════════════════════════════════════════════════════════════
# MODEL АРХИТЕКТУР
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
# MODEL АЧААЛАХ + CACHE
# ══════════════════════════════════════════════════════════════════
def _register_se_hook(model: ResNet2D_SE, cache_key: str) -> None:
    """Layer3-ын SEBlock-ын channel attention weights-ийг hook-оор авна."""
    def hook_fn(module, input, output):
        b, c, _, _ = input[0].size()
        y = module.pool(input[0]).view(b, c)
        w = module.fc(y).squeeze().detach().cpu().numpy()
        _se_cache[cache_key] = w

    model.layer3.se.register_forward_hook(hook_fn)


def load_model(model_name: str) -> ResNet2D_SE:
    """
    Model-ийг cache-ээс эсвэл дискнээс ачаална.
    Нэг model-ийг хоёр дахь удаа ачаалахгүй.
    """
    if model_name in _model_cache:
        return _model_cache[model_name]

    path = MODEL_PATHS.get(model_name)
    if not path:
        raise ValueError(f"Тохиргоонд байхгүй model: {model_name}")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model файл олдсонгүй: {path}\n"
            f"MODEL_PATHS дотрох замыг шалгана уу."
        )

    model = ResNet2D_SE()
    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    _register_se_hook(model, model_name)
    _model_cache[model_name] = model

    print(f"[OK] Model ачааллаа: {model_name} <- {path}")
    return model


# ══════════════════════════════════════════════════════════════════
# PREPROCESSING
# ══════════════════════════════════════════════════════════════════
def preprocess_audio(audio_path: str):
    """
    .wav файлыг уншиж mel-spectrogram гаргана.
    Training pipeline-тай яг адилхан параметр ашиглана.

    Returns:
        segment  : np.ndarray (4000,)         — нормчлогдоогүй waveform
        mel_db   : np.ndarray (128, 16)       — dB масштабын mel-spec
        tensor   : torch.Tensor (1,1,128,16)  — нормчлогдсон model input
    """
    # 1. Аудио унших + SR=2000 руу resample
    signal, _ = librosa.load(audio_path, sr=SR, mono=True)

    # 2. Эхний 4000 sample авах (2 сек), богино бол pad
    if len(signal) >= SEGMENT_LEN:
        segment = signal[:SEGMENT_LEN]
    else:
        segment = np.pad(signal, (0, SEGMENT_LEN - len(signal)))

    # 3. Mel-spectrogram — training-тай яг адилхан параметр
    mel = librosa.feature.melspectrogram(
        y=segment,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)  # shape: (128, 16)

    # 4. Global normalization — training stats ашиглана
    mel_norm = (mel_db - NORM_MEAN) / (NORM_STD + 1e-8)

    # 5. Tensor болгох
    tensor = torch.FloatTensor(mel_norm).unsqueeze(0).unsqueeze(0).to(DEVICE)

    return segment, mel_db, tensor


# ══════════════════════════════════════════════════════════════════
# VISUALIZATION ФУНКЦҮҮД
# ══════════════════════════════════════════════════════════════════
def plot_waveform(waveform: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 2.8))
    t = np.arange(len(waveform)) / SR
    ax.plot(t, waveform, color="#1565C0", linewidth=0.8, alpha=0.9)
    ax.fill_between(t, waveform, alpha=0.12, color="#1565C0")
    ax.axhline(0, color="gray", linewidth=0.4, linestyle="--")
    ax.set_xlabel("Хугацаа (секунд)", fontsize=11)
    ax.set_ylabel("Далайц", fontsize=11)
    ax.set_title("PCG Дохионы Хэлбэр (Waveform)", fontsize=13, fontweight="bold")
    ax.set_xlim(0, len(waveform) / SR)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def plot_mel_spectrogram(mel_db: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    img = ax.imshow(
        mel_db, aspect="auto", origin="lower",
        cmap="magma", interpolation="nearest"
    )
    plt.colorbar(img, ax=ax, label="дБ (dB)")
    ax.set_xlabel("Хугацааны Frame", fontsize=11)
    ax.set_ylabel("Mel Filter Bank Индекс", fontsize=11)
    ax.set_title("Mel-Spectrogram (128 × 16) — Model-ын Оролт", fontsize=13, fontweight="bold")
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
        "SE Attention Weights — Layer 3 (128 суваг)\n"
        "Улаан: хамгийн идэвхтэй Top-20 давтамжийн суваг",
        fontsize=12, fontweight="bold"
    )
    ax.set_xlim(-1, len(weights))
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def plot_metrics_card() -> plt.Figure:
    """Загварын test set гүйцэтгэлийн хүснэгт."""
    metrics = {
        "Accuracy":  0.9115,
        "Precision": 0.7438,
        "Recall":    0.9516,
        "F1-Score":  0.8350,
        "AUC-ROC":   0.9744,
    }
    fig, ax = plt.subplots(figsize=(9, 2.2))
    ax.axis("off")
    names  = list(metrics.keys())
    values = [f"{v:.4f}" for v in metrics.values()]
    raw    = list(metrics.values())

    table = ax.table(
        cellText=[values],
        colLabels=names,
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
            g = int(180 + raw[col] * 70)
            cell.set_facecolor(f"#{g:02x}e4ff"[:7])
            cell.set_text_props(fontweight="bold")

    ax.set_title(
        "ResNet2D + SE Attention + Mel-Spectrogram — Test Set Гүйцэтгэл",
        fontsize=12, fontweight="bold", pad=14
    )
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════
# ҮНДСЭН PREDICT ФУНКЦ
# ══════════════════════════════════════════════════════════════════
def predict(audio_path: str, model_name: str):
    """
    Inputs:
        audio_path  : gr.Audio filepath
        model_name  : dropdown сонголт

    Outputs (7):
        fig_wave, fig_mel, fig_attn, fig_metrics,
        label_dict (gr.Label), confidence_str, score_str
    """
    EMPTY = (None, None, None, None, {}, "N/A", "N/A")

    if audio_path is None:
        return EMPTY

    try:
        # 1. Model ачаалах (cache ашиглана)
        model = load_model(model_name)

        # 2. Preprocessing
        waveform, mel_db, tensor = preprocess_audio(audio_path)

        # 3. Inference
        with torch.no_grad():
            logit = model(tensor).squeeze().item()

        prob_abn = float(torch.sigmoid(torch.tensor(logit)))
        prob_nor = 1.0 - prob_abn
        abnormal = prob_abn >= 0.5

        # 4. Plots
        fig_wave    = plot_waveform(waveform)
        fig_mel     = plot_mel_spectrogram(mel_db)
        fig_metrics = plot_metrics_card()
        fig_attn    = (
            plot_se_attention(_se_cache[model_name])
            if model_name in _se_cache else None
        )

        # 5. gr.Label dict
        label_dict = {
            "🔴 ХЭВИЙН БУС (Abnormal)": round(prob_abn, 4),
            "🟢 ХЭВИЙН (Normal)":        round(prob_nor, 4),
        }

        # 6. Confidence + raw score
        conf_pct  = prob_abn * 100 if abnormal else prob_nor * 100
        conf_str  = f"{conf_pct:.1f}%"
        score_str = f"{logit:.4f}  →  sigmoid: {prob_abn:.4f}"

        return fig_wave, fig_mel, fig_attn, fig_metrics, label_dict, conf_str, score_str

    except FileNotFoundError as e:
        msg = str(e)
        print(f"[ERROR] {msg}")
        return None, None, None, None, {"❌ Model олдсонгүй": 1.0}, "N/A", msg[:80]

    except Exception as e:
        msg = str(e)
        print(f"[ERROR] {msg}")
        return None, None, None, None, {"❌ Алдаа": 1.0}, "N/A", msg[:80]


# ══════════════════════════════════════════════════════════════════
# GRADIO INTERFACE
# ══════════════════════════════════════════════════════════════════
with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="indigo",
        font=gr.themes.GoogleFont("IBM Plex Sans")
    ),
    title="Зүрхний Авиа Шинжилгээ"
) as demo:

    # ── Header ────────────────────────────────────────────────────
    gr.Markdown("""
    # 🫀 Зүрхний Авианы Хэвийн Бус Байдлыг Илрүүлэх Систем
    **ResNet2D + SE Attention + Mel-Spectrogram** архитектур дээр суурилсан

    > PhysioNet/CinC 2016 PCG өгөгдлийн сан — 3,240 бичлэг | SR: 2000 Hz
    > Бакалаврын судалгааны ажил — Г.Хишигжаргал, 2026
    ---
    """)

    # ── Input + Үр дүн ────────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📂 Оролт")
            audio_input = gr.Audio(
                label="PCG бичлэг оруулах (.wav)",
                type="filepath",
                sources=["upload", "microphone"]
            )
            model_dropdown = gr.Dropdown(
                choices=list(MODEL_PATHS.keys()),
                value=list(MODEL_PATHS.keys())[0],
                label="🧠 Model сонгох",
                info="Шилдэг: ResNet2D + Mel + SE Attention (F1: 0.835, AUC: 0.974)"
            )
            predict_btn = gr.Button("🔍 Шинжлэх", variant="primary", size="lg")
            gr.Markdown("""
            **Зөвлөмж:**
            - `.wav` формат, дор хаяж 2 секунд
            - PhysioNet датасетын файлуудыг туршиж болно
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
                    label="Итгэлцэл (Confidence)",
                    interactive=False,
                    scale=1
                )
                score_out = gr.Textbox(
                    label="Raw Logit → Sigmoid",
                    interactive=False,
                    scale=2
                )

    # ── Дохионы дүрслэл ───────────────────────────────────────────
    gr.Markdown("---\n### 📈 Дохионы Дүрслэл")
    with gr.Row():
        fig_waveform = gr.Plot(label="Waveform — Цуврал Дохио")
        fig_mel      = gr.Plot(label="Mel-Spectrogram — Model Оролт (128×16)")

    # ── SE Attention ──────────────────────────────────────────────
    gr.Markdown("""
    ---
    ### 🧠 SE Attention Visualization
    Загвар ямар Mel давтамжийн сувгуудад анхаарлаа хандуулж байгааг харуулна.
    **Улаан баганууд** = хамгийн идэвхтэй Top-20 channel.
    """)
    fig_attention = gr.Plot(label="SE Channel Attention Weights — Layer 3")

    # ── Metrics ───────────────────────────────────────────────────
    gr.Markdown("---\n### 📋 Загварын Гүйцэтгэл (Test Set)")
    fig_metrics = gr.Plot(label="Evaluation Metrics")

    # ── Event binding ─────────────────────────────────────────────
    predict_btn.click(
        fn=predict,
        inputs=[audio_input, model_dropdown],
        outputs=[
            fig_waveform, fig_mel, fig_attention, fig_metrics,
            result_label, confidence_out, score_out
        ]
    )

    # ── Footer ────────────────────────────────────────────────────
    gr.Markdown("""
    ---
    <div style='text-align:center; color:#888; font-size:12px;'>
    Компьютрын Ухааны Тэнхим | МКУТ | 2026 &nbsp;|&nbsp;
    Accuracy: 91.15% &nbsp;|&nbsp; F1: 83.50% &nbsp;|&nbsp; AUC: 97.44% &nbsp;|&nbsp; Recall: 95.16%
    </div>
    """)


# ══════════════════════════════════════════════════════════════════
# АЖИЛЛУУЛАХ
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"Device     : {DEVICE}")
    print(f"Sample Rate: {SR} Hz | N_Mels: {N_MELS} | Segment: {SEGMENT_LEN} samples")
    demo.launch(
        share=True,
        server_port=7860,
        show_error=True
    )