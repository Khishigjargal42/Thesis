"""
PCG Heart Sound Classifier - Gradio Demo
ResNet2D + SE Attention + Mel-Spectrogram

Зохиогч: Г.Хишигжаргал
Бакалаврын судалгааны ажил, 2026
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

# ══════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "resnet_mel_attention_v3.pth")
FEAT_DIR   = os.path.join(BASE_DIR, "data", "features")
RAW_DIR    = os.path.join(BASE_DIR, "data", "raw")

NORM_MEAN  = -60.8221
NORM_STD   = 21.9991
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
# LOAD MODEL + DATA
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

# Precomputed mel features
demo_mels   = np.load(os.path.join(FEAT_DIR, "demo_wav_mels.npy"))
demo_labels = np.load(os.path.join(FEAT_DIR, "demo_wav_labels.npy"))
demo_ids    = np.load(os.path.join(FEAT_DIR, "demo_wav_ids.npy"))

# record_id -> index map
id_to_idx = {rid: i for i, rid in enumerate(demo_ids)}

print(f"Demo mels loaded: {len(demo_mels)} files")
print(f"Normal: {(demo_labels==0).sum()}  Abnormal: {(demo_labels==1).sum()}")


# ══════════════════════════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════════════════════════
def plot_mel(mel_db: np.ndarray, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    img = ax.imshow(
        mel_db, aspect="auto", origin="lower",
        cmap="magma", interpolation="nearest",
        vmin=-80, vmax=0
    )
    plt.colorbar(img, ax=ax, label="дБ (dB)")
    ax.set_xlabel("Хугацааны Frame", fontsize=11)
    ax.set_ylabel("Mel Filter Bank", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
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
        "F1-Score":  0.8599,
        "AUC-ROC":   0.9804,
        "Precision": 0.8595,
        "Recall":    0.8604,
    }
    fig, ax = plt.subplots(figsize=(8, 2.2))
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
        "ResNet2D + SE Attention + Mel-Spectrogram — Validation Гүйцэтгэл",
        fontsize=12, fontweight="bold", pad=14
    )
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════════════════════════
def run_inference(mel_db: np.ndarray):
    mel_norm = (mel_db - NORM_MEAN) / (NORM_STD + 1e-8)
    tensor   = torch.FloatTensor(mel_norm).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logit = model(tensor).squeeze().item()
    prob_abn = float(torch.sigmoid(torch.tensor(logit)))
    return logit, prob_abn


# ══════════════════════════════════════════════════════════════════
# PREDICT — WAV UPLOAD
# ══════════════════════════════════════════════════════════════════
def predict_wav(audio_path: str):
    EMPTY = (None, None, None, {}, "N/A", "N/A")

    if audio_path is None:
        return EMPTY

    # Filename-ээс record_id олох (a0001.wav → a0001)
    fname     = os.path.basename(audio_path)
    record_id = os.path.splitext(fname)[0]

    if record_id not in id_to_idx:
        return (
            None, None, None,
            {"❌ Олдсонгүй": 1.0},
            "N/A",
            f"'{record_id}' датасетод байхгүй байна"
        )

    idx        = id_to_idx[record_id]
    mel_db     = demo_mels[idx]
    true_label = "Normal" if demo_labels[idx] == 0 else "Abnormal"

    logit, prob_abn = run_inference(mel_db)
    prob_nor = 1.0 - prob_abn
    abnormal = prob_abn >= 0.5

    fig_mel  = plot_mel(mel_db, f"Mel-Spectrogram — {record_id}  (Үнэн: {true_label})")
    fig_attn = plot_se_attention(_se_weights.get("layer3", np.zeros(128)))
    fig_met  = plot_metrics()

    label_dict = {
        "🔴 ХЭВИЙН БУС (Abnormal)": round(prob_abn, 4),
        "🟢 ХЭВИЙН (Normal)":        round(prob_nor, 4),
    }

    conf_pct  = prob_abn * 100 if abnormal else prob_nor * 100
    conf_str  = f"{conf_pct:.1f}%"
    score_str = f"logit: {logit:.4f}  sigmoid: {prob_abn:.4f}"

    return fig_mel, fig_attn, fig_met, label_dict, conf_str, score_str


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

    > PhysioNet/CinC 2016 PCG өгөгдлийн сан — 3,240 бичлэг
    > Бакалаврын судалгааны ажил — Г.Хишигжаргал, 2026
    ---
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📂 PCG Бичлэг Оруулах")
            audio_input = gr.Audio(
                label="Wav файл оруулна уу (PhysioNet формат)",
                type="filepath",
                sources=["upload"]
            )
            predict_btn = gr.Button("🔍 Шинжлэх", variant="primary", size="lg")
            gr.Markdown("""
            **Зөвлөмж:**
            - PhysioNet/CinC 2016 датасетын `.wav` файл
            - Жишээ: `a0001.wav`, `a0002.wav` гэх мэт
            - Файлын нэр record ID байх ёстой
            """)

        with gr.Column(scale=1):
            gr.Markdown("### 📊 Оношлогооны Үр Дүн")
            result_label = gr.Label(label="Ангиллын Үр Дүн", num_top_classes=2)
            with gr.Row():
                confidence_out = gr.Textbox(
                    label="Итгэлцэл", interactive=False, scale=1
                )
                score_out = gr.Textbox(
                    label="Raw Score", interactive=False, scale=2
                )

    gr.Markdown("---\n### 📈 Mel-Spectrogram — Model Оролт (128 × 16)")
    fig_mel_out = gr.Plot(label="Mel-Spectrogram")

    gr.Markdown(
        "---\n### 🧠 SE Attention Visualization\n"
        "Загвар ямар давтамжийн сувгуудад анхаарлаа хандуулж байгааг харуулна. "
        "**Улаан** = Top-20 идэвхтэй суваг."
    )
    fig_attn_out = gr.Plot(label="SE Channel Attention Weights")

    gr.Markdown("---\n### 📋 Загварын Гүйцэтгэл (Validation Set)")
    fig_metrics_out = gr.Plot(label="Evaluation Metrics")

    predict_btn.click(
        fn=predict_wav,
        inputs=[audio_input],
        outputs=[
            fig_mel_out, fig_attn_out, fig_metrics_out,
            result_label, confidence_out, score_out
        ]
    )

    gr.Markdown("""
    ---
    <div style='text-align:center; color:#888; font-size:12px;'>
    Компьютрын Ухааны Тэнхим | МКУТ | 2026
    &nbsp;|&nbsp; F1: 85.99% &nbsp;|&nbsp; AUC: 98.04%
    </div>
    """)


# ══════════════════════════════════════════════════════════════════
# LAUNCH
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    demo.launch(share=True, server_port=7860, show_error=True)