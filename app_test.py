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
DATA_DIR   = os.path.join(BASE_DIR, "data", "features")

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
# MODEL + DATA АЧААЛАХ
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

# Demo samples ачаалах
normal_mel   = np.load(os.path.join(DATA_DIR, "demo_normal_mel.npy"))
abnormal_mel = np.load(os.path.join(DATA_DIR, "demo_abnormal_mel.npy"))

print(f"Normal samples  : {len(normal_mel)}")
print(f"Abnormal samples: {len(abnormal_mel)}")

NORMAL_CHOICES   = [f"Normal Sample {i+1}"   for i in range(len(normal_mel))]
ABNORMAL_CHOICES = [f"Abnormal Sample {i+1}" for i in range(len(abnormal_mel))]
ALL_CHOICES      = NORMAL_CHOICES + ABNORMAL_CHOICES


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
    ax.set_title(title, fontsize=13, fontweight="bold")
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
# PREDICT
# ══════════════════════════════════════════════════════════════════
def predict(sample_name: str):
    if sample_name is None:
        return None, None, None, {}, "N/A", "N/A"

    if sample_name.startswith("Normal"):
        idx        = int(sample_name.split()[-1]) - 1
        mel_db     = normal_mel[idx]
        true_label = "Normal"
    else:
        idx        = int(sample_name.split()[-1]) - 1
        mel_db     = abnormal_mel[idx]
        true_label = "Abnormal"

    # Normalize + tensor
    mel_norm = (mel_db - NORM_MEAN) / (NORM_STD + 1e-8)
    tensor   = torch.FloatTensor(mel_norm).unsqueeze(0).unsqueeze(0).to(DEVICE)

    # Inference
    with torch.no_grad():
        logit = model(tensor).squeeze().item()

    prob_abn = float(torch.sigmoid(torch.tensor(logit)))
    prob_nor = 1.0 - prob_abn
    abnormal = prob_abn >= 0.5

    # Plots
    fig_mel  = plot_mel(mel_db, f"Mel-Spectrogram — {sample_name}  (Үнэн: {true_label})")
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

    > PhysioNet/CinC 2016 PCG өгөгдлийн сан | Бакалаврын судалгааны ажил — Г.Хишигжаргал, 2026
    ---
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🎵 Sample Сонгох")
            gr.Markdown(
                "Test set-ээс **15 Normal + 15 Abnormal** sample бэлдсэн.\n"
                "Dropdown-оос сонгоод шинжлэх товч дарна уу."
            )
            sample_dropdown = gr.Dropdown(
                choices=ALL_CHOICES,
                value=ALL_CHOICES[0],
                label="PCG Sample",
                info="Normal 1-15 эсвэл Abnormal 1-15"
            )
            predict_btn = gr.Button("🔍 Шинжлэх", variant="primary", size="lg")

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
        fn=predict,
        inputs=[sample_dropdown],
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