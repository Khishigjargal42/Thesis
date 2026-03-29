import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(14,4))

layers = [
    "Input\n(1×128×16)",
    "Conv2D\n16 filters\n3×3",
    "ReLU",
    "MaxPool\n2×2",
    "Conv2D\n32 filters\n3×3",
    "ReLU",
    "MaxPool\n2×2",
    "Flatten",
    "FC\n128",
    "Softmax\n2 classes"
]

x = 0

for i, layer in enumerate(layers):

    box = FancyBboxPatch(
        (x,0),
        1.2,
        1,
        boxstyle="round,pad=0.02",
        edgecolor="black",
        facecolor="#e6f2ff"
    )

    ax.add_patch(box)

    ax.text(
        x+0.6,
        0.5,
        layer,
        ha='center',
        va='center',
        fontsize=10
    )

    if i < len(layers)-1:
        ax.arrow(
            x+1.2,
            0.5,
            0.3,
            0,
            head_width=0.05,
            head_length=0.1,
            fc='black'
        )

    x += 1.7

ax.set_xlim(0,x)
ax.set_ylim(0,1)

ax.axis('off')

plt.title("Convolutional Neural Network Architecture")

plt.savefig("figures/cnn_architecture_improved.png", dpi=300)

plt.show()