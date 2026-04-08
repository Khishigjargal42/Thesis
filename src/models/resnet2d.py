# src/models/resnet2d.py
import torch
import torch.nn as nn


class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # shortcut
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += identity
        return self.relu(out)


class ResNet2D(nn.Module):
    """
    Improved ResNet2D
    ✔ Stable
    ✔ Less overfitting
    ✔ Works with any input size
    """

    def __init__(self, num_classes=1):
        super().__init__()

        # ===== STEM =====
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # ===== RESIDUAL LAYERS =====
        self.layer1 = ResidualBlock2D(32,  64,  stride=2)
        self.layer2 = ResidualBlock2D(64,  128, stride=2)
        self.layer3 = ResidualBlock2D(128, 128, stride=2)  # ← reduced

        # ===== HEAD =====
        self.gap     = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)  # ↑ stronger regularization
        self.fc      = nn.Linear(128, num_classes)

        # weight init
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.gap(x)
        x = x.view(x.size(0), -1)

        x = self.dropout(x)
        return self.fc(x)


# ===== TEST =====
if __name__ == '__main__':
    model = ResNet2D()

    shapes = {
        'mel_spectrogram': (4, 1, 128, 16),
        'mfcc':            (4, 1, 40, 8),
        'spec':            (4, 1, 128, 16),
        'logmel':          (4, 1, 128, 16),
    }

    for name, shape in shapes.items():
        x = torch.randn(*shape)
        out = model(x)
        print(f"{name:20s}: input {shape} → output {tuple(out.shape)}")