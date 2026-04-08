# src/models/resnet2d.py
import torch
import torch.nn as nn

class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


class ResNet2D(nn.Module):
    """
    Input shape: (B, 1, 128, 16)
    128 = frequency bins, 16 = time frames
    """
    def __init__(self, num_classes=1):
        super().__init__()
        # Stem: frequency-д stride=2, time-д stride=1 хэрэглэнэ
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=(2, 1), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        )
        # (B, 32, 32, 16)

        self.layer1 = ResidualBlock2D(32,  64,  stride=(2, 1))
        # (B, 64, 16, 16)

        self.layer2 = ResidualBlock2D(64,  128, stride=(2, 2))
        # (B, 128, 8, 8)

        self.layer3 = ResidualBlock2D(128, 256, stride=(2, 2))
        # (B, 256, 4, 4)

        self.gap     = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.4)
        self.fc      = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)


# Shape шалгах тест
if __name__ == '__main__':
    model = ResNet2D()
    x = torch.randn(4, 1, 128, 16)
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")   # (4, 1) байх ёстой
    print("OK")