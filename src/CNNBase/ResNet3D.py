from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1):
        super(ResidualBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, dilation, dilation),
                               dilation=(1, dilation, dilation), bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, dilation, dilation),
                               dilation=(1, dilation, dilation), bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = F.relu(out)
        return out


class ResNet3D(nn.Module):
    def __init__(self, input_channels: int = 200, num_classes: int = 16):
        super(ResNet3D, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.in_channels = 64
        self.dim = 3

        self.conv1 = nn.Conv3d(1, self.in_channels, kernel_size=(7, 3, 3), padding=(3, 1, 1), bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(64, 2, dilation=1)
        self.layer2 = self._make_layer(128, 2, dilation=2)
        self.layer3 = self._make_layer(256, 2, dilation=4)
        self.layer4 = self._make_layer(512, 2, dilation=8)

        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels: int, num_blocks: int, dilation: int = 1) -> nn.Sequential:
        layers: List[ResidualBlock3D] = [ResidualBlock3D(self.in_channels, out_channels, dilation=dilation)]
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock3D(out_channels, out_channels, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.relu(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def __str__(self) -> str:
        return f"ResNet3DHSI(input_channels={self.input_channels}, num_classes={self.num_classes})"


if __name__ == '__main__':
    # 测试代码
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size, in_channels = 16, 200
    n_classes = 16
    model = ResNet3D(input_channels=in_channels, num_classes=n_classes)
    model.to(device)

    # 测试不同输入尺寸
    for size in [(3, 3), (5, 5), (7, 7), (9, 9)]:
        height, width = size
        input_data = torch.randn(batch_size, 1, in_channels, height, width)
        input_data = input_data.to(device)

        # 前向传播
        output = model(input_data)

        print(f"Input shape: {input_data.shape}")
        print(f"Output shape: {output.shape}")

    print(f"Model structure: {model}")
