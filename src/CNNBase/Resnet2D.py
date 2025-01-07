from typing import List

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1):
        super(ResidualBlock2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet2D(nn.Module):
    def __init__(self, input_channels: int = 200, num_classes: int = 16):
        super(ResNet2D, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.in_channels = 64

        self.conv1 = nn.Conv2d(input_channels, self.in_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(64, 2, dilation=1)
        self.layer2 = self._make_layer(128, 2, dilation=2)
        self.layer3 = self._make_layer(256, 2, dilation=4)
        self.layer4 = self._make_layer(512, 2, dilation=8)

        self.final_conv = nn.Conv2d(512, num_classes, kernel_size=1)

    def _make_layer(self, out_channels: int, num_blocks: int, dilation: int = 1) -> nn.Sequential:
        layers: List[ResidualBlock2D] = [ResidualBlock2D(self.in_channels, out_channels, dilation=dilation)]
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock2D(out_channels, out_channels, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.relu(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.final_conv(x)

        return x

    def __str__(self) -> str:
        return f"ResNet2DHSI(input_channels={self.input_channels}, num_classes={self.num_classes})"


# # 测试代码
# batch_size, in_channels, height, width = 16, 200, 3, 3
# input_data = torch.randn(batch_size, in_channels, height, width)
#
# # 创建模型实例
# n_classes = 16
# model = ResNet2D(input_channels=in_channels, num_classes=n_classes)
#
# # 前向传播
# output = model(input_data)
#
# print(f"Input shape: {input_data.shape}")
# print(f"Output shape: {output.shape}")
# print(f"Model structure: {model}")
