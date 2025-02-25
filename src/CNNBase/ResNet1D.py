import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from datesets.Dataset import prepare_data


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet1D(nn.Module):
    def __init__(self, input_channels: int, num_classes: int, patch_size=3):
        super(ResNet1D, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.in_channels = 64
        self.dim = 1

        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels: int, num_blocks: int, stride: int = 1) -> nn.Sequential:
        layers = [ResidualBlock1D(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        # x shape: (batch_size, input_channels)
        x = x.unsqueeze(1)  # 添加通道维度: (batch_size, 1, input_channels)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def __str__(self) -> str:
        return f"ResNet1D(input_channels={self.input_channels}, num_classes={self.num_classes})"


if __name__ == '__main__':
    # 假设我们有一个高光谱图像数据集
    bands, rows, cols = 200, 100, 100
    num_classes = 16

    # 模拟高光谱数据和标签
    data = np.random.rand(bands, rows, cols)
    labels = np.random.randint(0, num_classes + 1, size=(rows, cols))  # 0 表示背景

    # 准备数据
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(data, labels, dim=1)

    # 创建模型实例
    model = ResNet1D(input_channels=bands, num_classes=num_classes)

    # 测试模型
    batch_size = 32
    input_data = torch.randn(batch_size, bands)
    output = model(input_data)
    print(f"Input shape: {input_data.shape}")
    print(f"Output shape: {output.shape}")
