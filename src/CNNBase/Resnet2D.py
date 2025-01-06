from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        """
        初始化一个二维残差块。

        Args:
            in_channels (int): 输入通道数。
            out_channels (int): 输出通道数。
            stride (int, optional): 卷积操作的步长。默认为1。
        """
        super(ResidualBlock2D, self).__init__()
        self.conv1: nn.Conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1: nn.BatchNorm2d = nn.BatchNorm2d(out_channels)
        self.conv2: nn.Conv2d = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2: nn.BatchNorm2d = nn.BatchNorm2d(out_channels)

        self.shortcut: nn.Sequential = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: Tensor) -> Tensor:
        """
        定义残差块的前向传播。

        Args:
            x (Tensor): 输入张量，形状为 (batch_size, in_channels, height, width)

        Returns:
            Tensor: 输出张量，形状为 (batch_size, out_channels, new_height, new_width)
        """
        out: Tensor = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet2D(nn.Module):
    def __init__(self, input_channels: int = 200, num_classes: int = 16):
        """
        初始化二维ResNet模型。

        Args:
            input_channels (int, optional): 输入数据的通道数。默认为200。
            num_classes (int, optional): 分类问题的类别数。默认为16。
        """
        super(ResNet2D, self).__init__()
        self.input_channels: int = input_channels
        self.num_classes: int = num_classes
        self.in_channels: int = 200

        # 初始卷积层
        self.conv1: nn.Conv2d = nn.Conv2d(input_channels, 200, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1: nn.BatchNorm2d = nn.BatchNorm2d(self.in_channels)
        self.relu: nn.ReLU = nn.ReLU(inplace=True)
        self.maxpool: nn.MaxPool2d = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet层
        self.layer1: nn.Sequential = self._make_layer(64, 2)
        self.layer2: nn.Sequential = self._make_layer(128, 2, stride=2)
        self.layer3: nn.Sequential = self._make_layer(256, 2, stride=2)
        self.layer4: nn.Sequential = self._make_layer(512, 2, stride=2)

        # 全局平均池化和全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels: int, num_blocks: int, stride: int = 1) -> nn.Sequential:
        """
        创建包含多个残差块的层。

        Args:
            out_channels (int): 该层的输出通道数。
            num_blocks (int): 该层中残差块的数量。
            stride (int, optional): 第一个残差块的步长。默认为1。

        Returns:
            nn.Sequential: 包含多个残差块的Sequential模块。
        """
        layers: List[ResidualBlock2D] = [ResidualBlock2D(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock2D(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        定义ResNet2D模型的前向传播。

        Args:
            x (Tensor): 输入张量，形状为 (batch_size, input_channels, height, width)

        Returns:
            Tensor: 输出张量，形状为 (batch_size, num_classes)
        """
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
        """
        返回模型的字符串表示。

        Returns:
            str: 描述模型结构的字符串。
        """
        return (f"ResNet2DHSI(input_channels={self.input_channels}, "
                f"num_classes={self.num_classes}, "
                f"input_spatial_size=145x145)")