import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import init

from Train_and_Eval.device import get_device


class LeeEtAl3D(nn.Module):
    """
    LeeEtAl3D 模型实现。
    """

    def __init__(self, input_channels: int, num_classes: int, patch_size=3):
        """
        初始化 LeeEtAl3D 模型。

        Args:
            input_channels (int): 输入通道数。
            num_classes (int): 类别数量。
        """
        super(LeeEtAl3D, self).__init__()
        self.name: str = 'LeeEtAl'

        self.inception: nn.ModuleDict = self._create_inception_module(input_channels)
        self.residual_blocks: nn.ModuleList = self._create_residual_blocks()
        self.classifier: nn.Sequential = self._create_classifier(num_classes)
        self.dim: int = 3

        self.lrn1: nn.LocalResponseNorm = nn.LocalResponseNorm(256)
        self.lrn2: nn.LocalResponseNorm = nn.LocalResponseNorm(128)
        self.dropout: nn.Dropout = nn.Dropout(p=0.4)

        self.apply(self._weight_init)

    def forward(self, x: Tensor) -> Tensor:
        """
        模型的前向传播。

        Args:
            x (Tensor): 输入张量，形状为 (batch_size, 1, input_channels, height, width)。

        Returns:
            Tensor: 输出张量，形状为 (batch_size, num_classes)。
        """

        x_3x3 = self.inception['conv_3x3'](x)
        x_1x1 = self.inception['conv_1x1'](x)

        x = torch.cat([x_3x3, x_1x1], dim=1)
        x = x.squeeze(2)
        x = F.relu(self.lrn1(x))

        x = self.residual_blocks[0](x)
        x = F.relu(self.lrn2(x))
        for block in self.residual_blocks[1:]:
            x = F.relu(x + block(x))

        for i, layer in enumerate(self.classifier):
            x = layer(x)
            if i < len(self.classifier) - 1:
                x = F.relu(x)
                x = self.dropout(x)

        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)

        return x

    @staticmethod
    def _create_inception_module(input_channels: int) -> nn.ModuleDict:
        """
        创建 inception 模块。

        Args:
            input_channels (int): 输入通道数。

        Returns:
            nn.ModuleDict: 包含 inception 模块的字典。
        """
        return nn.ModuleDict({
            'conv_3x3': nn.Conv3d(1, 128, (input_channels, 3, 3), stride=(input_channels, 1, 1), padding=(0, 1, 1)),
            'conv_1x1': nn.Conv3d(1, 128, (input_channels, 1, 1), stride=(input_channels, 1, 1), padding=0)
        })

    def _create_residual_blocks(self) -> nn.ModuleList:
        """
        创建残差块。

        Returns:
            nn.ModuleList: 包含残差块的列表。
        """
        return nn.ModuleList([
            self._create_residual_block(256, 128),
            self._create_residual_block(128, 128)
        ])

    @staticmethod
    def _create_residual_block(input_channels: int, out_channels: int) -> nn.Sequential:
        """
        创建单个残差块。

        Args:
            input_channels (int): 输入通道数。
            out_channels (int): 输出通道数。

        Returns:
            nn.Sequential: 残差块。
        """
        return nn.Sequential(
            nn.Conv2d(input_channels, out_channels, (1, 1)),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, (1, 1))
        )

    @staticmethod
    def _create_classifier(num_classes: int) -> nn.Sequential:
        """
        创建分类器。

        Args:
            num_classes (int): 类别数量。

        Returns:
            nn.Sequential: 分类器。
        """
        return nn.Sequential(
            nn.Conv2d(128, 128, (1, 1)),
            nn.ReLU(),
            nn.Conv2d(128, 128, (1, 1)),
            nn.ReLU(),
            nn.Conv2d(128, num_classes, (1, 1))
        )

    @staticmethod
    def _weight_init(m: nn.Module) -> None:
        """
        初始化模型权重。

        Args:
            m (nn.Module): 要初始化的模块。
        """
        if isinstance(m, (nn.Linear, nn.Conv3d, nn.Conv2d)):
            init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)

    def __str__(self) -> str:
        """
        返回模型的字符串表示。

        Returns:
            str: 模型的字符串表示。
        """
        return (f"LeeEtAl(in_channels={self.inception['conv_3x3'].kernel_size[0]}, "
                f"n_classes={self.classifier[-1].out_channels})")


if __name__ == '__main__':
    batch_size, in_channels = 16, 200
    input_sizes = [(3, 3), (5, 5), (7, 7)]
    n_classes = 16
    device = get_device()
    model = LeeEtAl3D(in_channels, n_classes).to(device)

    for size in input_sizes:
        input_data = torch.randn(batch_size, 1, in_channels, *size).to(device)
        output = model(input_data)
        print(f"Input shape: {input_data.shape}")
        print(f"Output shape: {output.shape}")

    print(f"Model structure: {model}")
