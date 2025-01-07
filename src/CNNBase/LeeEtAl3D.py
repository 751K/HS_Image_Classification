import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import init


class LeeEtAl3D(nn.Module):
    def __init__(self, in_channels: int, n_classes: int):
        super(LeeEtAl3D, self).__init__()
        self.name: str = 'LeeEtAl'

        self.inception: nn.ModuleDict = self._create_inception_module(in_channels)
        self.residual_blocks: nn.ModuleList = self._create_residual_blocks()
        self.classifier: nn.Sequential = self._create_classifier(n_classes)
        self.dim: int = 3

        self.lrn1: nn.LocalResponseNorm = nn.LocalResponseNorm(256)
        self.lrn2: nn.LocalResponseNorm = nn.LocalResponseNorm(128)
        self.dropout: nn.Dropout = nn.Dropout(p=0.4)

        self.apply(self._weight_init)

    def forward(self, x: Tensor) -> Tensor:
        # Inception forward
        x_3x3 = self.inception['conv_3x3'](x)
        x_1x1 = self.inception['conv_1x1'](x)
        x = torch.cat([x_3x3, x_1x1], dim=1)
        x = x.squeeze(2)  # 移除 in_channels 维度
        x = F.relu(self.lrn1(x))

        # Residual blocks forward
        x = self.residual_blocks[0](x)
        x = F.relu(self.lrn2(x))
        for block in self.residual_blocks[1:]:
            x = F.relu(x + block(x))

        # Classifier forward
        for i, layer in enumerate(self.classifier):
            x = layer(x)
            if i < len(self.classifier) - 1:
                x = F.relu(x)
                x = self.dropout(x)

        return x

    @staticmethod
    def _create_inception_module(in_channels: int) -> nn.ModuleDict:
        return nn.ModuleDict({
            'conv_3x3': nn.Conv3d(1, 128, (in_channels, 3, 3), stride=(in_channels, 1, 1), padding=(0, 1, 1)),
            'conv_1x1': nn.Conv3d(1, 128, (in_channels, 1, 1), stride=(in_channels, 1, 1), padding=0)
        })

    def _create_residual_blocks(self) -> nn.ModuleList:
        return nn.ModuleList([
            self._create_residual_block(256, 128),
            self._create_residual_block(128, 128)
        ])

    @staticmethod
    def _create_residual_block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (1, 1)),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, (1, 1))
        )

    @staticmethod
    def _create_classifier(n_classes: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(128, 128, (1, 1)),
            nn.ReLU(),
            nn.Conv2d(128, 128, (1, 1)),
            nn.ReLU(),
            nn.Conv2d(128, n_classes, (1, 1))
        )

    @staticmethod
    def _weight_init(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Conv3d, nn.Conv2d)):
            init.kaiming_uniform_(m.weight)
            init.zeros_(m.bias)

    def __str__(self) -> str:
        return (f"LeeEtAl(in_channels={self.inception['conv_3x3'].kernel_size[0]}, "
                f"n_classes={self.classifier[-1].out_channels})")


# # 测试代码
# batch_size, in_channels = 1, 200
# input_data = torch.randn(batch_size, 1, in_channels, 145, 145)
#
# # 创建模型实例
# n_classes = 16
# model = LeeEtAl3D(in_channels, n_classes)
#
# # 前向传播
# output = model(input_data)
#
# print(f"Input shape: {input_data.shape}")
# print(f"Output shape: {output.shape}")
# print(f"Model structure: {model}")
