import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import init


class LeeEtAl(nn.Module):
    def __init__(self, in_channels: int, n_classes: int):
        """
        初始化 LeeEtAl 模型。

        Args:
            in_channels (int): 输入数据的通道数。
            n_classes (int): 分类问题的类别数。
        """
        super(LeeEtAl, self).__init__()
        self.name: str = 'LeeEtAl'

        self.inception: nn.ModuleDict = self._create_inception_module(in_channels)
        self.residual_blocks: nn.ModuleList = self._create_residual_blocks()
        self.classifier: nn.Sequential = self._create_classifier(n_classes)

        self.lrn1: nn.LocalResponseNorm = nn.LocalResponseNorm(256)
        self.lrn2: nn.LocalResponseNorm = nn.LocalResponseNorm(128)
        self.dropout: nn.Dropout = nn.Dropout(p=0.4)

        self.apply(self._weight_init)

    def forward(self, x: Tensor) -> Tensor:
        """
        定义模型的前向传播。

        Args:
            x (Tensor): 输入张量。

        Returns:
            Tensor: 模型的输出张量。
        """
        x = self._inception_forward(x)
        x = self._residual_blocks_forward(x)
        x = self._classifier_forward(x)
        return x

    @staticmethod
    def _create_inception_module(in_channels: int) -> nn.ModuleDict:
        """
        创建 inception 模块。

        Args:
            in_channels (int): 输入通道数。

        Returns:
            nn.ModuleDict: 包含 inception 模块的字典。
        """
        return nn.ModuleDict({
            'conv_3x3': nn.Conv3d(1, 128, (3, 3, in_channels), stride=(1, 1, 2), padding=(1, 1, 0)),
            'conv_1x1': nn.Conv3d(1, 128, (1, 1, in_channels), stride=(1, 1, 1), padding=0)
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
    def _create_residual_block(in_channels: int, out_channels: int) -> nn.Sequential:
        """
        创建单个残差块。

        Args:
            in_channels (int): 输入通道数。
            out_channels (int): 输出通道数。

        Returns:
            nn.Sequential: 包含残差块层的序列。
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (1, 1)),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, (1, 1))
        )

    @staticmethod
    def _create_classifier(n_classes: int) -> nn.Sequential:
        """
        创建分类器。

        Args:
            n_classes (int): 分类问题的类别数。

        Returns:
            nn.Sequential: 包含分类器层的序列。
        """
        return nn.Sequential(
            nn.Conv2d(128, 128, (1, 1)),
            nn.ReLU(),
            nn.Conv2d(128, 128, (1, 1)),
            nn.ReLU(),
            nn.Conv2d(128, n_classes, (9, 9))
        )

    def _inception_forward(self, x: Tensor) -> Tensor:
        """
        Inception 模块的前向传播。

        Args:
            x (Tensor): 输入张量。

        Returns:
            Tensor: Inception 模块的输出张量。
        """
        x_3x3 = self.inception['conv_3x3'](x)
        x_1x1 = self.inception['conv_1x1'](x)
        x = torch.cat([x_3x3, x_1x1], dim=1)
        x = torch.squeeze(x)
        return F.relu(self.lrn1(x))

    def _residual_blocks_forward(self, x: Tensor) -> Tensor:
        """
        残差块的前向传播。

        Args:
            x (Tensor): 输入张量。

        Returns:
            Tensor: 残差块的输出张量。
        """
        x = self.residual_blocks[0](x)
        x = F.relu(self.lrn2(x))
        for block in self.residual_blocks[1:]:
            x = F.relu(x + block(x))
        return x

    def _classifier_forward(self, x: Tensor) -> Tensor:
        """
        分类器的前向传播。

        Args:
            x (Tensor): 输入张量。

        Returns:
            Tensor: 分类器的输出张量。
        """
        for i, layer in enumerate(self.classifier):
            x = layer(x)
            if i < len(self.classifier) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        return x.squeeze(2).squeeze(2)

    @staticmethod
    def _weight_init(m: nn.Module) -> None:
        """
        初始化模型权重。

        Args:
            m (nn.Module): 需要初始化的模块。
        """
        if isinstance(m, (nn.Linear, nn.Conv3d, nn.Conv2d)):
            init.kaiming_uniform_(m.weight)
            init.zeros_(m.bias)