import torch
import torch.nn as nn
from typing import Callable
from timm.layers import DropPath
from functools import partial
from src.MambaBase.SSM import SSM
import torch.nn.functional as F


class AdaptiveFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim * 2, 16),  # 计算空间和光谱特征的合并表示
            nn.ReLU(),
            nn.Linear(16, 2),  # 输出2个权重
            nn.Softmax(dim=-1)  # 归一化
        )

    def forward(self, x_spa, x_spe):
        # 获取全局特征
        global_spa = x_spa.mean(dim=1)  # (B, dim)
        global_spe = x_spe.mean(dim=1)  # (B, dim)

        # 计算融合权重
        global_features = torch.cat([global_spa, global_spe], dim=-1)  # (B, dim*2)
        weights = self.fc(global_features)  # (B, 2)

        # 广播计算融合
        fused = (weights[:, 0].unsqueeze(1).unsqueeze(1) * x_spa) + (weights[:, 1].unsqueeze(1).unsqueeze(1) * x_spe)

        return fused


class ResidualSpectralSpatialBlock(nn.Module):
    def __init__(self, embed_dim, **kwargs):
        super().__init__()
        self.block = SpectralSpatialBlock(embed_dim, **kwargs)
        self.shortcut = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, x_spa, x_spe):
        identity_spa = self.shortcut(x_spa)
        identity_spe = self.shortcut(x_spe)
        x_spa, x_spe = self.block(x_spa, x_spe)
        return x_spa + identity_spa, x_spe + identity_spe


class MultiScaleFeatureExtraction(nn.Module):
    def __init__(self, in_channels, out_channels, spatial_kernels=None, spectral_kernels=None):
        super().__init__()
        # 确保每个尺度的通道数加起来等于 out_channels
        if spectral_kernels is None:
            spectral_kernels = [5, 7, 9, 11]
        if spatial_kernels is None:
            spatial_kernels = [3, 5, 7, 9]
        scale_out_channels = out_channels // 4

        self.scale1 = SpatialSpectralFeatureExtraction(in_channels, scale_out_channels,
                                                       spatial_kernel_size=spatial_kernels[0],
                                                       spectral_kernel_size=spectral_kernels[0])
        self.scale2 = SpatialSpectralFeatureExtraction(in_channels, scale_out_channels,
                                                       spatial_kernel_size=spatial_kernels[1],
                                                       spectral_kernel_size=spectral_kernels[1])
        self.scale3 = SpatialSpectralFeatureExtraction(in_channels, scale_out_channels,
                                                       spatial_kernel_size=spatial_kernels[2],
                                                       spectral_kernel_size=spectral_kernels[2])
        self.scale4 = SpatialSpectralFeatureExtraction(in_channels, scale_out_channels,
                                                       spatial_kernel_size=spatial_kernels[3],
                                                       spectral_kernel_size=spectral_kernels[3])

        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, input_data):
        x1 = self.scale1(input_data)
        x2 = self.scale2(input_data)
        x3 = self.scale3(input_data)
        x4 = self.scale4(input_data)
        x_cat = torch.cat([x1, x2, x3, x4], dim=1)
        return self.fusion(x_cat)


class SpatialSpectralFeatureExtraction(nn.Module):
    def __init__(self, in_channels, out_channels, spatial_kernel_size=3, spectral_kernel_size=7):
        super().__init__()

        self.out_channels = out_channels

        # 3D卷积路径
        self.conv3d_path = nn.Sequential(
            # 光谱卷积
            nn.Conv3d(1, out_channels // 2,
                      kernel_size=(spectral_kernel_size, 1, 1),
                      padding=(spectral_kernel_size // 2, 0, 0)),
            nn.BatchNorm3d(out_channels // 2),
            nn.ReLU(),
            # 空间卷积（深度可分离卷积）
            nn.Conv3d(out_channels // 2,
                      out_channels // 2,
                      kernel_size=(1, spatial_kernel_size, spatial_kernel_size),
                      padding=(0, spatial_kernel_size // 2, spatial_kernel_size // 2),
                      groups=out_channels // 2),
            nn.BatchNorm3d(out_channels // 2),
            nn.ReLU()
        )

        # 2D卷积路径（深度可分离卷积）
        self.conv2d_depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.conv2d_pointwise = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1)

        self.bn2d = nn.BatchNorm2d(out_channels // 2)

    def forward(self, input_data):
        # x shape: (B, C, H, W)
        # 3D卷积路径
        x_3d = input_data.unsqueeze(1)  # (B, 1, C, H, W)
        x_3d_out = self.conv3d_path(x_3d)
        x_3d_out = x_3d_out.max(dim=2)[0]  # (B, out_channels // 2, H, W)

        # 2D卷积路径
        x_2d = self.conv2d_depthwise(input_data)
        x_2d = self.conv2d_pointwise(x_2d)
        x_2d = self.bn2d(x_2d)
        x_2d = F.relu(x_2d)  # ReLU 在 BatchNorm 后进行

        # 合并光谱和空间特征
        x_combined = torch.cat([x_3d_out, x_2d], dim=1)  # (B, out_channels, H, W)

        return x_combined


class Basic_Block(nn.Module):
    """
    一维块结构，包含自注意力机制和残差连接。
    这个模块实现了一个基于状态空间模型的一维块。
    """

    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            cls: bool = True,
            activation: str = 'relu',  # 可以选择激活函数类型
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SSM(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.cls = cls

        # 根据需要选择激活函数
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = None  # 如果没有激活函数，直接使用线性层

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。

        Args:
            input_data (torch.Tensor): 输入张量，形状为 (batch_size, sequence_length, hidden_dim)。

        Returns:
            torch.Tensor: 输出张量，形状与输入相同。
        """
        tmp_data = self.ln_1(input_data)
        x1 = self.self_attention(tmp_data)

        if self.activation:
            x1 = self.activation(x1)

        # 仅保留单向处理
        return self.drop_path(x1) + input_data


class SpectralSpatialBlock(nn.Module):
    """
    空间-光谱块类，用于处理和融合高光谱图像的空间和光谱信息。

    该模块分别使用 1D 块处理空间和光谱特征，并根据需求执行特征融合。

    Args:
        embed_dim (int): 嵌入向量的维度。
        drop_path (float, optional): 用于 1D 块的 Drop Path 概率。默认为 0.0。
        cls (bool, optional): 是否在 1D 块中包含分类令牌。默认为 True。
        fu (bool, optional): 是否启用特征融合。默认为 True。

    Attributes:
        spa_block (Basic_Block): 处理空间特征的 1D 块。
        spe_block (Basic_Block): 处理光谱特征的 1D 块。
        fu (bool): 是否启用特征融合。
        fusion_layer (nn.Sequential, optional): 特征融合层（如果 fu=True）。
    """

    def __init__(self, embed_dim, drop_path=0.0, cls=True, fu=True):
        super(SpectralSpatialBlock, self).__init__()

        # 空间和光谱特征的 1D 块
        self.spa_block = Basic_Block(hidden_dim=embed_dim, drop_path=drop_path, cls=cls)
        self.spe_block = Basic_Block(hidden_dim=embed_dim, drop_path=drop_path, cls=cls)

        # 特征融合选项
        self.fu = fu
        if self.fu:
            self.fusion_layer = nn.Sequential(
                nn.Linear(embed_dim, embed_dim, bias=False),
                nn.Sigmoid(),
            )

    def forward(self, x_spa: torch.Tensor, x_spe: torch.Tensor) -> tuple:
        """
        执行空间-光谱块的前向传播。

        分别处理空间和光谱特征，并在启用融合时进行特征融合。

        Args:
            x_spa (torch.Tensor): 空间特征张量，形状为 (B, N_spa, embed_dim)。
            x_spe (torch.Tensor): 光谱特征张量，形状为 (B, N_spe, embed_dim)。

        Returns:
            tuple: 更新后的空间和光谱特征。
                - x_spa (torch.Tensor): 更新后的空间特征。
                - x_spe (torch.Tensor): 更新后的光谱特征。

        Note:
            - 如果 fu=False，直接返回空间和光谱特征，不进行融合。
        """
        # 分别通过空间和光谱块处理输入特征
        x_spa = self.spa_block(x_spa)
        x_spe = self.spe_block(x_spe)

        if self.fu:
            # 计算空间和光谱特征的平均池化
            x_spa_c = x_spa.mean(1)  # (B, embed_dim)
            x_spe_c = x_spe.mean(1)  # (B, embed_dim)

            # 融合两者的特征，生成注意力权重
            fusion_weights = self.fusion_layer((x_spa_c + x_spe_c) / 2).unsqueeze(1)  # (B, 1, embed_dim)

            # 将权重应用到空间和光谱特征
            x_spa = x_spa * fusion_weights  # (B, N_spa, embed_dim)
            x_spe = x_spe * fusion_weights  # (B, N_spe, embed_dim)

        return x_spa, x_spe

    def __repr__(self):
        return f"SpectralSpatialBlock(embed_dim={self.spa_block.hidden_dim}, fu={self.fu})"


class MSAFMamba(nn.Module):
    def __init__(self, input_channels, num_classes, embed_dim=128, depth=5,
                 fu=True, spectral_kernels=None, spatial_kernels=None):
        super().__init__()
        self.dim = 2

        # 如果传入的是元组，转换成列表；如果没有传入，使用默认值
        spectral_kernels = list(spectral_kernels) if isinstance(spectral_kernels, tuple) else (
                spectral_kernels or [3, 5, 7, 9])
        spatial_kernels = list(spatial_kernels) if isinstance(spatial_kernels, tuple) else (
                spatial_kernels or [3, 5, 7, 9])

        # 1. 使用多尺度特征提取模块
        self.feature_extraction = MultiScaleFeatureExtraction(
            input_channels, embed_dim, spatial_kernels, spectral_kernels
        )

        # 2. 使用残差空间-光谱块
        self.blocks = nn.ModuleList([
            ResidualSpectralSpatialBlock(embed_dim, cls=True, fu=fu) for _ in range(depth)
        ])

        # 3. 使用自适应融合
        self.adaptive_fusion = AdaptiveFusion(embed_dim)

        # 4. 分类头部
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward_features(self, root_data):
        B, C, Height, Width = root_data.shape

        # 1. 使用多尺度特征提取
        root_data = self.feature_extraction(root_data)  # (B, embed_dim, H, W)
        root_data = root_data.permute(0, 2, 3, 1).reshape(B, Height * Width, -1)  # (B, H*W, embed_dim)

        # 2. 初始化空间和光谱特征
        spatial_features, spectral_features = root_data, root_data

        # 3. 通过残差空间-光谱块
        for blk in self.blocks:
            spatial_features, spectral_features = blk(spatial_features, spectral_features)

        # 4. 使用自适应融合
        outcome = self.adaptive_fusion(spatial_features, spectral_features)
        outcome = self.norm(outcome)

        return outcome.view(B, Height, Width, -1)  # (B, H, W, embed_dim)

    def forward(self, input_data):
        # 1. 提取特征
        features = self.forward_features(input_data)  # (B, H, W, embed_dim)

        # 2. 全局池化
        global_features = features.mean(dim=[1, 2])  # (B, embed_dim)

        # 3. 分类
        outcome = self.head(global_features)  # (B, num_classes)
        return outcome

    def __repr__(self):
        return f"MSAFMamba(input_channels={self.input_channels}, num_classes={self.head.out_features}, " \
               f"embed_dim={self.embed_dim}, depth={len(self.blocks)})"


if __name__ == "__main__":
    torch.manual_seed(42)

    # 设置模型参数
    in_chans = 64
    embed_dim = 128
    num_classes = 10

    # 创建改进后的模型
    model = MSAFMamba(input_channels=in_chans, num_classes=num_classes)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    # 测试不同输入尺寸
    test_sizes = [(3, 3), (5, 5), (7, 7), (9, 9)]

    for size in test_sizes:
        H, W = size
        x = torch.randn(4, in_chans, H, W).to(device)

        try:
            with torch.no_grad():
                output = model(x)

            print(f"Input shape: {x.shape}")
            print(f"Output shape: {output.shape}")

        except Exception as e:
            print(f"An error occurred for input size {size}: {e}\n")

    # 计算并打印模型参数总数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
