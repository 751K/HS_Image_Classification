import torch
import torch.nn as nn
from typing import Callable
from timm.layers import DropPath
from functools import partial
from src.MambaBase.SSM import SSM


class AdaptiveFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, x_spa, x_spe):
        # 获取全局特征
        global_spa = x_spa.mean(dim=1)  # (B, dim)
        global_spe = x_spe.mean(dim=1)  # (B, dim)

        # 计算融合权重
        global_features = torch.cat([global_spa, global_spe], dim=-1)  # (B, dim*2)
        weights = self.fc(global_features)  # (B, 2)

        # 应用权重
        fused = weights[:, 0].unsqueeze(1).unsqueeze(1) * x_spa + \
                weights[:, 1].unsqueeze(1).unsqueeze(1) * x_spe

        return fused


class ResidualSpectralSpatialBlock(nn.Module):
    def __init__(self, embed_dim, **kwargs):
        super().__init__()
        self.block = spectral_spatial_block(embed_dim, **kwargs)
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
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 确保每个尺度的通道数加起来等于 out_channels
        scale_out_channels = out_channels // 4

        self.scale1 = SpatialSpectralFeatureExtraction(in_channels, scale_out_channels,
                                                       spatial_kernel_sizes=[3], spectral_kernel_size=5)
        self.scale2 = SpatialSpectralFeatureExtraction(in_channels, scale_out_channels,
                                                       spatial_kernel_sizes=[5], spectral_kernel_size=7)
        self.scale3 = SpatialSpectralFeatureExtraction(in_channels, scale_out_channels,
                                                       spatial_kernel_sizes=[7], spectral_kernel_size=9)
        self.scale4 = SpatialSpectralFeatureExtraction(in_channels, scale_out_channels,
                                                       spatial_kernel_sizes=[9], spectral_kernel_size=11)

        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.scale1(x)
        x2 = self.scale2(x)
        x3 = self.scale3(x)
        x4 = self.scale4(x)
        x_cat = torch.cat([x1, x2, x3, x4], dim=1)
        return self.fusion(x_cat)


class SpatialSpectralFeatureExtraction(nn.Module):
    def __init__(self, in_channels, out_channels, spatial_kernel_sizes=[3], spectral_kernel_size=7):
        super().__init__()

        self.out_channels = out_channels

        # 3D convolution paths with different kernel sizes
        self.conv3d_paths = nn.ModuleList()
        for spatial_kernel_size in spatial_kernel_sizes:
            self.conv3d_paths.append(nn.Sequential(
                # Spectral convolution
                nn.Conv3d(1, out_channels // 2,
                          kernel_size=(spectral_kernel_size, 1, 1),
                          padding=(spectral_kernel_size // 2, 0, 0)),
                nn.BatchNorm3d(out_channels // 2),
                nn.ReLU(inplace=True),
                # Spatial convolution (depthwise)
                nn.Conv3d(out_channels // 2,
                          out_channels // 2,
                          kernel_size=(1, spatial_kernel_size, spatial_kernel_size),
                          padding=(0, spatial_kernel_size // 2, spatial_kernel_size // 2),
                          groups=out_channels // 2),
                nn.BatchNorm3d(out_channels // 2),
                nn.ReLU(inplace=True)
            ))

        # 2D convolution path (depthwise separable)
        self.conv2d_depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.conv2d_pointwise = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1)

        self.bn2d = nn.BatchNorm2d(out_channels // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x shape: (B, C, H, W)
        B, C, H, W = x.shape

        # 3D convolution paths
        x_3d = x.unsqueeze(1)  # (B, 1, C, H, W)
        x_3d_outputs = []
        for conv3d_path in self.conv3d_paths:
            x_3d_out = conv3d_path(x_3d)
            x_3d_out = x_3d_out.max(dim=2)[0]  # (B, out_channels // 2, H, W)
            x_3d_outputs.append(x_3d_out)

        x_3d_combined = torch.cat(x_3d_outputs, dim=1)  # (B, out_channels // 2, H, W)

        # 2D convolution path
        x_2d = self.conv2d_depthwise(x)
        x_2d = self.conv2d_pointwise(x_2d)
        x_2d = self.relu(self.bn2d(x_2d))

        # Concatenate 3D and 2D features
        x_combined = torch.cat([x_3d_combined, x_2d], dim=1)  # (B, out_channels, H, W)

        return x_combined


class block_1D(nn.Module):
    """
    一维块结构，包含自注意力机制和残差连接。

    这个模块实现了一个基于状态空间模型的一维块，可以选择是否使用双向处理和类别标记。

    Args:
        hidden_dim (int): 隐藏层的维度。默认为0。
        drop_path (float): DropPath的概率。默认为0。
        norm_layer (Callable[..., torch.nn.Module]): 归一化层的类型。默认为LayerNorm。
        attn_drop_rate (float): 注意力层的dropout率。默认为0。
        d_state (int): 状态空间模型的状态维度。默认为16。
        bi (bool): 是否使用双向处理。默认为True。
        cls (bool): 是否包含类别标记。默认为True。
        **kwargs: 传递给SSM的额外参数。

    Attributes:
        ln_1 (nn.Module): 层归一化。
        self_attention (SSM): 状态空间模型的自注意力层。
        drop_path (DropPath): DropPath层。
        bi (bool): 是否使用双向处理。
        cls (bool): 是否包含类别标记。
    """

    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            bi: bool = True,
            cls: bool = True,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)

        self.self_attention = SSM(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.bi = bi
        self.cls = cls

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。

        Args:
            input (torch.Tensor): 输入张量，形状为 (batch_size, sequence_length, hidden_dim)。

        Returns:
            torch.Tensor: 输出张量，形状与输入相同。

        Note:
            如果 self.bi 为 True，则进行双向处理。
            如果 self.cls 为 True，则特殊处理类别标记。
        """
        x = self.ln_1(input)
        x1 = self.self_attention(x)
        if self.bi:
            if self.cls:
                # 处理包含类别标记的双向情况
                x2 = x[:, 0:-1, :]  # 除去最后一个标记（类别标记）
                cls_token = x[:, -1:, :]  # 类别标记
                x2 = torch.flip(x2, dims=[1])  # 反转序列
                x2 = torch.cat((x2, cls_token), dim=1)  # 重新添加类别标记
                x3 = self.self_attention(x2)

                # 再次处理结果
                x2 = x3[:, 0:-1, :]
                cls_token = x3[:, -1:, :]
                x3 = torch.flip(x2, dims=[1])
                x3 = torch.cat((x3, cls_token), dim=1)
            else:
                # 不包含类别标记的双向处理
                x3 = torch.flip(x, dims=[1])
                x3 = self.self_attention(x3)
                x3 = torch.flip(x3, dims=[1])

            # 合并双向结果并应用残差连接
            return self.drop_path((x1 + x3) / 2) + input
        else:
            # 单向处理
            return self.drop_path(x1) + input


class spectral_spatial_block(nn.Module):
    """
    空间-光谱块类，用于处理和融合高光谱图像的空间和光谱信息。

    这个模块设计用于同时处理高光谱图像的空间和光谱特征。它包含两个独立的 1D 块
    （一个用于空间特征，一个用于光谱特征），以及一个可选的特征融合机制。
    这种设计允许模型分别捕获空间和光谱域的信息，然后通过融合步骤将它们结合起来。

    Args:
        embed_dim (int): 嵌入向量的维度。
        bi (bool, optional): 是否在 1D 块中使用双向处理。默认为 False。
        N (int, optional): 未使用的参数，为了兼容性保留。默认为 1。
        drop_path (float, optional): 用于 1D 块的 drop path 概率。默认为 0.0。
        norm_layer (nn.Module, optional): 用于 1D 块的归一化层类型。默认为 nn.LayerNorm。
        cls (bool, optional): 是否在 1D 块中包含分类令牌。默认为 True。
        fu (bool, optional): 是否启用特征融合。默认为 True。

    Attributes:
        spa_block (block_1D): 处理空间特征的 1D 块。
        spe_block (block_1D): 处理光谱特征的 1D 块。
        fu (bool): 是否启用特征融合。
        l1 (nn.Sequential): 特征融合使用的线性层和激活函数（如果 fu=True）。
    """

    def __init__(self, embed_dim, bi=False, N=1, drop_path=0.0, norm_layer=nn.LayerNorm, cls=True, fu=True):
        super(spectral_spatial_block, self).__init__()
        self.spa_block = block_1D(
            hidden_dim=embed_dim,
            drop_path=drop_path,
            bi=bi,
            cls=cls,
        )
        self.spe_block = block_1D(
            hidden_dim=embed_dim,
            drop_path=drop_path,
            bi=bi,
            cls=cls
        )
        self.fu = fu
        if self.fu:
            self.l1 = nn.Sequential(
                nn.Linear(embed_dim, embed_dim, bias=False),
                nn.Sigmoid(),
            )

    def forward(self, x_spa: torch.Tensor, x_spe: torch.Tensor) -> tuple:
        """
        执行空间-光谱块的前向传播。

        这个方法首先分别通过空间和光谱的 1D 块处理输入特征。
        如果启用了特征融合（fu=True），则会计算空间和光谱特征的平均值，
        通过一个线性层和 Sigmoid 激活函数生成注意力权重，
        然后将这个权重应用到原始的空间和光谱特征上。

        Args:
            x_spa (torch.Tensor): 空间特征张量，形状为 (B, N_spa, embed_dim)
                B: 批次大小
                N_spa: 空间序列长度（patch数量 + 1，如果 cls=True）
                embed_dim: 嵌入维度
            x_spe (torch.Tensor): 光谱特征张量，形状为 (B, N_spe, embed_dim)
                N_spe: 光谱序列长度（patch数量 + 1，如果 cls=True）

        Returns:
            tuple: 包含两个张量的元组：
                - x_spa (torch.Tensor): 更新后的空间特征，形状为 (B, N_spa, embed_dim)
                - x_spe (torch.Tensor): 更新后的光谱特征，形状为 (B, N_spe, embed_dim)

        Note:
            - 如果 fu=False，则直接返回经过各自 1D 块处理后的特征，不进行融合。
            - 特征融合步骤使用了平均池化和元素级乘法，这可能会影响计算复杂度和内存使用。
        """
        x_spa = self.spa_block(x_spa)
        x_spe = self.spe_block(x_spe)

        if self.fu:
            x_spa_c = x_spa.mean(1)  # (B, embed_dim)
            x_spe_c = x_spe.mean(1)  # (B, embed_dim)
            sig = self.l1((x_spa_c + x_spe_c) / 2).unsqueeze(1)  # (B, 1, embed_dim)
            x_spa = x_spa * sig  # (B, N_spa, embed_dim)
            x_spe = x_spe * sig  # (B, N_spe, embed_dim)

        return x_spa, x_spe


class MSAFMamba(nn.Module):
    def __init__(self, input_channels, num_classes, embed_dim=128, depth=3, bi=True,
                 norm_layer=nn.LayerNorm, fu=True):
        super().__init__()

        self.input_channels = input_channels
        self.embed_dim = embed_dim
        self.dim = 2

        # 使用多尺度特征提取模块
        self.feature_extraction = MultiScaleFeatureExtraction(input_channels, embed_dim)

        # 使用残差空间-光谱块
        self.blocks = nn.ModuleList([
            ResidualSpectralSpatialBlock(embed_dim, bi=bi, N=26, cls=True, fu=fu) for _ in range(depth)
        ])

        self.adaptive_fusion = AdaptiveFusion(embed_dim)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward_features(self, x):
        # x shape: (B, C, H, W)
        B, C, H, W = x.shape

        # 使用多尺度特征提取模块
        x = self.feature_extraction(x)  # (B, embed_dim, H, W)

        x = x.permute(0, 2, 3, 1).reshape(B, H * W, -1)  # (B, H*W, embed_dim)

        # 双路处理
        x_spa, x_spe = x, x

        # 通过残差光谱-空间块
        for blk in self.blocks:
            x_spa, x_spe = blk(x_spa, x_spe)

        # 使用自适应融合
        outcome = self.adaptive_fusion(x_spa, x_spe)
        outcome = self.norm(outcome)

        return outcome.view(B, H, W, -1)  # (B, H, W, embed_dim)

    def forward(self, x):
        features = self.forward_features(x)  # (B, H, W, embed_dim)

        # 对所有像素的特征进行平均池化
        global_features = features.mean(dim=[1, 2])  # (B, embed_dim)

        # 对全局特征进行分类
        output = self.head(global_features)  # (B, num_classes)
        return output


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
