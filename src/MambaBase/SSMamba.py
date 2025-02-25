import numpy as np
import torch
import torch.nn as nn
from typing import Callable
from timm.layers import DropPath
from functools import partial
from src.MambaBase.SSM import SSM


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    从网格生成1D正弦余弦位置嵌入。

    Args:
        embed_dim (int): 每个位置的输出维度。必须是偶数。
        pos (np.ndarray): 要编码的位置数组，形状为 (M,)，其中 M 是位置的数量。

    Returns:
        np.ndarray: 位置嵌入矩阵，形状为 (M, D)，其中 M 是位置的数量，D 是 embed_dim。

    Raises:
        AssertionError: 如果 embed_dim 不是偶数。

    Notes:
        - 该函数使用正弦和余弦函数来生成位置嵌入。
        - 输出的嵌入矩阵前半部分是正弦值，后半部分是余弦值。
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), 外积

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


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

    Example:
        >>> embed_dim = 128
        >>> block = spectral_spatial_block(embed_dim, bi=True, fu=True)
        >>> x_spa = torch.randn(32, 100, embed_dim)  # (batch_size, sequence_length, embed_dim)
        >>> x_spe = torch.randn(32, 100, embed_dim)
        >>> x_spa_out, x_spe_out = block(x_spa, x_spe)
        >>> print(x_spa_out.shape, x_spe_out.shape)
        torch.Size([32, 100, 128]) torch.Size([32, 100, 128])
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


class SSMamba(nn.Module):
    """
    SSMamba 模型类，用于处理高光谱图像数据并进行分类。

    这个模型使用空间-光谱块来处理输入的高光谱图像，并综合所有像素的信息进行分类。

    Args:
        input_channels (int): 输入图像的通道数。
        num_classes (int): 分类的类别数。
        embed_dim (int, optional): 嵌入维度。默认为 128。
        depth (int, optional): 空间-光谱块的数量。默认为 4。
        bi (bool, optional): 是否在空间-光谱块中使用双向处理。默认为 True。
        norm_layer (nn.Module, optional): 归一化层类型。默认为 nn.LayerNorm。
        fu (bool, optional): 是否在空间-光谱块中使用特征融合。默认为 True。

    Attributes:
        input_channels (int): 输入通道数。
        embed_dim (int): 嵌入维度。
        spa_embed (nn.Linear): 空间特征的嵌入层。
        spe_embed (nn.Linear): 光谱特征的嵌入层。
        blocks (nn.ModuleList): 空间-光谱块的列表。
        norm (nn.Module): 归一化层。
        head (nn.Linear): 分类头。
    """

    def __init__(self, input_channels, num_classes, embed_dim=128, depth=4, bi=True,
                 norm_layer=nn.LayerNorm, fu=True, patch_size=7):
        super().__init__()

        self.input_channels = input_channels
        self.embed_dim = embed_dim
        self.dim = 2
        self.spa_embed = nn.Linear(input_channels, embed_dim)
        self.spe_embed = nn.Linear(input_channels, embed_dim)

        self.blocks = nn.ModuleList([
            spectral_spatial_block(embed_dim, bi, N=26, cls=True, fu=fu) for _ in range(depth)
        ])

        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.fusion_weight = nn.Parameter(torch.FloatTensor([0.5, 0.5]))

    def forward_features(self, x):
        """
        前向传播特征提取部分。

        Args:
            x (torch.Tensor): 输入张量，形状为 (B, C, H, W)
                B: 批次大小, C: 输入通道数, H: 高度, W: 宽度

        Returns:
            torch.Tensor: 提取的特征，形状为 (B, H, W, embed_dim)
        """
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)

        x_spa = self.spa_embed(x)  # (B, H*W, embed_dim)
        x_spe = self.spe_embed(x)  # (B, H*W, embed_dim)

        # 通过光谱-空间块
        for blk in self.blocks:
            x_spa, x_spe = blk(x_spa, x_spe)

        outcome = self.fusion_weight[0] * x_spa + self.fusion_weight[1] * x_spe
        outcome = self.norm(outcome)

        return outcome.view(B, H, W, -1)  # (B, H, W, embed_dim)

    def forward(self, x):
        """
        模型的前向传播。

        Args:
            x (torch.Tensor): 输入张量，形状为 (B, C, H, W)
                B: 批次大小, C: 输入通道数, H: 高度, W: 宽度

        Returns:
            torch.Tensor: 分类输出，形状为 (B, num_classes)
        """
        B, C, H, W = x.shape
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

    # 创建模型
    model = SSMamba(input_channels=in_chans, num_classes=num_classes)

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
