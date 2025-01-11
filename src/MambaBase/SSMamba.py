import numpy as np
import torch
import torch.nn as nn
from typing import Callable
from timm.layers import DropPath
from functools import partial
from src.MambaBase.SSM import SSM


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """
    从网格生成2D正弦余弦位置嵌入
    """
    assert embed_dim % 2 == 0

    # 使用一半维度编码grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    从网格生成1D正弦余弦位置嵌入

    Args:
        embed_dim: 每个位置的输出维度
        pos: 要编码的位置列表：大小 (M,)

    输出: (M, D)
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
        x_spa = self.spa_block(x_spa)
        x_spe = self.spe_block(x_spe)
        if self.fu:
            x_spa_c = x_spa.mean(1)
            x_spe_c = x_spe.mean(1)
            sig = self.l1((x_spa_c + x_spe_c) / 2).unsqueeze(1)
            x_spa = x_spa * sig
            x_spe = x_spe * sig
        return x_spa, x_spe


class SSMamba(nn.Module):
    def __init__(self, input_channels, num_classes, embed_dim=128, depth=4, bi=True,
                 norm_layer=nn.LayerNorm, fu=True):
        super().__init__()

        self.input_channels = input_channels
        self.embed_dim = embed_dim
        self.dim = 2
        self.spa_embed = nn.Linear(input_channels, embed_dim)
        self.spe_embed = nn.Linear(input_channels, embed_dim)

        self.spa_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.spe_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.blocks = nn.ModuleList([
            spectral_spatial_block(embed_dim, bi, N=26, cls=True, fu=fu) for _ in range(depth)
        ])

        self.head = nn.Linear(embed_dim, num_classes)
        self.norm = norm_layer(embed_dim)

    def forward_features(self, x):
        B, C, H, W = x.shape

        # 将输入重塑并映射到嵌入空间
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, H*W, C)
        x_spa = self.spa_embed(x)  # (B, H*W, embed_dim)
        x_spe = self.spe_embed(x)  # (B, H*W, embed_dim)

        spa_cls_tokens = self.spa_cls_token.expand(B, -1, -1)
        spe_cls_tokens = self.spe_cls_token.expand(B, -1, -1)
        x_spa = torch.cat((x_spa, spa_cls_tokens), dim=1)
        x_spe = torch.cat((x_spe, spe_cls_tokens), dim=1)

        # 通过光谱-空间块
        for blk in self.blocks:
            x_spa, x_spe = blk(x_spa, x_spe)

        # 获取每个像素的特征
        x_spa = x_spa[:, :-1, :]  # 移除分类令牌
        x_spe = x_spe[:, :-1, :]  # 移除分类令牌
        outcome = (x_spa + x_spe) / 2

        return outcome.view(B, H, W, -1)  # (B, H, W, embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        features = self.forward_features(x)

        # 获取中心像素的特征
        center_h, center_w = H // 2, W // 2
        center_features = features[:, center_h, center_w, :]  # (B, embed_dim)

        # 对中心像素进行分类
        output = self.head(center_features)  # (B, num_classes)
        return output


if __name__ == "__main__":
    # TODO: fix model,现在效果有问题
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
