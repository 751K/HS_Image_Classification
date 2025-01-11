# https://github.com/li-yapeng/MambaHSI

# @ARTICLE{MambaHSI_TGRS24,
#   author={Li, Yapeng and Luo, Yong and Zhang, Lefei and Wang, Zengmao and Du, Bo},
#   journal={IEEE Transactions on Geoscience and Remote Sensing},
#   title={MambaHSI: Spatial-Spectral Mamba for Hyperspectral Image Classification},
#   year={2024},
#   volume={},
#   number={},
#   pages={1-16},
#   keywords={Hyperspectral Image Classification;Mamba;State Space Models;Transformer},
#   doi={10.1109/TGRS.2024.3430985}}

import math
import torch
from torch import nn
from mamba_ssm import Mamba


class SpeMamba(nn.Module):
    """
    光谱 Mamba 模块，用于处理光谱信息。

    Args:
        channels (int): 输入通道数。
        token_num (int, optional): 令牌数量。默认为 8。
        use_residual (bool, optional): 是否使用残差连接。默认为 True。
        group_num (int, optional): 组归一化的组数。默认为 4。

    Attributes:
        token_num (int): 令牌数量。
        use_residual (bool): 是否使用残差连接。
        group_channel_num (int): 每个组的通道数。
        channel_num (int): 总通道数。
        mamba (Mamba): Mamba 模块。
        proj (nn.Sequential): 投影层。
    """

    def __init__(self, channels, token_num=8, use_residual=True, group_num=4):
        super(SpeMamba, self).__init__()
        self.token_num = token_num
        self.use_residual = use_residual

        self.group_channel_num = math.ceil(channels / token_num)
        self.channel_num = self.token_num * self.group_channel_num

        self.mamba = Mamba(
            d_model=self.group_channel_num,
            d_state=16,
            d_conv=4,
            expand=2,
        )

        self.proj = nn.Sequential(
            nn.GroupNorm(group_num, self.channel_num),
            nn.SiLU()
        )

    def padding_feature(self, x):
        """
        对输入特征进行填充。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 填充后的张量。
        """
        B, C, H, W = x.shape
        if C < self.channel_num:
            pad_c = self.channel_num - C
            pad_features = torch.zeros((B, pad_c, H, W)).to(x.device)
            cat_features = torch.cat([x, pad_features], dim=1)
            return cat_features
        else:
            return x

    def forward(self, x):
        """
        前向传播。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 处理后的张量。
        """
        x_pad = self.padding_feature(x)
        x_pad = x_pad.permute(0, 2, 3, 1).contiguous()
        B, H, W, C_pad = x_pad.shape
        x_flat = x_pad.view(B * H * W, self.token_num, self.group_channel_num)
        x_flat = self.mamba(x_flat)
        x_recon = x_flat.view(B, H, W, C_pad)
        x_recon = x_recon.permute(0, 3, 1, 2).contiguous()
        x_proj = self.proj(x_recon)
        if self.use_residual:
            return x + x_proj
        else:
            return x_proj


class SpaMamba(nn.Module):
    """
    空间 Mamba 模块，用于处理空间信息。

    Args:
        channels (int): 输入通道数。
        use_residual (bool, optional): 是否使用残差连接。默认为 True。
        group_num (int, optional): 组归一化的组数。默认为 4。
        use_proj (bool, optional): 是否使用投影层。默认为 True。

    Attributes:
        use_residual (bool): 是否使用残差连接。
        use_proj (bool): 是否使用投影层。
        mamba (Mamba): Mamba 模块。
        proj (nn.Sequential): 投影层（如果 use_proj 为 True）。
    """

    def __init__(self, channels, use_residual=True, group_num=4, use_proj=True):
        super(SpaMamba, self).__init__()
        self.use_residual = use_residual
        self.use_proj = use_proj
        self.mamba = Mamba(
            d_model=channels,
            d_state=16,
            d_conv=4,
            expand=2,
        )
        if self.use_proj:
            self.proj = nn.Sequential(
                nn.GroupNorm(group_num, channels),
                nn.SiLU()
            )

    def forward(self, x):
        """
        前向传播。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 处理后的张量。
        """
        x_re = x.permute(0, 2, 3, 1).contiguous()
        B, H, W, C = x_re.shape
        x_flat = x_re.view(1, -1, C)
        x_flat = self.mamba(x_flat)

        x_recon = x_flat.view(B, H, W, C)
        x_recon = x_recon.permute(0, 3, 1, 2).contiguous()
        if self.use_proj:
            x_recon = self.proj(x_recon)
        if self.use_residual:
            return x_recon + x
        else:
            return x_recon


class BothMamba(nn.Module):
    """
    结合空间和光谱 Mamba 的模块。
    使用加权处理完成融合。

    Args:
        channels (int): 输入通道数。
        token_num (int): 光谱 Mamba 的令牌数量。
        use_residual (bool): 是否使用残差连接。
        group_num (int, optional): 组归一化的组数。默认为 4。
        use_att (bool, optional): 是否使用注意力机制进行融合。默认为 True。

    Attributes:
        use_att (bool): 是否使用注意力机制。
        use_residual (bool): 是否使用残差连接。
        weights (nn.Parameter): 注意力权重（如果 use_att 为 True）。
        softmax (nn.Softmax): Softmax 层（如果 use_att 为 True）。
        spa_mamba (SpaMamba): 空间 Mamba 模块。
        spe_mamba (SpeMamba): 光谱 Mamba 模块。
    """

    def __init__(self, channels, token_num, use_residual, group_num=4, use_att=True):
        super(BothMamba, self).__init__()
        self.use_att = use_att
        self.use_residual = use_residual
        if self.use_att:
            self.weights = nn.Parameter(torch.ones(2) / 2)
            self.softmax = nn.Softmax(dim=0)

        self.spa_mamba = SpaMamba(channels, use_residual=use_residual, group_num=group_num)
        self.spe_mamba = SpeMamba(channels, token_num=token_num, use_residual=use_residual, group_num=group_num)

    def forward(self, x):
        """
        前向传播。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 处理后的张量。
        """
        spa_x = self.spa_mamba(x)
        spe_x = self.spe_mamba(x)
        if self.use_att:
            weights = self.softmax(self.weights)
            fusion_x = spa_x * weights[0] + spe_x * weights[1]
        else:
            fusion_x = spa_x + spe_x
        if self.use_residual:
            return fusion_x + x
        else:
            return fusion_x


class MambaHSI(nn.Module):
    """
    Mamba-based 高光谱图像分类模型。

    这个模型使用 Mamba 架构处理高光谱图像数据，支持空间、光谱或两者结合的处理方式。

    Args:
        input_channels (int, optional): 输入图像的通道数。默认为 128。
        num_classes (int, optional): 分类类别数。默认为 10。
        hidden_dim (int, optional): 隐藏层维度。默认为 64。
        use_residual (bool, optional): 是否使用残差连接。默认为 True。
        mamba_type (str, optional): Mamba 模块类型，可选 'spa'、'spe' 或 'both'。默认为 'both'。
        token_num (int, optional): 光谱 Mamba 的令牌数量。默认为 4。
        group_num (int, optional): 组归一化的组数。默认为 4。
        use_att (bool, optional): 在 'both' 模式下是否使用注意力机制。默认为 True。

    Attributes:
        dim (int): 模型维度。
        mamba_type (str): 使用的 Mamba 类型。
        patch_embedding (nn.Sequential): 补丁嵌入层。
        mamba (nn.Sequential): Mamba 模块序列。
        cls_head (nn.Sequential): 分类头。
    """

    def __init__(self, input_channels=128, num_classes=10, hidden_dim=64, use_residual=True, mamba_type='both',
                 token_num=4, group_num=4, use_att=True):
        super(MambaHSI, self).__init__()
        self.dim = 2
        self.mamba_type = mamba_type
        self.num_layers = 3

        # 补丁嵌入层
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(group_num, hidden_dim),
            nn.SiLU())

        # 定义 Mamba 模块映射
        mamba_modules = {
            'spa': lambda: SpaMamba(hidden_dim, use_residual=use_residual, group_num=group_num),
            'spe': lambda: SpeMamba(hidden_dim, token_num=token_num, use_residual=use_residual, group_num=group_num),
            'both': lambda: BothMamba(channels=hidden_dim, token_num=token_num, use_residual=use_residual,
                                      group_num=group_num, use_att=use_att)
        }

        # 创建 Mamba 模块序列
        self.mamba = nn.Sequential(
            *[mamba_modules[mamba_type]() for _ in range(self.num_layers)]
        )

        # 分类头
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(group_num, 128),
            nn.SiLU(),
            nn.Conv2d(in_channels=128, out_channels=num_classes, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        """
        模型的前向传播。

        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, input_channels, height, width)。

        Returns:
            torch.Tensor: 模型输出的 logits，形状为 (batch_size, num_classes, height, width)。
        """
        x = self.patch_embedding(x)
        x = self.mamba(x)
        logits = self.cls_head(x)
        return logits


# 测试代码
if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 4
    in_channels = 128
    height = 5
    width = 5
    x = torch.randn(batch_size, in_channels, height, width)

    model = MambaHSI(input_channels=in_channels, hidden_dim=64, num_classes=10, mamba_type='both')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = model.to(device)
    x = x.to(device)

    try:
        with torch.no_grad():
            output = model(x)

        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params}")

    except Exception as e:
        print(f"An error occurred: {e}")
