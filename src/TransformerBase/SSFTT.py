import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


class Residual(nn.Module):
    """
    残差连接模块

    Args:
        fn (nn.Module): 需要应用残差连接的模块
    """

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        """
        Args:
            x (torch.Tensor): 输入张量
            **kwargs: 其他参数

        Returns:
            torch.Tensor: 残差连接的输出
        """
        return self.fn(x, **kwargs) + x


class LayerNormalize(nn.Module):
    """
    层归一化模块，等同于PreNorm

    Args:
        dim (int): 输入特征的维度
        fn (nn.Module): 需要应用层归一化的模块
    """

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        """
        Args:
            x (torch.Tensor): 输入张量
            **kwargs: 其他参数

        Returns:
            torch.Tensor: 层归一化后的输出
        """
        return self.fn(self.norm(x), **kwargs)


class MLP_Block(nn.Module):
    """
    多层感知机模块，等同于FeedForward

    Args:
        dim (int): 输入和输出的特征维度
        hidden_dim (int): 隐藏层的维度
        dropout (float): dropout率，默认为0.1
    """

    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入张量

        Returns:
            torch.Tensor: MLP处理后的输出
        """
        return self.net(x)


class Attention(nn.Module):
    """
    多头自注意力机制模块

    Args:
        dim (int): 输入特征的维度
        heads (int): 注意力头的数量，默认为8
        dropout (float): dropout的比率，默认为0.1
    """

    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5  # 缩放因子，防止点积结果过大

        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)  # 线性变换，将输入转换为query, key, value
        self.output_linear = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        前向传播

        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, dim)
            mask (torch.Tensor, 可选): 注意力掩码，默认为None

        Returns:
            torch.Tensor: 注意力机制的输出，形状与输入相同
        """
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.output_linear(out)
        out = self.dropout(out)
        return out


class Transformer(nn.Module):
    """
    Transformer模块

    Args:
        dim (int): 输入特征的维度
        depth (int): Transformer层的数量
        heads (int): 注意力头的数量
        mlp_dim (int): MLP中间层的维度
        dropout (float): dropout率
    """

    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(LayerNormalize(dim, Attention(dim, heads=heads, dropout=dropout))),
                Residual(LayerNormalize(dim, MLP_Block(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        """
        Args:
            x (torch.Tensor): 输入张量
            mask (torch.Tensor, 可选): 注意力掩码，默认为None

        Returns:
            torch.Tensor: Transformer处理后的输出
        """
        for attention, mlp in self.layers:
            x = attention(x, mask=mask)
            x = mlp(x)
        return x


class SSFTT(nn.Module):
    """
    SSFTTnet模型

    Args:
        input_channels (int): 输入光谱通道数。默认值: 200。
        num_classes (int): 输出类别数。默认值: 16。
        num_tokens (int): transformer的标记数。默认值: 4。
        hidden_dim (int): 隐藏层维度大小。默认值: 64。
        depth (int): transformer层数。默认值: 1。
        heads (int): transformer中的注意力头数。默认值: 8。
        mlp_dim (int): transformer中MLP的维度。默认值: 8。
        dropout (float): transformer中的dropout率。默认值: 0.1。
        emb_dropout (float): 嵌入层的dropout率。默认值: 0.1。
    """

    def __init__(self, input_channels: int = 200, num_classes: int = 16, num_tokens=4, hidden_dim=64, depth=1, heads=8,
                 mlp_dim=8, dropout=0.1, emb_dropout=0.1, patch_size=5):
        super(SSFTT, self).__init__()
        self.L = num_tokens
        self.cT = hidden_dim
        self.dim = 3

        # 3D卷积特征提取
        self.conv3d_features = nn.Sequential(
            nn.Conv3d(1, out_channels=8, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(8),
            nn.ReLU(),
        )

        # 2D卷积特征提取
        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=8 * (input_channels - 2), out_channels=64, kernel_size=(3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # Tokenization
        self.token_wA = nn.Parameter(torch.empty(1, self.L, 64), requires_grad=True)
        torch.nn.init.xavier_normal_(self.token_wA)
        self.token_wV = nn.Parameter(torch.empty(1, 64, self.cT), requires_grad=True)
        torch.nn.init.xavier_normal_(self.token_wV)

        # 位置编码
        self.pos_embedding = nn.Parameter(torch.empty(1, (num_tokens + 1), hidden_dim))
        torch.nn.init.normal_(self.pos_embedding, std=.02)

        # 分类token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.dropout = nn.Dropout(emb_dropout)

        # Transformer
        self.transformer = Transformer(hidden_dim, depth, heads, mlp_dim, dropout)

        self.to_cls_token = nn.Identity()

        # 分类头
        self.nn1 = nn.Linear(hidden_dim, num_classes)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std=1e-6)

    def forward(self, x, mask=None):
        """
        前向传播

        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, 1, in_channels, height, width)
            mask (torch.Tensor, 可选): 注意力掩码，默认为None

        Returns:
            torch.Tensor: 模型的输出，形状为 (batch_size, num_classes)
        """
        x = self.conv3d_features(x)
        x = rearrange(x, 'b c d h w -> b (c d) h w')
        x = self.conv2d_features(x)
        x = rearrange(x, 'b c h w -> b (h w) c')

        wa = rearrange(self.token_wA, 'b h w -> b w h')  # Transpose
        A = torch.einsum('bij,bjk->bik', x, wa)
        A = rearrange(A, 'b h w -> b w h')  # Transpose
        A = A.softmax(dim=-1)

        VV = torch.einsum('bij,bjk->bik', x, self.token_wV)
        T = torch.einsum('bij,bjk->bik', A, VV)

        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, T), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)
        x = self.transformer(x, mask)
        x = self.to_cls_token(x[:, 0])
        x = self.nn1(x)

        return x


if __name__ == '__main__':
    # 测试代码
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size, in_channels = 16, 200
    n_classes = 16
    model = SSFTT(input_channels=in_channels, num_classes=n_classes)
    model.to(device)

    # 测试不同输入尺寸
    for size in [(5, 5), (7, 7), (9, 9)]:
        height, width = size
        input_data = torch.randn(batch_size, 1, in_channels, height, width)
        input_data = input_data.to(device)

        # 前向传播
        output = model(input_data)

        print(f"Input shape: {input_data.shape}")
        print(f"Output shape: {output.shape}")
