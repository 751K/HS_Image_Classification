import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.nn import functional as F

from Train_and_Eval.device import get_device


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        # x:[b,n,dim]
        b, n, _ = x.shape
        h = self.heads

        # get qkv tuple:([b,n,head_num*head_dim],[...],[...])
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # split q,k,v from [b,n,head_num*head_dim] -> [b,head_num,n,head_dim]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        # transpose(k) * q / sqrt(head_dim) -> [b,head_num,n,n]
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        # mask value: -inf
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        # softmax normalization -> attention matrix
        attn = dots.softmax(dim=-1)
        # value * attention matrix -> output
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        # cat all output -> [b, n, head_num*head_dim]
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_head, dropout, num_channel):
        super().__init__()

        # 初始化 transformer 层
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_head, dropout=dropout)))
            ]))

        self.skipcat = nn.ModuleList([])
        for _ in range(depth - 2):
            self.skipcat.append(nn.Conv2d(num_channel + 1, num_channel + 1, (1, 2), 1, 0))

    def forward(self, x, mask=None):
        last_output = []
        nl = 0
        for attn, ff in self.layers:
            last_output.append(x)
            if nl > 1:
                x = self.skipcat[nl - 2](
                    torch.cat([x.unsqueeze(3), last_output[nl - 2].unsqueeze(3)], dim=3)).squeeze(3)
            x = attn(x, mask=mask)
            x = ff(x)
            nl += 1
        return x


class SFT(nn.Module):
    def __init__(self, input_channels, num_classes, patch_size=5, near_band=3, hidden_dim=64, depth=5, heads=8, mlp_dim=16,
                 dim_head=16, dropout=0.1, emb_dropout=0.1):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, input_channels + 1, hidden_dim))
        self.patch_to_embedding = nn.Linear(patch_size ** 2 * near_band, hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(hidden_dim, depth, heads, dim_head, mlp_dim, dropout, input_channels)
        self.near_band = near_band
        self.to_latent = nn.Identity()
        self.dim = 2

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes)
        )

    def prepare(self, input_data):
        nn = self.near_band // 2
        batch_size, band, H, W = input_data.shape
        pp = (W ** 2) // 2

        data = rearrange(input_data, 'b c h w -> b (h w) c')

        times_data = torch.zeros(
            (batch_size, H ** 2 * self.near_band, band),
            dtype=torch.float,
            device=input_data.device
        )

        # Set center band
        center_start = nn * H ** 2
        center_end = (nn + 1) * H ** 2
        times_data[:, center_start:center_end, :] = data

        if pp > 0:
            for i in range(nn):
                start_idx = i * H ** 2
                end_idx = (i + 1) * H ** 2
                times_data[:, start_idx:end_idx, :i + 1] = data[:, :, band - i - 1:]
                times_data[:, start_idx:end_idx, i + 1:] = data[:, :, :band - i - 1]

                start_idx = (nn + i + 1) * H ** 2
                end_idx = (nn + i + 2) * H ** 2
                times_data[:, start_idx:end_idx, :band - i - 1] = data[:, :, i + 1:]
                times_data[:, start_idx:end_idx, band - i - 1:] = data[:, :, :i + 1]
        else:
            for i in range(nn):
                times_data[:, i:(i + 1), :(nn - i)] = data[:, 0:1, band - nn + i:]
                times_data[:, i:(i + 1), (nn - i):] = data[:, 0:1, :band - nn + i]

                times_data[:, nn + 1 + i:nn + 2 + i, band - i - 1:] = data[:, 0:1, :i + 1]
                times_data[:, nn + 1 + i:nn + 2 + i, :band - i - 1] = data[:, 0:1, i + 1:]

        return rearrange(times_data, 'b n c -> b c n')

    def forward(self, x, mask=None):
        x = self.prepare(x)
        # 将每个补丁向量嵌入到嵌入大小：[batch, patch_num, embedding_size]
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape

        # 添加位置嵌入
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)  # [b,1,dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [b,n+1,dim]
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        # transformer: x[b,n + 1,dim] -> x[b,n + 1,dim]
        x = self.transformer(x, mask)

        # 分类：使用 cls_token 输出
        x = self.to_latent(x[:, 0])

        # MLP 分类层
        return self.mlp_head(x)


if __name__ == "__main__":
    torch.manual_seed(42)

    # 设置模型参数
    in_chans = 30  # 输入通道数
    num_classes = 21  # 类别数量
    patch = 5  # patch大小

    # 创建模型
    model = SFT(input_channels=in_chans, num_classes=num_classes)

    # 设置设备
    device = get_device()
    print(f"Using device: {device}")
    model = model.to(device)

    # 测试不同输入尺寸
    test_sizes = [(patch, patch)]

    for size in test_sizes:
        H, W = size
        x = torch.randn(4, in_chans, H, W).to(device)
        print(f"Input shape: {x.shape}")

        with torch.no_grad():
            output = model(x)

        print(f"Output shape: {output.shape}")

    # 计算并打印模型参数总数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
