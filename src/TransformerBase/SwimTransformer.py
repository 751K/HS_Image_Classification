import torch
import torch.nn as nn
import torch.nn.functional as F


class SwimTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
        self.num_heads = num_heads

    def forward(self, x):
        B, N, C = x.shape
        # 确保序列长度能被头数整除
        pad_len = (self.num_heads - (N % self.num_heads)) % self.num_heads
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))

        x_norm = self.norm1(x)
        x = x + self.attn(x_norm, x_norm, x_norm)[0]
        if pad_len > 0:
            x = x[:, :N, :]
        x = x + self.mlp(self.norm2(x))
        return x


class SwimTransformer(nn.Module):
    def __init__(self, input_channels, num_classes, embed_dim=64, depth=6, num_heads=8, mlp_ratio=4.,
                 qkv_bias=False, dropout=0.1, patch_size=3):
        super().__init__()
        self.dim = 2
        self.patch_embed = nn.Conv2d(input_channels, embed_dim, kernel_size=1, stride=1)
        self.blocks = nn.ModuleList([
            SwimTransformerBlock(embed_dim, num_heads, mlp_ratio, qkv_bias, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.dim = 2

    def forward(self, x):
        B, C, H, W = x.shape

        # Patch嵌入
        x = self.patch_embed(x)  # (B, embed_dim, H, W)

        # 重塑并转置用于transformer块
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, embed_dim)

        # 应用transformer块
        for block in self.blocks:
            x = block(x)

        # 应用最终的归一化
        x = self.norm(x)

        # 提取中心像素的嵌入
        center_idx = (H * W) // 2
        x = x[:, center_idx, :]  # (B, embed_dim)

        # 应用分类头
        x = self.head(x)  # (B, num_classes)

        return x


# 测试代码
if __name__ == "__main__":
    # 初始化模型
    input_channels = 3
    num_classes = 10
    model = SwimTransformer(input_channels, num_classes)

    # 测试不同尺寸的输入
    batch_size = 64
    for patch_size in [3, 5]:
        x = torch.randn(batch_size, input_channels, patch_size, patch_size)
        output = model(x)
        print(f"输入形状: {x.shape}")
        print(f"输出形状: {output.shape}")
        print(f"{patch_size}x{patch_size} patch测试通过")
        print()
