import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim):
        super().__init__()
        self.proj = nn.Linear(in_channels, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        return self.proj(x)  # (B, H*W, embed_dim)


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4., qkv_bias=False, dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, input_channels, num_classes, embed_dim=64, depth=6, num_heads=8, mlp_ratio=4., qkv_bias=False,
                 dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(input_channels, embed_dim)
        self.dim = 2
        self.pos_embed = nn.Parameter(torch.zeros(1, 1000, embed_dim))  # 支持最大1000个像素
        self.blocks = nn.ModuleList([
            TransformerEncoder(embed_dim, num_heads, mlp_ratio, qkv_bias, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)  # (B, H*W, embed_dim)

        # 添加位置编码
        x = x + self.pos_embed[:, :H * W, :]

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # 只取中心位置的输出
        center_idx = H * W // 2
        x = x[:, center_idx, :]  # (B, embed_dim)

        x = self.head(x)  # (B, num_classes)
        return x


if __name__ == "__main__":
    # 使用示例
    model = VisionTransformer(input_channels=3, num_classes=10, embed_dim=64, depth=6, num_heads=8)

    # 测试不同输入尺寸
    test_inputs = [
        torch.randn(1, 3, 3, 3),
        torch.randn(1, 3, 5, 5),
        torch.randn(1, 3, 7, 7)
    ]

    for input_tensor in test_inputs:
        output = model(input_tensor)
        print(f"Input shape: {input_tensor.shape}, Output shape: {output.shape}")
