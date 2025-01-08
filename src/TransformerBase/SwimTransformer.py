import torch
import torch.nn as nn


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

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class SwimTransformer(nn.Module):
    def __init__(self, input_channels, num_classes, embed_dim=64, depth=6, num_heads=8, mlp_ratio=4., qkv_bias=False,
                 dropout=0.1):
        super().__init__()
        self.dim = 2
        self.patch_embed = nn.Linear(input_channels, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, 25, embed_dim))  # 5x5=25
        self.blocks = nn.ModuleList([
            SwimTransformerBlock(embed_dim, num_heads, mlp_ratio, qkv_bias, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, 25, C)
        x = self.patch_embed(x)  # (B, 25, embed_dim)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = self.head(x)  # (B, 25, num_classes)
        return x.transpose(1, 2).reshape(B, -1, H, W)  # (B, num_classes, 5, 5)


if __name__ == '__main__':
    num_classes = 10
    model = SwimTransformer(200,20)

    # 创建一个示例输入
    x = torch.randn(32, 200, 5, 5)  # (batch_size, channels, height, width)

    # 前向传播
    output = model(x)

    print(output.shape)  # 应该输出 torch.Size([32, 5, 5, 10])
