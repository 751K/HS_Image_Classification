import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim):
        super().__init__()
        self.proj = nn.Linear(in_channels, embed_dim)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).flatten(1, 2)  # (B, 25, in_channels)
        return self.proj(x)  # (B, 25, embed_dim)


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
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 26, embed_dim))  # 25 + 1 for cls token
        self.blocks = nn.ModuleList([
            TransformerEncoder(embed_dim, num_heads, mlp_ratio, qkv_bias, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)  # (B, 25, embed_dim)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, 26, embed_dim)
        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        x = self.head(x[:, 1:])  # 不使用 cls token
        return x.transpose(1, 2).reshape(B, -1, H, W)  # (B, num_classes, 5, 5)


if __name__ == '__main__':
    # 实例化模型
    num_classes = 10
    model = VisionTransformer(200, num_classes)

    # 创建一个示例输入
    x = torch.randn(32, 200, 5, 5)  # (batch_size, channels, height, width)

    # 前向传播
    output = model(x)

    print(output.shape)  # 应该输出 torch.Size([32, 10, 5, 5])
