import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state, d_conv):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Δ, B, C 矩阵
        self.dt = nn.Parameter(torch.randn(d_model, d_state))
        self.A = nn.Linear(d_model, d_state, bias=False)
        self.B = nn.Linear(d_model, d_state, bias=False)
        self.C = nn.Linear(d_state, d_model, bias=False)

        # 卷积层
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=d_conv, padding=d_conv - 1, groups=d_model)

        # 投影层
        self.in_proj = nn.Linear(d_model, 2 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, D = x.shape

        x_and_res = self.in_proj(x)  # [B, L, 2D]
        x, res = x_and_res.chunk(2, dim=-1)  # 两个 [B, L, D]

        x = x.permute(0, 2, 1)  # [B, D, L]
        x = self.conv(x)[:, :, :L]
        x = x.permute(0, 2, 1)  # [B, L, D]

        # 应用 SSM
        dA = torch.exp(torch.einsum('bld,de->ble', x, self.dt))  # [B, L, d_state]
        dB = self.B(x)  # [B, L, d_state]

        # 使用累积乘积来替代循环
        dA_cumprod = torch.cumprod(dA, dim=1)
        y = torch.cumsum(dA_cumprod * dB, dim=1)
        y = torch.cat([dB[:, 0:1], y[:, 1:] - y[:, :-1] * dA[:, 1:]], dim=1)

        y = self.C(y)  # [B, L, D]

        # 残差连接和输出投影
        output = self.out_proj(y * F.silu(res))
        return output + x


class MambaModel(nn.Module):
    def __init__(self, input_channels, num_classes,d_model=64, d_state=16, d_conv=4, num_layers=6):
        super().__init__()
        self.embed = nn.Linear(input_channels, d_model)
        self.layers = nn.ModuleList([MambaBlock(d_model, d_state, d_conv) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        self.dim = 2

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B, 25, C]

        x = self.embed(x)  # [B, 25, d_model]

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        x = self.head(x)  # [B, 25, num_classes]

        return x.transpose(1, 2).reshape(B, -1, H, W)  # [B, num_classes, 5, 5]


if __name__ == '__main__':
    # 实例化模型
    num_classes = 16
    model = MambaModel(200, num_classes)

    # 创建一个示例输入
    x = torch.randn(32, 200, 5, 5)  # (batch_size, channels, height, width)

    # 前向传播
    output = model(x)

    print(output.shape)  # 应该输出 torch.Size([32, 10, 5, 5])