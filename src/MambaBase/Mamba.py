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

        # 添加批量归一化
        self.bn = nn.BatchNorm1d(d_model)

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
        output = output + x

        # 应用批量归一化
        output = self.bn(output.transpose(1, 2)).transpose(1, 2)

        return output


class MambaModel(nn.Module):
    def __init__(self, input_channels, num_classes, d_model=64, d_state=16, d_conv=4, num_layers=6):
        super().__init__()
        self.embed = nn.Linear(input_channels, d_model)
        self.layers = nn.ModuleList([MambaBlock(d_model, d_state, d_conv) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        self.dim = 2

        self.input_bn = nn.BatchNorm2d(input_channels)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.final_bn = nn.BatchNorm1d(d_model)

        self._init_weights()

    def _init_weights(self):
        def _init_weight(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

        # 应用初始化
        self.apply(_init_weight)

        # 特别处理 MambaBlock 中的 dt 参数
        for layer in self.layers:
            if hasattr(layer, 'dt'):
                nn.init.normal_(layer.dt, mean=0.0, std=0.02)

        # 初始化嵌入层
        if isinstance(self.embed, nn.Linear):
            nn.init.xavier_uniform_(self.embed.weight)
            if self.embed.bias is not None:
                nn.init.zeros_(self.embed.bias)

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.input_bn(x)
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B, H*W, C]
        x = self.embed(x)  # [B, H*W, d_model]

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        x = x.transpose(1, 2)  # [B, d_model, H*W]
        x = self.global_avg_pool(x).squeeze(-1)  # [B, d_model]
        x = self.final_bn(x)
        output = self.head(x)  # [B, num_classes]

        return output


if __name__ == '__main__':
    # 实例化模型
    num_classes = 16
    model = MambaModel(200, num_classes)

    # 测试不同输入尺寸
    test_sizes = [(3, 3), (5, 5), (7, 7), (9, 9)]

    for size in test_sizes:
        H, W = size
        x = torch.randn(32, 200, H, W)  # (batch_size, channels, height, width)

        # 前向传播
        output = model(x)

        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")

    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")