import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, einsum

from src.utils.device import get_device


# -----------------------------
# RMSNorm 定义
# -----------------------------
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


# -----------------------------
# MambaBlock 定义（完整版本，保留 selective_scan 部分）
# -----------------------------
class Mamba1(nn.Module):
    def __init__(
            self,
            d_model: int,
            d_state: int,
            expand: int,
            d_conv: int = 4,
            conv_bias: bool = True,
            bias: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.dt_rank = math.ceil(d_model / 16)
        self.d_conv = d_conv
        self.conv_bias = conv_bias
        self.bias = bias

        # 计算内部隐藏维度 d_inner = expand * d_model
        self.d_inner = int(expand * d_model)

        # in_proj 将输入投影到 2*d_inner 维（包含 x 部分和残差部分）
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=bias)

        # 分组卷积，对 x 部分进行局部处理
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=conv_bias,
        )

        # x_proj 将卷积后的输出映射到 (dt_rank + 2*d_state) 维，
        # dt_rank 为输入相关步长参数的秩，2*d_state 用于辅助因子 B 和 C（各为 d_state）
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)

        # dt_proj 将 delta 部分映射回 d_inner 维
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # 初始化 A_log 参数：形状 (d_inner, d_state)
        A = torch.arange(1, d_state + 1).float()  # shape (d_state,)
        A = A.unsqueeze(0).repeat(self.d_inner, 1)  # shape (d_inner, d_state)
        self.A_log = nn.Parameter(torch.log(A))

        # 参数 D：形状 (d_inner,)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # out_proj 将输出映射回 d_model 维
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量，形状 (b, l, d_model)
        Returns:
            output: 输出张量，形状 (b, l, d_model)
        """
        b, l, _ = x.shape

        # 1. 输入投影，得到 (b, l, 2*d_inner)
        x_and_res = self.in_proj(x)
        # 分割为 x_part 和残差 res，每个形状 (b, l, d_inner)
        x_part, res = x_and_res.split(self.d_inner, dim=-1)

        # 2. 卷积处理：先转置为 (b, d_inner, l)，经过卷积后截断到长度 l，再转回 (b, l, d_inner)
        x_conv = self.conv1d(x_part.transpose(1, 2))[:, :, :l]
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)

        # 3. x_proj 映射：得到融合张量 x_dbl，形状 (b, l, dt_rank + 2*d_state)
        x_dbl = self.x_proj(x_conv)
        # 拆分为 delta, B, C：
        # delta: (b, l, dt_rank), B: (b, l, d_state), C: (b, l, d_state)
        delta, B, C = x_dbl.split([self.dt_proj.in_features, self.A_log.shape[1], self.A_log.shape[1]], dim=-1)
        # 对 delta 进行激活映射，输出形状 (b, l, d_inner)
        delta = F.softplus(self.dt_proj(delta))

        # 4. 调用 selective_scan 进行状态空间计算，得到 y (b, l, d_inner)
        y = self.selective_scan(x_conv, delta, -torch.exp(self.A_log.float()), B, C, self.D.float())

        # 5. 融合残差：使用 SiLU 激活门控 residual 部分
        y = y * F.silu(res)

        # 6. 映射回 d_model 维度
        output = self.out_proj(y)
        return output

    @staticmethod
    def selective_scan(u: torch.Tensor, delta: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
                       D: torch.Tensor) -> torch.Tensor:
        """
        执行选择性扫描（selective scan），模拟状态空间模型的递归更新
        Args:
            u: 输入张量，形状 (b, l, d_in)
            delta: 步长参数，形状 (b, l, d_in)
            A: 状态空间参数 A，形状 (d_in, n)
            B: 辅助因子 B，形状 (b, l, n)
            C: 辅助因子 C，形状 (b, l, n)
            D: 参数 D，形状 (d_in,)
        Returns:
            y: 输出张量，形状 (b, l, d_in)
        """
        b, l, d_in = u.shape
        n = A.shape[1]

        # 离散化连续参数：
        # A 使用零阶保持（ZOH）离散化，B 使用简化的 Euler 离散化
        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')

        # 选择性扫描：依次更新状态
        x_state = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []
        for i in range(l):
            x_state = deltaA[:, i] * x_state + deltaB_u[:, i]
            y_i = einsum(x_state, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y_i)
        y = torch.stack(ys, dim=1)  # 形状 (b, l, d_in)

        # 加入直接传递的 u 部分，通过 D 调整
        y = y + u * D
        return y


# -----------------------------
# 测试代码：直接测试 MambaBlock
# -----------------------------
if __name__ == "__main__":
    device = get_device()
    print(device)
    # 参数设置：
    # d_model: 输入输出的隐藏维度 (512)
    # d_state: 状态空间模型中的状态维度 (64)
    # expand: 扩展因子，计算 d_inner = expand * d_model (2，即 d_inner = 1024)
    # dt_rank: 输入相关步长参数的秩，通常设为 ceil(d_model/16) (例如 32)
    # d_conv: 卷积核大小 (4)
    # conv_bias, bias: 是否使用偏置 (例如 conv_bias=True, bias=False)
    d_model = 512
    d_state = 64
    expand = 2
    d_conv = 4
    conv_bias = True
    bias = False

    # 初始化 MambaBlock，直接传入各参数
    block = Mamba1(
        d_model=d_model,
        d_state=d_state,
        expand=expand,
        d_conv=d_conv,
        conv_bias=conv_bias,
        bias=bias,
    ).to(device)

    # 构造输入数据，形状为 (batch, seq_len, d_model)
    batch_size = 2
    seq_len = 16
    input_tensor = torch.randn(batch_size, seq_len, d_model, device=device)
    print("Input shape:", input_tensor.shape)

    # 执行前向传播测试
    output = block(input_tensor)
    print("Output shape:", output.shape)
