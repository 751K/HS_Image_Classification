import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from Train_and_Eval.device import get_device


# -----------------------------
# RMSNorm 定义
# -----------------------------
class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5, device=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d, device=device))

    def forward(self, x, z=None):
        if z is not None:
            x = x * silu(z)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def silu(x):
    return x * F.sigmoid(x)


# -----------------------------
# Mamba2 定义（加入卷积操作版本）
# -----------------------------
class Mamba2(nn.Module):
    def __init__(self, d_model: int, d_state: int, headdim: int, chunk_size: int = 2, expand: int = 2,
                 device: str = 'mps', d_conv: int = 4):
        super().__init__()
        self.d_model = d_model
        self.d_inner = expand * d_model
        self.nheads = (expand * d_model) // headdim
        self.d_state = d_state
        self.headdim = headdim
        self.chunk_size = chunk_size
        self.device = device

        # in_proj: 将输入投影到 (z, x, B, C, dt)
        d_in_proj = 2 * self.d_inner + 2 * self.d_state + self.nheads
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=False, device=device)

        # 增加卷积层，对 xBC 进行局部卷积处理
        # 输入通道数为 d_inner + 2*d_state，使用分组卷积，每个通道独立
        self.d_conv = d_conv  # 卷积核大小
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner + 2 * self.d_state,
            out_channels=self.d_inner + 2 * self.d_state,
            kernel_size=self.d_conv,
            groups=self.d_inner + 2 * self.d_state,
            padding=self.d_conv - 1,
            device=device,
        )

        self.dt_bias = nn.Parameter(torch.empty(self.nheads, device=device))
        self.A_log = nn.Parameter(torch.empty(self.nheads, device=device))
        self.D = nn.Parameter(torch.empty(self.nheads, device=device))
        self.norm = RMSNorm(self.d_inner, device=device)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False, device=device)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            u: (batch, seqlen, d_model) 输入，要求序列长度为 chunk_size 的倍数。
        Return:
            y: (batch, seqlen, d_model) 输出
        """
        # 1. 输入投影与分割
        # u -> (batch, seqlen, d_model)
        zxbcdt = self.in_proj(u)  # (batch, seqlen, d_in_proj)
        # 分割成三个部分：z, xBC, dt
        z, xBC, dt = torch.split(zxbcdt, [self.d_inner, self.d_inner + 2 * self.d_state, self.nheads], dim=-1)
        dt = F.softplus(dt + self.dt_bias)  # (batch, seqlen, nheads)

        # 2. 卷积操作处理 xBC：
        # 先将 xBC 转置到 (batch, channels, seqlen)
        xBC = xBC.transpose(1, 2)
        # 经过 conv1d 卷积，再转回 (batch, seqlen, channels) 并截断到原序列长度
        xBC = self.conv1d(xBC).transpose(1, 2)[:, :u.shape[1], :]
        # 然后应用激活函数 SiLU
        xBC = silu(xBC)  # (batch, seqlen, d_inner + 2*d_state)

        # 3. 分割 xBC 得到 x, B, C
        x, B, C = torch.split(xBC, [self.d_inner, self.d_state, self.d_state], dim=-1)
        # 将 x 重排为 (batch, seqlen, nheads, headdim) ，满足 d_inner = nheads * headdim
        x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)

        # 4. 调用 SSD 模块
        A = -torch.exp(self.A_log)  # (nheads,)
        y = ssd(
            x=x * dt.unsqueeze(-1),
            A=A * dt,
            B=rearrange(B, "b l n -> b l 1 n"),
            C=rearrange(C, "b l n -> b l 1 n"),
            chunk_size=self.chunk_size,
            device=self.device,
        )

        # 5. 残差融合：将 SSD 输出与 x 按 head 权重加权融合
        y = y + x * self.D.unsqueeze(-1)
        # 重排回 (batch, seqlen, d_inner)
        y = rearrange(y, "b l h p -> b l (h p)")
        # 6. 归一化与线性投影
        y = self.norm(y, z)
        y = self.out_proj(y)
        return y


# 构造下三角矩阵，用于 SSD 模块计算
def segmeng_sum(x: torch.Tensor, device=None) -> torch.Tensor:
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def ssd(x, A, B, C, chunk_size, initial_states=None, device=None):
    # 确保序列长度可以被块大小整除
    assert x.shape[1] % chunk_size == 0

    # 将输入 x、A、B、C 按照 chunk_size 划分为块
    x, A, B, C = [rearrange(m, "b (c l) ... -> b c l ...", l=chunk_size) for m in (x, A, B, C)]
    A = rearrange(A, "b c l h -> b h c l")

    # 对 A 沿着块内序列长度维度（l）做累加，得到每个块内的 A 累加和
    A_cumsum = torch.cumsum(A, dim=-1)

    # 利用 segsum 计算下三角的分段累加矩阵，然后取指数
    # 这个矩阵 L 用于对块内的 A 进行加权累加
    L = torch.exp(segmeng_sum(A, device=device))

    # 计算块内的输出 Y_diag：利用爱因斯坦求和对 C、B、L 和 x 进行融合
    Y_diag = torch.einsum("bclhn, bcshn, bhcls, bcshp -> bclhp", C, B, L, x)

    # 计算衰减状态：利用 A 的累加和计算每个块内每个位置的衰减系数
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)

    # 计算块内状态：利用 B、衰减状态和 x 计算状态表示
    states = torch.einsum("bclhn, bhcl, bclhp -> bchpn", B, decay_states, x)

    # 如果没有初始状态，则初始化为与状态相同形状的零张量（只取第一块状态）
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])

    # 将初始状态和当前块状态拼接，用于后续块间状态的递归更新
    states = torch.cat([initial_states, states], dim=1)

    # 计算跨块衰减矩阵
    # 首先对 A_cumsum 最后一维（chunk 内最后一个位置）进行 pad，然后计算 segsum，再取指数
    decay_chunk = torch.exp(segmeng_sum(torch.nn.functional.pad(A_cumsum[:, :, :, -1], (1, 0)), device=device))

    # 利用跨块衰减矩阵和状态计算新的状态（跨块递归更新）
    new_states = torch.einsum("bhzc, bchpn -> bzhpn", decay_chunk, states)

    # 分离出各块的状态和最终状态：前面的块为更新后的状态
    states = new_states[:, :-1]

    # 计算状态到输出的转换矩阵：对 A_cumsum 取指数
    state_decay_out = torch.exp(A_cumsum)

    # 计算块外输出 Y_off：利用 C、更新后的状态和转换矩阵进行爱因斯坦求和
    Y_off = torch.einsum("bclhn, bchpn, bhcl -> bclhp", C, states, state_decay_out)

    # 将块内输出 Y_diag 与块间输出 Y_off 相加，再重排回原始序列维度
    Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")

    # 返回 SSD 模块的输出 Y
    return Y


# -----------------------------
# 测试代码：测试 Mamba2 模型（重新加入卷积操作）
# -----------------------------
if __name__ == "__main__":

    device = get_device()
    # 参数设置：
    # d_model：模型主维度，表示输入嵌入和输出表示的维度（512）。
    d_model = 512
    # d_state：状态空间模型中的状态维度（64）。
    d_state = 64
    # expand：扩展因子，用于计算内部隐藏维度 d_inner = expand * d_model
    expand = 2
    # headdim：每个 head 的维度（64），head 数量 = d_inner / headdim。
    headdim = 64
    # chunk_size：序列分块长度（8），用于 SSD 模块中块级计算。
    chunk_size = 8
    # 确保 d_inner 能被 headdim 整除
    assert (expand * d_model) % headdim == 0, "d_inner must be divisible by headdim"

    # 初始化 Mamba2 模型，卷积操作已经重新加回
    model = Mamba2(
        d_model=d_model,
        d_state=d_state,
        headdim=headdim,
        chunk_size=chunk_size,
        expand=expand,
    ).to(device)

    # 构造输入数据，形状为 (batch, seq_len, d_model)
    batch_size = 2
    seq_len = 16  # 必须是 chunk_size 的倍数
    input_tensor = torch.randn(batch_size, seq_len, d_model, device=device)
    print("Input shape:", input_tensor.shape)

    # 执行前向传播测试
    output = model(input_tensor)
    print("Output shape:", output.shape)
