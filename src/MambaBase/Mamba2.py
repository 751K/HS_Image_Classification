import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from src.utils.device import get_device


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
            x = x * F.silu(z)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


# -----------------------------
# Mamba2 定义
# -----------------------------
class Mamba2(nn.Module):
    def __init__(self, d_model: int, d_state: int, headdim: int = 16, chunk_size: int = 8,
                 expand: int = 2, d_conv: int = 4):
        super().__init__()
        self.d_model = d_model
        self.d_inner = expand * d_model
        self.nheads = (expand * d_model) // headdim
        self.d_state = d_state
        self.headdim = headdim
        self.chunk_size = chunk_size
        self.device = get_device()

        # in_proj: 将输入投影到 (z, x, B, C, dt)
        d_in_proj = 2 * self.d_inner + 2 * self.d_state + self.nheads
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=False, device=self.device)

        # 增加卷积层，对 xBC 进行局部卷积处理
        # 输入通道数为 d_inner + 2*d_state，使用分组卷积，每个通道独立
        self.d_conv = d_conv  # 卷积核大小
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner + 2 * self.d_state,
            out_channels=self.d_inner + 2 * self.d_state,
            kernel_size=self.d_conv,
            groups=self.d_inner + 2 * self.d_state,
            padding=self.d_conv - 1,
            device=self.device,
        )

        self.dt_bias = nn.Parameter(torch.zeros(self.nheads, device=self.device))
        self.A_log = nn.Parameter(torch.randn(self.nheads, device=self.device) * 0.1)
        self.D = nn.Parameter(torch.ones(self.nheads, device=self.device))

        self.norm = RMSNorm(self.d_inner, device=self.device)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False, device=self.device)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            u: (batch, seqlen, d_model) 输入，如果序列长度不是 chunk_size 的倍数，会自动填充。
        Return:
            y: (batch, seqlen, d_model) 输出，尺寸与输入相同
        """
        # 保存原始序列长度以便后续裁剪
        original_seq_len = u.shape[1]

        # 检查序列长度是否是 chunk_size 的整数倍
        if original_seq_len % self.chunk_size != 0:
            # 计算需要填充的长度
            pad_len = self.chunk_size - (original_seq_len % self.chunk_size)
            # 使用序列的开头值填充
            if original_seq_len >= 4:
                # 取前5个位置的平均值
                pad_values = torch.mean(u[:, :5], dim=1, keepdim=True)
                pad = pad_values.repeat(1, pad_len, 1)  # 使用平均值重复填充
            else:
                # 序列太短时使用第一个位置的值填充
                pad = u[:, :1].repeat(1, pad_len, 1)
            u = torch.cat([pad, u], dim=1)  # 在序列开头添加填充

        # 1. 输入投影与分割
        # u -> (batch, seqlen, d_model)
        zxbcdt = self.in_proj(u)  # (batch, seqlen, d_in_proj)

        # 分割成三个部分：z, xBC, dt (使用切片代替torch.split减少开销)
        z = zxbcdt[:, :, :self.d_inner]
        xBC = zxbcdt[:, :, self.d_inner:self.d_inner + self.d_inner + 2 * self.d_state]
        dt = zxbcdt[:, :, -self.nheads:]

        # 缓存序列长度，避免重复访问
        seq_len = u.shape[1]
        dt = F.softplus(dt + self.dt_bias)  # (batch, seqlen, nheads)

        # 2. 卷积操作处理 xBC：
        # 先将 xBC 转置到 (batch, channels, seqlen)
        xBC_t = xBC.transpose(1, 2)
        # 经过 conv1d 卷积，再转回 (batch, seqlen, channels) 并截断到原序列长度
        xBC = self.conv1d(xBC_t).transpose(1, 2)[:, :seq_len, :]
        # 然后应用激活函数 SiLU
        xBC = F.silu(xBC)  # (batch, seqlen, d_inner + 2*d_state)

        # 3. 分割 xBC 得到 x, B, C (使用切片代替torch.split)
        x = xBC[:, :, :self.d_inner]
        B = xBC[:, :, self.d_inner:self.d_inner + self.d_state]
        C = xBC[:, :, -self.d_state:]

        # 将 x 重排为 (batch, seqlen, nheads, headdim)，使用view更高效
        x = x.view(x.shape[0], seq_len, self.nheads, self.headdim)

        # 4. 调用 SSD 模块
        # 提前计算 A 值
        A = -torch.exp(self.A_log)  # (nheads,)

        # 提前计算 A*dt 和 x*dt 减少计算量
        A_dt = A.unsqueeze(0).unsqueeze(0) * dt
        x_dt = x * dt.unsqueeze(-1)

        # 重排 B 和 C，使用view代替rearrange
        B_reshaped = B.unsqueeze(2)  # (b l 1 n)
        C_reshaped = C.unsqueeze(2)  # (b l 1 n)

        y = ssd(
            x=x_dt,
            A=A_dt,
            B=B_reshaped,
            C=C_reshaped,
            chunk_size=self.chunk_size,
            device=self.device,
        )

        # 5. 残差融合：将 SSD 输出与 x 按 head 权重加权融合
        # 提前展开 D 以减少广播操作
        D_expanded = self.D.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        y = y + x * D_expanded

        # 重排回 (batch, seqlen, d_inner)，使用reshape代替rearrange
        y = y.reshape(y.shape[0], seq_len, self.d_inner)

        # 6. 归一化与线性投影
        y = self.norm(y, z)
        y = self.out_proj(y)

        # 如果进行了填充，则裁剪掉开头的填充部分
        if original_seq_len != seq_len:
            pad_len = seq_len - original_seq_len
            y = y[:, pad_len:, :]

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
# 测试代码：测试 Mamba2 模型
# -----------------------------
if __name__ == "__main__":
    device = get_device()
    print(device)
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
