import torch
from torch import nn
import math
import torch.nn.functional as F
from einops import rearrange, repeat


class SSM(nn.Module):
    """
    状态空间模型 (State Space Model, SSM) 实现。

    这个模块实现了一个基于状态空间模型的神经网络层，用于序列建模任务。

    Args:
        d_model (int): 模型的维度。默认为96。
        d_state (int or str): 状态的维度。如果为"auto"，则自动设置为 d_model/6。默认为4。
        ssm_ratio (int): SSM的扩展比率。默认为2。
        dt_rank (int or str): dt投影的秩。如果为"auto"，则自动设置为 d_model/16。默认为"auto"。
        dt_min (float): dt的最小值。默认为0.001。
        dt_max (float): dt的最大值。默认为0.1。
        dt_init (str): dt初始化方法，可以是"random"或"constant"。默认为"random"。
        dt_scale (float): dt缩放因子。默认为1.0。
        dt_init_floor (float): dt初始化的下限。默认为1e-4。
        bias (bool): 是否在线性层中使用偏置。默认为False。
        gaussian (bool): 是否使用高斯初始化。默认为False。
        **kwargs: 额外的关键字参数。

    Attributes:
        d_model (int): 模型的维度。
        d_state (int): 状态的维度。
        expand (int): 扩展比率。
        d_inner (int): 内部维度。
        dt_rank (int): dt投影的秩。
        in_proj (nn.Linear): 输入投影层。
        x_proj (nn.Linear): x投影层。
        dt_proj (nn.Linear): dt投影层。
        A_log (nn.Parameter): A参数的对数。
        D (nn.Parameter): D参数。
        selective_scan (function): 选择性扫描函数。
        out_norm (nn.LayerNorm): 输出归一化层。
        out_proj (nn.Linear): 输出投影层。
    """

    def __init__(self, d_model=96, d_state=4, ssm_ratio=2, dt_rank="auto",
                 dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0,
                 dt_init_floor=1e-4, bias=False, gaussian=False, **kwargs):
        super().__init__()
        factory_kwargs = {"device": None, "dtype": None}

        # 初始化模型参数
        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # 定义网络层
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.x_proj = nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
        self.dt_proj = self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                                    **factory_kwargs)

        # 初始化A和D参数
        self.A_log = self.A_log_init(self.d_state, self.d_inner)
        self.D = self.D_init(self.d_inner)

        from src.MambaBase.Mamba1 import Mamba1
        from src.Train_and_Eval.device import get_device
        device = get_device()
        if device == 'cuda':
            from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
        elif device == 'mps':
            # TODO： fix mps
            Mamba1 = Mamba1(d_model=self.d_model, d_state=self.d_state, expand=self.expand, d_conv=4, conv_bias=True)
            self.selective_scan = Mamba1.selective_scan
        elif device == 'cpu':
            pass

        # 输出层
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        """
        初始化dt投影层。

        Args:
            dt_rank (int): dt投影的秩。
            d_inner (int): 内部维度。
            dt_scale (float): dt缩放因子。
            dt_init (str): 初始化方法，"random"或"constant"。
            dt_min (float): dt的最小值。
            dt_max (float): dt的最大值。
            dt_init_floor (float): dt初始化的下限。
            **factory_kwargs: 额外的关键字参数。

        Returns:
            nn.Linear: 初始化后的dt投影层。
        """
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        """
        初始化A_log参数。

        Args:
            d_state (int): 状态的维度。
            d_inner (int): 内部维度。
            copies (int): 复制次数。默认为-1。
            device (torch.device): 设备。
            merge (bool): 是否合并复制。默认为True。

        Returns:
            nn.Parameter: 初始化后的A_log参数。
        """
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        """
        初始化D参数。

        Args:
            d_inner (int): 内部维度。
            copies (int): 复制次数。默认为-1。
            device (torch.device): 设备。
            merge (bool): 是否合并复制。默认为True。

        Returns:
            nn.Parameter: 初始化后的D参数。
        """
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        """
        核心前向传播过程。

        Args:
            x (torch.Tensor): 输入张量，形状为 (B, L, d)。

        Returns:
            torch.Tensor: 处理后的张量，形状为 (B, L, d_inner)。
        """
        B, L, d = x.shape
        x = x.permute(0, 2, 1)

        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=L)
        A = -torch.exp(self.A_log.float())
        B = rearrange(B, "(b l) dstate -> b dstate l", l=L).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=L).contiguous()

        y = self.selective_scan(
            x, dt,
            A, B, C, self.D.float(),
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
        )
        y = rearrange(y, "b d l -> b l d")
        y = self.out_norm(y)
        return y

    def forward(self, x: torch.Tensor):
        """
        模型的前向传播。

        Args:
            x (torch.Tensor): 输入张量，形状为 (B, L, d_model)。

        Returns:
            torch.Tensor: 输出张量，形状为 (B, L, d_model)。
        """
        B, L, d = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        y = self.forward_core(x)
        y = y * F.silu(z)
        out = self.out_proj(y)

        return out
