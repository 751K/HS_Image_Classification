from typing import List, Union, Dict, Any
import math
import torch
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineSchedule(_LRScheduler):
    """
    实现带有预热阶段的余弦学习率调度。

    首先线性增加学习率，然后使用余弦衰减。
    """

    def __init__(self, optimizer: 'torch.optim.Optimizer', warmup_steps: int, t_total: int,
                 cycles: float = 0.5, min_lr: float = 0.0, last_epoch: int = -1):
        # 参数验证
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps必须为非负数，但得到{warmup_steps}")
        if t_total <= 0:
            raise ValueError(f"t_total必须为正数，但得到{t_total}")

        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """计算当前步骤的学习率"""
        if self.last_epoch < self.warmup_steps:
            # 预热阶段
            warmup_factor = self.last_epoch / max(1.0, self.warmup_steps)
            return [warmup_factor * lr for lr in self.base_lrs]
        else:
            # 余弦衰减阶段
            progress = (self.last_epoch - self.warmup_steps) / max(1.0, self.t_total - self.warmup_steps)
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * self.cycles * 2.0 * progress))
            return [max(self.min_lr, lr * cosine_factor) for lr in self.base_lrs]

    def state_dict(self) -> Dict[str, Any]:
        """返回调度器的状态字典"""
        return {
            'warmup_steps': self.warmup_steps,
            't_total': self.t_total,
            'cycles': self.cycles,
            'min_lr': self.min_lr,
            'base_lrs': self.base_lrs,
            'last_epoch': self.last_epoch
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """从状态字典加载调度器的状态"""
        self.warmup_steps = state_dict['warmup_steps']
        self.t_total = state_dict['t_total']
        self.cycles = state_dict['cycles']
        self.base_lrs = state_dict['base_lrs']
        self.last_epoch = state_dict['last_epoch']
        self.min_lr = state_dict.get('min_lr', 0.0)  # 为了向后兼容
