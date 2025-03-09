from typing import List, Union, Dict
import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineSchedule(_LRScheduler):
    """
    实现带有预热阶段的余弦学习率调度。

    这个调度器首先线性增加学习率，然后使用余弦衰减。

    Args:
        optimizer (torch.optim.Optimizer): 要调度的优化器。
        warmup_steps (int): 预热阶段的步数。
        t_total (int): 总训练步数。
        cycles (float, optional): 余弦周期的数量。默认为 0.5。
        last_epoch (int, optional): 最后一个 epoch 的索引。默认为 -1。

    Attributes:
        warmup_steps (int): 预热阶段的步数。
        t_total (int): 总训练步数。
        cycles (float): 余弦周期的数量。
        optimizer (torch.optim.Optimizer): 被调度的优化器。
        base_lrs (List[float]): 初始学习率列表。
        last_epoch (int): 最后一个 epoch 的索引。

    """

    def __init__(self, optimizer: 'torch.optim.Optimizer', warmup_steps: int, t_total: int, cycles: float = 0.5,
                 last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        计算当前步骤的学习率。

        Returns:
            List[float]: 当前步骤的学习率列表。
        """
        if self.last_epoch < self.warmup_steps:
            warmup_factor = float(self.last_epoch) / float(max(1, self.warmup_steps))
            return [warmup_factor * lr for lr in self.base_lrs]
        else:
            progress = float(self.last_epoch - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
            cosine_factor = 0.5 * (1.0 + np.cos(np.pi * float(self.cycles) * 2.0 * progress))
            return [lr * cosine_factor for lr in self.base_lrs]

    def step(self, epoch: Union[int, None] = None) -> None:
        """
        更新学习率调度的步骤。

        Args:
            epoch (int or None, optional): 要步进到的 epoch。如果为 None，则使用 last_epoch + 1。默认为 None。
        """
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def state_dict(self) -> Dict[str, any]:
        """
        返回调度器的状态字典。

        Returns:
            Dict[str, any]: 包含调度器当前状态的字典。
        """
        return {
            'warmup_steps': self.warmup_steps,
            't_total': self.t_total,
            'cycles': self.cycles,
            'base_lrs': self.base_lrs,
            'last_epoch': self.last_epoch
        }

    def load_state_dict(self, state_dict: Dict[str, any]) -> None:
        """
        从状态字典加载调度器的状态。

        Args:
            state_dict (Dict[str, any]): 包含调度器状态的字典。
        """
        self.warmup_steps = state_dict['warmup_steps']
        self.t_total = state_dict['t_total']
        self.cycles = state_dict['cycles']
        self.base_lrs = state_dict['base_lrs']
        self.last_epoch = state_dict['last_epoch']
        self.step(self.last_epoch + 1)