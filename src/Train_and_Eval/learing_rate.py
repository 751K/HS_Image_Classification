from typing import List, Union

import numpy as np
import torch


class WarmupCosineSchedule:
    """
    实现带有预热期的余弦退火学习率调度。

    This class adjusts the learning rate over time:
    - During the warmup period, the learning rate increases linearly.
    - After warmup, the learning rate follows a cosine decay schedule.
    """

    def __init__(self, optimizer: 'torch.optim.Optimizer', warmup_steps: int, t_total: int, cycles: float = 0.5,
                 last_epoch: int = -1):
        """
        初始化 WarmupCosineSchedule。

        Args:
            optimizer (torch.optim.Optimizer): 优化器实例。
            warmup_steps (int): 预热步数。
            t_total (int): 总训练步数。
            cycles (float, optional): 余弦周期的数量。默认为 0.5。
            last_epoch (int, optional): 上一轮的 epoch。默认为 -1。

        """
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.step(self.last_epoch + 1)

    def get_lr(self) -> List[float]:
        """
        计算当前步骤的学习率。

        Returns:
            List[float]: 每个参数组的新学习率列表。
        """
        if self.last_epoch < self.warmup_steps:
            return [(float(self.last_epoch) / float(max(1, self.warmup_steps))) * lr for lr in self.base_lrs]
        progress = float(self.last_epoch - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return [lr * (0.5 * (1. + np.cos(np.pi * float(self.cycles) * 2.0 * progress))) for lr in self.base_lrs]

    def step(self, epoch: Union[int, None] = None) -> None:
        """
        更新学习率。

        Args:
            epoch (int or None, optional): 当前的 epoch。如果为 None，则使用 last_epoch + 1。

        """
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
