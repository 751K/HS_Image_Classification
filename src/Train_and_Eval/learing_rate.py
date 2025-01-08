from typing import List, Union, Dict
import numpy as np
import torch


class WarmupCosineSchedule:
    def __init__(self, optimizer: 'torch.optim.Optimizer', warmup_steps: int, t_total: int, cycles: float = 0.5,
                 last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        self.optimizer = optimizer
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.last_epoch = last_epoch
        self.step(self.last_epoch + 1)

    def get_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_steps:
            return [(float(self.last_epoch) / float(max(1, self.warmup_steps))) * lr for lr in self.base_lrs]
        progress = float(self.last_epoch - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return [lr * (0.5 * (1. + np.cos(np.pi * float(self.cycles) * 2.0 * progress))) for lr in self.base_lrs]

    def step(self, epoch: Union[int, None] = None) -> None:
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def state_dict(self) -> Dict[str, any]:
        """返回调度器的状态字典"""
        return {
            'warmup_steps': self.warmup_steps,
            't_total': self.t_total,
            'cycles': self.cycles,
            'base_lrs': self.base_lrs,
            'last_epoch': self.last_epoch
        }

    def load_state_dict(self, state_dict: Dict[str, any]) -> None:
        """从状态字典加载调度器的状态"""
        self.warmup_steps = state_dict['warmup_steps']
        self.t_total = state_dict['t_total']
        self.cycles = state_dict['cycles']
        self.base_lrs = state_dict['base_lrs']
        self.last_epoch = state_dict['last_epoch']
        self.step(self.last_epoch + 1)
