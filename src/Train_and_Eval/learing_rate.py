import numpy as np


class WarmupCosineSchedule:
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.step(self.last_epoch + 1)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [(float(self.last_epoch) / float(max(1, self.warmup_steps))) * lr for lr in self.base_lrs]
        progress = float(self.last_epoch - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return [lr * (0.5 * (1. + np.cos(np.pi * float(self.cycles) * 2.0 * progress))) for lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr