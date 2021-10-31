import math
import warnings

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class LinearWarmupCosineAnnealingLR(_LRScheduler):
    """
    This custom scheduler is based on torch.optim.lr_scheduler.CosineAnnealingLR
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations per half cycle step.
        warmup_start_lr (float, default=0.0): Starting learning rate of warmup.
        eta_min (float, default=0.0): Minimum learning rate.
        m_mul (float, default=1.0): Used to derive the initial learning rate of the i-th cycle:
        last_epoch (int, default=-1): The index of last epoch.
        verbose (bool, default=False): If True, prints a message to stdout for each update.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        T_max: int,
        warmup_epochs: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        m_mul: float = 1.0,
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        self.T_max = T_max
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        self.m_mul = m_mul
        self.m_mul_cycle = 1.0
        super(LinearWarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, " "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)

        elif self.last_epoch < self.warmup_epochs:
            return [
                group["lr"] + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        elif ((self.last_epoch - self.warmup_epochs) - 1 - self.T_max) % (2 * self.T_max) == 0:
            self.m_mul_cycle *= self.m_mul
            return [
                group["lr"] + (base_lr * self.m_mul_cycle - self.eta_min) * (1 - math.cos(math.pi / self.T_max)) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / self.T_max))
            / (1 + math.cos(math.pi * ((self.last_epoch - self.warmup_epochs) - 1) / self.T_max))
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self):
        return [
            self.eta_min
            + (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / self.T_max))
            / 2
            for base_lr in self.base_lrs
        ]
