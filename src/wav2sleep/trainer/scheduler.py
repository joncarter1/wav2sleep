import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class ExpWarmUpScheduler(LRScheduler):
    """Learning rate scheduler that uses a warm-up to a maximum, followed by exponential decay."""

    def __init__(
        self,
        optimizer: Optimizer,
        lr_max: int,
        warmup_steps: int,
        tau: float,
    ) -> None:
        self.lr_max = lr_max
        self.warmup_steps = warmup_steps
        self.tau = tau
        self.num_param_groups = len(optimizer.param_groups)
        super().__init__(optimizer, last_epoch=-1)

    def get_lr(self) -> float:
        # Use last_epoch (checkpointed) rather than _step_count (not checkpointed)
        # so LR schedule is consistent after resuming from checkpoints.
        # last_epoch starts at -1 before the first step, so use step = last_epoch + 1.
        step = self.last_epoch + 1
        if step <= self.warmup_steps:
            lr = self.lr_max * (step / self.warmup_steps)
        else:  # Exponential decay after warmup
            lr = self.lr_max * math.exp(-(step - self.warmup_steps) / self.tau)
        return [lr] * self.num_param_groups
