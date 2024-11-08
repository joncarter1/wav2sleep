import logging

import lightning
import torch

logger = logging.getLogger(__name__)


class ResettableEarlyStopping(lightning.pytorch.callbacks.EarlyStopping):
    """Resettable Early Stopping.

    Useful for fine-tuning after restoring a checkpoint that was originally
    trained with early stopping.
    """

    def __init__(self, *args, reset: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset = reset

    def on_fit_start(self, trainer, pl_module):
        if self.reset:
            self.wait_count = 0
            self.stopped_epoch = 0
            self.best_score = torch.tensor(torch.inf) if self.monitor_op == torch.lt else -torch.tensor(torch.inf)
        super().on_fit_start(trainer, pl_module)
