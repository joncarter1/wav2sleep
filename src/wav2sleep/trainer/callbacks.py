import logging
import sys
from typing import Any

import lightning
import torch
from lightning.pytorch.callbacks.progress.tqdm_progress import TQDMProgressBar, Tqdm

logger = logging.getLogger(__name__)


class EMACallback(lightning.pytorch.callbacks.Callback):
    """Exponential Moving Average (EMA) callback for model weights.

    Maintains an exponentially weighted moving average of model parameters.
    During validation and testing, the EMA weights are used instead of
    the current training weights, which typically improves generalization.

    The EMA update formula is:
        ema_params = decay * ema_params + (1 - decay) * model_params

    Args:
        decay: EMA decay rate. Higher values give more weight to past parameters.
            Typical values: 0.999 (faster adaptation) to 0.9999 (slower, smoother).
        start_step: Step at which to begin EMA updates. Set > 0 to skip warmup
            period where weights change rapidly.
        device: Device for EMA model. Use 'cpu' to save GPU memory, or None
            to use the same device as the model.
    """

    def __init__(self, decay: float = 0.9999, start_step: int = 0, device: str | None = None):
        super().__init__()
        if not 0.0 <= decay <= 1.0:
            raise ValueError(f'decay must be in [0, 1], got {decay}')
        self.decay = decay
        self.start_step = start_step
        self.device = device

        self._ema_state_dict: dict[str, torch.Tensor] | None = None
        self._original_state_dict: dict[str, torch.Tensor] | None = None
        self._step_count: int = 0

    def setup(self, trainer: lightning.Trainer, pl_module: lightning.LightningModule, stage: str) -> None:
        """Initialize EMA state dict from the model."""
        if stage == 'fit' and self._ema_state_dict is None:
            self._ema_state_dict = {
                k: v.clone().to(self.device if self.device else v.device) for k, v in pl_module.state_dict().items()
            }

    def _should_update(self) -> bool:
        """Determine if EMA should be updated at the current step."""
        return self._step_count >= self.start_step

    @torch.no_grad()
    def _update_ema(self, pl_module: lightning.LightningModule) -> None:
        """Update EMA parameters with current model parameters."""
        if self._ema_state_dict is None:
            return

        model_state = pl_module.state_dict()
        for key, ema_param in self._ema_state_dict.items():
            if key in model_state:
                model_param = model_state[key]
                if ema_param.device != model_param.device:
                    model_param = model_param.to(ema_param.device)
                ema_param.mul_(self.decay).add_(model_param, alpha=1 - self.decay)

    def on_train_batch_end(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Update EMA after each training batch."""
        self._step_count += 1
        if self._should_update():
            self._update_ema(pl_module)

    def _swap_to_ema(self, pl_module: lightning.LightningModule) -> None:
        """Swap model parameters with EMA parameters for evaluation."""
        if self._ema_state_dict is None:
            return
        self._original_state_dict = {k: v.clone() for k, v in pl_module.state_dict().items()}
        ema_state = {k: v.to(pl_module.device) for k, v in self._ema_state_dict.items()}
        pl_module.load_state_dict(ema_state)

    def _swap_to_original(self, pl_module: lightning.LightningModule) -> None:
        """Restore original model parameters after evaluation."""
        if self._original_state_dict is None:
            return
        pl_module.load_state_dict(self._original_state_dict)
        self._original_state_dict = None

    def on_validation_epoch_start(self, trainer: lightning.Trainer, pl_module: lightning.LightningModule) -> None:
        """Swap to EMA weights before validation."""
        self._swap_to_ema(pl_module)

    def on_validation_epoch_end(self, trainer: lightning.Trainer, pl_module: lightning.LightningModule) -> None:
        """Restore original weights after validation."""
        self._swap_to_original(pl_module)

    def on_test_epoch_start(self, trainer: lightning.Trainer, pl_module: lightning.LightningModule) -> None:
        """Swap to EMA weights before testing."""
        self._swap_to_ema(pl_module)

    def on_test_epoch_end(self, trainer: lightning.Trainer, pl_module: lightning.LightningModule) -> None:
        """Restore original weights after testing."""
        self._swap_to_original(pl_module)

    def on_train_end(self, trainer: lightning.Trainer, pl_module: lightning.LightningModule) -> None:
        """Replace model parameters with EMA at the end of training."""
        if self._ema_state_dict is not None:
            ema_state = {k: v.to(pl_module.device) for k, v in self._ema_state_dict.items()}
            pl_module.load_state_dict(ema_state)

    def state_dict(self) -> dict[str, Any]:
        """Return callback state for checkpointing."""
        return {
            'ema_state_dict': self._ema_state_dict,
            'step_count': self._step_count,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load callback state from checkpoint."""
        self._ema_state_dict = state_dict.get('ema_state_dict')
        self._step_count = state_dict.get('step_count', 0)


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


class CustomTQDMProgressBar(TQDMProgressBar):
    """Custom TQDM progress bar with exponential smoothing to track batches/second during training.

    The default progress bar uses smoothing=0, calculating batches/second over all previous iterations.
    """

    def __init__(self, *args, smoothing: float = 0.7, **kwargs):
        super().__init__(*args, **kwargs)
        self.smoothing = smoothing

    def init_train_tqdm(self) -> Tqdm:
        """Overridden with faster smoothing to track variation during training."""
        return Tqdm(
            desc=self.train_description,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=self.smoothing,
            bar_format=self.BAR_FORMAT,
        )
