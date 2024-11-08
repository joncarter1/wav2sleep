"""PyTorch lightning model classes for sleep transformer models."""

__all__ = ('SleepLightningModel',)
import logging
from collections import defaultdict
from typing import Callable, Iterator, Optional

import lightning
import torch
import torch.distributed as dist
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.trainer.states import RunningStage
from torch.nn import Parameter
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler
from torchmetrics.classification import MulticlassConfusionMatrix

from wav2sleep.log import log_aux_metrics
from wav2sleep.models.ppgnet import SleepPPGNet
from wav2sleep.models.wav2sleep import Wav2Sleep
from wav2sleep.trainer.scheduler import ExpWarmUpScheduler

from ..settings import CCSHS, CFS, CHAT, ECG, MESA, PPG, SHHS, TEST, THX, TRAIN, VAL
from .masker import SignalMasker

logger = logging.getLogger(__name__)


def sum_if_distributed(tensor):
    """Sums the given tensor across all GPUs."""
    if dist.is_initialized():
        dist.barrier()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def confusion_matrix(y_out, y_true, cmat_func: MulticlassConfusionMatrix, from_logits: bool = True):
    """Compute confusion matrix of PyTorch Tensors which may lie on GPU and should be in logit form."""
    # Remove invalid labels (-1) from calculation.
    y_out, y_true = y_out[y_true >= 0], y_true[y_true >= 0]
    y_out = y_out.detach()
    if from_logits:
        y_out = y_out.argmax(dim=-1)
    return cmat_func(y_out, y_true).detach()


class SleepLightningModel(lightning.LightningModule):
    """Wrapper around the underlying model that handles the training process.

    This includes computing the loss function.
    """

    tuning_mode: bool = False  # Patch to avoid logging when performing tuning at start-up.

    def __init__(
        self,
        model: Wav2Sleep | SleepPPGNet,
        criterion,
        optimizer: Callable[[Iterator[Parameter]], Optimizer],  # Func that returns optimizer given params
        aux_metrics=None,
        scheduler: Optional[Callable[[Optimizer], _LRScheduler]] = None,  # Func that returns scheduler given optimizer
        debug_level=2,
        on_step: bool = False,
        on_epoch: bool = True,
        num_classes: int = 4,
        masker: SignalMasker | None = None,
        flip_polarity: bool = True,
    ):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.cmat_func = MulticlassConfusionMatrix(num_classes=num_classes)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.aux_metrics = aux_metrics
        self.aux_outputs = {mode: defaultdict(lambda: defaultdict(lambda: 0.0)) for mode in (TRAIN, VAL, TEST)}
        self.debug_level = debug_level
        # When to log
        self.on_epoch = on_epoch
        self.on_step = on_step
        # Stochastic channel masking
        if isinstance(model, Wav2Sleep):
            self.masker = masker
        else:
            self.masker = None
        self.flip_polarity = flip_polarity
        # Is the model unified? i.e. does it work on multiple modalities
        self.unified = isinstance(model, Wav2Sleep) and len(model.signal_encoders) > 1

    def forward(self, x: dict[str, torch.Tensor], y: torch.Tensor | None = None) -> torch.Tensor:
        if isinstance(self.model, SleepPPGNet):
            # Turn to single tensor of e.g. ECG or PPG for SleepPPG-Net
            if len(x) != 1:
                raise ValueError(f'{x.keys()=} but expected unimodal input!')
            x = x[list(x.keys())[0]]
        return self.model(x)

    def reshape_for_loss(self, outputs: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Flatten sequence data for the loss function."""
        n_classes = outputs.size(-1)
        return outputs.view(-1, n_classes), labels.view(-1)

    def get_ds_name(self, dataloader_idx: int, mode: str):
        """Determine name of dataset from data loader."""
        if mode == TRAIN:  # Train dataloader combines all.
            ds_name = 'all'
        elif mode == VAL:
            ds_name = self.trainer.datamodule.val_dataset_map[dataloader_idx]
        else:
            ds_name = self.trainer.datamodule.test_dataset_map[dataloader_idx]
        return ds_name

    def _step(
        self,
        batch: tuple[dict[str, torch.Tensor], torch.Tensor],
        mode: str,
        dataloader_idx: int = 0,
        signals: tuple[str] | None = None,
    ):
        """Generic training, validation, test step."""
        ds_name = self.get_ds_name(dataloader_idx, mode)
        x, y_true_BSC = batch
        # Apply model to subset of signals
        if signals is not None:
            x = {s: x[s] for s in signals}
            sig_prefix = '_'.join(signals)
            loss_prefix = f'{mode}_{sig_prefix}'
        else:
            loss_prefix = mode
            if self.unified:
                sig_prefix = None
            else:  # Log metric for the specific signal.
                sig_prefix = '_'.join(x.keys())
        y_logits_BSC = self(x, y_true_BSC)
        y_logits_NC, y_true_NC = self.reshape_for_loss(y_logits_BSC, y_true_BSC)
        loss = self.criterion(y_logits_NC, y_true_NC.long())

        if dataloader_idx == 0:  # Combined dataloader
            loss_name = f'{loss_prefix}_loss'
        else:
            loss_name = f'{loss_prefix}_loss_{ds_name}'
        # Compute confusion matrices
        with torch.no_grad():
            cmat = confusion_matrix(y_logits_NC, y_true_NC, self.cmat_func)
            self.aux_outputs[mode][sig_prefix][ds_name] += sum_if_distributed(cmat)
        self.log(
            loss_name,
            float(loss),
            prog_bar=True,
            logger=True,
            on_step=self.on_step,
            on_epoch=self.on_epoch,
            add_dataloader_idx=False,
            sync_dist=True,
        )
        return loss

    def training_step(self, batch, batch_idx):
        signals, _ = batch
        if self.flip_polarity:
            invert_signals(signals)
        if self.unified and self.masker is not None:
            self.masker(signals)
        return self._step(batch, mode=TRAIN)

    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0):
        # Run validation on signal subsets as well as on all available.
        loss = self._step(batch, mode=VAL, dataloader_idx=dataloader_idx)
        # Don't evaluate on subsets for non-unified models.
        if dataloader_idx == 0 or not self.unified:
            return loss
        ds_name = self.get_ds_name(dataloader_idx, VAL)
        # Validate on ECG for all datasets
        if ECG in batch[0] and ECG in self.model.valid_signals:
            self._step(batch, mode=VAL, dataloader_idx=dataloader_idx, signals=(ECG,))
            # Validate ECG/THX on SHHS, MESA
            if THX in batch[0] and THX in self.model.valid_signals and ds_name in (SHHS, MESA):
                self._step(batch, mode=VAL, dataloader_idx=dataloader_idx, signals=(ECG, THX))
        # Validate on PPG for datasets where available.
        if PPG in batch[0] and PPG in self.model.valid_signals and ds_name in (MESA, CFS, CCSHS, CHAT):
            self._step(batch, mode=VAL, dataloader_idx=dataloader_idx, signals=(PPG,))
            if THX in batch[0] and THX in self.model.valid_signals and ds_name in (MESA,):
                self._step(batch, mode=VAL, dataloader_idx=dataloader_idx, signals=(PPG, THX))
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx: int = 0):
        loss = self._step(batch, mode=TEST, dataloader_idx=dataloader_idx)
        if not self.unified:
            return loss
        ds_name = self.get_ds_name(dataloader_idx, TEST)
        # Validate on ECG for all datasets.
        if ECG in batch[0] and ECG in self.model.valid_signals:
            self._step(batch, mode=TEST, dataloader_idx=dataloader_idx, signals=(ECG,))
            # Validate ECG/THX where available.
            if THX in batch[0] and THX in self.model.valid_signals:
                self._step(batch, mode=TEST, dataloader_idx=dataloader_idx, signals=(ECG, THX))
        # Validate on PPG for datasets where available.
        if PPG in batch[0] and PPG in self.model.valid_signals and ds_name in (MESA, CFS, CCSHS, CHAT):
            self._step(batch, mode=TEST, dataloader_idx=dataloader_idx, signals=(PPG,))
            if THX in batch[0] and THX in self.model.valid_signals and ds_name in (MESA,):
                self._step(batch, mode=TEST, dataloader_idx=dataloader_idx, signals=(PPG, THX))
        return loss

    def predict_step(self, batch):
        x, y_true_BSC = batch
        output_dict = {'labels': y_true_BSC}
        # Generate predictions just using ECG.
        if ECG in x:
            y_logits_ECG_BSC = self({ECG: x[ECG]}, y_true_BSC)
            output_dict[f'preds_{ECG}'] = y_logits_ECG_BSC.argmax(dim=-1)
        # Generate predictions using ECG + THX
        if ECG in x and THX in x:
            x_ECG_THX = {ECG: x[ECG], THX: x[THX]}
            y_logits_ECG_THX_BSC = self(x_ECG_THX, y_true_BSC)
            output_dict[f'preds_{ECG}_{THX}'] = y_logits_ECG_THX_BSC.argmax(dim=-1)
        # Generate predictions using all modalities.
        output_dict['preds'] = self(x, y_true_BSC).argmax(dim=-1)
        return output_dict

    def _epoch_end(self, mode) -> None:
        """Generic method called at the end of an epoch."""
        epoch = self.trainer.current_epoch
        if self.tuning_mode:
            logger.debug(f'Skipping {epoch=}, {mode=}. In tuning mode.')
        elif (
            self.debug_level < 2
            and self.trainer.state.stage != RunningStage.SANITY_CHECKING
            and isinstance(self.logger, MLFlowLogger)
        ):
            logger.info(f'Logging at {mode=}, epoch={self.trainer.current_epoch}!')
            # Make sure DDP processes in the same order to avoid deadlock.
            for sig_prefix in sorted(self.aux_outputs[mode].keys(), key=sortkey):
                aux_metrics = self.aux_outputs[mode][sig_prefix]
                for ds_name, cmat in sorted(aux_metrics.items()):
                    if sig_prefix is None:
                        prefix = f'{mode}_{ds_name}'
                    else:
                        prefix = f'{mode}_{sig_prefix}_{ds_name}'
                    log_aux_metrics(cmat.cpu().numpy(), epoch=epoch, prefix=prefix)
        self.aux_outputs[mode] = defaultdict(lambda: defaultdict(lambda: 0.0))  # Reset

    def on_train_epoch_end(self) -> None:
        self._epoch_end(mode=TRAIN)

    def on_validation_epoch_end(self) -> None:
        self._epoch_end(mode=VAL)

    def on_test_epoch_end(self) -> None:
        self._epoch_end(mode=TEST)

    def configure_optimizers(self) -> dict[str, Optimizer | str]:
        """Set-up neural network optimizer"""
        optimizer = self.optimizer(self.model.parameters())
        optimizer_dict = {'optimizer': optimizer}
        if self.scheduler is None:
            return optimizer_dict
        scheduler = self.scheduler(optimizer)
        if isinstance(scheduler, ReduceLROnPlateau):
            optimizer_dict['lr_scheduler'] = {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1,
                'strict': True,
            }
        elif isinstance(scheduler, ExpWarmUpScheduler):
            optimizer_dict['lr_scheduler'] = {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
                'strict': True,
            }
        else:
            raise ValueError(f'{scheduler} is not configured.')
        return optimizer_dict


def sortkey(x):
    """For sorting signals which may be None."""
    return (x is not None, x)


def invert_signals(signals: dict[str, torch.Tensor]):
    """Randomly invert signal polarity with p = 0.5.

    Should improve robustness to e.g. ECG leads connected wrong way around [1].

    [1] Expert-level sleep staging using an electrocardiography-only feed-forward neural network. Jones et al. 2024.
    """
    for signal_name, x_BT in signals.items():
        B, T = x_BT.shape
        flip_mask = 2 * torch.randint(0, 2, (B, 1), dtype=torch.float, device=x_BT.device).repeat(1, T) - 1
        signals[signal_name] *= flip_mask
    return signals
