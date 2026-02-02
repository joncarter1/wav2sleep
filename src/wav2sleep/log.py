import logging
import os

import hydra
import lightning
import matplotlib.pyplot as plt
import mlflow
import torch
from omegaconf import DictConfig, OmegaConf

from wav2sleep.plotting import plot_confusion_matrix
from wav2sleep.stats import (
    cohens_kappa,
    confusion_accuracy,
)
from wav2sleep.utils import rank_zero_only

logger = logging.getLogger(__name__)


# Mapping from num classes to categorisation
SLEEP_STAGE_CATEGORIES = {
    4: ['Wake', 'N1+N2', 'N3', 'REM'],
    5: ['Wake', 'N1', 'N2', 'N3', 'REM'],
}


@rank_zero_only
def log_aux_metrics(cmat, epoch, prefix: str):
    """Log auxiliary metrics."""
    num_classes = len(cmat)
    categories = SLEEP_STAGE_CATEGORIES[num_classes]
    fig, ax = plt.subplots(1, 1)
    plot_confusion_matrix(
        categories,
        cmat,
        ax=ax,
        description=None,
        heatmap_cmap='Purples',
    )
    fig.tight_layout()
    mlflow.log_figure(fig, f'{prefix}_conf_mats/{epoch:04d}.png')
    plt.close(fig)
    acc = float(confusion_accuracy(cmat))
    kappa = float(cohens_kappa(cmat, n_classes=len(cmat)))
    metrics = {f'{prefix}_acc': acc, f'{prefix}_kappa': kappa}
    mlflow.log_metrics(metrics, step=epoch)


def restore_checkpoint(pl_model):
    """Restore final checkpoint."""
    last_ckpt_path = os.path.join(mlflow.get_artifact_uri(), 'model/checkpoints/last/last.ckpt').replace('file://', '')
    if not os.path.exists(last_ckpt_path):
        logger.info("Checkpoint doesn't exist.")
        return None
    logger.info(f'Restoring checkpoint from: {last_ckpt_path}')
    state_dict = torch.load(last_ckpt_path, map_location='cpu')['state_dict']
    pl_model.load_state_dict(state_dict)
    return pl_model


# Re-instantiate the trained model from the best checkpoint for logging.
@rank_zero_only
def restore_and_log_ckpt(cfg):
    pl_model: lightning.LightningModule = hydra.utils.instantiate(cfg.training.module)
    pl_model_ckpt = restore_checkpoint(pl_model)
    # Log the underlying PyTorch model
    if pl_model_ckpt is not None:
        logger.info('Logging model state dict.')
        OmegaConf.resolve(cfg.model)
        log_model(pl_model_ckpt, cfg.model)
    else:
        logger.warning("Didn't find checkpoint to restore.")


@rank_zero_only
def log_model(pl_model: lightning.LightningModule, model_cfg: DictConfig):
    # # Log the state dict too, incase module paths change.
    logger.info('Logging model config.')
    mlflow.log_dict(OmegaConf.to_container(model_cfg), 'model/config.yaml')
    logger.info('Logging state dict.')
    mlflow.pytorch.log_state_dict(state_dict=pl_model.model.state_dict(), artifact_path='model')
    return
