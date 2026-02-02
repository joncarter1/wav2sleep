"""Sleep staging the NSRR datasets."""

import logging
import os
import warnings

warnings.filterwarnings('ignore', message='pkg_resources is deprecated')
import hydra
import mlflow
import numpy
import torch
from hydra_useful_callbacks.utils import is_rank_zero
from lightning.pytorch.tuner.tuning import Tuner
from lightning.pytorch.utilities.enums import GradClipAlgorithmType
from omegaconf import DictConfig

from wav2sleep.data.datamodule import SleepDataModule
from wav2sleep.log import restore_and_log_ckpt
from wav2sleep.trainer import SleepLightningModule
from wav2sleep.utils import fix_seeds

logger = logging.getLogger(__name__)

torch.backends.mha.set_fastpath_enabled(False)  # Issue with fastpath compilation in validation step...


def train_func(cfg: DictConfig):
    if is_rank_zero() and mlflow.active_run() is not None:
        mlflow.enable_system_metrics_logging()
    if cfg.compile:
        # Needs to be imported here to keep submitit happy (doesn't like pickling config modules...).
        from torch._dynamo import config as dynamo_config
        from torch._functorch import config as functorch_config

        dynamo_config.recompile_limit = 64  # Default is 8... we have many different shapes...
        functorch_config.activation_memory_budget = cfg.activation_memory_budget

    # Create the Pytorch Lightning module from Hydra configuration (see https://hydra.cc/docs/advanced/instantiate_objects/overview/ )
    pl_model: SleepLightningModule = hydra.utils.instantiate(cfg.training.module)
    if cfg.compile:
        logger.info('Compiling model...')
        pl_model.compile(fullgraph=False, dynamic=False)
    logger.info('Model compiled.')

    # PyTorch lightning trainer module : https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html
    trainer = hydra.utils.instantiate(cfg.training.trainer)
    datamodule: SleepDataModule = hydra.utils.instantiate(cfg.training.datamodule)
    # Find maximum batch size for the model.
    if cfg.tune_batch_size and cfg.num_gpus > 1:
        logger.warning("Can't use batch size finder with DDP yet. Using default batch size.")
    elif cfg.tune_batch_size:
        # Create a tuner for the trainer. Auto-scale batch size by growing it exponentially.
        logger.info('Tuning batch size on single GPU...')
        pl_model.tuning_mode = True
        tuner = Tuner(trainer)
        tuner.scale_batch_size(model=pl_model, datamodule=datamodule, mode='power')
        pl_model.tuning_mode = False
    # Accumulate gradients to achieve a target batch size.
    if cfg.target_batch_size is not None:
        num_gpus = cfg.num_gpus
        ratio = cfg.target_batch_size / (datamodule.batch_size * num_gpus)
        if ratio.is_integer():
            logger.info(
                f'Accumulating {int(ratio)} batch(es) of {datamodule.batch_size} on {num_gpus} GPU(s) for an effective batch size of {cfg.target_batch_size}.'
            )
            trainer.accumulate_grad_batches = int(ratio)
        elif ratio < 1:
            logger.info(
                'Current batch size is greater than or equal to the target batch size. Defaulting to no accumulation.'
            )
            trainer.accumulate_grad_batches = 1
        else:
            logger.warning(
                'The target batch size is not an integer multiple of the current batch size. Defaulting to no accumulation.'
            )
            trainer.accumulate_grad_batches = 1
    # Following is currently necessary to reload the Lightning checkpoints.
    torch.serialization.add_safe_globals(
        [
            numpy._core.multiarray.scalar,
            numpy.dtype,
            numpy.dtypes.Float64DType,
            GradClipAlgorithmType,
        ]
    )
    # Train the neural network
    logger.info('Training model...')
    if cfg.ckpt_path is not None:
        ckpt_path = cfg.ckpt_path.replace('file://', '')
        logger.info(f'Loading checkpoint from {ckpt_path=}')
    else:
        ckpt_path = None
    trainer.fit(pl_model, datamodule=datamodule, ckpt_path=ckpt_path)
    # Restore the best checkpoint for evaluation.
    last_ckpt_path = os.path.join(mlflow.get_artifact_uri(), 'model/checkpoints/last/last.ckpt').replace('file://', '')
    if cfg.restore_best and os.path.exists(last_ckpt_path):
        logger.info(f'Restoring checkpoint from: {last_ckpt_path}')
        state_dict = torch.load(last_ckpt_path)['state_dict']
        pl_model.load_state_dict(state_dict)
    else:
        logger.info('Not restoring checkpoint.')
    # Apply to test set.
    if cfg.test:
        logger.info('Evaluating checkpoint on the test set.')
        trainer.test(pl_model, datamodule=datamodule)
    return trainer


@hydra.main(version_base=None, config_path='config', config_name='main')
def main(cfg: DictConfig):
    logger.info('Starting run.')
    # Fix random seeds of numpy, pytorch etc.
    fix_seeds(cfg.seed)
    # Create storage for outputs
    os.makedirs(os.environ['WAV2SLEEP_STORAGE'], exist_ok=True)
    # Train the model.
    torch.set_float32_matmul_precision('high')
    train_func(cfg)
    # Restore the best checkpoint and log the model.
    restore_and_log_ckpt(cfg)
    return


if __name__ == '__main__':
    main()
