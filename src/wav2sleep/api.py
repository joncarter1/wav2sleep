"""Public API for running wav2sleep on new data.

This module provides a minimal, high-level interface that wraps the existing
CLI-oriented utilities into importable, user-friendly functions.

Typical usage:

    from wav2sleep import load_model, predict_on_folder

    model = load_model("/path/to/checkpoint", device="cuda")
    predict_on_folder(
        input_folder="/path/to/new_data",
        output_folder="/path/to/save_preds",
        model=model,  # alternatively pass model_folder instead
    )
"""

from __future__ import annotations

import logging
import os
import tempfile
from glob import glob
from pathlib import Path
from typing import Iterable, Optional, Tuple

import hydra
import numpy as np
import pandas as pd
import torch
import yaml
from omegaconf import OmegaConf
from tqdm import tqdm
from yaml import CLoader as Loader

from .data.dataset import ParquetDataset
from .data.edf import load_edf_data
from .data.preprocessing import process_waveform_dataframe
from .hub import download_from_hub, is_hf_repo_id
from .models.wav2sleep import Wav2Sleep
from .settings import LABEL, PRED, TIMESTAMP

logger = logging.getLogger(__name__)


def load_model(
    folder: str,
    device: str = 'auto',
    compile: bool = False,
    revision: str | None = None,
    cache_dir: str | None = None,
):
    """Load a pretrained wav2sleep model from a checkpoint folder or Hugging Face Hub.

    Args:
        folder: Local folder containing `config.yaml` and `state_dict.pth`,
            or a Hugging Face Hub URI (e.g., "hf://joncarter/wav2sleep").
        device: Torch device string, e.g. "cuda", "cpu", or "auto" (default).
            When "auto", selects CUDA if available, otherwise CPU.
        compile: If True, compile the model with torch.compile where available.
        revision: For HF Hub models, the git revision (branch, tag, or commit hash).
        cache_dir: For HF Hub models, local directory to cache downloads.

    Returns:
        A `Wav2Sleep` model moved to the requested device in eval mode.
    """

    # Download from HF Hub if repo ID detected
    if is_hf_repo_id(folder):
        folder = download_from_hub(folder, revision=revision, cache_dir=cache_dir)

    # Resolve device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_info = f' ({torch.cuda.get_device_name(0)})' if device.startswith('cuda') and torch.cuda.is_available() else ''
    logger.info(f'Using device: {device}{device_info}')

    # Load YAML configuration
    config_fp = os.path.join(folder, 'config.yaml')
    if not os.path.exists(config_fp):
        raise FileNotFoundError(f'No config file found at {config_fp}. Has the model been downloaded?')
    with open(config_fp, 'r') as f:
        model_cfg = yaml.load(f, Loader)
        model_cfg = OmegaConf.create(model_cfg)
    model: Wav2Sleep = hydra.utils.instantiate(model_cfg)
    ckpt_path = os.path.join(folder, 'state_dict.pth')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'No state dict found at {ckpt_path}. Has the model been downloaded?')
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    if compile:
        model.compile()
    model.eval()
    return model.to(device)


def prepare(
    input_folder: str,
    signals: Iterable[str],
    max_length_hours: int = 10,
    tmp_root_folder: str | None = None,
) -> str:
    """Preprocess EDF/CSV/Parquet into model-ready parquet files.

    Args:
        input_folder: Folder with raw EDF/CSV/Parquet files.
        signals: Iterable of signal names to use (e.g., ["ECG", "THX"]).
        max_length_hours: Maximum hours of data to keep per file.
        tmp_root_folder: Root directory for cached, preprocessed parquet files.
            Defaults to a platform-appropriate temp directory.

    Returns:
        Path to the folder containing preprocessed parquet files.
    """
    if tmp_root_folder is None:
        tmp_root_folder = os.path.join(tempfile.gettempdir(), 'wav2sleep')
    logger.info(f'Preparing dataset from {input_folder}...')
    signals = list(signals)
    tmp_subfolder = os.path.join(tmp_root_folder, '_'.join(signals) + f'_{max_length_hours}h')
    fps = _get_supported_files(input_folder)
    logger.debug(f'Found {len(fps)} files in {input_folder}')
    for fp in tqdm(fps):
        tmp_path = Path(tmp_subfolder) / Path(fp).relative_to(Path(fp).anchor).with_suffix('.parquet')
        if os.path.exists(tmp_path):
            logger.debug(f'Skipping {fp} because it already exists in {tmp_root_folder}')
            continue
        try:
            df = _load_file(fp, columns=signals)
        except (FileNotFoundError, ValueError, KeyError, pd.errors.ParserError) as e:
            logger.error(f'Failed to process {fp} due to {e}')
            continue
        df = process_waveform_dataframe(df, signals, max_length_hours=max_length_hours)
        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        df.to_parquet(tmp_path)
    return tmp_subfolder


def load_dataset(
    parquet_folder: str,
    signals: Iterable[str],
    num_classes: int = 4,
    max_length_hours: Optional[int] = None,
) -> ParquetDataset:
    """Create a `ParquetDataset` from a folder of parquet files."""
    signals = list(signals)
    input_fps = _get_parquet_files(parquet_folder)
    if len(input_fps) == 0:
        raise ValueError(f'No parquet files found in {parquet_folder}.')
    return ParquetDataset(
        parquet_fps=input_fps,
        num_classes=num_classes,
        columns=signals,
        require_labels=False,
        max_length_hours=max_length_hours,
    )


@torch.inference_mode()
def predict(
    model: Wav2Sleep,
    dataset: ParquetDataset,
    device: str = 'cuda',
    batch_size: int = 4,
    num_workers: int = 4,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Apply a wav2sleep model to a dataset and return predictions (and labels if present)."""
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=False
    )
    causal = model.signal_encoders.causal
    predictions = []
    labels = []
    for batch in tqdm(dataloader):
        x, batch_labels = batch
        x = {k: v.to(device) for k, v in x.items()}
        logits_BSC = model(x)
        batch_preds = logits_BSC.argmax(dim=-1)
        predictions.append(batch_preds)
        labels.append(batch_labels)
    predictions = torch.cat(predictions, dim=0).cpu()
    labels = torch.cat(labels, dim=0).cpu()
    if (labels == -1).all():
        labels = None
    return predictions, labels


def save_predictions(
    predictions: torch.Tensor,
    parquet_folder: str,
    output_folder: str,
    dataset: ParquetDataset,
    labels: Optional[torch.Tensor] = None,
    overwrite: bool = False,
    max_length_hours: Optional[int] = None,
) -> None:
    """Save predictions (and optional labels) to CSV files mirroring the input tree."""
    for idx, fp in enumerate(dataset.files):
        rel_path = Path(fp).relative_to(parquet_folder)
        out_fp = str(Path(output_folder) / rel_path.with_suffix('.preds.csv'))
        if os.path.exists(out_fp) and not overwrite:
            logger.warning(f'File {out_fp} exists. Skipping.')
            continue
        input_df = pd.read_parquet(fp)
        input_df = input_df[list(set(dataset.columns) & set(input_df.columns))]
        duration_epochs = int(len(predictions[idx]))
        start = input_df.index[0]
        output_index = pd.Index(np.arange(0, 60 * duration_epochs / 2, step=30) + 30.0, name=TIMESTAMP)
        if isinstance(input_df.index, pd.DatetimeIndex):
            output_index = start + pd.to_timedelta(output_index, unit='s')
        output_df = pd.DataFrame({PRED: predictions[idx][:duration_epochs]}, index=output_index)
        if labels is not None:
            output_df[LABEL] = labels[idx][:duration_epochs]
        os.makedirs(os.path.dirname(out_fp), exist_ok=True)
        output_df.to_csv(out_fp)


def predict_on_folder(
    input_folder: str,
    output_folder: str,
    *,
    model: Optional[Wav2Sleep] = None,
    model_folder: Optional[str] = None,
    signals: Optional[Iterable[str]] = None,
    device: str = 'cuda',
    batch_size: int = 4,
    num_workers: int = 4,
    preprocess: bool = True,
    max_length_hours: int = 10,
    overwrite: bool = False,
    compile: bool = False,
    return_tensors: bool = False,
) -> None | Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """End-to-end convenience function to run preprocessing and inference on a folder.

    Exactly mirrors the behavior of `scripts/predict.py` but is importable.

    Args:
        input_folder: Folder of EDF/CSV/Parquet to run inference on.
        output_folder: Where CSV prediction files will be written (tree mirrors input).
        model: Optionally provide a preloaded model (returned by `load_model`).
        model_folder: If `model` is not provided, path to checkpoint folder to load.
        signals: Optional subset of signals to use; defaults to any supported by the model.
        device: Torch device string.
        batch_size: Inference batch size.
        num_workers: DataLoader workers.
        preprocess: If True, preprocess raw files; if False, `input_folder` must already be parquet.
        max_length_hours: Maximum hours to process (affects preprocessing and dataset creation).
        overwrite: Overwrite existing output CSVs.
        compile: If True and loading from `model_folder`, compile the model.
    """
    if model is None:
        if model_folder is None:
            raise ValueError('Either `model` or `model_folder` must be provided.')
        model = load_model(model_folder, device=device, compile=compile)
    else:
        # Move provided model to device and eval just in case
        model = model.to(device)
        model.eval()

    # Determine signals: user-provided or use model's valid_signals
    if signals is None:
        if not hasattr(model, 'valid_signals'):
            raise AttributeError('Model does not expose `valid_signals`. Please pass `signals` explicitly.')
        signals = list(model.valid_signals)  # type: ignore[attr-defined]
    else:
        signals = list(signals)
        if hasattr(model, 'valid_signals'):
            valid = set(model.valid_signals)  # type: ignore[attr-defined]
            if not set(signals).issubset(valid):
                raise ValueError(f'Invalid signal subset: {signals}. Valid signals are: {sorted(valid)}')

    # Preprocess or assume parquet
    if preprocess:
        parquet_folder = prepare(input_folder=input_folder, signals=signals, max_length_hours=max_length_hours)
    else:
        parquet_folder = input_folder

    ds = load_dataset(
        parquet_folder=parquet_folder,
        signals=signals,
        num_classes=model.num_classes,  # type: ignore[attr-defined]
        max_length_hours=max_length_hours,
    )
    preds, labels = predict(model=model, dataset=ds, device=device, batch_size=batch_size, num_workers=num_workers)
    save_predictions(
        predictions=preds,
        parquet_folder=parquet_folder,
        output_folder=output_folder,
        dataset=ds,
        labels=labels,
        overwrite=overwrite,
        max_length_hours=max_length_hours,
    )
    return (preds, labels) if return_tensors else None


# ---------- internal helpers ----------


def _get_supported_files(input_folder: str) -> list[str]:
    files = []
    for ext in ('edf', 'csv', 'parquet'):
        files.extend(glob(os.path.join(input_folder, f'**/*.{ext}'), recursive=True))
    return files


def _get_parquet_files(folder: str) -> list[str]:
    return glob(os.path.join(folder, '**/*.parquet'), recursive=True)


def _load_file(fp: str, columns: list[str]) -> pd.DataFrame:
    if fp.endswith('.edf'):
        df, _metadata = load_edf_data(fp, columns=columns, convert_time=True, raise_on_missing=False)
        return df
    elif fp.endswith('.csv'):
        return pd.read_csv(fp, index_col=0, parse_dates=True)
    elif fp.endswith('.parquet'):
        return pd.read_parquet(fp)
    else:
        raise ValueError(f'Unsupported file extension for {fp}')
