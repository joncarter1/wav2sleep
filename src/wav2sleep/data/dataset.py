"""PyTorch dataset."""

import logging
from typing import Dict

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from ..settings import (
    CAUSAL_NORM_BASELINE_TAU_SECONDS,
    CAUSAL_NORM_MIN_SIGMA,
    CAUSAL_NORM_TAU_SECONDS,
    COLS_TO_SAMPLES_PER_EPOCH,
    INTEGER_LABEL_MAPS,
    LABEL,
    NORM_OUTLIER_THRESHOLD,
)
from .normalization import causal_rolling_normalize

logger = logging.getLogger(__name__)


class ParquetDataset(Dataset):
    """Subclass of torch.utils.data.Dataset.

    The methods implemented are:
    1. __get_item__, which is used by the PyTorch dataloader to get samples
    2. __len__, which tells the dataloader how many samples there are.

    See here for more info on custom PyTorch data loading:
    https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """

    def __init__(
        self,
        parquet_fps: list[str],
        columns: list[str],
        num_classes: int = 4,
        require_labels: bool = True,
        max_length_hours: int | None = None,
        causal: bool = False,
    ):
        """
        Class for creating a time-series sleep dataset for deep learning.

        Args:
            parquet_fps: List of parquet filepaths.
            columns: Subset of columns to use from the parquet files.
            num_classes: Number of sleep stage classes to use.
            require_labels: If True, raise an error if no labels are found in the parquet files.
            max_length_hours: Maximum recording length in hours (truncates longer recordings).
            causal: If True, apply causal normalization using exponential moving average.
                If False, apply non-causal z-score normalization using global statistics.

        Notes:
            Normalization behavior:
            - causal=True: Uses exponential moving average (EMA) normalization with parameters
              from settings.py (tau=15min, outlier_threshold=4.0). This is suitable for
              real-time or streaming applications where future information must not leak.
            - causal=False: Uses standard z-score normalization with global mean/std computed
              across the entire recording. This gives better normalization but uses future info.
        """
        self.files = parquet_fps
        self.columns = columns
        for col in self.columns:
            if col not in COLS_TO_SAMPLES_PER_EPOCH:
                raise ValueError(f'Column {col} unrecognised.')
        self.map = INTEGER_LABEL_MAPS[num_classes]
        self.require_labels = require_labels
        self.max_length_epochs = 1_000_000 if max_length_hours is None else max_length_hours * 60 * 2
        self.causal = causal

    @staticmethod
    def _zscore_normalize(signals: Dict[str, Tensor]) -> Dict[str, Tensor]:
        out: Dict[str, Tensor] = {}
        eps = 1e-6
        for k, x in signals.items():
            if x.numel() == 0 or not torch.isfinite(x).all():
                out[k] = x
                continue
            mu = torch.mean(x)
            std = torch.std(x)
            std = std if std > eps else torch.tensor(eps, dtype=x.dtype, device=x.device)
            out[k] = (x - mu) / std
        return out

    def _causal_normalize(self, signals: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Apply causal rolling normalization to signals using EMA.

        This normalization is causal (online), meaning it only uses past samples to compute
        statistics at each time point. This prevents future information leakage and makes
        it suitable for real-time or streaming applications.

        Args:
            signals: Dictionary mapping signal names to 1D tensors

        Returns:
            Dictionary with same keys as input, containing normalized signals
        """

        out: Dict[str, Tensor] = {}
        for k, x in signals.items():
            # Skip invalid/missing signals
            if x.numel() == 0 or not torch.isfinite(x).all() or torch.isinf(x).any():
                out[k] = x
                continue

            # Get signal-specific sampling frequency
            if k not in COLS_TO_SAMPLES_PER_EPOCH:
                logger.warning(f'Unknown signal {k}, skipping causal normalization')
                out[k] = x
                continue

            samples_per_epoch = COLS_TO_SAMPLES_PER_EPOCH[k]
            sampling_freq = samples_per_epoch / 30.0

            # Apply causal normalization with separate tau for baseline and variance
            # This handles signals with baseline drift (like CFS/CCSHS PPG) while
            # keeping variance estimation stable. min_sigma prevents NaN from flat epochs.
            out[k] = causal_rolling_normalize(
                signal=x,
                sampling_freq=sampling_freq,
                tau_seconds=CAUSAL_NORM_TAU_SECONDS,
                outlier_threshold_sigma=NORM_OUTLIER_THRESHOLD,
                baseline_tau_seconds=CAUSAL_NORM_BASELINE_TAU_SECONDS,
                min_sigma=CAUSAL_NORM_MIN_SIGMA,
            )
        return out

    def __getitem__(self, idx) -> tuple[dict[str, Tensor], Tensor]:
        """Return a sample of data for neural network training/evaluation.

        Samples are turned into training batches automatically by the torch.Dataloader class.
        """
        fp = self.files[idx]
        df = try_read_parquet(fp)
        signal_dict = {}
        found_col = False
        cols_to_pad = []
        prev_inferred_recording_length_epochs = None
        for col in self.columns:
            if col in df.columns:
                found_col = True
                x_T = torch.from_numpy(df[col].dropna().values).float()
                if torch.isinf(x_T).any():
                    raise ValueError(f'{fp=} has inf. values for {col=}')
                # Check that signals have the same effective length.
                inferred_recording_length_epochs = x_T.shape[0] // COLS_TO_SAMPLES_PER_EPOCH[col]
                if prev_inferred_recording_length_epochs is None:
                    prev_inferred_recording_length_epochs = inferred_recording_length_epochs
                elif prev_inferred_recording_length_epochs != inferred_recording_length_epochs:
                    raise ValueError(
                        f'{prev_inferred_recording_length_epochs=} != {inferred_recording_length_epochs=} for {fp=}'
                    )
                # Truncate as desired.
                signal_dict[col] = x_T[
                    : COLS_TO_SAMPLES_PER_EPOCH[col] * min(inferred_recording_length_epochs, self.max_length_epochs)
                ]
            else:
                cols_to_pad.append(col)
        if not found_col:
            raise ValueError(f'No relevant columns found in {fp=}. {self.columns=}')
        # Apply normalization based on causal mode.
        if self.causal:
            signal_dict = self._causal_normalize(signal_dict)
        else:
            signal_dict = self._zscore_normalize(signal_dict)
        for col in cols_to_pad:
            # Use recording length inferred from other signals to pad with correct length.
            sig_len = COLS_TO_SAMPLES_PER_EPOCH[col] * min(inferred_recording_length_epochs, self.max_length_epochs)
            signal_dict[col] = torch.full((sig_len,), float('-inf')).float()
        if self.require_labels or LABEL in df.columns:
            labels = df[LABEL].dropna().map(self.map)
            labels = torch.from_numpy(labels.fillna(-1).values.T).float()
            if labels.shape[0] != inferred_recording_length_epochs:
                raise ValueError(f'{labels.shape=} != {inferred_recording_length_epochs=} for {fp=}')
            # Truncate as desired.
            labels = labels[: self.max_length_epochs]
        else:
            labels = torch.full((inferred_recording_length_epochs,), -1).float()[: self.max_length_epochs]
        return signal_dict, labels

    def __len__(self) -> int:
        """Length of dataset."""
        return len(self.files)


def try_read_parquet(fp: str, columns: list[str] | None = None, max_retries: int = 3):
    """Read parquet with retries for flaky filesystems."""
    try:
        return pd.read_parquet(fp, columns=columns)
    except Exception as e:
        logger.error(f'Failed to read parquet {fp=} - {e}')
        if max_retries > 0:
            return try_read_parquet(fp, columns=columns, max_retries=max_retries - 1)
        else:
            raise ValueError(f'Failed to read parquet {fp=}')
