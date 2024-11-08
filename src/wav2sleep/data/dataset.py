"""PyTorch dataset."""

import logging

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from ..settings import COL_MAP, INTEGER_LABEL_MAPS, LABEL

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
    ):
        """
        Class for creating a time-series sleep dataset for deep learning.

        Args:
            datasets: List of datasets to use e.g. shhs, mesa
            col_map: Mapping from desired cols to expected shape.
                For unavailable cols, we return neg inf. tensors of the desired shape.
            data_location: Path that datasets are stored under i.e. /path/to/processed/nsrr
            partition: train, val or test.
            num_classes: Number of sleep stage classes to use.
            max_nights: Max. number of files to use, or number per folder.
        """
        self.files = parquet_fps
        self.columns = columns
        for col in self.columns:
            if col not in COL_MAP:
                raise ValueError(f'Column {col} unrecognised.')
        self.map = INTEGER_LABEL_MAPS[num_classes]

    def __getitem__(self, idx) -> tuple[dict[str, Tensor], Tensor]:
        """Return a sample of data for neural network training/evaluation.

        Samples are turned into training batches automatically by the torch.Dataloader class.
        """
        fp = self.files[idx]
        df = try_read_parquet(fp)
        signal_dict = {}
        found_col = False
        for col in self.columns:
            sig_len = COL_MAP[col]
            if col in df.columns:
                signal_dict[col] = torch.from_numpy(df[col].dropna().values).float()
                found_col = True
            else:
                signal_dict[col] = torch.full((sig_len,), float('-inf')).float()
        if not found_col:
            raise ValueError(f'No relevant columns found in {fp=}. {self.columns=}')
        labels = df[LABEL].dropna().map(self.map)
        labels = torch.from_numpy(labels.fillna(-1).values.T).float()
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
