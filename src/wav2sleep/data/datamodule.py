"""Lightning data module."""

__all__ = ('SleepDataModule',)
import logging
import os

import lightning
from torch.utils.data import DataLoader

from ..settings import CENSUS, PPG, TEST, TRAIN, VAL
from .dataset import ParquetDataset
from .nsrr import get_dataset
from .utils import get_parquet_cols, get_parquet_fps

logger = logging.getLogger(__name__)


MAX_NIGHTS = 1_000_000


def get_parquet_fps_for_dataset(
    datasets: list[str],
    partition: str,
    data_location: str,
    columns: list[str],
    exclude_issues: bool = True,
    max_nights: int = MAX_NIGHTS,
) -> list[str]:
    """Create Dataset object."""
    parquet_fps = []
    if len(datasets) == 0:
        raise ValueError(f'No datasets provided: {datasets}.')
    for dataset in datasets:
        folder = os.path.join(data_location, dataset, partition)
        if not os.path.exists(folder):
            raise FileNotFoundError(folder)
        folder_lim = MAX_NIGHTS
        logger.info(f'Using up to {folder_lim} records from {folder=}.')
        parquet_fps += get_parquet_fps(folder)[:folder_lim]
    prefiltered = len(parquet_fps)
    # Sessions with issues end with '.issues.parquet'
    if exclude_issues:
        files = [fp for fp in parquet_fps if '.issues' not in fp]
    num_removed = prefiltered - len(parquet_fps)
    if num_removed > 0:
        logger.info(f'Removed {num_removed} files due to scoring issues.')
    prefiltered = len(parquet_fps)
    # Remove files that don't have any of the columns we're using.
    # Relevant for PPG-only training.
    if len(columns) == 1 and PPG in columns:
        parquet_fps = [fp for fp in parquet_fps if bool(set(columns).intersection(get_parquet_cols(fp)))]
    num_removed = prefiltered - len(parquet_fps)
    if num_removed > 0:
        logger.info(f'Removed {num_removed} files because no relevant columns.')
    logger.info(f'Creating dataset from {len(parquet_fps)} (max {max_nights}) files.')
    parquet_fps = sorted(parquet_fps[:max_nights])

    if len(parquet_fps) == 0:
        raise ValueError('Filtered out all files.')
    return parquet_fps


class SleepDataModule(lightning.LightningDataModule):
    def __init__(
        self,
        columns: list[str],
        num_classes: int,
        data_location: str,
        train_datasets: list[str],
        val_datasets: list[str],
        test_datasets: list[str] | None = None,
        test: bool = False,
        max_nights: int = MAX_NIGHTS,
        batch_size: int = 32,
        num_workers: int = 10,
        pin_memory: bool = True,
        exclude_issues: bool = False,
        persistent_workers: bool = True,
        prepare_data_per_node: bool = True,
        val_batch_size: int | None = None,
        test_batch_size: int | None = None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prepare_data_per_node = prepare_data_per_node  # If True, 'prepare_data' called on all nodes.

        def _create_dataset(datasets: list[str], partition: str) -> ParquetDataset:
            parquet_fps = get_parquet_fps_for_dataset(
                datasets=datasets,
                partition=partition,
                data_location=data_location,
                columns=columns,
                exclude_issues=exclude_issues,
                max_nights=max_nights,
            )
            return ParquetDataset(parquet_fps=parquet_fps, columns=columns, num_classes=num_classes)

        self.train_dataset = _create_dataset(
            datasets=train_datasets,
            partition=TRAIN,
        )
        # 1st val dataloader contains all datasets to compute total val. loss
        # Don't use CENSUS for total validation loss. (To avoid repeated data)
        # Create mapping from dataloader idxes to dataset names, to aid logging of metrics.
        self.val_dataset_map = {}
        # Create total val dataloader and separate val dataloaders for each dataset.
        if len(val_datasets) > 1:
            total_val_datasets = [ds for ds in val_datasets if ds != CENSUS]
            self.val_datasets = [_create_dataset(datasets=total_val_datasets, partition=VAL)]
            self.val_dataset_map[0] = 'all'
            logger.info('Creating separate val dataloaders for each dataset.')
            for i, folder in enumerate(val_datasets):
                self.val_dataset_map[i + 1] = get_dataset(folder)
                self.val_datasets.append(_create_dataset(datasets=[folder], partition=VAL))
        else:  # Only one val dataset
            self.val_dataset_map[0] = get_dataset(val_datasets[0])
            self.val_datasets = [_create_dataset(datasets=val_datasets, partition=VAL)]
        if not test:
            return
        self.test_datasets = []
        self.test_dataset_map = {}
        if test_datasets is not None:
            for i, folder in enumerate(test_datasets):
                self.test_dataset_map[i] = get_dataset(folder)
                self.test_datasets.append(_create_dataset(datasets=[folder], partition=TEST))
        else:
            self.test_datasets = None

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self, batch_size: int | None = None) -> list[DataLoader]:
        ws_per_loader = self.num_workers
        if batch_size is None and self.val_batch_size is None:
            batch_size = self.batch_size
        elif self.val_batch_size is not None:
            batch_size = self.val_batch_size
        return [
            DataLoader(
                ds,
                batch_size=batch_size,
                num_workers=ws_per_loader,
                pin_memory=self.pin_memory,
                shuffle=False,
                persistent_workers=self.persistent_workers,
            )
            for ds in self.val_datasets
        ]

    def test_dataloader(self, batch_size: int | None = None) -> list[DataLoader]:
        if self.test_datasets is None:
            raise ValueError('No test datasets specified.')
        ws_per_loader = self.num_workers
        if batch_size is None and self.test_batch_size is None:
            batch_size = self.batch_size
        elif self.test_batch_size is not None:
            batch_size = self.test_batch_size
        return [
            DataLoader(
                ds,
                batch_size=batch_size,
                num_workers=ws_per_loader,
                pin_memory=self.pin_memory,
                shuffle=False,
                persistent_workers=False,
            )
            for ds in self.test_datasets
        ]

    def predict_dataloader(self):
        return None
