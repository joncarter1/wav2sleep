"""Lightning data module."""

__all__ = ('SleepDataModule',)
import logging
import os
import subprocess  # noqa: S404
from concurrent.futures import ThreadPoolExecutor, as_completed

import lightning
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..settings import CENSUS, PPG, TEST, TRAIN, VAL
from .dataset import ParquetDataset
from .nsrr import get_dataset
from .utils import get_parquet_cols, get_parquet_fps

logger = logging.getLogger(__name__)


MAX_NIGHTS = 1_000_000


def _get_directory_size_bytes(path: str) -> int:
    """Get total size of directory in bytes, following symlinks."""
    result = subprocess.run(
        ['du', '-sLb', path],  # noqa: S603, S607
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return int(result.stdout.split()[0])
    return 0


def _check_destination_space(
    sync_tasks: list[tuple[str, str, str]],
    dest_path: str,
    buffer_fraction: float = 0.1,
) -> None:
    """Check destination has enough space for sync. Raises RuntimeError if not."""
    total_source_bytes = sum(_get_directory_size_bytes(src) for _, _, src in sync_tasks)

    os.makedirs(dest_path, exist_ok=True)
    stat = os.statvfs(dest_path)
    available_bytes = stat.f_bavail * stat.f_frsize

    required_bytes = int(total_source_bytes * (1 + buffer_fraction))

    source_gb = total_source_bytes / (1024**3)
    avail_gb = available_bytes / (1024**3)

    logger.info(f'Source data: {source_gb:.1f}GB, Available at destination: {avail_gb:.1f}GB')

    if available_bytes < required_bytes:
        raise RuntimeError(
            f'Insufficient space in {dest_path}: '
            f'need ~{source_gb * (1 + buffer_fraction):.1f}GB but only {avail_gb:.1f}GB available'
        )


def _rsync_directory(source_path: str, dest_path: str) -> tuple[bool, str]:
    """Rsync a single directory. Returns (success, error_message)."""
    os.makedirs(dest_path, exist_ok=True)

    cmd = [
        'rsync',
        '-Lav',
        '--inplace',
        '--no-whole-file',
        '--partial',
        '--size-only',
        f'{source_path}/',
        f'{dest_path}/',
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)  # noqa: S603, S607
        return (True, '')
    except subprocess.CalledProcessError as e:
        return (False, e.stderr.strip() if e.stderr else str(e))


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
        parquet_fps = [fp for fp in parquet_fps if '.issues' not in fp]
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
        drop_last: bool = False,
        causal: bool = False,
        sync_to_local: bool = False,
        local_data_cache: str | None = None,
        max_parallel_rsyncs: int = 8,
        seed: int = 42,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prepare_data_per_node = prepare_data_per_node  # If True, 'prepare_data' called on all nodes.
        self.drop_last = drop_last
        self.causal = causal
        self.sync_to_local = sync_to_local
        self.local_data_cache = local_data_cache
        self.max_parallel_rsyncs = max_parallel_rsyncs
        self.original_data_location = data_location
        self.data_location = data_location
        self.train_datasets = train_datasets
        self.val_datasets = val_datasets
        self.test_datasets = test_datasets
        self.test = test
        self._sync_completed = False
        self.seed = seed

        # Sync data to local cache if enabled, before creating datasets
        if self.sync_to_local and self.local_data_cache is not None:
            self._sync_data_to_local()

        def _create_dataset(datasets: list[str], partition: str) -> ParquetDataset:
            parquet_fps = get_parquet_fps_for_dataset(
                datasets=datasets,
                partition=partition,
                data_location=self.data_location,
                columns=columns,
                exclude_issues=exclude_issues,
                max_nights=max_nights,
            )
            return ParquetDataset(
                parquet_fps=parquet_fps,
                columns=columns,
                num_classes=num_classes,
                causal=self.causal,
            )

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
        g = torch.Generator()
        current_epoch = self.trainer.current_epoch if hasattr(self, 'trainer') and self.trainer else 0
        g.manual_seed(self.seed + current_epoch)

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            generator=g,
            persistent_workers=self.persistent_workers,
            drop_last=self.drop_last,
        )

    def val_dataloader(self, batch_size: int | None = None) -> list[DataLoader]:
        if batch_size is None and self.val_batch_size is None:
            batch_size = self.batch_size
        elif self.val_batch_size is not None:
            batch_size = self.val_batch_size
        return [
            DataLoader(
                ds,
                batch_size=batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                shuffle=False,
                persistent_workers=True,
                drop_last=self.drop_last,
                prefetch_factor=1,
            )
            for ds in self.val_datasets
        ]

    def test_dataloader(self, batch_size: int | None = None) -> list[DataLoader]:
        if self.test_datasets is None:
            raise ValueError('No test datasets specified.')
        ws_per_loader = max(1, self.num_workers // len(self.test_datasets))
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
                drop_last=self.drop_last,
            )
            for ds in self.test_datasets
        ]

    def predict_dataloader(self):
        return None

    def _sync_data_to_local(self) -> None:
        """Sync datasets from NFS to local cache."""
        if self.data_location == self.local_data_cache:
            logger.info(f'Data location already set to local cache: {self.data_location}')
            self._sync_completed = True
            return

        logger.info(f'Syncing from {self.original_data_location} to {self.local_data_cache}...')

        sync_tasks = self._build_sync_tasks()
        if not sync_tasks:
            logger.warning('No datasets/partitions found to sync.')
            return

        _check_destination_space(sync_tasks, self.local_data_cache)

        failed = self._run_parallel_sync(sync_tasks)

        if failed:
            error_summary = '\n'.join([f'  {ds}/{part}: {err}' for ds, part, err in failed])
            raise RuntimeError(f'Failed to sync {len(failed)} partitions:\n{error_summary}')

        self.data_location = self.local_data_cache
        self._sync_completed = True
        logger.info(f'Sync complete. Using local cache at {self.data_location}')

    def _build_sync_tasks(self) -> list[tuple[str, str, str]]:
        """Build list of (dataset, partition, source_path) to sync."""
        all_datasets = set(self.train_datasets)
        all_datasets.update(self.val_datasets)
        if self.test_datasets is not None:
            all_datasets.update(self.test_datasets)

        partitions = [TRAIN, VAL]
        if self.test:
            partitions.append(TEST)

        sync_tasks = []
        for dataset in sorted(all_datasets):
            for partition in partitions:
                source_path = os.path.join(self.original_data_location, dataset, partition)
                if os.path.exists(source_path):
                    sync_tasks.append((dataset, partition, source_path))

        return sync_tasks

    def _run_parallel_sync(
        self,
        sync_tasks: list[tuple[str, str, str]],
    ) -> list[tuple[str, str, str]]:
        """Run rsync in parallel. Returns list of (dataset, partition, error) for failures."""
        failed: list[tuple[str, str, str]] = []

        pbar = tqdm(total=len(sync_tasks), desc='Syncing', unit='partition', ncols=100)

        with ThreadPoolExecutor(max_workers=self.max_parallel_rsyncs) as executor:
            futures = {}
            for dataset, partition, source_path in sync_tasks:
                dest_path = os.path.join(self.local_data_cache, dataset, partition)
                future = executor.submit(_rsync_directory, source_path, dest_path)
                futures[future] = (dataset, partition)

            for future in as_completed(futures):
                dataset, partition = futures[future]
                success, error = future.result()
                pbar.update(1)
                if success:
                    pbar.set_postfix_str(f'{dataset}/{partition} ✓')
                else:
                    failed.append((dataset, partition, error))
                    pbar.set_postfix_str(f'{dataset}/{partition} ✗')

        pbar.close()
        return failed

    def prepare_data(self) -> None:
        """Sync datasets from NFS to local cache if enabled (Lightning hook)."""
        # If sync already happened in __init__, skip
        if self._sync_completed:
            return
        # Otherwise, perform sync now
        if self.sync_to_local and self.local_data_cache is not None:
            self._sync_data_to_local()
