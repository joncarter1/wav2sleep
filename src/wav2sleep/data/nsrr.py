import logging
import os

from ..settings import KNOWN_DATASETS

logger = logging.getLogger(__name__)


def get_split(dataset: str, split: str) -> list[str]:
    """Get dataset splits used."""
    folder = os.path.dirname(__file__)
    fp = os.path.join(folder, 'splits', dataset, f'{split}.txt')
    if not os.path.exists(fp):
        logger.info(f"Couldn't find {fp=} for {dataset=}, {split=}")
        return []
    with open(fp, 'r') as f:
        return [session_id.strip() for session_id in f.readlines()]


def get_dataset(fp: str):
    """Infer source dataset of filepath."""
    for ds in KNOWN_DATASETS:
        if ds in fp:
            return ds
    else:
        raise ValueError(f"Couldn't determine source dataset of {fp=}")
