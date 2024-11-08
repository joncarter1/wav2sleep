import os

from ..settings import KNOWN_DATASETS


def get_split(dataset: str, split: str):
    """Get dataset splits used."""
    folder = os.path.dirname(__file__)
    fp = os.path.join(folder, 'splits', dataset, f'{split}.txt')
    if not os.path.exists(fp):
        raise FileNotFoundError(f"Couldn't find {fp=} for {dataset=}, {split=}")
    with open(fp, 'r') as f:
        return [session_id.strip() for session_id in f.readlines()]


def get_dataset(fp: str):
    """Infer source dataset of filepath."""
    for ds in KNOWN_DATASETS:
        if ds in fp:
            return ds
    else:
        raise ValueError(f"Couldn't determine source dataset of {fp=}")
