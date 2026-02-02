import argparse
import os
import shutil
from glob import glob

from tqdm import tqdm

from wav2sleep.data.nsrr import get_split
from wav2sleep.settings import CENSUS, INGEST, TEST, VAL

JONES_SPLITS = {VAL: get_split(CENSUS, VAL), TEST: get_split(CENSUS, TEST)}


def build_set(folder: str, split: str, all_parquet_fps: list[str], copy: bool = False) -> None:
    if split not in (VAL, TEST):
        raise ValueError(f'Split must be either {VAL} or {TEST}')

    session_ids = JONES_SPLITS[split]
    found = {}
    # Create mapping from session ID to file path.
    for fp in all_parquet_fps:
        session_id = (
            os.path.basename(fp).replace('.parquet', '').replace('.issues', '')
        )  # n.b. one file from CFS used was marked as having scoring issues.
        if session_id in session_ids:
            found[session_id] = fp
    # Check all found
    if len(found) != len(session_ids):
        missing_files = set(session_ids).difference(found.keys())
        print(len(missing_files), missing_files)
        raise ValueError(f'Found {len(found)} files, but expected {len(session_ids)}')
    else:
        print(f'Found all {len(found)} files necessary for {split} split. Copying...')
    # Copy files to new folder - remove .issues from filename to avoid ignoring it during evaluation.
    for session_id, fp in tqdm(found.items()):
        o_fp = os.path.join(folder, CENSUS, split, os.path.basename(fp).replace('.issues', ''))
        os.makedirs(os.path.dirname(o_fp), exist_ok=True)
        if copy:
            shutil.copy2(fp, o_fp)
        else:
            os.symlink(fp, o_fp)


def parse_args():
    parser = argparse.ArgumentParser(prog='Dataset Splitter', description='Split dataset into train, val, test sets.')
    parser.add_argument('--folder', help='Location of processed NSRR datasets.')
    parser.add_argument('--copy', action='store_true', help='Copy files to new folder.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print('Globbing all ingested files...')
    all_parquet_fps = glob(f'{args.folder}/*/{INGEST}/*.parquet')
    print('Found ', len(all_parquet_fps), 'files.')
    build_set(args.folder, VAL, all_parquet_fps, copy=args.copy)
    build_set(args.folder, TEST, all_parquet_fps, copy=args.copy)


if __name__ == '__main__':
    main()
