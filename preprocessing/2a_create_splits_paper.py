"""Create dataset splits for training, validation, and testing as used in wav2sleep paper."""

import argparse
import logging
import os
import shutil
from glob import glob

from tqdm import tqdm

from wav2sleep.data.nsrr import get_split
from wav2sleep.settings import INGEST, TEST, TRAIN, VAL

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(prog='Dataset Splitter', description='Split dataset into train, val, test sets.')
    parser.add_argument('--folder', help='Location of dataset.')
    parser.add_argument(
        '--output-folder',
        type=str,
        default=None,
        help='Output folder for splits. Defaults to adjacent to ingest folder.',
    )
    parser.add_argument(
        '--copy',
        action='store_true',
        help='Copy files to output folder rather than symlinking.',
        default=False,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    folder = args.folder
    fps = glob(f'{folder}/{INGEST}/*.parquet')
    print(f'Found {len(fps)} files in {folder}/{INGEST}. Splitting into train, val, test sets...')
    dataset = args.folder.split('/')[-1]  # Infer name of dataset e.g. path/to/mros
    train, val, test = get_split(dataset, 'train'), get_split(dataset, 'val'), get_split(dataset, 'test')
    if args.output_folder is not None:
        output_folder = args.output_folder
    else:
        output_folder = folder
    num_train, num_val, num_test = 0, 0, 0
    for fp in tqdm(fps):
        o_fp = None
        session_id = os.path.basename(fp).replace('.parquet', '').replace('.issues', '')
        # Determine whether file is for train/val/test set.
        if session_id in train:
            o_fp = os.path.join(output_folder, f'{TRAIN}', os.path.basename(fp))
            num_train += 1
        elif session_id in val:
            o_fp = os.path.join(output_folder, f'{VAL}', os.path.basename(fp))
            num_val += 1
        elif session_id in test:
            o_fp = os.path.join(output_folder, f'{TEST}', os.path.basename(fp))
            num_test += 1
        else:
            # Ignored due e.g. to scoring issues.
            logger.debug(f'Session {session_id} not found in train/val/test sets.')
            continue
        if o_fp is not None:
            os.makedirs(os.path.dirname(o_fp), exist_ok=True)
            if args.copy:
                shutil.copy2(fp, o_fp)
            else:
                os.symlink(fp, o_fp)
    print(f'Created {num_train} train, {num_val} val, {num_test} test files.')


if __name__ == '__main__':
    main()
