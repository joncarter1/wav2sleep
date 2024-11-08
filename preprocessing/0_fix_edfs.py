"""Fix broken EDF signals (physical maximum of 0.0) by setting them to a fill value.

Broken signals discovered: CHIN, SNORE 2, Rchin, SNORE2, Chin, ECG

We fill with sensible physical min/max from other files.
Given signals are unit-normalized, this shouldn't affect the final processed ECG signals.
"""

import argparse
import os
import shutil
from glob import glob

import pyedflib
from tqdm import tqdm


def _fix_edf_header(filename, fix_dict, fill_val: float = 3.28):
    """Fix signals that a physical maximum of 0.0.

    Instead we set them to a reasonable fill_val.
    """
    for _, (pos_min, pos_max) in fix_dict.items():
        with open(filename, 'rb+') as f:
            f.seek(pos_min)
            f.write(f'{-fill_val:.2f}'.ljust(8).encode())
            f.seek(pos_max)
            f.write(f'{fill_val:.2f}'.ljust(8).encode())
    return


def _debug_header(filename):
    """Find out which signals need fixing.

    Inspired by pyedflib.edfreader._debug_parse_header"""
    with open(filename, 'rb') as f:
        f.seek(252)
        nsigs = int(f.read(4).decode())
        label = [f.read(16).decode() for i in range(nsigs)]
        pmax_start = 256 + (16 * nsigs) + (80 * nsigs) + (8 * nsigs) + (8 * nsigs)
        f.seek(pmax_start)
        fix_dict = {}  # Store mapping of broken signal names to positions in the files.
        for i in range(nsigs):
            pos = f.tell()
            pmax_val = f.read(8).decode()
            if float(pmax_val) == 0.0:  # Store the min and max position in the header.
                fix_dict[label[i]] = (pos - 8 * nsigs, pos)
    return fix_dict


def triage_edf_fp(filename: str, overwrite: bool = False) -> bool:
    """Triage EDF files for broken signals."""
    broken_signals = _debug_header(filename)
    if not bool(broken_signals):
        return False
    fixed_filename = filename.replace('.edf', '_fixed.edf')
    if os.path.exists(fixed_filename) and not overwrite:
        return False
    shutil.copyfile(filename, fixed_filename)
    _fix_edf_header(fixed_filename, broken_signals)
    try:
        with pyedflib.EdfReader(fixed_filename):
            return True
    except OSError as e:
        print(f'Failed to fix {filename} due to {e}')
        return False


def parse_args():
    parser = argparse.ArgumentParser(prog='Fix EDFsr', description='Fix EDFs from the CHAT dataset.')
    parser.add_argument('--folder', help='Location of CHAT dataset.')
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing fixed EDF files.',
        default=False,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    folder = args.folder
    overwrite = args.overwrite
    edf_fps = glob(f'{folder}/**/*.edf', recursive=True)
    print(f'Found {len(edf_fps)} EDF files.')
    fixed = 0
    for fp in tqdm(edf_fps):
        fixed += triage_edf_fp(fp, overwrite=overwrite)
    print(f'Fixed {fixed} EDF files.')


if __name__ == '__main__':
    main()
