"""Fix broken EDF signals (physical maximum of 0.0) by setting them to a fill value.

Broken signals discovered: CHIN, SNORE 2, Rchin, SNORE2, Chin, ECG

We fill with sensible physical min/max from other files.
Given signals are unit-normalized, this shouldn't affect the final processed ECG signals.
This fix allows the files to be read using the pyedf library.
"""

import argparse
import os
import shutil
import tempfile
from glob import glob

import pyedflib
from tqdm import tqdm


def _fix_edf_header(filename, fix_dict, fill_val: float = 3.28):
    """Fix signals that have a physical maximum of 0.0.

    Instead we set them to a reasonable fill_val.
    Also marks the unit as 'BROKEN' so the EDF reader knows to skip this channel.
    """
    for _, (pos_unit, pos_min, pos_max) in fix_dict.items():
        with open(filename, 'rb+') as f:
            # Mark unit as BROKEN so reader skips this channel
            f.seek(pos_unit)
            f.write('BROKEN'.ljust(8).encode())
            # Fix physical min/max so file is readable
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
            if float(pmax_val) == 0.0:  # Store unit, min, and max positions in the header.
                pos_unit = pos - 16 * nsigs  # unit field is 16*nsigs bytes before pmax
                pos_min = pos - 8 * nsigs
                fix_dict[label[i]] = (pos_unit, pos_min, pos)
    return fix_dict


def try_read_edf(fp: str) -> bool:
    try:
        with pyedflib.EdfReader(fp):
            return True
    except OSError as e:
        print(f'Failed to read {fp} due to {e}')
        return False


def triage_edf_fp(filename: str, overwrite: bool = False) -> bool:
    """Triage EDF files for broken signals."""
    fixed_filename = filename.replace('.edf', '_fixed.edf')
    if os.path.exists(fixed_filename) and not overwrite:
        return False
    broken_signals = _debug_header(filename)
    if not bool(broken_signals):
        return False
    with tempfile.NamedTemporaryFile() as tmp_file:
        shutil.copyfile(filename, tmp_file.name)
        _fix_edf_header(tmp_file.name, broken_signals)
        if try_read_edf(tmp_file.name):
            shutil.copyfile(tmp_file.name, fixed_filename)
            return True
    return False


def parse_args():
    parser = argparse.ArgumentParser(prog='Fix EDFs', description='Fix EDFs from the CHAT dataset.')
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
