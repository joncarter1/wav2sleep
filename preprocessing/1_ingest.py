"""Process the NSRR datasets.

Turns EDF files into parquet files containing sleep labels and the signals used.
"""

import argparse
import logging
import os
from glob import glob

import numpy as np
import pandas as pd
import ray
from tqdm import tqdm

from wav2sleep.data.edf import load_edf_data
from wav2sleep.data.txt import parse_txt_annotations
from wav2sleep.data.utils import interpolate_index
from wav2sleep.data.xml import parse_xml_annotations
from wav2sleep.parallel import parallelise
from wav2sleep.settings import ABD, CHAT, ECG, INGEST, MROS, PPG, SHHS, THX, WSC

logger = logging.getLogger(__name__)

EDF_COLS = [ECG, PPG, ABD, THX]

MAX_LENGTH = 60 * 60 * 10  # Recording length in seconds (trimmed to 10h)
TARGET_LABEL_INDEX = pd.Index(np.arange(0, MAX_LENGTH + 1, 30.0)[1:])
HIGH_FREQ_SIGNAL_INDEX = pd.Index(np.arange(0, MAX_LENGTH + 1e-9, 30 / 1024)[1:])  # ~ 34 Hz
LOW_FREQ_SIGNAL_INDEX = pd.Index(np.arange(0, MAX_LENGTH + 1e-9, 30 / 256)[1:])  # ~ 8 Hz


def process_edf(edf: pd.DataFrame):
    """Process dataframe of EDF data."""
    signals = []

    def _process_edf_column(col, target_index):
        """Process signal column of EDF"""
        if col in edf:
            resampled_wav = interpolate_index(edf[col].dropna(), target_index)
            normalized_wav = (resampled_wav - resampled_wav.mean()) / resampled_wav.std()
            signals.append(normalized_wav)

    _process_edf_column(ECG, HIGH_FREQ_SIGNAL_INDEX)
    _process_edf_column(PPG, HIGH_FREQ_SIGNAL_INDEX)
    _process_edf_column(ABD, LOW_FREQ_SIGNAL_INDEX)
    _process_edf_column(THX, LOW_FREQ_SIGNAL_INDEX)
    return pd.concat(signals, axis=1).astype(np.float32)


def process(edf_fp: str, label_fp: str, output_fp: str, overwrite: bool = False) -> bool:
    """Process night of data."""
    if os.path.exists(output_fp) and not overwrite:
        logger.debug(f'Skipping {edf_fp=}, {output_fp=}, already exists')
        return False
    else:
        os.makedirs(os.path.dirname(output_fp), exist_ok=True)
    # Process labels
    if label_fp.endswith('.xml'):
        try:
            labels = parse_xml_annotations(label_fp)
        except Exception as e:
            logger.error(f'Failed to parse: {label_fp}.')
            logger.error(e)
            return False
    else:
        labels = parse_txt_annotations(fp=label_fp)
        if labels is None:
            logger.error(f'Failed to parse: {label_fp}.')
            return False
    labels = labels.reindex(TARGET_LABEL_INDEX).fillna(-1)
    # Check for N1, N3 or REM presence. (Recordings with just sleep-wake typically use N2 as sole sleep class)
    stage_counts = labels.value_counts()
    if stage_counts.get(1.0) is None and stage_counts.get(3.0) is None and stage_counts.get(4.0) is None:
        logger.error(f'No N1, N3 or REM in {label_fp}.')
        output_fp = output_fp.replace('.parquet', '.issues.parquet')
    edf = load_edf_data(edf_fp, columns=EDF_COLS, raise_on_missing=False)
    waveform_df = process_edf(edf)
    output_df = pd.concat([waveform_df, labels], axis=1)
    output_df.to_parquet(output_fp)
    return True


def get_edf_path(session_id: str, dataset: str, folder: str):
    if dataset == SHHS:
        partition, _ = session_id.split('-')  # shhs1 or shhs2
        edf_fp = os.path.join(folder, 'polysomnography/edfs', partition, f'{session_id}.edf')
    elif dataset == MROS:
        _, partition, *_ = session_id.split('-')  # mros visit 1 or 2
        edf_fp = os.path.join(folder, 'polysomnography/edfs', partition, f'{session_id}.edf')
    elif dataset == CHAT:
        if 'nonrandomized' in session_id:  # e.g. chat-baseline-nonrandomized-xxxx
            partition = 'nonrandomized'
        else:
            partition = session_id.split('-')[1]  # e.g. chat-baseline-xxxx
        edf_fp = os.path.join(folder, 'polysomnography/edfs', partition, f'{session_id}.edf')
        fixed_edf_fp = edf_fp.replace('.edf', '_fixed.edf')
        # Check for existence of fixed EDF (physical maximum is 0.0 in some files and needed fixing.)
        if os.path.exists(fixed_edf_fp):
            edf_fp = fixed_edf_fp
    else:
        edf_fp = os.path.join(folder, 'polysomnography/edfs', f'{session_id}.edf')
    return edf_fp


def prepare_dataset(folder: str, output_folder: str, dataset: str):
    """Prepare dataset IO locations for parallel processing."""
    # WSC uses .txt annotation files
    fp_dict = {}
    if dataset == WSC:
        edf_fps = glob(f'{folder}/polysomnography/*.edf', recursive=True)
        label_fps = []
        for edf_fp in edf_fps:
            all_score_fp = edf_fp.replace('.edf', '.allscore.txt')
            stg_fp = edf_fp.replace('.edf', '.stg.txt')
            if os.path.exists(stg_fp):
                label_fp = stg_fp
            elif os.path.exists(all_score_fp):
                label_fp = all_score_fp
            else:
                continue
            session_id = os.path.basename(edf_fp).replace('.edf', '')
            output_fp = os.path.join(output_folder, dataset, INGEST, f'{session_id}.parquet')
            fp_dict[session_id] = {'edf_fp': edf_fp, 'label_fp': label_fp, 'output_fp': output_fp}
        return fp_dict
    # Other datasets use NSRR standardized XML files
    label_fps = glob(f'{folder}/polysomnography/annotations-events-nsrr/**/**.xml', recursive=True)
    for label_fp in label_fps:
        session_id = os.path.basename(label_fp).replace('-nsrr.xml', '')
        edf_fp = get_edf_path(session_id, dataset, folder)
        if not os.path.exists(edf_fp):
            logger.warning(f"{edf_fp=} doesn't exist. Skipping...")
            continue
        output_fp = os.path.join(output_folder, dataset, INGEST, f'{session_id}.parquet')
        fp_dict[session_id] = {'edf_fp': edf_fp, 'label_fp': label_fp, 'output_fp': output_fp}
    return fp_dict


def process_files(
    fp_dict: dict[str, dict[str, str]], max_parallel: int = 1, overwrite: bool = False, address: str = 'local'
):
    print(f'Preparing to process {len(fp_dict)} files.')

    def proc(arg_dict):
        try:
            return process(overwrite=overwrite, **arg_dict)
        except Exception as e:
            logger.error(f'Failed on {arg_dict} - {e}')
            print(f'Failed on {arg_dict} - {e}')
            return False

    if max_parallel > 1:
        MEM_GB = int(os.environ.get('SLURM_MEM_PER_NODE', 8 * 1024)) / 1024  # SLURM env var. is in MB
        ray.init(
            address=address,
            num_cpus=max_parallel,
            object_store_memory=MEM_GB * 1024**3,
            ignore_reinit_error=True,
            include_dashboard=False,
        )
        num_converted = sum(parallelise(proc, fp_dict.values(), use_tqdm=True, max_parallel=max_parallel))
    else:
        num_converted = 0
        for fp_map in tqdm(fp_dict.values()):
            num_converted += process(**fp_map)
    print(f'Converted {num_converted} files.')


def parse_args():
    parser = argparse.ArgumentParser(prog='Dataset Processor', description='Process dataset.')
    parser.add_argument('--folder', help='Location of dataset.')
    parser.add_argument('--max-parallel', default=1, type=int, help='Parallel processes.')
    parser.add_argument('--cluster-address', default='local', type=str, help='Ray cluster address (defaults to local).')
    parser.add_argument('--output-folder', required=True, help='Base output folder for processed datasets.')
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing parquet files.',
        default=False,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = args.folder.split('/')[-1]  # Infer name of dataset e.g. path/to/mros
    print(f'Processing {dataset=}...')
    fp_dict = prepare_dataset(folder=args.folder, output_folder=args.output_folder, dataset=dataset)
    process_files(fp_dict, max_parallel=args.max_parallel, overwrite=args.overwrite, address=args.cluster_address)


if __name__ == '__main__':
    main()
