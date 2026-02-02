"""Process the NSRR datasets.

Turns EDF files into parquet files containing sleep labels and the signals used.
"""

import argparse
import json
import logging
import os
from glob import glob
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from wav2sleep.data.edf import INV_ALT_UNIT_NAMES, VOLTAGE_SIGNALS, load_edf_data
from wav2sleep.data.preprocessing import TARGET_LABEL_INDEX, process_waveform_dataframe
from wav2sleep.data.txt import parse_txt_annotations
from wav2sleep.data.xml import parse_xml_annotations
from wav2sleep.parallel import parallelise
from wav2sleep.settings import (
    ABD,
    CCSHS,
    CFS,
    CHAT,
    ECG,
    EOG_L,
    EOG_R,
    INGEST,
    MESA,
    MROS,
    PPG,
    SHHS,
    THX,
    WSC,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

CARDIO_RESP_COLS = [ECG, PPG, ABD, THX]
NEURAL_COLS = [EOG_L, EOG_R]
EDF_COLS = CARDIO_RESP_COLS + NEURAL_COLS

PROJECT_DIR = str(Path(__file__).parent.parent.parent)

# Valid voltage units (from edf.py)
VALID_VOLTAGE_UNITS = set(INV_ALT_UNIT_NAMES.keys())


def check_voltage_signal_units(signal_metadata: dict[str, dict]) -> list[str]:
    """Check if voltage signals have valid units.

    Returns list of signal names with invalid units.
    """
    invalid_signals = []
    for sig_name, meta in signal_metadata.items():
        if sig_name in VOLTAGE_SIGNALS:
            unit = meta.get('unit', '').strip()
            if unit not in VALID_VOLTAGE_UNITS:
                invalid_signals.append(f"{sig_name} (unit='{unit}')")
    return invalid_signals


# Minimum std threshold for a signal to be considered "alive" (not flat/dead)
MIN_SIGNAL_STD = 0.001


def check_and_drop_flat_signals(
    edf: pd.DataFrame,
    signal_metadata: dict[str, dict],
    signals_to_check: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict], list[str]]:
    """Check for flat/dead signals and drop them from the DataFrame.

    Args:
        edf: DataFrame with signal data
        signal_metadata: Dict of signal metadata from load_edf_data
        signals_to_check: List of signals to check (default: PPG only)

    Returns:
        edf: DataFrame with flat signals removed
        signal_metadata: Updated metadata with flat signals removed
        dropped: List of dropped signal names
    """
    if signals_to_check is None:
        signals_to_check = [PPG]  # Only check PPG by default

    dropped = []
    for sig_name in signals_to_check:
        if sig_name in edf.columns:
            std = edf[sig_name].std()
            if std < MIN_SIGNAL_STD:
                logger.warning(f'Dropping flat signal {sig_name} (std={std:.6f})')
                edf = edf.drop(columns=[sig_name])
                if sig_name in signal_metadata:
                    del signal_metadata[sig_name]
                dropped.append(sig_name)

    return edf, signal_metadata, dropped


def process_night(
    edf_fp: str,
    label_fp: str | None,
    output_fp: str,
    columns: list[str],
    overwrite: bool = False,
) -> bool:
    """Process night of data."""
    if os.path.exists(output_fp) and not overwrite:
        logger.debug(f'Skipping {edf_fp=}, {output_fp=}, already exists')
        return False
    else:
        os.makedirs(os.path.dirname(output_fp), exist_ok=True)
    # Process labels
    if label_fp is not None:
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
    else:
        labels = None
    # Process signals (now returns tuple with metadata)
    edf, signal_metadata = load_edf_data(edf_fp, columns=columns, raise_on_missing=False)

    # Check for voltage signals with invalid units (ECG, EOG)
    invalid_voltage_signals = check_voltage_signal_units(signal_metadata)
    if invalid_voltage_signals:
        logger.warning(f'{edf_fp}: Invalid units for voltage signals: {invalid_voltage_signals}')
        output_fp = output_fp.replace('.parquet', '.issues.parquet')

    # Check for and drop flat/dead signals (PPG)
    edf, signal_metadata, dropped_signals = check_and_drop_flat_signals(edf, signal_metadata)

    waveform_df = process_waveform_dataframe(edf, columns=columns)
    if labels is None:
        output_df = waveform_df
    else:
        output_df = pd.concat([waveform_df, labels], axis=1)

    # Store signal metadata in parquet file metadata
    # This enables later reconstruction of normalization for real-time inference
    table = pa.Table.from_pandas(output_df)
    # Add signal metadata to the parquet schema metadata
    existing_metadata = table.schema.metadata or {}
    existing_metadata[b'signal_metadata'] = json.dumps(signal_metadata).encode('utf-8')
    table = table.replace_schema_metadata(existing_metadata)
    pq.write_table(table, output_fp)
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
        edf_fps = glob(f'{folder}/**/*.edf', recursive=True)
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
    # Other datasets use NSRR standardized XML files
    elif dataset in (SHHS, MROS, CHAT, MESA, CCSHS, CFS):
        label_fps = glob(f'{folder}/polysomnography/annotations-events-nsrr/**/**.xml', recursive=True)
        for label_fp in label_fps:
            session_id = os.path.basename(label_fp).replace('-nsrr.xml', '')
            edf_fp = get_edf_path(session_id, dataset, folder)
            if not os.path.exists(edf_fp):
                logger.warning(f"{edf_fp=} doesn't exist. Skipping...")
                continue
            output_fp = os.path.join(output_folder, dataset, INGEST, f'{session_id}.parquet')
            fp_dict[session_id] = {'edf_fp': edf_fp, 'label_fp': label_fp, 'output_fp': output_fp}
    else:
        logger.warning(f'Unknown dataset: {dataset}. Only processing EDF files.')
        edf_fps = glob(f'{folder}/**/*.edf', recursive=True)
        for edf_fp in edf_fps:
            fixed_edf_fp = edf_fp.replace('.edf', '_fixed.edf')
            # Check for existence of fixed EDF (physical maximum is 0.0 in some files and needed fixing.)
            if os.path.exists(fixed_edf_fp):
                continue
            output_fp = edf_fp.replace(folder, output_folder).replace('.edf', '.parquet')
            fp_dict[os.path.basename(edf_fp)] = {'edf_fp': edf_fp, 'label_fp': None, 'output_fp': output_fp}
    return fp_dict


def process_files(
    fp_dict: dict[str, dict[str, str]],
    max_parallel: int = 1,
    overwrite: bool = False,
    columns: list[str] = EDF_COLS,
):
    print(f'Preparing to process {len(fp_dict)} files.')

    def proc(arg_dict):
        try:
            return process_night(columns=columns, overwrite=overwrite, **arg_dict)
        except Exception as e:
            logger.error(f'Failed on {arg_dict} - {e}')
            print(f'Failed on {arg_dict} - {e}')
            return False

    if max_parallel > 1:
        num_converted = sum(parallelise(proc, fp_dict.values(), use_tqdm=True, max_parallel=max_parallel))
    else:
        num_converted = 0
        for fp_map in tqdm(fp_dict.values()):
            num_converted += process_night(columns=columns, **fp_map)
    print(f'Converted {num_converted} files.')


def parse_args():
    parser = argparse.ArgumentParser(prog='Dataset Processor', description='Process dataset.')
    parser.add_argument('--folder', help='Location of dataset.')
    parser.add_argument('--columns', nargs='+', help='Signals to process. (e.g. ECG PPG ABD THX)')
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
    # Prepare dataset
    print('Preparing dataset...')
    fp_dict = prepare_dataset(folder=args.folder, output_folder=args.output_folder, dataset=dataset)
    # Process files
    print('Processing files...')
    process_files(
        fp_dict,
        max_parallel=args.max_parallel,
        overwrite=args.overwrite,
        columns=args.columns,
    )


if __name__ == '__main__':
    main()
