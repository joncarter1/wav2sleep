"""Utility functions for reading one or more columns from an EDF file."""

import datetime

import numpy as np
import pandas as pd
import pyedflib

from ..settings import ABD, ECG, PPG, THX

# Possible alternative names for PSG signal columns in an EDF.
ALT_COLUMNS = {
    ECG: ('EKG', 'ECG1', 'ECG L', 'ECGL', 'ECG L-ECG R'),
    PPG: (
        'Pleth',
        'PlethWV',
        'PWF',
        'PlethMasimo',
        'PletMasimo',
        'PlethMasino',
        'PLETHMASIMO',
        'plethmasimo',
        'Plethmasimo',
    ),  # Handle typos galore in the CHAT dataset...
    ABD: ('Abdo', 'ABDO RES', 'ABDO EFFORT', 'Abdominal', 'abdomen'),
    THX: ('Thor', 'THOR RES', 'THOR EFFORT', 'Thoracic', 'Chest', 'thorax', 'CHEST'),
}
INV_ALT_COLUMNS = {v_i: k for k, v in ALT_COLUMNS.items() for v_i in v}


def get_column_match(target_col: str, available_cols: list[str], raise_error: bool = True):
    """Get a column from an EDF file that might be under an alternative name."""
    if target_col in available_cols:
        return target_col
    if target_col in ALT_COLUMNS:
        alt_col_names = ALT_COLUMNS[target_col]
        for alt_col in alt_col_names:
            if alt_col in available_cols:
                return alt_col
    if raise_error:
        raise KeyError(f'EDF has no signal called {target_col}')
    else:
        return None


def load_edf_data(
    filepath: str, columns: list[str], convert_time: bool = False, raise_on_missing: bool = True
) -> pd.DataFrame:
    """Load selected columns of EDF data into a Pandas DataFrame.

    timestamp | col 1 | col 2 | (label)

    Args:
        filepath (str): EDF filepath
        columns (str|list): Name of column or list of column names e.g. ['EEG', 'EKG']
    """
    if isinstance(columns, str):
        columns = [columns]
    with pyedflib.EdfReader(filepath) as f:
        signal_map = {}
        for ind, channel_dict in enumerate(f.getSignalHeaders()):  # Map channel names to numbers
            signal_map[channel_dict['label']] = ind
        dfs = []
        for sig_name in columns:
            actual_col_name = get_column_match(sig_name, signal_map.keys(), raise_error=raise_on_missing)
            if actual_col_name is None:
                continue
            idx = signal_map[actual_col_name]
            sig = f.readSignal(idx)
            sampling_freq = f.getSampleFrequency(idx)
            t = np.arange(0, len(sig)) / sampling_freq
            dfs.append(pd.DataFrame({sig_name: sig}, index=t))
        df = pd.concat(dfs, axis=1).sort_index()
        if convert_time:
            start = f.getStartdatetime()
            df.index = start + pd.TimedeltaIndex(df.index, unit='s')
    return df


def get_edf_start(filepath: str) -> datetime.datetime:
    with pyedflib.EdfReader(filepath) as f:
        return f.getStartdatetime()


def get_edf_end(filepath: str) -> datetime.datetime:
    with pyedflib.EdfReader(filepath) as f:
        return f.getStartdatetime() + datetime.timedelta(seconds=f.getFileDuration())


def get_edf_signals(filepath: str, convert: bool = True) -> dict[str, float]:
    """Get dict of signal names to sampling rates from an EDF."""
    with pyedflib.EdfReader(filepath) as f:
        channel_map = {
            channel_dict['label']: f.getSampleFrequency(ind) for ind, channel_dict in enumerate(f.getSignalHeaders())
        }
    if convert:  # Try to convert to common names e.g. EKG -> ECG
        channel_map = {INV_ALT_COLUMNS.get(k, k): v for k, v in channel_map.items()}
    return channel_map
