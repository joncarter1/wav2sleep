import logging

import numpy as np
import pandas as pd

from ..settings import ABD, COLS_TO_SAMPLES_PER_EPOCH, ECG, EOG_L, EOG_R, PPG, THX, TRAINING_LENGTH_HOURS
from .utils import interpolate_index

logger = logging.getLogger(__name__)

CARDIO_RESP_COLS = [ECG, PPG, ABD, THX]
NEURAL_COLS = [EOG_L, EOG_R]
EDF_COLS = CARDIO_RESP_COLS + NEURAL_COLS


TARGET_LABEL_INDEX = pd.Index(np.arange(0, TRAINING_LENGTH_HOURS * 60 * 60 + 1, 30.0)[1:])


def process_waveform_dataframe(df: pd.DataFrame, columns: list[str], max_length_hours: int = TRAINING_LENGTH_HOURS):
    """Process dataframe of waveform data.

    Args:
        df: DataFrame of waveform data.
        columns: List of columns to process.
        max_length_hours: Maximum length of the waveform data in hours.

    Returns:
        DataFrame of processed waveform data.
    """
    signals = []

    def _process_edf_column(col):
        """Process signal column of EDF"""
        samples_per_epoch = COLS_TO_SAMPLES_PER_EPOCH[col]  # Data points for each 30-second sleep epoch.
        target_index = pd.Index(np.arange(0, max_length_hours * 60 * 60 + 1e-9, 30 / samples_per_epoch)[1:])
        if col in df:
            resampled_wav = interpolate_index(df[col].dropna(), target_index, limit_area='inside').fillna(0.0)
            signals.append(resampled_wav)

    # Save start information.
    df_start = df.index[0]
    if isinstance(df.index, pd.DatetimeIndex):
        timestamp = True
        df.index = (df.index - df.index[0]).astype(int) / 10**9
    else:
        timestamp = False
    for col in columns:
        _process_edf_column(col)
    df = pd.concat(signals, axis=1).astype(np.float32)
    if timestamp:  # Restore timestamp information.
        df.index = df_start + pd.to_timedelta(df.index, unit='s')
    return df
