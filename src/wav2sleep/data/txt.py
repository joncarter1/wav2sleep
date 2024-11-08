"""Code to convert Wisconsin Sleep Cohort (WSC) sleep stage annotations to consistent format."""

import datetime
import logging
import os

import numpy as np
import pandas as pd

from ..settings import LABEL, TIMESTAMP
from .edf import get_edf_end, get_edf_start
from .utils import convert_int_stage, convert_str_stage

logger = logging.getLogger(__name__)

# Check recordings between 4 and 14 hours long.
MIN_RECORDING_LENGTH = 60 * 60 * 4
MAX_RECORDING_LENGTH = 60 * 60 * 14


def convert_index(hour_minute_index, start_ts: datetime.datetime) -> pd.DatetimeIndex:
    """Convert hh:mm:.. string timestamp index to datetimes.

    This function expects recordings to start between 5pm and 3am to correctly convert
    timestamps in hh:mm format to the correct datetime.
    """
    # Get hour for each timestamp.
    hours = hour_minute_index.str.slice(start=0, stop=2).astype(int)
    # Assuming recording is less than 24h, should span two days if end hour is less than start.
    end_hour = hours[-1]
    if start_ts.hour < end_hour:
        multiday = False
    else:
        multiday = True
    start_day = start_ts.date()
    start_day_str = start_day.strftime('%Y/%m/%d')
    end_day = start_day + datetime.timedelta(days=1)
    end_day_str = end_day.strftime('%Y/%m/%d')
    same_day = start_ts.hour <= hours
    if not multiday and ((~same_day).any()):
        raise ValueError(f'{multiday=}, but {start_ts=} and found timestamps on a possibly different day.')
    day_strings = np.repeat(start_day_str, len(hour_minute_index))
    day_strings[~same_day] = end_day_str
    string_index = day_strings + (' ' + hour_minute_index)
    datetime_index = pd.DatetimeIndex(string_index)
    return datetime_index


def parse_all_score(fp, convert_time: bool = False) -> pd.Series | None:
    """Parse all_score files from WSC.

    (From the WSC docs): There is a stage indicated only when there was a stage change..
    There is not a stage indicated for every epoch
    (e.g. assume they stayed in the previous stage until you see an indication for the next stage)
    """
    txt = (
        pd.read_csv(fp, encoding='unicode_escape', delimiter='\t', index_col=0, header=None)
        .squeeze()
        .dropna()
        .rename(LABEL)
    )
    txt.index = txt.index.rename(TIMESTAMP)
    # Exclude recordings with power failures.
    if txt.str.contains('POWER FAILURE RECOVERY').any():
        logger.info(f'Recording contains power failure for {fp=}')
        return None
    # Remove extra annotations
    df = txt[txt.str.contains('STAGE|START')]
    # Check for start annotation
    if not df.str.contains('START RECORDING').any():
        logger.info(f"Didn't find START RECORDING annotation for {fp=}.")
        return None
    # Remove any annotations before 'START RECORDING'
    # (Found atleast one record with annotations long before 'START RECORDING' i.e. at like 4 pm.)
    df_ = df.reset_index()
    start_pos = df_.query(f'{LABEL} == "START RECORDING"').index[0]
    df_ = df_.iloc[start_pos:]
    df = df_.set_index(TIMESTAMP).squeeze()
    start = df[df == 'START RECORDING']
    if len(start) != 1:
        logger.info(f'Found multiple START RECORDING annotations for {fp=}.')
        return None
    # Check start matches the EDF
    hour, minute, second = int(start.index[0][:2]), int(start.index[0][3:5]), int(start.index[0][6:8])
    # Check EDF start time.
    edf_fp = fp.replace('allscore.txt', 'edf')
    edf_start = get_edf_start(edf_fp)
    if edf_start.hour != hour or edf_start.minute != minute or edf_start.second != second:
        logger.warning(f'{edf_start=} did not match allscore file start: {start.index[0]} for {fp=}. Skipping...')
        raise ValueError
        return None
    # Create mock timestamp to create timedeltas.
    start_ts = f'01/01/2000 {start.index[0]}'
    start_ts = datetime.datetime.strptime(start_ts, '%d/%m/%Y %H:%M:%S.%f')
    # Convert H:M:S timestamps into absolute timestamps
    try:
        df.index = convert_index(df.index, start_ts=start_ts)  # type: ignore
    except Exception as e:
        logger.warning(f'Failed to convert timestamps for {fp=}')
        return None
    # Remove index duplicates if present
    df = df.loc[~df.index.duplicated()]
    df.index = df.index - df.index[0]
    # Forward fill at 30s intervals.
    df = df.resample('30s').ffill()
    df.index = df.index.total_seconds()
    df = df.map(convert_str_stage)
    # Switch labels to correspond to previous 30 seconds. (Assuming existing labels correspond to start)
    df.index += 30.0
    # Check assumption that timestamps were sorted. (Might break index converter otherwise)
    if not (df.sort_index().index == df.index).all():
        logger.warning(f"Timestamps in {fp=} weren't already sorted.")
        return None
    if df.index[-1] < MIN_RECORDING_LENGTH:
        logger.warning(f'Recording less than {MIN_RECORDING_LENGTH=} for {fp=}')
        return None
    elif df.index[-1] > MAX_RECORDING_LENGTH:
        logger.warning(f'Recording greater than {MAX_RECORDING_LENGTH=} for {fp=}')
        return None
    stage_counts = df.value_counts(dropna=False)
    # Check for N1, N3 or REM presence. (Recordings with just sleep-wake typically use N2 as sole sleep class)
    if stage_counts.get(1.0) is None and stage_counts.get(3.0) is None and stage_counts.get(4.0) is None:
        print(df)
        print(stage_counts)
        print(fp)
        raise ValueError
    if convert_time:
        df.index = edf_start + pd.TimedeltaIndex(df.index, unit='s')
    return df


def midnight_dist(hh_mm_string: str):
    hour = float(hh_mm_string[0][:2])
    minute = float(hh_mm_string[0][3:5])
    return (hour + minute / 60 - 24) % 24


def get_start_from_log(fp: str) -> pd.Series:
    """Determine the start of the recording from the .log.txt file.

    For some recordings, there are restarts, so we assume the closest to
    midnight is the true start then check that matches the EDF.

    This process could be cleaner, and does discard a small number of EDF files.
    """
    with open(fp, 'r') as f:
        log_contents = f.readlines()
    starts = []
    for line in log_contents:
        contents = line.strip().split('\t')
        if len(contents) < 2:
            continue
        time_epoch, annotation, *_ = contents
        if annotation == 'Recording Started':
            time, epoch_no = time_epoch.split(' ', maxsplit=1)
            starts.append((time.strip(), epoch_no.strip()))
    if not bool(starts):
        logger.warning(f"Couldn't find 'Recording Start' annotation in {fp=}")
        return None, None
    if len(starts) > 1:
        logger.warning(f'Found multiple starts in {fp=}. Using closest to midnight...')
        starts = sorted(starts, key=midnight_dist)
    return starts[-1]


COL = 'User-Defined Stage'


def parse_stg_file(fp, convert_time: bool = False) -> pd.Series | None:
    # "wsc-visit1-10119-nsrr.stg.txt"
    df = pd.read_csv(fp, index_col=0, delimiter='\t').squeeze()
    # Some files are seemingly missing the header.
    if COL in df.columns:
        df = df[COL]
    else:
        df = pd.read_csv(fp, index_col=0, delimiter='\t', names=[COL, 'X'])[COL]
    df = df.rename(LABEL)
    # Get start time from log.
    log_fp = fp.replace('stg', 'log')
    if not os.path.exists(log_fp):
        raise FileNotFoundError(f"Couldn't find corresponding log file for {fp=}")
    start_time, epoch = get_start_from_log(log_fp)
    if start_time is None:
        return None
    hour, minute, second = map(int, start_time.split(':'))
    # Check EDF start time.
    edf_fp = fp.replace('stg.txt', 'edf')
    edf_start, edf_end = get_edf_start(edf_fp), get_edf_end(edf_fp)
    if edf_start.hour != hour or edf_start.minute != minute or edf_start.second != second:
        logger.warning(f'{edf_start=} did not match log file start: {start_time} for {fp=}. Skipping...')
        return None
    edf_duration = (edf_end - edf_start).total_seconds()
    if edf_duration < MIN_RECORDING_LENGTH:
        logger.warning(f'EDF less than {MIN_RECORDING_LENGTH=} for {fp=}')
        return None
    elif edf_duration > MAX_RECORDING_LENGTH:
        logger.warning(f'EDF greater than {MAX_RECORDING_LENGTH=} for {fp=}')
        return None
    # Convert epoch index to seconds from start.
    # Start is 1st epoch so labels already correspond to right bin edge.
    df.index *= 30.0
    df = df.map(convert_int_stage)
    stage_counts = df.value_counts(dropna=False)
    # Check for N1, N3 or REM presence. (Recordings with just sleep-wake typically use N2 as sole sleep class)
    if stage_counts.get(1.0) is None and stage_counts.get(3.0) is None and stage_counts.get(4.0) is None:
        print(df)
        print(stage_counts)
        print(fp)
        raise ValueError
    if convert_time:
        df.index = edf_start + pd.TimedeltaIndex(df.index, unit='s')
    return df


def parse_txt_annotations(fp: str):
    """Parse annotations from a WSC .txt file"""
    if fp.endswith('.stg.txt'):
        return parse_stg_file(fp)
    elif fp.endswith('.allscore.txt'):
        return parse_all_score(fp)
    else:
        raise ValueError(f'File extension {fp=} unsupported. Expected .stg.txt or .allscore.txt')
