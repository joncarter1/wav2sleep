import os
from glob import glob

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def get_parquet_cols(fp: str) -> set[str]:
    """Get list of column names from parquet file."""
    cols = pq.read_schema(fp, memory_map=True).names
    if '__index_level_0__' in cols:
        cols.remove('__index_level_0__')
    return cols


def get_parquet_fps(folder: str, recursive: bool = False) -> list[str]:
    """Return parquet files in a folder."""
    if not os.path.exists(folder):
        raise FileNotFoundError(folder)
    if recursive:
        return glob(f'{folder}/**/*.parquet', recursive=True)
    else:
        return glob(f'{folder}/*.parquet')


def convert_int_stage(stage: int | str):
    stage = int(stage)
    if stage not in [0, 1, 2, 3, 4, 5, 6, 7, 9]:  # 6 = mvmt, 9 = unscored
        raise ValueError(f'{stage} not a valid sleep stage.')
    # Map any N4 to 3 (N3), REM to 4.
    if stage == 4:
        stage = 3
    elif stage == 5:
        stage = 4
    elif stage in [6, 7, 9]:
        stage = np.nan
    return stage


def convert_str_stage(stage: str):
    if 'STAGE' not in stage:
        return None
    if 'NO STAGE' in stage:
        return None
    elif 'W' in stage:
        return 0
    elif 'N1' in stage:
        return 1
    elif 'N2' in stage:
        return 2
    elif 'N3' in stage:
        return 3
    elif 'R' in stage:
        return 4
    elif 'MVT' in stage:
        return None
    else:
        raise ValueError(f'Encountered unseen value: {stage=}')


def interpolate_index(
    source_df: pd.Series | pd.DataFrame,
    target_index: pd.Index | pd.DatetimeIndex,
    method: str | None = None,
    squeeze: bool = True,
    **kwargs,
) -> pd.Series | pd.DataFrame:
    """Re-sample pandas Data (Series or DataFrame) to match a target index.

    This function takes an input pandas series or dataframe (source_df)
    and interpolates/resamples it to align with a target_index

    kwargs are passed as parameters to the interpolate method:
    https://pandas.pydata.org/docs/reference/api/pandas.Series.interpolate.html
    """
    if target_index.__class__ != source_df.index.__class__:
        raise ValueError('target_index must be the same type as the source_index.')
    if method is None:
        if isinstance(source_df.index, (pd.DatetimeIndex, pd.TimedeltaIndex)):
            method = 'time'
            # Check for timezone info - think this still needs to be removed before for interpolation.
            if hasattr(source_df.index, 'tz_localize'):
                raise ValueError('sourcet_index contains TZ information. Remove and reinstate for interpolation.')
            elif hasattr(target_index, 'tz_localize'):
                raise ValueError('target_index contains TZ information. Remove and reinstate for interpolation.')
        else:
            method = 'index'
    # Add NaNs at interpolation timestamps where no data is available
    if isinstance(source_df, pd.Series):
        source_df = source_df.to_frame()
    nan_padded_df = source_df.join(pd.DataFrame(index=target_index), how='outer')
    # Re-sample dataframe at timestamps then slice at the interpolated values
    resampled_df = nan_padded_df.interpolate(method=method, limit_direction='both', **kwargs).loc[target_index]
    if squeeze:
        return resampled_df.squeeze(axis='columns')
    else:
        return resampled_df
