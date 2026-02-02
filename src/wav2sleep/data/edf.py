"""Utility functions for reading one or more columns from an EDF file."""

import datetime
import logging

import numpy as np
import pandas as pd
import pyedflib

from ..settings import ABD, ECG, EOG_L, EOG_R, PPG, THX, TIMESTAMP

_logger = logging.getLogger(__name__)

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
    EOG_L: ('EOG-L', 'EOG(L)', 'E1', 'LOC', 'EOGl'),
    EOG_R: ('EOG-R', 'EOG(R)', 'E2', 'ROC', 'EOGr'),
}
INV_ALT_COLUMNS = {v_i: k for k, v in ALT_COLUMNS.items() for v_i in v}

MICRO_V = 'uV'
MILLI_V = 'mV'
VOLTS = 'V'
ALT_UNIT_NAMES = {
    MICRO_V: {'uV', 'uv'},
    MILLI_V: {'mV', 'mv'},
    VOLTS: {'V', 'v', 'Volts'},
}
INV_ALT_UNIT_NAMES = {v_i: k for k, v in ALT_UNIT_NAMES.items() for v_i in v}

# Signals that are true voltage signals (apply unit conversion to mV)
VOLTAGE_SIGNALS = {ECG, EOG_L, EOG_R}

# Signals with arbitrary units (normalize using physical range instead)
ARBITRARY_UNIT_SIGNALS = {ABD, THX, PPG}

# Scale voltages to mV
UNIT_SCALING = {
    MICRO_V: 1e-3,
    MILLI_V: 1,
    VOLTS: 1e3,
}


def get_unit_scaling(col: str, unit: str) -> float:
    """Get scaling factor for voltage signals only (ECG, EOG).

    Non-voltage signals (ABD, THX, PPG) return 1.0 - they should use
    physical range normalization instead.

    For blank or unknown units, returns 1.0 (no scaling) and logs a warning.
    """
    # Only apply voltage scaling to true voltage signals
    if col not in VOLTAGE_SIGNALS:
        return 1.0

    # Handle blank/whitespace-only units
    unit_stripped = unit.strip()
    if not unit_stripped:
        _logger.warning(f"Blank unit for voltage signal '{col}' - assuming no scaling needed")
        return 1.0

    # Look up known units
    if unit_stripped in ALT_UNIT_NAMES:
        return UNIT_SCALING[unit_stripped]
    elif unit_stripped in INV_ALT_UNIT_NAMES:
        return UNIT_SCALING[INV_ALT_UNIT_NAMES[unit_stripped]]
    else:
        _logger.warning(f"Unknown unit '{unit}' for voltage signal '{col}' - assuming no scaling needed")
        return 1.0


BROKEN_UNIT = 'BROKEN'


def get_column_match(
    target_col: str,
    available_cols: list[str],
    units_map: dict[str, str] | None = None,
    raise_error: bool = True,
):
    """Get a column from an EDF file that might be under an alternative name.

    Args:
        target_col: The canonical column name we want (e.g., 'ECG')
        available_cols: List of column names in the EDF file
        units_map: Optional dict of {col_name: unit_string} to check for broken channels
        raise_error: If True, raise KeyError if no match found

    Returns:
        The matched column name, or None if not found and raise_error=False
    """

    def is_broken(col: str) -> bool:
        """Check if a column is marked as broken."""
        if units_map is None:
            return False
        return units_map.get(col, '').strip().upper() == BROKEN_UNIT

    # Try exact match first (skip if broken)
    if target_col in available_cols and not is_broken(target_col):
        return target_col

    # Try alternative names
    if target_col in ALT_COLUMNS:
        alt_col_names = ALT_COLUMNS[target_col]
        for alt_col in alt_col_names:
            if alt_col in available_cols and not is_broken(alt_col):
                return alt_col

    if raise_error:
        raise KeyError(f'EDF has no valid signal called {target_col}')
    else:
        return None


def _warn_signal_issues(
    filepath: str,
    sig_name: str,
    sig: np.ndarray,
    raw_mean: float,
    raw_std: float,
    raw_min: float,
    raw_max: float,
    physical_min: float,
    physical_max: float,
    unit: str,
) -> None:
    """Log warnings for problematic signal statistics.

    Only warns about serious issues that likely indicate data problems:
    - NaN values (data corruption)
    - Constant/dead channels (sensor issue)
    - Zero physical range (cannot normalize)
    - Extremely large amplitudes after unit conversion (likely wrong unit)
    """
    basename = filepath.split('/')[-1]  # Just filename for cleaner logs

    # Check for NaN values (data corruption)
    nan_count = np.isnan(sig).sum()
    if nan_count > 0:
        nan_pct = 100 * nan_count / len(sig)
        _logger.warning(f'{basename}: {sig_name} has {nan_count} NaN values ({nan_pct:.1f}%)')

    # Check for constant signal (dead channel)
    if raw_std == 0 or np.isnan(raw_std):
        _logger.warning(f'{basename}: {sig_name} is constant (std=0) - possible dead channel')

    # Check for zero physical range (cannot normalize)
    physical_range = physical_max - physical_min
    if physical_range == 0:
        _logger.warning(
            f'{basename}: {sig_name} has zero physical range (min={physical_min}, max={physical_max}) - cannot normalize'
        )

    # Check for extremely large values after unit conversion (likely wrong unit in header)
    if sig_name in VOLTAGE_SIGNALS:
        unit_scale = get_unit_scaling(sig_name, unit)
        scaled_max = max(abs(raw_min), abs(raw_max)) * unit_scale
        # ECG/EOG > 200 mV is almost certainly wrong (normal is < 5 mV)
        if scaled_max > 200:
            _logger.warning(
                f'{basename}: {sig_name} has extreme amplitude ({scaled_max:.1f} mV after scaling) '
                f"- likely incorrect unit '{unit}' in header"
            )


def load_edf_data(
    filepath: str,
    columns: list[str],
    convert_time: bool = False,
    convert_units: bool = True,
    normalize_arbitrary: bool = True,
    raise_on_missing: bool = True,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Load selected columns of EDF data into a Pandas DataFrame.

    timestamp | col 1 | col 2 | (label)

    Args:
        filepath (str): EDF filepath
        columns (str|list): Name of column or list of column names e.g. ['EEG', 'EKG']
        convert_time (bool): If True, convert the index to a datetime index.
        convert_units (bool): If True, convert voltage signal units to mV.
        normalize_arbitrary (bool): If True, normalize arbitrary-unit signals (ABD, THX, PPG)
            to [-1, 1] range using their physical_min/max from the EDF header.
        raise_on_missing (bool): If True, raise an error if a requested column is not found.

    Returns:
        df: DataFrame with signal data
        metadata: Dict of {signal_name: {mean, std, min, max, physical_min, physical_max, unit}}
            for use in downstream normalization or real-time inference.
    """
    if isinstance(columns, str):
        columns = [columns]

    metadata = {}

    with pyedflib.EdfReader(filepath) as f:
        signal_map = {}
        units_map = {}
        for ind, channel_dict in enumerate(f.getSignalHeaders()):  # Map channel names to numbers
            label = channel_dict['label']
            signal_map[label] = ind
            units_map[label] = f.getPhysicalDimension(ind)
        dfs = []
        for sig_name in columns:
            actual_col_name = get_column_match(
                sig_name, signal_map.keys(), units_map=units_map, raise_error=raise_on_missing
            )
            if actual_col_name is None:
                continue
            idx = signal_map[actual_col_name]
            sig = f.readSignal(idx)
            sampling_freq = f.getSampleFrequency(idx)
            unit = f.getPhysicalDimension(idx)
            physical_min = f.getPhysicalMinimum(idx)
            physical_max = f.getPhysicalMaximum(idx)

            # Store raw signal stats before any normalization
            raw_mean = float(np.nanmean(sig))
            raw_std = float(np.nanstd(sig))
            raw_min = float(np.nanmin(sig))
            raw_max = float(np.nanmax(sig))

            # Check for problematic signal statistics
            _warn_signal_issues(
                filepath=filepath,
                sig_name=sig_name,
                sig=sig,
                raw_mean=raw_mean,
                raw_std=raw_std,
                raw_min=raw_min,
                raw_max=raw_max,
                physical_min=physical_min,
                physical_max=physical_max,
                unit=unit,
            )

            # Apply appropriate scaling based on signal type
            if sig_name in VOLTAGE_SIGNALS:
                # Voltage signals: convert to mV
                scale = get_unit_scaling(sig_name, unit) if convert_units else 1.0
                sig = sig * scale
                norm_method = 'voltage_to_mV'
                norm_scale = scale
                norm_offset = 0.0
            elif sig_name in ARBITRARY_UNIT_SIGNALS and normalize_arbitrary:
                # Arbitrary-unit signals: normalize to [-1, 1] using physical range
                # Use abs() to handle inverted ranges (min > max)
                physical_range = abs(physical_max - physical_min)
                if physical_range > 0:
                    # Normalize to [-1, 1]
                    physical_center = (physical_max + physical_min) / 2
                    sig = (sig - physical_center) / (physical_range / 2)
                    norm_method = 'physical_range'
                    norm_scale = 2.0 / physical_range
                    norm_offset = -physical_center * norm_scale
                else:
                    # Fallback if physical range is zero (truly invalid)
                    norm_method = 'none'
                    norm_scale = 1.0
                    norm_offset = 0.0
            else:
                norm_method = 'none'
                norm_scale = 1.0
                norm_offset = 0.0

            # Store metadata for this signal
            metadata[sig_name] = {
                'unit': unit,
                'physical_min': physical_min,
                'physical_max': physical_max,
                'physical_range_inverted': physical_max < physical_min,
                'raw_mean': raw_mean,
                'raw_std': raw_std,
                'raw_min': raw_min,
                'raw_max': raw_max,
                'norm_method': norm_method,
                'norm_scale': norm_scale,
                'norm_offset': norm_offset,
                'sampling_freq': sampling_freq,
            }

            t = pd.Index(np.arange(0, len(sig)) / sampling_freq, name=TIMESTAMP)
            dfs.append(pd.DataFrame({sig_name: sig}, index=t))

        if not bool(dfs):
            print(f'No signals found in {filepath} for {columns}')
        df = pd.concat(dfs, axis=1).sort_index()
        if convert_time:
            start = f.getStartdatetime()
            df.index = start + pd.to_timedelta(df.index, unit='s')

    return df, metadata


def get_edf_start(filepath: str) -> datetime.datetime:
    with pyedflib.EdfReader(filepath) as f:
        return f.getStartdatetime()


def get_edf_end(filepath: str) -> datetime.datetime:
    with pyedflib.EdfReader(filepath) as f:
        return f.getStartdatetime() + datetime.timedelta(seconds=f.getFileDuration())


def get_edf_signals(filepath: str, convert: bool = True, columns: list[str] | None = None) -> dict[str, float]:
    """Get dict of signal names to sampling rates and units from an EDF."""
    with pyedflib.EdfReader(filepath) as f:
        channel_map = {
            channel_dict['label']: {
                'sampling_rate': f.getSampleFrequency(ind),
                'unit': f.getPhysicalDimension(ind),
                'physical_min': f.getPhysicalMinimum(ind),
                'physical_max': f.getPhysicalMaximum(ind),
                'digital_min': f.getDigitalMinimum(ind),
                'digital_max': f.getDigitalMaximum(ind),
            }
            for ind, channel_dict in enumerate(f.getSignalHeaders())
        }
    if convert:  # Try to convert to common names e.g. EKG -> ECG
        channel_map = {INV_ALT_COLUMNS.get(k, k): v for k, v in channel_map.items()}
    if columns is not None:
        channel_map = {k: v for k, v in channel_map.items() if k in columns}
    return channel_map
