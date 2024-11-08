"""Parse Sleep stages from XML files."""

import re

import numpy as np
import pandas as pd

from ..settings import LABEL, TIMESTAMP
from .utils import convert_int_stage


def parse_xml_annotations(filepath) -> pd.Series:
    """Parse an annotations XML file to retrieve a series of sleep stages indexed in seconds.

    Inspired by:
    https://github.com/drasros/sleep_staging_shhs/blob/master/shhs.py
    """
    with open(filepath) as f:
        content = f.read()
    patterns_start = re.findall(r'<EventConcept>Recording Start Time</EventConcept>\n<Start>0</Start>', content)
    if len(patterns_start) == 0:
        raise ValueError(f'{filepath=} had no start time.')
    elif len(patterns_start) > 1:
        raise ValueError(f'{filepath=} had multiple start times.')
    # Find annotations within the XML via regex matching.
    stage_patterns = find_stages(content)
    return create_sleep_series(stage_patterns)


def find_stages(file_contents: str) -> list[str]:
    """Find sleep stages within an XML using regex matching."""
    return re.findall(
        r'<EventType>Stages.Stages</EventType>\n'
        + r'<EventConcept>.+</EventConcept>\n'
        + r'<Start>.+</Start>\n'
        + r'<Duration>.+</Duration>\n'
        + r'</ScoredEvent>',
        file_contents,
    )


def create_sleep_series(stage_patterns: list[str]) -> pd.Series:
    """Create pandas series of sleep stages from a list of stage patterns."""
    stages = []

    for ind, pattern in enumerate(stage_patterns):
        _, sleep_stage_str, start_str, duration_str, *_ = pattern.splitlines()
        stage = convert_int_stage(sleep_stage_str[-16])
        start = float(start_str[7:-8])
        if ind == 0 and start != 0.0:
            raise ValueError(f'First stage did not start at 0.0s: {start}')
        duration = float(duration_str[10:-11])
        if duration % 30 != 0.0:
            raise ValueError(f'Non-30s epoch duration: {duration}')
        num_epochs = int(duration) // 30
        stages += [stage] * num_epochs
    ts = np.arange(0, 30 * len(stages), 30.0)  # Timestamps in seconds from start
    # Make labels correspond to previous 30s rather than next 30s
    ts += 30
    return pd.DataFrame({LABEL: stages, TIMESTAMP: ts}).set_index(TIMESTAMP).squeeze().sort_index()
