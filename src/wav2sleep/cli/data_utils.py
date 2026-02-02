"""CLI compatibility layer that re-exports the public API.

This file exists to maintain backward compatibility for any internal or external
imports that referenced `wav2sleep.cli.data_utils`. All functionality now lives
in `wav2sleep.api`.
"""

from wav2sleep.api import load_dataset, save_predictions
from wav2sleep.api import prepare as prepare_dataset

__all__ = ['prepare_dataset', 'load_dataset', 'save_predictions']
