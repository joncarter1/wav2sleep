"""CLI compatibility layer that re-exports the public API.

This file exists to maintain backward compatibility for any internal or external
imports that referenced `wav2sleep.cli.model_utils`. All functionality now lives
in `wav2sleep.api`.
"""

from wav2sleep.api import load_model
from wav2sleep.api import predict as apply_model

__all__ = ['load_model', 'apply_model']
