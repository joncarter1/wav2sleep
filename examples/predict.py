# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "wav2sleep @ git+https://github.com/joncarter1/wav2sleep.git",
# ]
# ///
"""Run wav2sleep inference on a folder of EDF files.

Usage:
    uv run predict.py --input-folder /path/to/edfs --output-folder /path/to/outputs
    uv run predict.py --input-folder /path/to/edfs --output-folder /path/to/outputs --model hf://joncarter/wav2sleep
"""

import argparse

from wav2sleep import load_model, predict_on_folder

parser = argparse.ArgumentParser(description='Run wav2sleep inference on EDF files.')
parser.add_argument('--input-folder', required=True, help='Folder containing EDF files.')
parser.add_argument('--output-folder', required=True, help='Folder to save predictions.')
parser.add_argument('--model', default='hf://joncarter/wav2sleep-eog', help='HF Hub URI or local model folder.')
parser.add_argument('--device', default='auto', help='Device to use (auto, cuda, cpu).')
args = parser.parse_args()

model = load_model(args.model, device=args.device)
predict_on_folder(args.input_folder, args.output_folder, model=model, device=args.device)
