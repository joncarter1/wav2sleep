"""Generate predictions using a trained model."""

import argparse
import logging
import os

import hydra
import numpy as np
import pandas as pd
import torch
import yaml
from dotenv import load_dotenv
from omegaconf import OmegaConf
from tqdm import tqdm
from yaml import CLoader as Loader

from wav2sleep.data.dataset import ParquetDataset
from wav2sleep.data.utils import get_parquet_fps
from wav2sleep.settings import ECG, PPG, THX, TIMESTAMP

logger = logging.getLogger(__name__)

SUBSETS = [(ECG,), (PPG,), (ECG, THX), 'all']

CHECKPOINT_FOLDER = 'checkpoints'
MODEL_CONFIG_FP = 'config.yaml'
STATE_DICT_FP = 'state_dict.pth'
PRED_COL = 'Pred'


def load_model(folder: str | None = None, device: str = 'cuda'):
    """Load the model from the specified folder."""
    # Use repository model by default.
    if folder is None:
        folder = os.path.join(os.path.dirname(__file__), CHECKPOINT_FOLDER, 'wav2sleep')
    # Load YAML configuration
    with open(os.path.join(folder, MODEL_CONFIG_FP), 'r') as f:
        model_cfg = yaml.load(f, Loader)
        model_cfg = OmegaConf.create(model_cfg)
    model = hydra.utils.instantiate(model_cfg)
    ckpt_path = os.path.join(folder, STATE_DICT_FP)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'No state dict found at {ckpt_path}. Has the model been downloaded?')
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    model.eval()
    return model.to(device)


def load_dataset(parquet_folder, signals):
    parquet_fps = get_parquet_fps(parquet_folder)
    return ParquetDataset(parquet_fps=parquet_fps, columns=signals)


@torch.no_grad
def apply_model(model, dataset, device: str, batch_size: int = 32):
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=False
    )
    predictions = []
    for batch in tqdm(dataloader):
        x, _ = batch
        x = {k: v.to(device) for k, v in x.items()}
        logits_BSC = model(x)
        predictions.append(logits_BSC.argmax(dim=-1))
    predictions = torch.cat(predictions, dim=0).cpu().numpy()
    return predictions


def save_predictions(predictions, output_folder, dataset, overwrite: bool = False):
    """Save the predictions to CSVs in the output folder."""
    os.makedirs(output_folder, exist_ok=True)
    index = pd.Index(np.arange(0, 36_000, step=30) + 30.0, name=TIMESTAMP)  # Elapsed time in seconds
    for idx, fp in enumerate(dataset.files):
        out_fp = os.path.join(output_folder, os.path.basename(fp).replace('.parquet', '.preds.csv'))
        if os.path.exists(out_fp) and not overwrite:
            logger.warning(f'File {out_fp} exists. Skipping.')
            continue
        else:
            pd.Series(predictions[idx], index=index, name=PRED_COL).to_csv(out_fp)


def parse_args():
    parser = argparse.ArgumentParser(
        prog='Inference', description='Apply trained wav2sleep model to folders of parquet.'
    )
    parser.add_argument('--parquet-folder', required=True, help='Folder containing parquet files.')
    parser.add_argument('--output-folder', required=True, help='Base output folder for predictions.')
    parser.add_argument(
        '--model-folder',
        default=None,
        help='Folder containing model state dict and YAML config file. Defaults to the model stored in the repository.',
    )
    parser.add_argument(
        '--signals',
        default=None,
        help='Subset of signals to use e.g. ECG,THX. If unspecified, all signals will be used.',
    )
    parser.add_argument('--device', required=False, type=str, default='cuda', help='Device to use for inference.')
    parser.add_argument('--batch-size', type=int, default=32, help='Base output folder for predictions.')
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing predictions.',
        default=False,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print('Starting evaluation...')
    # Load the model
    model = load_model(folder=args.model_folder, device=args.device)
    if args.signals is not None:
        signals = [sig.strip() for sig in args.signals.split(',')]
    else:  # Use all signals from the parquet file.
        signals = model.valid_signals
    # Load the dataset
    dataset = load_dataset(parquet_folder=args.parquet_folder, signals=signals)
    print(f'Found {len(dataset)} files.')
    # Evaluate the model
    print('Applying model...')
    predictions = apply_model(model, dataset, device=args.device, batch_size=args.batch_size)
    save_predictions(predictions, args.output_folder, dataset)
    print('Eval complete.')
    return


if __name__ == '__main__':
    load_dotenv()
    main()
