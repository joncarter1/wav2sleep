"""Generate predictions using a trained wav2sleep model.

The model can be applied to EDF, CSV and/or parquet files containing input signals.

The input CSV/parquet files are expected to be in the following format:
timestamp, signal_name (e.g. ECG, THX, etc.)
```input.csv
2025-01-01 00:00:00, 0.01
2025-01-01 00:00:01, 0.20
2025-01-01 00:00:02, 0.20
```
with timestamps in datetime format or measured in seconds from the start of the recording.

The output predictions will be saved in the following format:
```output.csv
timestamp, pred, label (if available)
2025-01-01 00:00:00, 0, 0
2025-01-01 00:00:00, 1, 1
2025-01-01 00:00:00, 2, 2
```

The output folder will have the same directory structure as the input folder.
"""

import argparse
import logging
import os

from torchmetrics.classification import MulticlassConfusionMatrix

from wav2sleep.api import predict_on_folder
from wav2sleep.stats import cohens_kappa, confusion_accuracy

logger = logging.getLogger(__name__)


logger.info = print


def parse_args():
    parser = argparse.ArgumentParser(
        prog='Inference', description='Apply trained wav2sleep model to folders of parquet.'
    )
    parser.add_argument(
        '--input-folder', required=True, help='Folder containing EDF/csv/parquet files.', type=os.path.abspath
    )
    parser.add_argument(
        '--output-folder',
        required=True,
        help='Base output folder for predictions. Directory structure will be copied from input folder.',
        type=os.path.abspath,
    )
    parser.add_argument(
        '--model-folder',
        default=None,
        help='Folder containing model state dict and YAML config file. Defaults to the model stored in the repository.',
        type=os.path.abspath,
    )
    parser.add_argument(
        '--signals',
        default=None,
        help='Subset of signals to use, e.g. ECG,THX. If unspecified, all supported signals will be used.',
    )
    parser.add_argument('--device', required=False, type=str, default='cuda', help='Device to use for inference.')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for inference.')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers for inference.')
    parser.add_argument(
        '--no-preprocess',
        action='store_true',
        help='Assume input data is already preprocessed.',
        default=False,
    )
    parser.add_argument(
        '--max-length-hours',
        type=int,
        default=10,
        help='Maximum length of the input data in hours.',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing predictions.',
        default=False,
    )
    parser.add_argument(
        '--compile',
        action='store_true',
        help='Compile the model.',
        default=False,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info('Starting inference...')
    # Parse signals (optional)
    signals = None
    if args.signals is not None:
        signals = [sig.strip() for sig in args.signals.split(',')]
        logger.info(f'Using any available signals from: {signals}')
    # Run end-to-end via public API and return tensors for metrics
    result = predict_on_folder(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        model_folder=args.model_folder,
        signals=signals,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        preprocess=not args.no_preprocess,
        max_length_hours=args.max_length_hours,
        overwrite=args.overwrite,
        compile=args.compile,
        return_tensors=True,
    )
    predictions, labels = result
    if labels is not None:
        logger.info('Evaluating model...')
        # We cannot directly access model.num_classes here; infer from preds/labels
        n_classes = int(predictions.max().item() + 1)
        cmat_func = MulticlassConfusionMatrix(num_classes=n_classes, ignore_index=-1)
        cmat = cmat_func(predictions, labels).numpy()
        kappa = cohens_kappa(cmat, n_classes=n_classes)
        acc = 100 * confusion_accuracy(cmat)
        logger.info(f'Kappa: {kappa:.3f}, Accuracy: {acc:.3f}')
    return
