#!/usr/bin/env python
"""Upload wav2sleep checkpoints to Hugging Face Hub.

Usage:
    uv run scripts/upload_to_hub.py \\
        --local-folder checkpoints/wav2sleep \\
        --repo-id joncarter/wav2sleep \\
        --variant wav2sleep \\
        --private
"""

from __future__ import annotations

import argparse
import sys

from wav2sleep.hub import MODEL_VARIANTS, upload_to_hub


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Upload wav2sleep checkpoints to Hugging Face Hub.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Upload cardio-respiratory model (private)
    uv run scripts/upload_to_hub.py \\
        --local-folder checkpoints/wav2sleep \\
        --repo-id joncarter/wav2sleep \\
        --variant wav2sleep \\
        --private

    # Upload EOG model (public)
    uv run scripts/upload_to_hub.py \\
        --local-folder checkpoints/wav2sleep-eog \\
        --repo-id joncarter/wav2sleep-eog \\
        --variant wav2sleep-eog
        """,
    )
    parser.add_argument(
        '--local-folder',
        required=True,
        help='Path to local checkpoint folder containing config.yaml and state_dict.pth',
    )
    parser.add_argument(
        '--repo-id',
        required=True,
        help='Target Hugging Face Hub repo ID (e.g., "joncarter/wav2sleep")',
    )
    parser.add_argument(
        '--variant',
        choices=list(MODEL_VARIANTS.keys()),
        help='Model variant name for auto-generating model card',
    )
    parser.add_argument(
        '--private',
        action='store_true',
        help='Create a private repository',
    )
    parser.add_argument(
        '--token',
        help='Hugging Face API token (defaults to cached token from huggingface-cli login)',
    )

    args = parser.parse_args()

    print(f'Uploading {args.local_folder} to {args.repo_id}...')

    url = upload_to_hub(
        local_folder=args.local_folder,
        repo_id=args.repo_id,
        variant_name=args.variant,
        private=args.private,
        token=args.token,
    )

    print(f'Uploaded to: {url}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
