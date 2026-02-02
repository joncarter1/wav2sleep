"""Hugging Face Hub integration for wav2sleep models."""

from __future__ import annotations

from typing import Optional

from huggingface_hub import HfApi, snapshot_download

# Model variant configurations for generating model cards
MODEL_VARIANTS = {
    'wav2sleep': {
        'signals': ['ECG', 'PPG', 'ABD', 'THX'],
        'num_classes': 4,
        'causal': False,
        'description': 'Cardio-respiratory sleep staging (4-class: Wake, Light, Deep, REM)',
    },
    'wav2sleep-eog': {
        'signals': ['EOG-L', 'EOG-R'],
        'num_classes': 5,
        'causal': False,
        'description': 'EOG-based sleep staging (5-class: Wake, N1, N2, N3, REM)',
    },
}


def is_hf_repo_id(path_or_repo: str) -> bool:
    """Check if string is a Hugging Face Hub URI.

    Args:
        path_or_repo: Either a local path or a HF Hub URI (hf://username/repo-name).

    Returns:
        True if the string starts with 'hf://', False otherwise.
    """
    return path_or_repo.startswith('hf://')


def download_from_hub(
    repo_id: str,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> str:
    """Download a wav2sleep model from Hugging Face Hub.

    Downloads only the necessary files (config.yaml, state_dict.pth, README.md).

    Args:
        repo_id: The HF Hub repo ID (e.g., "joncarter/wav2sleep" or "hf://joncarter/wav2sleep").
        revision: Git revision (branch, tag, or commit hash). Defaults to main.
        cache_dir: Local directory to cache downloads. Defaults to HF cache.

    Returns:
        Local path to the downloaded model folder.
    """
    # Strip hf:// prefix if present
    if repo_id.startswith('hf://'):
        repo_id = repo_id[5:]

    return snapshot_download(
        repo_id=repo_id,
        revision=revision,
        cache_dir=cache_dir,
        allow_patterns=['config.yaml', 'state_dict.pth', 'README.md'],
    )


def upload_to_hub(
    local_folder: str,
    repo_id: str,
    variant_name: Optional[str] = None,
    private: bool = False,
    token: Optional[str] = None,
) -> str:
    """Upload a wav2sleep checkpoint to Hugging Face Hub.

    Uploads config.yaml and state_dict.pth, with an auto-generated model card
    if a variant name is provided.

    Args:
        local_folder: Path to local checkpoint folder containing config.yaml and state_dict.pth.
        repo_id: Target HF Hub repo ID (e.g., "joncarter/wav2sleep").
        variant_name: Optional model variant name (e.g., "wav2sleep-eog") for model card generation.
        private: If True, create a private repository.
        token: HF API token. If None, uses cached token from `huggingface-cli login`.

    Returns:
        URL of the uploaded model on HF Hub.
    """
    api = HfApi(token=token)

    # Create repo if it doesn't exist
    api.create_repo(repo_id=repo_id, exist_ok=True, private=private)

    # Upload model files
    api.upload_folder(
        folder_path=local_folder,
        repo_id=repo_id,
        allow_patterns=['config.yaml', 'state_dict.pth'],
    )

    # Generate and upload model card if variant specified
    if variant_name:
        model_card = generate_model_card(variant_name)
        api.upload_file(
            path_or_fileobj=model_card.encode('utf-8'),
            path_in_repo='README.md',
            repo_id=repo_id,
        )

    return f'https://huggingface.co/{repo_id}'


def generate_model_card(variant_name: str) -> str:
    """Generate a Hugging Face model card with YAML frontmatter.

    Args:
        variant_name: Model variant name from MODEL_VARIANTS.

    Returns:
        Markdown string for README.md with HF-compatible frontmatter.

    Raises:
        ValueError: If variant_name is not in MODEL_VARIANTS.
    """
    if variant_name not in MODEL_VARIANTS:
        raise ValueError(f"Unknown variant '{variant_name}'. Valid variants: {list(MODEL_VARIANTS.keys())}")

    variant = MODEL_VARIANTS[variant_name]
    signals = variant['signals']
    num_classes = variant['num_classes']
    causal = variant['causal']
    description = variant['description']

    # Determine signal type for tags
    if 'EOG-L' in signals:
        signal_desc = 'electrooculography (EOG)'
    else:
        signal_desc = 'cardio-respiratory signals (ECG, PPG, respiratory)'

    causal_desc = 'Causal (real-time capable)' if causal else 'Non-causal (bidirectional)'

    return f"""---
license: mit
tags:
  - sleep-staging
  - wav2sleep
  - polysomnography
  - time-series
  - pytorch
library_name: wav2sleep
pipeline_tag: other
---

# {variant_name}

{description}

## Model Description

This is a **wav2sleep** model for automatic sleep stage classification from {signal_desc}.
wav2sleep is a unified multi-modal deep learning approach that can process various combinations
of physiological signals for sleep staging.

- **Paper**: [wav2sleep: A Unified Multi-Modal Approach to Sleep Stage Classification](https://arxiv.org/abs/2411.04644)
- **Repository**: [GitHub](https://github.com/joncarter1/wav2sleep)
- **Conference**: ML4H 2024

## Model Details

| Property | Value |
|----------|-------|
| **Input Signals** | {', '.join(signals)} |
| **Output Classes** | {num_classes} |
| **Architecture** | {causal_desc} |

### Signal Specifications

| Signal | Samples per 30s epoch |
|--------|----------------------|
| ECG, PPG | 1,024 |
| ABD, THX | 256 |
| EOG-L, EOG-R | 4,096 |

## Usage

```python
from wav2sleep import load_model

# Load model from Hugging Face Hub
model = load_model("hf://joncarter/{variant_name}")

# Or load from local checkpoint
model = load_model("/path/to/checkpoint")
```

For inference on new data:

```python
from wav2sleep import load_model, predict_on_folder

model = load_model("hf://joncarter/{variant_name}")
predict_on_folder(
    input_folder="/path/to/edf_files",
    output_folder="/path/to/predictions",
    model=model,
)
```

## Training Data

The model was trained on polysomnography data from multiple publicly available datasets
managed by the National Sleep Research Resource (NSRR).

## Citation

```bibtex
@misc{{carter2024wav2sleep,
    title={{wav2sleep: A Unified Multi-Modal Approach to Sleep Stage Classification from Physiological Signals}},
    author={{Jonathan F. Carter and Lionel Tarassenko}},
    year={{2024}},
    eprint={{2411.04644}},
    archivePrefix={{arXiv}},
    primaryClass={{cs.LG}},
}}
```

## License

MIT
"""
