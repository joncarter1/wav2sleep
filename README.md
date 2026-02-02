<div align="center">
  <h2><b>wav2sleep 💤: A Unified Multi-Modal Approach to Sleep Stage Classification from Physiological Signals
  </b></h2>
</div>

![image](figures/wav2sleep.jpg)

## Overview

This repository contains the official implementation of **wav2sleep**💤, which has been accepted to **[Machine Learning for Health (ML4H) 2024](https://ahli.cc/ml4h/)**.

wav2sleep is a unified model for sleep staging from sets of physiological signals, including cardio-respiratory (ECG, PPG, respiratory) and neural (EOG) modalities. It can be jointly trained across heterogeneous datasets, where the availability of input signals can vary. At test-time, the model can be applied to *any* subset of the modalities used during training.

After jointly training on over 10,000 overnight recordings from publicly available polysomnography datasets, including SHHS and MESA, wav2sleep outperforms existing sleep stage classification models across a range of input signal combinations.

To find out more, check out our paper: https://arxiv.org/abs/2411.04644

<a href="https://arxiv.org/abs/2411.04644"><img src="https://img.shields.io/badge/Arxiv-2411.04644-B31B1B.svg"></a>

## Updates

**February 2026**
- **EOG Model Release**: We've released `wav2sleep-eog`, a 5-class sleep staging model using EOG signals (Wake, N1, N2, N3, REM).
- **Python API**: New `load_model()` and `predict_on_folder()` functions make it easy to run inference on your own data:

```python
from wav2sleep import load_model, predict_on_folder

model = load_model("hf://joncarter/wav2sleep-eog")
predict_on_folder("/path/to/edfs", "/path/to/outputs", model=model)
```

## Features 🔥
### Models
Implementation of `Wav2Sleep` and baseline models in PyTorch, using [Lightning](https://lightning.ai/) for training.

![image](figures/wav2sleep_arch.png)
*Figure: wav2sleep architecture for sets of signals.*

### High-performance data processing pipelines
Scripts for transforming EDF files and sleep stage annotations from all 7 datasets used into efficient, columnar parquet files for model training. This can be parallelised over multiple CPU cores or an entire cluster using [Ray](https://www.anyscale.com/ray-open-source), e.g.:
```bash
uv run preprocessing/1_ingest.py --folder /path/to/shhs --output-folder /path/to/processed/datasets --max-parallel 16
```

## Table of Contents
1. [Set-up and Installation](#set-up-and-installation)
2. [Training and Evaluation](#training-and-evaluation)
3. [Inference](#inference)
4. [Hugging Face Hub](#hugging-face-hub)
5. [Visualising Results](#visualising-results)
6. [Citation](#citation)
7. [License](#license)

## Set-up and Installation

Install directly from GitHub:
```bash
pip install git+https://github.com/joncarter1/wav2sleep.git
```

Or clone and install for development:
```bash
git clone https://github.com/joncarter1/wav2sleep
cd wav2sleep
uv sync
```

### b. Datasets
Our work uses datasets managed by the National Sleep Research Resource ([NSRR](https://sleepdata.org/)). To reproduce our results, you will need to apply for access to the following datasets:
- SHHS
- MESA
- WSC
- CCSHS
- CFS
- CHAT
- MROS

Once approved, these can be downloaded with the NSRR Ruby gem, e.g. `nsrr download shhs`. More details can be found on the NSRR website.

Once downloaded, we provide high-performance processing scripts to process each dataset and split it into training, validation and test partitions. Instructions on how to do this can be found [here](preprocessing/README.md).

## Training and Evaluation
To train and evaluate the model on all datasets, just run:
```bash
uv run scripts/train.py model=wav2sleep num_gpus=1 tune_batch_size=True target_batch_size=16 name=wav2sleep-repro inputs=all datasets=all test=True
```
This will find the largest batch size that fits on your GPU, and accumulate batches for an effective batch size of at least 16. If you're lucky enough to have more than one GPU, you can specify e.g. `num_gpus=2` to run across them using distributed data parallel (DDP) training.


## Inference
The `predict.py` script can be used to run either cardio-respiratory or EOG models.

## Example inference commands
Note: You may need to reduce batch size depending on GPU size.

Run the EOG model on already-processed parquet files using the `no-preprocess` flag:
```bash
uv run scripts/predict.py --input-folder /path/to/processed/mesa/test  \
--output-folder /tmp/example-mesa-outputs \
--model-folder checkpoints/wav2sleep-eog \
--batch-size 16 --no-preprocess
```

Run the EOG model on raw EDF files, with analysis of up to 14 hours per file:
```bash
uv run scripts/predict.py --input-folder /path/to/edf/folder \
--output-folder /tmp/example-outputs \
--model-folder checkpoints/wav2sleep-eog \
--batch-size 16 --max-length-hours 14
```
(This will skip broken EDF files)

## Hugging Face Hub

### Available Models

| Model | Signals | Classes | Description |
|-------|---------|---------|-------------|
| `hf://joncarter/wav2sleep` | ECG, PPG, ABD, THX | 4 | Cardio-respiratory (Wake, Light, Deep, REM) |
| `hf://joncarter/wav2sleep-eog` | EOG-L, EOG-R | 5 | EOG-based (Wake, N1, N2, N3, REM) |

Models can be loaded directly from the Hugging Face Hub:

```python
from wav2sleep import load_model, predict_on_folder

# Load from HF Hub
model = load_model("hf://joncarter/wav2sleep-eog")

# Run inference
predict_on_folder(
    input_folder="/path/to/edf_files",
    output_folder="/path/to/predictions",
    model=model,
)
```

To upload checkpoints to the Hub:

```bash
uv run scripts/upload_to_hub.py \
    --local-folder checkpoints/wav2sleep-eog \
    --repo-id your-username/wav2sleep-eog \
    --variant wav2sleep-eog
```

## Visualising Results

We use [MLFlow](https://mlflow.org) to log trained models and evaluation metrics. By default, these will be stored in a local directory (`./mlruns`) and can be visualized results by running:
```bash
uv run mlflow server
```
and visiting http://localhost:5000 in your browser:

![image](figures/mlflow.png)
*Figure: Screenshot from an MLFlow dashboard*

## Citation

If you find this code useful for your research, please cite our paper:

```bibtex
@misc{carter2024wav2sleepunifiedmultimodalapproach,
      title={wav2sleep: A Unified Multi-Modal Approach to Sleep Stage Classification from Physiological Signals},
      author={Jonathan F. Carter and Lionel Tarassenko},
      year={2024},
      eprint={2411.04644},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2411.04644},
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This software is provided for research and development purposes only. It is not a medical device, and is not intended for use in clinical decision-making.

## Credits
This project was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [joncarter1/cookiecutter_research](https://github.com/joncarter1/cookiecutter_research) template.
