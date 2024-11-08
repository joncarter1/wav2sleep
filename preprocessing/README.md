# Dataset Pre-processing
Our work uses datasets managed by the National Sleep Research Resource (NSRR, https://sleepdata.org/). To reproduce our results, you will need to apply for access to the following datasets:
- SHHS
- MESA
- WSC
- CCSHS
- CFS
- CHAT
- MROS

Once approved, they can be downloaded with the NSRR gem using your access token:
```bash
nsrr download shhs # Downloads into your current working directory.
```

For the CHAT dataset, you will then need to run:
```bash
python 0_fix_edfs.py /path/to/chat/polysomnography/edfs
```
to fix some of the EDF files for parsing with `pyedflib`.

### Ingestion
Once downloaded, we provide scripts to extract only the required signals and sleep stage annotations for all datasets, saving them as high-performance, columnar parquet e.g.:
```bash
python 1_ingest.py --folder /path/to/shhs --output-folder /path/to/processed/datasets --overwrite
```
will save a processed version of SHHS at `/path/to/processed/datasets/shhs/ingest`. This can be parallelised over multiple cores with Ray using the `--max-parallel` flag.

This should be repeated for all datasets, after which you should end up with a folder structure like:
```
/path/to/processed/datasets
  /shhs
    /ingest
  /mesa
    /ingest
  ...
```
All ingested datasets should be stored under the same root (or symlinked appropriately) for subsequent scripts to work.

### Split the datasets
Next, the datasets need to be split into `train`, `val` and `test` subfolders. This can be done for each dataset with:
```bash
python 2a_create_splits_paper.py --folder /path/to/processed/datasets/shhs
...
python 2a_create_splits_paper.py --folder /path/to/processed/datasets/mros
python 2b_create_census_split.py --folder /path/to/processed/datasets
```
This will create `train`, `val` and `test` folders, and copy the relevant files over from the `ingest` folder(s).

### Environment variables
Create a copy of the `.env.example` file (`cp .env.example .env`) and set the variables within the file. For example, `WAV2SLEEP_DATA` should be set to the folder (`/path/to/processed/datasets`) containing the processed NSRR datasets.
