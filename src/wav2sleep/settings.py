# Output column names
PPG = 'PPG'
ECG = 'ECG'
ABD = 'ABD'
THX = 'THX'
EOG_L = 'EOG-L'
EOG_R = 'EOG-R'
LABEL = 'Stage'
TIMESTAMP = 'Timestamp'
SLEEP = 'Sleep'
PRED = 'Pred'

TRAINING_LENGTH_HOURS = 10  # Recording length in hours during training.

# Mapping of signals to sampling rates. These are measured in samples per epoch (30-second window of data).
LOW_FREQ_SAMPLES_PER_EPOCH = 256
MEDIUM_FREQ_SAMPLES_PER_EPOCH = 1024
HIGH_FREQ_SAMPLES_PER_EPOCH = 4096
COLS_TO_SAMPLES_PER_EPOCH = {
    ABD: LOW_FREQ_SAMPLES_PER_EPOCH,
    THX: LOW_FREQ_SAMPLES_PER_EPOCH,
    ECG: MEDIUM_FREQ_SAMPLES_PER_EPOCH,
    PPG: MEDIUM_FREQ_SAMPLES_PER_EPOCH,
    EOG_L: HIGH_FREQ_SAMPLES_PER_EPOCH,
    EOG_R: HIGH_FREQ_SAMPLES_PER_EPOCH,
}

# Causal normalization parameters (used when causal=True in dataset loading)
CAUSAL_NORM_TAU_SECONDS = 900.0  # 15 minutes time constant for variance tracking
NORM_OUTLIER_THRESHOLD = 4.0  # Sigma threshold for outlier clipping in both causal and non-causal normalization
CAUSAL_NORM_BASELINE_TAU_SECONDS = 120.0  # 2 minutes time constant for baseline (mean) tracking
CAUSAL_NORM_MIN_SIGMA = 0.1  # Minimum sigma floor to prevent division by near-zero variance

# PSG datasets
SHHS = 'shhs'
MESA = 'mesa'
CFS = 'cfs'
CHAT = 'chat'
CCSHS = 'ccshs'
MROS = 'mros'
WSC = 'wsc'

# Folder for census-balanced dataset used by Jones et al. paper
CENSUS = 'census'

KNOWN_DATASETS = [SHHS, MESA, CFS, CHAT, CCSHS, MROS, WSC, CENSUS]

INGEST = 'ingest'  # Temporary folder for each dataset to store parquet before splitting into train/val/test.
TRAIN, VAL, TEST = 'train', 'val', 'test'


# Mappings from five class sleep stages to integer labels for different num_classes.
INTEGER_LABEL_MAPS = {
    4: {0: 0, 1: 1, 2: 1, 3: 2, 4: 3},  # 4-class
    5: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4},  # 5-class
}
