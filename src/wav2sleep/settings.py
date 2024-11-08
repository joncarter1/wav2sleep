# Output column names
PPG = 'PPG'
ECG = 'ECG'
ABD = 'ABD'
THX = 'THX'
LABEL = 'Stage'
TIMESTAMP = 'Timestamp'
SLEEP = 'Sleep'

# Mapping of signals to expected shape
HIGH_FREQ_LEN = 1_228_800
LOW_FREQ_LEN = 307_200
COL_MAP = {ECG: HIGH_FREQ_LEN, PPG: HIGH_FREQ_LEN, ABD: LOW_FREQ_LEN, THX: LOW_FREQ_LEN}

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

# Mappings from five class sleep stages to integers.
INTEGER_LABEL_MAPS = {
    4: {0: 0, 1: 1, 2: 1, 3: 2, 4: 3},
}
