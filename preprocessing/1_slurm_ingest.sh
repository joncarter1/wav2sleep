#!/bin/bash
echo "Submitting preprocessor script for dataset $1"

LOGFILE="${HOME}/slurm/preprocess/${1}_%j.out"

if [ -z "$1" ]; then
    echo "Error: Dataset name not provided.."
    exit 1
fi
if [ -z "$2" ]; then
    echo "Error: Signals not provided. Use 'neural' or 'cardiorespiratory'.."
    exit 1
fi
DATASET="$1"
SIGNALS="$2"

if [ "$SIGNALS" == "neural" ]; then
    columns="EOG-L EOG-R"
    subfolder="eog-raw"
elif [ "$SIGNALS" == "cardiorespiratory" ]; then
    columns="ECG PPG ABD THX"
    subfolder="cardiorespiratory"
else
    echo "Error: Invalid signals provided.."
    exit 1
fi

sbatch <<EOT
#!/bin/bash
#SBATCH --job-name="${DATASET}_preprocess"
#SBATCH --time=12:00:00
#SBATCH -o ${LOGFILE} --cpus-per-task 34 --mem 128GB --partition=short --nodes 1 --clusters=all --constraint="os:redhat9"

source .venv/bin/activate
python preprocessing/1_ingest.py \
    --folder $DATA/nsrr/${DATASET} \
    --output-folder $DATA/datasets/${subfolder} \
    --max-parallel 32 \
    --columns ${columns}

EOT