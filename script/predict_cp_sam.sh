#!/bin/bash
#   predict_cp_sam.sh    —  Predict tumor mask using SAM.

set -euo pipefail

# --- Clean environment to avoid ~/.local issues ---
module purge
module load anaconda3/2023.09

source $(conda info --base)/etc/profile.d/conda.sh

# --- Activate conda env ---
conda activate /projects/gbm_modeling/.conda/envs/cp


LOG_DIR="logs"
LOG_LEVEL="DEBUG"
mkdir -p "$LOG_DIR"

BASE_DATA_DIR="output/data"
PYTHON="/projects/gbm_modeling/.conda/envs/cp/bin/python"
MAIN="src/predict_cp_sam.py"

OUTPUT_DIR="output/metrics"
mkdir -p "$OUTPUT_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/predict_cp_sam_${TIMESTAMP}.log"

set +e
$PYTHON "$MAIN" \
    --checkpoint_dir /path/to/your/run_dir \
    --test_dir /path/to/test_dir \
    --predict_dir /path/to/preds_cp \
    --prefer "t1*ce*,T1*CE*"
    --log_level "$LOG_LEVEL" \
    --log_file "$LOG_FILE"
exit_code=$?
set -e


if [[ ${exit_code} -eq 0 ]]; then
    echo "OK: Preprocessing finished at $(date)" | tee -a "${LOG_FILE}"
else
    echo "ERROR: Preprocessing failed with exit code ${exit_code} at $(date)" | tee -a "${LOG_FILE}"
    # Uncomment to stop on first failure:
    exit ${exit_code}
fi    