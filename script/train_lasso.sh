#!/bin/bash
#   train_lasso.sh    —  Train Lasso model.
#SBATCH --job-name=train_lasso
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/train_lasso_%j.out
#SBATCH --error=logs/train_lasso_%j.err

set -euo pipefail

module purge
module load anaconda3/2023.09
module load proxy/jh-proxy-1.0
source $(conda info --base)/etc/profile.d/conda.sh

# --- Paths (edit if needed) ---
ENV_PREFIX="/projects/gbm_modeling/.conda/envs/mri"
PIP_CACHE_DIR="/projects/gbm_modeling/.pip_cache"
CONDA_PKGS_DIRS="/projects/gbm_modeling/.conda/pkgs"

# --- Keep installs off $HOME and avoid user-site leakage ---
mkdir -p "${PIP_CACHE_DIR}" "${CONDA_PKGS_DIRS}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS}"
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true

conda activate "${ENV_PREFIX}"

# Set nnUNet paths
source script/set_unet_path.sh

LOG_DIR="logs"
LOG_LEVEL="DEBUG"
mkdir -p "$LOG_DIR"

PYTHON="${ENV_PREFIX}/bin/python"
MAIN="src/train_lasso.py"


MODEL_TYPE="preop"
DATA_DIR="data/CP"
C_BINARY=0.1
C_MULTINOMIAL=.01
# C_BINARY=1
# C_MULTINOMIAL=1
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/${MODEL_TYPE}_train_lasso_${TIMESTAMP}.log"

    
echo "Data dir: ${DATA_DIR}"
echo "Model type: ${MODEL_TYPE}"
echo "Log file: ${LOG_FILE}"
echo "Log level: ${LOG_LEVEL}"


set +e
$PYTHON "$MAIN" \
    --log_level "$LOG_LEVEL" \
    --log_file "$LOG_FILE" \
    --data_dir "$DATA_DIR" \
    --model_type "$MODEL_TYPE" \
    --C_binary "$C_BINARY" \
    --C_multinomial "$C_MULTINOMIAL"
exit_code=$?
set -e

if [[ ${exit_code} -eq 0 ]]; then
    echo "OK: script finished at $(date)" | tee -a "${LOG_FILE}"
else
    echo "ERROR: script failed with exit code ${exit_code} at $(date)" | tee -a "${LOG_FILE}"
    exit ${exit_code}
fi