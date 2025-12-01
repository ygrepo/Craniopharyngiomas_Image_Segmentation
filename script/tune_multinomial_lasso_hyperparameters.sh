#!/bin/bash
#   tune_multinomial_lasso_hyperparameters.sh    —  Tune Multinomial Lasso hyperparameters.
#SBATCH --job-name=tune_multinomial_lasso_hyperparameters
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/tune_multinomial_lasso_hyperparameters_%j.out
#SBATCH --error=logs/tune_multinomial_lasso_hyperparameters_%j.err

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
MAIN="src/tune_multinomial_lasso_hyperparams.py"


MODEL_TYPE="postop"
PENALTY="l1"
C_GRID="0.001,0.01,0.1,1.0,3.0,10.0,30.0,100.0,150.0,200.0,250.0,300.0"
#K="40"
L1_RATIO=0.3
L1_RATIO_GRID="0.1,0.3,0.5,0.7,0.9"

DATA_DIR="data/CP"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/${MODEL_TYPE}_${PENALTY}_l1ratio_${L1_RATIO}_tune_multinomial_lasso_hyperparams_${TIMESTAMP}.log"

    
echo "Data dir: ${DATA_DIR}"
echo "Model type: ${MODEL_TYPE}"
echo "Penalty: ${PENALTY}"
echo "L1 ratio: ${L1_RATIO}"
#echo "K: ${K}"
echo "C grid: ${C_GRID}"
echo "L1 ratio grid: ${L1_RATIO_GRID}"

echo "Log file: ${LOG_FILE}"
echo "Log level: ${LOG_LEVEL}"


set +e
$PYTHON "$MAIN" \
    --log_level "$LOG_LEVEL" \
    --log_file "$LOG_FILE" \
    --data_dir "$DATA_DIR" \
    --model_type "$MODEL_TYPE" \
    --penalty "$PENALTY" \
    --l1_ratio "$L1_RATIO" \
    --C_grid "$C_GRID" \
    --l1_ratio_grid "$L1_RATIO_GRID"
exit_code=$?
set -e

if [[ ${exit_code} -eq 0 ]]; then
    echo "OK: script finished at $(date)" | tee -a "${LOG_FILE}"
else
    echo "ERROR: script failed with exit code ${exit_code} at $(date)" | tee -a "${LOG_FILE}"
    exit ${exit_code}
fi