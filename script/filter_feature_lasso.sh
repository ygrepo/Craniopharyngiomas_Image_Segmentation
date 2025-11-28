#!/bin/bash
#   filter_feature_lasso.sh    —  Filter features based on Lasso importance.
#SBATCH --job-name=filter_feature_lasso
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/filter_feature_lasso_%j.out
#SBATCH --error=logs/filter_feature_lasso_%j.err

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
MAIN="src/filter_feature_lasso.py"


MODEL_TYPE="preop"
#TASK="multinomial"
TASK="binary"
#C=0.1
C=0.0010
#PENALTY="elasticnet"
PENALTY="l2"
L1_RATIO=0.3
#NPZ_PATH="data/CP/preop_train_multinomial_scaled.npz"
NPZ_PATH="data/CP/preop_train_binary_scaled.npz"
IMPORTANCE_CSV="data/CP/preop_binary_l2_feature_importance.csv"
#IMPORTANCE_CSV="data/CP/preop_multinomial_C_0.1_elasticnet_l1ratio_0.3_feature_importance.csv"
K=60
DATA_DIR="data/CP"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/${MODEL_TYPE}_${PENALTY}_l1ratio_${L1_RATIO}_filter_feature_lasso_${TIMESTAMP}.log"

    
echo "Data dir: ${DATA_DIR}"
echo "Model type: ${MODEL_TYPE}"
echo "Penalty: ${PENALTY}"
echo "L1 ratio: ${L1_RATIO}"
echo "C: ${C}"
echo "NPZ path: ${NPZ_PATH}"
echo "Importance CSV: ${IMPORTANCE_CSV}"
echo "K: ${K}"
echo "Task: ${TASK}"
echo "Log file: ${LOG_FILE}"
echo "Log level: ${LOG_LEVEL}"


set +e
$PYTHON "$MAIN" \
    --log_level "$LOG_LEVEL" \
    --log_file "$LOG_FILE" \
    --model_type "$MODEL_TYPE" \
    --task "$TASK" \
    --npz_path "$NPZ_PATH" \
    --importance_csv "$IMPORTANCE_CSV" \
    --k "$K" \
    --C "$C" \
    --penalty "$PENALTY" \
    --l1_ratio "$L1_RATIO"
exit_code=$?
set -e

if [[ ${exit_code} -eq 0 ]]; then
    echo "OK: script finished at $(date)" | tee -a "${LOG_FILE}"
else
    echo "ERROR: script failed with exit code ${exit_code} at $(date)" | tee -a "${LOG_FILE}"
    exit ${exit_code}
fi