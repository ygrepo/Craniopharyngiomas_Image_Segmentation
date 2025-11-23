#!/bin/bash
#   nnunet_compute_chiasm_metrics.sh    —  Compute tumor–chiasm metrics.
#SBATCH --job-name=nnunet_compute_chiasm_metrics
#SBATCH --output=logs/nnunet_compute_chiasm_metrics_%A_%a.out
#SBATCH --error=logs/nnunet_compute_chiasm_metrics_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G

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

PYTHON="${ENV_PREFIX}/bin/python"
MAIN="src/nnunet_compute_chiasm_metrics_2.py"
BASE_DIR="nnUNet_raw/Dataset503_CP"
PRED_DIR="nnUNet_results/Dataset503_CP/EmaDiceEarlyStopTrainer__nnUNetResEncUNetMPlans__3d_fullres/fold_0/predictions/test"


LOG_DIR="logs"
LOG_LEVEL="DEBUG"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/nnunet_compute_chiasm_metrics_${TIMESTAMP}.log"

set +e
$PYTHON "$MAIN" \
    --base_dir "$BASE_DIR" \
    --pred_dir "$PRED_DIR" \
    --log_level "$LOG_LEVEL" \
    --log_file "$LOG_FILE"
exit_code=$?
set -e

if [[ ${exit_code} -eq 0 ]]; then
    echo "OK: script finished at $(date)" | tee -a "${LOG_FILE}"
else
    echo "ERROR: script failed with exit code ${exit_code} at $(date)" | tee -a "${LOG_FILE}"
    # Uncomment to stop on first failure:
    exit ${exit_code}
fi    