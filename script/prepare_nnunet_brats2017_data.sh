#!/bin/bash
#   prepare_nnunet_brats2017_data.sh    —  Prepare nnU-Net data for BraTS2017.
#SBATCH --job-name=prepare_nnunet_brats2017_data
#SBATCH --output=logs/prepare_nnunet_brats2017_data_%A_%a.out
#SBATCH --error=logs/prepare_nnunet_brats2017_data_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G

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

LOG_DIR="logs"
LOG_LEVEL="DEBUG"
mkdir -p "$LOG_DIR"

DATA_DIR="data/BraTS2017"
OUTPUT_DIR="data/nnUNet_raw/BraTS2017"
mkdir -p "$OUTPUT_DIR"

PYTHON="${ENV_PREFIX}/bin/python"
MAIN="src/nnUnet_util.py"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/nnunet_util_${TIMESTAMP}.log"

set +e
$PYTHON "$MAIN" \
    --src_root "$DATA_DIR" \
    --dst_root "$OUTPUT_DIR" \
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