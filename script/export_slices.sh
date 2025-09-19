#!/bin/bash
#   export_slices.sh    —  Export per-slice PNGs and metadata.

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

BASE_DATA_DIR="output/preprocessed"
PYTHON="${ENV_PREFIX}/bin/python"
MAIN="src/export_slices.py"

OUTPUT_DIR="output/slices"
mkdir -p "$OUTPUT_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/export_slices_${TIMESTAMP}.log"


set +e
$PYTHON "$MAIN" \
    --nii_root "$BASE_DATA_DIR" \
    --out_root "$OUTPUT_DIR" \
    --prefer "*T1_CE_3D_AX_ALIGNED*.nii.gz" \
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