#!/bin/bash
#   make_nnunet_input.sh    —  Make nnU-Net inputs (duplicate T1-CE into missing channels) and optionally run prediction.

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

DATA_DIR="output/nifti"
PYTHON="${ENV_PREFIX}/bin/python"
MAIN="src/make_nnunet_v2_input.py"

OUTPUT_DIR="output/nnunet_input"
mkdir -p "$OUTPUT_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/make_nnunet_v2_input_${TIMESTAMP}.log"

set +e
$PYTHON "$MAIN" \
    --src_root "$DATA_DIR" \
    --dst_root "$OUTPUT_DIR" \
    --log_level "$LOG_LEVEL" \
    --log_file "$LOG_FILE" \
    --mode copy \
    --overwrite
exit_code=$?
set -e


if [[ ${exit_code} -eq 0 ]]; then
    echo "OK: script finished at $(date)" | tee -a "${LOG_FILE}"
else
    echo "ERROR: script failed with exit code ${exit_code} at $(date)" | tee -a "${LOG_FILE}"
    # Uncomment to stop on first failure:
    exit ${exit_code}
fi    