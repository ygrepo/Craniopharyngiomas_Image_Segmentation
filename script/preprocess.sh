#!/bin/bash
#   preprocess.sh    —  Preprocess craniopharyngioma MRI

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

BASE_DATA_DIR="data"
PYTHON="${ENV_PREFIX}/bin/python"   
MAIN="src/preprocess.py"

OUTPUT_DIR="output/preprocessed"
mkdir -p "$OUTPUT_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/preprocess_${TIMESTAMP}.log"

set +e
$PYTHON "$MAIN" \
    --in_dir "$BASE_DATA_DIR" \
    --out_dir "$OUTPUT_DIR" \
    --modalities T1_CE_3D_AX_ALIGNED \
    --spacing 1.0 \
    --roi_from_mask "centroid" \
    --roi_size_mm 96 96 96 \
    --mask_tag "Tumor.seg" \
    --save_mask \
    --log_level "$LOG_LEVEL" \
    --log_file "$LOG_FILE"
exit_code=$?
set -e

# set +e
# $PYTHON "$MAIN" \
#     --in_dir "$BASE_DATA_DIR" \
#     --out_dir "$OUTPUT_DIR" \
#     --modalities T1_CE_3D_AX_ALIGNED \
#     --spacing 1.0 \
#     --roi_from_mask bbox \
#     --bbox_pad_mm 8 8 8 \
#     --mask_tag "Tumor.seg" \
#     --save_mask \
#     --log_level "$LOG_LEVEL" \
#     --log_file "$LOG_FILE"
# exit_code=$?
# set -e


if [[ ${exit_code} -eq 0 ]]; then
    echo "OK: Preprocessing finished at $(date)" | tee -a "${LOG_FILE}"
else
    echo "ERROR: Preprocessing failed with exit code ${exit_code} at $(date)" | tee -a "${LOG_FILE}"
    # Uncomment to stop on first failure:
    exit ${exit_code}
fi    