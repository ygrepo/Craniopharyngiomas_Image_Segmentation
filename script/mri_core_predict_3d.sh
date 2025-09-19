#!/bin/bash
#   mri_core_predict_3d.sh    —  Predict 3D mask using MRI-CORE.
#SBATCH --job-name=mri_core_predict_3d
#SBATCH --output=logs/mri_core_predict_3d_%A_%a.out
#SBATCH --error=logs/mri_core_predict_3d_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
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

LOG_DIR="logs"
LOG_LEVEL="DEBUG"
mkdir -p "$LOG_DIR"

BASE_DATA_DIR="output/slices"
PYTHON="${ENV_PREFIX}/bin/python"
MAIN="src/mricore_predict_3d.py"

OUTPUT_DIR="output/mri_core_3d"
mkdir -p "$OUTPUT_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/mri_core_predict_3d_${TIMESTAMP}.log"


set +e
$PYTHON "$MAIN" \
  --slices_root "$BASE_DATA_DIR" \
  --predict_root "$OUTPUT_DIR" \
  --checkpoint "pretrained_weights/sam_vit_b_mricore.pth" \
  --arch "vit_b" \
  --image_size 1024 \
  --device "cuda" \   # use "cpu" if you're not actually on a GPU node
  --log_level "$LOG_LEVEL" \
  --log_file "$LOG_FILE"
exit_code=$?
set -e


if [[ ${exit_code} -eq 0 ]]; then
    echo "OK: script  finished at $(date)" | tee -a "${LOG_FILE}"
else
    echo "ERROR: script failed with exit code ${exit_code} at $(date)" | tee -a "${LOG_FILE}"
    # Uncomment to stop on first failure:
    exit ${exit_code}
fi    