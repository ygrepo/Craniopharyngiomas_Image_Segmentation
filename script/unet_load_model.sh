#!/bin/bash

set -euo pipefail

# --------- Config (edit if needed) ----------
DATASET_ID=501
DATASET_NAME=BraTS2017_4ch
FOLD=0
CFG=3d_fullres
TR=nnUNetTrainer
PLANS_ID=nnUNetPlans
# -------------------------------------------

# --- env setup ---
module purge
module load anaconda3/2023.09
module load proxy/jh-proxy-1.0
source "$(conda info --base)/etc/profile.d/conda.sh"

ENV_PREFIX="/projects/gbm_modeling/.conda/envs/mri"
conda activate "${ENV_PREFIX}"
PYTHON="${ENV_PREFIX}/bin/python"

LOG_DIR="logs"
LOG_LEVEL="DEBUG"
mkdir -p "$LOG_DIR"

# Set nnUNet paths (this defines $nnUNet_raw, $nnUNet_preprocessed, $nnUNet_results)
source script/set_unet_path.sh

# --- derive paths from envs ---
RES="${nnUNet_results}/Dataset${DATASET_ID}_${DATASET_NAME}/${TR}__${PLANS_ID}__${CFG}"

# --- sanity checks ---
[[ -d "${nnUNet_preprocessed:-}" ]] || { echo "ERROR: nnUNet_preprocessed not set"; exit 1; }
[[ -d "${nnUNet_results:-}" ]] || { echo "ERROR: nnUNet_results not set"; exit 1; }


[[ -f "${RES}/fold_${FOLD}/checkpoint_best.pth" ]] || { echo "ERROR: checkpoint_best.pth missing under ${RES}/fold_${FOLD}"; exit 1; }
MODEL_DIR="${RES}/fold_${FOLD}"
echo "[info] RES =$RES"
echo "[info] MODEL_DIR=$MODEL_DIR"
MAIN="src/nnunet_load_model.py"

ts=$(date +"%Y%m%d_%H%M%S")
log_file="${LOG_DIR}/${ts}_unet_load_model.log"
echo "  log_file : ${log_file}"

set +e
"${PYTHON}" "${MAIN}" \
  --log_file "${log_file}" \
  --log_level "${LOG_LEVEL}" \
  --model_dir "${MODEL_DIR}" \
  --fold "${FOLD}"
  exit_code=$?
set -e

if [[ ${exit_code} -eq 0 ]]; then
  echo "OK: script finished at $(date)" | tee -a "${log_file}"
else
  echo "ERROR: script failed with exit code ${exit_code} at $(date)" | tee -a "${log_file}"
  exit ${exit_code}
fi