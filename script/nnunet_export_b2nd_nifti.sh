#!/bin/bash

set -euo pipefail



# --- env setup ---
module purge
module load anaconda3/2023.09
module load proxy/jh-proxy-1.0
source "$(conda info --base)/etc/profile.d/conda.sh"

ENV_PREFIX="/projects/gbm_modeling/.conda/envs/mri"
conda activate "${ENV_PREFIX}"
PYTHON="${ENV_PREFIX}/bin/python"

# Set nnUNet paths (this defines $nnUNet_raw, $nnUNet_preprocessed, $nnUNet_results)
source script/set_unet_path.sh


DATASET_ID=503
DATASET_NAME=CP
FOLD=0
CFG=3d_fullres
TR=EmaDiceEarlyStopTrainer
PLANS_ID=nnUNetResEncUNetMPlans

MAIN="src/export_b2nd_to_nifti.py"


LOG_DIR="logs"
LOG_LEVEL="DEBUG"
mkdir -p "$LOG_DIR"
ts=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/${ts}_${DATASET_ID}_compute_validation_metrics.log"

OUT_DIR="out_nifti"
mkdir -p "$OUT_DIR"

"${PYTHON}" "${MAIN}" \
  --root "$nnUNet_preprocessed/Dataset${DATASET_ID}_${DATASET_NAME}/nnUNetPlans_${CFG}" \
  --case 75062101 \
  --out "$OUT_DIR" \
  --log_file "${LOG_FILE}" \
  --log_level "${LOG_LEVEL}"

echo "[ok] Wrote: ${OUT_SUMMARY_FN}"
echo "[ok] Wrote: ${OUT_CASE_FN}"
echo "[ok] Wrote: ${OUT_MICRO_FN}"
