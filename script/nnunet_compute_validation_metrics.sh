#!/bin/bash

set -euo pipefail

# --------- Config (edit if needed) ----------
# DATASET_ID=501
# DATASET_NAME=BraTS2017_4ch
# FOLD=0
# CFG=3d_fullres
# TR=nnUNetTrainer
# PLANS_ID=nnUNetPlans

# DATASET_ID=502
# DATASET_NAME=BraTS2017_4ch
# FOLD=0
# CFG=3d_fullres
# TR=nnUNetTrainer
# PLANS_ID=nnUNetResEncUNetMPlans

DATASET_ID=503
DATASET_NAME=CP
FOLD=0
CFG=3d_fullres
TR=EmaDiceEarlyStopTrainer
PLANS_ID=nnUNetResEncUNetMPlans

# -------------------------------------------

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

# --- derive paths from envs ---
RES="${nnUNet_results}/Dataset${DATASET_ID}_${DATASET_NAME}/${TR}__${PLANS_ID}__${CFG}"

# --- sanity checks ---
[[ -d "${nnUNet_preprocessed:-}" ]] || { echo "ERROR: nnUNet_preprocessed not set"; exit 1; }
[[ -d "${nnUNet_results:-}" ]] || { echo "ERROR: nnUNet_results not set"; exit 1; }


[[ -f "${RES}/fold_${FOLD}/checkpoint_best.pth" ]] || { echo "ERROR: checkpoint_best.pth missing under ${RES}/fold_${FOLD}"; exit 1; }

echo "[info] RES =$RES"

RAW="${nnUNet_raw}/Dataset${DATASET_ID}_${DATASET_NAME}"
echo "[info] RAW =$RAW"

# --- prediction output ---
OUTP="${RES}/fold_${FOLD}/predictions/validation"
mkdir -p "$OUTP"
IN_JSON_HD95="${OUTP}/${DATASET_ID}_fold${FOLD}_summary_with_hd95.json"
OUT_CASE_FN="${OUTP}/${DATASET_ID}_validation_metrics_cases.csv"
OUT_SUMMARY_FN="${OUTP}/${DATASET_ID}_validation_metrics_summary.csv"
OUT_MICRO_FN="${OUTP}/${DATASET_ID}_validation_metrics_micro.csv"
OUT_COUNTS_FN="${OUTP}/${DATASET_ID}_validation_metrics_counts.csv"

HD95_QUANTILES="90,95"
STD_TYPE="population"
# STD_TYPE="sample"

MAIN="src/nnunet_compute_validation_metrics.py"
DJ="${RAW}/dataset.json"


LOG_DIR="logs"
LOG_LEVEL="DEBUG"
mkdir -p "$LOG_DIR"
ts=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/${ts}_{DATASET_ID}_compute_validation_metrics.log"

"${PYTHON}" "${MAIN}" \
  --in_fn "${IN_JSON_HD95}" \
  --out_cases_fn "${OUT_CASE_FN}" \
  --out_summary_fn "${OUT_SUMMARY_FN}" \
  --out_micro_fn "${OUT_MICRO_FN}" \
  --counts_out_fn "${OUT_COUNTS_FN}" \
  --std_type "${STD_TYPE}" \
  --hd95_quantiles "${HD95_QUANTILES}" \
  --granularity labels \
  --rename_hd95_mm --round 3 \
  --log_file "${LOG_FILE}" \
  --log_level "${LOG_LEVEL}"
echo "[ok] Wrote: ${OUT_SUMMARY_FN}"
echo "[ok] Wrote: ${OUT_CASE_FN}"
echo "[ok] Wrote: ${OUT_MICRO_FN}"
