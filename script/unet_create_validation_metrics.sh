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

# Set nnUNet paths (this defines $nnUNet_raw, $nnUNet_preprocessed, $nnUNet_results)
source script/set_unet_path.sh

# --- derive paths from envs ---
RES="${nnUNet_results}/Dataset${DATASET_ID}_${DATASET_NAME}/${TR}__${PLANS_ID}__${CFG}"

# --- sanity checks ---
[[ -d "${nnUNet_preprocessed:-}" ]] || { echo "ERROR: nnUNet_preprocessed not set"; exit 1; }
[[ -d "${nnUNet_results:-}" ]] || { echo "ERROR: nnUNet_results not set"; exit 1; }


[[ -f "${RES}/fold_${FOLD}/checkpoint_best.pth" ]] || { echo "ERROR: checkpoint_best.pth missing under ${RES}/fold_${FOLD}"; exit 1; }

echo "[info] RES =$RES"

# --- prediction output ---
OUTP="${RES}/fold_${FOLD}/predictions/validation"
mkdir -p "$OUTP"
IN_JSON_HD95="${OUTP}/summary_with_hd95.json"
OUT_CSV="summary_with_hd95.json"
OUT_CASES="summary_with_hd95_cases.csv"
OUT_CASES="${OUTP}/validation_count_metrics.csv"
HD95_QUANTILES="90,95"
STD_TYPE="sample"
LABEL_MAP="brats"
echo "[info] Creating validation metrics with to: ${OUT_CSV}"
MAIN="src/nnunet_eval_to_csv.py"

"${PYTHON}" "${MAIN}" \
  --in_fn "${IN_JSON_HD95}" \
  --out_cases_fn "${OUT_CASES}" \
  --out_summary_fn "${OUT_CSV}" \
  --std_type "${STD_TYPE}" \
  --hd95_quantiles "${HD95_QUANTILES}" \
  --label_map "${LABEL_MAP}"

echo "[ok] Wrote: ${OUT_CASES}"
echo "[ok] Wrote: ${OUT_CSV}"
