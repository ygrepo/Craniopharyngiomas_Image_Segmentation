#!/bin/bash
set -euo pipefail

# === Config you may edit ===
DATASET_ID=501
DATASET_NAME=BraTS2017_4ch
FOLD=0
CFG=3d_fullres
TR=nnUNetTrainer
PLANS_ID=nnUNetPlans
# ==========================

module purge
module load anaconda3/2023.09
module load proxy/jh-proxy-1.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "/projects/gbm_modeling/.conda/envs/mri"

# Bring back your nnU-Net env paths
source script/set_unet_path.sh

# Resolve paths from envs set above
RAW="${nnUNet_raw}/Dataset${DATASET_ID}_${DATASET_NAME}"
PREP="${nnUNet_preprocessed}/Dataset${DATASET_ID}_${DATASET_NAME}"
RES="${nnUNet_results}/Dataset${DATASET_ID}_${DATASET_NAME}/${TR}__${PLANS_ID}__${CFG}"

DJ="${RAW}/dataset.json"
PL="${RES}/plans.json"

# --- sanity checks ---
[[ -d "${nnUNet_raw:-}" ]] || { echo "nnUNet_raw not set by set_unet_path.sh"; exit 1; }
[[ -d "${nnUNet_preprocessed:-}" ]] || { echo "nnUNet_preprocessed not set by set_unet_path.sh"; exit 1; }
[[ -d "${nnUNet_results:-}" ]] || { echo "nnUNet_results not set by set_unet_path.sh"; exit 1; }

[[ -f "$DJ" ]] || { echo "Missing dataset.json at $DJ"; exit 1; }
[[ -f "$PL" ]] || { echo "Missing plans.json at $PL"; exit 1; }
[[ -d "$PREP" ]] || { echo "Missing preprocessed dir $PREP"; exit 1; }
[[ -d "${RAW}/imagesTr" && -d "${RAW}/labelsTr" ]] || { echo "RAW imagesTr/labelsTr missing under $RAW"; exit 1; }
[[ -f "${RES}/fold_${FOLD}/checkpoint_best.pth" ]] || { echo "Missing checkpoint_best.pth under ${RES}/fold_${FOLD}"; exit 1; }

# --- extract fold-0 validation IDs from splits_final.json ---
python - <<'PY' "$PREP"
import json, os, sys
prep = sys.argv[1]
with open(os.path.join(prep, "splits_final.json")) as f:
    splits = json.load(f)
val_ids = sorted(set(splits[0]["val"]))
with open("val_ids_fold0.txt","w") as g:
    g.write("\n".join(val_ids))
print(f"Wrote {len(val_ids)} IDs to val_ids_fold0.txt")
PY

# --- build subset folders with symlinks (validation images & labels) ---
VAL_ROOT="$PWD/tmp_val"
VALI="${VAL_ROOT}/imagesTr"; mkdir -p "$VALI"
VALL="${VAL_ROOT}/labelsTr"; mkdir -p "$VALL"

while read -r ID; do
  for ch in 0000 0001 0002 0003; do
    ln -sf "${RAW}/imagesTr/${ID}_${ch}.nii.gz" "${VALI}/${ID}_${ch}.nii.gz"
  done
  ln -sf "${RAW}/labelsTr/${ID}.nii.gz" "${VALL}/${ID}.nii.gz"
done < val_ids_fold0.txt

# --- prediction output (under your results tree) ---
OUTP="${RES}/fold_${FOLD}/predictions/validation"
mkdir -p "$OUTP"

# IMPORTANT: pass the CHECKPOINT *NAME* only, not a path!
nnUNetv2_predict \
  -i "$VALI" \
  -o "$OUTP" \
  -d "$DATASET_ID" \
  -p "$PLANS_ID" \
  -tr "$TR" \
  -c "$CFG" \
  -f "$FOLD" \
  -chk checkpoint_best.pth \
  --disable_tta \
  --disable_progress_bar

# --- evaluate against matching labels subset ---
OUT_JSON="${OUTP}/val_fold${FOLD}_results.json"
nnUNetv2_evaluate_folder \
  -djfile "$DJ" \
  -pfile  "$PL" \
  -o      "$OUT_JSON" \
  "$VALL" \
  "$OUTP"

echo "Wrote metrics to: $OUT_JSON"
