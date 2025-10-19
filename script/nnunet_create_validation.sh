#!/bin/bash
# Usage: ./unet_create_validation.sh [OPTIONAL_PREP_DIR]
#SBATCH --job-name=unet_create_validation
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/unet_create_validation_%j.out
#SBATCH --error=logs/unet_create_validation_%j.err

set -euo pipefail

# --------- Config (edit if needed) ----------
# DATASET_ID=501
# DATASET_NAME=BraTS2017_4ch
# FOLD=0
# CFG=3d_fullres
# TR=nnUNetTrainer
# PLANS_ID=nnUNetPlans

DATASET_ID=502
DATASET_NAME=BraTS2017_4ch
FOLD=0
CFG=3d_fullres
TR=nnUNetTrainer
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
RAW="${nnUNet_raw}/Dataset${DATASET_ID}_${DATASET_NAME}"
RES="${nnUNet_results}/Dataset${DATASET_ID}_${DATASET_NAME}/${TR}__${PLANS_ID}__${CFG}"

# PREP: CLI arg wins; else use env-based default
if [[ $# -ge 1 ]]; then
  PREP="$1"
else
  PREP="${nnUNet_preprocessed}/Dataset${DATASET_ID}_${DATASET_NAME}"
fi

DJ="${RAW}/dataset.json"
PL="${RES}/plans.json"

# --- sanity checks ---
[[ -d "${nnUNet_raw:-}" ]] || { echo "ERROR: nnUNet_raw not set"; exit 1; }
[[ -d "${nnUNet_preprocessed:-}" ]] || { echo "ERROR: nnUNet_preprocessed not set"; exit 1; }
[[ -d "${nnUNet_results:-}" ]] || { echo "ERROR: nnUNet_results not set"; exit 1; }

[[ -d "$RAW" ]] || { echo "ERROR: RAW dataset dir missing: $RAW"; exit 1; }
[[ -f "$DJ"  ]] || { echo "ERROR: dataset.json missing: $DJ"; exit 1; }
[[ -d "$PREP" ]] || { echo "ERROR: PREP dir missing: $PREP"; exit 1; }

[[ -f "$PL" ]] || { echo "ERROR: plans.json missing: $PL"; exit 1; }
[[ -d "$RAW/imagesTr" && -d "$RAW/labelsTr" ]] || { echo "ERROR: imagesTr/labelsTr missing under $RAW"; exit 1; }
[[ -f "${RES}/fold_${FOLD}/checkpoint_best.pth" ]] || { echo "ERROR: checkpoint_best.pth missing under ${RES}/fold_${FOLD}"; exit 1; }

echo "[info] RAW=$RAW"
echo "[info] PREP=$PREP"
echo "[info] RES =$RES"
echo "[info] DJ  =$DJ"
echo "[info] PL  =$PL"

# --- extract fold-0 validation IDs ---
python - <<'PY' "$PREP"
import json, os, sys
prep = sys.argv[1]
spl = os.path.join(prep, "splits_final.json")
with open(spl) as f:
    splits = json.load(f)
val_ids = sorted(set(splits[0]["val"]))
with open("val_ids_${DATASET_ID}_fold${FOLD}.txt","w") as g:
    g.write("\n".join(val_ids))
print(f"[ok] Wrote {len(val_ids)} IDs to val_ids_${DATASET_ID}_fold${FOLD}.txt")
PY

# --- build subset (symlinks) ---
VAL_ROOT="$PWD/tmp_${DATASET_ID}_fold${FOLD}_val"
VALI="${VAL_ROOT}/imagesTr"; mkdir -p "$VALI"
VALL="${VAL_ROOT}/labelsTr"; mkdir -p "$VALL"

n_linked=0
while read -r ID; do
  # link channels
  for ch in 0000 0001 0002 0003; do
    src="${RAW}/imagesTr/${ID}_${ch}.nii.gz"
    dst="${VALI}/${ID}_${ch}.nii.gz"
    if [[ -f "$src" ]]; then
      ln -sf "$src" "$dst"
      ((n_linked++)) || true
    else
      echo "[warn] missing image channel: $src"
    fi
  done
  # link label
  lab_src="${RAW}/labelsTr/${ID}.nii.gz"
  lab_dst="${VALL}/${ID}.nii.gz"
  if [[ -f "$lab_src" ]]; then
    ln -sf "$lab_src" "$lab_dst"
  else
    echo "[warn] missing label: $lab_src"
  fi
done < val_ids_${DATASET_ID}_fold${FOLD}.txt

n_cases=$(ls -1 "$VALI"/*_0000.nii.gz 2>/dev/null | wc -l | awk '{print $1}')
echo "[info] Symlinked $n_cases validation cases (counted by *_0000.nii.gz)."
if [[ "$n_cases" -eq 0 ]]; then
  echo "[error] No validation cases were linked. Check splits_final.json IDs and RAW paths."
  exit 2
fi

# --- prediction output ---
OUTP="${RES}/fold_${FOLD}/predictions/validation"
mkdir -p "$OUTP"

export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

echo "[info] Running prediction into: $OUTP"
# IMPORTANT: pass checkpoint NAME only (nnUNet constructs the path)
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
  -device cuda \
  --save_probabilities \
  --disable_progress_bar

# verify predictions were created
n_preds=$(ls -1 "$OUTP"/*.nii.gz 2>/dev/null | wc -l | awk '{print $1}')
echo "[info] Predicted $n_preds files."
if [[ "$n_preds" -eq 0 ]]; then
  echo "[error] No predictions written to $OUTP. Check logs above."
  exit 3
fi

# --- evaluate ---
OUT_JSON="${OUTP}/summary.json"
echo "[info] Evaluating to: $OUT_JSON"
nnUNetv2_evaluate_folder \
  -djfile "$DJ" \
  -pfile  "$PL" \
  -o      "$OUT_JSON" \
  "$VALL" \
  "$OUTP"
echo "[ok] Wrote metrics to: $OUT_JSON"

# --- add HD95 per class to the JSON (writes summary_with_hd95.json) ---
ADD_HD95_PY="src/nnunet_add_hd95_to_eval_json.py"   # path to the helper script
OUT_JSON_HD95="${OUTP}/summary_with_hd95.json"

echo "[info] Adding HD95 to: ${OUT_JSON}"
# Option A: specify exactly which classes to compute (BraTS: 1,2,3)
# "${PYTHON}" "${ADD_HD95_PY}" \
#   -i "${OUT_JSON}" \
#   -o "${OUT_JSON_HD95}"

  # \
  # --classes 1,2,3

"${PYTHON}" "${ADD_HD95_PY}" \
  -i "$OUT_JSON" \
  -o "$OUT_JSON_HD95" \
  --dataset_json "${DJ}"


echo "[ok] Wrote: ${OUT_JSON_HD95}"
