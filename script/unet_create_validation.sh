#!/bin/bash
# Usage: ./unet_create_validation.sh
DATASET_ID=501
DATASET_NAME=BraTS2017_4ch
PREP=nnUNet_preprocessed/Dataset501_BraTS2017_4ch

# # Prefer CLI arg → PREP env → nnUNet_preprocessed envs → hard fail
# PREP="${1:-${PREP:-${NNUNet_preprocessed:-${nnUNet_preprocessed:-}}/Dataset${DATASET_ID}_${DATASET_NAME}}}"
# if [ ! -d "$PREP" ]; then
#   echo "ERROR: PREP not found: $PREP"
#   echo "Set PREP or pass the path as the first argument."
#   exit 1
# fi
export PREP


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
source script/set_unet_path.sh


$PYTHON - <<'PY' "$PREP"
import json, os, sys
prep = sys.argv[1]
with open(os.path.join(prep, "splits_final.json")) as f:
    splits = json.load(f)
val_ids = sorted(set(splits[0]["val"]))
open("val_ids_fold0.txt","w").write("\n".join(val_ids))
print(f"Wrote {len(val_ids)} IDs to val_ids_fold0.txt")
PY

VALI=$PWD/tmp_val/images     ; mkdir -p "$VALI"
VALL=$PWD/tmp_val/labelsTr   ; mkdir -p "$VALL"

while read ID; do
  for ch in 0000 0001 0002 0003; do
    ln -sf "${RAW}/imagesTr/${ID}_${ch}.nii.gz" "${VALI}/${ID}_${ch}.nii.gz"
  done
  ln -sf "${RAW}/labelsTr/${ID}.nii.gz" "${VALL}/${ID}.nii.gz"
done < val_ids_fold0.txt

RES=nnUNet_results/Dataset501_BraTS2017_4ch/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0
OUTP=${RES}/predictions/validation
mkdir -p "$OUTP"

nnUNetv2_predict \
  -i "$VALI" \
  -o "$OUTP" \
  -d 501 -c 3d_fullres -f 0 \
  -chk ${RES}/fold_0/checkpoint_best.pth \
  -device cuda

OUT=${RES}/predictions/val_fold0_results.json
nnUNetv2_evaluate_folder -djfile "$DJ" -pfile "$PL" -o "$OUT" \
  "$VALL" "$OUTP"
echo "Wrote metrics to: $OUT"
