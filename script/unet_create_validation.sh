#!/bin/bash
# Usage: ./unet_create_validation.sh [/path/to/nnUNet_preprocessed/Dataset501_BraTS2017_4ch]
DATASET_ID=501
DATASET_NAME=BraTS2017_4ch

# Prefer CLI arg → PREP env → nnUNet_preprocessed envs → hard fail
PREP="${1:-${PREP:-${NNUNet_preprocessed:-${nnUNet_preprocessed:-}}/Dataset${DATASET_ID}_${DATASET_NAME}}}"
if [ ! -d "$PREP" ]; then
  echo "ERROR: PREP not found: $PREP"
  echo "Set PREP or pass the path as the first argument."
  exit 1
fi
export PREP

python - <<'PY' "$PREP"
import json, os, sys
prep = sys.argv[1]
with open(os.path.join(prep, "splits_final.json")) as f:
    splits = json.load(f)
val_ids = sorted(set(splits[0]["val"]))
open("val_ids_fold0.txt","w").write("\n".join(val_ids))
print(f"Wrote {len(val_ids)} IDs to val_ids_fold0.txt")
PY