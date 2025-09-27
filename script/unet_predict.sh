#!/usr/bin/env bash
set -euo pipefail

# --------- CONFIG (edit the top 4 lines as needed) ---------
RESULTS_DIR="${RESULTS_DIR:-$PWD/nnUNet_results}"  # where models live
ZIP_PATH="${1:-}"                                  # optional: /path/to/Dataset002_BraTS2021_3d_fullres.zip
DATASET_ID="${DATASET_ID:-002}"                    # BraTS21 is commonly 002
IN_DIR="${IN_DIR:-output/nnunet_input}"            # your prepared inputs (per-case _0000.._0003)
OUT_DIR="${OUT_DIR:-output/nnunet_pred}"           # predictions go here
FOLDS="${FOLDS:-all}"
CONFIG="${CONFIG:-3d_fullres}"
TRAINER="${TRAINER:-nnUNetTrainer}"
EXTRA_PRED_ARGS="${EXTRA_PRED_ARGS:---save_probabilities --disable_tta}"  # tweak as you like
# ----------------------------------------------------------

echo "[1/4] Ensuring nnUNet env"
mkdir -p "$RESULTS_DIR" "$OUT_DIR"
export nnUNet_results="$RESULTS_DIR"
echo "  nnUNet_results=$nnUNet_results"

if [[ -n "$ZIP_PATH" ]]; then
  echo "[2/4] Installing pretrained model from ZIP: $ZIP_PATH"
  if ! command -v nnUNetv2_install_pretrained_model_from_zip >/dev/null 2>&1; then
    echo "  ERROR: nnUNetv2_install_pretrained_model_from_zip not found. Activate the nnUNet v2 env." >&2
    exit 2
  fi
  nnUNetv2_install_pretrained_model_from_zip "$ZIP_PATH"
else
  echo "[2/4] No ZIP provided, skipping install (assuming model already installed)."
fi

echo "[info] Models under \$nnUNet_results:"
ls -1 "$nnUNet_results" || true

echo "[3/4] Materializing symlinks in $IN_DIR (cp -L → real files)"
if [[ ! -d "$IN_DIR" ]]; then
  echo "  ERROR: Input dir '$IN_DIR' not found." >&2
  exit 3
fi
# Replace any symlinked channel file with a real file (follows the link)
find "$IN_DIR" -type l -name "*_000[0-3].nii.gz" | while read -r L; do
  TMP="${L}.tmp"
  cp -L "$L" "$TMP"
  mv -f "$TMP" "$L"
done

echo "[4/4] Running nnUNetv2_predict"
set -x
nnUNetv2_predict \
  -d "$DATASET_ID" \
  -i "$IN_DIR" \
  -o "$OUT_DIR" \
  -f "$FOLDS" \
  -c "$CONFIG" \
  -tr "$TRAINER" \
  $EXTRA_PRED_ARGS
set +x

echo "[done] Predictions → $OUT_DIR"
