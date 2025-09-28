#!/bin/bash
#   prepare_cases_unnet_v1.sh    —  Prepare inputs for nnUNet v1

set -euo pipefail

source script/set_unet_path.sh

in=output/nifti
out=output/kaist_input_v1
mkdir -p "$out"
for id in 71899681 76686586 77364735 77402268 79391843; do
  src="$in/$id/${id}_T1_CE_3D_AX_ALIGNED.nii.gz"
  ln -sf "$src" "$out/BraTS2021_${id}_t1ce.nii.gz"
  ln -sf "$src" "$out/BraTS2021_${id}_t1.nii.gz"
  ln -sf "$src" "$out/BraTS2021_${id}_flair.nii.gz"
  ln -sf "$src" "$out/BraTS2021_${id}_t2.nii.gz"
done
