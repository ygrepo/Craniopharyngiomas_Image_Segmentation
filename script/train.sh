#!/bin/bash

set -euo pipefail

export nnUNet_raw="$PWD/data/nnUNet_raw"
export nnUNet_preprocessed="$PWD/data/nnUNet_preprocessed"
export nnUNet_results="$PWD/data/nnUNet_results"
# 3D full-res (all folds)
for f in 0 1 2 3 4; do
  nnUNetv2_train 501 3d_fullres $f -p nnUNetResEncUNetXLPlans
done

# (Optional) 2D (all folds)
for f in 0 1 2 3 4; do
  nnUNetv2_train 501 2d $f -p nnUNetResEncUNetXLPlans
done
