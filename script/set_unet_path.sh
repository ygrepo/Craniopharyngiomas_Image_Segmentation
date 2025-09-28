#!/bin/bash
#   set_path.sh    —  Set paths for nnUNet

set -euo pipefail

export nnUNet_raw="$PWD/nnUNet_raw"
export nnUNet_preprocessed="$PWD/nnUNet_preprocessed"
export nnUNet_results="$PWD/nnUNet_results"
export RESULTS_FOLDER=$PWD/trained_models          # v1 uses RESULTS_FOLDER (caps)
