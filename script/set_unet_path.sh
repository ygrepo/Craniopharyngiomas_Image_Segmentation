#!/bin/bash
# nnU-Net v1 + v2 paths (safe to export both)

# ----- v1 (used by KAIST) -----
export nnUNet_raw_data_base="$PWD/nnUNet_raw"
export nnUNet_preprocessed="$PWD/nnUNet_preprocessed"
export RESULTS_FOLDER="$PWD/trained_models"     # <- KAIST weights live here

# ----- v2 (used when you run v2) -----
export nnUNet_raw="$PWD/nnUNet_raw"
export nnUNet_results="$PWD/nnUNet_results"
