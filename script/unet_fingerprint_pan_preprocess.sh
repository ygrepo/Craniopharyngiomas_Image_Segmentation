#!/bin/bash
# Predict 3D masks with Fingerprint and PAN nnU-Net v2 data
#
set -euo pipefail

module purge
module load anaconda3/2023.09
module load proxy/jh-proxy-1.0
source "$(conda info --base)/etc/profile.d/conda.sh"

ENV_PREFIX="/projects/gbm_modeling/.conda/envs/mri"
conda activate "${ENV_PREFIX}"


mkdir -p logs
source script/set_unet_path.sh

nnUNetv2_extract_fingerprint -d 501
nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity


echo "[ok]"
