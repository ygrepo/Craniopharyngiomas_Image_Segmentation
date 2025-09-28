#!/bin/bash
# Predict 3D masks with KAIST nnU-Net v1 models and ensemble
#SBATCH --job-name=unet_v1_predict
#SBATCH --output=logs/unet_v1_predict_%A.out
#SBATCH --error=logs/unet_v1_predict_%A.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G

set -euo pipefail

module purge
module load anaconda3/2023.09
module load proxy/jh-proxy-1.0
source "$(conda info --base)/etc/profile.d/conda.sh"

ENV_PREFIX="/projects/gbm_modeling/.conda/envs/mri"
conda activate "${ENV_PREFIX}"

# Make sure nnU-Net v1 CLIs exist (install once if needed)
if ! command -v nnUNet_predict >/dev/null 2>&1; then
  pip install "nnunet==1.7.0"
fi

mkdir -p logs
source script/set_unet_path.sh

IN=output/kaist_input
OUT_A=output/kaist_pred_A
OUT_B=output/kaist_pred_B
OUT_F=output/kaist_pred_final

# Model A (BatchNorm)
nnUNet_predict \
  -i "$IN" -o "$OUT_A" \
  -t Task500_BraTS2021 -m 3d_fullres -p nnUNetPlansv2.1 \
  -tr nnUNetTrainerV2BraTSRegions_DA4_BN_BD \
  -f 0 1 2 3 4 \
  --save_npz

# Model B (GroupNorm, large UNet)
nnUNet_predict \
  -i "$IN" -o "$OUT_B" \
  -t Task500_BraTS2021 -m 3d_fullres -p nnUNetPlansv2.1 \
  -tr nnUNetTrainerV2BraTSRegions_DA4_BN_BD_largeUnet_Groupnorm \
  -f 0 1 2 3 4 \
  --save_npz

# Ensemble A + B
nnUNet_ensemble -f "$OUT_A" "$OUT_B" -o "$OUT_F"

echo "[ok] KAIST predictions written to: $OUT_F"
