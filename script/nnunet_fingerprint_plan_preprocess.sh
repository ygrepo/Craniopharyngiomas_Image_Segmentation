#!/bin/bash
#   nnunet_fingerprint_plan_preprocess.sh    —  Convert CP to nnU-Net format.
#SBATCH --job-name=nnunet_convert_CP_data
#SBATCH --output=logs/nnunet_convert_CP_data_%A_%a.out
#SBATCH --error=logs/nnunet_convert_CP_data_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --partition=cpu
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G


set -euo pipefail

module purge
module load anaconda3/2023.09
module load proxy/jh-proxy-1.0
source "$(conda info --base)/etc/profile.d/conda.sh"

ENV_PREFIX="/projects/gbm_modeling/.conda/envs/mri"
conda activate "${ENV_PREFIX}"


mkdir -p logs
source script/set_unet_path.sh

#nnUNetv2_extract_fingerprint -d 502 --clean
#nnUNetv2_plan_and_preprocess -d 502 --verify_dataset_integrity --clean
#nnUNetv2_plan_and_preprocess -d 502 -pl nnUNetPlannerResEncM
nnUNetv2_plan_and_preprocess -d 503 --verify_dataset_integrity --clean -pl nnUNetPlannerResEncM -np 8

echo "[ok]"
