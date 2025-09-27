#!/bin/bash
#   unet_predict.sh    —  Predict 3D mask using nnU-Net.
#SBATCH --job-name=unet_predict
#SBATCH --output=logs/unet_predict_%A_%a.out
#SBATCH --error=logs/unet_predict_%A_%a.err
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
source $(conda info --base)/etc/profile.d/conda.sh

# --- Paths (edit if needed) ---
ENV_PREFIX="/projects/gbm_modeling/.conda/envs/mri"
PIP_CACHE_DIR="/projects/gbm_modeling/.pip_cache"
CONDA_PKGS_DIRS="/projects/gbm_modeling/.conda/pkgs"

# --- Keep installs off $HOME and avoid user-site leakage ---
mkdir -p "${PIP_CACHE_DIR}" "${CONDA_PKGS_DIRS}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS}"
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true

conda activate "${ENV_PREFIX}"

LOG_DIR="logs"
LOG_LEVEL="DEBUG"
mkdir -p "$LOG_DIR"

# make a flat folder and copy (follow symlinks) all channels into it
mkdir -p output/nnunet_input_flat
# find output/nnunet_input -type f -name "*_000[0-3].nii.gz" -exec cp -L {} output/nnunet_input_flat/ \;

# ls -1 output/nnunet_input_flat | wc -l
# ls -1 output/nnunet_input_flat/*_0000.nii.gz

# export nnUNet_results="$PWD/nnUNet_results"
# ls -R nnUNet_results/Dataset002_BRATS19/nnUNetTrainer__nnUNetPlans__3d_fullres

nnUNetv2_predict \
  -d 002 \
  -i output/nnunet_input_flat \
  -o output/nnunet_pred \
  -f 0 1 2 3 4 \
  -c 3d_fullres \
  -tr nnUNetTrainer \
  --save_probabilities \
  --disable_tta
