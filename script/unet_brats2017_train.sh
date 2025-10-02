#!/bin/bash
#   unet_brats2017_train.sh    —  Train 3D mask using nnU-Net v2.
#SBATCH --job-name=unet_brats2017_train
#SBATCH --output=logs/unet_brats2017_train_%A_%a.out
#SBATCH --error=logs/unet_brats2017_train_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G

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

source script/set_unet_path.sh

# Choose a GPU id if needed:
export CUDA_VISIBLE_DEVICES=0

# Train all 5 folds:
nnUNetv2_train 501 3d_fullres 0 -num_gpus 2 -p nnUNetResEncUNetMPlans

# nnUNetv2_train 501 3d_fullres 1
# nnUNetv2_train 501 3d_fullres 2
# nnUNetv2_train 501 3d_fullres 3
# nnUNetv2_train 501 3d_fullres 4

