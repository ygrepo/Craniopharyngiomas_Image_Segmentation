#!/bin/bash
#   unet_brats2017_predict.sh    —  Predict 3D mask using nnU-Net v2.
#SBATCH --job-name=unet_brats2017_predict
#SBATCH --output=logs/unet_brats2017_predict_%A_%a.out
#SBATCH --error=logs/unet_brats2017_predict_%A_%a.err
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

source script/set_unet_path.sh

# Choose a GPU id if needed:
#export CUDA_VISIBLE_DEVICES=0
# DATASET_ID=501
# DATASET_NAME=BraTS2017_4ch
# FOLD=0
# CFG=3d_fullres
# TR=nnUNetTrainer
# PLANS_ID=nnUNetPlans


DATASET_ID=502
DATASET_NAME=BraTS2017_4ch
FOLD=0
CFG=3d_fullres
TR=nnUNetTrainer
PLANS_ID=nnUNetResEncUNetMPlans

# --- derive paths from envs ---
RAW="${nnUNet_raw}/Dataset${DATASET_ID}_${DATASET_NAME}"
RES="${nnUNet_results}/Dataset${DATASET_ID}_${DATASET_NAME}/${TR}__${PLANS_ID}__${CFG}"

# Predict Single fold (e.g., fold 0)
nnUNetv2_predict \
  -i ${RAW}/imagesTs/ \
  -o ${RES}/fold_${FOLD}/predictions/test \
  -d ${DATASET_ID} \
  -c ${CFG} \
  -f ${FOLD} \
  -tr ${TR} \
  -p ${PLANS_ID} \
  -chk ${RES}/fold_${FOLD}/checkpoint_best.pth \
  -device cuda
