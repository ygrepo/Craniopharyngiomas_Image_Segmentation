#!/bin/bash
#   unet_brats2017_deepdream.sh    —  Predict 3D mask using nnU-Net v2.
#SBATCH --job-name=unet_brats2017_deepdream
#SBATCH --output=logs/unet_brats2017_deepdream_%A_%a.out
#SBATCH --error=logs/unet_brats2017_deepdream_%A_%a.err
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

PYTHON="${ENV_PREFIX}/bin/python"

MAIN="src/nnunet_deep_dream_keras.py"
TRAINER_NAME="nnUNetTrainer"
CONFIGURATION="nnUNetPlans_3d_fullres"

$PYTHON "$MAIN" \
    --model_dir "nnUNet_results/Dataset501_BraTS2017_4ch/${TRAINER_NAME}__nnUNetPlans__${CONFIGURATION}/" \
    --preprocessed_dir "nnUNet_preprocessed/Dataset501_BraTS2017_4ch/" \
    --case_id "Brats17_CBICA_AAG_1" \
    --fold 0 \
    --checkpoint "checkpoint_final.pth" \
    --trainer_name "${TRAINER_NAME}" \
    --configuration "${CONFIGURATION}" \
    --log_file "$LOG_DIR/nnunet_deepdream_keras.log" \
    --log_level "$LOG_LEVEL"

echo "[ok]"
