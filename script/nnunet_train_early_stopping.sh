#!/bin/bash
#   nnunet_train_ema_dice.sh    —  Train 3D mask using nnU-Net v2.
#SBATCH --job-name=nnunet_train_ema_dice
#SBATCH --output=logs/nnunet_train_ema_dice_%A_%a.out
#SBATCH --error=logs/nnunet_train_ema_dice_%A_%a.err
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
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
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7     # or however many GPUs you have

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN          # or INFO when debugging comms
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
python - <<'PY'
import torch
print("PyTorch sees", torch.cuda.device_count(), "GPUs")
PY

# Train all 5 folds:
export nnUNet_compile=1     # or set in your shell rc


DATASET_ID=504
CFG=3d_fullres
TR=nnUNetTrainerEarlyStopping
PLANS_ID=nnUNetResEncUNetXLPlans

FOLD=0
nnUNetv2_train $DATASET_ID $CFG  $FOLD -tr $TR -num_gpus 4 -p $PLANS_ID --npz
FOLD=1
nnUNetv2_train $DATASET_ID $CFG  $FOLD -tr $TR -num_gpus 4 -p $PLANS_ID --npz
FOLD=2
nnUNetv2_train $DATASET_ID $CFG  $FOLD -tr $TR -num_gpus 4 -p $PLANS_ID --npz
FOLD=3
nnUNetv2_train $DATASET_ID $CFG  $FOLD -tr $TR -num_gpus 4 -p $PLANS_ID --npz
FOLD=4
nnUNetv2_train $DATASET_ID $CFG  $FOLD -tr $TR -num_gpus 4 -p $PLANS_ID --npz
