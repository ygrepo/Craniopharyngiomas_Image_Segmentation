#!/bin/bash
#   unet_brats2017_find_best_configuration.sh — Find best configuration
#SBATCH --job-name=unet_brats2017_find_best_configuration
#SBATCH --output=logs/unet_brats2017_find_best_configuration_%j.out
#SBATCH --error=logs/unet_brats2017_find_best_configuration_%j.err
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G


set -euo pipefail

module purge
module load anaconda3/2023.09
module load proxy/jh-proxy-1.0
source "$(conda info --base)/etc/profile.d/conda.sh"

# --- Paths (same as your train script) ---
ENV_PREFIX="/projects/gbm_modeling/.conda/envs/mri"
PIP_CACHE_DIR="/projects/gbm_modeling/.pip_cache"
CONDA_PKGS_DIRS="/projects/gbm_modeling/.conda/pkgs"

export PIP_CACHE_DIR CONDA_PKGS_DIRS
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true

conda activate "${ENV_PREFIX}"


# nnU-Net envs (your helper)
source script/set_unet_path.sh

# ------------------ config ------------------

LOG_DIR="logs"
LOG_LEVEL="DEBUG"
mkdir -p "$LOG_DIR"


DATASET_ID=504
CFG=3d_fullres
TR=nnUNetTrainerEarlyStopping
PLANS_ID=nnUNetResEncUNetMPlans

NUM_GPUS=4
NUM_CPUS=8

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN          # or INFO when debugging comms
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
python - <<'PY'
import torch
print("PyTorch sees", torch.cuda.device_count(), "GPUs")
PY


# Results dir where fold_0 lives (this is nnU-Net’s default layout)
RESULTS_DIR="${nnUNet_results}/Dataset${DATASET_ID}_BraTS2017_4ch/${TR}__${PLANS_ID}__${CFG}"
echo "[info] RESULTS_DIR=${RESULTS_DIR}"

nnUNetv2_find_best_configuration -c ${CFG} -p ${PLANS_ID} -tr ${TR} -np  ${NUM_CPUS} ${DATASET_ID}