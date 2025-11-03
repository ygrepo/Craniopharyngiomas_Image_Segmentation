#!/bin/bash
#   unet_brats2017_resume.sh — Resume training (fold 0) with nnU-Net v2
#SBATCH --job-name=unet_brats2017_resume
#SBATCH --output=logs/unet_brats2017_resume_%j.out
#SBATCH --error=logs/unet_brats2017_resume_%j.err
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
FOlD=1

NUM_GPUS=4

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN          # or INFO when debugging comms
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
python - <<'PY'
import torch
print("PyTorch sees", torch.cuda.device_count(), "GPUs")
PY

export nnUNet_compile=1

# Results dir where fold_0 lives (this is nnU-Net’s default layout)
RESULTS_DIR="${nnUNet_results}/Dataset${DATASET_ID}_BraTS2017_4ch/${TR}__${PLANS_ID}__${CFG}"
CHK_LATEST="${RESULTS_DIR}/fold_${FOLD}/checkpoint_latest.pth"
CHK_BEST="${RESULTS_DIR}/fold_${FOLD}/checkpoint_best.pth"

echo "[info] RESULTS_DIR=${RESULTS_DIR}"
echo "[info] Checking checkpoints…"
if [[ -f "${CHK_LATEST}" ]]; then
  echo "[resume] Found checkpoint_latest.pth → continuing training with --c"
  nnUNetv2_train "${DATASET_ID}" "${CFG}" "${FOLD}" --c -num_gpus "${NUM_GPUS}" -p ${PLANS_ID} --npz
elif [[ -f "${CHK_BEST}" ]]; then
  echo "[warmstart] checkpoint_latest.pth not found. Using checkpoint_best.pth as pretrained weights (new optimizer/LR schedule)."
  nnUNetv2_train "${DATASET_ID}" "${CFG}" "${FOLD}" \
    -pretrained_weights "${CHK_BEST}" \
    -num_gpus "${NUM_GPUS}" -p ${PLANS_ID} --npz
else
  echo "[error] No checkpoint_latest.pth or checkpoint_best.pth found in:"
  echo "        ${RESULTS_DIR}/fold_${FOLD}"
  exit 1
fi