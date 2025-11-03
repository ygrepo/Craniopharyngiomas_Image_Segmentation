#!/bin/bash
#SBATCH --job-name=brats_f${FOLD}
#SBATCH --output=logs/brats_f${FOLD}_%j.out
#SBATCH --error=logs/brats_f${FOLD}_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=72:00:00
## Optional: uncomment to avoid port clashes if multiple jobs share a node
#SBATCH --exclusive

set -euo pipefail

module purge
module load anaconda3/2023.09
module load proxy/jh-proxy-1.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /projects/gbm_modeling/.conda/envs/mri

# nnU-Net paths (must export nnUNet_raw / nnUNet_preprocessed / nnUNet_results)
source script/set_unet_path.sh

# ---------------- config ----------------
DATASET_ID=504
CFG=3d_fullres
TR=nnUNetTrainerEarlyStopping
PLANS_ID=nnUNetResEncUNetMPlans
FOLD=${FOLD:?FOLD not exported}     # 0..4

# 4 GPUs per fold (DDP)
NUM_GPUS=4

# DDP/NCCL hygiene
export nnUNet_compile=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# Unique port per fold (prevents clashes if multiple multi-GPU jobs share a node)
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((12340 + FOLD))

echo "[info] fold=${FOLD}  gpus=${NUM_GPUS}"
python - <<'PY'
import torch
print("PyTorch sees", torch.cuda.device_count(), "GPUs")
PY

RESULTS_DIR="${nnUNet_results}/Dataset${DATASET_ID}_BraTS2017_4ch/${TR}__${PLANS_ID}__${CFG}"
CHK_LATEST="${RESULTS_DIR}/fold_${FOLD}/checkpoint_latest.pth"
CHK_BEST="${RESULTS_DIR}/fold_${FOLD}/checkpoint_best.pth"

if [[ -f "${CHK_LATEST}" ]]; then
  echo "[resume] Using checkpoint_latest.pth"
  nnUNetv2_train "$DATASET_ID" "$CFG" "$FOLD" \
    -tr "$TR" -p "$PLANS_ID" -num_gpus "$NUM_GPUS" --npz --c
elif [[ -f "${CHK_BEST}" ]]; then
  echo "[warmstart] Using checkpoint_best.pth"
  nnUNetv2_train "$DATASET_ID" "$CFG" "$FOLD" \
    -tr "$TR" -p "$PLANS_ID" -num_gpus "$NUM_GPUS" --npz \
    -pretrained_weights "$CHK_BEST"
else
  echo "[fresh] No checkpoints found; starting from scratch"
  nnUNetv2_train "$DATASET_ID" "$CFG" "$FOLD" \
    -tr "$TR" -p "$PLANS_ID" -num_gpus "$NUM_GPUS" --npz
fi
