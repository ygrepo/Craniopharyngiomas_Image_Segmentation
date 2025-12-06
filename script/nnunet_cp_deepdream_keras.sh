#!/bin/bash
#   nnunet_cp_deepdream_keras.sh    —  Predict 3D mask using nnU-Net v2.
#SBATCH --job-name=nnunet_cp_deepdream_keras
#SBATCH --output=logs/nnunet_cp_deepdream_keras_%A_%a.out
#SBATCH --error=logs/nnunet_cp_deepdream_keras_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G

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


export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN          # or INFO when debugging comms
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
python - <<'PY'
import torch
print("PyTorch sees", torch.cuda.device_count(), "GPUs")
PY


PYTHON="${ENV_PREFIX}/bin/python"

MAIN="src/nnunet_deep_dream_keras_cp.py"


DATASET_ID=503
DATASET_NAME=CP
FOLD=0
PLANS_ID=nnUNetResEncUNetMPlans
TR="nnUNetTrainerEarlyStopping"
CONFIGURATION="3d_fullres"

RES="nnUNet_results/Dataset${DATASET_ID}_${DATASET_NAME}/${TR}__${PLANS_ID}__${CONFIGURATION}"
PREP="nnUNet_preprocessed/Dataset${DATASET_ID}_${DATASET_NAME}"
DATA_DIR="${PREP}/nnUNetPlans_${CONFIGURATION}"
OUTPUT_DIR="output/deepdream_cp_keras"
export TORCHDYNAMO_DISABLE=1

mkdir -p "$OUTPUT_DIR"
$PYTHON "$MAIN" \
    --model_dir "${RES}" \
    --preprocessed_dir "${DATA_DIR}" \
    --fold 0 \
    --checkpoint "checkpoint_best.pth" \
    --case_id 35340174 \
    --output_dir "$OUTPUT_DIR" \
    --layer_regex "encoder" \
    --target_idx 0 \
    --iterations 80 \
    --step_size 0.02 \
    --num_octaves 3 \
    --octave_scale 1.4 \
    --slice_idx "40,60,80" \
    --modality_idx 1 \
    --feature_importance_csv output/metrics/preop_binary_elasticnet_l1ratio_0.3_feature_importance.csv \
    --latent_rank 1 \
    --log_file "$LOG_DIR/nnunet_deepdream_cp_keras.log" \
    --log_level "$LOG_LEVEL"

echo "[ok]"
