#!/bin/bash
#   nnunet_predict_postprocessing.sh    —  Predict 3D mask using nnU-Net v2.
#SBATCH --job-name=nnunet_predict_postprocessing
#SBATCH --output=logs/nnunet_predict_postprocessing_%A_%a.out
#SBATCH --error=logs/nnunet_predict_postprocessing_%A_%a.err
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


export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN          # or INFO when debugging comms
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
python - <<'PY'
import torch
print("PyTorch sees", torch.cuda.device_count(), "GPUs")
PY

DATASET_ID=504
DATASET_NAME=BraTS2017_4ch
FOLD=0
CFG=3d_fullres
TR=nnUNetTrainerEarlyStopping
PLANS_ID=nnUNetResEncUNetMPlans

# --- derive paths from envs ---
RAW="${nnUNet_raw}/Dataset${DATASET_ID}_${DATASET_NAME}"
RES="${nnUNet_results}/Dataset${DATASET_ID}_${DATASET_NAME}/${TR}__${PLANS_ID}__${CFG}"

nnUNetv2_apply_postprocessing -i OUTPUT_FOLDER -o OUTPUT_FOLDER_PP -pp_pkl_file /projects/gbm_modeling/github/Craniopharyngiomas_Image_Segmentation/nnUNet_results/Dataset504_
nnUNetv2_apply_postprocessing -i OUTPUT_FOLDER -o OUTPUT_FOLDER_PP -pp_pkl_file /projects/gbm_modeling/github/Craniopharyngiomas_Image_Segmentation/nnUNet_results/Dataset504_
BraTS2017_4ch/nnUNetTrainerEarlyStopping__nnUNetResEncUNetMPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/postprocessing.pkl -np 8 -plans_json /projects/gbm_modeling/gith
ub/Craniopharyngiomas_Image_Segmentation/nnUNet_results/Dataset504_BraTS2017_4ch/nnUNetTrainerEarlyStopping__nnUNetResEncUNetMPlans__3d_fullres/crossval_results_folds_0_1_2_3
_4/plans.json


# Predict all folds
nnUNetv2_apply_postprocessing \
  -i ${RAW}/imagesTs/ \
  -o ${RES}/folds/predictions/test \
  -d ${DATASET_ID} \
  -c ${CFG} \
  -f 0 1 2 3 4 \
  -tr ${TR} \
  -p ${PLANS_ID} \
  -device cuda \
  -save_probabilities \
  -disable_postprocessing