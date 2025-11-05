#!/bin/bash
#   nnunet_predict_folds.sh    —  Predict 3D mask using nnU-Net v2.
#SBATCH --job-name=nnunet_predict_fold_array  # <-- Renamed for clarity
#SBATCH --output=logs/nnunet_predict_folds_%A_%a.out
#SBATCH --error=logs/nnunet_predict_folds_%A_%a.err
#SBATCH --time=24:00:00       
#SBATCH --partition=gpu
#SBATCH --array=0-4           
#SBATCH --gres=gpu:1          
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G             

set -euo pipefail

# ... (All your module/conda/path setup is correct) ...
module purge
module load anaconda3/2023.09
module load proxy/jh-proxy-1.0
source "$(conda info --base)/etc/profile.d/conda.sh"

ENV_PREFIX="/projects/gbm_modeling/.conda/envs/mri"
PIP_CACHE_DIR="/projects/gbm_modeling/.pip_cache"
CONDA_PKGS_DIRS="/projects/gbm_modeling/.conda/pkgs"

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
# ... (All your env vars and debug prints are correct) ...
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

echo "Job ID: ${SLURM_JOB_ID}, Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

# --- Use the array task ID to select the fold ---
DATASET_ID=504
DATASET_NAME=BraTS2017_4ch
FOLD=${SLURM_ARRAY_TASK_ID}  
CFG=3d_fullres
TR=nnUNetTrainerEarlyStopping
PLANS_ID=nnUNetResEncUNetMPlans

# --- derive paths from envs ---
RAW="${nnUNet_raw}/Dataset${DATASET_ID}_${DATASET_NAME}"
RES="${nnUNet_results}/Dataset${DATASET_ID}_${DATASET_NAME}/${TR}__${PLANS_ID}__${CFG}"

# --- Define output dir for this specific fold ---
OUTPUT_DIR="${RES}/folds_${FOLD}/test_predictions"
mkdir -p "${OUTPUT_DIR}"

echo "Predicting for FOLD=${FOLD}"
echo "Outputting to: ${OUTPUT_DIR}"

nnUNetv2_predict \
  -i ${RAW}/imagesTs/ \
  -o ${OUTPUT_DIR} \
  -d ${DATASET_ID} \
  -c ${CFG} \
  -f ${FOLD} \
  -tr ${TR} \
  -p ${PLANS_ID} \
  -device cuda \
  --save_probabilities 