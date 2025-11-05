#!/bin/bash
#   nnunet_apply_pp.sh    — Apply post-processing to ensembled predictions
#SBATCH --job-name=nnunet_apply_pp
#SBATCH --output=logs/nnunet_apply_pp_%j.out
#SBATCH --error=logs/nnunet_apply_pp_%j.err
#SBATCH --time=04:00:00        # 4h is a safe start
#SBATCH --partition=cpu        # <-- This is a CPU-only job
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8      # <-- Matches -np 8
#SBATCH --mem=64G              # 64G is likely plenty

set -euo pipefail

# --- Environment Setup (Same as your other scripts) ---
module purge
module load anaconda3/2023.09
module load proxy/jh-proxy-1.0
source "$(conda info --base)/etc/profile.d/conda.sh"

ENV_PREFIX="/projects/gbm_modeling/.conda/envs/mri"
PIP_CACHE_DIR="/projects/gbm_modeling/.pip_cache"
CONDA_PKGS_DIRS="/projects/gbm_modeling/.conda/pkgs"

export PIP_CACHE_DIR CONDA_PKGS_DIRS
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true

conda activate "${ENV_PREFIX}"
source script/set_unet_path.sh
# --- End of Environment Setup ---


# --- Define paths ---
DATASET_ID=504
DATASET_NAME=BraTS2017_4ch
TR=nnUNetTrainerEarlyStopping
PLANS_ID=nnUNetResEncUNetMPlans
CFG=3d_fullres

# This is the base directory for this model
RES_DIR="${nnUNet_results}/Dataset${DATASET_ID}_${DATASET_NAME}/${TR}__${PLANS_ID}__${CFG}"

# 1. INPUT: The folder created by your previous ensemble script
INPUT_DIR="${RES_DIR}/ensemble_predictions_final"

# 2. OUTPUT: A new folder for the final, clean .nii.gz files
OUTPUT_DIR="${RES_DIR}/ensemble_predictions_final_pp"

# 3. Path to the cross-validation results (from nnU-Net's recommendation)
CROSSVAL_DIR="${RES_DIR}/crossval_results_folds_0_1_2_3_4"

# 4. The specific files needed from that folder
PP_FILE="${CROSSVAL_DIR}/postprocessing.pkl"
PLANS_FILE="${CROSSVAL_DIR}/plans.json"

# Create the new output directory
mkdir -p "${OUTPUT_DIR}"

# --- Run the apply_postprocessing command ---
echo "Applying postprocessing..."
echo "Input (raw .npz):   ${INPUT_DIR}"
echo "Output (clean .nii.gz): ${OUTPUT_DIR}"
echo "Postprocessing file: ${PP_FILE}"
echo "Plans file:          ${PLANS_FILE}"

nnUNetv2_apply_postprocessing \
    -i "${INPUT_DIR}" \
    -o "${OUTPUT_DIR}" \
    -pp_pkl_file "${PP_FILE}" \
    -plans_json "${PLANS_FILE}" \
    -np 8  # Match --cpus-per-task

echo "Postprocessing complete. Final segmentations are in ${OUTPUT_DIR}"