#!/bin/bash
#   nnunet_ensemble.sh    —  Ensemble 5 folds and apply post-processing
#SBATCH --job-name=nnunet_ensemble
#SBATCH --output=logs/nnunet_ensemble_%j.out
#SBATCH --error=logs/nnunet_ensemble_%j.err
#SBATCH --time=04:00:00        # Adjust as needed, 4h is a safe start
#SBATCH --partition=cpu   
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8     
#SBATCH --mem=64G            

set -euo pipefail

# --- Environment Setup ---
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


# --- Define your main results directory ---
DATASET_ID=504
DATASET_NAME=BraTS2017_4ch
TR=nnUNetTrainerEarlyStopping
PLANS_ID=nnUNetResEncUNetMPlans
CFG=3d_fullres

RES_DIR="${nnUNet_results}/Dataset${DATASET_ID}_${DATASET_NAME}/${TR}__${PLANS_ID}__${CFG}"

# --- Run the ensemble command ---
echo "Starting ensemble..."
nnUNetv2_ensemble \
    -i "${RES_DIR}/folds_0/test_predictions" \
       "${RES_DIR}/folds_1/test_predictions" \
       "${RES_DIR}/folds_2/test_predictions" \
       "${RES_DIR}/folds_3/test_predictions" \
       "${RES_DIR}/folds_4/test_predictions" \
    -o "${RES_DIR}/ensemble_predictions_final" \
    --save_npz \
    -np 8  # Use 8 processes, matching cpus-per-task

echo "Ensemble complete. Final predictions are in ${RES_DIR}/ensemble_predictions_final"