#!/bin/bash
#   nnunet_extract_latent_vectors.sh    —  Extract latent vectors from nnU-Net v2.
#SBATCH --job-name=nnunet_extract_latent_vectors
#SBATCH --output=logs/nnunet_extract_latent_vectors_%A_%a.out
#SBATCH --error=logs/nnunet_extract_latent_vectors_%A_%a.err
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
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

# Set nnUNet paths
source script/set_unet_path.sh

LOG_DIR="logs"
LOG_LEVEL="DEBUG"
mkdir -p "$LOG_DIR"

PYTHON="${ENV_PREFIX}/bin/python"
MAIN="src/nnunet_extract_latent_vectors.py"

# Configuration
DATASET_ID=503
DATASET_NAME=CP
FOLD=0
CFG=3d_fullres
TR=nnUNetTrainerEarlyStopping
PLANS_ID=nnUNetResEncUNetMPlans

MODEL_FOLDER="${nnUNet_results}/Dataset${DATASET_ID}_${DATASET_NAME}/${TR}__${PLANS_ID}__${CFG}"
OUTPUT_DIR="${MODEL_FOLDER}/latent_features"
CASE_ID="06780898"  # Edit as needed

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/nnunet_extract_latent_vectors_${TIMESTAMP}.log"

echo "Model folder: ${MODEL_FOLDER}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Case ID: ${CASE_ID}"

set +e
$PYTHON "$MAIN" \
    --log_level "$LOG_LEVEL" \
    --log_file "$LOG_FILE" \
    --model_folder "$MODEL_FOLDER" \
    --case_id "$CASE_ID" \
    --output_dir "$OUTPUT_DIR" \
    --device "cuda"
exit_code=$?
set -e

if [[ ${exit_code} -eq 0 ]]; then
    echo "OK: script finished at $(date)" | tee -a "${LOG_FILE}"
else
    echo "ERROR: script failed with exit code ${exit_code} at $(date)" | tee -a "${LOG_FILE}"
    exit ${exit_code}
fi