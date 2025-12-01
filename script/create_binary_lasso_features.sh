#!/bin/bash
#   create_binary_lasso_features.sh    —  Create Binary Lasso features.


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
MAIN="src/create_binary_lasso_features.py"

# Configuration
DATASET_ID=503
DATASET_NAME=CP
CFG=3d_fullres
TR=nnUNetTrainerEarlyStopping
PLANS_ID=nnUNetResEncUNetMPlans

MODEL_FOLDER="${nnUNet_results}/Dataset${DATASET_ID}_${DATASET_NAME}/${TR}__${PLANS_ID}__${CFG}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/create_lasso_features_${TIMESTAMP}.log"
LATENT_DIR="${MODEL_FOLDER}/latent_features"
RADIOMICS_CSV="nnUNet_raw/Dataset503_CP/radiomics_results.csv"
CLINICAL_CSV="data/CP"
MODEL_TYPE="postop"
OUTPUT_DIR="data/CP"
TEST_FRAC=0.20
    
echo "Model folder: ${MODEL_FOLDER}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Latent dir: ${LATENT_DIR}"
echo "Radiomics CSV: ${RADIOMICS_CSV}"
echo "Clinical CSV: ${CLINICAL_CSV}"
echo "Model type: ${MODEL_TYPE}"
echo "Test fraction: ${TEST_FRAC}"
echo "Log file: ${LOG_FILE}"
echo "Log level: ${LOG_LEVEL}"


set +e
$PYTHON "$MAIN" \
    --log_level "$LOG_LEVEL" \
    --log_file "$LOG_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --latent_dir "$LATENT_DIR" \
    --radiomics_csv "$RADIOMICS_CSV" \
    --clinical_csv "$CLINICAL_CSV" \
    --model_type "$MODEL_TYPE" \
    --test_frac "$TEST_FRAC"
exit_code=$?
set -e

if [[ ${exit_code} -eq 0 ]]; then
    echo "OK: script finished at $(date)" | tee -a "${LOG_FILE}"
else
    echo "ERROR: script failed with exit code ${exit_code} at $(date)" | tee -a "${LOG_FILE}"
    exit ${exit_code}
fi