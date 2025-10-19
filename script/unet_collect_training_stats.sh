#!/bin/bash
#   unet_collect_training_stats.sh    —  Collect training stats from nnU-Net v2 logs.
set -euo pipefail

module purge
module load anaconda3/2023.09
module load proxy/jh-proxy-1.0
source "$(conda info --base)/etc/profile.d/conda.sh"

ENV_PREFIX="/projects/gbm_modeling/.conda/envs/mri"
conda activate "${ENV_PREFIX}"
PYTHON="${ENV_PREFIX}/bin/python"
PYTHON="/Users/yvesgreatti/.pyenv/shims/python"
MAIN="src/nnunet_collect_training_stats.py"

LOG_DIR="logs"
LOG_LEVEL="DEBUG"
mkdir -p "$LOG_DIR"
source script/set_unet_path.sh

# INPUT_DIR="nnUNet_results/Dataset501_BraTS2017_4ch/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/"
# OUTPUT_FN="nnUNet_results/Dataset501_BraTS2017_4ch/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/training_metrics.csv"
INPUT_DIR="nnUNet_results/Dataset502_BraTS2017_4ch/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/fold_0/"
OUTPUT_FN="nnUNet_results/Dataset502_BraTS2017_4ch/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/fold_0/training_metrics.csv"

ts=$(date +"%Y%m%d_%H%M%S")
log_file="${LOG_DIR}/${ts}_unet_collect_training_stats.log"
echo "  log_file : ${log_file}"

set +e
"${PYTHON}" "${MAIN}" \
  --log_file "${log_file}" \
  --log_level "${LOG_LEVEL}" \
  --input_dir "${INPUT_DIR}" \
  --output_fn "${OUTPUT_FN}"
exit_code=$?
set -e

if [[ ${exit_code} -eq 0 ]]; then
  echo "OK: script finished at $(date)" | tee -a "${log_file}"
else
  echo "ERROR: script failed with exit code ${exit_code} at $(date)" | tee -a "${log_file}"
  exit ${exit_code}
fi