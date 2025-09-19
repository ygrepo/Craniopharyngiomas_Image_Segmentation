#!/bin/bash
#SBATCH --job-name=cp_predict
#SBATCH --output=logs/predict_%A_%a.out
#SBATCH --error=logs/predict_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --array=1-$(wc -l < case_list.txt)

set -euo pipefail

module load anaconda3/2023.09
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /projects/gbm_modeling/.conda/envs/cp

PYTHON="/projects/gbm_modeling/.conda/envs/cp/bin/python3"

# Set up log file with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs"
LOG_LEVEL="INFO"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/test_job_${TIMESTAMP}.log"
$PYTHON "$PWD/src/test_job.py" \
    2>&1 | tee -a "$LOG_FILE"

# Check the exit status of the Python script
EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo "Script completed successfully at $(date)" | tee -a "$LOG_FILE"
    exit 0
else
    echo "Error: Script failed with exit code $EXIT_CODE" | tee -a "$LOG_FILE"
    echo "Check the log file for details: $LOG_FILE"
    exit $EXIT_CODE
fi