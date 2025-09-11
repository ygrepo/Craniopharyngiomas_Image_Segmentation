#!/bin/bash
#SBATCH --job-name=test_job       # Job name
#SBATCH --output=test.out     # Standard output log
#SBATCH --error=test.err      # Standard error log
#SBATCH --time=00:05:00            # Time limit (hh:mm:ss)
#SBATCH --partition=standard       # Partition/queue name
#SBATCH --ntasks=1                 # Number of tasks (processes)
#SBATCH --cpus-per-task=1          # Number of CPU cores per task
#SBATCH --mem=1G                   # Memory per node
#SBATCH --mail-type=END,FAIL       # Mail events (NONE, BEGIN, END, FAIL, ALL)


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