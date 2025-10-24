#!/bin/bash
#SBATCH --job-name=nnunet_convert_CP_data
#SBATCH --output=logs/nnunet_fingerprint_plan_preprocess_%j.out
#SBATCH --error=logs/nnunet_fingerprint_plan_preprocess_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail
mkdir -p logs

# Threading to match SLURM allocation
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=${SLURM_CPUS_PER_TASK}


set -euo pipefail

module purge
module load anaconda3/2023.09
module load proxy/jh-proxy-1.0
source "$(conda info --base)/etc/profile.d/conda.sh"

ENV_PREFIX="/projects/gbm_modeling/.conda/envs/mri"
conda activate "${ENV_PREFIX}"


mkdir -p logs
source script/set_unet_path.sh

#nnUNetv2_extract_fingerprint -d 502 --clean
#nnUNetv2_plan_and_preprocess -d 502 --verify_dataset_integrity --clean
#nnUNetv2_plan_and_preprocess -d 502 -pl nnUNetPlannerResEncM
nnUNetv2_plan_and_preprocess -d 503 --verify_dataset_integrity --clean -pl nnUNetPlannerResEncM -np 8

echo "[ok]"
