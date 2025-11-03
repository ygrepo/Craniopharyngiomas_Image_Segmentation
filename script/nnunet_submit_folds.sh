#!/usr/bin/env bash
set -euo pipefail
mkdir -p logs
#for F in 0 1 2 3 4; do

for F in 1 2 3 4; do
  echo "Submitting fold $F with GPUs…"
  sbatch --export=ALL,FOLD=$F script/nnunet_train_fold.sh
done
