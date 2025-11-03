#!/usr/bin/env bash
set -euo pipefail
mkdir -p logs
#for F in 0 1 2 3 4; do

for F in 1 2 3 4; do
  sbatch \
    --export=ALL,FOLD=$F \
    --job-name=brats_f$F \
    --output=logs/brats_f${F}_%j.out \
    --error=logs/brats_f${F}_%j.err \
    script/nnunet_train_fold.sh
done