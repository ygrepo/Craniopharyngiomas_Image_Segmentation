#!/bin/bash
module purge
module load anaconda3/2023.09

source $(conda info --base)/etc/profile.d/conda.sh

# Create new env named cp in project .conda
conda create --prefix /projects/gbm_modeling/.conda/envs/cp python=3.12 -y

# Activate it
conda activate /projects/gbm_modeling/.conda/envs/cp


# Upgrade pip first
pip install --upgrade pip
