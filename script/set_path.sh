#!/bin/bash

# Learning Rate Finder Script for Favorita Grocery Sales Forecasting
# This script runs the learning rate finder to help determine optimal learning rates

set -euo pipefail

export nnUNet_raw="$PWD/data/nnUNet_raw"
export nnUNet_preprocessed="$PWD/data/nnUNet_preprocessed"
export nnUNet_results="$PWD/data/nnUNet_results"
