#!/bin/bash

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate genomeai-py312


srun \
  --job-name=genomeai \
  --partition=partition-a \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=32G \
  --time=12:00:00 \
  --qos=test \
  python train.py

