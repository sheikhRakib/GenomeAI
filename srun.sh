#!/bin/bash

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate genomeai-py312

python build_dna_windows.py --input dataset/input.csv --reference-fasta data/hg38.fa --subset non-vus

DATA=data/VUS_64_genes_rating_dna_8192_non-vus.csv.gz

# one GPU allocation for all three modes (baseline, LLM-driven single agent, five agents): the same steps as
# the batch job, so both runs stay identical
srun \
  --job-name=genomeai \
  --partition=partition-a \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=32G \
  --time=12:00:00 \
  --qos=test \
  bash run.slurm "$DATA"
