#!/bin/bash

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate genomeai-py312


python build_dna_windows.py --input dataset/clinvar_cm_rare_missense_dna_8192.csv --reference-fasta data/hg38.fa --subset non-vus

DATA=data/VUS_64_genes_rating_dna_8192_non-vus.csv.gz

# one GPU allocation for all three modes: baseline (no agents), single agent, five agents
srun \
  --job-name=genomeai \
  --partition=partition-a \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=32G \
  --time=12:00:00 \
  --qos=test \
  bash -c 'set -euo pipefail; for mode in baseline single agents; do echo "== mode $mode"; python -u main.py "$0" --mode "$mode" --security mock; done' "$DATA"

