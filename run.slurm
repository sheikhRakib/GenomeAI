#!/bin/bash
#SBATCH --job-name=genomeai-full
#SBATCH --partition=partition-a
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --qos=test
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Full-dataset run of all three modes (baseline, single agent, five agents) in one job, each saving to
# results/<dataset>_<timestamp>[_<mode>_<security>]/.
# Submit from the repo root: sbatch run.slurm [windows.csv.gz from build_dna_windows.py] [main.py options, e.g. --security permissive]
# The job sets --mode itself, so do not pass it.
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs results

source ~/miniconda3/etc/profile.d/conda.sh
conda activate genomeai-py312

DATASET="${1:-data/VUS_64_genes_rating_dna_8192_non-vus.csv.gz}"
echo "job ${SLURM_JOB_ID:-local} on $(hostname) | dataset ${DATASET}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for mode in baseline single agents; do
  echo "== mode ${mode}"
  python -u main.py "${DATASET}" --out results --mode "${mode}" "${@:2}"
done
