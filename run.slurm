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

# Full-dataset run: validate, score with frozen Evo2, save results to results/<dataset>_<timestamp>[_agents_<security>]/.
# Submit from the repo root: sbatch run.slurm [dataset.csv] [main.py options, e.g. --mode agents --security mock]
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs results

source ~/miniconda3/etc/profile.d/conda.sh
conda activate genomeai-py312

DATASET="${1:-dataset/clinvar_cm_rare_missense_dna_8192.csv}"
echo "job ${SLURM_JOB_ID:-local} on $(hostname) | dataset ${DATASET}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python -u main.py "${DATASET}" --out results "${@:2}"
