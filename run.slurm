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

# Full-dataset run of the chosen modes (--baseline, --single-agent, --multi-agent; none given runs all three) in one
# job, each saving to results/<dataset>_<timestamp>[_<mode>_<security>]/.
# The agents of both agent modes are driven by an LLM (Qwen3-8B) on a local vLLM server, started for those modes only.
# Submit from the repo root: sbatch run.slurm [windows.csv.gz from build_dna_windows.py] [mode flags] [main.py options]
# e.g. sbatch run.slurm data/VUS_64_genes_rating_dna_8192_non-vus.csv.gz --baseline --single-agent --security permissive
# srun.sh runs this same script interactively.
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs results

source ~/miniconda3/etc/profile.d/conda.sh
conda activate genomeai-py312

DATASET=data/VUS_64_genes_rating_dna_8192_non-vus.csv.gz
if [ $# -gt 0 ] && [[ "$1" != -* ]]; then
  DATASET="$1"
  shift
fi
modes=()
opts=()
for arg in "$@"; do
  case "${arg}" in
    --baseline|--single-agent|--multi-agent) modes+=("${arg}") ;;
    *) opts+=("${arg}") ;;
  esac
done
[ ${#modes[@]} -gt 0 ] || modes=(--baseline --single-agent --multi-agent)
LLM_MODEL="${LLM_MODEL:-Qwen/Qwen3-8B}"
PORT=$((8000 + ${SLURM_JOB_ID:-0} % 1000))  # two jobs on one node must not share a port
LLM_LOG="logs/vllm-${SLURM_JOB_ID:-local}.log"
echo "job ${SLURM_JOB_ID:-local} on $(hostname) | dataset ${DATASET} | llm ${LLM_MODEL}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

LLM_PID=
trap '[ -n "${LLM_PID}" ] && kill "${LLM_PID}" 2>/dev/null' EXIT  # never leave the server running

start_llm() {
  # 127.0.0.1: no other user on the node can reach it. Half the GPU: Evo2 runs next to the server in the agent modes
  # (no score cache unless --cache), and needs up to 16.7 GiB for an 8192 bp window.
  HF_HUB_OFFLINE=1 vllm serve "${LLM_MODEL}" --host 127.0.0.1 --port "${PORT}" --max-model-len 8192 \
    --gpu-memory-utilization 0.5 --enable-auto-tool-choice --tool-call-parser hermes > "${LLM_LOG}" 2>&1 &
  LLM_PID=$!
  until curl -sf "http://127.0.0.1:${PORT}/health" > /dev/null; do
    if ! kill -0 "${LLM_PID}" 2>/dev/null; then
      echo "vLLM failed to start, see ${LLM_LOG}"
      exit 1
    fi
    sleep 5
  done
  echo "vLLM serving ${LLM_MODEL} on port ${PORT}"
}

stop_llm() {
  kill "${LLM_PID}" 2>/dev/null || true
  wait "${LLM_PID}" 2>/dev/null || true
  LLM_PID=
}

# one main.py call per mode, in this order; vLLM holds the GPU only during the agent modes.
# expandable_segments: without it Evo2's 8192 bp forward pass runs out of memory next to vLLM (fragmentation)
for mode in --baseline --single-agent --multi-agent; do
  [[ " ${modes[*]} " == *" ${mode} "* ]] || continue
  echo "== ${mode}"
  llm_args=()
  if [ "${mode}" != --baseline ]; then
    start_llm
    llm_args=(--llm-url "http://127.0.0.1:${PORT}/v1" --llm-model "${LLM_MODEL}")
  fi
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python -u main.py "${DATASET}" --out results "${mode}" "${llm_args[@]}" "${opts[@]}"
  if [ "${mode}" != --baseline ]; then
    stop_llm
  fi
done
