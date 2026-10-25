#!/usr/bin/env bash
# Full benchmark set on one OSCAR GPU; writes plain-text outputs to results/<date>-<gpu>-<job>/.
# Headline numbers should come from this script.
#   sbatch scripts/oscar_bench.sbatch          (or srun ... bash scripts/oscar_bench.sbatch)
# BENCH_COMMIT=<sha> records which source tree was benchmarked (the OSCAR copy has no .git).
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -t 0:45:00
#SBATCH -o artifacts/oscar-bench-%j.out
set -euo pipefail

VENV=${VENV:-$HOME/hoad.venv}
module load cuda/12.9.0-cinr >/dev/null 2>&1 || true
export PYTHONPATH=src
export PATH="$VENV/bin:$PATH"
mkdir -p artifacts

gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | sed 's/NVIDIA //; s/ /-/g' | tr -cd '[:alnum:]-')
out="results/$(date +%Y-%m-%d)-${gpu}-${SLURM_JOB_ID:-nojob}"
mkdir -p "$out"
echo "${BENCH_COMMIT:-unknown}" > "$out/commit.txt"
nvidia-smi > "$out/nvidia-smi.txt"

"$VENV/bin/python" setup.py -q build_ext --inplace > "$out/build.txt" 2>&1
"$VENV/bin/python" -m pytest -q > "$out/pytest.txt" 2>&1
"$VENV/bin/python" -m benchmarks.benchmark_poisson --device cuda > "$out/inference.txt" 2>&1
"$VENV/bin/python" -m benchmarks.count_launches > "$out/launches.txt" 2>&1
"$VENV/bin/python" -m benchmarks.benchmark_training --cuda-graph > "$out/training.txt" 2>&1
echo "$out"
