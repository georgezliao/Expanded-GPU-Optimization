#!/usr/bin/env bash
# Nsight Compute summary of the fused kernels. Run on a GPU node after oscar_job.sh built
# the extension:  srun -p gpu --gres=gpu:1 --mem=16G -t 0:15:00 bash scripts/ncu_fused.sh
set -euo pipefail

VENV=${VENV:-$HOME/hoad.venv}
module load cuda/12.9.0-cinr >/dev/null 2>&1 || true
export PYTHONPATH=src
mkdir -p artifacts/profiles

nvidia-smi --query-gpu=name --format=csv,noheader
ncu --section SpeedOfLight --section Occupancy --section LaunchStats \
  --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis \
  --export artifacts/profiles/fused --force-overwrite \
  "$VENV/bin/python" -m benchmarks.fused_probe "$@" >/dev/null
ncu --import artifacts/profiles/fused.ncu-rep --page details 2>&1 \
  | grep -E "fused_forward|ERR|Duration|Throughput|Registers Per|Achieved Occupancy|Theoretical Occupancy|Block Limit|Executed Ipc A|Local|Dynamic Shared" \
  || true
