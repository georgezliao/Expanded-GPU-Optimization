#!/usr/bin/env bash
# Run tests and the backend benchmark on one OSCAR GPU.
#   Interactive: srun -p gpu --gres=gpu:1 --mem=16G -t 0:20:00 bash scripts/oscar_job.sh [benchmark args]
#   Batch:       sbatch scripts/oscar_job.sh [benchmark args]
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 0:20:00
#SBATCH --mem=16G  # nvcc on torch headers gets OOM-killed at the default limit
#SBATCH -o artifacts/oscar-%j.out
set -euo pipefail

VENV=${VENV:-$HOME/hoad.venv}
module load cuda/12.9.0-cinr >/dev/null 2>&1 || true
export PYTHONPATH=src
export PATH="$VENV/bin:$PATH"  # lets BuildExtension find ninja
mkdir -p artifacts

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"$VENV/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

# Build the extension in place for this node's GPU architecture (ninja makes rebuilds cheap).
"$VENV/bin/python" setup.py -q build_ext --inplace 2>&1 | grep -v -i warning || true
"$VENV/bin/python" -c "from fused_hoad._extension import load_extension; load_extension()"

"$VENV/bin/python" -m pytest -q
"$VENV/bin/python" -m benchmarks.benchmark_poisson --device cuda "$@"
"$VENV/bin/python" -m benchmarks.count_launches

if [[ "${PROFILE:-0}" == 1 ]] && command -v nsys >/dev/null; then
  nsys profile --trace=cuda,nvtx --output=artifacts/profiles/backends --force-overwrite=true \
    "$VENV/bin/python" -m benchmarks.benchmark_poisson --device cuda \
    --batch-sizes 16384 --warmup 5 --iterations 20 >/dev/null
  nsys stats --report cuda_gpu_kern_sum --format csv artifacts/profiles/backends.nsys-rep \
    | head -25 || true  # head closes the pipe early (SIGPIPE) by design
fi
