#!/usr/bin/env bash
set -euo pipefail

mkdir -p artifacts/profiles
nsys profile \
  --trace=cuda,nvtx,osrt \
  --output=artifacts/profiles/reference \
  --force-overwrite=true \
  python -m benchmarks.benchmark_poisson "$@"

