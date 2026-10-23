"""Benchmark every registered backend on the 2D Poisson model contract.

Each backend is checked against the eager oracle at every batch size before it is timed.
"""

import argparse
import time
from dataclasses import dataclass

import torch

from fused_hoad import BACKENDS, DEFAULT_POISSON_MLP, build_mlp, error_metrics, get_backend
from fused_hoad.backends import REFERENCE_BACKEND

_TOLERANCE = {torch.float32: 1e-3, torch.float64: 1e-8}


@dataclass(frozen=True, slots=True)
class Result:
    backend: str
    batch_size: int
    milliseconds: float
    points_per_second: float
    peak_memory_mib: float | None
    max_abs_error: float


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_batch(
    backend: str,
    batch_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    warmup: int,
    iterations: int,
) -> Result:
    """Check one backend against the oracle, then measure it at one batch size."""
    torch.manual_seed(0)
    model = build_mlp(DEFAULT_POISSON_MLP).to(device=device, dtype=dtype)
    model.requires_grad_(False).eval()
    coordinates = torch.empty(batch_size, 2, device=device, dtype=dtype).uniform_(-1.0, 1.0)
    run = get_backend(backend)

    _, expected = get_backend(REFERENCE_BACKEND)(model, coordinates)
    _, actual = run(model, coordinates)
    metrics = error_metrics(actual.detach(), expected.detach())
    if metrics.max_absolute > _TOLERANCE[dtype]:
        raise SystemExit(
            f"{backend} disagrees with {REFERENCE_BACKEND} at batch {batch_size}: "
            f"max abs error {metrics.max_absolute:.3e}"
        )

    for _ in range(warmup):
        run(model, coordinates)
    _synchronize(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        # Earlier backends' CUDA-graph pools stay allocated, so report the peak above this.
        baseline_bytes = torch.cuda.memory_allocated(device)

    start = time.perf_counter()
    for _ in range(iterations):
        run(model, coordinates)
    _synchronize(device)
    elapsed = time.perf_counter() - start

    milliseconds = elapsed * 1_000.0 / iterations
    peak_memory_mib = None
    if device.type == "cuda":
        peak_memory_mib = (torch.cuda.max_memory_allocated(device) - baseline_bytes) / 1024**2

    return Result(
        backend=backend,
        batch_size=batch_size,
        milliseconds=milliseconds,
        points_per_second=batch_size / (milliseconds / 1_000.0),
        peak_memory_mib=peak_memory_mib,
        max_abs_error=metrics.max_absolute,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", nargs="+", choices=sorted(BACKENDS), default=list(BACKENDS))
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1_024, 4_096, 16_384, 65_536],
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    if any(size < 1 for size in args.batch_sizes):
        parser.error("batch sizes must be positive")
    if args.warmup < 0 or args.iterations < 1:
        parser.error("warmup must be non-negative and iterations must be positive")
    return args


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available")
    dtype = getattr(torch, args.dtype)

    print(f"device={device} dtype={args.dtype} model=2x{DEFAULT_POISSON_MLP.hidden_dim}x4x1")
    print(
        f"{'backend':>38} {'batch':>10} {'ms':>12} {'points/s':>16} "
        f"{'peak MiB':>12} {'max abs err':>12}"
    )
    for backend in args.backends:
        for batch_size in args.batch_sizes:
            result = benchmark_batch(
                backend,
                batch_size,
                device=device,
                dtype=dtype,
                warmup=args.warmup,
                iterations=args.iterations,
            )
            memory = "n/a" if result.peak_memory_mib is None else f"{result.peak_memory_mib:.1f}"
            print(
                f"{result.backend:>38} {result.batch_size:>10,d} {result.milliseconds:>12.3f} "
                f"{result.points_per_second:>16,.0f} {memory:>12} {result.max_abs_error:>12.2e}"
            )


if __name__ == "__main__":
    main()
