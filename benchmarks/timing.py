"""Shared timing loop: wall-clock mean plus per-iteration CUDA-event median and p90."""

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class Timing:
    mean_ms: float  # Python-to-Python wall time over the whole loop, divided by iterations
    median_ms: float  # per-iteration CUDA-event time (wall clock per iteration on CPU)
    p90_ms: float


def time_calls(fn: Callable[[], object], iterations: int, device: torch.device) -> Timing:
    """Run ``fn`` ``iterations`` times after the caller's warmup and summarize the timings.

    On CUDA, each iteration is bracketed by events on the current stream, so the per-iteration
    numbers are GPU time from the first to the last enqueued op, including gaps where the GPU
    waited for the host. The mean is wall time and also includes the final synchronize.
    """
    samples: list[float] = []
    if device.type == "cuda":
        pairs = [
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(iterations)
        ]
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        for begin, end in pairs:
            begin.record()
            fn()
            end.record()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        samples = [begin.elapsed_time(end) for begin, end in pairs]
    else:
        start = time.perf_counter()
        for _ in range(iterations):
            tick = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - tick) * 1_000.0)
        elapsed = time.perf_counter() - start

    samples.sort()
    p90 = samples[min(len(samples) - 1, int(0.9 * len(samples)))]
    return Timing(
        mean_ms=elapsed * 1_000.0 / iterations,
        median_ms=statistics.median(samples),
        p90_ms=p90,
    )
