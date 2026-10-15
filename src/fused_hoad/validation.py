"""Numerical comparison primitives for reference and optimized backends."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class ErrorMetrics:
    max_absolute: float
    max_relative: float
    mean_absolute: float


def error_metrics(actual: Tensor, expected: Tensor) -> ErrorMetrics:
    """Summarize elementwise numerical error for tensors of equal shape."""
    if actual.shape != expected.shape:
        raise ValueError(f"shape mismatch: {actual.shape} != {expected.shape}")
    if actual.numel() == 0:
        raise ValueError("cannot compare empty tensors")

    absolute = (actual - expected).abs()
    epsilon = torch.full_like(expected, torch.finfo(expected.dtype).eps)
    scale = torch.maximum(expected.abs(), epsilon)
    relative = absolute / scale
    return ErrorMetrics(
        max_absolute=absolute.max().item(),
        max_relative=relative.max().item(),
        mean_absolute=absolute.mean().item(),
    )

