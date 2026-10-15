"""Fused CUDA primitives for higher-order automatic differentiation."""

from fused_hoad.autodiff import diagonal_hessian, gradient, laplacian
from fused_hoad.models import build_mlp
from fused_hoad.problems.poisson import (
    DEFAULT_POISSON_MLP,
    evaluate_poisson_residual,
    manufactured_solution,
    manufactured_source,
    poisson_residual,
)
from fused_hoad.reference import mlp_second_derivative
from fused_hoad.residuals import evaluate_residual
from fused_hoad.specs import MLPConfig
from fused_hoad.validation import ErrorMetrics, error_metrics

__all__ = [
    "ErrorMetrics",
    "MLPConfig",
    "DEFAULT_POISSON_MLP",
    "build_mlp",
    "diagonal_hessian",
    "error_metrics",
    "evaluate_poisson_residual",
    "evaluate_residual",
    "gradient",
    "laplacian",
    "manufactured_solution",
    "manufactured_source",
    "mlp_second_derivative",
    "poisson_residual",
]
__version__ = "0.1.0"

