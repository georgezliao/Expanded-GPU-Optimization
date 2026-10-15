"""Problem definitions used to drive reference and fused implementations."""

from fused_hoad.problems.poisson import (
    DEFAULT_POISSON_MLP,
    evaluate_poisson_residual,
    manufactured_solution,
    manufactured_source,
    poisson_residual,
)

__all__ = [
    "DEFAULT_POISSON_MLP",
    "evaluate_poisson_residual",
    "manufactured_solution",
    "manufactured_source",
    "poisson_residual",
]

