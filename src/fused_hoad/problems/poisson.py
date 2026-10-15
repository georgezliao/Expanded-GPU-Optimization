"""Two-dimensional Poisson benchmark with an analytical reference solution."""

from collections.abc import Callable
from math import pi

import torch
from torch import Tensor, nn

from fused_hoad.autodiff import laplacian
from fused_hoad.residuals import evaluate_residual
from fused_hoad.specs import MLPConfig

Source = Tensor | Callable[[Tensor], Tensor]

DEFAULT_POISSON_MLP = MLPConfig(
    input_dim=2,
    hidden_dim=64,
    hidden_layers=4,
    output_dim=1,
    activation="tanh",
)


def _validate_coordinates(coordinates: Tensor) -> None:
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("Poisson coordinates must have shape [batch, 2]")


def manufactured_solution(coordinates: Tensor) -> Tensor:
    """Return ``sin(pi*x) sin(pi*y)`` on the square ``[-1, 1]^2``."""
    _validate_coordinates(coordinates)
    x = coordinates[:, 0:1]
    y = coordinates[:, 1:2]
    return torch.sin(pi * x) * torch.sin(pi * y)


def manufactured_source(coordinates: Tensor) -> Tensor:
    """Return the source ``f = Laplacian(u)`` for the manufactured solution."""
    return -2.0 * pi**2 * manufactured_solution(coordinates)


def poisson_residual(
    output: Tensor,
    coordinates: Tensor,
    source: Source = manufactured_source,
) -> Tensor:
    """Compute ``Laplacian(output) - source`` for the 2D Poisson equation."""
    _validate_coordinates(coordinates)
    if output.ndim != 2 or output.shape != (coordinates.shape[0], 1):
        raise ValueError("Poisson output must have shape [batch, 1]")

    source_values = source(coordinates) if callable(source) else source
    if source_values.shape != output.shape:
        raise ValueError("Poisson source must have shape [batch, 1]")
    return laplacian(output, coordinates) - source_values


def evaluate_poisson_residual(
    model: nn.Module,
    coordinates: Tensor,
    source: Source = manufactured_source,
) -> tuple[Tensor, Tensor]:
    """Evaluate a model and its Poisson residual at the supplied coordinates."""

    def operator(output: Tensor, x: Tensor) -> Tensor:
        return poisson_residual(output, x, source)

    return evaluate_residual(model, coordinates, operator)

