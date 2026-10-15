"""Residual composition kept independent from models and derivative mechanics."""

from collections.abc import Callable

from torch import Tensor, nn

from fused_hoad.autodiff import differentiable_coordinates

ResidualOperator = Callable[[Tensor, Tensor], Tensor]


def evaluate_residual(
    model: nn.Module,
    coordinates: Tensor,
    operator: ResidualOperator,
) -> tuple[Tensor, Tensor]:
    """Evaluate a model and apply a residual operator to ``(output, coordinates)``."""
    x = differentiable_coordinates(coordinates)
    output = model(x)
    if output.ndim != 2 or output.shape[0] != x.shape[0]:
        raise ValueError("model output must have shape [batch, outputs]")

    residual = operator(output, x)
    if residual.ndim != 2 or residual.shape[0] != x.shape[0]:
        raise ValueError("residual must have shape [batch, terms]")
    return output, residual

