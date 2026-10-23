"""Eager autograd backend: the correctness oracle."""

from torch import Tensor, nn

from fused_hoad.reference import mlp_second_derivative


def eager_laplacian(model: nn.Module, coordinates: Tensor) -> tuple[Tensor, Tensor]:
    """Evaluate ``model`` and its input Laplacian with nested ``autograd.grad`` calls."""
    return mlp_second_derivative(model, coordinates)
