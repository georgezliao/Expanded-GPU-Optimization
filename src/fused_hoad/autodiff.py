"""Small, composable higher-order automatic-differentiation operators."""

import torch
from torch import Tensor


def differentiable_coordinates(coordinates: Tensor) -> Tensor:
    """Validate coordinates and return an independent differentiable leaf tensor."""
    if coordinates.ndim != 2:
        raise ValueError("coordinates must have shape [batch, dimensions]")
    if not coordinates.is_floating_point():
        raise TypeError("coordinates must use a floating-point dtype")
    return coordinates.detach().clone().requires_grad_(True)


def gradient(output: Tensor, coordinates: Tensor, *, create_graph: bool = True) -> Tensor:
    """Compute the per-sample gradient of a scalar field."""
    if output.ndim != 2 or output.shape != (coordinates.shape[0], 1):
        raise ValueError("output must have shape [batch, 1]")
    return torch.autograd.grad(output.sum(), coordinates, create_graph=create_graph)[0]


def diagonal_hessian(output: Tensor, coordinates: Tensor) -> Tensor:
    """Compute only the Hessian diagonal, avoiding unused mixed partials."""
    first_derivative = gradient(output, coordinates)
    terms: list[Tensor] = []
    for dimension in range(coordinates.shape[1]):
        derivative = torch.autograd.grad(
            first_derivative[:, dimension].sum(),
            coordinates,
            create_graph=True,
            retain_graph=True,
        )[0]
        terms.append(derivative[:, dimension])
    return torch.stack(terms, dim=1)


def laplacian(output: Tensor, coordinates: Tensor) -> Tensor:
    """Compute the Laplacian of a batched scalar field."""
    return diagonal_hessian(output, coordinates).sum(dim=1, keepdim=True)

