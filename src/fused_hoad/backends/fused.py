"""Fused CUDA backend: the whole forward Laplacian in one kernel launch.

Fixed to the Poisson contract (2 -> 64x4 -> 1 tanh, FP32, CUDA). The forward pass is one
kernel. The backward pass is either one fused CUDA kernel (default) or a reference that
recomputes the jet in PyTorch from the packed parameters and differentiates that.
"""

import torch
from torch import Tensor, nn

from fused_hoad._extension import fused_forward_laplacian, fused_forward_laplacian_backward
from fused_hoad.backends.forward_laplacian import tanh_mlp_forward_laplacian
from fused_hoad.parameters import (
    POISSON_PARAMETER_LAYOUT,
    pack_poisson_parameters,
    unpack_poisson_parameters,
)

_cache: dict[str, object] = {}

# Backward modes: fused CUDA kernels by version, or the PyTorch recompute reference.
_CUDA_BACKWARD_VERSIONS = {"cuda": 2, "cuda_v1": 1, "cuda_v3": 3}
BACKWARD_MODES = (*_CUDA_BACKWARD_VERSIONS, "recompute")


class FusedForwardLaplacian(torch.autograd.Function):
    """Fused-kernel forward; fused-kernel or recompute-and-differentiate backward."""

    @staticmethod
    def forward(ctx, coordinates: Tensor, parameters: Tensor, version: int, backward: str):
        ctx.save_for_backward(coordinates, parameters)
        ctx.backward_mode = backward
        return fused_forward_laplacian(coordinates, parameters, version)

    @staticmethod
    def backward(ctx, grad_output: Tensor, grad_laplacian: Tensor):
        coordinates, parameters = ctx.saved_tensors
        needs_coordinates, needs_parameters = ctx.needs_input_grad[:2]
        if ctx.backward_mode in _CUDA_BACKWARD_VERSIONS:
            grad_parameters, grad_coordinates = fused_forward_laplacian_backward(
                coordinates,
                parameters,
                grad_output,
                grad_laplacian,
                _CUDA_BACKWARD_VERSIONS[ctx.backward_mode],
            )
            return (
                grad_coordinates if needs_coordinates else None,
                grad_parameters if needs_parameters else None,
                None,
                None,
            )
        return (
            *_recompute_backward(
                coordinates,
                parameters,
                grad_output,
                grad_laplacian,
                needs_coordinates,
                needs_parameters,
            ),
            None,
            None,
        )


def _recompute_backward(
    coordinates: Tensor,
    parameters: Tensor,
    grad_output: Tensor,
    grad_laplacian: Tensor,
    needs_coordinates: bool,
    needs_parameters: bool,
) -> tuple[Tensor | None, Tensor | None]:
    """Reference backward: rebuild the jet in PyTorch and differentiate it."""
    with torch.enable_grad():
        x = coordinates.detach().requires_grad_(needs_coordinates)
        flat = parameters.detach().requires_grad_(needs_parameters)
        views = unpack_poisson_parameters(flat)
        linears = [
            (views[f"{spec.name}.weight"], views[f"{spec.name}.bias"])
            for spec in POISSON_PARAMETER_LAYOUT
        ]
        outputs = tanh_mlp_forward_laplacian(linears, x)
        wanted = ((x, needs_coordinates), (flat, needs_parameters))
        inputs = [tensor for tensor, needed in wanted if needed]
        grads = iter(
            torch.autograd.grad(outputs, inputs, (grad_output, grad_laplacian), allow_unused=True)
        )
    grad_coordinates = next(grads) if needs_coordinates else None
    grad_parameters = next(grads) if needs_parameters else None
    return grad_coordinates, grad_parameters


def _packed_parameters(model: nn.Module) -> Tensor:
    """Differentiable pack when training; otherwise pack once per in-place weight update."""
    if torch.is_grad_enabled() and any(p.requires_grad for p in model.parameters()):
        return pack_poisson_parameters(model, differentiable=True)
    key = (id(model), tuple(p._version for p in model.parameters()))
    if _cache.get("key") != key:
        _cache.update(key=key, packed=pack_poisson_parameters(model))
    return _cache["packed"]


def fused_laplacian(
    model: nn.Module, coordinates: Tensor, version: int = 2, backward: str = "cuda"
) -> tuple[Tensor, Tensor]:
    """Evaluate the fixed Poisson MLP and its input Laplacian with one CUDA kernel.

    ``backward`` picks the gradient path when autograd is active: ``"cuda"`` (fused v2
    kernel), ``"cuda_v1"`` / ``"cuda_v3"`` (other fused versions) or ``"recompute"``
    (PyTorch reference).
    """
    if backward not in BACKWARD_MODES:
        raise ValueError(f"backward must be one of {BACKWARD_MODES}")
    parameters = _packed_parameters(model)
    if parameters.requires_grad or coordinates.requires_grad:
        return FusedForwardLaplacian.apply(coordinates.contiguous(), parameters, version, backward)
    return fused_forward_laplacian(coordinates.contiguous(), parameters, version)
