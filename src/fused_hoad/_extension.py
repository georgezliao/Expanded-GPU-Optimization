"""Lazy access to the compiled CUDA extension."""

from types import ModuleType

from torch import Tensor


def load_extension() -> ModuleType:
    """Return the compiled extension or raise an actionable error."""
    try:
        from fused_hoad import _C
    except ImportError as exc:
        raise RuntimeError(
            "The fused_hoad CUDA extension is not built. Install the project on a "
            "CUDA-capable system with `pip install -e . --no-build-isolation`."
        ) from exc
    return _C


def extension_available() -> bool:
    """Return whether the compiled extension can be imported."""
    try:
        load_extension()
    except RuntimeError:
        return False
    return True


def fused_forward_laplacian(
    coordinates: Tensor, parameters: Tensor, version: int = 2
) -> tuple[Tensor, Tensor]:
    """Evaluate the fixed MLP and its input Laplacian with the CUDA extension.

    ``version=0``: naive one thread per point. ``1``: 32-point shared-memory tiles.
    ``2``: 64-point tiles with two points per lane. ``3``: 32-point tiles with TF32
    tensor-core GEMMs (lower precision; sm_80+).
    """
    output, laplacian = load_extension().fused_forward_laplacian(coordinates, parameters, version)
    return output, laplacian


def fused_forward_laplacian_backward(
    coordinates: Tensor,
    parameters: Tensor,
    grad_output: Tensor,
    grad_laplacian: Tensor,
    version: int = 4,
) -> tuple[Tensor, Tensor]:
    """Return ``(grad_parameters, grad_coordinates)`` for the fused forward Laplacian.

    ``version=1``: 32-point tiles, 1 block/SM. ``2``: 16-point tiles, 2 blocks/SM, one
    private dW row per block. ``3``: as 2, but blocks on an SM share a row via atomicAdd.
    ``4``: as 2, but layer 0's constant J/L jet components are not stored in scratch.
    ``5``: as 4, with 8 inputs per thread in the hbar GEMM.
    """
    grad_parameters, grad_coordinates = load_extension().fused_forward_laplacian_backward(
        coordinates,
        parameters,
        grad_output.reshape(-1).contiguous(),
        grad_laplacian.reshape(-1).contiguous(),
        version,
    )
    return grad_parameters, grad_coordinates
