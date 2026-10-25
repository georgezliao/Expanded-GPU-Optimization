"""Interchangeable implementations of ``model -> (u, Laplacian(u))``.

Every backend takes a dense MLP and ``[batch, dimensions]`` coordinates and returns the
model output and its input Laplacian, both shaped ``[batch, 1]``. The eager backend is the
numerical oracle; every other backend must match it before it is benchmarked.
"""

from collections.abc import Callable

from torch import Tensor, nn

from fused_hoad._extension import extension_available
from fused_hoad.backends.eager import eager_laplacian
from fused_hoad.backends.forward_laplacian import forward_laplacian
from fused_hoad.backends.func import func_laplacian
from fused_hoad.backends.graphs import compiled, cuda_graphed

Backend = Callable[[nn.Module, Tensor], tuple[Tensor, Tensor]]

# CUDA Graph variants are inference-only. Compiled and graph variants fall back to their
# base backend off CUDA.
BACKENDS: dict[str, Backend] = {
    "eager": eager_laplacian,
    "eager_cudagraph": cuda_graphed(eager_laplacian),
    "func": func_laplacian,
    "forward_laplacian": forward_laplacian,
    "forward_laplacian_cudagraph": cuda_graphed(forward_laplacian),
    "forward_laplacian_compiled": compiled(forward_laplacian, mode="default"),
    "forward_laplacian_compiled_cudagraph": compiled(forward_laplacian, mode="reduce-overhead"),
}

TRAINABLE_BACKENDS = ("eager", "func", "forward_laplacian")

# Backends hard-wired to the Poisson contract (2 -> 64x4 -> 1 tanh, FP32, CUDA). They are
# registered only when the CUDA extension is built.
FIXED_CONTRACT_BACKENDS: tuple[str, ...] = ()
if extension_available():
    from functools import partial

    from fused_hoad.backends.fused import fused_laplacian

    BACKENDS["fused_cuda_v0"] = partial(fused_laplacian, version=0)
    BACKENDS["fused_cuda_v1"] = partial(fused_laplacian, version=1)
    BACKENDS["fused_cuda"] = fused_laplacian
    BACKENDS["fused_cuda_bwd_v1"] = partial(fused_laplacian, backward="cuda_v1")
    BACKENDS["fused_cuda_bwd_v3"] = partial(fused_laplacian, backward="cuda_v3")
    BACKENDS["fused_cuda_recompute"] = partial(fused_laplacian, backward="recompute")
    FIXED_CONTRACT_BACKENDS = (
        "fused_cuda_v0",
        "fused_cuda_v1",
        "fused_cuda",
        "fused_cuda_bwd_v1",
        "fused_cuda_bwd_v3",
        "fused_cuda_recompute",
    )

REFERENCE_BACKEND = "eager"


def get_backend(name: str) -> Backend:
    """Return the backend registered under ``name``."""
    try:
        return BACKENDS[name]
    except KeyError:
        available = ", ".join(sorted(BACKENDS))
        raise KeyError(f"unknown backend {name!r}; available: {available}") from None


__all__ = [
    "BACKENDS",
    "FIXED_CONTRACT_BACKENDS",
    "REFERENCE_BACKEND",
    "TRAINABLE_BACKENDS",
    "Backend",
    "compiled",
    "cuda_graphed",
    "eager_laplacian",
    "forward_laplacian",
    "func_laplacian",
    "get_backend",
]
