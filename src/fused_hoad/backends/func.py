"""``torch.func`` backend: per-sample Hessian via ``vmap(jacfwd(jacrev))``."""

import torch
from torch import Tensor, nn
from torch.func import functional_call, jacfwd, jacrev, vmap


def func_laplacian(model: nn.Module, coordinates: Tensor) -> tuple[Tensor, Tensor]:
    """Evaluate ``model`` and the trace of its per-sample input Hessian."""
    if coordinates.ndim != 2:
        raise ValueError("coordinates must have shape [batch, dimensions]")
    state = {**dict(model.named_parameters()), **dict(model.named_buffers())}

    def scalar_field(point: Tensor) -> Tensor:
        return functional_call(model, state, (point.unsqueeze(0),)).squeeze()

    hessians = vmap(jacfwd(jacrev(scalar_field)))(coordinates)
    laplacian = torch.diagonal(hessians, dim1=1, dim2=2).sum(dim=1, keepdim=True)
    return model(coordinates), laplacian
