"""Training-step helpers, including whole-step CUDA Graph capture."""

from collections.abc import Callable

import torch
from torch import Tensor


def cuda_graph_step(
    loss_fn: Callable[[], Tensor], optimizer: torch.optim.Optimizer, *, warmup: int = 3
) -> Callable[[], Tensor]:
    """Capture ``loss_fn() -> backward -> optimizer.step()`` as one CUDA Graph.

    Follows PyTorch's whole-network capture recipe: warm up on a side stream, drop grads
    so backward allocates them inside the graph's pool, then capture. The optimizer must be
    built with ``capturable=True``. Inputs that ``loss_fn`` closes over are static: copy new
    data into them in place between replays. Returns a function that replays the step and
    returns the (static) loss tensor.
    """
    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        for _ in range(warmup):
            optimizer.zero_grad(set_to_none=True)
            loss_fn().backward()
            optimizer.step()
    torch.cuda.current_stream().wait_stream(side_stream)

    graph = torch.cuda.CUDAGraph()
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.graph(graph):
        static_loss = loss_fn()
        static_loss.backward()
        optimizer.step()

    def replay() -> Tensor:
        graph.replay()
        return static_loss

    return replay


def boundary_points(count: int, *, device, dtype=torch.float32, generator=None) -> Tensor:
    """Sample ``count`` points uniformly on the boundary of ``[-1, 1]^2``."""
    t = torch.rand(count, device=device, dtype=dtype, generator=generator) * 2 - 1
    side = torch.randint(0, 4, (count,), device=device, generator=generator)
    fixed = torch.where(side % 2 == 0, -1.0, 1.0).to(dtype)  # sides: x=-1, x=1, y=-1, y=1
    x_is_fixed = side < 2
    x = torch.where(x_is_fixed, fixed, t)
    y = torch.where(x_is_fixed, t, fixed)
    return torch.stack((x, y), dim=1)


def relative_l2_error(model: torch.nn.Module, *, resolution: int = 101) -> float:
    """Relative L2 error of ``model`` against ``sin(pi x) sin(pi y)`` on a uniform grid."""
    from fused_hoad.problems.poisson import manufactured_solution

    parameter = next(model.parameters())
    axis = torch.linspace(-1.0, 1.0, resolution, device=parameter.device, dtype=parameter.dtype)
    grid = torch.cartesian_prod(axis, axis)
    with torch.no_grad():
        error = model(grid) - manufactured_solution(grid)
        return (error.norm() / manufactured_solution(grid).norm()).item()


def poisson_loss_fn(
    backend: Callable,
    model: torch.nn.Module,
    interior: Tensor,
    boundary: Tensor,
    *,
    boundary_weight: float = 100.0,
) -> Callable[[], Tensor]:
    """Return a closure for the Poisson PINN loss: interior residual + weighted u=0 on boundary.

    ``backend`` computes ``(u, Laplacian u)`` on the interior points; the boundary term uses a
    plain forward pass so every backend shares it.
    """
    from fused_hoad.problems.poisson import manufactured_source

    source = manufactured_source(interior)

    def loss_fn() -> Tensor:
        _, lap = backend(model, interior)
        residual = (lap - source).pow(2).mean()
        return residual + boundary_weight * model(boundary).pow(2).mean()

    return loss_fn
