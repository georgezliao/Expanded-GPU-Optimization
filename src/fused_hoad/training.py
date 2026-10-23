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
