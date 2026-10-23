"""Launch-overhead removal: CUDA Graph capture and ``torch.compile`` wrappers.

These are inference-only: replays reuse static output buffers and do not record autograd
history for the weights. Each wrapper falls back to the plain backend off CUDA.
"""

from collections.abc import Callable

import torch
from torch import Tensor, nn

Backend = Callable[[nn.Module, Tensor], tuple[Tensor, Tensor]]


def cuda_graphed(backend: Backend) -> Backend:
    """Capture ``backend`` once per (model, input shape) and replay it afterwards.

    Same pattern as pinns-torch: warm up on a side stream, capture with a static input
    buffer, then copy new coordinates into that buffer before each replay.
    """
    state: dict[str, object] = {}

    def run(model: nn.Module, coordinates: Tensor) -> tuple[Tensor, Tensor]:
        if coordinates.device.type != "cuda":
            return backend(model, coordinates)

        key = (id(model), coordinates.shape, coordinates.dtype, coordinates.device)
        if state.get("key") != key:
            static_input = coordinates.clone()
            side_stream = torch.cuda.Stream()
            side_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side_stream):
                for _ in range(3):
                    backend(model, static_input)
            torch.cuda.current_stream().wait_stream(side_stream)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_output = backend(model, static_input)
            state.update(key=key, graph=graph, input=static_input, output=static_output)

        state["input"].copy_(coordinates)
        state["graph"].replay()
        return state["output"]

    return run


def compiled(backend: Backend, *, mode: str) -> Backend:
    """Compile ``backend`` with Inductor; ``mode="reduce-overhead"`` adds CUDA Graphs."""
    compiled_backend = torch.compile(backend, mode=mode, dynamic=False)

    def run(model: nn.Module, coordinates: Tensor) -> tuple[Tensor, Tensor]:
        if coordinates.device.type != "cuda":
            return backend(model, coordinates)
        if mode == "reduce-overhead":
            torch.compiler.cudagraph_mark_step_begin()
        return compiled_backend(model, coordinates)

    return run
