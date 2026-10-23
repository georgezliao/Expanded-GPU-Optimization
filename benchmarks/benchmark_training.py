"""Benchmark one PINN training step (forward + backward + Adam) per trainable backend.

``--cuda-graph`` also times each backend with the whole step captured as one CUDA Graph.
"""

import argparse
import time

import torch

from fused_hoad import BACKENDS, DEFAULT_POISSON_MLP, build_mlp, get_backend, manufactured_source
from fused_hoad.training import cuda_graph_step

_TRAINABLE = (
    "eager",
    "forward_laplacian",
    "forward_laplacian_compiled",
    "fused_cuda_recompute",
    "fused_cuda_bwd_v1",
    "fused_cuda",
)
# torch.compile manages its own graphs; capturing it inside an outer graph is unsupported.
_GRAPHABLE = tuple(name for name in _TRAINABLE if name != "forward_laplacian_compiled")


def time_step(name: str, batch_size: int, warmup: int, iterations: int, graph: bool) -> float:
    torch.manual_seed(0)
    model = build_mlp(DEFAULT_POISSON_MLP).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, capturable=graph)
    coordinates = torch.empty(batch_size, 2, device="cuda").uniform_(-1.0, 1.0)
    source = manufactured_source(coordinates)
    run = get_backend(name)

    def loss_fn() -> torch.Tensor:
        _, lap = run(model, coordinates)
        return (lap - source).pow(2).mean()

    if graph:
        step = cuda_graph_step(loss_fn, optimizer)
    else:

        def step() -> None:
            optimizer.zero_grad(set_to_none=True)
            loss_fn().backward()
            optimizer.step()

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        step()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1_000.0 / iterations


def main() -> None:
    available = [name for name in _TRAINABLE if name in BACKENDS]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", nargs="+", choices=available, default=available)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[4_096, 16_384, 65_536])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--cuda-graph", action="store_true")
    args = parser.parse_args()

    print(f"{'backend':>28} {'graph':>6} {'batch':>10} {'step ms':>10}")
    for name in args.backends:
        modes = (False, True) if args.cuda_graph and name in _GRAPHABLE else (False,)
        for graph in modes:
            for batch_size in args.batch_sizes:
                ms = time_step(name, batch_size, args.warmup, args.iterations, graph)
                print(f"{name:>28} {'yes' if graph else 'no':>6} {batch_size:>10,d} {ms:>10.3f}")


if __name__ == "__main__":
    main()
