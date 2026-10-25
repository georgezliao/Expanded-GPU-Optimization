"""Benchmark one PINN training step (forward + backward + Adam) per trainable backend.

``--cuda-graph`` also times each backend with the whole step captured as one CUDA Graph.
"""

import argparse

import torch

from benchmarks.environment import describe
from benchmarks.timing import Timing, time_calls
from fused_hoad import BACKENDS, DEFAULT_POISSON_MLP, build_mlp, get_backend, manufactured_source
from fused_hoad.training import cuda_graph_step

_TRAINABLE = (
    "eager",
    "forward_laplacian",
    "forward_laplacian_compiled",
    "forward_laplacian_compiled_cudagraph",
    "fused_cuda_recompute",
    "fused_cuda_bwd_v1",
    "fused_cuda_bwd_v3",
    "fused_cuda",
)
# torch.compile manages its own graphs (reduce-overhead = CUDA Graph trees over forward and
# backward, optimizer outside); capturing it inside an outer graph is unsupported.
_GRAPHABLE = tuple(name for name in _TRAINABLE if "compiled" not in name)


def time_step(name: str, batch_size: int, warmup: int, iterations: int, graph: bool) -> Timing:
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
    return time_calls(step, iterations, torch.device("cuda"))


def main() -> None:
    available = [name for name in _TRAINABLE if name in BACKENDS]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", nargs="+", choices=available, default=available)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[4_096, 16_384, 65_536])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--cuda-graph", action="store_true")
    args = parser.parse_args()

    print(describe())
    print(
        f"{'backend':>38} {'graph':>6} {'batch':>10} {'mean ms':>10} {'median ms':>10} "
        f"{'p90 ms':>10}"
    )
    print("(mean: wall clock over the loop; median/p90: per-step CUDA events)")
    for name in args.backends:
        modes = (False, True) if args.cuda_graph and name in _GRAPHABLE else (False,)
        for graph in modes:
            for batch_size in args.batch_sizes:
                timing = time_step(name, batch_size, args.warmup, args.iterations, graph)
                print(
                    f"{name:>38} {'yes' if graph else 'no':>6} {batch_size:>10,d} "
                    f"{timing.mean_ms:>10.3f} {timing.median_ms:>10.3f} {timing.p90_ms:>10.3f}"
                )


if __name__ == "__main__":
    main()
