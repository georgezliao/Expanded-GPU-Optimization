"""Train the 2D Poisson PINN end to end per backend; report wall time and relative L2 error."""

import argparse
import time

import torch

from benchmarks.environment import describe
from fused_hoad import BACKENDS, DEFAULT_POISSON_MLP, build_mlp, get_backend
from fused_hoad.training import (
    boundary_points,
    cuda_graph_step,
    poisson_loss_fn,
    relative_l2_error,
)


def train(name: str, args: argparse.Namespace, graph: bool) -> tuple[float, float, float]:
    torch.manual_seed(args.seed)
    model = build_mlp(DEFAULT_POISSON_MLP).cuda()
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    interior = torch.rand(args.interior, 2, device="cuda", generator=generator) * 2 - 1
    boundary = boundary_points(args.boundary, device="cuda", generator=generator)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, capturable=graph)
    loss_fn = poisson_loss_fn(get_backend(name), model, interior, boundary)

    torch.cuda.synchronize()
    start = time.perf_counter()
    if graph:
        step = cuda_graph_step(loss_fn, optimizer)  # its 3 warmup steps count as training
        for _ in range(args.steps - 3):
            loss = step()
    else:
        for _ in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn()
            loss.backward()
            optimizer.step()
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    return seconds, loss.item(), relative_l2_error(model)


def main() -> None:
    trainable = (
        "eager",
        "forward_laplacian",
        "forward_laplacian_compiled",
        "fused_cuda",
        "fused_cuda_bwd_v1",
        "fused_cuda_recompute",
    )
    choices = [n for n in trainable if n in BACKENDS]
    default = [n for n in ("eager", "forward_laplacian_compiled", "fused_cuda") if n in BACKENDS]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", nargs="+", choices=choices, default=default)
    parser.add_argument("--steps", type=int, default=5_000)
    parser.add_argument("--interior", type=int, default=16_384)
    parser.add_argument("--boundary", type=int, default=1_024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda-graph", action="store_true")
    args = parser.parse_args()

    print(describe())
    print(
        f"steps={args.steps} interior={args.interior} boundary={args.boundary} lr={args.lr} "
        f"seed={args.seed} loss=mean(res^2)+100*mean(u_b^2) (time includes compile/capture)"
    )
    print(f"{'backend':>30} {'graph':>6} {'seconds':>9} {'final loss':>12} {'rel L2':>10}")
    for name in args.backends:
        modes = (False, True) if args.cuda_graph and "compiled" not in name else (False,)
        for graph in modes:
            seconds, loss, error = train(name, args, graph)
            flag = "yes" if graph else "no"
            print(f"{name:>30} {flag:>6} {seconds:>9.2f} {loss:>12.3e} {error:>10.3e}")


if __name__ == "__main__":
    main()
