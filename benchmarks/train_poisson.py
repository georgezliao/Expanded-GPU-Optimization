"""Train the 2D Poisson PINN end to end per backend; report wall time and relative L2 error."""

import argparse
import hashlib
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


def train(
    name: str, args: argparse.Namespace, graph: bool
) -> tuple[float, float, float, str, float | None, int | None]:
    """Train one backend. Returns seconds, final loss, final rel L2, weights hash, and the
    training seconds and step at which rel L2 first dropped below ``args.target``.

    Rel L2 is evaluated every ``args.eval_every`` steps with the clock paused, so the
    time-to-target excludes evaluation; the step resolution is ``eval_every``.
    """
    torch.manual_seed(args.seed)
    model = build_mlp(DEFAULT_POISSON_MLP).cuda()
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    interior = torch.rand(args.interior, 2, device="cuda", generator=generator) * 2 - 1
    boundary = boundary_points(args.boundary, device="cuda", generator=generator)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, capturable=graph)
    loss_fn = poisson_loss_fn(get_backend(name), model, interior, boundary)

    def eager_step() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn()
        loss.backward()
        optimizer.step()
        return loss

    torch.cuda.synchronize()
    seconds = 0.0
    start = time.perf_counter()
    done = 0
    if graph:
        step = cuda_graph_step(loss_fn, optimizer)  # its 3 warmup steps count as training
        done = 3
    else:
        step = eager_step
    hit_seconds: float | None = None
    hit_step: int | None = None
    while done < args.steps:
        for _ in range(min(args.eval_every, args.steps - done)):
            loss = step()
        done = min(done + args.eval_every, args.steps)
        torch.cuda.synchronize()
        seconds += time.perf_counter() - start
        if hit_seconds is None and relative_l2_error(model) < args.target:
            hit_seconds, hit_step = seconds, done
        start = time.perf_counter()
    digest = hashlib.sha256()
    for parameter in model.parameters():
        digest.update(parameter.detach().cpu().numpy().tobytes())
    error = relative_l2_error(model)
    return seconds, loss.item(), error, digest.hexdigest()[:12], hit_seconds, hit_step


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
    parser.add_argument("--target", type=float, default=5e-3, help="rel L2 for time-to-target")
    parser.add_argument("--eval-every", type=int, default=250)
    args = parser.parse_args()

    print(describe())
    print(
        f"steps={args.steps} interior={args.interior} boundary={args.boundary} lr={args.lr} "
        f"seed={args.seed} loss=mean(res^2)+100*mean(u_b^2) (time includes compile/capture)"
    )
    print(
        f"{'backend':>30} {'graph':>6} {'seconds':>9} {'final loss':>14} {'rel L2':>12} "
        f"{'weights sha256':>14} {'s to target':>12} {'step':>6}"
    )
    print(
        f"(seconds exclude rel-L2 evaluation every {args.eval_every} steps; target {args.target})"
    )
    for name in args.backends:
        modes = (False, True) if args.cuda_graph and "compiled" not in name else (False,)
        for graph in modes:
            seconds, loss, error, digest, hit_seconds, hit_step = train(name, args, graph)
            flag = "yes" if graph else "no"
            hit = "never" if hit_seconds is None else f"{hit_seconds:.2f}"
            print(
                f"{name:>30} {flag:>6} {seconds:>9.2f} {loss:>14.6e} {error:>12.6e} "
                f"{digest:>14} {hit:>12} {hit_step or '-':>6}"
            )


if __name__ == "__main__":
    main()
