"""Count host-side launches and GPU kernels for one call of each backend."""

import argparse

import torch
from torch.profiler import ProfilerActivity, profile

from benchmarks.environment import describe
from fused_hoad import BACKENDS, DEFAULT_POISSON_MLP, build_mlp, get_backend

_LAUNCH_CALLS = ("cudaLaunchKernel", "cuLaunchKernel", "cudaGraphLaunch", "cuGraphLaunch")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", nargs="+", choices=sorted(BACKENDS), default=list(BACKENDS))
    parser.add_argument("--batch-size", type=int, default=16_384)
    args = parser.parse_args()

    torch.manual_seed(0)
    model = build_mlp(DEFAULT_POISSON_MLP).cuda().requires_grad_(False).eval()
    coordinates = torch.empty(args.batch_size, 2, device="cuda").uniform_(-1.0, 1.0)

    print(describe())
    print(f"{'backend':>38} {'host launches':>14} {'gpu kernels':>12}")
    for name in args.backends:
        run = get_backend(name)
        for _ in range(5):
            run(model, coordinates)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            run(model, coordinates)
            torch.cuda.synchronize()
        events = prof.events()
        launches = sum(1 for e in events if e.name in _LAUNCH_CALLS)
        kernels = sum(1 for e in events if e.device_type == torch.autograd.DeviceType.CUDA)
        print(f"{name:>38} {launches:>14} {kernels:>12}")


if __name__ == "__main__":
    main()
