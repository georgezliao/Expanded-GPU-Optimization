"""Break one training step into its top GPU kernels and host-side launch count."""

import argparse

import torch
from torch.profiler import ProfilerActivity, profile

from fused_hoad import DEFAULT_POISSON_MLP, build_mlp, get_backend, manufactured_source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="fused_cuda")
    parser.add_argument("--batch-size", type=int, default=65_536)
    args = parser.parse_args()

    torch.manual_seed(0)
    model = build_mlp(DEFAULT_POISSON_MLP).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    coordinates = torch.empty(args.batch_size, 2, device="cuda").uniform_(-1.0, 1.0)
    source = manufactured_source(coordinates)
    run = get_backend(args.backend)

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
        _, lap = run(model, coordinates)
        (lap - source).pow(2).mean().backward()
        optimizer.step()

    for _ in range(5):
        step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        step()
        torch.cuda.synchronize()

    kernels = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    total = sum(e.device_time for e in kernels)
    print(f"{args.backend} batch={args.batch_size:,} kernels={len(kernels)} gpu_us={total:.0f}")
    by_name: dict[str, float] = {}
    for event in kernels:
        by_name[event.name[:70]] = by_name.get(event.name[:70], 0.0) + event.device_time
    for name, micros in sorted(by_name.items(), key=lambda item: -item[1])[:6]:
        print(f"{micros:>10.0f} us  {name}")


if __name__ == "__main__":
    main()
