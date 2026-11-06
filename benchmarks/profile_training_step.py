"""Break one training step into its top GPU kernels and host-side launch count."""

import argparse
import statistics
import time

import torch
from torch.profiler import ProfilerActivity, profile

from benchmarks.environment import describe
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

    forward_ms: list[float] = []
    backward_ms: list[float] = []
    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        start = time.perf_counter()
        _, lap = run(model, coordinates)
        loss = (lap - source).pow(2).mean()
        torch.cuda.synchronize()
        middle = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        forward_ms.append((middle - start) * 1e3)
        backward_ms.append((time.perf_counter() - middle) * 1e3)
        optimizer.step()
    print(
        f"synced split, median of 20: forward+loss {statistics.median(forward_ms):.3f} ms, "
        f"backward {statistics.median(backward_ms):.3f} ms"
    )

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        step()
        torch.cuda.synchronize()

    kernels = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    total = sum(e.device_time for e in kernels)
    print(describe())
    print(f"{args.backend} batch={args.batch_size:,} kernels={len(kernels)} gpu_us={total:.0f}")
    by_name: dict[str, float] = {}
    for event in kernels:
        by_name[event.name[:70]] = by_name.get(event.name[:70], 0.0) + event.device_time
    for name, micros in sorted(by_name.items(), key=lambda item: -item[1])[:6]:
        print(f"{micros:>10.0f} us  {name}")


if __name__ == "__main__":
    main()
