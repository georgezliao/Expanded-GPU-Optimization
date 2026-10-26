"""Split a small-batch fused forward call into host-side pieces and GPU kernel time."""

import argparse
import time

import torch
from torch.profiler import ProfilerActivity, profile

from benchmarks.environment import describe
from fused_hoad import DEFAULT_POISSON_MLP, build_mlp, fused_forward_laplacian, get_backend
from fused_hoad.backends.fused import _packed_parameters
from fused_hoad.parameters import pack_poisson_parameters

_legacy_cache: dict[str, object] = {}


def legacy_packed_parameters(model: torch.nn.Module) -> torch.Tensor:
    """The pre-2026-10-26 lookup, kept here so old and new can be timed in one process."""
    if torch.is_grad_enabled() and any(p.requires_grad for p in model.parameters()):
        return pack_poisson_parameters(model, differentiable=True)
    key = (id(model), tuple(p._version for p in model.parameters()))
    if _legacy_cache.get("key") != key:
        _legacy_cache.update(key=key, packed=pack_poisson_parameters(model))
    return _legacy_cache["packed"]


def host_us(fn, calls: int) -> float:
    """Mean host time per call in µs, with no synchronize inside the loop."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(calls):
        fn()
    elapsed = time.perf_counter() - start
    torch.cuda.synchronize()
    return elapsed * 1e6 / calls


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1_024, 4_096])
    parser.add_argument("--calls", type=int, default=2_000)
    args = parser.parse_args()

    print(describe())
    print(
        f"{'batch':>7} {'legacy pack us':>15} {'pack us':>8} {'ext call us':>12} "
        f"{'backend us':>11} {'synced us':>10} {'kernel us':>10}"
    )
    torch.manual_seed(0)
    model = build_mlp(DEFAULT_POISSON_MLP).cuda().requires_grad_(False)
    run = get_backend("fused_cuda")
    with torch.no_grad():
        for batch in args.batch_sizes:
            x = torch.empty(batch, 2, device="cuda").uniform_(-1.0, 1.0)
            packed = _packed_parameters(model)
            for _ in range(50):
                run(model, x)

            legacy = host_us(lambda: legacy_packed_parameters(model), args.calls)
            pack = host_us(lambda: _packed_parameters(model), args.calls)
            ext = host_us(lambda: fused_forward_laplacian(x, packed), args.calls)
            backend = host_us(lambda: run(model, x), args.calls)

            def synced() -> None:
                run(model, x)
                torch.cuda.synchronize()

            synced_us = host_us(synced, args.calls // 4)
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for _ in range(100):
                    run(model, x)
                torch.cuda.synchronize()
            kernels = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
            kernel_us = sum(e.device_time for e in kernels) / 100
            print(
                f"{batch:>7,d} {legacy:>15.1f} {pack:>8.1f} {ext:>12.1f} {backend:>11.1f} "
                f"{synced_us:>10.1f} {kernel_us:>10.1f}"
            )


if __name__ == "__main__":
    main()
