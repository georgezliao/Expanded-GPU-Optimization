"""Measure forward v3 (TF32) error against an FP64 forward Laplacian."""

import torch

from benchmarks.environment import describe
from fused_hoad import (
    DEFAULT_POISSON_MLP,
    build_mlp,
    forward_laplacian,
    fused_forward_laplacian,
    pack_poisson_parameters,
)


def main() -> None:
    print(describe())
    print(
        f"{'weights':>8} {'version':>7} {'u max abs':>10} {'u rel':>9} "
        f"{'lap max abs':>12} {'lap rel':>9}"
    )
    for scale in (1.0, 3.0):
        torch.manual_seed(0)
        model = build_mlp(DEFAULT_POISSON_MLP).cuda()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.mul_(scale)
        x = torch.empty(65_536, 2, device="cuda").uniform_(-1.0, 1.0)
        with torch.no_grad():
            ref_u, ref_lap = forward_laplacian(model.double(), x.double())
            model.float()
            packed = pack_poisson_parameters(model)
            for version in (2, 3, 4):
                u, lap = fused_forward_laplacian(x, packed, version)
                du = (u.double() - ref_u).abs().max().item()
                dl = (lap.double() - ref_lap).abs().max().item()
                print(
                    f"{scale:>8} {version:>7} {du:>10.2e} {du / ref_u.abs().max().item():>9.2e} "
                    f"{dl:>12.2e} {dl / ref_lap.abs().max().item():>9.2e}"
                )


if __name__ == "__main__":
    main()
