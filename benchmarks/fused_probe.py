"""Launch each fused kernel version once at a fixed batch; a target for Nsight Compute."""

import argparse

import torch

from fused_hoad import (
    DEFAULT_POISSON_MLP,
    build_mlp,
    fused_forward_laplacian,
    pack_poisson_parameters,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=65_536)
    parser.add_argument("--versions", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    torch.manual_seed(0)
    parameters = pack_poisson_parameters(build_mlp(DEFAULT_POISSON_MLP).cuda())
    coordinates = torch.empty(args.batch_size, 2, device="cuda").uniform_(-1.0, 1.0)
    for version in args.versions:
        fused_forward_laplacian(coordinates, parameters, version)
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
