"""Launch the fused backward kernel once at a fixed batch; a target for Nsight Compute."""

import argparse

import torch

from fused_hoad import DEFAULT_POISSON_MLP, build_mlp, pack_poisson_parameters
from fused_hoad._extension import fused_forward_laplacian_backward


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=65_536)
    parser.add_argument("--versions", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    args = parser.parse_args()

    torch.manual_seed(0)
    parameters = pack_poisson_parameters(build_mlp(DEFAULT_POISSON_MLP).cuda())
    coordinates = torch.empty(args.batch_size, 2, device="cuda").uniform_(-1.0, 1.0)
    grad = torch.randn(args.batch_size, 1, device="cuda")
    for version in args.versions:
        fused_forward_laplacian_backward(coordinates, parameters, grad, grad, version)
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
