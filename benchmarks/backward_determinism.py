"""Check whether fused backward gradients depend on the persistent grid size.

Runs the default backward with ``FUSED_HOAD_BWD_BLOCKS`` set to several block counts and
compares every gradient bitwise against the run with the natural grid size.
"""

import argparse
import os

import torch

from benchmarks.environment import describe
from fused_hoad import DEFAULT_POISSON_MLP, build_mlp, pack_poisson_parameters
from fused_hoad._extension import fused_forward_laplacian_backward


def gradients(x, parameters, gu, gl, blocks: int | None):
    if blocks is None:
        os.environ.pop("FUSED_HOAD_BWD_BLOCKS", None)
    else:
        os.environ["FUSED_HOAD_BWD_BLOCKS"] = str(blocks)
    return fused_forward_laplacian_backward(x, parameters, gu, gl)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=65_536)
    parser.add_argument("--blocks", type=int, nargs="+", default=[1, 7, 16, 64, 100, 168, 1000])
    args = parser.parse_args()

    print(describe())
    torch.manual_seed(0)
    parameters = pack_poisson_parameters(build_mlp(DEFAULT_POISSON_MLP).cuda())
    x = torch.empty(args.batch_size, 2, device="cuda").uniform_(-1.0, 1.0)
    gu = torch.randn(args.batch_size, 1, device="cuda")
    gl = torch.randn(args.batch_size, 1, device="cuda")

    ref_p, ref_x = gradients(x, parameters, gu, gl, None)
    again_p, again_x = gradients(x, parameters, gu, gl, None)
    same = torch.equal(ref_p, again_p) and torch.equal(ref_x, again_x)
    print(f"natural grid, repeated: bit-identical={same}")
    print(
        f"{'blocks':>7} {'params identical':>17} {'max |dp|':>10} {'max |dp|/max|p|':>16} "
        f"{'coords identical':>17}"
    )
    scale = ref_p.abs().max().item()
    for blocks in args.blocks:
        grad_p, grad_x = gradients(x, parameters, gu, gl, blocks)
        diff = (grad_p - ref_p).abs().max().item()
        print(
            f"{blocks:>7} {str(torch.equal(grad_p, ref_p)):>17} {diff:>10.2e} "
            f"{diff / scale:>16.2e} {str(torch.equal(grad_x, ref_x)):>17}"
        )
    os.environ.pop("FUSED_HOAD_BWD_BLOCKS", None)


if __name__ == "__main__":
    main()
