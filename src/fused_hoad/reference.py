"""Correctness-first PyTorch implementations used as test oracles."""

from torch import Tensor, nn

from fused_hoad.autodiff import laplacian
from fused_hoad.residuals import evaluate_residual


def mlp_second_derivative(model: nn.Module, coordinates: Tensor) -> tuple[Tensor, Tensor]:
    """Evaluate an MLP and its diagonal second derivative with respect to inputs.

    The function intentionally uses eager autograd. It provides a transparent baseline
    against which fused kernels can be checked before performance comparisons are made.
    """
    def laplacian_operator(output: Tensor, x: Tensor) -> Tensor:
        if output.shape[1] != 1:
            raise ValueError("model output must have shape [batch, 1]")
        return laplacian(output, x)

    return evaluate_residual(model, coordinates, laplacian_operator)

