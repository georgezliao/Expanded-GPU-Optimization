"""Forward-mode Laplacian: one forward pass, no double backward.

Each hidden state carries a second-order jet per point (Li et al., 2023):

    v : activations                     [batch, width]
    J : d v / d x                       [batch, dims, width]
    L : sum_d d^2 v / d x_d^2           [batch, width]

These are stacked into one ``[batch, dims + 2, width]`` tensor so each linear layer is a
single matmul. Activations apply the chain rule elementwise:

    h_v = act(z_v)
    h_J = act'(z_v) * z_J
    h_L = act'(z_v) * z_L + act''(z_v) * sum_d z_J[d]^2

This is the exact math the fused CUDA kernel implements, so it doubles as its oracle.
"""

from collections.abc import Sequence

import torch
from torch import Tensor, nn

# Each entry returns (act, act', act'') evaluated at z.
_ActivationJet = tuple[Tensor, Tensor, Tensor]


def _tanh_jet(z: Tensor) -> _ActivationJet:
    a = torch.tanh(z)
    s = 1.0 - a * a
    return a, s, -2.0 * a * s


def _silu_jet(z: Tensor) -> _ActivationJet:
    sigma = torch.sigmoid(z)
    first = sigma * (1.0 + z * (1.0 - sigma))
    second = sigma * (1.0 - sigma) * (2.0 + z * (1.0 - 2.0 * sigma))
    return z * sigma, first, second


_ACTIVATION_JETS = {
    nn.Tanh: _tanh_jet,
    nn.SiLU: _silu_jet,
}


# Both helpers update row slices in place instead of concatenating rows: a cat copies the
# whole jet, while the in-place fixups only touch the value and Laplacian rows. Autograd
# allows this because neither matmul nor mul saves its output for the backward pass.


def _linear(jet: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
    out = jet @ weight.T
    if bias is not None:
        # Only the value row receives the bias; derivatives of a constant vanish.
        out[:, 0].add_(bias)
    return out


def _activation(jet: Tensor, jet_fn) -> Tensor:
    z_v, z_j = jet[:, :1], jet[:, 1:-1]
    act, first, second = jet_fn(z_v)
    # first * jet is already correct for the J rows and the first term of the L row.
    out = first * jet
    out[:, -1:].add_(second * (z_j * z_j).sum(dim=1, keepdim=True))
    out[:, :1].copy_(act)
    return out


def _validate_model(model: nn.Module) -> list[nn.Module]:
    if not isinstance(model, nn.Sequential):
        raise TypeError("forward_laplacian requires an nn.Sequential MLP")
    modules = list(model)
    for module in modules:
        if not isinstance(module, nn.Linear) and type(module) not in _ACTIVATION_JETS:
            raise TypeError(f"unsupported module for forward_laplacian: {type(module).__name__}")
    if not modules or not isinstance(modules[-1], nn.Linear):
        raise TypeError("forward_laplacian requires the MLP to end with nn.Linear")
    if modules[-1].out_features != 1:
        raise ValueError("model output must have shape [batch, 1]")
    return modules


def tanh_mlp_forward_laplacian(
    linears: Sequence[tuple[Tensor, Tensor | None]], coordinates: Tensor
) -> tuple[Tensor, Tensor]:
    """Forward Laplacian of a tanh MLP given raw ``(weight, bias)`` pairs.

    Used to differentiate the fused kernel: its packed parameters unpack into views that
    feed straight into this function.
    """
    return _propagate(_initial_jet(coordinates), _tanh_layers(linears))


def forward_laplacian(model: nn.Module, coordinates: Tensor) -> tuple[Tensor, Tensor]:
    """Evaluate a tanh/SiLU MLP and its input Laplacian in a single forward pass."""
    modules = _validate_model(model)
    steps = [
        (module.weight, module.bias)
        if isinstance(module, nn.Linear)
        else _ACTIVATION_JETS[type(module)]
        for module in modules
    ]
    return _propagate(_initial_jet(coordinates), steps)


def _tanh_layers(linears: Sequence[tuple[Tensor, Tensor | None]]) -> list:
    steps: list = []
    for index, linear in enumerate(linears):
        steps.append(linear)
        if index < len(linears) - 1:
            steps.append(_tanh_jet)
    return steps


def _initial_jet(coordinates: Tensor) -> Tensor:
    if coordinates.ndim != 2:
        raise ValueError("coordinates must have shape [batch, dimensions]")
    if not coordinates.is_floating_point():
        raise TypeError("coordinates must use a floating-point dtype")
    batch, dims = coordinates.shape
    jet = coordinates.new_zeros(batch, dims + 2, dims)
    jet[:, 0] = coordinates
    jet[:, 1:-1] = torch.eye(dims, dtype=coordinates.dtype, device=coordinates.device)
    return jet


def _propagate(jet: Tensor, steps: list) -> tuple[Tensor, Tensor]:
    """Apply ``(weight, bias)`` steps as linear layers and callables as activation jets."""
    for step in steps:
        jet = _linear(jet, *step) if isinstance(step, tuple) else _activation(jet, step)
    return jet[:, 0], jet[:, -1]
