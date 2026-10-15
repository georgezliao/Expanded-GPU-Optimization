"""Model construction primitives."""

from collections.abc import Callable

from torch import nn

from fused_hoad.specs import MLPConfig

_ACTIVATIONS: dict[str, Callable[[], nn.Module]] = {
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
}


def build_mlp(config: MLPConfig) -> nn.Sequential:
    """Build the canonical dense MLP described by ``config``."""
    layers: list[nn.Module] = []
    in_features = config.input_dim

    for _ in range(config.hidden_layers):
        layers.append(nn.Linear(in_features, config.hidden_dim))
        layers.append(_ACTIVATIONS[config.activation]())
        in_features = config.hidden_dim

    layers.append(nn.Linear(in_features, config.output_dim))
    return nn.Sequential(*layers)

