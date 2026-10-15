"""Validated specifications shared by reference and fused implementations."""

from dataclasses import dataclass
from typing import Literal

ActivationName = Literal["tanh", "silu"]


@dataclass(frozen=True, slots=True)
class MLPConfig:
    """Shape contract for the fixed MLP targeted by a fused kernel."""

    input_dim: int
    hidden_dim: int
    hidden_layers: int
    output_dim: int = 1
    activation: ActivationName = "tanh"

    def __post_init__(self) -> None:
        for name in ("input_dim", "hidden_dim", "hidden_layers", "output_dim"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")

