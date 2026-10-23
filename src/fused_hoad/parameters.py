"""Stable flat-parameter ABI for the fixed Poisson MLP."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class ParameterSlice:
    """Offsets and shape for one linear layer in the flat parameter buffer."""

    name: str
    out_features: int
    in_features: int
    weight_offset: int
    bias_offset: int

    @property
    def weight_elements(self) -> int:
        return self.out_features * self.in_features

    @property
    def end_offset(self) -> int:
        return self.bias_offset + self.out_features


def _make_layout() -> tuple[ParameterSlice, ...]:
    shapes = ((64, 2), (64, 64), (64, 64), (64, 64), (1, 64))
    layout: list[ParameterSlice] = []
    offset = 0
    for index, (out_features, in_features) in enumerate(shapes):
        weight_elements = out_features * in_features
        layout.append(
            ParameterSlice(
                name=f"layer{index}",
                out_features=out_features,
                in_features=in_features,
                weight_offset=offset,
                bias_offset=offset + weight_elements,
            )
        )
        offset += weight_elements + out_features
    return tuple(layout)


POISSON_PARAMETER_LAYOUT = _make_layout()
POISSON_PARAMETER_COUNT = POISSON_PARAMETER_LAYOUT[-1].end_offset


def _linear_layers(model: nn.Module) -> list[nn.Linear]:
    return [module for module in model.modules() if isinstance(module, nn.Linear)]


def pack_poisson_parameters(model: nn.Module, *, differentiable: bool = False) -> Tensor:
    """Pack the fixed 2x64x64x64x64x1 MLP into its CUDA ABI layout.

    With ``differentiable=True`` the buffer is a ``torch.cat`` of the parameters, so
    gradients on it flow back to the model; otherwise it is a detached copy.
    """
    layers = _linear_layers(model)
    if len(layers) != len(POISSON_PARAMETER_LAYOUT):
        raise ValueError(f"expected {len(POISSON_PARAMETER_LAYOUT)} linear layers")

    segments: list[Tensor] = []
    for layer, spec in zip(layers, POISSON_PARAMETER_LAYOUT, strict=True):
        expected_shape = (spec.out_features, spec.in_features)
        if tuple(layer.weight.shape) != expected_shape:
            raise ValueError(
                f"{spec.name}.weight must have shape {expected_shape}, "
                f"got {tuple(layer.weight.shape)}"
            )
        if layer.bias is None or tuple(layer.bias.shape) != (spec.out_features,):
            raise ValueError(f"{spec.name}.bias must have shape ({spec.out_features},)")
        segments.extend((layer.weight.reshape(-1), layer.bias.reshape(-1)))

    if differentiable:
        return torch.cat(segments)
    packed = segments[0].new_empty(POISSON_PARAMETER_COUNT)
    offset = 0
    for segment in segments:
        packed[offset : offset + segment.numel()].copy_(segment.detach())
        offset += segment.numel()
    return packed.contiguous()


def unpack_poisson_parameters(parameters: Tensor) -> dict[str, Tensor]:
    """Return shaped views into a packed parameter buffer for inspection."""
    if parameters.ndim != 1 or parameters.numel() != POISSON_PARAMETER_COUNT:
        raise ValueError(f"parameters must have shape [{POISSON_PARAMETER_COUNT}]")

    views: dict[str, Tensor] = {}
    for spec in POISSON_PARAMETER_LAYOUT:
        weight_end = spec.weight_offset + spec.weight_elements
        views[f"{spec.name}.weight"] = parameters[
            spec.weight_offset:weight_end
        ].view(spec.out_features, spec.in_features)
        views[f"{spec.name}.bias"] = parameters[spec.bias_offset : spec.end_offset]
    return views

