from math import pi

import pytest
import torch

from fused_hoad import (
    DEFAULT_POISSON_MLP,
    evaluate_poisson_residual,
    manufactured_solution,
    manufactured_source,
)


class ManufacturedSolution(torch.nn.Module):
    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        return manufactured_solution(coordinates)


def test_default_model_contract_is_fixed() -> None:
    assert DEFAULT_POISSON_MLP.input_dim == 2
    assert DEFAULT_POISSON_MLP.hidden_dim == 64
    assert DEFAULT_POISSON_MLP.hidden_layers == 4
    assert DEFAULT_POISSON_MLP.output_dim == 1
    assert DEFAULT_POISSON_MLP.activation == "tanh"


def test_manufactured_source_matches_analytical_laplacian() -> None:
    coordinates = torch.tensor([[0.25, 0.5], [-0.5, 0.25]], dtype=torch.float64)
    expected = -2.0 * pi**2 * manufactured_solution(coordinates)
    torch.testing.assert_close(manufactured_source(coordinates), expected)


def test_manufactured_solution_has_negligible_poisson_residual() -> None:
    torch.manual_seed(0)
    coordinates = torch.empty(32, 2, dtype=torch.float64).uniform_(-1.0, 1.0)
    _, residual = evaluate_poisson_residual(ManufacturedSolution(), coordinates)
    torch.testing.assert_close(residual, torch.zeros_like(residual), atol=1e-12, rtol=1e-12)


def test_manufactured_solution_is_zero_on_boundary() -> None:
    coordinates = torch.tensor(
        [[-1.0, -0.4], [1.0, 0.2], [0.3, -1.0], [-0.7, 1.0]], dtype=torch.float64
    )
    torch.testing.assert_close(
        manufactured_solution(coordinates),
        torch.zeros(4, 1, dtype=torch.float64),
        atol=1e-15,
        rtol=0.0,
    )


def test_poisson_problem_requires_two_coordinates() -> None:
    with pytest.raises(ValueError, match=r"\[batch, 2\]"):
        manufactured_solution(torch.randn(4, 3))
