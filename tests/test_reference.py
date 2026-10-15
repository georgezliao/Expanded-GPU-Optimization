import pytest
import torch

from fused_hoad.reference import mlp_second_derivative


def test_quadratic_model_has_expected_laplacian() -> None:
    class Quadratic(torch.nn.Module):
        def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
            return (coordinates**2).sum(dim=1, keepdim=True)

    coordinates = torch.randn(8, 3, dtype=torch.float64)
    output, laplacian = mlp_second_derivative(Quadratic(), coordinates)

    torch.testing.assert_close(output, (coordinates**2).sum(dim=1, keepdim=True))
    torch.testing.assert_close(laplacian, torch.full((8, 1), 6.0, dtype=torch.float64))


def test_rejects_non_matrix_input() -> None:
    with pytest.raises(ValueError, match="batch, dimensions"):
        mlp_second_derivative(torch.nn.Identity(), torch.randn(8))

