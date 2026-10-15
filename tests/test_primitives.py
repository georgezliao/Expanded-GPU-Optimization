import pytest
import torch

from fused_hoad import MLPConfig, build_mlp, error_metrics, evaluate_residual, gradient


def test_mlp_matches_configured_shape() -> None:
    model = build_mlp(MLPConfig(input_dim=2, hidden_dim=16, hidden_layers=3))
    assert model(torch.randn(5, 2)).shape == (5, 1)


def test_config_rejects_non_positive_dimensions() -> None:
    with pytest.raises(ValueError, match="hidden_layers"):
        MLPConfig(input_dim=2, hidden_dim=16, hidden_layers=0)


def test_residual_operator_is_composable() -> None:
    class Sum(torch.nn.Module):
        def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
            return coordinates.sum(dim=1, keepdim=True)

    def first_derivative(output: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
        return gradient(output, coordinates)

    output, residual = evaluate_residual(Sum(), torch.randn(4, 3), first_derivative)
    assert output.shape == (4, 1)
    torch.testing.assert_close(residual, torch.ones(4, 3))


def test_error_metrics_reports_known_difference() -> None:
    metrics = error_metrics(torch.tensor([1.0, 4.0]), torch.tensor([1.0, 2.0]))
    assert metrics.max_absolute == 2.0
    assert metrics.max_relative == 1.0
    assert metrics.mean_absolute == 1.0

