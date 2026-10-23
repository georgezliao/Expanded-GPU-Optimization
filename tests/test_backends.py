import pytest
import torch

from fused_hoad import (
    BACKENDS,
    DEFAULT_POISSON_MLP,
    MLPConfig,
    build_mlp,
    forward_laplacian,
    get_backend,
)
from fused_hoad.backends import FIXED_CONTRACT_BACKENDS, REFERENCE_BACKEND, TRAINABLE_BACKENDS

GENERAL_BACKENDS = sorted(set(BACKENDS) - set(FIXED_CONTRACT_BACKENDS))

CONFIGS = [
    DEFAULT_POISSON_MLP,
    MLPConfig(input_dim=3, hidden_dim=16, hidden_layers=2, activation="tanh"),
    MLPConfig(input_dim=2, hidden_dim=32, hidden_layers=3, activation="silu"),
]

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _model_and_coordinates(
    config: MLPConfig, batch: int = 64, device: str = "cpu", dtype=torch.float64
):
    torch.manual_seed(0)
    model = build_mlp(config).to(device=device, dtype=dtype)
    coordinates = torch.empty(batch, config.input_dim, device=device, dtype=dtype)
    return model, coordinates.uniform_(-1.0, 1.0)


def test_registry_contains_reference_and_forward_laplacian() -> None:
    assert REFERENCE_BACKEND in BACKENDS
    assert set(TRAINABLE_BACKENDS) <= set(BACKENDS)
    assert get_backend("forward_laplacian") is forward_laplacian


def test_unknown_backend_lists_available_names() -> None:
    with pytest.raises(KeyError, match="forward_laplacian"):
        get_backend("does-not-exist")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", GENERAL_BACKENDS)
@pytest.mark.parametrize("config", CONFIGS, ids=lambda c: f"{c.activation}-{c.input_dim}d")
def test_backend_matches_eager_in_float64(name: str, config: MLPConfig, device: str) -> None:
    model, coordinates = _model_and_coordinates(config, device=device)
    expected_u, expected_lap = get_backend(REFERENCE_BACKEND)(model, coordinates)
    u, lap = get_backend(name)(model, coordinates)

    assert u.shape == lap.shape == (coordinates.shape[0], 1)
    torch.testing.assert_close(u, expected_u, atol=1e-12, rtol=1e-10)
    torch.testing.assert_close(lap, expected_lap, atol=1e-10, rtol=1e-8)


@pytest.mark.parametrize("device", DEVICES)
def test_forward_laplacian_matches_eager_in_float32(device: str) -> None:
    model, coordinates = _model_and_coordinates(
        DEFAULT_POISSON_MLP, batch=256, device=device, dtype=torch.float32
    )
    _, expected = get_backend(REFERENCE_BACKEND)(model, coordinates)
    _, actual = forward_laplacian(model, coordinates)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("name", TRAINABLE_BACKENDS)
def test_parameter_gradients_match_eager(name: str) -> None:
    """Trainable backends must stay differentiable in the weights."""
    model, coordinates = _model_and_coordinates(DEFAULT_POISSON_MLP, batch=32)

    def parameter_gradients(backend: str) -> list[torch.Tensor]:
        model.zero_grad()
        _, lap = get_backend(backend)(model, coordinates)
        lap.pow(2).mean().backward()
        # The output bias does not affect the Laplacian, so eager leaves its grad unset.
        return [
            torch.zeros_like(p) if p.grad is None else p.grad.clone()
            for p in model.parameters()
        ]

    for actual, expected in zip(
        parameter_gradients(name), parameter_gradients(REFERENCE_BACKEND), strict=True
    ):
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-8)


@requires_cuda
@pytest.mark.parametrize("name", [n for n in BACKENDS if "cudagraph" in n])
def test_graph_replay_uses_new_coordinates(name: str) -> None:
    """A replay must read freshly copied inputs, not the coordinates seen at capture."""
    model, first = _model_and_coordinates(DEFAULT_POISSON_MLP, device="cuda")
    second = torch.empty_like(first).uniform_(-1.0, 1.0)
    run = get_backend(name)
    run(model, first)
    _, lap = run(model, second)
    _, expected = get_backend(REFERENCE_BACKEND)(model, second)
    torch.testing.assert_close(lap, expected, atol=1e-10, rtol=1e-8)


def test_forward_laplacian_of_linear_model_is_zero() -> None:
    """A model with no activations is linear, so its Laplacian is exactly zero."""
    model = torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.Linear(4, 1)).double()
    _, lap = forward_laplacian(model, torch.randn(8, 2, dtype=torch.float64))
    torch.testing.assert_close(lap, torch.zeros_like(lap))


def test_forward_laplacian_rejects_unsupported_modules() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.ReLU(), torch.nn.Linear(4, 1))
    with pytest.raises(TypeError, match="ReLU"):
        forward_laplacian(model, torch.randn(3, 2))
