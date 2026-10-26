import pytest
import torch

from fused_hoad import (
    DEFAULT_POISSON_MLP,
    build_mlp,
    forward_laplacian,
    fused_forward_laplacian,
    pack_poisson_parameters,
)
from fused_hoad._extension import extension_available

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and extension_available()),
    reason="needs CUDA and the built extension",
)


def _model(seed: int = 0) -> torch.nn.Module:
    torch.manual_seed(seed)
    return build_mlp(DEFAULT_POISSON_MLP).cuda().requires_grad_(False)


VERSIONS = [0, 1, 2]


@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("batch", [1, 2, 31, 32, 33, 63, 64, 65, 1_000, 65_536])
def test_fused_matches_forward_laplacian(batch: int, version: int) -> None:
    model = _model()
    coordinates = torch.empty(batch, 2, device="cuda").uniform_(-1.0, 1.0)
    expected_u, expected_lap = forward_laplacian(model, coordinates)
    u, lap = fused_forward_laplacian(coordinates, pack_poisson_parameters(model), version)
    torch.testing.assert_close(u, expected_u, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(lap, expected_lap, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("version", VERSIONS)
def test_fused_matches_float64_oracle_on_large_weights(version: int) -> None:
    """Scaled weights push tanh into saturation, where cancellation errors would show."""
    model = _model(seed=1)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.mul_(3.0)
    coordinates = torch.empty(4_096, 2, device="cuda").uniform_(-1.0, 1.0)
    _, expected = forward_laplacian(model.double(), coordinates.double())
    _, lap = fused_forward_laplacian(coordinates, pack_poisson_parameters(model.float()), version)
    scale = expected.abs().max().item()
    assert (lap.double() - expected).abs().max().item() <= 1e-4 * scale


def test_fused_handles_empty_batch() -> None:
    u, lap = fused_forward_laplacian(
        torch.empty(0, 2, device="cuda"), pack_poisson_parameters(_model())
    )
    assert u.shape == lap.shape == (0, 1)


@pytest.mark.parametrize(
    ("coordinates", "message"),
    [
        (lambda: torch.zeros(4, 2, device="cuda", dtype=torch.float64), "float32"),
        (lambda: torch.zeros(4, 3, device="cuda"), r"\[batch, 2\]"),
        (lambda: torch.zeros(4, 2), "CUDA"),
    ],
)
def test_fused_rejects_invalid_coordinates(coordinates, message: str) -> None:
    with pytest.raises(RuntimeError, match=message):
        fused_forward_laplacian(coordinates(), pack_poisson_parameters(_model()))


def test_fused_rejects_unknown_version() -> None:
    coordinates = torch.zeros(4, 2, device="cuda")
    with pytest.raises(RuntimeError, match="version"):
        fused_forward_laplacian(coordinates, pack_poisson_parameters(_model()), 7)


def test_fused_backend_repacks_after_in_place_weight_update() -> None:
    from fused_hoad import get_backend

    model = _model()
    coordinates = torch.empty(256, 2, device="cuda").uniform_(-1.0, 1.0)
    run = get_backend("fused_cuda")
    run(model, coordinates)
    with torch.no_grad():
        model[0].weight.mul_(0.5)
    _, lap = run(model, coordinates)
    _, expected = forward_laplacian(model, coordinates)
    torch.testing.assert_close(lap, expected, atol=1e-5, rtol=1e-4)


def _poisson_loss(u: torch.Tensor, lap: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
    from fused_hoad import manufactured_source

    return (lap - manufactured_source(coordinates)).pow(2).mean() + 0.1 * u.pow(2).mean()


@pytest.mark.parametrize(
    "name",
    [
        "fused_cuda_v0",
        "fused_cuda_v1",
        "fused_cuda",
        "fused_cuda_bwd_v1",
        "fused_cuda_bwd_v3",
        "fused_cuda_bwd_v2",
        "fused_cuda_recompute",
    ],
)
def test_fused_gradients_match_forward_laplacian(name: str) -> None:
    from fused_hoad import get_backend

    model = _model().requires_grad_(True)
    coordinates = torch.empty(4_096, 2, device="cuda").uniform_(-1.0, 1.0)

    def gradients(run) -> list[torch.Tensor]:
        model.zero_grad()
        x = coordinates.clone().requires_grad_(True)
        _poisson_loss(*run(model, x), coordinates).backward()
        return [p.grad.clone() for p in model.parameters()] + [x.grad.clone()]

    expected = gradients(forward_laplacian)
    actual = gradients(get_backend(name))
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-4)


def test_fused_trains_like_forward_laplacian() -> None:
    from fused_hoad import get_backend

    coordinates = torch.empty(2_048, 2, device="cuda").uniform_(-1.0, 1.0)

    def train(run) -> list[float]:
        model = _model().requires_grad_(True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        losses = []
        for _ in range(50):
            optimizer.zero_grad()
            loss = _poisson_loss(*run(model, coordinates), coordinates)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        return losses

    fused = train(get_backend("fused_cuda"))
    reference = train(forward_laplacian)
    assert fused[-1] < 0.5 * fused[0]
    torch.testing.assert_close(torch.tensor(fused), torch.tensor(reference), rtol=1e-3, atol=0)


def test_fused_inference_path_stays_out_of_autograd() -> None:
    from fused_hoad import get_backend

    model = _model().requires_grad_(True)
    with torch.no_grad():
        u, lap = get_backend("fused_cuda")(model, torch.zeros(8, 2, device="cuda"))
    assert not u.requires_grad and not lap.requires_grad


BACKWARD_VERSIONS = [1, 2, 3, 4]


def _backward_pair(batch: int, weight_scale: float = 1.0, seed: int = 0, version: int = 2):
    from fused_hoad._extension import fused_forward_laplacian_backward
    from fused_hoad.backends.fused import _recompute_backward

    model = _model(seed)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.mul_(weight_scale)
    parameters = pack_poisson_parameters(model)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    coordinates = torch.rand(batch, 2, device="cuda", generator=generator) * 2 - 1
    grad_u = torch.randn(batch, 1, device="cuda", generator=generator)
    grad_lap = torch.randn(batch, 1, device="cuda", generator=generator)
    actual = fused_forward_laplacian_backward(coordinates, parameters, grad_u, grad_lap, version)
    expected_x, expected_p = _recompute_backward(
        coordinates, parameters, grad_u, grad_lap, True, True
    )
    return actual, (expected_p, expected_x)


def _assert_scaled_close(actual: torch.Tensor, expected: torch.Tensor, tolerance: float) -> None:
    """Sums over thousands of points: compare against the gradient's own scale."""
    scale = expected.abs().max().item()
    error = (actual - expected).abs().max().item()
    assert error <= tolerance * scale, f"max error {error:.3e} vs scale {scale:.3e}"


@pytest.mark.parametrize("version", BACKWARD_VERSIONS)
@pytest.mark.parametrize("batch", [1, 15, 16, 17, 31, 32, 33, 1_000, 65_536])
def test_cuda_backward_matches_recompute(batch: int, version: int) -> None:
    (grad_p, grad_x), (expected_p, expected_x) = _backward_pair(batch, version=version)
    _assert_scaled_close(grad_p, expected_p, 1e-4)
    _assert_scaled_close(grad_x, expected_x, 1e-4)


@pytest.mark.parametrize("version", BACKWARD_VERSIONS)
def test_cuda_backward_matches_recompute_with_saturated_tanh(version: int) -> None:
    (grad_p, grad_x), (expected_p, expected_x) = _backward_pair(
        4_096, weight_scale=3.0, seed=1, version=version
    )
    _assert_scaled_close(grad_p, expected_p, 1e-4)
    _assert_scaled_close(grad_x, expected_x, 1e-4)


@pytest.mark.parametrize("version", BACKWARD_VERSIONS)
def test_cuda_backward_matches_per_parameter_group(version: int) -> None:
    """Catch layout bugs: every weight and bias block must match, not just the max."""
    from fused_hoad import unpack_poisson_parameters

    (grad_p, _), (expected_p, _) = _backward_pair(2_048, version=version)
    actual = unpack_poisson_parameters(grad_p)
    expected = unpack_poisson_parameters(expected_p)
    for name, want in expected.items():
        _assert_scaled_close(actual[name], want, 1e-4)


def test_cuda_backward_handles_empty_batch() -> None:
    from fused_hoad._extension import fused_forward_laplacian_backward

    parameters = pack_poisson_parameters(_model())
    empty = torch.empty(0, 2, device="cuda")
    grad_p, grad_x = fused_forward_laplacian_backward(
        empty, parameters, torch.empty(0, 1, device="cuda"), torch.empty(0, 1, device="cuda")
    )
    assert grad_x.shape == (0, 2)
    assert torch.count_nonzero(grad_p) == 0


def test_cuda_graph_training_step_matches_eager_step() -> None:
    from fused_hoad import get_backend
    from fused_hoad.training import cuda_graph_step

    coordinates = torch.empty(2_048, 2, device="cuda").uniform_(-1.0, 1.0)
    run = get_backend("fused_cuda")

    def train(graph: bool) -> list[float]:
        model = _model().requires_grad_(True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, capturable=graph)

        def loss_fn() -> torch.Tensor:
            return _poisson_loss(*run(model, coordinates), coordinates)

        if graph:
            step = cuda_graph_step(loss_fn, optimizer, warmup=3)
            return [step().item() for _ in range(20)]
        losses = []
        for _ in range(23):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        return losses[3:]

    # Warmup runs 3 real steps (Adam state must exist before capture), so skip those.
    torch.testing.assert_close(
        torch.tensor(train(True)), torch.tensor(train(False)), rtol=1e-4, atol=0
    )


# TF32 keeps 10 mantissa bits in the GEMM inputs. Measured on an RTX A5000, error is
# max |error| / max |reference| over the batch: 64K points gave 6.6e-4 (Laplacian) at default
# init and 2.3e-3 with weights x3; a single point with weights x3 gave 1.1e-2.
TF32_TOLERANCE = {1.0: 5e-3, 3.0: 2e-2}  # by weight scale
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8,
    reason="TF32 tensor cores need sm_80+",
)
@pytest.mark.parametrize("weight_scale", [1.0, 3.0])
@pytest.mark.parametrize("batch", [1, 31, 32, 33, 4_096])
def test_tf32_forward_within_stated_tolerance(batch: int, weight_scale: float) -> None:
    model = _model()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.mul_(weight_scale)
    coordinates = torch.empty(batch, 2, device="cuda").uniform_(-1.0, 1.0)
    expected_u, expected_lap = forward_laplacian(model.double(), coordinates.double())
    u, lap = fused_forward_laplacian(coordinates, pack_poisson_parameters(model.float()), 3)
    _assert_scaled_close(u.double(), expected_u, TF32_TOLERANCE[weight_scale])
    _assert_scaled_close(lap.double(), expected_lap, TF32_TOLERANCE[weight_scale])
