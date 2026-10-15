# Architecture

The repository is split into four layers:

- `src/fused_hoad/` contains the public Python API, eager PyTorch reference code, and
  the lazy native-extension loader.
- `csrc/` contains the PyTorch C++ binding and CUDA kernels.
- `benchmarks/` contains repeatable performance measurements.
- `tests/` checks numerical behavior and the Python/native boundary.

## Foundation primitives

Build upward from small contracts that can be tested independently:

1. `specs.py` defines the fixed MLP shape and activation contract.
2. `models.py` constructs the canonical network from that contract.
3. `autodiff.py` provides gradient, Hessian-diagonal, and Laplacian operators.
4. `residuals.py` composes a model with a problem-specific differential operator.
5. `validation.py` measures numerical error between reference and optimized results.
6. `_extension.py` is the boundary through which fused implementations are loaded.

This ordering keeps PDE choice, model mechanics, differentiation, validation, and native
execution separate. A fused kernel can replace several layers for speed while retaining
the same externally tested contract.

## Kernel contract to finalize

The CUDA source currently provides a buildable identity-kernel seam. Before implementing
the fused residual, record the fixed MLP's layer sizes, activation, parameter layout,
target differential operator, supported dtypes, and expected forward/backward behavior.
Those choices determine the kernel signature and should not be guessed from benchmark
goals alone.

## Initial benchmark contract

The first concrete target is the two-dimensional Poisson equation on `[-1, 1]^2`:

```text
Laplacian(u) = f
u(x, y) = sin(pi*x) sin(pi*y)
f(x, y) = -2*pi^2 sin(pi*x) sin(pi*y)
residual = Laplacian(u) - f
```

The optimized model contract is a `2 -> 64 -> 64 -> 64 -> 64 -> 1` tanh MLP. The eager
implementation is benchmarked at 1K, 4K, 16K, and 64K points in FP32. The analytical
solution and FP64 execution provide correctness references.

## Development flow

1. Express the target residual by composing primitives in `residuals.py`.
2. Add numerical tests covering representative and edge-case inputs.
3. Implement the equivalent operation under `csrc/`.
4. Compare outputs and gradients before collecting timing data.
5. Store generated Nsight reports under `artifacts/` (ignored by Git).

