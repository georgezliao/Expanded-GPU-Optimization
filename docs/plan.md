# Plan: from scaffold to a fused Laplacian kernel

Goal: a working end-to-end comparison on the 2D Poisson contract
(`2 -> 64x4 -> 1` tanh MLP, FP32, 1K–64K points). Every implementation returns
`(u, Laplacian(u))`, is checked against the eager oracle, and is then timed and profiled.
Get something simple working across the whole pipeline first. Optimize after that.

## Where the repo stands

Status (2026-10-23): milestones 1–3 and 5 done, plus a fused CUDA backward (forward 17x, training
step 8x eager at 64K); see findings.md.

Done: specs, model, autodiff primitives, Poisson residual, flat parameter ABI
(`parameters.py`, 12,737 floats), validation metrics, eager benchmark, nsys script, tests.

Gaps and bugs:
- ~~bindings/kernel name mismatch~~ fixed with the CUDA v0 kernel.
- This machine has no CUDA (Mac, MPS only). Build and benchmark on a remote NVIDIA GPU:
  Brown's OSCAR `gpu` partition, or Colab/Lambda as a fallback. Pure-PyTorch rungs and
  tests still run locally on CPU.
- `parameters.py` and the binding changes are not committed yet.

## Core idea: forward-mode Laplacian (no double backward)

Eager autograd builds a graph, walks it backward once for the gradient, then walks it
backward once per dimension for the Hessian diagonal. That causes all three bottlenecks
in the README: repeated traversal, materialized intermediates and many small kernels.

Instead, push a second-order Taylor jet forward through the network in a single pass
(the "Forward Laplacian", Li et al. 2023, *Nature Machine Intelligence*; also what
Taylor-mode AD does). Each point carries, per layer:

```text
v  : activations            [64]
J  : d v / d x              [64, 2]
L  : sum_d d^2 v / d x_d^2  [64]
```

Linear layer `z = W h + b`:
```text
z_v = W h_v + b,   z_J = W h_J,   z_L = W h_L
```
tanh, with `a = tanh(z_v)`, `s = 1 - a^2`, `s' = -2 a s`:
```text
h_v = a
h_J = s * z_J                      (broadcast over the 2 columns)
h_L = s * z_L + s' * sum_d z_J[:, d]^2
```
Input: `v = x`, `J = I_2`, `L = 0`. Output: `u = v_out`, `Laplacian(u) = L_out`.

So each layer is one GEMM of `W` against a `[64, 4·B]` matrix (columns for v, J0, J1 and L),
plus one elementwise epilogue. That shape is ideal for fusion. It is the whole kernel.

## Implementation ladder

Each rung goes behind the same function signature and the same correctness test.
Each rung removes one of the README's bottlenecks, so the benchmark tells a clear story.

| # | Backend | File | Removes |
|---|---------|------|---------|
| 0 | Eager `autograd.grad(create_graph=True)` | `problems/poisson.py` (exists) | — (baseline) |
| 1 | `torch.func` (`vmap(hessian)` / `jacfwd(jacrev)`) | `backends/func.py` | Python-level graph walking |
| 2 | Rung 0 or 3 + `torch.compile(mode="max-autotune")` + CUDA Graphs | `backends/compiled.py` | Launch overhead (pinns-torch's approach) |
| 3 | Forward Laplacian in plain PyTorch (`torch.addmm` + epilogue) | `backends/forward_laplacian.py` | Double backward and most materialization |
| 4 | Fused CUDA kernel (rung 3 in one launch) | `csrc/fused_residual.cu` | Global-memory round trips and launch overhead |

Rung 3 matters most. It is a second, independent oracle for the CUDA kernel, which is
written as the same math, line for line. It is also a strong baseline that is easy to
explain: "the algorithm change gives X, and fusion adds Y on top."

Prior art to borrow from, not rebuild:
- **pinns-torch** (Bafghi & Raissi, NeurIPS 2023 DLDE workshop): reports up to 9x speedup
  over TF1 by using TorchScript plus CUDA Graphs. Copy the capture/replay pattern from
  `pinnstorch/utils/jit_compiler.py` for rung 2, and its `gradient_fn.py` derivative
  helper as a cross-check.
- **PhysicsNeMo Sym** `physicsnemo/sym/eq/derivatives.py`: autodiff via a TorchScript-ed
  `autograd.grad`, plus `MeshlessFiniteDerivative`, which approximates the Laplacian with
  extra forward passes at stencil offsets. Add the meshless finite derivative as an
  optional "approximate" baseline (it trades accuracy for speed), useful in the writeup.
- **tiny-cuda-nn fully fused MLP** (Müller et al.): reference design for keeping 64-wide
  weights in shared memory and activations on-chip. It is a source of tiling ideas only. It
  has no second-derivative support, which is the gap this project fills.

## CUDA kernel, in three passes

Keep the signature fixed:
`fused_forward_laplacian(coords[B,2] f32, params[12737] f32) -> (u[B,1], lap[B,1])`.

**v0: correct, naive (target: about one day).** One thread per point, with `v/J/L` in
local arrays (4×64 floats; spills are fine for now). Weights are read from global memory
through `__ldg`. Fix the bindings name mismatch. Pass `torch.testing.assert_close`
against rung 3 and the eager oracle at FP32 (`rtol≈1e-4`) on random weights and on a
trained model.

**v1: shared-memory weights + point tiles.** Each block stages one layer's `W` (64×64 =
16 KB) and `b` into shared memory. A block handles a tile of points (for example 32 points
× 4 jet columns = 128 columns), with threads mapped to (neuron, column). Run the
epilogue right after each layer's GEMM, and keep activations in shared memory between
layers. Nothing goes back to global memory except the final `u` and `lap`.

**v2 (stretch): tensor cores.** Run the per-layer GEMM with WMMA/`mma.sync` in TF32, or
generate the same fused kernel in Triton for comparison. Only do this if v1 profiling
shows the kernel is compute-bound.

Training (backward with respect to parameters) is out of scope until v1 is benchmarked.
The cheap path then: wrap rung 3 in autograd for training, and use the fused kernel for
residual evaluation and inference. A custom `torch.autograd.Function` backward is a stretch goal.

## Milestones

1. **Housekeeping.** Commit the parameter ABI. Add a `backends/` package with a registry
   `{name: fn(model_or_params, coords) -> (u, lap)}`. Make the benchmark iterate over
   backends.
   Done when: `pytest` passes on CPU and the benchmark prints a table for every backend.
2. **Rung 3 + parity tests.** Implement the forward Laplacian in PyTorch, and test it
   against eager in FP64 (`atol 1e-10`).
   Done when: it passes on CPU and is faster than eager on a GPU.
3. **Rungs 1–2.** Add `torch.func` and compiled/CUDA-Graph versions. These are a few
   lines each.
4. **GPU environment.** Set up an OSCAR job script (`scripts/oscar_job.sh`) that builds
   the extension and runs `pytest` and the benchmark.
5. **CUDA v0.** Done when: the parity test passes on GPU and appears in the benchmark table.
6. **Profile.** Run `nsys` on rung 0 and v0 to show kernel counts and launch gaps
   (the README's three claims). Run `ncu` on v0/v1 for memory throughput, occupancy and
   a roofline plot. Save reports to `artifacts/`.
7. **CUDA v1.** Done when: it is faster than rung 2 at 16K–64K points, with parity tests
   still passing.
8. **Writeup.** Plot ms and peak MiB against batch size for each backend, compare
   nsys timelines, add a roofline point, and record the accuracy table.

## Measurement rules

- Check correctness before timing, every run (the benchmark asserts parity first).
- Time with CUDA events, after warmup, with `torch.cuda.synchronize`. Report the median
  of N runs, plus peak memory.
- Use the same weights and the same coordinates (seeded) for every backend.
- Record GPU model, driver, CUDA and PyTorch versions with each result.
