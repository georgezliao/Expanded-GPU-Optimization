# Fused CUDA Kernels for Higher-Order Autodiff

Computes the input Laplacian of a fixed MLP in one fused CUDA kernel, so PINN residuals and training steps don't have to go through PyTorch's autograd graph.

On an RTX A5000 at 16K points, a residual evaluation takes 0.131 ms, compared with 2.075 ms for eager autograd and 0.975 ms for a forward Laplacian under `torch.compile`. It uses 1 kernel launch per call where eager uses 94. At 64K points, a full training step takes 3.300 ms instead of 22.261 ms. The full tables are under [Results](#results), along with the kernel designs that didn't help.

The kernels are written for one architecture, the tanh MLP described under [Method](#method), so this isn't a general speedup for arbitrary PyTorch models.

## The problem

PINNs need derivatives of a network's output with respect to its inputs, often second order or higher. Nsight Systems profiles of the PyTorch pipeline point to three costs:

- **Graph traversal:** the framework builds and walks execution graphs for each higher-order derivative.
- **Tensor materialization:** short-lived intermediate tensors are allocated and written to VRAM, which saturates memory bandwidth.
- **Kernel launches:** many small sequential kernels leave the GPU waiting on launches.

## The approach

For a fixed MLP, a custom C++/CUDA implementation replaces the higher-order autograd pipeline:

- **Kernel fusion:** elementwise operations and the residual evaluation run in a single kernel, which keeps intermediate data in registers and shared memory.
- **Fewer VRAM reads and writes:** this moves the workload from memory-bound toward compute-bound.
- **PyTorch extension:** `torch.utils.cpp_extension` exposes the kernels to Python, so they can be called from PyTorch training and inference scripts.

## Method

The target is the input Laplacian of a fixed `2 -> 64 -> 64 -> 64 -> 64 -> 1` tanh MLP, used as the residual of the 2D Poisson problem `Laplacian(u) = f` on `[-1, 1]^2` with `u = sin(pi x) sin(pi y)`. All runs use FP32.

Eager autograd computes the Laplacian with one backward pass for the gradient and one more per input dimension. The fused path propagates a second-order jet forward instead (value, input gradient, Laplacian; Li et al., 2023). That turns each layer into one GEMM over `4 x batch` columns plus an elementwise tanh step, and the whole thing runs in a single CUDA kernel:

| kernel | design |
|---|---|
| forward v0 | one thread per point; jet held in local memory |
| forward v1 | 32-point tiles, jets and staged weights in shared memory, register-tiled GEMM |
| forward v2 (default) | v1 with two points per lane (64-point tiles), float2 loads |
| forward v3 | 32-point tiles, hidden GEMMs on TF32 tensor cores (WMMA) |
| forward v4 | v3 with `cp.async` double-buffered weights and no separate Z buffer (TF32, opt-in) |
| backward v1 | persistent blocks re-run the tile forward, save pre-activations to scratch, walk the layers back with the adjoint jet in shared memory; per-block weight-gradient rows, summed on the host |
| backward v2 | 16-point tiles, weights read through L1, 2 blocks/SM |
| backward v3 | v2 with one weight-gradient row per SM, updated with `atomicAdd` |
| backward v4 | v2 without storing layer 0's constant jet components in scratch |
| backward v5 | v4 with 8 inputs per thread in the `Wᵀ·zbar` GEMM |
| backward v6 (default) | v4 with stride-20 shared-memory rows and float4 loads in the weight-gradient loop |

The backward kernel returns gradients for the parameters and the input coordinates. A PyTorch reference backward, which recomputes the jet and calls `autograd.grad`, is kept for testing.

Correctness is checked before any timing. The fused forward matches the PyTorch forward Laplacian within atol 1e-5 / rtol 1e-4, and the PyTorch forward Laplacian matches eager autograd to 1e-10 in FP64. The CUDA backward matches the reference backward to within 1e-4 of each gradient block's scale. TF32 has separate, wider tolerances (see `tests/test_fused.py`).

## Results

Headline numbers come from `scripts/oscar_bench.sbatch`. Raw outputs are in `results/<date>-<gpu>-<job>/`. The tables below are from `results/2026-10-26-RTX-A5000-7960100/` (RTX A5000, driver 575.57.08, torch 2.8.0+cu128, commit `35318f6`). Times are medians of per-iteration CUDA events, in milliseconds.

Residual evaluation, `(u, Laplacian u)`:

| backend | 1K | 16K | 64K |
|---|---|---|---|
| eager autograd | 2.089 | 2.075 | 6.817 |
| forward Laplacian + `torch.compile` | 0.359 | 0.985 | 3.695 |
| forward Laplacian + `torch.compile` (reduce-overhead) | 0.134 | 0.975 | 3.692 |
| fused CUDA forward v2 | 0.038 | 0.131 | 0.512 |
| fused CUDA forward v3 (TF32) | 0.034 | 0.186 | 0.701 |
| fused CUDA forward v4 (TF32, pipelined) | 0.033 | 0.123 | 0.482 |

At 16K, eager launches 94 kernels per call, `torch.compile` 19, and the fused path 1.

Training step (forward, backward, Adam):

| backend | 4K | 16K | 64K |
|---|---|---|---|
| eager autograd | 5.991 | 6.085 | 22.261 |
| eager autograd, step captured as a CUDA graph | 1.887 | 5.290 | 21.922 |
| forward Laplacian + `torch.compile` | 1.717 | 2.578 | 9.361 |
| fused CUDA forward v2 + backward v2 | 0.947 | 0.954 | 3.622 |
| fused CUDA forward v2 + backward v4 | 0.947 | 0.954 | 3.534 |
| fused CUDA forward v2 + backward v6 | 0.951 | 0.956 | 3.300 |
| same, step captured as a CUDA graph | 0.269 | 0.878 | 3.348 |

Times vary between GPU models and between Slurm jobs on the same model. At 1K–4K the call is short enough that host CPU speed matters. Before the parameter-pack lookup was cached, the same kernel measured 23 µs and 45 µs of host time per call on two nodes. After the change it measured 16–20 µs, against about 30 µs of kernel time. Compare rows within one results directory. Earlier runs are kept under `results/` for reference.

Things that did not help:

- Forward v0 was slower than eager at 64K. Its per-thread jet lives in local memory, and ncu reported 93% DRAM throughput.
- Forward v3 (TF32) was slower than v2 at 16K and 64K, and its Laplacian error was about 1e3 times larger. On an A40, ncu showed 0.25 instructions issued per cycle against 0.62 for v2, with the tensor pipe active 15% of the time. Forward v4 prefetches the weights with `cp.async`; that cut the `long_scoreboard` stall ratio from 2.97 to 1.23 and made TF32 faster than v2 at 16K–64K. The error is the same as v3, so v4 is opt-in (`fused_cuda_tf32_pipelined`).
- With no shared-memory padding (leading dims 128/64), the TF32 kernel had 48.8M load bank conflicts and was 1.5× slower.
- Backward v3 (SM-shared atomic rows) was slower than v2 on an RTX A5500. DRAM throughput dropped only from 46% to 41%.
- Backward v2 reaches 2 blocks/SM, but its training step was no faster than v1 on the A40 or the A5500.
- Backward v5 lowered the `mio_throttle` stall ratio from 1.65 to 1.45 on an A40, but the 64K training step stayed at about 2.76 ms, the same as v4.
- `torch.func` (`vmap(jacfwd(jacrev))`) was slower than eager at every batch size tested.

## Open questions

Results that are still unexplained or need another experiment:

- ncu and the step benchmark disagree on the small backward changes. On an A40, v4 showed the same 2.62 ms as v2 in ncu but a 2.6% faster 64K step, while v6 was 11.6% faster than v4 in ncu and about 3% faster per step. The step also contains the forward kernel, packing, autograd and Adam, which accounts for only part of this. ncu's clock control may also matter, but that hasn't been tested.
- Backward v2 raised achieved occupancy from about 17% to about 30% on the A40, and DRAM throughput rose from 23% to 61% in a separate ncu run. Backward v3 reduced that DRAM traffic only slightly, so the per-tile weight-gradient updates are not its main source. The remaining global loads have not been attributed with source counters.
- Backward v5 lowered `mio_throttle`, but `long_scoreboard` rose and the step did not change. It may have moved the stall from shared-memory instructions to L1 weight loads.
- The `wait` stall in the TF32 kernels barely changed between forward v3 and v4 (2.41 to 2.27). The chained `mma_sync` calls are the suspected cause. Bank conflicts also did not change.
- Final PINN error depends on the GPU model. Reruns on one node were bit-identical, and a rerun on another A5500 matched to every printed digit. With the same seed and code, fused seed 0 ended at 1.97e-2 on an A5500 and 3.33e-3 in a job whose GPU was not recorded. Summation order in the backward grid is one candidate.
- One or two seeds can end much worse than the rest. Over 8 seeds on an A40, eager had no final relative L2 above 1e-2, while `torch.compile` and the fused path had one each, at different seeds. Eight seeds cannot separate the backends. The cause of the bad seeds is not known.
- Host-side time varies between nodes. Two A40 nodes measured 0.80 ms and 1.08 ms for the same 16K backward v4 step.
- The PyTorch forward Laplacian trains slower than eager at 64K although its forward is faster. Its backward has not been profiled.

Directions to try next: build a roofline for these kernels, including a shared-memory roofline, since shared-memory bandwidth limited several versions. Test the kernel design on a higher-order operator such as the biharmonic, and on a PDE harder than this Poisson problem. Find what prevents `torch.compile` from fusing below 19 kernels. Check whether gradients stay bit-identical when only the backward block count changes.

## Repository layout

```text
GPU-Optimization/
├── benchmarks/          # Inference, training, end-to-end PINN, launch counts, ncu probes
├── csrc/                # PyTorch bindings and the forward/backward CUDA kernels
├── docs/                # Architecture and kernel-contract notes
├── results/             # Raw benchmark outputs from scripts/oscar_bench.sbatch
├── scripts/             # OSCAR job, benchmark and profiling scripts
├── src/fused_hoad/      # Package: backends, autodiff reference, parameter ABI, training
└── tests/               # Numerical parity and extension tests
```

## Getting started

Requirements: Python 3.10+, PyTorch 2.2+, the CUDA toolkit, and an NVIDIA GPU (sm_80+ for the TF32 kernel).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]' --no-build-isolation
pytest
```

The CUDA tests are skipped when no GPU or built extension is present.

```bash
python -m benchmarks.benchmark_poisson                  # residual evaluation, all backends
python -m benchmarks.benchmark_training --cuda-graph    # training step
python -m benchmarks.train_poisson --cuda-graph         # end-to-end PINN: wall time, relative L2
```

For a quick CPU check of the PyTorch backends, pass `--device cpu --batch-sizes 32 --warmup 1 --iterations 2` to `benchmark_poisson`.

On a Slurm cluster, `sbatch scripts/oscar_bench.sbatch` builds the extension, runs the tests and all benchmarks, and writes the outputs to `results/`.

## References

Forward-mode and high-order derivatives:

- Li et al., [Forward Laplacian: A New Computational Framework for Neural Network-based Variational Monte Carlo](https://arxiv.org/abs/2307.08214), 2023. [folx](https://github.com/netket/folx) is a JAX implementation of this approach.
- Shi, Hu, Lin, Kawaguchi, [Stochastic Taylor Derivative Estimator](https://arxiv.org/abs/2412.00088), 2024. It uses Taylor-mode AD with random tangents for arbitrary differential operators; the forward Laplacian computes one such operator exactly.
- [Hutchinson Trace Estimation for High-Dimensional and High-Order Physics-Informed Neural Networks](https://arxiv.org/abs/2312.14499)
- [Derivative Computation in PINNs: Automatic Differentiation, Finite Differences and Beyond](https://arxiv.org/abs/2608.11020)
- [Verified residual-specific explicit derivative kernels for physics-informed learning and discretized PDE adjoints](https://arxiv.org/abs/2606.29702)

PINN training:

- Wang, Teng, Perdikaris, [Understanding and mitigating gradient pathologies in physics-informed neural networks](https://arxiv.org/abs/2001.04536), 2020.
- [Mitigating Gradient Pathology in PINNs through Aligned Constraint](https://arxiv.org/abs/2605.25001), 2026.

GPU kernels and profiling:

- [Optimization of Tensor-product Operations in Nekbone on GPUs](https://arxiv.org/abs/2005.13425) proposes a shared-memory roofline.
- [Acceleration of Tensor-Product Operations with Tensor Cores](https://arxiv.org/abs/2407.09621)
- [Accelerating HPC Applications with NVIDIA Nsight Compute Roofline Analysis](https://developer.nvidia.com/blog/accelerating-hpc-applications-with-nsight-compute-roofline-analysis)
- [Roofline on NVIDIA GPUs hackathon materials](https://nersc.gov/roofline-on-nvidia-gpus-hackathon) (NERSC, 2020)

Talks:

- [Roofline and NVIDIA Ampere GPU Architecture Analysis](https://www.youtube.com/watch?v=VtkxhygfNsY)
- [GTC 2020 roofline session](https://developer.nvidia.com/gtc/2020/video/s21565-vid) (NERSC and Berkeley Lab)
- [Nsight Compute videos](https://developer.nvidia.com/nsight-compute-videos)
- [Shared Memory Kernel Optimization](https://www.youtube.com/watch?v=3bko4iL5MJk)
- [Shared Memory Atomics and Dynamic Allocation in CUDA](https://www.youtube.com/watch?v=Bwv5J7dHYjU)
- [CUDA Crash Course: Optimizing Histogram Kernels](https://www.youtube.com/watch?v=v4DGiuU53h8)
- [Stochastic Taylor Derivative Estimator](https://www.youtube.com/watch?v=5zYRENUOyW4)
