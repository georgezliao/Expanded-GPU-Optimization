# Fused CUDA Kernels for Higher-Order Autodiff

Computes the input Laplacian of a fixed MLP in one fused CUDA kernel, so PINN residuals and training steps don't have to go through PyTorch's autograd graph.

On an RTX A5000 at 16K points, a residual evaluation takes 0.131 ms, compared with 2.109 ms for eager autograd and 0.975 ms for a forward Laplacian under `torch.compile`. It uses 1 kernel launch per call where eager uses 94. At 64K points, a full training step takes 3.334 ms instead of 22.292 ms, and peak memory during a step is 15.3 MiB instead of 977.7 MiB (measured above a post-warmup baseline). The full tables are under [Results](#results), along with the kernel designs that didn't help.

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
| forward v5 | v4 with 16 warps per block instead of 8 (TF32, opt-in) |
| backward v1 | persistent blocks re-run the tile forward, save pre-activations to scratch, walk the layers back with the adjoint jet in shared memory; per-block weight-gradient rows, summed on the host |
| backward v2 | 16-point tiles, weights read through L1, 2 blocks/SM |
| backward v3 | v2 with one weight-gradient row per SM, updated with `atomicAdd` |
| backward v4 | v2 without storing layer 0's constant jet components in scratch |
| backward v5 | v4 with 8 inputs per thread in the `Wᵀ·zbar` GEMM |
| backward v6 (default) | v4 with stride-20 shared-memory rows and float4 loads in the weight-gradient loop |

The backward kernel returns gradients for the parameters and the input coordinates. A PyTorch reference backward, which recomputes the jet and calls `autograd.grad`, is kept for testing.

Correctness is checked before any timing. The fused forward matches the PyTorch forward Laplacian within atol 1e-5 / rtol 1e-4, and the PyTorch forward Laplacian matches eager autograd to 1e-10 in FP64. The CUDA backward matches the reference backward to within 1e-4 of each gradient block's scale. TF32 has separate, wider tolerances (see `tests/test_fused.py`).

## Results

Headline numbers come from `scripts/oscar_bench.sbatch`. Raw outputs are in `results/<date>-<gpu>-<job>/`. The tables below are from `results/2026-11-06-RTX-A5000-8454035/` (RTX A5000, driver 575.57.08, torch 2.8.0+cu128, commit `b289645`). Times are medians of per-iteration CUDA events, in milliseconds.

Residual evaluation, `(u, Laplacian u)`:

| backend | 1K | 16K | 64K |
|---|---|---|---|
| eager autograd | 2.131 | 2.109 | 6.855 |
| forward Laplacian + `torch.compile` | 0.357 | 0.981 | 3.694 |
| forward Laplacian + `torch.compile` (reduce-overhead) | 0.133 | 0.975 | 3.688 |
| fused CUDA forward v2 | 0.038 | 0.131 | 0.538 |
| fused CUDA forward v3 (TF32) | 0.035 | 0.185 | 0.699 |
| fused CUDA forward v4 (TF32, pipelined) | 0.033 | 0.121 | 0.478 |
| fused CUDA forward v5 (TF32, 16 warps) | 0.031 | 0.112 | 0.453 |

At 16K, eager launches 94 kernels per call, `torch.compile` 19, and the fused path 1. Forward v4 and v5 are faster than the FP32 default here, but their Laplacian error is about 1e3 times larger, so they are opt-in.

Training step (forward, backward, Adam):

| backend | 4K | 16K | 64K |
|---|---|---|---|
| eager autograd | 6.112 | 6.047 | 22.292 |
| eager autograd, step captured as a CUDA graph | 1.879 | 5.296 | 21.927 |
| forward Laplacian + `torch.compile` | 1.716 | 2.592 | 9.358 |
| fused CUDA forward v2 + backward v2 | 0.943 | 0.952 | 3.642 |
| fused CUDA forward v2 + backward v4 | 0.943 | 0.958 | 3.565 |
| fused CUDA forward v2 + backward v6 | 0.948 | 0.962 | 3.334 |
| same, step captured as a CUDA graph | 0.268 | 0.827 | 3.411 |

Peak memory during a training step, in MiB above a baseline taken right after warmup:

| backend | 4K | 16K | 64K |
|---|---|---|---|
| eager autograd | 61.1 | 244.4 | 977.7 |
| forward Laplacian + `torch.compile` | 41.1 | 164.7 | 659.0 |
| fused CUDA forward v2 + backward v6 | 14.4 | 14.6 | 15.3 |

The baseline already holds the weights, gradients and Adam state. Captured CUDA graphs are left out of this table, because their memory pool is allocated during warmup and also sits in the baseline. Fused memory depends on the number of SMs: the block grid sets the size of the scratch and partials buffers, so an A40 (84 SMs) measured 19.8 MiB at 64K.

End-to-end PINN training: 5000 Adam steps, 16K interior and 1K boundary points, seeds 0–2. Time to target is the first evaluation (every 250 steps, clock paused) with relative L2 below 5e-3 against the analytic solution.

| backend | total seconds, median | seconds to target, median | final relative L2 by seed |
|---|---|---|---|
| eager autograd | 34.60 | 24.51 | 5.35e-3, 3.88e-3, 1.77e-2 |
| forward Laplacian + `torch.compile` | 17.67 | 13.47 | 1.00e-2, 7.57e-3, 1.02e-2 |
| fused CUDA forward v2 + backward v6 | 9.10 | 6.34 | 4.20e-3, 9.83e-3, 3.50e-3 |

The backends need a similar number of steps to reach the target, so the gap comes from time per step. Error is not monotonic: in this run every backend ended at least one seed above the 5e-3 it had already reached. Three seeds can't rank the backends on accuracy.

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
- Backward v2 raised achieved occupancy from about 17% to about 30% on the A40, and DRAM throughput rose from 23% to 61% in a separate ncu run, with no faster step. In backward v6 source counters (A40, 64K), scratch reads are about 46% of global load sectors, GEMM weights 29% and weight-gradient partials 18%, which fits backward v3 barely changing DRAM. The two weight-load lines collect the most stall samples. Prefetching them with `cp.async` won't fit the 49 KB limit for 2 blocks/SM, so that experiment would trade occupancy for it.
- Backward v5 lowered `mio_throttle`, but `long_scoreboard` rose and the step did not change. It may have moved the stall from shared-memory instructions to L1 weight loads.
- The `wait` stall ratio in the TF32 kernels barely moved across forward v3, v4 and v5 (2.41, 2.27, 2.21). The chained `mma_sync` calls are the suspected cause. Forward v5 still got faster by doubling warps active, and `math_pipe_throttle` rose from 0.42 to 1.37, which may mean the tensor pipe is close to saturated.
- Final PINN error depends on the GPU model. Reruns on one node were bit-identical, and runs on different nodes of the same model matched to every printed digit. With the same seed, code and backward, fused seed 0 ended at 1.97e-2 on an A5500 and 3.33e-3 in a job whose GPU was not recorded. Changing the backward block count changes weight gradients by up to 2.65e-7 relative, and different models get different grids, so summation order is one source. It isn't the only one, because eager also differs between models.
- One or two seeds can end much worse than the rest. Over 8 seeds on an A40, eager had no final relative L2 above 1e-2, while `torch.compile` and the fused path had one each, at different seeds. Eight seeds cannot separate the backends. The cause of the bad seeds is not known.
- Host-side time varies between nodes. Two A40 nodes measured 0.80 ms and 1.08 ms for the same 16K backward v4 step.
- The PyTorch forward Laplacian trains slower than eager at 64K although its forward is faster: 5.96 ms forward and 24.19 ms backward, against 7.46 ms and 16.57 ms (A40). Its backward spends 4.8 ms in device-to-device copies. Autograd's handling of the in-place row updates is the likely source, but this has not been traced.

Directions to try next: build a roofline for these kernels, including a shared-memory roofline, since shared-memory bandwidth limited several versions. Test the kernel design on a higher-order operator such as the biharmonic, and on a PDE harder than this Poisson problem. Find what prevents `torch.compile` from fusing below 19 kernels.

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
