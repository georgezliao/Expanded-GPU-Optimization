# Findings

All runs on OSCAR `gpu` nodes: torch 2.8.0+cu128, FP32, `2→64×4→1` tanh MLP, 2D Poisson residual. The GPU differs between runs (Slurm picks the node), so compare times within a run, not across runs. Run with `scripts/oscar_job.sh` unless noted.

Inference tables report the `(u, Laplacian u)` call, timed Python-to-Python over 100 iterations after 10 warmup calls (`benchmarks/benchmark_poisson.py`). Training tables report one full step (forward, backward, Adam), timed Python-to-Python over 50 iterations after 10 warmup steps (`benchmarks/benchmark_training.py`). Kernel-only times are from ncu or the torch profiler and say so.

## Run 1: eager vs forward Laplacian (RTX A6000)

| batch | eager ms | forward ms | eager peak MiB | forward peak MiB |
|---|---|---|---|---|
| 1K | 2.90 | 1.41 | 43.9 | 28.8 |
| 4K | 2.80 | 1.44 | 126.5 | 66.4 |
| 16K | 2.84 | 1.52 | 457.2 | 216.9 |
| 64K | 6.12 | 6.05 | 1780.1 | 818.6 |

Peak memory here is absolute `max_memory_allocated`; later runs subtract a baseline.

Eager time doesn't move between 1K and 16K. I read that as launch/host-bound rather than compute-bound, but I didn't profile host time in this run. At 64K the two backends take the same time.

The nsys kernel summary (both backends at 16K in one profile) had `CatArrayBatchedCopy` kernels at about 14% of total GPU kernel time. I replaced the `torch.cat` calls in the forward Laplacian with in-place row updates after this.

## Run 2: launch-overhead removal (GPU not recorded)

| backend | host launches | GPU kernels | 1K ms | 16K ms | 64K ms |
|---|---|---|---|---|---|
| eager | 94 | 97 | 2.03 | 2.03 | 5.26 |
| eager + CUDA graph | 1 | 98 | 0.21 | 1.21 | 5.16 |
| torch.func | 118 | 123 | 4.91 | 4.90 | 6.76 |
| forward Laplacian | 59 | 59 | 1.01 | 0.98 | 3.81 |
| forward + CUDA graph | 1 | 60 | 0.11 | 0.91 | 3.72 |
| forward + compile | 19 | 19 | 0.33 | 0.73 | 2.72 |
| forward + compile + CUDA graph | 2 | 20 | 0.10 | 0.71 | 2.70 |

Launch and kernel counts are for one call at 16K (`benchmarks/count_launches.py`). `torch.compile` used Inductor's default mode with `dynamic=False`. `max-autotune` was skipped because it would have autotuned a GEMM for every batch size.

The forward Laplacian without the cats went from 5.63 to 3.81 ms at 64K, but those two numbers come from different jobs and possibly different nodes.

CUDA graphs cut eager at 1K from 2.03 to 0.21 ms and change almost nothing at 64K (5.26 → 5.16 ms). torch.func (`vmap(jacfwd(jacrev))`) launched more kernels than eager and was slower at every batch size. I didn't look into why.

Compile is the only change here that moved the 64K time for the forward Laplacian (3.81 → 2.72 ms). It took the kernel count from 59 to 19.

The graph variants report ~0 MiB peak. The benchmark records `memory_allocated` after warmup as a baseline and reports the peak above it. Graph capture happens during warmup, so the graph's private pool is already in the baseline. The graphs do hold memory; this statistic doesn't show it.

The first `count_launches` run failed with "CUDA driver initialization failed" in a fresh process on the GPU node. The rerun worked, and I don't know the cause.

## Run 3: fused CUDA forward kernel (A40)

| backend | launches | 1K ms | 4K ms | 16K ms | 64K ms |
|---|---|---|---|---|---|
| eager | 94 | 2.33 | 2.32 | 2.31 | 7.28 |
| forward + compile + CUDA graph | 2 | 0.126 | 0.197 | 1.04 | 3.96 |
| fused v0 (thread per point) | 1 | 0.69 | 0.74 | 4.18 | 23.2 |
| fused v1 (32-pt smem tiles) | 1 | 0.033 | 0.047 | 0.154 | 0.536 |
| fused v2 (64-pt tiles, 2 pts/lane) | 1 | 0.034 | 0.034 | 0.130 | 0.418 |

v0 holds the whole jet per thread (`v`, `J0`, `J1` and `L` for 64 neurons, two buffers, about 2 KB) and was slower than eager at 64K. ncu at 64K: DRAM throughput 92.8%, executed IPC 0.11, 44 registers/thread. The jet arrays are dynamically indexed, so they live in local memory. My reading is that local-memory traffic accounts for the DRAM number, but I didn't check the local-load counters.

v1: each block owns 32 points. The jets live in shared memory as `[component][neuron][point]`, a warp owns 8 neurons, and each lane owns one point. Each layer's weights are staged transposed into smem so a warp reads its 8 outputs as two float4s. ncu at 64K: L1/TEX 72.6%, SM 61.3%, DRAM 0.18%, 64 registers. Theoretical occupancy is 33.3%; smem limits it to 2 blocks/SM at 49 KB each.

v2 is v1 with 2 points per lane, loaded as float2. ncu: 563 µs vs 717 µs for v1 under the profiler. It uses 128 registers and 82 KB smem. smem limits it to 1 block/SM (registers would allow 2), for 16.7% theoretical occupancy. It was faster than v1 at lower occupancy.

Throughput estimate for v2 at 64K: about 99K FLOP/point from the three 64×64 hidden GEMMs on 4 jet components, so ~6.5 GFLOP in 0.418 ms ≈ 15.7 TFLOP/s. That ignores tanh and layer 0. The A40's FP32 peak is 37.4 TFLOP/s.

The 1K and 4K times sit at ~34 µs for v1 and v2. I'd guess this is per-call Python/launch overhead, but I haven't measured it.

Build notes:
- `nvcc` on the torch headers was OOM-killed at Slurm's default memory, so the job requests `--mem=16G`.
- Each job runs `build_ext`, which targets the node's arch (sm_86 so far, plus compute_86 PTX). On a node with a different arch I expect ninja to rebuild because the nvcc command line changes. Every node whose name I recorded has been sm_86 (A6000/A40/A5000), so that hasn't been tested.

## Run 4: backward via recompute (RTX A5000)

The autograd Function runs the fused kernel forward. Its backward rebuilds the jet in PyTorch from the packed parameters and calls `autograd.grad` on that.

| backend | 4K step ms | 16K step ms | 64K step ms |
|---|---|---|---|
| eager | 6.01 | 6.00 | 22.4 |
| forward Laplacian | 4.38 | 7.18 | 28.0 |
| forward + compile | 1.75 | 2.59 | 9.37 |
| fused fwd + recompute bwd | 5.32 | 7.39 | 28.6 |

Weight and coordinate gradients matched the forward Laplacian within atol 1e-5 / rtol 1e-4. A 50-step Adam run tracked it to rtol 1e-3. 110/110 tests passed.

With the recompute backward, the fused path trained at about the same speed as the plain forward Laplacian. I didn't profile the split between forward and backward in this run.

The forward Laplacian trained slower than eager at 64K (28.0 vs 22.4 ms) although its forward is faster. One possible reason is that its backward moves `[batch, 4, 64]` jet tensors where eager moves `[batch, 64]` activations. That's untested.

## Run 5: fused CUDA backward, v1 (A40)

| backend | 4K step ms | 16K step ms | 64K step ms |
|---|---|---|---|
| eager | 6.48 | 6.46 | 23.6 |
| forward + compile | 2.12 | 2.72 | 9.75 |
| fused, recompute bwd | 5.71 | 7.74 | 30.0 |
| fused, CUDA bwd v1 | 1.02 | 1.02 | 2.94 |

Kernel design: one persistent block per SM walks 32-point tiles. For each tile it:
- re-runs the forward pass and stores each layer's pre-activation jet in a per-block global scratch buffer;
- seeds the adjoint jet in smem from the linear output layer;
- for layers 3..1, applies the tanh-jet backward in place, recomputes the layer input from scratch, adds `Σ_p Σ_c zbar_c h_cᵀ` into register accumulators (a 4×4 block per thread), and computes `hbar = Wᵀ zbar`.

Layer 0 has its own closed form because its input jet is `(x, I, 0)`. Coordinate gradients come out of that step. Each block writes one row of a `[blocks, 12737]` partials buffer, and the host sums the rows. There are no atomics.

Parity against the recompute backward: max error ≤ 1e-4 × max |grad|, checked per weight/bias block, for batches 1, 31, 32, 33, 1K and 64K, and with weights scaled ×3 to saturate tanh. 120/120 tests passed.

torch profiler, one fused step at 64K: the backward kernel took 2424 µs and the forward kernel 421 µs, out of 2927 µs of total GPU kernel time. At 4K the backward kernel took 197 µs, while the step took ~1 ms wall time.

ncu at 64K (separate run, same GPU model): 2.97 ms, 229 registers/thread, 86.3 KB dynamic smem. Both registers and smem limit it to 1 block/SM, for 16.7% theoretical and achieved occupancy. 0.58 eligible warps per scheduler and 0.37 issued per cycle. DRAM 23%, L1/TEX 61%, SM 60%.

## Run 6: occupancy-oriented backward (v2) and CUDA-graphed training step (A40)

Training steps, `graph = yes` meaning the whole step is captured with `fused_hoad.training.cuda_graph_step` (Adam `capturable=True`):

| backend | graph | 4K step ms | 16K step ms | 64K step ms |
|---|---|---|---|---|
| eager | no | 6.49 | 6.45 | 23.55 |
| eager | yes | 2.01 | 5.28 | 23.11 |
| forward Laplacian | no | 4.69 | 7.54 | 29.46 |
| forward Laplacian | yes | 1.61 | 7.22 | 29.14 |
| forward + compile | no | 2.00 | 2.73 | 9.75 |
| fused, recompute bwd | no | 5.70 | 7.74 | 29.95 |
| fused, recompute bwd | yes | 1.69 | 7.39 | 29.61 |
| fused, CUDA bwd v1 | no | 1.006 | 1.003 | 2.917 |
| fused, CUDA bwd v1 | yes | 0.309 | 0.884 | 2.933 |
| fused, CUDA bwd v2 | no | 1.038 | 1.033 | 2.902 |
| fused, CUDA bwd v2 | yes | 0.325 | 0.842 | 2.907 |

Compile wasn't captured in the outer graph.

Backward v2 changes from v1:
- 16-point tiles: 36.9 KB smem.
- Weights read through L1 with `__ldg`. The forward GEMM uses a host-side transposed copy so its loads are float4.
- Weight gradients added into the block's own partials row (float4 read-modify-write, one owner thread per entry) after each layer, instead of being held in registers.
- `__launch_bounds__(256, 2)`.

ncu at 64K: 2.67 ms, 128 registers, 2 blocks/SM (both register and smem limits are 2), 33.3% theoretical / 29.6% achieved occupancy. DRAM 61%, L1/TEX 75%, SM 73%, L2 32%.

Under ncu, v2 took 2.67 ms against v1's 2.97 ms (from Run 5's separate ncu job). The step-level benchmark shows no difference at 64K: 2.902 vs 2.917 ms without a graph.

DRAM throughput went from 23% to 61%. I suspect the per-tile read-modify-write of the dW partials. That's 48 KB per 16-point tile per direction, and 2 × 84 rows of 51 KB don't fit in the 6 MB L2. I haven't confirmed it with memory counters. Two things to try: accumulate dW over several tiles before writing, or keep one layer's dW in smem.

`fused_cuda` now defaults to backward v2. Since the step times are the same, that choice doesn't matter much yet.

64-point backward tiles weren't built. Holding the forward jet and the adjoint jet in smem for 64 points takes 2 × 4 × 64 × 65 × 4 B ≈ 133 KB. sm_86 allows at most 99 KB of dynamic smem per block, and larger tiles also work against 2 blocks/SM.

The CUDA-graphed step is about 3.2× faster at 4K for both fused backward versions (1.04 → 0.33 ms for v2). It makes no difference at 64K. At 16K the gain is smaller (1.03 → 0.84 ms).

The graphed and non-graphed fused steps produce the same loss trajectory over 20 steps (rtol 1e-4) once both have run 3 warmup steps. Capturing before Adam has created its state would replay the state initialization on every step.

136/136 tests passed on the GPU.

## Corrections to earlier notes

- An earlier version of Run 1 said `torch.cat` was ~14% of the forward Laplacian's GPU time. The profile contained both eager and forward-Laplacian runs at 16K, and eager's `torch.stack` in `diagonal_hessian` also launches `CatArrayBatchedCopy`. So 14% is a share of total kernel time across both backends.
- An earlier version said Run 2 was on an RTX A6000. That job's output didn't print the GPU name, so the device is unknown.
- An earlier version of Run 5 gave 12.5% occupancy for backward v1. ncu reports 16.67% theoretical (8 of 48 warps), and registers (229/thread) also limit it to 1 block/SM, not just smem.
- An earlier version of Run 4 said the recompute backward was ~98% of the step. That was inferred from the totals, not measured.
- An earlier version of Run 2 called the compiled backends inference-only. Only the CUDA-graph wrappers are. `forward_laplacian_compiled` (Inductor default mode) trains, as Runs 4–6 show. The comment in `src/fused_hoad/backends/__init__.py` is fixed. `TRAINABLE_BACKENDS` still leaves out the compiled backend, because on CPU it falls back to the uncompiled function and the CPU gradient test wouldn't exercise compile.

## Things to try

- Backward v2: buffer dW partials across tiles to cut global read-modify-write traffic, then check DRAM in ncu again.
- Forward v2 sits at 1 block/SM with 82 KB smem. TF32 `mma.sync` for the 64×64 GEMMs hasn't been tried.
- Graph-capture the compiled path with `mode="reduce-overhead"` and compare it with the graphed fused step at 4K–16K.
