# Fused CUDA Accelerators for Higher-Order Automatic Differentiation

## Optimizing higher-order automatic differentiation (AD) workloads by eliminating graph traversal overhead and tensor materialization bottlenecks using custom CUDA kernel fusion.

### Overview:
Higher-order automatic differentiation in modern deep learning frameworks frequently suffers from severe performance degradation. While frameworks like PyTorch excel at first-order gradients, computing higher-order derivatives introduces repeated computational graph traversals, excessive tensor materialization, and severe CUDA kernel-launch overhead.
This project profiles these specific bottlenecks using Nvidia Nsight Systems and implements a highly optimized, fused CUDA residual kernel that bypasses framework-level abstractions to dramatically accelerate residual evaluation while preserving strict numerical accuracy.

### Performance Bottlenecks Identified (Nsight Systems)

• Repeated Graph Traversal: Framework overhead from recursively building and walking execution graphs for secondary and tertiary gradients.
• Tensor Materialization: Massive memory bandwidth saturation caused by allocating and writing intermediate, short-lived tensors to Global Memory (VRAM).
• Kernel-Launch Overhead: A high volume of small, sequential kernel launches that leave the GPU execution pipelines starved and latency-bound.

### Solution

To resolve these inefficiencies, this project, for a fixed MLP, replaces the native PyTorch higher-order AD pipeline with a custom C++/CUDA implementation:
• Kernel Fusion: Combines elementwise operations and residual evaluations into a single monolithic CUDA kernel, keeping intermediate data entirely within high-speed GPU registers and Shared Memory.
• Memory Optimization: Reduces redundant VRAM read/write cycles, shifting the workload from a memory-bandwidth-bound state to a compute-bound state.
• PyTorch C++ Extension: Exposes the optimized CUDA backend seamlessly to Python using torch.utils.cpp_extension, allowing it to drop directly into existing PyTorch training and inference scripts.

### Applications:

1. Physics-Informed Neural Networks (PINNs):
To enforce a PDE like the Navier-Stokes or Schrödinger equation, they must compute second, third, or fourth-order derivatives (\[u_{xx}\], \[u_{tttt}\]) of the network’s outputs with respect to the input coordinates. Doing this via native torch.autograd.grad(..., create_graph=True) creates a massive, deeply nested computational graph. It causes severe tensor materialization (allocating VRAM for intermediate derivatives at every layer) and floods the GPU with tiny elementwise kernel launches. Academic labs and industrial groups (like NVIDIA’s Modulus team) frequently write custom CUDA extensions or use Triton to fuse the final layers of the network directly with the PDE residual evaluation.
2. Modern Compiler Framework Teams (PyTorch Inductor & JAX/XLA):
If you look at the engineering blogs behind torch.compile, their primary goal is automating what you did manually. PyTorch Inductor uses AOTAutograd to trace execution graphs, isolate elementwise operators, and generate fused Triton kernels to eliminate the exact VRAM round-trips and launch overhead you identified.  JAX was built explicitly because PyTorch's eager mode struggled with higher-order differentiation. JAX converts functional code into intermediate expressions (Jaxprs) and hands them to the XLA compiler, which aggressively performs vertical and horizontal kernel fusion. Even then, JAX developers still write custom C++/CUDA loops when XLA fails to optimally fuse deeply recursive higher-order graphs.
3. Quantitative Finance & Risk Management Engineers
Computing "Speed" or "Gamma" requires evaluating second and third-order derivatives of a pricing model or a neural network approximation. • High-frequency trading (HFT) firms and quantitative desks at major banks write custom CUDA architectures to compute these higher-order Greeks in real-time. They cannot tolerate framework graph-traversal latencies, making your exact approach (C++ extensions, registers, and Shared Memory optimization) standard practice for production code.

## Repository layout

```text
GPU-Optimization/
├── benchmarks/          # Reproducible baseline and fused-kernel benchmarks
├── csrc/                # PyTorch C++ bindings and CUDA kernels
├── docs/                # Architecture and kernel-contract notes
├── scripts/             # Profiling entry points
├── src/fused_hoad/      # Python package and correctness baselines
└── tests/               # Numerical and extension-boundary tests
```

## Getting started

Requirements: Python 3.10+, PyTorch 2.2+, the CUDA toolkit, and an NVIDIA GPU.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]' --no-build-isolation
pytest
```

Run the eager Poisson-residual baseline with:

```bash
python -m benchmarks.benchmark_poisson
```

The benchmark uses a fixed `2 -> 64 -> 64 -> 64 -> 64 -> 1` tanh MLP and sweeps batches
of 1K, 4K, 16K, and 64K points. For a quick CPU smoke test, pass
`--device cpu --batch-sizes 32 --warmup 1 --iterations 2`.

The native CUDA operation is currently a buildable integration seam. Its final signature
will be implemented after the fixed MLP layout and target differential equation are
specified; see `docs/architecture.md`.

## Build order

The project is intentionally structured from stable primitives toward specialized code:

1. Lock the MLP contract in `specs.py` and construct it through `models.py`.
2. Verify reusable derivative operators in `autodiff.py`.
3. Express the target equation as a composition in `residuals.py`.
4. Treat the eager implementation as the numerical oracle.
5. Implement the matching fused CUDA path behind `_extension.py`.
6. Require correctness checks before benchmarking and profiling either path.

