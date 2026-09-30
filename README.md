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

To resolve these inefficiencies, this project replaces the native PyTorch higher-order AD pipeline with a custom C++/CUDA implementation:
• Kernel Fusion: Combines elementwise operations and residual evaluations into a single monolithic CUDA kernel, keeping intermediate data entirely within high-speed GPU registers and Shared Memory.
• Memory Optimization: Reduces redundant VRAM read/write cycles, shifting the workload from a memory-bandwidth-bound state to a compute-bound state.
• PyTorch C++ Extension: Exposes the optimized CUDA backend seamlessly to Python using torch.utils.cpp_extension, allowing it to drop directly into existing PyTorch training and inference scripts.


