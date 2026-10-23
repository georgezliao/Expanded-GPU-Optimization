#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <vector>

#include "mlp_abi.cuh"

// Forward-mode Laplacian for the fixed 2 -> 64x4 -> 1 tanh MLP.
//
// Each point carries a second-order jet through the network (same math as
// src/fused_hoad/backends/forward_laplacian.py):
//   v : activations          J0, J1 : d v / d x_d          L : sum_d d^2 v / d x_d^2
// Linear:  z_v = W v + b,  z_J = W J,  z_L = W L
// tanh:    a = tanh(z_v), s = 1 - a^2
//          v = a,  J = s z_J,  L = s z_L - 2 a s (z_J0^2 + z_J1^2)
//
// v0: one thread per point, weights read through the read-only cache. Correctness first;
//     the per-thread jet (2 KB) spills to local memory.
// v1: a block owns a tile of 32 points. Jets live in shared memory as [component][neuron]
//     [point]; each warp owns 8 neurons and each lane one point, so every layer is a small
//     register-tiled GEMM against the layer's weights staged in shared memory.
// v2: v1 with 2 points per lane (64-point tiles): float2 jet loads halve shared-memory
//     instructions per FMA, at the cost of 80 KB shared memory per block.

namespace {

using namespace fused_hoad;

constexpr int kThreads = 128;

struct Jet {
  float v[kHidden];
  float j0[kHidden];
  float j1[kHidden];
  float l[kHidden];
};

__device__ __forceinline__ void tanh_jet(float z, float zj0, float zj1, float zl,
                                         Jet& out, int o) {
  const float a = tanhf(z);
  const float s = 1.0f - a * a;
  out.v[o] = a;
  out.j0[o] = s * zj0;
  out.j1[o] = s * zj1;
  out.l[o] = s * zl - 2.0f * a * s * (zj0 * zj0 + zj1 * zj1);
}

__global__ void __launch_bounds__(kThreads)
fused_forward_laplacian_v0_kernel(const float* __restrict__ coordinates,
                               const float* __restrict__ params,
                               float* __restrict__ output,
                               float* __restrict__ laplacian,
                               int64_t batch) {
  const int64_t point = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (point >= batch) {
    return;
  }

  const float x0 = coordinates[2 * point];
  const float x1 = coordinates[2 * point + 1];

  Jet buffers[2];
  Jet* in = &buffers[0];
  Jet* out = &buffers[1];

  // Layer 0: J = W columns and L = 0 because the input jet is (x, I, 0).
  for (int o = 0; o < kHidden; ++o) {
    const float w0 = __ldg(params + o * kInputs);
    const float w1 = __ldg(params + o * kInputs + 1);
    const float z = w0 * x0 + w1 * x1 + __ldg(params + kLayer0Bias + o);
    tanh_jet(z, w0, w1, 0.0f, *in, o);
  }

  for (int layer = 0; layer < kHiddenLayers - 1; ++layer) {
    const float* weight = params + kHiddenBase + layer * kHiddenStride;
    const float* bias = weight + kHidden * kHidden;
    for (int o = 0; o < kHidden; ++o) {
      float z = __ldg(bias + o), zj0 = 0.0f, zj1 = 0.0f, zl = 0.0f;
#pragma unroll 16
      for (int i = 0; i < kHidden; ++i) {
        const float w = __ldg(weight + o * kHidden + i);
        z = fmaf(w, in->v[i], z);
        zj0 = fmaf(w, in->j0[i], zj0);
        zj1 = fmaf(w, in->j1[i], zj1);
        zl = fmaf(w, in->l[i], zl);
      }
      tanh_jet(z, zj0, zj1, zl, *out, o);
    }
    Jet* swap = in;
    in = out;
    out = swap;
  }

  // Output layer is linear, so its Laplacian is just W L.
  const float* weight = params + kOutputBase;
  float u = __ldg(weight + kHidden), lap = 0.0f;
#pragma unroll 16
  for (int i = 0; i < kHidden; ++i) {
    const float w = __ldg(weight + i);
    u = fmaf(w, in->v[i], u);
    lap = fmaf(w, in->l[i], lap);
  }
  output[point] = u;
  laplacian[point] = lap;
}

constexpr int kTileThreads = 256;
constexpr int kNeuronsPerThread = kHidden / (kTileThreads / 32);
static_assert(kNeuronsPerThread == 8, "each warp must own 8 neurons");

// A block owns 32 * kPoints points; lane l owns the kPoints adjacent columns starting at
// l * kPoints, so jet reads are float (v1) or float2 (v2) vector loads.
template <int kPoints>
struct Tile {
  static constexpr int kSize = 32 * kPoints;
  static constexpr int kPlane = kHidden * kSize;
  static constexpr size_t kSharedBytes = (4 * kPlane + kHidden * kHidden) * sizeof(float);
};

template <int kPoints>
__device__ __forceinline__ void load_points(const float* source, float (&target)[kPoints]) {
  if constexpr (kPoints == 1) {
    target[0] = source[0];
  } else {
    static_assert(kPoints == 2, "only 1 or 2 points per lane");
    const float2 pair = *reinterpret_cast<const float2*>(source);
    target[0] = pair.x;
    target[1] = pair.y;
  }
}

template <int kPoints>
__device__ __forceinline__ void store_tanh(float* jet, float z, float zj0, float zj1, float zl,
                                           int index) {
  constexpr int kPlane = Tile<kPoints>::kPlane;
  const float a = tanhf(z);
  const float s = 1.0f - a * a;
  jet[index] = a;
  jet[kPlane + index] = s * zj0;
  jet[2 * kPlane + index] = s * zj1;
  jet[3 * kPlane + index] = s * zl - 2.0f * a * s * (zj0 * zj0 + zj1 * zj1);
}

template <int kPoints>
__global__ void __launch_bounds__(kTileThreads)
fused_forward_laplacian_tiled_kernel(const float* __restrict__ coordinates,
                                     const float* __restrict__ params,
                                     float* __restrict__ output,
                                     float* __restrict__ laplacian,
                                     int64_t batch) {
  using T = Tile<kPoints>;
  extern __shared__ float shared[];
  // Jet planes v, J0, J1, L, each [neuron][point]; then weights staged transposed,
  // [input][output], so a warp reads its 8 outputs as 2 float4s.
  float* jet = shared;
  float* weight_t = shared + 4 * T::kPlane;

  const int column = (threadIdx.x % 32) * kPoints;
  const int o0 = (threadIdx.x / 32) * kNeuronsPerThread;
  const int64_t first_point = static_cast<int64_t>(blockIdx.x) * T::kSize + column;

  float x0[kPoints], x1[kPoints];
#pragma unroll
  for (int q = 0; q < kPoints; ++q) {
    const bool active = first_point + q < batch;
    x0[q] = active ? coordinates[2 * (first_point + q)] : 0.0f;
    x1[q] = active ? coordinates[2 * (first_point + q) + 1] : 0.0f;
  }

#pragma unroll
  for (int k = 0; k < kNeuronsPerThread; ++k) {
    const int o = o0 + k;
    const float w0 = __ldg(params + o * kInputs);
    const float w1 = __ldg(params + o * kInputs + 1);
    const float b = __ldg(params + kLayer0Bias + o);
#pragma unroll
    for (int q = 0; q < kPoints; ++q) {
      store_tanh<kPoints>(jet, w0 * x0[q] + w1 * x1[q] + b, w0, w1, 0.0f,
                          o * T::kSize + column + q);
    }
  }

  for (int layer = 0; layer < kHiddenLayers - 1; ++layer) {
    const float* weight = params + kHiddenBase + layer * kHiddenStride;
    const float* bias = weight + kHidden * kHidden;
    for (int k = threadIdx.x; k < kHidden * kHidden; k += kTileThreads) {
      const int i = k / kHidden;
      const int o = k % kHidden;
      weight_t[k] = __ldg(weight + o * kHidden + i);
    }
    __syncthreads();  // weights staged and the previous layer's jet fully written

    float zv[kPoints][kNeuronsPerThread], zj0[kPoints][kNeuronsPerThread],
        zj1[kPoints][kNeuronsPerThread], zl[kPoints][kNeuronsPerThread];
#pragma unroll
    for (int k = 0; k < kNeuronsPerThread; ++k) {
      const float b = __ldg(bias + o0 + k);
#pragma unroll
      for (int q = 0; q < kPoints; ++q) {
        zv[q][k] = b;
        zj0[q][k] = zj1[q][k] = zl[q][k] = 0.0f;
      }
    }

#pragma unroll 4
    for (int i = 0; i < kHidden; ++i) {
      const int index = i * T::kSize + column;
      float hv[kPoints], hj0[kPoints], hj1[kPoints], hl[kPoints];
      load_points<kPoints>(jet + index, hv);
      load_points<kPoints>(jet + T::kPlane + index, hj0);
      load_points<kPoints>(jet + 2 * T::kPlane + index, hj1);
      load_points<kPoints>(jet + 3 * T::kPlane + index, hl);
      const float4 wa = *reinterpret_cast<const float4*>(weight_t + i * kHidden + o0);
      const float4 wb = *reinterpret_cast<const float4*>(weight_t + i * kHidden + o0 + 4);
      const float w[kNeuronsPerThread] = {wa.x, wa.y, wa.z, wa.w, wb.x, wb.y, wb.z, wb.w};
#pragma unroll
      for (int q = 0; q < kPoints; ++q) {
#pragma unroll
        for (int k = 0; k < kNeuronsPerThread; ++k) {
          zv[q][k] = fmaf(w[k], hv[q], zv[q][k]);
          zj0[q][k] = fmaf(w[k], hj0[q], zj0[q][k]);
          zj1[q][k] = fmaf(w[k], hj1[q], zj1[q][k]);
          zl[q][k] = fmaf(w[k], hl[q], zl[q][k]);
        }
      }
    }
    __syncthreads();  // every read of this layer's jet and weights is done; update in place

#pragma unroll
    for (int k = 0; k < kNeuronsPerThread; ++k) {
#pragma unroll
      for (int q = 0; q < kPoints; ++q) {
        store_tanh<kPoints>(jet, zv[q][k], zj0[q][k], zj1[q][k], zl[q][k],
                            (o0 + k) * T::kSize + column + q);
      }
    }
  }
  __syncthreads();

  // Output layer is linear, so its Laplacian is just W L. One warp finishes the tile.
  if (threadIdx.x < 32) {
    const float* weight = params + kOutputBase;
    float u[kPoints], lap[kPoints];
#pragma unroll
    for (int q = 0; q < kPoints; ++q) {
      u[q] = __ldg(weight + kHidden);
      lap[q] = 0.0f;
    }
#pragma unroll 16
    for (int i = 0; i < kHidden; ++i) {
      const float w = __ldg(weight + i);
#pragma unroll
      for (int q = 0; q < kPoints; ++q) {
        u[q] = fmaf(w, jet[i * T::kSize + column + q], u[q]);
        lap[q] = fmaf(w, jet[3 * T::kPlane + i * T::kSize + column + q], lap[q]);
      }
    }
#pragma unroll
    for (int q = 0; q < kPoints; ++q) {
      if (first_point + q < batch) {
        output[first_point + q] = u[q];
        laplacian[first_point + q] = lap[q];
      }
    }
  }
}

template <int kPoints>
void launch_tiled(const torch::Tensor& coordinates, const torch::Tensor& parameters,
                  torch::Tensor& output, torch::Tensor& laplacian, int64_t batch,
                  cudaStream_t stream) {
  using T = Tile<kPoints>;
  auto kernel = fused_forward_laplacian_tiled_kernel<kPoints>;
  if constexpr (T::kSharedBytes > 48 * 1024) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(T::kSharedBytes)));
  }
  const int blocks = static_cast<int>((batch + T::kSize - 1) / T::kSize);
  kernel<<<blocks, kTileThreads, T::kSharedBytes, stream>>>(
      coordinates.data_ptr<float>(), parameters.data_ptr<float>(), output.data_ptr<float>(),
      laplacian.data_ptr<float>(), batch);
}

}  // namespace

std::vector<torch::Tensor> fused_forward_laplacian_cuda(torch::Tensor coordinates,
                                                        torch::Tensor parameters,
                                                        int64_t version) {
  const c10::cuda::CUDAGuard device_guard(coordinates.device());
  const int64_t batch = coordinates.size(0);
  auto output = torch::empty({batch, 1}, coordinates.options());
  auto laplacian = torch::empty({batch, 1}, coordinates.options());
  if (batch == 0) {
    return {output, laplacian};
  }

  const auto stream = at::cuda::getCurrentCUDAStream();
  if (version == 0) {
    const int blocks = static_cast<int>((batch + kThreads - 1) / kThreads);
    fused_forward_laplacian_v0_kernel<<<blocks, kThreads, 0, stream>>>(
        coordinates.data_ptr<float>(), parameters.data_ptr<float>(), output.data_ptr<float>(),
        laplacian.data_ptr<float>(), batch);
  } else if (version == 1) {
    launch_tiled<1>(coordinates, parameters, output, laplacian, batch, stream);
  } else {
    launch_tiled<2>(coordinates, parameters, output, laplacian, batch, stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, laplacian};
}
