#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <mma.h>

#include <cstdint>
#include <cstdlib>
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
// v3: 32-point tiles with the hidden 64x64 GEMMs on tensor cores (WMMA m16n16k8, TF32
//     inputs, FP32 accumulate); the tanh-jet epilogue stays FP32. Needs sm_80+.
// v4: v3 without the separate Z buffer (accumulators go back into H), so there is room to
//     double-buffer each layer's weights + bias with cp.async while the previous GEMM runs.
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


// ---- v3: TF32 tensor-core GEMMs ----
// Per hidden layer, Z[64 x 128] = W[64 x 64] * H[64 x 128], where the 128 columns are the
// 4 jet components x 32 points (column = component * 32 + point). Each of the 8 warps owns
// one 16-column strip and all 4 row tiles, so it loads each B fragment once per k-step.
constexpr int kTcTile = 32;
constexpr int kTcColumns = 4 * kTcTile;
constexpr int kTcLdh = kTcColumns + 4;  // padded leading dims, multiples of 4 for WMMA
constexpr int kTcLdw = kHidden + 4;
constexpr size_t kTcSharedBytes =
    (kHidden * kTcLdw + 2 * kHidden * kTcLdh) * sizeof(float);  // W, H, Z

__global__ void __launch_bounds__(256)
fused_forward_laplacian_tf32_kernel(const float* __restrict__ coordinates,
                                    const float* __restrict__ params,
                                    float* __restrict__ output,
                                    float* __restrict__ laplacian,
                                    int64_t batch) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  namespace wmma = nvcuda::wmma;
  extern __shared__ float shared[];
  float* w_s = shared;                       // [out][in], ld kTcLdw
  float* h_s = w_s + kHidden * kTcLdw;       // [neuron][component * 32 + point], ld kTcLdh
  float* z_s = h_s + kHidden * kTcLdh;       // same layout, pre-activations

  const int t = threadIdx.x;
  const int lane = t % kTcTile;
  const int warp = t / kTcTile;
  const int o0 = warp * 8;  // epilogue mapping: 8 neurons per warp, one point per lane
  const int64_t point = static_cast<int64_t>(blockIdx.x) * kTcTile + lane;
  const bool active = point < batch;
  const float x0 = active ? coordinates[2 * point] : 0.0f;
  const float x1 = active ? coordinates[2 * point + 1] : 0.0f;

  auto store_jet = [&](float* plane, int o, float v, float j0, float j1, float l) {
    plane[o * kTcLdh + lane] = v;
    plane[o * kTcLdh + kTcTile + lane] = j0;
    plane[o * kTcLdh + 2 * kTcTile + lane] = j1;
    plane[o * kTcLdh + 3 * kTcTile + lane] = l;
  };

#pragma unroll
  for (int k = 0; k < 8; ++k) {
    const int o = o0 + k;
    const float w0 = __ldg(params + o * kInputs);
    const float w1 = __ldg(params + o * kInputs + 1);
    const float zv = w0 * x0 + w1 * x1 + __ldg(params + kLayer0Bias + o);
    const float a = tanhf(zv);
    const float s = 1.0f - a * a;
    store_jet(h_s, o, a, s * w0, s * w1, -2.0f * a * s * (w0 * w0 + w1 * w1));
  }

  for (int layer = 0; layer < kHiddenLayers - 1; ++layer) {
    const float* weight = params + kHiddenBase + layer * kHiddenStride;
    const float* bias = weight + kHidden * kHidden;
    for (int k = t; k < kHidden * kHidden; k += blockDim.x) {
      w_s[(k / kHidden) * kTcLdw + k % kHidden] = __ldg(weight + k);
    }
    __syncthreads();  // weights staged, previous epilogue's H written

    wmma::fragment<wmma::accumulator, 16, 16, 8, float> acc[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      wmma::fill_fragment(acc[r], 0.0f);
    }
    const int column = warp * 16;
#pragma unroll
    for (int k = 0; k < kHidden; k += 8) {
      wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> b;
      wmma::load_matrix_sync(b, h_s + k * kTcLdh + column, kTcLdh);
#pragma unroll
      for (int e = 0; e < b.num_elements; ++e) {
        b.x[e] = wmma::__float_to_tf32(b.x[e]);
      }
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> a;
        wmma::load_matrix_sync(a, w_s + (r * 16) * kTcLdw + k, kTcLdw);
#pragma unroll
        for (int e = 0; e < a.num_elements; ++e) {
          a.x[e] = wmma::__float_to_tf32(a.x[e]);
        }
        wmma::mma_sync(acc[r], a, b, acc[r]);
      }
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      wmma::store_matrix_sync(z_s + (r * 16) * kTcLdh + column, acc[r], kTcLdh,
                              wmma::mem_row_major);
    }
    __syncthreads();  // Z complete; all reads of W and H for this layer done

#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const int o = o0 + k;
      const float* z = z_s + o * kTcLdh;
      const float zv = z[lane] + __ldg(bias + o);
      const float zj0 = z[kTcTile + lane];
      const float zj1 = z[2 * kTcTile + lane];
      const float zl = z[3 * kTcTile + lane];
      const float a = tanhf(zv);
      const float s = 1.0f - a * a;
      store_jet(h_s, o, a, s * zj0, s * zj1, s * zl - 2.0f * a * s * (zj0 * zj0 + zj1 * zj1));
    }
  }
  __syncthreads();

  if (t < kTcTile && active) {
    const float* w_out = params + kOutputBase;
    float u = __ldg(w_out + kHidden), lap = 0.0f;
    for (int i = 0; i < kHidden; ++i) {
      const float w = __ldg(w_out + i);
      u = fmaf(w, h_s[i * kTcLdh + lane], u);
      lap = fmaf(w, h_s[i * kTcLdh + 3 * kTcTile + lane], lap);
    }
    output[point] = u;
    laplacian[point] = lap;
  }
#endif
}


// ---- v4: pipelined TF32 ----
// Templated on the padded leading dimensions so different paddings can be compared.
template <int Ldw>
constexpr int pipe_weight_floats() {
  return kHidden * Ldw + kHidden;  // one weight buffer + its bias
}
template <int Ldh, int Ldw>
constexpr size_t pipe_shared_bytes() {
  return (2 * pipe_weight_floats<Ldw>() + kHidden * Ldh) * sizeof(float);
}

// Copy hidden layer `layer`'s weights (row-major, padded to kTcLdw) and bias into `buffer`
// with 16-byte cp.async transfers; one commit group per call.
template <int Ldw>
__device__ __forceinline__ void prefetch_layer(float* buffer, const float* params, int layer) {
  static_assert(Ldw % 4 == 0, "cp.async and WMMA need 16-byte row steps");
  const float* weight = params + kHiddenBase + layer * kHiddenStride;
  const float* bias = weight + kHidden * kHidden;
  for (int chunk = threadIdx.x; chunk < kHidden * kHidden / 4; chunk += blockDim.x) {
    const int row = chunk / (kHidden / 4);
    const int column = (chunk % (kHidden / 4)) * 4;
    __pipeline_memcpy_async(buffer + row * Ldw + column, weight + row * kHidden + column, 16);
  }
  for (int chunk = threadIdx.x; chunk < kHidden / 4; chunk += blockDim.x) {
    __pipeline_memcpy_async(buffer + kHidden * Ldw + chunk * 4, bias + chunk * 4, 16);
  }
  __pipeline_commit();
}

template <int Ldh, int Ldw>
__global__ void __launch_bounds__(256)
fused_forward_laplacian_tf32_pipelined_kernel(const float* __restrict__ coordinates,
                                              const float* __restrict__ params,
                                              float* __restrict__ output,
                                              float* __restrict__ laplacian,
                                              int64_t batch) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  namespace wmma = nvcuda::wmma;
  extern __shared__ float shared[];
  constexpr int kPwFloats = pipe_weight_floats<Ldw>();
  float* buffers[2] = {shared, shared + kPwFloats};
  float* h_s = shared + 2 * kPwFloats;  // [neuron][component * 32 + point], ld Ldh

  const int t = threadIdx.x;
  const int lane = t % kTcTile;
  const int warp = t / kTcTile;
  const int o0 = warp * 8;
  const int64_t point = static_cast<int64_t>(blockIdx.x) * kTcTile + lane;
  const bool active = point < batch;

  prefetch_layer<Ldw>(buffers[0], params, 0);  // overlaps with layer 0 below

  const float x0 = active ? coordinates[2 * point] : 0.0f;
  const float x1 = active ? coordinates[2 * point + 1] : 0.0f;
  auto store_jet = [&](int o, float v, float j0, float j1, float l) {
    float* row = h_s + o * Ldh;
    row[lane] = v;
    row[kTcTile + lane] = j0;
    row[2 * kTcTile + lane] = j1;
    row[3 * kTcTile + lane] = l;
  };
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    const int o = o0 + k;
    const float w0 = __ldg(params + o * kInputs);
    const float w1 = __ldg(params + o * kInputs + 1);
    const float zv = w0 * x0 + w1 * x1 + __ldg(params + kLayer0Bias + o);
    const float a = tanhf(zv);
    const float s = 1.0f - a * a;
    store_jet(o, a, s * w0, s * w1, -2.0f * a * s * (w0 * w0 + w1 * w1));
  }

  for (int layer = 0; layer < kHiddenLayers - 1; ++layer) {
    __syncthreads();  // H written by the previous epilogue; the buffer about to be refilled is free
    const bool has_next = layer + 1 < kHiddenLayers - 1;
    if (has_next) {
      prefetch_layer<Ldw>(buffers[(layer + 1) % 2], params, layer + 1);
      __pipeline_wait_prior(1);  // this layer's group is done; the next one may be in flight
    } else {
      __pipeline_wait_prior(0);
    }
    __syncthreads();  // every thread's copies for this layer are visible

    const float* w_s = buffers[layer % 2];
    const float* bias_s = w_s + kHidden * Ldw;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> acc[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      wmma::fill_fragment(acc[r], 0.0f);
    }
    const int column = warp * 16;
#pragma unroll
    for (int k = 0; k < kHidden; k += 8) {
      wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> b;
      wmma::load_matrix_sync(b, h_s + k * Ldh + column, Ldh);
#pragma unroll
      for (int e = 0; e < b.num_elements; ++e) {
        b.x[e] = wmma::__float_to_tf32(b.x[e]);
      }
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> a;
        wmma::load_matrix_sync(a, w_s + (r * 16) * Ldw + k, Ldw);
#pragma unroll
        for (int e = 0; e < a.num_elements; ++e) {
          a.x[e] = wmma::__float_to_tf32(a.x[e]);
        }
        wmma::mma_sync(acc[r], a, b, acc[r]);
      }
    }
    __syncthreads();  // every warp has finished reading H
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      wmma::store_matrix_sync(h_s + (r * 16) * Ldh + column, acc[r], Ldh,
                              wmma::mem_row_major);
    }
    __syncthreads();  // H now holds Z

#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const int o = o0 + k;
      const float* z = h_s + o * Ldh;
      const float zv = z[lane] + bias_s[o];
      const float zj0 = z[kTcTile + lane];
      const float zj1 = z[2 * kTcTile + lane];
      const float zl = z[3 * kTcTile + lane];
      const float a = tanhf(zv);
      const float s = 1.0f - a * a;
      store_jet(o, a, s * zj0, s * zj1, s * zl - 2.0f * a * s * (zj0 * zj0 + zj1 * zj1));
    }
  }
  __syncthreads();

  if (t < kTcTile && active) {
    const float* w_out = params + kOutputBase;
    float u = __ldg(w_out + kHidden), lap = 0.0f;
    for (int i = 0; i < kHidden; ++i) {
      const float w = __ldg(w_out + i);
      u = fmaf(w, h_s[i * Ldh + lane], u);
      lap = fmaf(w, h_s[i * Ldh + 3 * kTcTile + lane], lap);
    }
    output[point] = u;
    laplacian[point] = lap;
  }
#endif
}

template <int Ldh, int Ldw>
void launch_tf32_pipelined(const torch::Tensor& coordinates, const torch::Tensor& parameters,
                           torch::Tensor& output, torch::Tensor& laplacian, int64_t batch,
                           cudaStream_t stream) {
  auto kernel = fused_forward_laplacian_tf32_pipelined_kernel<Ldh, Ldw>;
  constexpr size_t bytes = pipe_shared_bytes<Ldh, Ldw>();
  C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      static_cast<int>(bytes)));
  const int blocks = static_cast<int>((batch + kTcTile - 1) / kTcTile);
  kernel<<<blocks, 256, bytes, stream>>>(coordinates.data_ptr<float>(),
                                         parameters.data_ptr<float>(), output.data_ptr<float>(),
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
  } else if (version == 2) {
    launch_tiled<2>(coordinates, parameters, output, laplacian, batch, stream);
  } else if (version == 4) {
    const auto* props = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(props->major >= 8, "forward v4 (TF32 tensor cores) needs sm_80 or newer");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(parameters.data_ptr<float>()) % 16 == 0,
                "parameters must be 16-byte aligned for forward v4 (cp.async)");
    // FUSED_HOAD_TC_PAD picks the smem padding (experiment knob, read per call):
    // 0 = ld 132/68 (default), 1 = 136/72, 2 = 128/64 (none), 3 = 140/76, 4 = 144/80.
    const char* pad_env = std::getenv("FUSED_HOAD_TC_PAD");
    const int pad = pad_env ? std::atoi(pad_env) : 0;
    switch (pad) {
      case 1: launch_tf32_pipelined<136, 72>(coordinates, parameters, output, laplacian, batch, stream); break;
      case 2: launch_tf32_pipelined<128, 64>(coordinates, parameters, output, laplacian, batch, stream); break;
      case 3: launch_tf32_pipelined<140, 76>(coordinates, parameters, output, laplacian, batch, stream); break;
      case 4: launch_tf32_pipelined<144, 80>(coordinates, parameters, output, laplacian, batch, stream); break;
      default: launch_tf32_pipelined<132, 68>(coordinates, parameters, output, laplacian, batch, stream);
    }
  } else {
    const auto* props = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(props->major >= 8, "forward v3 (TF32 tensor cores) needs sm_80 or newer");
    C10_CUDA_CHECK(cudaFuncSetAttribute(fused_forward_laplacian_tf32_kernel,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(kTcSharedBytes)));
    const int blocks = static_cast<int>((batch + kTcTile - 1) / kTcTile);
    fused_forward_laplacian_tf32_kernel<<<blocks, 256, kTcSharedBytes, stream>>>(
        coordinates.data_ptr<float>(), parameters.data_ptr<float>(), output.data_ptr<float>(),
        laplacian.data_ptr<float>(), batch);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, laplacian};
}
