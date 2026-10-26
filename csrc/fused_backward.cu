#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "mlp_abi.cuh"

// Reverse-mode pass through the forward-Laplacian jet of the fixed tanh MLP.
//
// Given dLoss/du and dLoss/dLaplacian per point, computes dLoss/dparameters and
// dLoss/dcoordinates in one launch. A persistent grid walks 32-point tiles:
//   1. Re-run the tile's forward pass, saving each layer's pre-activation jet
//      z = (z_v, z_J0, z_J1, z_L) to a per-block global scratch area.
//   2. Seed the adjoint jet from the linear output layer, then for layers 3..1:
//        tanh backward (in place)      zbar = act'(z)^T hbar
//        weight gradient (registers)   dW  += sum_points sum_c zbar_c h_c^T
//        input adjoint                  hbar = W^T zbar
//   3. Layer 0 gets its own closed form because its input jet is (x, I, 0).
// Each block accumulates weight gradients across all of its tiles and writes one row of
// a [blocks, parameters] partials buffer, which the host sums. No atomics.
//
// tanh jet backward, with a = tanh(z_v), s = 1 - a^2, s2 = -2 a s, s3 = -2 s^2 + 4 a^2 s,
// q = z_J0^2 + z_J1^2 (forward: v = a, J = s z_J, L = s z_L + s2 q):
//   zbar_v  = vbar s + (Jbar . z_J) s2 + Lbar (z_L s2 + q s3)
//   zbar_Jd = Jbar_d s + 2 Lbar s2 z_Jd
//   zbar_L  = Lbar s

namespace {

using namespace fused_hoad;

constexpr int kTile = 32;
constexpr int kThreads = 256;
constexpr int kNeuronsPerThread = kHidden / (kThreads / kTile);  // jet mapping: warp -> 8 neurons
constexpr int kStride = kTile + 1;           // pad [neuron][point] rows to avoid bank conflicts
constexpr int kPlane = kHidden * kStride;    // one jet component in shared memory
constexpr int kScratchPlane = kHidden * kTile;
constexpr int kScratchLayer = 4 * kScratchPlane;
constexpr int kScratchPerBlock = kHiddenLayers * kScratchLayer;

// Small gradients accumulated in shared memory: W0, biases of layers 0..3, w_out, b_out.
constexpr int kSmallW0 = 0;
constexpr int kSmallBias = kSmallW0 + kHidden * kInputs;
constexpr int kSmallWout = kSmallBias + kHiddenLayers * kHidden;
constexpr int kSmallBout = kSmallWout + kHidden;
constexpr int kSmallCount = kSmallBout + 4;

constexpr int kSharedFloats =
    kHidden * kHidden + 8 * kPlane + kSmallCount + 4 * kTile;  // weights, adj, hid, small, io
constexpr size_t kSharedBytes = kSharedFloats * sizeof(float);
static_assert(kNeuronsPerThread == 8, "each warp must own 8 neurons");
static_assert(kThreads == 16 * 16, "weight-gradient mapping uses 16x16 blocks of 4x4");

__device__ __forceinline__ int scratch_index(int layer, int component, int neuron, int point) {
  return ((layer * 4 + component) * kHidden + neuron) * kTile + point;
}

__device__ __forceinline__ void tanh_forward(float zv, float zj0, float zj1, float zl, float& v,
                                             float& j0, float& j1, float& l) {
  const float a = tanhf(zv);
  const float s = 1.0f - a * a;
  v = a;
  j0 = s * zj0;
  j1 = s * zj1;
  l = s * zl - 2.0f * a * s * (zj0 * zj0 + zj1 * zj1);
}

__device__ __forceinline__ void tanh_backward(float zv, float zj0, float zj1, float zl,
                                              float& av, float& aj0, float& aj1, float& al) {
  const float a = tanhf(zv);
  const float s = 1.0f - a * a;
  const float s2 = -2.0f * a * s;
  const float s3 = -2.0f * s * s + 4.0f * a * a * s;
  const float q = zj0 * zj0 + zj1 * zj1;
  const float gv = av * s + (aj0 * zj0 + aj1 * zj1) * s2 + al * (zl * s2 + q * s3);
  const float g0 = aj0 * s + 2.0f * al * s2 * zj0;
  const float g1 = aj1 * s + 2.0f * al * s2 * zj1;
  av = gv;
  aj0 = g0;
  aj1 = g1;
  al = al * s;
}

__global__ void __launch_bounds__(kThreads, 1)
fused_forward_laplacian_backward_kernel(const float* __restrict__ coordinates,
                                        const float* __restrict__ params,
                                        const float* __restrict__ grad_output,
                                        const float* __restrict__ grad_laplacian,
                                        float* __restrict__ grad_coordinates,
                                        float* __restrict__ partials,
                                        float* __restrict__ scratch,
                                        int64_t batch,
                                        int64_t tiles) {
  extern __shared__ float shared[];
  float* weights = shared;                       // staged layer weights (float4-aligned)
  float* adj = weights + kHidden * kHidden;      // adjoint jet, 4 planes [neuron][point]
  float* hid = adj + 4 * kPlane;                 // forward jet, 4 planes [neuron][point]
  float* small = hid + 4 * kPlane;
  float* s_gu = small + kSmallCount;
  float* s_gl = s_gu + kTile;
  float* s_x0 = s_gl + kTile;
  float* s_x1 = s_x0 + kTile;
  float* block_scratch = scratch + static_cast<int64_t>(blockIdx.x) * kScratchPerBlock;

  const int t = threadIdx.x;
  const int lane = t % kTile;
  const int o0 = (t / kTile) * kNeuronsPerThread;  // jet mapping: neurons o0..o0+7, point lane
  const int wo = (t / 16) * 4;                     // weight-gradient mapping: 4x4 block
  const int wi = (t % 16) * 4;

  for (int k = t; k < kSmallCount; k += kThreads) {
    small[k] = 0.0f;
  }
  float dw[kHiddenLayers - 1][16];
#pragma unroll
  for (int l = 0; l < kHiddenLayers - 1; ++l) {
#pragma unroll
    for (int k = 0; k < 16; ++k) {
      dw[l][k] = 0.0f;
    }
  }

  for (int64_t tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
    const int64_t point = tile * kTile + lane;
    const bool active = point < batch;
    __syncthreads();  // previous tile is done with every shared buffer
    if (t < kTile) {
      s_x0[t] = active ? coordinates[2 * point] : 0.0f;
      s_x1[t] = active ? coordinates[2 * point + 1] : 0.0f;
      s_gu[t] = active ? grad_output[point] : 0.0f;
      s_gl[t] = active ? grad_laplacian[point] : 0.0f;
    }
    __syncthreads();

    // ---- 1. Forward pass, saving pre-activations ----
    {
      const float x0 = s_x0[lane];
      const float x1 = s_x1[lane];
#pragma unroll
      for (int k = 0; k < kNeuronsPerThread; ++k) {
        const int o = o0 + k;
        const float w0 = __ldg(params + o * kInputs);
        const float w1 = __ldg(params + o * kInputs + 1);
        const float zv = w0 * x0 + w1 * x1 + __ldg(params + kLayer0Bias + o);
        block_scratch[scratch_index(0, 0, o, lane)] = zv;
        block_scratch[scratch_index(0, 1, o, lane)] = w0;
        block_scratch[scratch_index(0, 2, o, lane)] = w1;
        block_scratch[scratch_index(0, 3, o, lane)] = 0.0f;
        const int h = o * kStride + lane;
        tanh_forward(zv, w0, w1, 0.0f, hid[h], hid[kPlane + h], hid[2 * kPlane + h],
                     hid[3 * kPlane + h]);
      }
    }

    for (int layer = 1; layer < kHiddenLayers; ++layer) {
      const float* weight = params + kHiddenBase + (layer - 1) * kHiddenStride;
      const float* bias = weight + kHidden * kHidden;
      for (int k = t; k < kHidden * kHidden; k += kThreads) {
        weights[k] = __ldg(weight + (k % kHidden) * kHidden + k / kHidden);  // [in][out]
      }
      __syncthreads();

      float zv[8], zj0[8], zj1[8], zl[8];
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        zv[k] = __ldg(bias + o0 + k);
        zj0[k] = zj1[k] = zl[k] = 0.0f;
      }
#pragma unroll 4
      for (int i = 0; i < kHidden; ++i) {
        const int h = i * kStride + lane;
        const float hv = hid[h], hj0 = hid[kPlane + h], hj1 = hid[2 * kPlane + h],
                    hl = hid[3 * kPlane + h];
        const float4 wa = *reinterpret_cast<const float4*>(weights + i * kHidden + o0);
        const float4 wb = *reinterpret_cast<const float4*>(weights + i * kHidden + o0 + 4);
        const float w[8] = {wa.x, wa.y, wa.z, wa.w, wb.x, wb.y, wb.z, wb.w};
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          zv[k] = fmaf(w[k], hv, zv[k]);
          zj0[k] = fmaf(w[k], hj0, zj0[k]);
          zj1[k] = fmaf(w[k], hj1, zj1[k]);
          zl[k] = fmaf(w[k], hl, zl[k]);
        }
      }
      __syncthreads();  // all reads of hid and weights done

#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const int o = o0 + k;
        block_scratch[scratch_index(layer, 0, o, lane)] = zv[k];
        block_scratch[scratch_index(layer, 1, o, lane)] = zj0[k];
        block_scratch[scratch_index(layer, 2, o, lane)] = zj1[k];
        block_scratch[scratch_index(layer, 3, o, lane)] = zl[k];
        const int h = o * kStride + lane;
        tanh_forward(zv[k], zj0[k], zj1[k], zl[k], hid[h], hid[kPlane + h],
                     hid[2 * kPlane + h], hid[3 * kPlane + h]);
      }
    }
    __syncthreads();  // hid now holds the last hidden jet

    // ---- 2a. Output layer: u = w.v + b, lap = w.L ----
    const float* w_out = params + kOutputBase;
    if (t < kHidden) {
      float acc = 0.0f;
      for (int p = 0; p < kTile; ++p) {
        acc += s_gu[p] * hid[t * kStride + p] + s_gl[p] * hid[3 * kPlane + t * kStride + p];
      }
      small[kSmallWout + t] += acc;
    }
    if (t == kHidden) {
      float acc = 0.0f;
      for (int p = 0; p < kTile; ++p) {
        acc += s_gu[p];
      }
      small[kSmallBout] += acc;
    }
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const int i = o0 + k;
      const float w = __ldg(w_out + i);
      const int a = i * kStride + lane;
      adj[a] = s_gu[lane] * w;
      adj[kPlane + a] = 0.0f;
      adj[2 * kPlane + a] = 0.0f;
      adj[3 * kPlane + a] = s_gl[lane] * w;
    }
    __syncthreads();  // output-layer reads of hid are done before hid is recomputed

    // ---- 2b. Hidden layers 3..1 ----
#pragma unroll
    for (int layer = kHiddenLayers - 1; layer >= 1; --layer) {
      const float* weight = params + kHiddenBase + (layer - 1) * kHiddenStride;

      // tanh backward in place, and recompute this layer's input jet from saved z.
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const int o = o0 + k;
        const int a = o * kStride + lane;
        tanh_backward(block_scratch[scratch_index(layer, 0, o, lane)],
                      block_scratch[scratch_index(layer, 1, o, lane)],
                      block_scratch[scratch_index(layer, 2, o, lane)],
                      block_scratch[scratch_index(layer, 3, o, lane)], adj[a],
                      adj[kPlane + a], adj[2 * kPlane + a], adj[3 * kPlane + a]);
        tanh_forward(block_scratch[scratch_index(layer - 1, 0, o, lane)],
                     block_scratch[scratch_index(layer - 1, 1, o, lane)],
                     block_scratch[scratch_index(layer - 1, 2, o, lane)],
                     block_scratch[scratch_index(layer - 1, 3, o, lane)], hid[a],
                     hid[kPlane + a], hid[2 * kPlane + a], hid[3 * kPlane + a]);
      }
      for (int k = t; k < kHidden * kHidden; k += kThreads) {
        weights[k] = __ldg(weight + k);  // [out][in]
      }
      __syncthreads();

      // dW[o][i] += sum_p sum_c zbar_c[o][p] h_c[i][p], 4x4 register block per thread.
#pragma unroll 2
      for (int p = 0; p < kTile; ++p) {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          float zb[4], hb[4];
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            zb[k] = adj[c * kPlane + (wo + k) * kStride + p];
            hb[k] = hid[c * kPlane + (wi + k) * kStride + p];
          }
#pragma unroll
          for (int k = 0; k < 4; ++k) {
#pragma unroll
            for (int m = 0; m < 4; ++m) {
              dw[layer - 1][k * 4 + m] = fmaf(zb[k], hb[m], dw[layer - 1][k * 4 + m]);
            }
          }
        }
      }
      if (t < kHidden) {
        float acc = 0.0f;
        for (int p = 0; p < kTile; ++p) {
          acc += adj[t * kStride + p];
        }
        small[kSmallBias + layer * kHidden + t] += acc;
      }

      // hbar[i][p] = sum_o W[o][i] zbar[o][p] for this thread's 8 inputs.
      float gv[8], gj0[8], gj1[8], gl[8];
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        gv[k] = gj0[k] = gj1[k] = gl[k] = 0.0f;
      }
#pragma unroll 4
      for (int o = 0; o < kHidden; ++o) {
        const int a = o * kStride + lane;
        const float av = adj[a], aj0 = adj[kPlane + a], aj1 = adj[2 * kPlane + a],
                    al = adj[3 * kPlane + a];
        const float4 wa = *reinterpret_cast<const float4*>(weights + o * kHidden + o0);
        const float4 wb = *reinterpret_cast<const float4*>(weights + o * kHidden + o0 + 4);
        const float w[8] = {wa.x, wa.y, wa.z, wa.w, wb.x, wb.y, wb.z, wb.w};
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          gv[k] = fmaf(w[k], av, gv[k]);
          gj0[k] = fmaf(w[k], aj0, gj0[k]);
          gj1[k] = fmaf(w[k], aj1, gj1[k]);
          gl[k] = fmaf(w[k], al, gl[k]);
        }
      }
      __syncthreads();  // every read of adj, hid and weights for this layer is done

#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const int a = (o0 + k) * kStride + lane;
        adj[a] = gv[k];
        adj[kPlane + a] = gj0[k];
        adj[2 * kPlane + a] = gj1[k];
        adj[3 * kPlane + a] = gl[k];
      }
    }

    // ---- 3. Layer 0: z_v = W0 x + b0, z_J = W0, z_L = 0 ----
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const int o = o0 + k;
      const int a = o * kStride + lane;
      tanh_backward(block_scratch[scratch_index(0, 0, o, lane)],
                    block_scratch[scratch_index(0, 1, o, lane)],
                    block_scratch[scratch_index(0, 2, o, lane)],
                    block_scratch[scratch_index(0, 3, o, lane)], adj[a], adj[kPlane + a],
                    adj[2 * kPlane + a], adj[3 * kPlane + a]);
    }
    __syncthreads();

    if (t < kHidden) {
      float gb = 0.0f, g0 = 0.0f, g1 = 0.0f;
      for (int p = 0; p < kTile; ++p) {
        const int a = t * kStride + p;
        const float zbar_v = adj[a];
        gb += zbar_v;
        g0 += zbar_v * s_x0[p] + adj[kPlane + a];
        g1 += zbar_v * s_x1[p] + adj[2 * kPlane + a];
      }
      small[kSmallBias + t] += gb;
      small[kSmallW0 + t * kInputs] += g0;
      small[kSmallW0 + t * kInputs + 1] += g1;
    }
    if (t < kTile && active) {
      float gx0 = 0.0f, gx1 = 0.0f;
      for (int o = 0; o < kHidden; ++o) {
        const float zbar_v = adj[o * kStride + lane];
        gx0 = fmaf(__ldg(params + o * kInputs), zbar_v, gx0);
        gx1 = fmaf(__ldg(params + o * kInputs + 1), zbar_v, gx1);
      }
      grad_coordinates[2 * point] = gx0;
      grad_coordinates[2 * point + 1] = gx1;
    }
  }
  __syncthreads();

  // ---- Write this block's partial gradient row in the packed ABI layout ----
  float* row = partials + static_cast<int64_t>(blockIdx.x) * kParameterCount;
  for (int k = t; k < kHidden * kInputs; k += kThreads) {
    row[k] = small[kSmallW0 + k];
  }
  if (t < kHidden) {
    row[kLayer0Bias + t] = small[kSmallBias + t];
    for (int layer = 1; layer < kHiddenLayers; ++layer) {
      row[kHiddenBase + (layer - 1) * kHiddenStride + kHidden * kHidden + t] =
          small[kSmallBias + layer * kHidden + t];
    }
    row[kOutputBase + t] = small[kSmallWout + t];
  }
  if (t == 0) {
    row[kOutputBase + kHidden] = small[kSmallBout];
  }
#pragma unroll
  for (int layer = 0; layer < kHiddenLayers - 1; ++layer) {
    float* weight_row = row + kHiddenBase + layer * kHiddenStride;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
#pragma unroll
      for (int m = 0; m < 4; ++m) {
        weight_row[(wo + k) * kHidden + wi + m] = dw[layer][k * 4 + m];
      }
    }
  }
}


// ---- v2: occupancy-oriented variant ----
// 16-point tiles (37 KB shared memory), weights read through L1 instead of being staged
// (the forward GEMM uses a host-transposed copy for float4 loads), and weight gradients
// added into the block's own partials row after every layer instead of living in
// registers. Together these fit 2 blocks (16 warps) per SM.
constexpr int kOTile = 16;
constexpr int kONeurons = 4;  // per thread; 16 neuron groups x 16 points = 256 threads
constexpr int kOStride = kOTile + 1;
constexpr int kOPlane = kHidden * kOStride;
constexpr int kOScratchPerBlock = kHiddenLayers * 4 * kHidden * kOTile;
constexpr int kOSharedFloats = 8 * kOPlane + kSmallCount + 4 * kOTile;
constexpr size_t kOSharedBytes = kOSharedFloats * sizeof(float);
constexpr int kRowStride = 12740;  // partials rows padded to 16 bytes for float4 updates
static_assert(kRowStride % 4 == 0 && kRowStride >= kParameterCount, "bad partials stride");
static_assert(kOSharedBytes <= 48 * 1024, "v2 must fit 2 blocks per SM");
static_assert(kHiddenBase % 4 == 0 && kHiddenStride % 4 == 0, "hidden weights must be float4-aligned");

__device__ __forceinline__ int o_scratch(int layer, int component, int neuron, int point) {
  return ((layer * 4 + component) * kHidden + neuron) * kOTile + point;
}

// kSharedRows (v3): blocks on the same SM share one partials row and add into it with
// atomicAdd (compiled to fire-and-forget RED). 84 rows fit in L2 where 168 private rows don't.
// kCompactScratch (v4): layer 0's J (= W0 columns) and L (= 0) are constants, so only its z_v
// is stored; the backward rebuilds the rest from the parameters.
template <bool kSharedRows, bool kCompactScratch>
__global__ void __launch_bounds__(kThreads, 2)
fused_backward_occupancy_kernel(const float* __restrict__ coordinates,
                                const float* __restrict__ params,
                                const float* __restrict__ hidden_t,
                                const float* __restrict__ grad_output,
                                const float* __restrict__ grad_laplacian,
                                float* __restrict__ grad_coordinates,
                                float* __restrict__ partials,
                                float* __restrict__ scratch,
                                int64_t batch,
                                int64_t tiles,
                                int rows) {
  extern __shared__ float shared[];
  float* adj = shared;
  float* hid = adj + 4 * kOPlane;
  float* small = hid + 4 * kOPlane;
  float* s_gu = small + kSmallCount;
  float* s_gl = s_gu + kOTile;
  float* s_x0 = s_gl + kOTile;
  float* s_x1 = s_x0 + kOTile;
  float* block_scratch = scratch + static_cast<int64_t>(blockIdx.x) * kOScratchPerBlock;
  const int64_t row_index = kSharedRows ? blockIdx.x % rows : blockIdx.x;
  float* row = partials + row_index * kRowStride;
  // Private rows are overwritten; shared rows are pre-zeroed on the host and added into.
  auto put = [&](int index, float value) {
    if constexpr (kSharedRows) {
      atomicAdd(row + index, value);
    } else {
      row[index] = value;
    }
  };

  const int t = threadIdx.x;
  const int lane = t % kOTile;
  // Pair neuron groups 4 apart within a warp so the two half-warps' rows land 16 banks
  // apart (4 neurons * 4 groups * 17 floats = 272 = 16 mod 32): conflict-free stores.
  const int warp = t / 32;
  const int group = (warp / 4) * 8 + (warp % 4) + 4 * ((t % 32) / 16);
  const int o0 = group * kONeurons;
  const int wo = (t / 16) * 4;  // weight-gradient 4x4 block
  const int wi = (t % 16) * 4;
  auto z_load = [&](int layer, int component, int neuron) -> float {
    if constexpr (kCompactScratch) {
      if (layer == 0 && component > 0) {
        return component == 3 ? 0.0f : __ldg(params + neuron * kInputs + component - 1);
      }
    }
    return block_scratch[o_scratch(layer, component, neuron, lane)];
  };

  for (int k = t; k < kSmallCount; k += kThreads) {
    small[k] = 0.0f;
  }
  // Each thread owns its 4x4 weight-gradient blocks in this block's row: zero them here,
  // update them after every layer, and never share them, so no synchronization is needed.
  if constexpr (!kSharedRows) {
    for (int layer = 0; layer < kHiddenLayers - 1; ++layer) {
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        *reinterpret_cast<float4*>(row + kHiddenBase + layer * kHiddenStride +
                                   (wo + k) * kHidden + wi) = make_float4(0.f, 0.f, 0.f, 0.f);
      }
    }
  }

  for (int64_t tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
    const int64_t point = tile * kOTile + lane;
    const bool active = point < batch;
    __syncthreads();
    if (t < kOTile) {
      s_x0[t] = active ? coordinates[2 * point] : 0.0f;
      s_x1[t] = active ? coordinates[2 * point + 1] : 0.0f;
      s_gu[t] = active ? grad_output[point] : 0.0f;
      s_gl[t] = active ? grad_laplacian[point] : 0.0f;
    }
    __syncthreads();

    // ---- 1. Forward pass, saving pre-activations ----
#pragma unroll
    for (int k = 0; k < kONeurons; ++k) {
      const int o = o0 + k;
      const float w0 = __ldg(params + o * kInputs);
      const float w1 = __ldg(params + o * kInputs + 1);
      const float zv = w0 * s_x0[lane] + w1 * s_x1[lane] + __ldg(params + kLayer0Bias + o);
      block_scratch[o_scratch(0, 0, o, lane)] = zv;
      if constexpr (!kCompactScratch) {
        block_scratch[o_scratch(0, 1, o, lane)] = w0;
        block_scratch[o_scratch(0, 2, o, lane)] = w1;
        block_scratch[o_scratch(0, 3, o, lane)] = 0.0f;
      }
      const int h = o * kOStride + lane;
      tanh_forward(zv, w0, w1, 0.0f, hid[h], hid[kOPlane + h], hid[2 * kOPlane + h],
                   hid[3 * kOPlane + h]);
    }
    for (int layer = 1; layer < kHiddenLayers; ++layer) {
      const float* weight_t = hidden_t + (layer - 1) * kHidden * kHidden;  // [in][out]
      const float* bias = params + kHiddenBase + (layer - 1) * kHiddenStride + kHidden * kHidden;
      __syncthreads();  // previous layer's jet is written

      float zv[kONeurons], zj0[kONeurons], zj1[kONeurons], zl[kONeurons];
#pragma unroll
      for (int k = 0; k < kONeurons; ++k) {
        zv[k] = __ldg(bias + o0 + k);
        zj0[k] = zj1[k] = zl[k] = 0.0f;
      }
#pragma unroll 8
      for (int i = 0; i < kHidden; ++i) {
        const int h = i * kOStride + lane;
        const float hv = hid[h], hj0 = hid[kOPlane + h], hj1 = hid[2 * kOPlane + h],
                    hl = hid[3 * kOPlane + h];
        const float4 w4 = __ldg(reinterpret_cast<const float4*>(weight_t + i * kHidden + o0));
        const float w[kONeurons] = {w4.x, w4.y, w4.z, w4.w};
#pragma unroll
        for (int k = 0; k < kONeurons; ++k) {
          zv[k] = fmaf(w[k], hv, zv[k]);
          zj0[k] = fmaf(w[k], hj0, zj0[k]);
          zj1[k] = fmaf(w[k], hj1, zj1[k]);
          zl[k] = fmaf(w[k], hl, zl[k]);
        }
      }
      __syncthreads();  // all reads of hid done

#pragma unroll
      for (int k = 0; k < kONeurons; ++k) {
        const int o = o0 + k;
        block_scratch[o_scratch(layer, 0, o, lane)] = zv[k];
        block_scratch[o_scratch(layer, 1, o, lane)] = zj0[k];
        block_scratch[o_scratch(layer, 2, o, lane)] = zj1[k];
        block_scratch[o_scratch(layer, 3, o, lane)] = zl[k];
        const int h = o * kOStride + lane;
        tanh_forward(zv[k], zj0[k], zj1[k], zl[k], hid[h], hid[kOPlane + h],
                     hid[2 * kOPlane + h], hid[3 * kOPlane + h]);
      }
    }
    __syncthreads();

    // ---- 2a. Output layer ----
    const float* w_out = params + kOutputBase;
    if (t < kHidden) {
      float acc = 0.0f;
      for (int p = 0; p < kOTile; ++p) {
        acc += s_gu[p] * hid[t * kOStride + p] + s_gl[p] * hid[3 * kOPlane + t * kOStride + p];
      }
      small[kSmallWout + t] += acc;
    }
    if (t == kHidden) {
      float acc = 0.0f;
      for (int p = 0; p < kOTile; ++p) {
        acc += s_gu[p];
      }
      small[kSmallBout] += acc;
    }
#pragma unroll
    for (int k = 0; k < kONeurons; ++k) {
      const int i = o0 + k;
      const float w = __ldg(w_out + i);
      const int a = i * kOStride + lane;
      adj[a] = s_gu[lane] * w;
      adj[kOPlane + a] = 0.0f;
      adj[2 * kOPlane + a] = 0.0f;
      adj[3 * kOPlane + a] = s_gl[lane] * w;
    }
    __syncthreads();

    // ---- 2b. Hidden layers 3..1 ----
    for (int layer = kHiddenLayers - 1; layer >= 1; --layer) {
      const float* weight = params + kHiddenBase + (layer - 1) * kHiddenStride;  // [out][in]
#pragma unroll
      for (int k = 0; k < kONeurons; ++k) {
        const int o = o0 + k;
        const int a = o * kOStride + lane;
        tanh_backward(z_load(layer, 0, o), z_load(layer, 1, o), z_load(layer, 2, o),
                      z_load(layer, 3, o), adj[a], adj[kOPlane + a],
                      adj[2 * kOPlane + a], adj[3 * kOPlane + a]);
        tanh_forward(z_load(layer - 1, 0, o), z_load(layer - 1, 1, o), z_load(layer - 1, 2, o),
                     z_load(layer - 1, 3, o), hid[a], hid[kOPlane + a],
                     hid[2 * kOPlane + a], hid[3 * kOPlane + a]);
      }
      __syncthreads();

      float d[16];
#pragma unroll
      for (int k = 0; k < 16; ++k) {
        d[k] = 0.0f;
      }
#pragma unroll 4
      for (int p = 0; p < kOTile; ++p) {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          float zb[4], hb[4];
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            zb[k] = adj[c * kOPlane + (wo + k) * kOStride + p];
            hb[k] = hid[c * kOPlane + (wi + k) * kOStride + p];
          }
#pragma unroll
          for (int k = 0; k < 4; ++k) {
#pragma unroll
            for (int m = 0; m < 4; ++m) {
              d[k * 4 + m] = fmaf(zb[k], hb[m], d[k * 4 + m]);
            }
          }
        }
      }
      float* weight_row = row + kHiddenBase + (layer - 1) * kHiddenStride;
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        float* entry = weight_row + (wo + k) * kHidden + wi;
        if constexpr (kSharedRows) {
#pragma unroll
          for (int m = 0; m < 4; ++m) {
            atomicAdd(entry + m, d[k * 4 + m]);
          }
        } else {
          float4* slot = reinterpret_cast<float4*>(entry);
          float4 value = *slot;
          value.x += d[k * 4];
          value.y += d[k * 4 + 1];
          value.z += d[k * 4 + 2];
          value.w += d[k * 4 + 3];
          *slot = value;
        }
      }
      if (t < kHidden) {
        float acc = 0.0f;
        for (int p = 0; p < kOTile; ++p) {
          acc += adj[t * kOStride + p];
        }
        small[kSmallBias + layer * kHidden + t] += acc;
      }

      float gv[kONeurons], gj0[kONeurons], gj1[kONeurons], gl[kONeurons];
#pragma unroll
      for (int k = 0; k < kONeurons; ++k) {
        gv[k] = gj0[k] = gj1[k] = gl[k] = 0.0f;
      }
#pragma unroll 8
      for (int o = 0; o < kHidden; ++o) {
        const int a = o * kOStride + lane;
        const float av = adj[a], aj0 = adj[kOPlane + a], aj1 = adj[2 * kOPlane + a],
                    al = adj[3 * kOPlane + a];
        const float4 w4 = __ldg(reinterpret_cast<const float4*>(weight + o * kHidden + o0));
        const float w[kONeurons] = {w4.x, w4.y, w4.z, w4.w};
#pragma unroll
        for (int k = 0; k < kONeurons; ++k) {
          gv[k] = fmaf(w[k], av, gv[k]);
          gj0[k] = fmaf(w[k], aj0, gj0[k]);
          gj1[k] = fmaf(w[k], aj1, gj1[k]);
          gl[k] = fmaf(w[k], al, gl[k]);
        }
      }
      __syncthreads();

#pragma unroll
      for (int k = 0; k < kONeurons; ++k) {
        const int a = (o0 + k) * kOStride + lane;
        adj[a] = gv[k];
        adj[kOPlane + a] = gj0[k];
        adj[2 * kOPlane + a] = gj1[k];
        adj[3 * kOPlane + a] = gl[k];
      }
    }

    // ---- 3. Layer 0 ----
#pragma unroll
    for (int k = 0; k < kONeurons; ++k) {
      const int o = o0 + k;
      const int a = o * kOStride + lane;
      tanh_backward(z_load(0, 0, o), z_load(0, 1, o), z_load(0, 2, o), z_load(0, 3, o), adj[a],
                    adj[kOPlane + a],
                    adj[2 * kOPlane + a], adj[3 * kOPlane + a]);
    }
    __syncthreads();

    if (t < kHidden) {
      float gb = 0.0f, g0 = 0.0f, g1 = 0.0f;
      for (int p = 0; p < kOTile; ++p) {
        const int a = t * kOStride + p;
        const float zbar_v = adj[a];
        gb += zbar_v;
        g0 += zbar_v * s_x0[p] + adj[kOPlane + a];
        g1 += zbar_v * s_x1[p] + adj[2 * kOPlane + a];
      }
      small[kSmallBias + t] += gb;
      small[kSmallW0 + t * kInputs] += g0;
      small[kSmallW0 + t * kInputs + 1] += g1;
    }
    if (t < kOTile && active) {
      float gx0 = 0.0f, gx1 = 0.0f;
      for (int o = 0; o < kHidden; ++o) {
        const float zbar_v = adj[o * kOStride + lane];
        gx0 = fmaf(__ldg(params + o * kInputs), zbar_v, gx0);
        gx1 = fmaf(__ldg(params + o * kInputs + 1), zbar_v, gx1);
      }
      grad_coordinates[2 * point] = gx0;
      grad_coordinates[2 * point + 1] = gx1;
    }
  }
  __syncthreads();

  for (int k = t; k < kHidden * kInputs; k += kThreads) {
    put(k, small[kSmallW0 + k]);
  }
  if (t < kHidden) {
    put(kLayer0Bias + t, small[kSmallBias + t]);
    for (int layer = 1; layer < kHiddenLayers; ++layer) {
      put(kHiddenBase + (layer - 1) * kHiddenStride + kHidden * kHidden + t,
          small[kSmallBias + layer * kHidden + t]);
    }
    put(kOutputBase + t, small[kSmallWout + t]);
  }
  if (t == 0) {
    put(kOutputBase + kHidden, small[kSmallBout]);
  }
}

bool& attribute_configured(int device) {
  static bool configured[64] = {};
  return configured[device];
}

}  // namespace

std::vector<torch::Tensor> fused_forward_laplacian_backward_cuda(torch::Tensor coordinates,
                                                                 torch::Tensor parameters,
                                                                 torch::Tensor grad_output,
                                                                 torch::Tensor grad_laplacian,
                                                                 int64_t version) {
  const c10::cuda::CUDAGuard device_guard(coordinates.device());
  const int64_t batch = coordinates.size(0);
  auto grad_coordinates = torch::empty_like(coordinates);
  if (batch == 0) {
    return {torch::zeros_like(parameters), grad_coordinates};
  }

  const auto stream = at::cuda::getCurrentCUDAStream();
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  auto options = parameters.options();

  if (version == 1) {
    const int64_t tiles = (batch + kTile - 1) / kTile;
    const int blocks = static_cast<int>(std::min<int64_t>(tiles, sms));
    auto partials = torch::empty({blocks, kParameterCount}, options);
    auto scratch = torch::empty({static_cast<int64_t>(blocks) * kScratchPerBlock}, options);
    // Set once per device: it is not a stream operation, so it stays out of CUDA graphs.
    bool& configured = attribute_configured(coordinates.get_device());
    if (!configured) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(fused_forward_laplacian_backward_kernel,
                                          cudaFuncAttributeMaxDynamicSharedMemorySize,
                                          static_cast<int>(kSharedBytes)));
      configured = true;
    }
    fused_forward_laplacian_backward_kernel<<<blocks, kThreads, kSharedBytes, stream>>>(
        coordinates.data_ptr<float>(), parameters.data_ptr<float>(),
        grad_output.data_ptr<float>(), grad_laplacian.data_ptr<float>(),
        grad_coordinates.data_ptr<float>(), partials.data_ptr<float>(),
        scratch.data_ptr<float>(), batch, tiles);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {partials.sum(0), grad_coordinates};
  }

  TORCH_CHECK(reinterpret_cast<uintptr_t>(parameters.data_ptr<float>()) % 16 == 0,
              "parameters must be 16-byte aligned for the v2 backward");
  // Transposed hidden weights, [layer][in][out], for float4 loads in the forward GEMM.
  auto hidden = parameters.narrow(0, kHiddenBase, (kHiddenLayers - 1) * kHiddenStride)
                    .view({kHiddenLayers - 1, kHiddenStride});
  auto hidden_t = hidden.narrow(1, 0, kHidden * kHidden)
                      .view({kHiddenLayers - 1, kHidden, kHidden})
                      .transpose(1, 2)
                      .contiguous();
  const int64_t tiles = (batch + kOTile - 1) / kOTile;
  const int blocks = static_cast<int>(std::min<int64_t>(tiles, 2 * sms));
  const bool shared_rows = version == 3;
  const int rows = shared_rows ? std::min(blocks, sms) : blocks;
  auto partials = shared_rows ? torch::zeros({rows, kRowStride}, options)
                              : torch::empty({rows, kRowStride}, options);
  auto scratch = torch::empty({static_cast<int64_t>(blocks) * kOScratchPerBlock}, options);
  auto kernel = shared_rows        ? fused_backward_occupancy_kernel<true, false>
                : version == 4 ? fused_backward_occupancy_kernel<false, true>
                               : fused_backward_occupancy_kernel<false, false>;
  kernel<<<blocks, kThreads, kOSharedBytes, stream>>>(
      coordinates.data_ptr<float>(), parameters.data_ptr<float>(), hidden_t.data_ptr<float>(),
      grad_output.data_ptr<float>(), grad_laplacian.data_ptr<float>(),
      grad_coordinates.data_ptr<float>(), partials.data_ptr<float>(), scratch.data_ptr<float>(),
      batch, tiles, rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {partials.narrow(1, 0, kParameterCount).sum(0), grad_coordinates};
}
