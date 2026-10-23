#pragma once

// Shape and packed-parameter layout of the fixed 2 -> 64x4 -> 1 tanh MLP, shared by the
// forward and backward kernels (see src/fused_hoad/parameters.py).

namespace fused_hoad {

constexpr int kInputs = 2;
constexpr int kHidden = 64;
constexpr int kHiddenLayers = 4;

constexpr int kLayer0Bias = kHidden * kInputs;
constexpr int kHiddenBase = kLayer0Bias + kHidden;
constexpr int kHiddenStride = kHidden * kHidden + kHidden;
constexpr int kOutputBase = kHiddenBase + (kHiddenLayers - 1) * kHiddenStride;
constexpr int kParameterCount = kOutputBase + kHidden + 1;
static_assert(kParameterCount == 12737, "parameter ABI drifted from parameters.py");

}  // namespace fused_hoad
