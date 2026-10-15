#include <torch/extension.h>

torch::Tensor fused_residual_cuda(torch::Tensor coordinates);

torch::Tensor fused_residual(torch::Tensor coordinates) {
  TORCH_CHECK(coordinates.is_cuda(), "coordinates must be a CUDA tensor");
  TORCH_CHECK(coordinates.is_contiguous(), "coordinates must be contiguous");
  TORCH_CHECK(coordinates.scalar_type() == torch::kFloat32,
              "coordinates must use float32");
  TORCH_CHECK(coordinates.dim() == 2, "coordinates must have shape [batch, dimensions]");
  return fused_residual_cuda(coordinates);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("fused_residual", &fused_residual,
             "Prototype fused residual operation (CUDA)");
}

