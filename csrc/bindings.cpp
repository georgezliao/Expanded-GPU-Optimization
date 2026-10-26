#include <torch/extension.h>

#include <vector>

namespace {

constexpr int64_t kParameterCount = 12737;

void check_inputs(const torch::Tensor& coordinates, const torch::Tensor& parameters) {
  TORCH_CHECK(coordinates.is_cuda(), "coordinates must be a CUDA tensor");
  TORCH_CHECK(parameters.is_cuda(), "parameters must be a CUDA tensor");
  TORCH_CHECK(coordinates.is_contiguous(), "coordinates must be contiguous");
  TORCH_CHECK(parameters.is_contiguous(), "parameters must be contiguous");
  TORCH_CHECK(coordinates.scalar_type() == torch::kFloat32,
              "coordinates must use float32");
  TORCH_CHECK(parameters.scalar_type() == torch::kFloat32,
              "parameters must use float32");
  TORCH_CHECK(coordinates.dim() == 2 && coordinates.size(1) == 2,
              "coordinates must have shape [batch, 2]");
  TORCH_CHECK(parameters.dim() == 1 && parameters.numel() == kParameterCount,
              "parameters must have shape [12737]");
  TORCH_CHECK(coordinates.device() == parameters.device(),
              "coordinates and parameters must be on the same device");
}

}  // namespace

std::vector<torch::Tensor> fused_forward_laplacian_cuda(
    torch::Tensor coordinates, torch::Tensor parameters, int64_t version);

std::vector<torch::Tensor> fused_forward_laplacian(
    torch::Tensor coordinates, torch::Tensor parameters, int64_t version) {
  TORCH_CHECK(version >= 0 && version <= 3, "version must be 0, 1, 2 or 3");
  check_inputs(coordinates, parameters);
  return fused_forward_laplacian_cuda(coordinates, parameters, version);
}

std::vector<torch::Tensor> fused_forward_laplacian_backward_cuda(
    torch::Tensor coordinates, torch::Tensor parameters, torch::Tensor grad_output,
    torch::Tensor grad_laplacian, int64_t version);

std::vector<torch::Tensor> fused_forward_laplacian_backward(
    torch::Tensor coordinates, torch::Tensor parameters, torch::Tensor grad_output,
    torch::Tensor grad_laplacian, int64_t version) {
  TORCH_CHECK(version >= 1 && version <= 6, "backward version must be 1 to 6");
  check_inputs(coordinates, parameters);
  for (const auto& grad : {grad_output, grad_laplacian}) {
    TORCH_CHECK(grad.is_cuda() && grad.is_contiguous() && grad.scalar_type() == torch::kFloat32,
                "upstream gradients must be contiguous float32 CUDA tensors");
    TORCH_CHECK(grad.numel() == coordinates.size(0), "upstream gradients must have [batch] values");
  }
  return fused_forward_laplacian_backward_cuda(coordinates, parameters, grad_output,
                                               grad_laplacian, version);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("fused_forward_laplacian", &fused_forward_laplacian,
             "Fixed MLP forward pass and input Laplacian (CUDA)",
             pybind11::arg("coordinates"), pybind11::arg("parameters"),
             pybind11::arg("version") = 2);
  module.def("fused_forward_laplacian_backward", &fused_forward_laplacian_backward,
             "Gradients of the fused forward Laplacian w.r.t. parameters and coordinates (CUDA)",
             pybind11::arg("coordinates"), pybind11::arg("parameters"),
             pybind11::arg("grad_output"), pybind11::arg("grad_laplacian"),
             pybind11::arg("version") = 6);
}

