// C++ wrapper for the CUDA LayerNorm kernel

#include <torch/extension.h>

torch::Tensor layer_norm_forward(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    float eps
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &layer_norm_forward, "LayerNorm forward (CUDA)");
}
