#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/dataType.h"
#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/runtime/torchUtils.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include "tensorrt_llm/thop/thUtils.h"

// CUDA forward declarations
torch::Tensor tinygemm2_cuda_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);

// C++ interface
namespace torch_ext
{
torch::Tensor tinygemm2_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias)
{
    TORCH_CHECK(input.dim() == 2, "input must be 2D");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D");
    TORCH_CHECK(input.sizes()[1] == weight.sizes()[1], "input.size(1) must match weight.size(1)");
    TORCH_CHECK(weight.sizes()[0] == bias.sizes()[0], "weight.size(0) must match bias.size(0)");
    CHECK_INPUT(input, torch::kBFloat16);
    CHECK_INPUT(weight, torch::kBFloat16);
    CHECK_INPUT(bias, torch::kBFloat16);
    return tinygemm2_cuda_forward(input, weight, bias);
}
} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("tinygemm2(Tensor input, Tensor weight, Tensor bias) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("tinygemm2", &torch_ext::tinygemm2_forward);
}
