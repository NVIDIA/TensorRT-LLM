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
    CHECK_INPUT(input, torch::kBFloat16);
    CHECK_INPUT(weight, torch::kBFloat16);
    CHECK_INPUT(bias, torch::kBFloat16);
    return tinygemm2_cuda_forward(input, weight, bias);
}
} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "gptoss_tinygemm(Tensor input, Tensor conv_weight, "
        "Tensor bias) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("gptoss_tinygemm", &torch_ext::tinygemm2_forward);
}
