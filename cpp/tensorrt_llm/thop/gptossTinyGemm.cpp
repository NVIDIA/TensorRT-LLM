#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/dataType.h"
#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/runtime/torchUtils.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include "tensorrt_llm/thop/thUtils.h"

// CUDA forward declarations
torch::Tensor tinygemm2_cuda_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                                                                 \
    CHECK_CUDA(x);                                                                                                     \
    CHECK_CONTIGUOUS(x)

torch::Tensor tinygemm2_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias)
{
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);
    return tinygemm2_cuda_forward(input, weight, bias);
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "gptoss_tinygemm(Tensor input, Tensor conv_weight, "
        "Tensor bias) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("gptoss_tinygemm", &torch_ext::gptoss_tinygemm);
}
