#include "tensorrt_llm/kernels/deepseekV4QNormKernel.h"

#include <ATen/cuda/CUDAContext.h>
#include <limits>
#include <torch/extension.h>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

torch::Tensor deepseekV4QNorm(torch::Tensor q, int64_t numHeads, int64_t headDim, double eps)
{
    TORCH_CHECK(q.is_cuda(), "deepseek_v4_q_norm expects a CUDA tensor");
    TORCH_CHECK(q.is_contiguous(), "deepseek_v4_q_norm expects a contiguous tensor");
    TORCH_CHECK(q.scalar_type() == torch::kFloat16 || q.scalar_type() == torch::kBFloat16,
        "deepseek_v4_q_norm expects fp16/bf16 input, got ", q.scalar_type());
    TORCH_CHECK(headDim == 512, "deepseek_v4_q_norm only supports head_dim=512, got ", headDim);

    TORCH_CHECK(q.dim() == 2, "deepseek_v4_q_norm expects 2D q [num_tokens, num_heads * head_dim], got ", q.dim(), "D");
    TORCH_CHECK(q.size(1) == numHeads * headDim, "q.shape[1] must equal num_heads * head_dim");

    auto output = torch::empty_like(q);
    int64_t const totalRows64 = q.size(0) * numHeads;
    TORCH_CHECK(totalRows64 <= std::numeric_limits<int>::max(), "deepseek_v4_q_norm total rows exceed int range");
    int const totalRows = static_cast<int>(totalRows64);
    auto stream = at::cuda::getCurrentCUDAStream(q.get_device());

    if (q.scalar_type() == torch::kFloat16)
    {
        tensorrt_llm::kernels::invokeDeepseekV4QNorm(q.data_ptr(), output.data_ptr(), totalRows,
            static_cast<int>(headDim), false, static_cast<float>(eps), stream);
    }
    else
    {
        tensorrt_llm::kernels::invokeDeepseekV4QNorm(q.data_ptr(), output.data_ptr(), totalRows,
            static_cast<int>(headDim), true, static_cast<float>(eps), stream);
    }
    return output;
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("deepseek_v4_q_norm(Tensor q, int num_heads, int head_dim, float eps) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("deepseek_v4_q_norm", &tensorrt_llm::torch_ext::deepseekV4QNorm);
}
