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

// Returns (FP8 nope, bf16/fp16 rope). When `interleavedQuantQ` is true the FP8
// tensor is laid out [N, H*head_dim] so RoPE_OptContext can fill the rope slot;
// otherwise it is packed [N, H*nope_dim].
std::tuple<torch::Tensor, torch::Tensor> deepseekV4QNormFusedFp8(torch::Tensor q, int64_t numHeads, int64_t headDim,
    int64_t nopeDim, double eps, torch::optional<torch::Tensor> quantScaleQkv, bool interleavedQuantQ)
{
    TORCH_CHECK(q.is_cuda(), "deepseek_v4_q_norm_fused_fp8 expects a CUDA tensor");
    TORCH_CHECK(q.is_contiguous(), "deepseek_v4_q_norm_fused_fp8 expects a contiguous tensor");
    TORCH_CHECK(q.scalar_type() == torch::kFloat16 || q.scalar_type() == torch::kBFloat16,
        "deepseek_v4_q_norm_fused_fp8 expects fp16/bf16 input, got ", q.scalar_type());
    TORCH_CHECK(headDim == 512, "deepseek_v4_q_norm_fused_fp8 only supports head_dim=512, got ", headDim);
    TORCH_CHECK(nopeDim == 448, "deepseek_v4_q_norm_fused_fp8 only supports nope_dim=448, got ", nopeDim);

    TORCH_CHECK(q.dim() == 2, "deepseek_v4_q_norm_fused_fp8 expects 2D q [num_tokens, num_heads * head_dim], got ",
        q.dim(), "D");
    TORCH_CHECK(q.size(1) == numHeads * headDim, "q.shape[1] must equal num_heads * head_dim");

    int64_t const numTokens = q.size(0);
    int64_t const ropeDim = headDim - nopeDim;
    auto deviceOpts = q.options().device(q.device());

    auto quantQNope = torch::empty(
        {numTokens, numHeads * (interleavedQuantQ ? headDim : nopeDim)}, deviceOpts.dtype(torch::kFloat8_e4m3fn));
    auto qPeOut = torch::empty({numTokens, numHeads * ropeDim}, q.options());

    int64_t const totalRows64 = numTokens * numHeads;
    TORCH_CHECK(
        totalRows64 <= std::numeric_limits<int>::max(), "deepseek_v4_q_norm_fused_fp8 total rows exceed int range");
    int const totalRows = static_cast<int>(totalRows64);

    void const* quantScalePtr = nullptr;
    if (quantScaleQkv.has_value() && quantScaleQkv->defined())
    {
        auto const& s = quantScaleQkv.value();
        TORCH_CHECK(s.is_cuda(), "quant_scale_qkv must be on CUDA");
        TORCH_CHECK(s.scalar_type() == torch::kFloat32, "quant_scale_qkv must be float32");
        TORCH_CHECK(s.numel() >= 1, "quant_scale_qkv must have at least 1 element");
        quantScalePtr = s.data_ptr();
    }

    int const quantQNopeRowStrideBytes = interleavedQuantQ ? static_cast<int>(headDim) : static_cast<int>(nopeDim);

    auto stream = at::cuda::getCurrentCUDAStream(q.get_device());
    bool const isBfloat16 = (q.scalar_type() == torch::kBFloat16);
    tensorrt_llm::kernels::invokeDeepseekV4QNormFusedFp8(q.data_ptr(), quantQNope.data_ptr(), qPeOut.data_ptr(),
        quantScalePtr, totalRows, static_cast<int>(headDim), static_cast<int>(nopeDim), quantQNopeRowStrideBytes,
        isBfloat16, static_cast<float>(eps), stream);

    return std::make_tuple(std::move(quantQNope), std::move(qPeOut));
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("deepseek_v4_q_norm(Tensor q, int num_heads, int head_dim, float eps) -> Tensor");
    m.def(
        "deepseek_v4_q_norm_fused_fp8(Tensor q, int num_heads, int head_dim, int nope_dim, float eps, "
        "Tensor? quant_scale_qkv, bool interleaved_quant_q=False) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("deepseek_v4_q_norm", &tensorrt_llm::torch_ext::deepseekV4QNorm);
    m.impl("deepseek_v4_q_norm_fused_fp8", &tensorrt_llm::torch_ext::deepseekV4QNormFusedFp8);
}
