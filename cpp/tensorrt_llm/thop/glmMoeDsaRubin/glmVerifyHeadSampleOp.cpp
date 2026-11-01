/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/glmMoeDsaRubin/glmVerifyHeadSample.h"
#include "tensorrt_llm/thop/thUtils.h"
#include <c10/cuda/CUDAGuard.h>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{
namespace
{
void checkArray(at::Tensor const& tensor, at::Device device, at::ScalarType dtype, int64_t minimum, char const* name)
{
    TORCH_CHECK(tensor.device() == device && tensor.scalar_type() == dtype && tensor.is_contiguous()
            && tensor.dim() == 1 && tensor.numel() >= minimum,
        name, " must be a contiguous CUDA vector of the required dtype and length");
}
} // namespace

at::Tensor glmVerifyHeadSampleTokens(at::Tensor const& logits, at::Tensor const& temperatures, at::Tensor const& topK,
    at::Tensor const& topP, at::Tensor const& minP, at::Tensor const& seed, at::Tensor const& offset,
    at::Tensor& counters, at::Tensor& workspace)
{
    TORCH_CHECK(logits.is_cuda() && logits.scalar_type() == at::kFloat && logits.is_contiguous() && logits.dim() == 2
            && logits.size(0) >= 1 && logits.size(0) <= 24 && logits.size(1) == 154880
            && reinterpret_cast<uintptr_t>(logits.const_data_ptr()) % 16 == 0,
        "GLM verify sampling requires contiguous aligned FP32 logits [1..24, 154880]");
    c10::cuda::CUDAGuard guard(logits.device());
    TORCH_CHECK(tensorrt_llm::common::getSMVersion() == 107, "GLM verify sampling requires SM107");
    int64_t const rows = logits.size(0);
    checkArray(temperatures, logits.device(), at::kFloat, rows, "temperatures");
    checkArray(topK, logits.device(), at::kInt, rows, "top_ks");
    checkArray(topP, logits.device(), at::kFloat, rows, "top_ps");
    checkArray(minP, logits.device(), at::kFloat, rows, "min_ps");
    checkArray(seed, logits.device(), at::kLong, 1, "seed");
    checkArray(offset, logits.device(), at::kLong, 1, "offset");
    TORCH_CHECK(seed.numel() == offset.numel() && (seed.numel() == 1 || seed.numel() >= rows),
        "RNG arrays must have equal length, either one shared entry or at least one per row");
    checkArray(counters, logits.device(), at::kInt, rows, "row_counters");
    namespace glm = kernels::glm_moe_dsa;
    auto const bytes = glm::getGlmVerifyHeadSampleWorkspaceSize(rows, logits.size(1));
    checkArray(workspace, logits.device(), at::kByte, bytes, "workspace");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(workspace.const_data_ptr()) % 16 == 0, "workspace must be 16-byte aligned");
    auto tokens = at::empty({rows}, logits.options().dtype(at::kInt));
    glm::GlmVerifyHeadSampleParams p{};
    p.logits = logits.const_data_ptr<float>();
    p.temperatures = temperatures.const_data_ptr<float>();
    p.topK = topK.const_data_ptr<int32_t>();
    p.topP = topP.const_data_ptr<float>();
    p.minP = minP.const_data_ptr<float>();
    p.seed = seed.const_data_ptr<int64_t>();
    p.offset = offset.const_data_ptr<int64_t>();
    // The singleton case is per-row, matching current fused sampling's contract.
    p.perRowRng = seed.numel() >= rows;
    p.rowCounters = counters.mutable_data_ptr<int32_t>();
    p.tokens = tokens.mutable_data_ptr<int32_t>();
    p.workspace = workspace.mutable_data_ptr();
    p.numRows = rows;
    p.vocabSize = logits.size(1);
    p.stream = at::cuda::getCurrentCUDAStream(logits.get_device());
    glm::invokeGlmVerifyHeadSample(p);
    return tokens;
}
} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "glm_verify_head_sample_tokens(Tensor logits, Tensor temperatures, Tensor top_ks, Tensor top_ps, "
        "Tensor min_ps, Tensor seed, Tensor offset, Tensor(a!) row_counters, Tensor(b!) workspace) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("glm_verify_head_sample_tokens", &tensorrt_llm::torch_ext::glmVerifyHeadSampleTokens);
}
