/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/glmMoeDsa/glmLowMBf16Moe.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/CUDAContext.h>

#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{
bool isAligned16(at::Tensor const& t)
{
    return reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0;
}
} // namespace

// One token through an MoE layer with unquantized BF16 SwiGLU experts held as TP slices and one-group DeepSeek-V3
// routing: routing from the router logits, FC1 + SwiGLU, and FC2 + the weighted sum over the routed experts, in two
// kernels (glmLowMBf16Moe.h). The weights are read in the layout CutlassFusedMoE keeps: w3_w1 [E, 2I, H] with the up
// (W3) rows first, w2 [E, H, I]. Returns [1, H].
at::Tensor glmLowmBf16Moe(at::Tensor const& x, at::Tensor const& routerLogits, at::Tensor const& eScoreCorrectionBias,
    at::Tensor const& w3W1, at::Tensor const& w2, int64_t topK, double routedScalingFactor)
{
    namespace tk = tensorrt_llm::kernels::glm_moe_dsa;

    CHECK_INPUT(x, at::ScalarType::BFloat16);
    CHECK_INPUT(routerLogits, at::ScalarType::Float);
    CHECK_TH_CUDA(eScoreCorrectionBias);
    CHECK_CONTIGUOUS(eScoreCorrectionBias);
    CHECK_INPUT(w3W1, at::ScalarType::BFloat16);
    CHECK_INPUT(w2, at::ScalarType::BFloat16);
    TORCH_CHECK(eScoreCorrectionBias.scalar_type() == at::ScalarType::Float
            || eScoreCorrectionBias.scalar_type() == at::ScalarType::BFloat16,
        "glm_lowm_bf16_moe expects an FP32 or BF16 e_score_correction_bias, got ", eScoreCorrectionBias.scalar_type());
    for (at::Tensor const* t : {&routerLogits, &eScoreCorrectionBias, &w3W1, &w2})
    {
        TORCH_CHECK(t->device() == x.device(), "glm_lowm_bf16_moe: all tensors must be on the device of x");
    }
    TORCH_CHECK(tensorrt_llm::common::getSMVersion() >= 90, "glm_lowm_bf16_moe needs SM90 or newer");

    TORCH_CHECK(x.dim() == 2 && x.size(0) == 1, "glm_lowm_bf16_moe takes one token, got x of shape ", x.sizes());
    TORCH_CHECK(w3W1.dim() == 3 && w2.dim() == 3, "glm_lowm_bf16_moe expects 3D expert weights");
    int64_t const hiddenSize = x.size(1);
    int64_t const numExperts = w3W1.size(0);
    int64_t const interSize = w2.size(2);
    TORCH_CHECK(numExperts == tk::kLowMBf16MoeNumExperts, "glm_lowm_bf16_moe is built for ", tk::kLowMBf16MoeNumExperts,
        " experts, got ", numExperts);
    TORCH_CHECK(
        topK == tk::kLowMBf16MoeTopK, "glm_lowm_bf16_moe is built for top-", tk::kLowMBf16MoeTopK, ", got top-", topK);
    TORCH_CHECK(w3W1.size(1) == 2 * interSize && w3W1.size(2) == hiddenSize, "glm_lowm_bf16_moe: w3_w1 of shape ",
        w3W1.sizes(), " does not match x ", x.sizes(), " and w2 ", w2.sizes());
    TORCH_CHECK(w2.size(0) == numExperts && w2.size(1) == hiddenSize, "glm_lowm_bf16_moe: w2 of shape ", w2.sizes(),
        " does not match x ", x.sizes(), " and w3_w1 ", w3W1.sizes());
    TORCH_CHECK(routerLogits.dim() == 2 && routerLogits.size(0) == 1 && routerLogits.size(1) == numExperts,
        "glm_lowm_bf16_moe: router_logits must be [1, ", numExperts, "], got ", routerLogits.sizes());
    TORCH_CHECK(eScoreCorrectionBias.dim() == 1 && eScoreCorrectionBias.size(0) == numExperts,
        "glm_lowm_bf16_moe: e_score_correction_bias must be [", numExperts, "], got ", eScoreCorrectionBias.sizes());
    TORCH_CHECK(tk::lowMBf16MoeSupportsShape(hiddenSize, interSize), "glm_lowm_bf16_moe: unsupported hidden size ",
        hiddenSize, " or intermediate size ", interSize);
    TORCH_CHECK(isAligned16(x) && isAligned16(w3W1) && isAligned16(w2),
        "glm_lowm_bf16_moe needs 16-byte aligned x, w3_w1 and w2");

    at::Tensor out = at::empty({1, hiddenSize}, x.options());
    at::Tensor act = at::empty({topK, interSize}, x.options());
    auto const stream = at::cuda::getCurrentCUDAStream(x.get_device());
    auto* outPtr = static_cast<__nv_bfloat16*>(out.data_ptr());
    auto* actPtr = static_cast<__nv_bfloat16*>(act.data_ptr());
    auto const* xPtr = static_cast<__nv_bfloat16 const*>(x.data_ptr());
    auto const* logitsPtr = static_cast<float const*>(routerLogits.data_ptr());
    auto const* w3W1Ptr = static_cast<__nv_bfloat16 const*>(w3W1.data_ptr());
    auto const* w2Ptr = static_cast<__nv_bfloat16 const*>(w2.data_ptr());
    if (eScoreCorrectionBias.scalar_type() == at::ScalarType::Float)
    {
        tk::invokeLowMBf16Moe<float>(outPtr, actPtr, xPtr, logitsPtr,
            static_cast<float const*>(eScoreCorrectionBias.data_ptr()), w3W1Ptr, w2Ptr, static_cast<int>(hiddenSize),
            static_cast<int>(interSize), routedScalingFactor, stream);
    }
    else
    {
        tk::invokeLowMBf16Moe<__nv_bfloat16>(outPtr, actPtr, xPtr, logitsPtr,
            static_cast<__nv_bfloat16 const*>(eScoreCorrectionBias.data_ptr()), w3W1Ptr, w2Ptr,
            static_cast<int>(hiddenSize), static_cast<int>(interSize), routedScalingFactor, stream);
    }
    return out;
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "glm_lowm_bf16_moe(Tensor x, Tensor router_logits, Tensor e_score_correction_bias, Tensor w3_w1, Tensor w2, "
        "int top_k, float routed_scaling_factor) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("glm_lowm_bf16_moe", &tensorrt_llm::torch_ext::glmLowmBf16Moe);
}
