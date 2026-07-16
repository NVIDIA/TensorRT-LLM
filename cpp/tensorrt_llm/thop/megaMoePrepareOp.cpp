/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/megaMoePrepareKernel.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/CUDAContext.h>
#include <limits>
#include <torch/extension.h>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{
namespace
{

using kernels::MegaMoePrepareExpertType;
using kernels::MegaMoePrepareScaleType;

MegaMoePrepareExpertType getExpertType(torch::Tensor const& tensor)
{
    if (tensor.scalar_type() == torch::kInt32)
    {
        return MegaMoePrepareExpertType::INT32;
    }
    if (tensor.scalar_type() == torch::kInt64)
    {
        return MegaMoePrepareExpertType::INT64;
    }
    TORCH_CHECK(false, "megamoe_prepare: token_selected_experts must be int32 or int64, got ", tensor.scalar_type());
    return MegaMoePrepareExpertType::INT32;
}

MegaMoePrepareScaleType getScaleType(torch::Tensor const& tensor)
{
    if (tensor.scalar_type() == torch::kFloat32)
    {
        return MegaMoePrepareScaleType::FP32;
    }
    if (tensor.scalar_type() == torch::kFloat16)
    {
        return MegaMoePrepareScaleType::FP16;
    }
    if (tensor.scalar_type() == torch::kBFloat16)
    {
        return MegaMoePrepareScaleType::BF16;
    }
    TORCH_CHECK(false, "megamoe_prepare: token_final_scales must be fp32/fp16/bf16, got ", tensor.scalar_type());
    return MegaMoePrepareScaleType::FP32;
}

} // namespace

void megaMoePrepare(torch::Tensor input, torch::Tensor tokenSelectedExperts, torch::Tensor tokenFinalScales,
    torch::Tensor xOut, torch::Tensor xSfOut, torch::Tensor topkIdxOut, torch::Tensor topkWeightsOut)
{
    CHECK_TH_CUDA(input);
    CHECK_TH_CUDA(tokenSelectedExperts);
    CHECK_TH_CUDA(tokenFinalScales);
    CHECK_TH_CUDA(xOut);
    CHECK_TH_CUDA(xSfOut);
    CHECK_TH_CUDA(topkIdxOut);
    CHECK_TH_CUDA(topkWeightsOut);

    CHECK_CONTIGUOUS(input);
    CHECK_CONTIGUOUS(tokenSelectedExperts);
    CHECK_CONTIGUOUS(tokenFinalScales);
    CHECK_CONTIGUOUS(xOut);
    CHECK_CONTIGUOUS(xSfOut);
    CHECK_CONTIGUOUS(topkIdxOut);
    CHECK_CONTIGUOUS(topkWeightsOut);

    TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "megamoe_prepare: input must be bfloat16");
    TORCH_CHECK(xOut.scalar_type() == torch::kFloat8_e4m3fn, "megamoe_prepare: x_out must be Float8_e4m3fn");
    TORCH_CHECK(xSfOut.scalar_type() == torch::kInt32, "megamoe_prepare: x_sf_out must be int32");
    TORCH_CHECK(topkIdxOut.scalar_type() == torch::kInt64, "megamoe_prepare: topk_idx_out must be int64");
    TORCH_CHECK(topkWeightsOut.scalar_type() == torch::kFloat32, "megamoe_prepare: topk_weights_out must be fp32");
    TORCH_CHECK(tensorrt_llm::common::getSMVersion() >= 100, "megamoe_prepare requires SM100 or newer");

    TORCH_CHECK(input.dim() == 2, "megamoe_prepare: input must be [num_tokens, hidden_size]");
    TORCH_CHECK(tokenSelectedExperts.dim() == 2, "megamoe_prepare: token_selected_experts must be [num_tokens, top_k]");
    TORCH_CHECK(tokenFinalScales.sizes() == tokenSelectedExperts.sizes(),
        "megamoe_prepare: token_final_scales shape must match token_selected_experts");

    int64_t const numTokens64 = input.size(0);
    int64_t const hiddenSize64 = input.size(1);
    int64_t const topK64 = tokenSelectedExperts.size(1);
    TORCH_CHECK(tokenSelectedExperts.size(0) == numTokens64,
        "megamoe_prepare: token_selected_experts first dim must match input");
    TORCH_CHECK(
        hiddenSize64 % 128 == 0, "megamoe_prepare: hidden_size must be divisible by 128 for packed UE8M0 int32 scales");
    TORCH_CHECK(numTokens64 <= std::numeric_limits<int>::max(), "megamoe_prepare: num_tokens exceeds int range");
    TORCH_CHECK(hiddenSize64 <= std::numeric_limits<int>::max(), "megamoe_prepare: hidden_size exceeds int range");
    TORCH_CHECK(topK64 <= std::numeric_limits<int>::max(), "megamoe_prepare: top_k exceeds int range");

    TORCH_CHECK(xOut.dim() == 2 && xOut.size(0) >= numTokens64 && xOut.size(1) == hiddenSize64,
        "megamoe_prepare: x_out must be [>=num_tokens, hidden_size]");
    TORCH_CHECK(xSfOut.dim() == 2 && xSfOut.size(0) >= numTokens64 && xSfOut.size(1) == hiddenSize64 / 128,
        "megamoe_prepare: x_sf_out must be [>=num_tokens, hidden_size / 128] int32");
    TORCH_CHECK(topkIdxOut.dim() == 2 && topkIdxOut.size(0) >= numTokens64 && topkIdxOut.size(1) == topK64,
        "megamoe_prepare: topk_idx_out must be [>=num_tokens, top_k]");
    TORCH_CHECK(topkWeightsOut.dim() == 2 && topkWeightsOut.size(0) >= numTokens64 && topkWeightsOut.size(1) == topK64,
        "megamoe_prepare: topk_weights_out must be [>=num_tokens, top_k]");

    if (numTokens64 == 0)
    {
        return;
    }

    auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
    int const multiProcessorCount = tensorrt_llm::common::getMultiProcessorCount();
    kernels::invokeMegaMoePrepare(input.data_ptr(), tokenSelectedExperts.data_ptr(), tokenFinalScales.data_ptr(),
        xOut.data_ptr(), xSfOut.data_ptr(), topkIdxOut.data_ptr<int64_t>(), topkWeightsOut.data_ptr<float>(),
        static_cast<int>(numTokens64), static_cast<int>(hiddenSize64), static_cast<int>(topK64),
        getExpertType(tokenSelectedExperts), getScaleType(tokenFinalScales), multiProcessorCount, stream);
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "megamoe_prepare("
        "Tensor input, Tensor token_selected_experts, Tensor token_final_scales, "
        "Tensor(a!) x_out, Tensor(b!) x_sf_out, Tensor(c!) topk_idx_out, Tensor(d!) topk_weights_out) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("megamoe_prepare", &tensorrt_llm::torch_ext::megaMoePrepare);
}
