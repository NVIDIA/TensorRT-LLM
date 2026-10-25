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
#include "tensorrt_llm/kernels/glmMoeDsa/glmFusedAGemm.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>
#include <torch/extension.h>

namespace tk = tensorrt_llm::kernels;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

using tk::glm_moe_dsa::FusedAL2Prefetch;

constexpr int64_t kTmaAlignmentBytes = 16;

bool isAligned(torch::Tensor const& tensor)
{
    return reinterpret_cast<uintptr_t>(tensor.data_ptr()) % kTmaAlignmentBytes == 0;
}

// One L2 prefetch range per tensor: its contiguous bytes on the GEMM's device.
FusedAL2Prefetch makeL2Prefetch(torch::TensorList tensors, torch::Device const& device)
{
    TORCH_CHECK(tensors.size() <= static_cast<size_t>(FusedAL2Prefetch::kMaxRanges),
        "glm_fused_a_gemm prefetches at most ", FusedAL2Prefetch::kMaxRanges, " tensors, got ", tensors.size());
    FusedAL2Prefetch ranges{};
    for (size_t i = 0; i < tensors.size(); ++i)
    {
        torch::Tensor const& tensor = tensors[i];
        TORCH_CHECK(tensor.device() == device, "l2_prefetch tensors must be on the GEMM's device");
        TORCH_CHECK(tensor.is_contiguous(), "l2_prefetch tensors must be contiguous");
        TORCH_CHECK(isAligned(tensor), "l2_prefetch tensors must be 16-byte aligned");
        ranges.ptr[i] = tensor.data_ptr();
        ranges.bytes[i] = static_cast<size_t>(tensor.numel()) * tensor.element_size();
    }
    return ranges;
}

} // namespace

//! The GLM-5 fused-A GEMM (kv_a_proj_with_mqa) for up to 8 tokens.
//!
//! @param hiddenStates BF16 CUDA tensor [num_tokens, 6144], contiguous, num_tokens in [1, 8].
//! @param weight BF16 CUDA tensor [out_features, 6144], contiguous: the kv_a_proj_with_mqa weight (2624 rows: q_a,
//! compressed KV, k_pe) or that weight with the DSA indexer's wk and weights_proj rows appended (2784 rows).
//! @param l2Prefetch Up to four contiguous, 16-byte aligned tensors on the same device that the kernel prefetches into
//! L2 before its grid-dependency wait (weights later kernels of the step read); may be empty.
//! @return BF16 tensor [num_tokens, out_features] = hiddenStates @ weight^T, FP32 accumulation.
torch::Tensor glmFusedAGemm(
    torch::Tensor const& hiddenStates, torch::Tensor const& weight, torch::TensorList l2Prefetch)
{
    namespace glm = tk::glm_moe_dsa;
    TORCH_CHECK(hiddenStates.is_cuda() && weight.is_cuda(), "glm_fused_a_gemm expects CUDA tensors");
    TORCH_CHECK(hiddenStates.device() == weight.device(), "hidden_states and weight must be on the same device");
    TORCH_CHECK(hiddenStates.scalar_type() == torch::kBFloat16 && weight.scalar_type() == torch::kBFloat16,
        "glm_fused_a_gemm expects BF16 hidden_states and weight, got ", hiddenStates.scalar_type(), " and ",
        weight.scalar_type());
    TORCH_CHECK(hiddenStates.dim() == 2 && weight.dim() == 2,
        "glm_fused_a_gemm expects hidden_states [num_tokens, 6144] and weight [out_features, 6144]");
    TORCH_CHECK(hiddenStates.size(1) == glm::kFusedAHiddenSize && weight.size(1) == glm::kFusedAHiddenSize,
        "glm_fused_a_gemm serves hidden size ", glm::kFusedAHiddenSize, ", got hidden_states ", hiddenStates.sizes(),
        " and weight ", weight.sizes());
    TORCH_CHECK(weight.size(0) == glm::kFusedAOutFeatures || weight.size(0) == glm::kFusedAIndexerOutFeatures,
        "glm_fused_a_gemm serves ", glm::kFusedAOutFeatures, " or ", glm::kFusedAIndexerOutFeatures,
        " output features, got ", weight.size(0));
    TORCH_CHECK(hiddenStates.size(0) >= 1 && hiddenStates.size(0) <= glm::kFusedAMaxTokens,
        "glm_fused_a_gemm takes 1 to ", glm::kFusedAMaxTokens, " tokens, got ", hiddenStates.size(0));
    TORCH_CHECK(hiddenStates.is_contiguous() && weight.is_contiguous(),
        "glm_fused_a_gemm expects contiguous row-major hidden_states and weight");
    TORCH_CHECK(isAligned(hiddenStates) && isAligned(weight),
        "glm_fused_a_gemm expects 16-byte aligned hidden_states and weight (tensor copies)");

    c10::cuda::CUDAGuard const deviceGuard{hiddenStates.device()};
    TORCH_CHECK(tensorrt_llm::common::getSMVersion() >= 90, "glm_fused_a_gemm needs SM90 or later");
    int const numTokens = static_cast<int>(hiddenStates.size(0));
    int const outFeatures = static_cast<int>(weight.size(0));
    torch::Tensor output = torch::empty({numTokens, outFeatures}, hiddenStates.options());
    auto const stream = at::cuda::getCurrentCUDAStream(hiddenStates.get_device());
    auto* outputPtr = reinterpret_cast<__nv_bfloat16*>(output.mutable_data_ptr());
    auto const* hiddenStatesPtr = reinterpret_cast<__nv_bfloat16 const*>(hiddenStates.data_ptr());
    auto const* weightPtr = reinterpret_cast<__nv_bfloat16 const*>(weight.data_ptr());
    if (l2Prefetch.empty())
    {
        glm::invokeFusedAGemm(outputPtr, hiddenStatesPtr, weightPtr, numTokens, outFeatures, nullptr, stream);
    }
    else
    {
        FusedAL2Prefetch const ranges = makeL2Prefetch(l2Prefetch, hiddenStates.device());
        glm::invokeFusedAGemm(outputPtr, hiddenStatesPtr, weightPtr, numTokens, outFeatures, &ranges, stream);
    }
    return output;
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("glm_fused_a_gemm(Tensor hidden_states, Tensor weight, Tensor[] l2_prefetch) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("glm_fused_a_gemm", &tensorrt_llm::torch_ext::glmFusedAGemm);
}
