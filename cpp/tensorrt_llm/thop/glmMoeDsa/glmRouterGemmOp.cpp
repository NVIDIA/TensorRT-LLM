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

#include "tensorrt_llm/kernels/glmMoeDsa/glmRouterGemm.h"

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

// The kernel reads both operands in 16-byte vectors.
constexpr int64_t kVectorBytes = 16;

bool isAligned(torch::Tensor const& tensor)
{
    return reinterpret_cast<uintptr_t>(tensor.data_ptr()) % kVectorBytes == 0;
}

} // namespace

//! The GLM-5 MoE router logits for up to 16 tokens.
//!
//! @param hiddenStates BF16 CUDA tensor [num_tokens, 6144], contiguous, num_tokens in [1, 16].
//! @param weight BF16 CUDA tensor [256, 6144], contiguous: the router (gate) weight as stored.
//! @return FP32 tensor [num_tokens, 256] = hiddenStates @ weight^T, FP32 accumulation.
torch::Tensor glmRouterGemm(torch::Tensor const& hiddenStates, torch::Tensor const& weight)
{
    namespace glm = tk::glm_moe_dsa;
    TORCH_CHECK(hiddenStates.is_cuda() && weight.is_cuda(), "glm_router_gemm expects CUDA tensors");
    TORCH_CHECK(hiddenStates.device() == weight.device(), "hidden_states and weight must be on the same device");
    TORCH_CHECK(hiddenStates.scalar_type() == torch::kBFloat16 && weight.scalar_type() == torch::kBFloat16,
        "glm_router_gemm expects BF16 hidden_states and weight, got ", hiddenStates.scalar_type(), " and ",
        weight.scalar_type());
    TORCH_CHECK(hiddenStates.dim() == 2 && weight.dim() == 2,
        "glm_router_gemm expects hidden_states [num_tokens, 6144] and weight [256, 6144]");
    TORCH_CHECK(hiddenStates.size(1) == glm::kRouterHiddenSize && weight.size(1) == glm::kRouterHiddenSize,
        "glm_router_gemm serves hidden size ", glm::kRouterHiddenSize, ", got hidden_states ", hiddenStates.sizes(),
        " and weight ", weight.sizes());
    TORCH_CHECK(weight.size(0) == glm::kRouterNumExperts, "glm_router_gemm serves ", glm::kRouterNumExperts,
        " experts, got ", weight.size(0));
    TORCH_CHECK(hiddenStates.size(0) >= 1 && hiddenStates.size(0) <= glm::kRouterMaxTokens,
        "glm_router_gemm takes 1 to ", glm::kRouterMaxTokens, " tokens, got ", hiddenStates.size(0));
    TORCH_CHECK(hiddenStates.is_contiguous() && weight.is_contiguous(),
        "glm_router_gemm expects contiguous row-major hidden_states and weight");
    TORCH_CHECK(isAligned(hiddenStates) && isAligned(weight),
        "glm_router_gemm expects 16-byte aligned hidden_states and weight");

    c10::cuda::CUDAGuard const deviceGuard{hiddenStates.device()};
    int const numTokens = static_cast<int>(hiddenStates.size(0));
    torch::Tensor logits
        = torch::empty({numTokens, glm::kRouterNumExperts}, hiddenStates.options().dtype(torch::kFloat32));
    auto const stream = at::cuda::getCurrentCUDAStream(hiddenStates.get_device());
    glm::invokeRouterGemm(logits.data_ptr<float>(), reinterpret_cast<__nv_bfloat16 const*>(hiddenStates.data_ptr()),
        reinterpret_cast<__nv_bfloat16 const*>(weight.data_ptr()), numTokens, stream);
    return logits;
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("glm_router_gemm(Tensor hidden_states, Tensor weight) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("glm_router_gemm", &tensorrt_llm::torch_ext::glmRouterGemm);
}
