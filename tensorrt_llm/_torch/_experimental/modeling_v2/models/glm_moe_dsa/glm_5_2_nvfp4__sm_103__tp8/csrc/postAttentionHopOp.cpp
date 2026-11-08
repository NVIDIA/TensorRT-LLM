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

#include "glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/csrc/postAttentionHop.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include "tensorrt_llm/modeling_v2/common/torchLibrary.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <cstdlib>
#include <cstring>
#include <optional>
#include <vector>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

namespace tk = TRTLLM_MV2_NS::post_attention_hop;

// Same switch as the native kernels: PDL is on unless TRTLLM_ENABLE_PDL=0.
bool pdlEnabled()
{
    static bool const enabled = []()
    {
        char const* env = std::getenv("TRTLLM_ENABLE_PDL");
        return !(env != nullptr && std::strcmp(env, "0") == 0);
    }();
    return enabled;
}

void checkRows(at::Tensor const& t, char const* name)
{
    TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kBFloat16, "[glm_moe_dsa_post_attention_hop] ", name,
        " must be a CUDA bf16 tensor");
    TORCH_CHECK(t.is_contiguous(), "[glm_moe_dsa_post_attention_hop] ", name, " must be contiguous");
    TORCH_CHECK(t.dim() >= 1 && t.size(-1) == tk::kHiddenDim, "[glm_moe_dsa_post_attention_hop] ", name,
        " must have hidden size ", tk::kHiddenDim);
}

} // namespace

//! Fused post-attention hop: one-shot cross-rank sum of ``input`` + residual + RMSNorm (+ router logits).
//! Returns ``[normed, residual_out, router_logits]``; ``router_logits`` is an empty tensor when ``router_weight`` is
//! not given. See postAttentionHop.h for the arithmetic and the workspace protocol.
std::vector<at::Tensor> postAttentionHop(at::Tensor const& input, at::Tensor const& residual, at::Tensor const& gamma,
    double eps, std::optional<at::Tensor> const& routerWeight, at::Tensor const& ucBuffer, int64_t mcPtr,
    at::Tensor const& flags, int64_t rank, int64_t nRanks)
{
    checkRows(input, "input");
    checkRows(residual, "residual");
    TORCH_CHECK(
        residual.sizes() == input.sizes(), "[glm_moe_dsa_post_attention_hop] residual must have the shape of input");
    checkRows(gamma, "gamma");
    TORCH_CHECK(gamma.dim() == 1, "[glm_moe_dsa_post_attention_hop] gamma must be 1-D");
    int64_t const numRows = input.numel() / tk::kHiddenDim;
    TORCH_CHECK(numRows >= 1 && numRows <= tk::kMaxRows, "[glm_moe_dsa_post_attention_hop] number of rows ", numRows,
        " is outside the supported range [1, ", tk::kMaxRows, "]");
    TORCH_CHECK(nRanks == tk::kNumRanks, "[glm_moe_dsa_post_attention_hop] only TP", tk::kNumRanks,
        " is supported, got ", nRanks);
    TORCH_CHECK(rank >= 0 && rank < nRanks, "[glm_moe_dsa_post_attention_hop] invalid rank ", rank);
    TORCH_CHECK(ucBuffer.is_cuda() && ucBuffer.is_contiguous()
            && static_cast<size_t>(ucBuffer.numel() * ucBuffer.element_size()) >= tk::kWorkspaceBytes,
        "[glm_moe_dsa_post_attention_hop] uc_buffer must be a contiguous CUDA tensor of at least ", tk::kWorkspaceBytes,
        " bytes");
    TORCH_CHECK(
        mcPtr != 0, "[glm_moe_dsa_post_attention_hop] mc_ptr must be the multicast address of uc_buffer's workspace");
    TORCH_CHECK(flags.is_cuda() && flags.is_contiguous() && flags.scalar_type() == at::kInt && flags.numel() >= 4,
        "[glm_moe_dsa_post_attention_hop] flags must be a contiguous CUDA int32 tensor with at least 4 elements");
    TORCH_CHECK(input.get_device() == ucBuffer.get_device() && input.get_device() == flags.get_device(),
        "[glm_moe_dsa_post_attention_hop] all tensors must be on the same device");

    at::Tensor routerLogits;
    if (routerWeight.has_value())
    {
        auto const& w = routerWeight.value();
        TORCH_CHECK(w.is_cuda() && w.scalar_type() == at::kBFloat16 && w.is_contiguous() && w.dim() == 2
                && w.size(0) == tk::kNumExperts && w.size(1) == tk::kHiddenDim,
            "[glm_moe_dsa_post_attention_hop] router_weight must be a contiguous CUDA bf16 tensor of shape [",
            tk::kNumExperts, ", ", tk::kHiddenDim, "]");
        routerLogits = at::empty({numRows, tk::kNumExperts}, input.options().dtype(at::kFloat));
    }
    else
    {
        routerLogits = at::empty({0}, input.options().dtype(at::kFloat));
    }

    c10::cuda::CUDAGuard const deviceGuard(input.device());
    at::Tensor normOut = at::empty_like(input);
    at::Tensor residualOut = at::empty_like(residual);

    tk::PostAttnHopParams params;
    params.input = input.const_data_ptr();
    params.residualIn = residual.const_data_ptr();
    params.gamma = gamma.const_data_ptr();
    params.routerWeight = routerWeight.has_value() ? routerWeight->const_data_ptr() : nullptr;
    params.normOut = normOut.mutable_data_ptr();
    params.residualOut = residualOut.mutable_data_ptr();
    params.routerLogits = routerWeight.has_value() ? routerLogits.mutable_data_ptr<float>() : nullptr;
    params.ucBuffer = ucBuffer.mutable_data_ptr();
    params.mcBuffer = reinterpret_cast<void*>(mcPtr);
    params.flags = reinterpret_cast<uint32_t*>(flags.mutable_data_ptr<int32_t>());
    params.numRows = static_cast<int>(numRows);
    params.rank = static_cast<int>(rank);
    params.eps = static_cast<float>(eps);
    params.enablePdl = pdlEnabled();
    params.stream = at::cuda::getCurrentCUDAStream(input.get_device());

    cudaError_t const status = tk::invokePostAttentionHop(params);
    TORCH_CHECK(
        status == cudaSuccess, "[glm_moe_dsa_post_attention_hop] kernel launch failed: ", cudaGetErrorString(status));
    return {normOut, residualOut, routerLogits};
}

} // namespace torch_ext

TRTLLM_MV2_NAMESPACE_END

TRTLLM_MV2_LIBRARY_FRAGMENT(m)
{
    m.def(
        "post_attention_hop(Tensor input, Tensor residual, Tensor gamma, float eps, Tensor? router_weight, "
        "Tensor uc_buffer, int mc_ptr, Tensor flags, int rank, int nranks) -> Tensor[]");
}

TRTLLM_MV2_LIBRARY_IMPL(CUDA, m)
{
    m.impl("post_attention_hop", &TRTLLM_MV2_NS::torch_ext::postAttentionHop);
}
