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

#include "glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/csrc/postAttentionHopFromPartials.h"
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

namespace tk = TRTLLM_MV2_NS::attention_output_into_hop;

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
    TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kBFloat16, "[glm_moe_dsa_post_attention_hop_from_partials] ",
        name, " must be a CUDA bf16 tensor");
    TORCH_CHECK(t.is_contiguous(), "[glm_moe_dsa_post_attention_hop_from_partials] ", name, " must be contiguous");
    TORCH_CHECK(t.dim() >= 1 && t.size(-1) == tk::kHiddenDim, "[glm_moe_dsa_post_attention_hop_from_partials] ", name,
        " must have hidden size ", tk::kHiddenDim);
}

} // namespace

//! Post-attention hop whose rank-local partials were already published into the hop's transport by the attention
//! output tail. ``residual`` gives the row count. Returns ``[normed, residual_out, router_logits]``; ``router_logits``
//! is an empty tensor when ``router_weight`` is not given. See postAttentionHopFromPartials.h.
std::vector<at::Tensor> postAttentionHopFromPartials(at::Tensor const& residual, at::Tensor const& gamma, double eps,
    std::optional<at::Tensor> const& routerWeight, at::Tensor const& ucBuffer, at::Tensor const& flags, int64_t rank,
    int64_t nRanks, bool earlyPoll, int64_t routerLoad)
{
    checkRows(residual, "residual");
    checkRows(gamma, "gamma");
    TORCH_CHECK(gamma.dim() == 1, "[glm_moe_dsa_post_attention_hop_from_partials] gamma must be 1-D");
    int64_t const numRows = residual.numel() / tk::kHiddenDim;
    TORCH_CHECK(numRows >= 1 && numRows <= tk::kMaxRows,
        "[glm_moe_dsa_post_attention_hop_from_partials] number of rows ", numRows,
        " is outside the supported range [1, ", tk::kMaxRows, "]");
    TORCH_CHECK(nRanks == tk::kNumRanks, "[glm_moe_dsa_post_attention_hop_from_partials] only TP", tk::kNumRanks,
        " is supported, got ", nRanks);
    TORCH_CHECK(rank >= 0 && rank < nRanks, "[glm_moe_dsa_post_attention_hop_from_partials] invalid rank ", rank);
    TORCH_CHECK(routerLoad >= 0 && routerLoad <= 2,
        "[glm_moe_dsa_post_attention_hop_from_partials] router_load must be 0, 1 or 2, got ", routerLoad);
    TORCH_CHECK(ucBuffer.is_cuda() && ucBuffer.is_contiguous()
            && static_cast<size_t>(ucBuffer.numel() * ucBuffer.element_size()) >= tk::kWorkspaceBytes,
        "[glm_moe_dsa_post_attention_hop_from_partials] uc_buffer must be a contiguous CUDA tensor of at least ",
        tk::kWorkspaceBytes, " bytes");
    TORCH_CHECK(flags.is_cuda() && flags.is_contiguous() && flags.scalar_type() == at::kInt && flags.numel() >= 4,
        "[glm_moe_dsa_post_attention_hop_from_partials] flags must be a contiguous CUDA int32 tensor with at least 4 "
        "elements");
    TORCH_CHECK(residual.get_device() == ucBuffer.get_device() && residual.get_device() == flags.get_device(),
        "[glm_moe_dsa_post_attention_hop_from_partials] all tensors must be on the same device");

    at::Tensor routerLogits;
    if (routerWeight.has_value())
    {
        auto const& w = routerWeight.value();
        TORCH_CHECK(w.is_cuda() && w.scalar_type() == at::kBFloat16 && w.is_contiguous() && w.dim() == 2
                && w.size(0) == tk::kNumExperts && w.size(1) == tk::kHiddenDim,
            "[glm_moe_dsa_post_attention_hop_from_partials] router_weight must be a contiguous CUDA bf16 tensor of "
            "shape [",
            tk::kNumExperts, ", ", tk::kHiddenDim, "]");
        routerLogits = at::empty({numRows, tk::kNumExperts}, residual.options().dtype(at::kFloat));
    }
    else
    {
        routerLogits = at::empty({0}, residual.options().dtype(at::kFloat));
    }

    c10::cuda::CUDAGuard const deviceGuard(residual.device());
    at::Tensor normOut = at::empty_like(residual);
    at::Tensor residualOut = at::empty_like(residual);

    tk::PostAttentionHopFromPartialsParams params;
    params.residualIn = residual.const_data_ptr();
    params.gamma = gamma.const_data_ptr();
    params.routerWeight = routerWeight.has_value() ? routerWeight->const_data_ptr() : nullptr;
    params.normOut = normOut.mutable_data_ptr();
    params.residualOut = residualOut.mutable_data_ptr();
    params.routerLogits = routerWeight.has_value() ? routerLogits.mutable_data_ptr<float>() : nullptr;
    params.ucBuffer = ucBuffer.mutable_data_ptr();
    params.flags = reinterpret_cast<uint32_t*>(flags.mutable_data_ptr<int32_t>());
    params.numRows = static_cast<int>(numRows);
    params.rank = static_cast<int>(rank);
    params.eps = static_cast<float>(eps);
    params.enablePdl = pdlEnabled();
    params.earlyPoll = earlyPoll;
    params.routerLoad = static_cast<int>(routerLoad);
    params.stream = at::cuda::getCurrentCUDAStream(residual.get_device());

    cudaError_t const status = tk::invokePostAttentionHopFromPartials(params);
    TORCH_CHECK(status == cudaSuccess,
        "[glm_moe_dsa_post_attention_hop_from_partials] kernel launch failed: ", cudaGetErrorString(status));
    return {normOut, residualOut, routerLogits};
}

} // namespace torch_ext

TRTLLM_MV2_NAMESPACE_END

TRTLLM_MV2_LIBRARY_FRAGMENT(m)
{
    m.def(
        "post_attention_hop_from_partials(Tensor residual, Tensor gamma, float eps, Tensor? router_weight, "
        "Tensor uc_buffer, Tensor flags, int rank, int nranks, bool early_poll, int router_load=1) -> Tensor[]");
}

TRTLLM_MV2_LIBRARY_IMPL(CUDA, m)
{
    m.impl("post_attention_hop_from_partials", &TRTLLM_MV2_NS::torch_ext::postAttentionHopFromPartials);
}
