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

#include "glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/csrc/postAttentionHopRouting.h"
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

namespace tk = TRTLLM_MV2_NS::hop_routing;

// Same switch as the native kernels: PDL is on unless TRTLLM_ENABLE_PDL=0.
bool routingPdlEnabled()
{
    static bool const enabled = []()
    {
        char const* env = std::getenv("TRTLLM_ENABLE_PDL");
        return !(env != nullptr && std::strcmp(env, "0") == 0);
    }();
    return enabled;
}

void checkBf16(at::Tensor const& t, char const* name, int64_t lastDim)
{
    TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kBFloat16, "[glm_moe_dsa_post_attention_hop_routing] ", name,
        " must be a CUDA bf16 tensor");
    TORCH_CHECK(t.is_contiguous(), "[glm_moe_dsa_post_attention_hop_routing] ", name, " must be contiguous");
    TORCH_CHECK(t.dim() >= 1 && t.size(-1) == lastDim, "[glm_moe_dsa_post_attention_hop_routing] ", name,
        " must have last dimension ", lastDim);
}

void checkWords(at::Tensor const& t, char const* name, int64_t minWords)
{
    TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type() == at::kInt && t.numel() >= minWords,
        "[glm_moe_dsa_post_attention_hop_routing] ", name, " must be a contiguous CUDA int32 tensor with at least ",
        minWords, " elements");
}

} // namespace

//! Pre-MoE hop with top-8 routing: the transport-input hop with router logits, plus the native
//! DeepSeek-V3 routing of every row. Returns ``[normed, residual_out, router_logits, routing]`` where ``routing`` is
//! an int32 ``[2, R, 8]`` tensor: ``routing[0]`` holds the expert ids and ``routing[1]`` the FP32 bits of the
//! bf16-rounded weights, both in descending native key order. See postAttentionHopRouting.h.
std::vector<at::Tensor> postAttentionHopRouting(at::Tensor const& residual, at::Tensor const& gamma, double eps,
    at::Tensor const& routerWeight, at::Tensor const& routingBias, at::Tensor const& ucBuffer, at::Tensor const& flags,
    at::Tensor const& gather, at::Tensor const& zstate, int64_t rank, int64_t nRanks, bool earlyPoll,
    int64_t routerLoad, double routedScaling, std::optional<at::Tensor> const& logitsIn,
    std::optional<at::Tensor> const& stamps)
{
    checkBf16(residual, "residual", tk::kHiddenDim);
    checkBf16(gamma, "gamma", tk::kHiddenDim);
    TORCH_CHECK(gamma.dim() == 1, "[glm_moe_dsa_post_attention_hop_routing] gamma must be 1-D");
    checkBf16(routerWeight, "router_weight", tk::kHiddenDim);
    TORCH_CHECK(routerWeight.dim() == 2 && routerWeight.size(0) == tk::kNumExperts,
        "[glm_moe_dsa_post_attention_hop_routing] router_weight must have shape [", tk::kNumExperts, ", ",
        tk::kHiddenDim, "]");
    checkBf16(routingBias, "routing_bias", tk::kNumExperts);
    TORCH_CHECK(routingBias.dim() == 1, "[glm_moe_dsa_post_attention_hop_routing] routing_bias must be 1-D");
    int64_t const numRows = residual.numel() / tk::kHiddenDim;
    TORCH_CHECK(numRows >= 1 && numRows <= tk::kMaxRows, "[glm_moe_dsa_post_attention_hop_routing] number of rows ",
        numRows, " is outside the supported range [1, ", tk::kMaxRows, "]");
    TORCH_CHECK(nRanks == tk::kNumRanks, "[glm_moe_dsa_post_attention_hop_routing] only TP", tk::kNumRanks,
        " is supported, got ", nRanks);
    TORCH_CHECK(rank >= 0 && rank < nRanks, "[glm_moe_dsa_post_attention_hop_routing] invalid rank ", rank);
    TORCH_CHECK(routerLoad >= 0 && routerLoad <= 2,
        "[glm_moe_dsa_post_attention_hop_routing] router_load must be 0, 1 or 2, got ", routerLoad);
    TORCH_CHECK(ucBuffer.is_cuda() && ucBuffer.is_contiguous()
            && static_cast<size_t>(ucBuffer.numel() * ucBuffer.element_size()) >= tk::kWorkspaceBytes,
        "[glm_moe_dsa_post_attention_hop_routing] uc_buffer must be a contiguous CUDA tensor of at least ",
        tk::kWorkspaceBytes, " bytes");
    checkWords(flags, "flags", 4);
    checkWords(gather, "gather", tk::kGatherWords);
    checkWords(zstate, "zstate", tk::kStateWords);
    auto const device = residual.get_device();
    TORCH_CHECK(gamma.get_device() == device && routerWeight.get_device() == device
            && routingBias.get_device() == device && ucBuffer.get_device() == device && flags.get_device() == device
            && gather.get_device() == device && zstate.get_device() == device,
        "[glm_moe_dsa_post_attention_hop_routing] all tensors must be on the same device");
    if (logitsIn.has_value())
    {
        auto const& l = logitsIn.value();
        TORCH_CHECK(l.is_cuda() && l.get_device() == device && l.scalar_type() == at::kFloat && l.is_contiguous()
                && l.numel() == numRows * tk::kNumExperts,
            "[glm_moe_dsa_post_attention_hop_routing] logits_in must be a contiguous CUDA fp32 tensor of shape [",
            numRows, ", ", tk::kNumExperts, "]");
        TORCH_CHECK(routerLoad == 1, "[glm_moe_dsa_post_attention_hop_routing] logits_in needs router_load=1");
    }
    if (stamps.has_value())
    {
        auto const& s = stamps.value();
        TORCH_CHECK(tk::stampsCompiled(),
            "[glm_moe_dsa_post_attention_hop_routing] stamps need a TLLM_GLM_MOE_DSA_HOP_ROUTING_STAMPS build");
        TORCH_CHECK(s.is_cuda() && s.get_device() == device && s.scalar_type() == at::kLong && s.is_contiguous()
                && s.numel() >= tk::kMaxGridCtas * tk::kNumStamps,
            "[glm_moe_dsa_post_attention_hop_routing] stamps must be a contiguous CUDA int64 tensor with at least ",
            tk::kMaxGridCtas * tk::kNumStamps, " elements");
    }

    c10::cuda::CUDAGuard const deviceGuard(residual.device());
    at::Tensor normOut = at::empty_like(residual);
    at::Tensor residualOut = at::empty_like(residual);
    at::Tensor routerLogits = at::empty({numRows, tk::kNumExperts}, residual.options().dtype(at::kFloat));
    at::Tensor routing = at::empty({2, numRows, tk::kTopK}, residual.options().dtype(at::kInt));

    tk::PostAttentionHopRoutingParams params;
    params.residualIn = residual.const_data_ptr();
    params.gamma = gamma.const_data_ptr();
    params.routerWeight = routerWeight.const_data_ptr();
    params.routingBias = routingBias.const_data_ptr();
    params.logitsIn = logitsIn.has_value() ? logitsIn->const_data_ptr<float>() : nullptr;
    params.normOut = normOut.mutable_data_ptr();
    params.residualOut = residualOut.mutable_data_ptr();
    params.routerLogits = routerLogits.mutable_data_ptr<float>();
    params.topkIds = routing.mutable_data_ptr<int32_t>();
    params.topkWeights = reinterpret_cast<float*>(routing.mutable_data_ptr<int32_t>() + numRows * tk::kTopK);
    params.ucBuffer = ucBuffer.mutable_data_ptr();
    params.flags = reinterpret_cast<uint32_t*>(flags.mutable_data_ptr<int32_t>());
    params.gather = reinterpret_cast<uint32_t*>(gather.mutable_data_ptr<int32_t>());
    params.zstate = reinterpret_cast<uint32_t*>(zstate.mutable_data_ptr<int32_t>());
    params.stamps
        = stamps.has_value() ? reinterpret_cast<unsigned long long*>(stamps->mutable_data_ptr<int64_t>()) : nullptr;
    params.numRows = static_cast<int>(numRows);
    params.rank = static_cast<int>(rank);
    params.eps = static_cast<float>(eps);
    params.routedScaling = static_cast<float>(routedScaling);
    params.enablePdl = routingPdlEnabled();
    params.earlyPoll = earlyPoll;
    params.routerLoad = static_cast<int>(routerLoad);
    params.stream = at::cuda::getCurrentCUDAStream(device);

    cudaError_t const status = tk::invokePostAttentionHopRouting(params);
    TORCH_CHECK(status == cudaSuccess,
        "[glm_moe_dsa_post_attention_hop_routing] kernel launch failed: ", cudaGetErrorString(status));
    return {normOut, residualOut, routerLogits, routing};
}

} // namespace torch_ext

TRTLLM_MV2_NAMESPACE_END

TRTLLM_MV2_LIBRARY_FRAGMENT(m)
{
    m.def(
        "post_attention_hop_routing(Tensor residual, Tensor gamma, float eps, Tensor router_weight, Tensor "
        "routing_bias, "
        "Tensor uc_buffer, Tensor flags, Tensor gather, Tensor zstate, int rank, int nranks, bool early_poll, "
        "int router_load, float routed_scaling, Tensor? logits_in=None, Tensor? stamps=None) -> Tensor[]");
}

TRTLLM_MV2_LIBRARY_IMPL(CUDA, m)
{
    m.impl("post_attention_hop_routing", &TRTLLM_MV2_NS::torch_ext::postAttentionHopRouting);
}
