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

// The stock `fp8_block_scale_moe_runner` logits path (thop/moe/trtllmGen/fp8BlockScaleMoe.cpp) at an explicit tactic
// (tile N, config), routed by fp8BlockScaleMoeFront.h; no routing bias, precomputed top-k, shared experts or groups.

#include "qwen4_exp/qwen3_8_flash_next_fp8__sm_103__unified/csrc/fp8BlockScaleMoeFront.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/kernels/moe/trtllmGen/runner.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include "tensorrt_llm/modeling_v2/common/torchLibrary.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/EmptyTensor.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <tuple>
#include <vector>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

namespace btg = ::batchedGemm::trtllm::gen;
namespace trtllmGenMoe = ::tensorrt_llm::kernels::trtllmGenFp8BlockScaleMoe;
namespace front = TRTLLM_MV2_NS::kernels::fp8_block_scale_moe_front;

using trtllmGenMoe::Routing::RoutingMethodType;

constexpr std::array<int32_t, 5> kSupportedTileN{8, 16, 32, 64, 128}; // FP8BlockScaleMoERunner's mSupportedTileN

// Built as FP8BlockScaleMoERunner builds its runner for the tile, so a tactic (tile N, config) tuned through that
// class names the same FC1 / FC2 kernels here. One per tile, built on first use and kept for the process.
trtllmGenMoe::MoE::Runner& moeRunner(int64_t tileN)
{
    static std::mutex mutex;
    static std::array<std::unique_ptr<trtllmGenMoe::MoE::Runner>, kSupportedTileN.size()> runners;
    auto const it = std::find(kSupportedTileN.begin(), kSupportedTileN.end(), tileN);
    TORCH_CHECK(it != kSupportedTileN.end(), "fp8_block_scale_moe_front: unsupported tile_n ", tileN);
    std::lock_guard<std::mutex> lock(mutex);
    auto& runner = runners[std::distance(kSupportedTileN.begin(), it)];
    if (runner == nullptr)
    {
        runner = std::make_unique<trtllmGenMoe::MoE::Runner>(
            /*dtypeElt=*/btg::Dtype::E4m3, /*useDeepSeekFp8=*/true, /*tileTokensDim=*/static_cast<int32_t>(tileN));
    }
    return *runner;
}

void checkOperands(at::Tensor const& routing_logits, std::optional<at::Tensor> const& hidden_states_bf16,
    at::Tensor const& hidden_states, at::Tensor const& hidden_states_scale, at::Tensor const& gemm1_weights,
    at::Tensor const& gemm1_weights_scale, at::Tensor const& gemm2_weights, at::Tensor const& gemm2_weights_scale,
    int64_t const num_experts, int64_t const top_k, int64_t const intermediate_size, int64_t const local_num_experts,
    int64_t const routing_method_type)
{
    TORCH_CHECK(routing_logits.scalar_type() == at::ScalarType::BFloat16 && routing_logits.is_contiguous(),
        "fp8_block_scale_moe_front: contiguous bf16 routing_logits required");
    TORCH_CHECK(routing_logits.dim() == 2, "routing_logits must be 2D.");
    TORCH_CHECK(routing_logits.sizes()[1] == num_experts, "routing_logits dim1 must match num_experts.");
    TORCH_CHECK(hidden_states.dim() == 2, "hidden_states must be 2D.");
    TORCH_CHECK(routing_logits.sizes()[0] == hidden_states.sizes()[0],
        "routing_logits and hidden_states must have the same number of tokens.");
    TORCH_CHECK(routing_logits.device() == hidden_states.device(),
        "routing_logits and hidden_states must be on the same device");
    // The kernel computes Renormalize (softmax over the top-k logits), so RenormalizeNaive is not taken.
    TORCH_CHECK(static_cast<RoutingMethodType>(routing_method_type) == RoutingMethodType::Renormalize,
        "fp8_block_scale_moe_front: logits-routed Renormalize without bias or fused shared experts only");
    TORCH_CHECK(top_k <= 32 && top_k > 0,
        "Current routing kernel (no groups, renormalize) only supports top_k<=32 && top_k>0.");
    TORCH_CHECK(num_experts > top_k, "num_experts must be greater than top_k");

    int64_t const num_tokens = hidden_states.sizes()[0];
    int64_t const hidden_size = hidden_states.sizes()[1];
    TORCH_CHECK(hidden_states.scalar_type() == at::ScalarType::Float8_e4m3fn, "hidden_states must be fp8.");
    TORCH_CHECK(hidden_states_scale.scalar_type() == at::ScalarType::Float, "hidden_states_scale must be float.");
    TORCH_CHECK(hidden_states_scale.dim() == 2, "hidden_states_scale must be 2D.");
    TORCH_CHECK(hidden_states_scale.sizes()[0] == hidden_size / 128,
        "hidden_states_scale dim0 must match hidden_states dim1 / 128.");
    TORCH_CHECK(hidden_states_scale.sizes()[1] == num_tokens, "hidden_states_scale dim1 must match num_tokens.");
    TORCH_CHECK(gemm1_weights.scalar_type() == at::ScalarType::Float8_e4m3fn, "gemm1_weights must be fp8.");
    TORCH_CHECK(gemm1_weights.dim() == 3, "gemm1_weights must be 3D.");
    TORCH_CHECK(gemm1_weights.sizes()[0] == local_num_experts, "gemm1_weights has incorrect shape.");
    TORCH_CHECK(gemm1_weights.sizes()[1] % 2 == 0, "the second dimension of weights must be even.");
    TORCH_CHECK(intermediate_size == gemm1_weights.sizes()[1] / 2, "intermediate_size has incorrect shape.");
    TORCH_CHECK(
        gemm1_weights.sizes()[2] == hidden_size, "the third dimension of weights must be equal to hidden_size.");
    TORCH_CHECK(gemm1_weights_scale.scalar_type() == at::ScalarType::Float, "gemm1_weights_scale must be float.");
    TORCH_CHECK(gemm1_weights_scale.dim() == 3, "gemm1_weights_scale must be 3D.");
    TORCH_CHECK(gemm1_weights_scale.sizes()[0] == local_num_experts, "gemm1_weights_scale has incorrect dim 0.");
    TORCH_CHECK(intermediate_size % 128 == 0, "the second dimension of weights must be a multiple of 128.");
    TORCH_CHECK(
        gemm1_weights_scale.sizes()[1] == 2 * intermediate_size / 128, "gemm1_weights_scale has incorrect shape.");
    TORCH_CHECK(gemm1_weights_scale.sizes()[2] == hidden_size / 128, "gemm1_weights_scale has incorrect shape.");
    TORCH_CHECK(gemm2_weights.scalar_type() == at::ScalarType::Float8_e4m3fn, "gemm2_weights must be fp8.");
    TORCH_CHECK(gemm2_weights.dim() == 3, "gemm2_weights must be 3D.");
    TORCH_CHECK(gemm2_weights.sizes()[0] == local_num_experts, "gemm2_weights has incorrect shape.");
    TORCH_CHECK(gemm2_weights.sizes()[2] == intermediate_size,
        "the third dimension of weights must be equal to intermediate_size.");
    TORCH_CHECK(gemm2_weights_scale.scalar_type() == at::ScalarType::Float, "gemm2_weights_scale must be float.");
    TORCH_CHECK(gemm2_weights_scale.dim() == 3, "gemm2_weights_scale must be 3D.");
    TORCH_CHECK(gemm2_weights_scale.sizes()[0] == local_num_experts, "gemm2_weights_scale has incorrect dim 0.");
    TORCH_CHECK(gemm2_weights_scale.sizes()[1] == hidden_size / 128, "gemm2_weights_scale has incorrect shape.");
    TORCH_CHECK(gemm2_weights_scale.sizes()[2] == intermediate_size / 128, "gemm2_weights_scale has incorrect shape.");

    TORCH_CHECK(
        front::supports(static_cast<int32_t>(num_tokens), static_cast<int32_t>(num_experts),
            static_cast<int32_t>(top_k), static_cast<int32_t>(local_num_experts), static_cast<int32_t>(hidden_size)),
        "fp8_block_scale_moe_front: unsupported problem size");
    if (hidden_states_bf16.has_value())
    {
        TORCH_CHECK(hidden_states_bf16->scalar_type() == at::ScalarType::BFloat16 && hidden_states_bf16->is_contiguous()
                && hidden_states_bf16->sizes() == hidden_states.sizes(),
            "fp8_block_scale_moe_front: hidden_states_bf16 must be contiguous bf16 of hidden_states' shape");
        TORCH_CHECK(hidden_states.is_contiguous() && hidden_states_scale.is_contiguous(),
            "fp8_block_scale_moe_front: the quantization outputs must be contiguous");
        TORCH_CHECK(hidden_states_bf16->device() == hidden_states.device(),
            "hidden_states_bf16 and hidden_states must be on the same device");
    }
}

// Returns [T, H] bf16. Given `hidden_states_bf16` [T, H], the routing launch first writes its 1x128 e4m3 quantization
// (bitwise `fp8_quantize_1x128`) into `hidden_states` [T, H] and `hidden_states_scale` [H / 128, T] fp32.
at::Tensor fp8BlockScaleMoeFront(at::Tensor const& routing_logits, std::optional<at::Tensor> const& hidden_states_bf16,
    at::Tensor& hidden_states, at::Tensor& hidden_states_scale, at::Tensor const& gemm1_weights,
    at::Tensor const& gemm1_weights_scale, at::Tensor const& gemm2_weights, at::Tensor const& gemm2_weights_scale,
    int64_t const num_experts, int64_t const top_k, int64_t const intermediate_size, int64_t const local_expert_offset,
    int64_t const local_num_experts, int64_t const routing_method_type, int64_t const tile_n, int64_t const config)
{
    c10::cuda::CUDAGuard const deviceGuard{hidden_states.device()};
    TORCH_CHECK(::tensorrt_llm::common::isSM100Family(), "Only SM100f is supported by FP8 block scale MOE");
    TORCH_CHECK(config >= 0, "fp8_block_scale_moe_front: needs an explicit config, got ", config);
    checkOperands(routing_logits, hidden_states_bf16, hidden_states, hidden_states_scale, gemm1_weights,
        gemm1_weights_scale, gemm2_weights, gemm2_weights_scale, num_experts, top_k, intermediate_size,
        local_num_experts, routing_method_type);
    trtllmGenMoe::MoE::Runner& runner = moeRunner(tile_n);
    int64_t const tile_tokens_dim = tile_n;
    int64_t const num_tokens = hidden_states.sizes()[0];
    int64_t const hidden_size = hidden_states.sizes()[1];
    auto const device = hidden_states.device();
    auto const stream = at::cuda::getCurrentCUDAStream(hidden_states.get_device());

    trtllmGenMoe::MoE::MoERunnerArgs args;
    trtllmGenMoe::MoE::MoEWorkspace workspace;

    // The output is bf16 (MoERunnerArgs' default mDtypeOut).
    args.mDtypeElt = btg::Dtype::E4m3;
    args.mDtypeBias = btg::Dtype::Bfloat16;
    args.routing_logits = routing_logits.data_ptr();
    args.hidden_states = hidden_states.data_ptr();
    args.hidden_states_scale = hidden_states_scale.data_ptr<float>();
    args.gemm1_weights = gemm1_weights.data_ptr();
    args.gemm1_weights_scale = gemm1_weights_scale.data_ptr<float>();
    args.gemm2_weights = gemm2_weights.data_ptr();
    args.gemm2_weights_scale = gemm2_weights_scale.data_ptr<float>();
    args.num_tokens = static_cast<int32_t>(num_tokens);
    args.num_experts = static_cast<int32_t>(num_experts);
    args.hidden_size = static_cast<int32_t>(hidden_size);
    args.top_k = static_cast<int32_t>(top_k);
    args.num_fused_shared_experts = 0;
    args.n_group = 0;
    args.topk_group = 0;
    args.local_expert_offset = static_cast<int32_t>(local_expert_offset);
    args.local_num_experts = static_cast<int32_t>(local_num_experts);
    args.routed_scaling_factor = 1.0;
    args.intermediate_size = static_cast<int32_t>(intermediate_size);
    args.mUseDeepSeekFp8 = true;

    int32_t const max_num_padded_tokens = trtllmGenMoe::Routing::getMaxPermutedPaddedCount(
        args.num_tokens, args.top_k, args.num_experts, static_cast<int32_t>(tile_tokens_dim));
    int32_t const max_num_padded_tokens_gemm1 = trtllmGenMoe::Routing::maybeGetMinTokenCount(
        max_num_padded_tokens, 2 * args.intermediate_size, btg::dtypeGetNumBits(args.mDtypeElt));
    // maybeGetMinTokenCount pads to a 128 KiB floor at the given row width, so each buffer passes its own width.
    int32_t const max_num_padded_tokens_activation = trtllmGenMoe::Routing::maybeGetMinTokenCount(
        max_num_padded_tokens, args.intermediate_size, btg::dtypeGetNumBits(args.mDtypeElt));
    int32_t const max_num_padded_tokens_gemm2 = trtllmGenMoe::Routing::maybeGetMinTokenCount(
        max_num_padded_tokens, args.hidden_size, btg::dtypeGetNumBits(args.mDtypeOut));
    at::Tensor total_num_padded_tokens = at::empty({}, at::TensorOptions().device(device).dtype(at::ScalarType::Int));
    at::Tensor expanded_idx_to_permuted_idx
        = at::detail::empty_cuda({num_tokens * top_k}, at::ScalarType::Int, device, std::nullopt);
    // getRouteMapAllocCount: the gemm1 cubins read one element past the end of the route map.
    at::Tensor permuted_idx_to_token_idx
        = at::detail::empty_cuda({trtllmGenMoe::Routing::getRouteMapAllocCount(max_num_padded_tokens)},
            at::ScalarType::Int, device, std::nullopt);
    // The routing writes bf16 weights, which the finalize reads as mDtypeExpW against the bf16 gemm2_output.
    at::Tensor expert_weights
        = at::detail::empty_cuda({num_tokens, top_k}, at::ScalarType::BFloat16, device, std::nullopt);

    // The rgTma TMA-OOB descriptors use OOB_ADDR_GEN_MODE_BASE_128kB on Blackwell, so TMA may probe up to a 128 KiB
    // boundary past the base: gemm1_output, activation_output and gemm2_output need 128 KiB alignment and >= 128 KiB
    // mapped headroom (empty_cuda aligns to 512 B). The SF buffers are loaded via LDGSTS and exempt.
    static constexpr int64_t kTmaOob128k = 128 * 1024;
    std::vector<at::Tensor> tmaOobKeepAlive; // owns the backing storage for the aligned pointers
    auto allocTmaOobWorkspace = [&](int64_t dataBytes) -> void*
    {
        int64_t const allocBytes = dataBytes + 2 * kTmaOob128k; // up-align slack + tail headroom
        at::Tensor buf = at::detail::empty_cuda({allocBytes}, at::ScalarType::Byte, device, std::nullopt);
        uint64_t const base = reinterpret_cast<uint64_t>(buf.data_ptr());
        uint64_t const aligned = (base + kTmaOob128k - 1) & ~(static_cast<uint64_t>(kTmaOob128k) - 1);
        tmaOobKeepAlive.push_back(std::move(buf));
        return reinterpret_cast<void*>(aligned);
    };
    void* gemm1_output_ptr
        = allocTmaOobWorkspace(static_cast<int64_t>(max_num_padded_tokens_gemm1) * (2 * intermediate_size)); // e4m3
    at::Tensor gemm1_output_scale = at::detail::empty_cuda(
        {2 * intermediate_size / 128, max_num_padded_tokens_gemm1}, at::ScalarType::Float, device, std::nullopt);
    void* activation_output_ptr
        = allocTmaOobWorkspace(static_cast<int64_t>(max_num_padded_tokens_activation) * intermediate_size); // e4m3
    at::Tensor activation_output_scale = at::detail::empty_cuda(
        {intermediate_size / 128, max_num_padded_tokens_activation}, at::ScalarType::Float, device, std::nullopt);
    void* gemm2_output_ptr
        = allocTmaOobWorkspace(static_cast<int64_t>(max_num_padded_tokens_gemm2) * args.hidden_size * 2); // bf16

    int32_t const max_num_ctas = trtllmGenMoe::Routing::getMaxNumCtasInBatchDim(
        args.num_tokens, args.top_k, args.num_experts, static_cast<int32_t>(tile_tokens_dim));
    at::Tensor cta_idx_xy_to_batch_idx
        = at::detail::empty_cuda({max_num_ctas}, at::ScalarType::Int, device, std::nullopt);
    at::Tensor cta_idx_xy_to_mn_limit
        = at::detail::empty_cuda({max_num_ctas}, at::ScalarType::Int, device, std::nullopt);
    at::Tensor num_non_exiting_ctas = at::empty({}, at::TensorOptions().device(device).dtype(at::ScalarType::Int));

    front::Data data;
    data.ptrScores = routing_logits.data_ptr();
    data.numTokens = args.num_tokens;
    data.numExperts = args.num_experts;
    data.topK = args.top_k;
    data.tileTokensDim = static_cast<int32_t>(tile_tokens_dim);
    data.localExpertsStartIdx = args.local_expert_offset;
    data.numLocalExperts = args.local_num_experts;
    data.hiddenDim = args.hidden_size;
    data.usePdl = ::tensorrt_llm::common::getEnvEnablePDL();
    data.ptrTopKWeights = expert_weights.data_ptr();
    data.ptrExpandedIdxToPermutedIdx = expanded_idx_to_permuted_idx.data_ptr<int>();
    data.ptrPermutedIdxToTokenIdx = permuted_idx_to_token_idx.data_ptr<int>();
    data.ptrPermutedIdxSize = total_num_padded_tokens.data_ptr<int>();
    data.ptrCtaIdxXyToBatchIdx = cta_idx_xy_to_batch_idx.data_ptr<int>();
    data.ptrCtaIdxXyToMnLimit = cta_idx_xy_to_mn_limit.data_ptr<int>();
    data.ptrNumNonExitingCtas = num_non_exiting_ctas.data_ptr<int>();
    if (hidden_states_bf16.has_value())
    {
        data.ptrHiddenBf16 = hidden_states_bf16->data_ptr();
        data.ptrHiddenFp8 = hidden_states.data_ptr();
        data.ptrHiddenScale = hidden_states_scale.data_ptr<float>();
    }
    front::run(data, stream);

    at::Tensor output
        = at::detail::empty_cuda({num_tokens, hidden_size}, at::ScalarType::BFloat16, device, std::nullopt);

    workspace.total_num_padded_tokens = total_num_padded_tokens.data_ptr<int>();
    workspace.total_max_padded_tokens
        = std::max({max_num_padded_tokens_gemm1, max_num_padded_tokens_activation, max_num_padded_tokens_gemm2});
    workspace.permuted_idx_size = total_num_padded_tokens.data_ptr<int>();
    workspace.expanded_idx_to_permuted_idx = expanded_idx_to_permuted_idx.data_ptr<int>(); // activation / finalize
    workspace.permuted_idx_to_token_idx = permuted_idx_to_token_idx.data_ptr<int>();       // permute + gemm1
    workspace.expert_weights = expert_weights.data_ptr();                                  // finalize
    workspace.cta_idx_xy_to_batch_idx = cta_idx_xy_to_batch_idx.data_ptr<int>();
    workspace.cta_idx_xy_to_mn_limit = cta_idx_xy_to_mn_limit.data_ptr<int>();
    workspace.num_non_exiting_ctas = num_non_exiting_ctas.data_ptr<int>();
    workspace.gemm1_output = gemm1_output_ptr;
    workspace.gemm1_output_scale = gemm1_output_scale.data_ptr<float>();
    workspace.activation_output = activation_output_ptr;
    workspace.activation_output_scale = activation_output_scale.data_ptr<float>();
    workspace.gemm2_output = gemm2_output_ptr;
    workspace.gemm2_output_scale = nullptr;
    args.output = output.data_ptr();
    args.output_scale = nullptr;

    auto const workspace_sizes = runner.getWorkspaceSizeInBytes(args, config);
    at::Tensor workspace_fc1
        = at::detail::empty_cuda({std::get<0>(workspace_sizes)}, at::ScalarType::Char, device, std::nullopt);
    at::Tensor workspace_fc2
        = at::detail::empty_cuda({std::get<1>(workspace_sizes)}, at::ScalarType::Char, device, std::nullopt);
    workspace.bmm1_workspace = workspace_fc1.data_ptr();
    workspace.bmm2_workspace = workspace_fc2.data_ptr();

    runner.run(args, workspace, hidden_states.get_device(), stream, config);
    return output;
}

} // namespace

} // namespace torch_ext

TRTLLM_MV2_NAMESPACE_END

TRTLLM_MV2_LIBRARY_FRAGMENT(m)
{
    m.def(
        "fp8_block_scale_moe_front(Tensor routing_logits, Tensor? hidden_states_bf16, Tensor(a!) hidden_states, "
        "Tensor(b!) hidden_states_scale, Tensor gemm1_weights, Tensor gemm1_weights_scale, Tensor gemm2_weights, "
        "Tensor gemm2_weights_scale, int num_experts, int top_k, int intermediate_size, int local_expert_offset, "
        "int local_num_experts, int routing_method_type, int tile_n, int config) -> Tensor");
}

TRTLLM_MV2_LIBRARY_IMPL(CUDA, m)
{
    m.impl("fp8_block_scale_moe_front", &TRTLLM_MV2_NS::torch_ext::fp8BlockScaleMoeFront);
}
