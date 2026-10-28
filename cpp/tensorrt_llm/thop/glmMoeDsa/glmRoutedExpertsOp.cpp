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
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/kernels/glmMoeDsa/glmMoeFusedRouter.h"
#include "tensorrt_llm/kernels/glmMoeDsa/glmMoeGemm2K128.h"
#include "tensorrt_llm/kernels/moe/trtllmGen/DevKernel.h"
#include "tensorrt_llm/kernels/moe/trtllmGen/runner.h"
#include "tensorrt_llm/kernels/quantization.h"
#include "tensorrt_llm/runtime/torchUtils.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/EmptyTensor.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <tuple>
#include <vector>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{
namespace
{

namespace btg = batchedGemm::trtllm::gen;
namespace trtllmGenMoe = tensorrt_llm::kernels::trtllmGenFp8BlockScaleMoe;
namespace glmKernels = tensorrt_llm::kernels::glm_moe_dsa;

using trtllmGenMoe::Routing::RoutingMethodType;

// FC1 and FC2 of GLM-5's routed experts for one token tile size: the TRTLLM-Gen permute + FC1 + SwiGLU + NVFP4
// requantization kernel, then FC2 on the TRTLLM-Gen kernel or on glmMoeGemm2K128 where it covers the call. A config
// is a pair of passing FC1 and FC2 kernel configs, enumerated as the TRTLLM-Gen MoE runner enumerates them.
class RoutedExpertsGemms
{
public:
    RoutedExpertsGemms(int32_t tileTokensDim, tensorrt_llm::kernels::ActType actType)
        : mPermuteGemm1(btg::Dtype::E2m1, btg::Dtype::E2m1, /* useDeepSeekFp8 */ false, tileTokensDim, actType,
            /* useFineGrained */ false)
        , mGemm2(btg::Dtype::E2m1, btg::Dtype::E2m1, btg::Dtype::Bfloat16, /* useDeepSeekFp8 */ false, tileTokensDim,
              /* useFineGrained */ false)
        , mTileTokensDim(tileTokensDim)
    {
        for (auto const indexGemm1 : mPermuteGemm1.getPassingConfigIndices())
        {
            for (auto const indexGemm2 : mGemm2.getPassingConfigIndices())
            {
                mConfigs.push_back(trtllmGenMoe::MoE::MoEConfig{indexGemm1, indexGemm2});
            }
        }
        TLLM_CHECK_WITH_INFO(!mConfigs.empty(), "No FC1 / FC2 config pair for tile N %d.", tileTokensDim);
    }

    [[nodiscard]] std::vector<int64_t> getValidConfigIndices(
        int32_t topK, int32_t hiddenSize, int32_t intermediateSize, int32_t numLocalExperts, int32_t numTokens) const
    {
        std::vector<int64_t> validIndices;
        for (size_t i = 0; i < mConfigs.size(); ++i)
        {
            if (mPermuteGemm1.isValidConfigIndex(
                    mConfigs[i].gemm1Config, topK, hiddenSize, intermediateSize, numLocalExperts, numTokens)
                && mGemm2.isValidConfigIndex(
                    mConfigs[i].gemm2Config, topK, hiddenSize, intermediateSize, numLocalExperts, numTokens))
            {
                validIndices.push_back(static_cast<int64_t>(i));
            }
        }
        return validIndices;
    }

    [[nodiscard]] int64_t getDefaultValidConfigIndex(
        int32_t topK, int32_t hiddenSize, int32_t intermediateSize, int32_t numLocalExperts, int32_t numTokens) const
    {
        int64_t const indexGemm1
            = mPermuteGemm1.getDefaultValidConfigIndex(topK, hiddenSize, intermediateSize, numLocalExperts, numTokens);
        int64_t const indexGemm2
            = mGemm2.getDefaultValidConfigIndex(topK, hiddenSize, intermediateSize, numLocalExperts, numTokens);
        auto const it = std::find_if(mConfigs.begin(), mConfigs.end(),
            [&](trtllmGenMoe::MoE::MoEConfig const& config)
            { return config.gemm1Config == indexGemm1 && config.gemm2Config == indexGemm2; });
        TLLM_CHECK_WITH_INFO(it != mConfigs.end(), "No FC1 / FC2 config pair holds the default configs.");
        return std::distance(mConfigs.begin(), it);
    }

    [[nodiscard]] std::tuple<int64_t, int64_t> getWorkspaceSizeInBytes(int32_t topK, int32_t hiddenSize,
        int32_t intermediateSize, int32_t numLocalExperts, int32_t numTokens, int64_t configIndex) const
    {
        auto const& config = mConfigs[configIndex];
        return {static_cast<int64_t>(mPermuteGemm1.getWorkspaceSizeInBytes(
                    topK, hiddenSize, intermediateSize, numLocalExperts, numTokens, config.gemm1Config)),
            static_cast<int64_t>(mGemm2.getWorkspaceSizeInBytes(
                topK, hiddenSize, intermediateSize, numLocalExperts, numTokens, config.gemm2Config))};
    }

    // hiddenStatesReady: an event recorded once hidden_states and their scales are written, or null. FC1, their first
    // reader, waits on it, so the routing launched before FC1 does not.
    void run(trtllmGenMoe::MoE::MoERunnerArgs const& args, trtllmGenMoe::MoE::MoEWorkspace const& workspace,
        cudaEvent_t hiddenStatesReady, bool preferGemm2K128, int device, cudaStream_t stream, int64_t configIndex)
    {
        sync_check_cuda_error(stream);
        auto const& config = mConfigs[configIndex];
        int32_t const validHiddenSize = args.valid_hidden_size.value_or(args.hidden_size);
        int32_t const validIntermediateSize = args.valid_intermediate_size.value_or(args.intermediate_size);

        if (hiddenStatesReady != nullptr)
        {
            TLLM_CUDA_CHECK(cudaStreamWaitEvent(stream, hiddenStatesReady, 0));
        }
        mPermuteGemm1.run(args.hidden_states, args.hidden_states_scale, args.gemm1_weights, args.gemm1_weights_scale,
            workspace.expert_weights, args.output1_scales_scalar, args.output1_scales_gate_scalar, args.gemm1_bias,
            args.gemm1_alpha, args.gemm1_beta, args.gemm1_clamp_limit, workspace.gemm1_output,
            workspace.gemm1_output_scale, args.top_k, args.hidden_size, args.intermediate_size, args.local_num_experts,
            args.num_tokens, workspace.permuted_idx_to_token_idx, workspace.num_non_exiting_ctas,
            workspace.total_num_padded_tokens, workspace.cta_idx_xy_to_batch_idx, workspace.cta_idx_xy_to_mn_limit,
            workspace.bmm1_workspace, args.mUseRoutingScalesOnInput, device, stream, config.gemm1Config,
            validHiddenSize, validIntermediateSize);

        int32_t const gemm2HiddenSize = args.output_hidden_size.value_or(args.hidden_size);
        int32_t const gemm2MaxNumCtas = trtllmGenMoe::Routing::getMaxNumCgasInBatchDim(
            args.num_tokens, args.top_k, args.local_num_experts, mTileTokensDim);
        // glmMoeGemm2K128 covers FC2 at intermediate size 128 and tile N 8 (GLM-5 decode at MoE TP16), with the output
        // rows of the TRTLLM-Gen FC2 kernel, bitwise.
        if (preferGemm2K128 && mTileTokensDim == 8 && args.intermediate_size == 128
            && validIntermediateSize == args.intermediate_size && validHiddenSize == gemm2HiddenSize
            && args.gemm2_bias == nullptr && args.output2_scales_scalar != nullptr
            && glmKernels::gemm2K128::isSupported(gemm2HiddenSize, gemm2MaxNumCtas))
        {
            glmKernels::gemm2K128::Data gemm2Data{};
            gemm2Data.weights = args.gemm2_weights;
            gemm2Data.weightsScale = args.gemm2_weights_scale;
            gemm2Data.act = workspace.gemm1_output;
            gemm2Data.actScale = workspace.gemm1_output_scale;
            gemm2Data.outputScale = args.output2_scales_scalar;
            gemm2Data.numNonExitingCtas = workspace.num_non_exiting_ctas;
            gemm2Data.ctaIdxXyToBatchIdx = workspace.cta_idx_xy_to_batch_idx;
            gemm2Data.ctaIdxXyToMnLimit = workspace.cta_idx_xy_to_mn_limit;
            gemm2Data.output = workspace.gemm2_output;
            gemm2Data.hiddenSize = gemm2HiddenSize;
            gemm2Data.numLocalExperts = args.local_num_experts;
            gemm2Data.maxNumCtas = gemm2MaxNumCtas;
            gemm2Data.usePdl = tensorrt_llm::common::getEnvEnablePDL();
            // The maximum shared-memory carveout preferred. Without a preference the driver can run the 141 KB CTAs in
            // the 164 KB configuration. A kernel that FC2 lets launch early (for example an all-reduce hop with the
            // finalize step fused) then shares those SMs and keeps them in that configuration until it ends, so the
            // next layer's large-shared-memory CTAs that land there start only after it. A scheduling hint only: the
            // outputs are unchanged.
            gemm2Data.sharedMemCarveout = static_cast<int32_t>(cudaSharedmemCarveoutMaxShared);
            glmKernels::gemm2K128::run(gemm2Data, stream);
        }
        else
        {
            mGemm2.run(workspace.gemm1_output, workspace.gemm1_output_scale, args.gemm2_weights,
                args.gemm2_weights_scale, args.output2_scales_scalar, args.gemm2_bias, workspace.gemm2_output,
                workspace.gemm2_output_scale, args.top_k, gemm2HiddenSize, args.intermediate_size,
                args.local_num_experts, args.num_tokens, workspace.num_non_exiting_ctas,
                workspace.total_num_padded_tokens, workspace.cta_idx_xy_to_batch_idx, workspace.cta_idx_xy_to_mn_limit,
                workspace.bmm2_workspace, device, stream, config.gemm2Config, validHiddenSize, validIntermediateSize);
        }

        if (args.do_finalize)
        {
            ::moe::dev::finalize::Data finalizeData;
            finalizeData.mDtypeElt = args.mDtypeOut;
            finalizeData.mDtypeExpW = args.mDtypeExpW;
            finalizeData.mUsePdl = tensorrt_llm::common::getEnvEnablePDL();
            finalizeData.mUseDeepSeekFp8 = false;
            finalizeData.inPtr = workspace.gemm2_output;
            finalizeData.outPtr = args.output;
            finalizeData.inDqSfsPtr = workspace.gemm2_output_scale;
            finalizeData.outDqSfsPtr = args.output_scale;
            finalizeData.expertWeightsPtr = workspace.expert_weights;
            finalizeData.expandedIdxToPermutedIdx = workspace.expanded_idx_to_permuted_idx;
            finalizeData.numTokens = args.num_tokens;
            finalizeData.numExperts = args.num_experts;
            finalizeData.topK = args.top_k;
            // The finalize step drops the hidden padding: it reads padded rows and writes the valid columns.
            finalizeData.hiddenDim = validHiddenSize;
            finalizeData.hiddenDimPadded = gemm2HiddenSize;
            finalizeData.totalNumPaddedTokens = workspace.total_num_padded_tokens;
            ::moe::dev::finalize::run(finalizeData, stream);
            sync_check_cuda_error(stream);
        }
    }

private:
    trtllmGenMoe::PermuteGemm1::Runner mPermuteGemm1;
    trtllmGenMoe::Gemm2::Runner mGemm2;
    int32_t mTileTokensDim;
    std::vector<trtllmGenMoe::MoE::MoEConfig> mConfigs;
};

std::vector<torch::Tensor> runRoutedExperts(torch::optional<torch::Tensor> const& routing_logits,
    torch::optional<torch::Tensor> const& routing_bias, torch::Tensor const& hidden_states,
    torch::Tensor const& hidden_states_scale, torch::Tensor const& gemm1_weights,
    torch::Tensor const& gemm1_weights_scale, std::optional<torch::Tensor> const& gemm1_bias,
    std::optional<torch::Tensor> const& gemm1_alpha, std::optional<torch::Tensor> const& gemm1_beta,
    std::optional<torch::Tensor> const& gemm1_clamp_limit, torch::Tensor const& gemm2_weights,
    torch::Tensor const& gemm2_weights_scale, std::optional<torch::Tensor> const& gemm2_bias,
    torch::Tensor const& output1_scales_scalar, torch::Tensor const& output1_scales_gate_scalar,
    torch::Tensor const& output2_scales_scalar, int64_t const num_experts, int64_t const top_k,
    std::optional<int64_t> const n_group, std::optional<int64_t> const topk_group, int64_t const intermediate_size,
    int64_t const local_expert_offset, int64_t const local_num_experts,
    std::optional<double> const routed_scaling_factor, int64_t const tile_tokens_dim, int64_t const routing_method_type,
    bool const do_finalize, RoutedExpertsGemms& gemms, int64_t const configIndex,
    torch::optional<torch::Tensor> const& topk_weights, torch::optional<torch::Tensor> const& topk_ids,
    torch::optional<torch::Tensor> const& out_tensor, std::optional<int64_t> const hidden_states_ready_event,
    torch::optional<torch::Tensor> const& router_hidden_states, torch::optional<torch::Tensor> const& router_weight,
    torch::optional<torch::Tensor> const& router_workspace, bool const prefer_gemm2_k128)
{
    TORCH_CHECK(tensorrt_llm::common::isSM100Family(), "GLM routed experts run on SM100-family GPUs only.");
    TORCH_CHECK(tile_tokens_dim == 8 || tile_tokens_dim == 16 || tile_tokens_dim == 32 || tile_tokens_dim == 64
            || tile_tokens_dim == 128 || tile_tokens_dim == 256,
        "tile_tokens_dim must be 8, 16, 32, 64, 128, 256");

    if (topk_ids.has_value() && topk_weights.has_value())
    {
        TORCH_CHECK(topk_ids.value().scalar_type() == at::ScalarType::Int, "topk_ids must be int");
        TORCH_CHECK(topk_weights.value().scalar_type() == at::ScalarType::BFloat16, "topk_weights must be bfloat16.");
        TORCH_CHECK(topk_ids.value().dim() == 2, "topk_ids must be 2D.");
        TORCH_CHECK(topk_ids.value().sizes()[0] == hidden_states.sizes()[0],
            "topk_ids and hidden_states must have the same number of tokens.");
        TORCH_CHECK(topk_ids.value().sizes()[1] == top_k, "topk_ids dim1 must match top_k.");
        TORCH_CHECK(topk_weights.value().dim() == 2, "topk_weights must be 2D.");
        TORCH_CHECK(topk_weights.value().sizes()[0] == hidden_states.sizes()[0],
            "topk_weights and hidden_states must have the same number of tokens.");
        TORCH_CHECK(topk_weights.value().sizes()[1] == top_k, "topk_weights dim1 must match top_k.");
    }
    else if (routing_logits.has_value())
    {
        TORCH_CHECK(routing_logits.value().scalar_type() == at::ScalarType::BFloat16
                || routing_logits.value().scalar_type() == at::ScalarType::Float,
            "routing_logits must be bfloat16 or float32");
        TORCH_CHECK(routing_logits.value().dim() == 2, "routing_logits must be 2D.");
        TORCH_CHECK(routing_logits.value().sizes()[1] == num_experts, "routing_logits has incorrect shape.");
        TORCH_CHECK(routing_logits.value().sizes()[0] == hidden_states.sizes()[0],
            "routing_logits and hidden_states must have the same number of tokens.");
    }
    else
    {
        TORCH_CHECK(false, "routing_logits or (topk_ids and topk_weights) must be provided.");
    }

    if (routing_bias.has_value())
    {
        TORCH_CHECK(routing_bias.value().scalar_type() == at::ScalarType::BFloat16
                || routing_bias.value().scalar_type() == at::ScalarType::Float,
            "routing_bias must be bfloat16 or float32.");
        TORCH_CHECK(routing_bias.value().dim() == 1, "routing_bias must be 1D.");
        TORCH_CHECK(routing_bias.value().sizes()[0] == num_experts, "routing_bias has incorrect shape.");
    }

    if (n_group.has_value() && n_group.value() > 1)
    {
        TORCH_CHECK(static_cast<RoutingMethodType>(routing_method_type) == RoutingMethodType::DeepSeekV3,
            "Routing kernel with groups implies DeepSeekV3 routing method.");
        TORCH_CHECK(topk_group.has_value(), "if n_group is given, topk_group must be given");
        TORCH_CHECK(num_experts % n_group.value() == 0, "num_experts must be divisible by n_group");
        TORCH_CHECK(top_k <= 8 && top_k > 0, "Current routing kernel (with groups) only supports top_k<=8 && top_k>0.");
        TORCH_CHECK(topk_group.value() <= 4 && topk_group.value() > 0,
            "Current routing kernel only (with groups) supports topk_group<=4 && topk_group > 0.");
        TORCH_CHECK(topk_group.value() <= n_group.value(), "n_group must not be smaller than topk_group.");
        TORCH_CHECK(top_k < (topk_group.value() * num_experts / n_group.value()),
            "top_k must be less than total number of experts in selected groups");
    }
    TORCH_CHECK(num_experts > top_k, "num_experts must be greater than top_k");
    TORCH_CHECK(num_experts <= 2048, "num_experts must be less than or equal to 2048");
    if (routing_logits.has_value() && topk_ids.has_value())
    {
        TORCH_CHECK(
            routing_logits->device() == topk_ids->device(), "routing_logits and topk_ids must be on the same device");
    }

    trtllmGenMoe::MoE::MoERunnerArgs args;
    trtllmGenMoe::MoE::MoEWorkspace workspace;

    // The output is bf16 (MoERunnerArgs' default mDtypeOut).
    args.mDtypeElt = btg::Dtype::E2m1;
    args.mDtypeBias = routing_bias.has_value() && routing_bias.value().scalar_type() == at::ScalarType::Float
        ? btg::Dtype::Fp32
        : btg::Dtype::Bfloat16;
    args.routing_logits = routing_logits.has_value() ? routing_logits.value().data_ptr() : nullptr;
    args.routing_bias = routing_bias.has_value() ? routing_bias.value().data_ptr() : nullptr;
    args.hidden_states = hidden_states.data_ptr();
    args.hidden_states_scale = hidden_states_scale.data_ptr();
    args.topk_weights = topk_weights.has_value() ? topk_weights.value().data_ptr() : nullptr;
    args.topk_ids = topk_ids.has_value() ? static_cast<int32_t*>(topk_ids.value().data_ptr()) : nullptr;
    args.gemm1_weights = gemm1_weights.data_ptr();
    args.gemm1_weights_scale = gemm1_weights_scale.data_ptr();
    args.gemm1_bias = gemm1_bias.has_value() ? gemm1_bias.value().data_ptr<float>() : nullptr;
    args.gemm1_alpha = gemm1_alpha.has_value() ? gemm1_alpha.value().data_ptr<float>() : nullptr;
    args.gemm1_beta = gemm1_beta.has_value() ? gemm1_beta.value().data_ptr<float>() : nullptr;
    args.gemm1_clamp_limit = gemm1_clamp_limit.has_value() ? gemm1_clamp_limit.value().data_ptr<float>() : nullptr;
    args.gemm2_weights = gemm2_weights.data_ptr();
    args.gemm2_weights_scale = gemm2_weights_scale.data_ptr();
    args.gemm2_bias = gemm2_bias.has_value() ? gemm2_bias.value().data_ptr<float>() : nullptr;
    args.num_tokens = hidden_states.sizes()[0];
    args.num_experts = num_experts;
    // Two e2m1 values per byte.
    args.hidden_size = hidden_states.sizes()[1] * 2;
    args.top_k = top_k;
    args.n_group = n_group.value_or(0);
    args.topk_group = topk_group.value_or(0);
    args.local_expert_offset = local_expert_offset;
    args.local_num_experts = local_num_experts;
    args.routed_scaling_factor = routed_scaling_factor.value_or(1.0);
    args.intermediate_size = intermediate_size;

    // Routing outputs and the FC1 / FC2 buffers, sized for the experts of this rank.
    auto const routing_device
        = routing_logits.has_value() ? routing_logits.value().device() : topk_ids.value().device();
    at::Tensor num_tokens_per_expert
        = at::detail::empty_cuda({num_experts}, at::ScalarType::Int, routing_device, std::nullopt);
    int32_t const max_num_padded_tokens
        = trtllmGenMoe::Routing::getMaxPermutedPaddedCount(args.num_tokens, top_k, local_num_experts, tile_tokens_dim);
    int32_t const max_num_padded_tokens_gemm1 = trtllmGenMoe::Routing::maybeGetMinTokenCount(
        max_num_padded_tokens, args.intermediate_size, btg::dtypeGetNumBits(args.mDtypeElt));
    int32_t const max_num_padded_tokens_gemm2 = trtllmGenMoe::Routing::maybeGetMinTokenCount(
        max_num_padded_tokens, args.hidden_size, btg::dtypeGetNumBits(args.mDtypeOut));
    at::Tensor total_num_padded_tokens
        = at::empty({}, at::TensorOptions().device(routing_device).dtype(at::ScalarType::Int));
    at::Tensor expanded_idx_to_permuted_idx
        = at::detail::empty_cuda({args.num_tokens, args.top_k}, at::ScalarType::Int, routing_device, std::nullopt);
    // getRouteMapAllocCount: the FC1 kernels read one element past the end of the route map.
    at::Tensor permuted_idx_to_token_idx
        = at::detail::empty_cuda({trtllmGenMoe::Routing::getRouteMapAllocCount(max_num_padded_tokens)},
            at::ScalarType::Int, routing_device, std::nullopt);
    // The routing's expert weights, read by the finalize step in the dtype of the FC2 output.
    at::Tensor expert_weights
        = at::detail::empty_cuda({args.num_tokens, args.top_k}, at::ScalarType::BFloat16, routing_device, std::nullopt);
    at::Tensor expert_indexes
        = at::detail::empty_cuda({args.num_tokens, args.top_k}, at::ScalarType::Int, routing_device, std::nullopt);
    int64_t const size_of_expert_count_histogram = std::max(num_experts * 2, int64_t(256 * 2));
    at::Tensor expert_count_histogram
        = at::detail::empty_cuda({size_of_expert_count_histogram}, at::ScalarType::Int, routing_device, std::nullopt);

    at::Tensor gemm1_output = at::detail::empty_cuda({max_num_padded_tokens_gemm1, intermediate_size / 2},
        at::ScalarType::Float8_e4m3fn, routing_device, std::nullopt);
    at::Tensor gemm1_output_scale = at::detail::empty_cuda(
        {tensorrt_llm::computeSwizzledLayoutSFSize(max_num_padded_tokens_gemm1, intermediate_size / 16)},
        at::ScalarType::Float8_e4m3fn, routing_device, std::nullopt);
    at::Tensor gemm2_output = at::detail::empty_cuda(
        {max_num_padded_tokens_gemm2, args.hidden_size}, at::ScalarType::BFloat16, routing_device, std::nullopt);

    int32_t const max_num_ctas = trtllmGenMoe::Routing::getMaxNumCtasInBatchDim(
        args.num_tokens, args.top_k, args.local_num_experts, tile_tokens_dim);
    at::Tensor cta_idx_xy_to_batch_idx
        = at::detail::empty_cuda({max_num_ctas}, at::ScalarType::Int, routing_device, std::nullopt);
    at::Tensor cta_idx_xy_to_mn_limit
        = at::detail::empty_cuda({max_num_ctas}, at::ScalarType::Int, routing_device, std::nullopt);
    at::Tensor num_non_exiting_ctas
        = at::empty({}, at::TensorOptions().device(routing_device).dtype(at::ScalarType::Int));

    void* expert_weights_ptr = args.topk_weights ? args.topk_weights : expert_weights.data_ptr();

    //
    // Routing
    //

    auto const& stream = at::cuda::getCurrentCUDAStream(
        routing_logits.has_value() ? routing_logits.value().get_device() : topk_ids.value().get_device());
    if (router_hidden_states.has_value())
    {
        // The router GEMV runs in the routing kernel: routing_logits only gives the shape and is not read or written.
        TORCH_CHECK(router_weight.has_value() && router_workspace.has_value(),
            "router_hidden_states needs router_weight and router_workspace.");
        TORCH_CHECK(!args.topk_ids
                && static_cast<RoutingMethodType>(routing_method_type) == RoutingMethodType::DeepSeekV3
                && args.n_group <= 1 && routing_bias.has_value(),
            "The fused router covers DeepSeekV3 routing without groups, with a routing bias, from the router inputs.");
        auto const& hs = router_hidden_states.value();
        auto const& rw = router_weight.value();
        auto const& ws = router_workspace.value();
        TORCH_CHECK(hs.scalar_type() == at::ScalarType::BFloat16 && hs.dim() == 2 && hs.is_contiguous()
                && hs.sizes()[0] == args.num_tokens && hs.sizes()[1] == args.hidden_size
                && hs.device() == routing_device,
            "router_hidden_states must be a contiguous bf16 [num_tokens, hidden_size] tensor on the routing device.");
        TORCH_CHECK(rw.scalar_type() == at::ScalarType::BFloat16 && rw.dim() == 2 && rw.is_contiguous()
                && rw.sizes()[0] == num_experts && rw.sizes()[1] == hs.sizes()[1] && rw.device() == routing_device,
            "router_weight must be a contiguous bf16 [num_experts, hidden] tensor on the routing device.");
        TORCH_CHECK(ws.scalar_type() == at::ScalarType::Int && ws.is_contiguous()
                && ws.numel() == glmKernels::kFusedRouterWorkspaceWords && ws.device() == routing_device,
            "router_workspace must be a contiguous int32 tensor of ", glmKernels::kFusedRouterWorkspaceWords,
            " words on the routing device.");
        glmKernels::invokeFusedRouter(hs.data_ptr(), rw.data_ptr(), static_cast<int32_t>(hs.sizes()[1]),
            ws.data_ptr<int>(), args.routing_bias, args.mDtypeBias, args.num_tokens, args.num_experts, args.top_k,
            args.local_expert_offset, args.local_num_experts, args.routed_scaling_factor,
            static_cast<int32_t>(tile_tokens_dim), total_num_padded_tokens.data_ptr<int>(),
            expanded_idx_to_permuted_idx.data_ptr<int>(), permuted_idx_to_token_idx.data_ptr<int>(), expert_weights_ptr,
            cta_idx_xy_to_batch_idx.data_ptr<int>(), cta_idx_xy_to_mn_limit.data_ptr<int>(),
            num_non_exiting_ctas.data_ptr<int>(), stream);
    }
    else
    {
        auto const dtypeRoutingLogits
            = routing_logits.has_value() && routing_logits.value().scalar_type() == at::ScalarType::Float
            ? btg::Dtype::Fp32
            : btg::Dtype::Bfloat16;
        trtllmGenMoe::Routing::Runner routing_runner(tile_tokens_dim);
        routing_runner.run(args.routing_logits, args.routing_bias, args.num_tokens, args.num_experts, args.top_k,
            /* num_fused_shared_expert */ 0, args.n_group, args.topk_group, args.local_expert_offset,
            args.local_num_experts, args.routed_scaling_factor, expert_indexes.data_ptr<int>(),
            expert_count_histogram.data_ptr<int>(), total_num_padded_tokens.data_ptr<int>(),
            expanded_idx_to_permuted_idx.data_ptr<int>(), /* permuted_idx_to_expanded_idx */ nullptr,
            permuted_idx_to_token_idx.data_ptr<int>(), expert_weights_ptr, args.topk_ids,
            num_tokens_per_expert.data_ptr<int>(), cta_idx_xy_to_batch_idx.data_ptr<int>(),
            cta_idx_xy_to_mn_limit.data_ptr<int>(), num_non_exiting_ctas.data_ptr<int>(), args.mDtypeElt,
            /* use_routing_scales_on_input */ false, /* use_deep_seek_fp8 */ false,
            static_cast<RoutingMethodType>(routing_method_type), stream, dtypeRoutingLogits, args.mDtypeBias);
    }

    //
    // FC1 + FC2
    //

    TORCH_CHECK(hidden_states.scalar_type() == FLOAT4_E2M1X2, "hidden_states must be byte.");
    TORCH_CHECK(hidden_states_scale.scalar_type() == at::ScalarType::Float8_e4m3fn, "hidden_states_scale must be fp8.");
    TORCH_CHECK(hidden_states_scale.dim() == 1, "hidden_states_scale must be 1D.");
    TORCH_CHECK(hidden_states_scale.sizes()[0]
            == tensorrt_llm::computeLinearLayoutSFSize(args.num_tokens, args.hidden_size / 16),
        "hidden_states_scale has incorrect size");
    TORCH_CHECK(gemm1_weights.scalar_type() == FLOAT4_E2M1X2, "gemm1_weights must be byte.");
    TORCH_CHECK(gemm1_weights.dim() == 3, "gemm1_weights must be 3D.");
    // Both are in bytes, two e2m1 values each.
    TORCH_CHECK(gemm1_weights.sizes()[2] == hidden_states.sizes()[1],
        "the third dimension of weights must be equal to hidden_size.");
    TORCH_CHECK(gemm1_weights.sizes()[1] % 2 == 0, "the second dimension of weights must be even.");
    // Gated activations (SwiGLU): FC1 has 2 * intermediate_size rows per expert; non-gated: intermediate_size.
    TORCH_CHECK(intermediate_size == gemm1_weights.sizes()[1] || 2 * intermediate_size == gemm1_weights.sizes()[1],
        "intermediate_size has incorrect dim 1 (expected: ", intermediate_size, " or ", 2 * intermediate_size,
        ", got: ", gemm1_weights.sizes()[1], ").");
    bool const is_gated_activation = (gemm1_weights.sizes()[1] == 2 * intermediate_size);
    TORCH_CHECK(gemm1_weights_scale.scalar_type() == at::ScalarType::Float8_e4m3fn, "gemm1_weights_scale must be fp8.");
    TORCH_CHECK(gemm1_weights_scale.dim() == 3, "gemm1_weights_scale must be 3D.");
    TORCH_CHECK(gemm1_weights_scale.sizes()[0] == local_num_experts, "gemm1_weights_scale has incorrect dim 0.");
    TORCH_CHECK(intermediate_size % 16 == 0, "the second dimension of weights must be a multiple of 16.");
    TORCH_CHECK(gemm1_weights_scale.sizes()[1] == (is_gated_activation ? 2 * intermediate_size : intermediate_size),
        "gemm1_weights_scale has incorrect dim 1.");
    TORCH_CHECK(gemm1_weights_scale.sizes()[2] == args.hidden_size / 16, "gemm1_weights_scale has incorrect dim 2.");
    for (auto const& [name, tensor] : {std::make_pair("gemm1_alpha", &gemm1_alpha),
             std::make_pair("gemm1_beta", &gemm1_beta), std::make_pair("gemm1_clamp_limit", &gemm1_clamp_limit)})
    {
        if (tensor->has_value())
        {
            TORCH_CHECK(tensor->value().scalar_type() == at::ScalarType::Float, name, " must be float.");
            TORCH_CHECK(tensor->value().dim() == 1 && tensor->value().sizes()[0] == local_num_experts, name,
                " must be [local_num_experts].");
        }
    }
    if (gemm1_bias.has_value())
    {
        TORCH_CHECK(gemm1_bias.value().scalar_type() == at::ScalarType::Float, "gemm1_bias must be float.");
        TORCH_CHECK(gemm1_bias.value().dim() == 2 && gemm1_bias.value().sizes()[0] == local_num_experts
                && gemm1_bias.value().sizes()[1] == (is_gated_activation ? 2 * intermediate_size : intermediate_size),
            "gemm1_bias has incorrect shape.");
    }
    TORCH_CHECK(gemm2_weights.scalar_type() == FLOAT4_E2M1X2, "gemm2_weights must be byte.");
    TORCH_CHECK(gemm2_weights.dim() == 3, "gemm2_weights must be 3D.");
    TORCH_CHECK(gemm2_weights.sizes()[2] == intermediate_size / 2,
        "the third dimension of weights must be equal to intermediate_size.");
    TORCH_CHECK(gemm2_weights_scale.scalar_type() == at::ScalarType::Float8_e4m3fn, "gemm2_weights_scale must be fp8.");
    TORCH_CHECK(gemm2_weights_scale.dim() == 3, "gemm2_weights_scale must be 3D.");
    TORCH_CHECK(gemm2_weights_scale.sizes()[0] == local_num_experts, "gemm2_weights_scale has incorrect dim 0.");
    TORCH_CHECK(gemm2_weights_scale.sizes()[1] == args.hidden_size, "gemm2_weights_scale has incorrect dim 1.");
    TORCH_CHECK(gemm2_weights_scale.sizes()[2] == intermediate_size / 16, "gemm2_weights_scale has incorrect dim 2.");
    if (gemm2_bias.has_value())
    {
        TORCH_CHECK(gemm2_bias.value().scalar_type() == at::ScalarType::Float, "gemm2_bias must be float.");
        TORCH_CHECK(gemm2_bias.value().dim() == 2 && gemm2_bias.value().sizes()[0] == local_num_experts
                && gemm2_bias.value().sizes()[1] == args.hidden_size,
            "gemm2_bias has incorrect shape.");
    }
    for (auto const& [name, tensor] : {std::make_pair("output1_scales_scalar", &output1_scales_scalar),
             std::make_pair("output1_scales_gate_scalar", &output1_scales_gate_scalar),
             std::make_pair("output2_scales_scalar", &output2_scales_scalar)})
    {
        TORCH_CHECK(tensor->scalar_type() == at::ScalarType::Float, name, " must be float.");
        TORCH_CHECK(
            tensor->dim() == 1 && tensor->sizes()[0] == local_num_experts, name, " must be [local_num_experts].");
    }

    at::Tensor output;
    if (out_tensor.has_value())
    {
        TORCH_CHECK(do_finalize, "out_tensor is only supported when do_finalize=true.");
        TORCH_CHECK(out_tensor->scalar_type() == at::ScalarType::BFloat16, "out_tensor must be bfloat16.");
        TORCH_CHECK(out_tensor->dim() == 2, "out_tensor must be 2D.");
        TORCH_CHECK(out_tensor->sizes()[0] == args.num_tokens, "out_tensor dim0 must match num_tokens.");
        TORCH_CHECK(out_tensor->device() == hidden_states.device(), "out_tensor must be on the same device as inputs.");
        auto const out_hidden = out_tensor->sizes()[1];
        if (out_hidden < args.hidden_size)
        {
            // An unpadded out_tensor: FC2 computes the padded width, the finalize step writes the valid columns.
            args.valid_hidden_size = out_hidden;
            args.output_hidden_size = args.hidden_size;
        }
        else
        {
            TORCH_CHECK(out_hidden == args.hidden_size, "out_tensor hidden dim must match hidden_size.");
        }
        output = out_tensor.value();
    }
    else if (do_finalize)
    {
        output = at::detail::empty_cuda(
            {args.num_tokens, args.hidden_size}, at::ScalarType::BFloat16, hidden_states.device(), std::nullopt);
    }

    workspace.total_num_padded_tokens = total_num_padded_tokens.data_ptr<int>();
    workspace.total_max_padded_tokens = std::max(max_num_padded_tokens_gemm1, max_num_padded_tokens_gemm2);
    workspace.routing_expert_indexes = expert_indexes.data_ptr<int>();
    workspace.permuted_idx_size = total_num_padded_tokens.data_ptr<int>();
    workspace.expanded_idx_to_permuted_idx = expanded_idx_to_permuted_idx.data_ptr<int>();
    workspace.permuted_idx_to_token_idx = permuted_idx_to_token_idx.data_ptr<int>();
    workspace.expert_weights = expert_weights_ptr;
    workspace.cta_idx_xy_to_batch_idx = cta_idx_xy_to_batch_idx.data_ptr<int>();
    workspace.cta_idx_xy_to_mn_limit = cta_idx_xy_to_mn_limit.data_ptr<int>();
    workspace.num_non_exiting_ctas = num_non_exiting_ctas.data_ptr<int>();
    workspace.gemm1_output = gemm1_output.data_ptr();
    workspace.gemm1_output_scale = reinterpret_cast<float*>(gemm1_output_scale.data_ptr());
    workspace.gemm2_output = gemm2_output.data_ptr();
    workspace.gemm2_output_scale = nullptr;
    args.output = do_finalize ? output.data_ptr() : nullptr;
    args.output_scale = nullptr;
    args.output1_scales_scalar = output1_scales_scalar.data_ptr<float>();
    args.output1_scales_gate_scalar = output1_scales_gate_scalar.data_ptr<float>();
    args.output2_scales_scalar = output2_scales_scalar.data_ptr<float>();
    args.do_finalize = do_finalize;

    auto const [workspace_size_fc1, workspace_size_fc2] = gemms.getWorkspaceSizeInBytes(
        args.top_k, args.hidden_size, args.intermediate_size, args.local_num_experts, args.num_tokens, configIndex);
    at::Tensor workspace_fc1
        = at::detail::empty_cuda({workspace_size_fc1}, at::ScalarType::Char, hidden_states.device(), std::nullopt);
    at::Tensor workspace_fc2
        = at::detail::empty_cuda({workspace_size_fc2}, at::ScalarType::Char, hidden_states.device(), std::nullopt);
    workspace.bmm1_workspace = workspace_fc1.data_ptr();
    workspace.bmm2_workspace = workspace_fc2.data_ptr();

    // A raw cudaEvent_t (torch.cuda.Event.cuda_event), recorded where hidden_states are written.
    cudaEvent_t const hidden_states_ready = hidden_states_ready_event.has_value()
        ? reinterpret_cast<cudaEvent_t>(static_cast<intptr_t>(hidden_states_ready_event.value()))
        : nullptr;
    auto const& moe_stream = at::cuda::getCurrentCUDAStream(hidden_states.get_device());
    gemms.run(
        args, workspace, hidden_states_ready, prefer_gemm2_k128, hidden_states.get_device(), moe_stream, configIndex);

    if (!do_finalize)
    {
        return {gemm2_output, expert_weights, expanded_idx_to_permuted_idx};
    }
    return {output};
}

} // namespace

// GLM-5's routed experts over NVFP4 input, as a Torch custom class for the autotuner: the tactics are (tile N, FC1 /
// FC2 config pair), as for the TRTLLM-Gen FP4 MoE runner.
class GlmRoutedExpertsRunner : public torch::CustomClassHolder
{
public:
    explicit GlmRoutedExpertsRunner(int64_t actType)
        : mSupportedTileN{8, 16, 32, 64, 128, 256}
        , mActType{actType}
    {
    }

    [[nodiscard]] std::vector<std::vector<int64_t>> getValidConfigs(
        int64_t topK, int64_t hiddenSize, int64_t intermediateSize, int64_t numLocalExperts, int64_t numTokens)
    {
        auto const chosen = trtllmGenMoe::computeSelectedTileN(mSupportedTileN, numTokens, topK, numLocalExperts);
        std::vector<std::vector<int64_t>> result;
        for (int32_t tileN : mSupportedTileN)
        {
            if (chosen.find(tileN) == chosen.end())
            {
                continue;
            }
            auto* gemms = getGemmsOrNull(tileN);
            if (gemms == nullptr)
            {
                continue;
            }
            for (auto config :
                gemms->getValidConfigIndices(topK, hiddenSize, intermediateSize, numLocalExperts, numTokens))
            {
                result.push_back({static_cast<int64_t>(tileN), config});
            }
        }
        return result;
    }

    [[nodiscard]] std::vector<torch::Tensor> run(torch::optional<torch::Tensor> const& routing_logits,
        torch::optional<torch::Tensor> const& routing_bias, torch::Tensor const& hidden_states,
        torch::Tensor const& hidden_states_scale, torch::Tensor const& gemm1_weights,
        torch::Tensor const& gemm1_weights_scale, std::optional<torch::Tensor> const& gemm1_bias,
        std::optional<torch::Tensor> const& gemm1_alpha, std::optional<torch::Tensor> const& gemm1_beta,
        std::optional<torch::Tensor> const& gemm1_clamp_limit, torch::Tensor const& gemm2_weights,
        torch::Tensor const& gemm2_weights_scale, std::optional<torch::Tensor> const& gemm2_bias,
        torch::Tensor const& output1_scales_scalar, torch::Tensor const& output1_scales_gate_scalar,
        torch::Tensor const& output2_scales_scalar, int64_t const num_experts, int64_t const top_k,
        std::optional<int64_t> const n_group, std::optional<int64_t> const topk_group, int64_t const intermediate_size,
        int64_t const local_expert_offset, int64_t const local_num_experts,
        std::optional<double> const routed_scaling_factor, int64_t const routing_method_type, bool const do_finalize,
        std::vector<int64_t> moeConfigIndex, torch::optional<torch::Tensor> const& topk_weights,
        torch::optional<torch::Tensor> const& topk_ids, torch::optional<torch::Tensor> const& output,
        std::optional<int64_t> const hidden_states_ready_event,
        torch::optional<torch::Tensor> const& router_hidden_states, torch::optional<torch::Tensor> const& router_weight,
        torch::optional<torch::Tensor> const& router_workspace, bool const prefer_gemm2_k128)
    {
        int64_t tileN = moeConfigIndex[0];
        int64_t config = moeConfigIndex[1];
        // No tuned tactic: the tile from the average tokens per expert and its default config pair.
        if (tileN == -1 || config == -1)
        {
            auto const num_tokens = hidden_states.sizes()[0];
            auto const hidden_size = 2 * hidden_states.sizes()[1];
            float const avg_tokens_per_expert = static_cast<float>(num_tokens * top_k) / local_num_experts;
            tileN = std::clamp(nextPowerOfTwo(avg_tokens_per_expert), mSupportedTileN.front(), mSupportedTileN.back());
            config = getGemms(tileN).getDefaultValidConfigIndex(
                top_k, hidden_size, intermediate_size, local_num_experts, num_tokens);
        }
        return runRoutedExperts(routing_logits, routing_bias, hidden_states, hidden_states_scale, gemm1_weights,
            gemm1_weights_scale, gemm1_bias, gemm1_alpha, gemm1_beta, gemm1_clamp_limit, gemm2_weights,
            gemm2_weights_scale, gemm2_bias, output1_scales_scalar, output1_scales_gate_scalar, output2_scales_scalar,
            num_experts, top_k, n_group, topk_group, intermediate_size, local_expert_offset, local_num_experts,
            routed_scaling_factor, tileN, routing_method_type, do_finalize, getGemms(tileN), config, topk_weights,
            topk_ids, output, hidden_states_ready_event, router_hidden_states, router_weight, router_workspace,
            prefer_gemm2_k128);
    }

private:
    // Null (cached) when no kernel exists for the tile.
    RoutedExpertsGemms* getGemmsOrNull(int32_t tileN)
    {
        auto it = mGemms.find(tileN);
        if (it == mGemms.end())
        {
            std::unique_ptr<RoutedExpertsGemms> gemms;
            try
            {
                gemms = std::make_unique<RoutedExpertsGemms>(
                    tileN, static_cast<tensorrt_llm::kernels::ActType>(mActType));
            }
            catch (std::exception const& e)
            {
                TLLM_LOG_DEBUG("No GLM routed-experts kernels for tileN=%d: %s", tileN, e.what());
            }
            it = mGemms.emplace(tileN, std::move(gemms)).first;
        }
        return it->second.get();
    }

    RoutedExpertsGemms& getGemms(int32_t tileN)
    {
        auto* gemms = getGemmsOrNull(tileN);
        TLLM_CHECK_WITH_INFO(gemms != nullptr, "No GLM routed-experts kernels for tileN=%d", tileN);
        return *gemms;
    }

    std::vector<int32_t> const mSupportedTileN;
    int64_t mActType;
    std::map<int32_t, std::unique_ptr<RoutedExpertsGemms>> mGemms;
};

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.class_<tensorrt_llm::torch_ext::GlmRoutedExpertsRunner>("GlmRoutedExpertsRunner")
        .def(torch::init<int64_t>())
        .def("get_valid_configs", &tensorrt_llm::torch_ext::GlmRoutedExpertsRunner::getValidConfigs)
        .def("run_moe", &tensorrt_llm::torch_ext::GlmRoutedExpertsRunner::run);
}
