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
#include "tensorrt_llm/kernels/moe/trtllmGen/runner.h"
#include "tensorrt_llm/thop/moe/trtllmGen/tmaOobWorkspace.h"
#include "tensorrt_llm/thop/thUtils.h"
#include <ATen/cuda/EmptyTensor.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <vector>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{
namespace btg = batchedGemm::trtllm::gen;
using tensorrt_llm::kernels::trtllmGenFp8BlockScaleMoe::Routing::RoutingMethodType;
using tensorrt_llm::kernels::trtllmGenFp8BlockScaleMoe::computeSelectedTileN;
namespace routing = tensorrt_llm::kernels::trtllmGenFp8BlockScaleMoe::Routing;

namespace
{
using MoeRunnerType = tensorrt_llm::kernels::trtllmGenFp8BlockScaleMoe::MoE::Runner;

//! The BF16 cubins read their weights in the shuffled BlockMajorK layout: per expert [K / kBlockK, M, kBlockK], with
//! 128-byte (64-element) K blocks.
constexpr int64_t kBlockK = 64;
//! Every BF16 cubin has tileM 128 and tileK 128 (or 64), so the hidden size (gemm1's K, gemm2's M) and the
//! intermediate size (gemm2's K, gemm1's M or half of it) must be multiples of 128.
constexpr int64_t kShapeAlignment = 128;

at::ScalarType toScalarType(btg::Dtype dtype)
{
    switch (dtype)
    {
    case btg::Dtype::Bfloat16: return at::ScalarType::BFloat16;
    case btg::Dtype::Fp16: return at::ScalarType::Half;
    case btg::Dtype::Fp32: return at::ScalarType::Float;
    default:
        TORCH_CHECK(
            false, "Unsupported MoE output dtype ", btg::dtypeToString(dtype), ". Expected Bfloat16/Fp16/Fp32.");
    }
}

void checkBlockMajorKWeights(torch::Tensor const& weights, char const* name, int64_t numExperts, int64_t m, int64_t k)
{
    TORCH_CHECK(weights.scalar_type() == at::ScalarType::BFloat16, name, " must be bfloat16, got ",
        c10::toString(weights.scalar_type()), ".");
    TORCH_CHECK(weights.dim() == 4 && weights.sizes()[0] == numExperts && weights.sizes()[1] * kBlockK == k
            && weights.sizes()[2] == m && weights.sizes()[3] == kBlockK,
        name, " must be BlockMajorK [", numExperts, ", ", k / kBlockK, ", ", m, ", ", kBlockK, "], got ",
        weights.sizes(), ".");
    TORCH_CHECK(weights.is_contiguous(), name, " must be contiguous.");
}

std::vector<torch::Tensor> runBf16Moe(torch::optional<torch::Tensor> const& routingLogits,
    torch::optional<torch::Tensor> const& routingBias, torch::Tensor const& hiddenStates,
    torch::Tensor const& gemm1Weights, torch::Tensor const& gemm2Weights, int64_t const numExperts, int64_t const topK,
    std::optional<int64_t> const nGroup, std::optional<int64_t> const topkGroup, int64_t const intermediateSize,
    int64_t const localExpertOffset, int64_t const localNumExperts, std::optional<double> const routedScalingFactor,
    int64_t const tileTokensDim, int64_t const routingMethodType, bool const isGatedActivation, bool const doFinalize,
    MoeRunnerType& moeRunner, int64_t const moeConfigIndex, torch::optional<torch::Tensor> const& topkWeights,
    torch::optional<torch::Tensor> const& topkIds, torch::optional<torch::Tensor> const& outTensor)
{
    TORCH_CHECK(tensorrt_llm::common::isSM100Family(), "Only SM100f is supported by the BF16 TRTLLM-Gen MoE.");

    TORCH_CHECK(hiddenStates.scalar_type() == at::ScalarType::BFloat16, "hidden_states must be bfloat16, got ",
        c10::toString(hiddenStates.scalar_type()), ".");
    TORCH_CHECK(hiddenStates.dim() == 2, "hidden_states must be 2D.");
    TORCH_CHECK(hiddenStates.is_contiguous(), "hidden_states must be contiguous.");
    int64_t const numTokens = hiddenStates.sizes()[0];
    int64_t const hiddenSize = hiddenStates.sizes()[1];
    auto const device = hiddenStates.device();

    TORCH_CHECK(topkIds.has_value() == topkWeights.has_value(), "topk_ids and topk_weights must be given together.");
    bool const hasTopK = topkIds.has_value();
    if (hasTopK)
    {
        TORCH_CHECK(topkIds->scalar_type() == at::ScalarType::Int, "topk_ids must be int32.");
        TORCH_CHECK(topkWeights->scalar_type() == at::ScalarType::BFloat16, "topk_weights must be bfloat16.");
        TORCH_CHECK(topkIds->dim() == 2 && topkIds->sizes()[0] == numTokens && topkIds->sizes()[1] == topK,
            "topk_ids must be [num_tokens, top_k].");
        TORCH_CHECK(topkWeights->dim() == 2 && topkWeights->sizes()[0] == numTokens && topkWeights->sizes()[1] == topK,
            "topk_weights must be [num_tokens, top_k].");
        TORCH_CHECK(
            topkIds->is_contiguous() && topkWeights->is_contiguous(), "topk_ids and topk_weights must be contiguous.");
        TORCH_CHECK(topkIds->device() == device && topkWeights->device() == device,
            "topk_ids and topk_weights must be on the same device as hidden_states.");
    }
    else
    {
        TORCH_CHECK(routingLogits.has_value(), "routing_logits or (topk_ids and topk_weights) must be provided.");
        TORCH_CHECK(routingLogits->scalar_type() == at::ScalarType::BFloat16
                || routingLogits->scalar_type() == at::ScalarType::Float,
            "routing_logits must be bfloat16 or float32.");
        TORCH_CHECK(routingLogits->dim() == 2 && routingLogits->sizes()[0] == numTokens
                && routingLogits->sizes()[1] == numExperts,
            "routing_logits must be [num_tokens, num_experts].");
        TORCH_CHECK(routingLogits->is_contiguous(), "routing_logits must be contiguous.");
        TORCH_CHECK(routingLogits->device() == device, "routing_logits must be on the same device as hidden_states.");
    }

    if (routingBias.has_value())
    {
        TORCH_CHECK(routingBias->scalar_type() == at::ScalarType::BFloat16
                || routingBias->scalar_type() == at::ScalarType::Float,
            "routing_bias must be bfloat16 or float32.");
        TORCH_CHECK(
            routingBias->dim() == 1 && routingBias->sizes()[0] == numExperts, "routing_bias must be [num_experts].");
        TORCH_CHECK(routingBias->is_contiguous(), "routing_bias must be contiguous.");
    }

    auto const methodType = static_cast<RoutingMethodType>(routingMethodType);
    if (nGroup.has_value() && nGroup.value() > 1)
    {
        TORCH_CHECK(methodType == RoutingMethodType::DeepSeekV3,
            "Routing kernel with groups implies DeepSeekV3 routing method.");
        TORCH_CHECK(topkGroup.has_value(), "if n_group is given, topk_group must be given");
        TORCH_CHECK(numExperts % nGroup.value() == 0, "num_experts must be divisible by n_group");
        TORCH_CHECK(topK <= 8 && topK > 0, "Current routing kernel (with groups) only supports top_k<=8 && top_k>0.");
        TORCH_CHECK(topkGroup.value() <= 4 && topkGroup.value() > 0,
            "Current routing kernel only (with groups) supports topk_group<=4 && topk_group > 0.");
        TORCH_CHECK(topkGroup.value() <= nGroup.value(), "n_group must not be smaller than topk_group.");
        TORCH_CHECK(topK < (topkGroup.value() * numExperts / nGroup.value()),
            "top_k must be less than total number of experts in selected groups");
    }
    else if (methodType == RoutingMethodType::Renormalize || methodType == RoutingMethodType::RenormalizeNaive)
    {
        TORCH_CHECK(topK <= 32 && topK > 0,
            "Current routing kernel (no groups, renormalize) only supports top_k<=32 && top_k>0.");
    }
    else if (methodType == RoutingMethodType::Llama4)
    {
        TORCH_CHECK(topK == 1, "Current routing kernel (no groups, Llama4) only supports top_k=1.");
    }
    TORCH_CHECK(numExperts > topK, "num_experts must be greater than top_k");
    TORCH_CHECK(numExperts <= 2048, "num_experts must be less than or equal to 2048");

    TORCH_CHECK(hiddenSize % kShapeAlignment == 0 && intermediateSize % kShapeAlignment == 0,
        "hidden_size and intermediate_size must be multiples of ", kShapeAlignment, " for the BF16 cubins, got ",
        hiddenSize, " and ", intermediateSize, ".");
    checkBlockMajorKWeights(gemm1Weights, "gemm1_weights", localNumExperts,
        isGatedActivation ? 2 * intermediateSize : intermediateSize, hiddenSize);
    checkBlockMajorKWeights(gemm2Weights, "gemm2_weights", localNumExperts, hiddenSize, intermediateSize);

    // Validated before any kernel is launched.
    if (outTensor.has_value())
    {
        TORCH_CHECK(doFinalize, "out_tensor is only supported when do_finalize=true.");
        TORCH_CHECK(outTensor->scalar_type() == at::ScalarType::BFloat16, "out_tensor must be bfloat16.");
        TORCH_CHECK(outTensor->dim() == 2 && outTensor->sizes()[0] == numTokens && outTensor->sizes()[1] == hiddenSize,
            "out_tensor must be [num_tokens, hidden_size].");
        TORCH_CHECK(outTensor->is_contiguous(), "out_tensor must be contiguous.");
        TORCH_CHECK(outTensor->device() == device, "out_tensor must be on the same device as inputs.");
    }

    tensorrt_llm::kernels::trtllmGenFp8BlockScaleMoe::MoE::MoERunnerArgs args;
    tensorrt_llm::kernels::trtllmGenFp8BlockScaleMoe::MoE::MoEWorkspace workspace;

    auto const routingBiasDtype = routingBias.has_value() ? routingBias->scalar_type() : at::ScalarType::BFloat16;
    args.mDtypeElt = btg::Dtype::Bfloat16;
    args.mDtypeBias = routingBiasDtype == at::ScalarType::Float ? btg::Dtype::Fp32 : btg::Dtype::Bfloat16;
    args.routing_logits = hasTopK ? nullptr : routingLogits->data_ptr();
    args.routing_bias = routingBias.has_value() ? routingBias->data_ptr() : nullptr;
    args.hidden_states = hiddenStates.data_ptr();
    args.topk_weights = hasTopK ? topkWeights->data_ptr() : nullptr;
    args.topk_ids = hasTopK ? static_cast<int32_t*>(topkIds->data_ptr()) : nullptr;
    args.gemm1_weights = gemm1Weights.data_ptr();
    args.gemm2_weights = gemm2Weights.data_ptr();
    args.num_tokens = numTokens;
    args.num_experts = numExperts;
    args.hidden_size = hiddenSize;
    args.top_k = topK;
    args.n_group = nGroup.value_or(0);
    args.topk_group = topkGroup.value_or(0);
    args.local_expert_offset = localExpertOffset;
    args.local_num_experts = localNumExperts;
    args.routed_scaling_factor = routedScalingFactor.value_or(1.0);
    args.intermediate_size = intermediateSize;

    at::Tensor const numTokensPerExpert
        = at::detail::empty_cuda({numExperts}, at::ScalarType::Int, device, /*memory_format_opt=*/std::nullopt);
    // Only the local experts' rows are materialized, so size the permuted buffers by them.
    int32_t const maxNumPaddedTokens
        = routing::getMaxPermutedPaddedCount(args.num_tokens, topK, localNumExperts, tileTokensDim);
    int32_t const maxNumPaddedTokensGemm1 = routing::maybeGetMinTokenCount(
        maxNumPaddedTokens, args.intermediate_size, btg::dtypeGetNumBits(args.mDtypeElt));
    int32_t const maxNumPaddedTokensGemm2
        = routing::maybeGetMinTokenCount(maxNumPaddedTokens, args.hidden_size, btg::dtypeGetNumBits(args.mDtypeOut));
    int32_t const maxNumCtas
        = routing::getMaxNumCtasInBatchDim(args.num_tokens, args.top_k, args.local_num_experts, tileTokensDim);

    at::Tensor const totalNumPaddedTokens
        = at::empty({}, at::TensorOptions().device(device).dtype(at::ScalarType::Int));
    at::Tensor const expandedIdxToPermutedIdx = at::detail::empty_cuda(
        {args.num_tokens, args.top_k}, at::ScalarType::Int, device, /*memory_format_opt=*/std::nullopt);
    // Sized via getRouteMapAllocCount: the gemm1 cubins read one element past the end of the route map, see the
    // comment on that helper.
    at::Tensor const permutedIdxToTokenIdx
        = at::detail::empty_cuda({routing::getRouteMapAllocCount(maxNumPaddedTokens)}, at::ScalarType::Int, device,
            /*memory_format_opt=*/std::nullopt);
    // The routing kernel's top-k weights feed moe_finalize, which reads them in the output dtype.
    at::Tensor const expertWeights = at::detail::empty_cuda(
        {args.num_tokens, args.top_k}, toScalarType(args.mDtypeOut), device, /*memory_format_opt=*/std::nullopt);
    at::Tensor const expertIndexes = at::detail::empty_cuda(
        {args.num_tokens, args.top_k}, at::ScalarType::Int, device, /*memory_format_opt=*/std::nullopt);
    int64_t const sizeOfExpertCountHistogram = std::max(numExperts * 2, static_cast<int64_t>(256 * 2));
    at::Tensor const expertCountHistogram = at::detail::empty_cuda(
        {sizeOfExpertCountHistogram}, at::ScalarType::Int, device, /*memory_format_opt=*/std::nullopt);
    at::Tensor const ctaIdxXyToBatchIdx
        = at::detail::empty_cuda({maxNumCtas}, at::ScalarType::Int, device, /*memory_format_opt=*/std::nullopt);
    at::Tensor const ctaIdxXyToMnLimit
        = at::detail::empty_cuda({maxNumCtas}, at::ScalarType::Int, device, /*memory_format_opt=*/std::nullopt);
    at::Tensor const numNonExitingCtas = at::empty({}, at::TensorOptions().device(device).dtype(at::ScalarType::Int));

    // FC1 TMA-stores gemm1Output, FC2 TMA-loads it and TMA-stores gemm2Output, all through the TMA-OOB descriptors.
    // The gated activation is fused into gemm1, so its output is intermediate_size wide either way.
    at::Tensor const gemm1Output
        = allocTmaOobWorkspace(maxNumPaddedTokensGemm1, intermediateSize, at::ScalarType::BFloat16, device);
    at::Tensor const gemm2Output
        = allocTmaOobWorkspace(maxNumPaddedTokensGemm2, args.hidden_size, at::ScalarType::BFloat16, device);

    void* expertWeightsPtr = hasTopK ? args.topk_weights : expertWeights.data_ptr();

    routing::Runner routingRunner(tileTokensDim);
    auto const& stream = at::cuda::getCurrentCUDAStream(device.index());
    auto const dtypeRoutingLogits
        = !hasTopK && routingLogits->scalar_type() == at::ScalarType::Float ? btg::Dtype::Fp32 : btg::Dtype::Bfloat16;
    routingRunner.run(args.routing_logits, args.routing_bias, args.num_tokens, args.num_experts, args.top_k,
        /*numFusedSharedExpert=*/0, args.n_group, args.topk_group, args.local_expert_offset, args.local_num_experts,
        args.routed_scaling_factor, expertIndexes.data_ptr<int>(), expertCountHistogram.data_ptr<int>(),
        totalNumPaddedTokens.data_ptr<int>(), expandedIdxToPermutedIdx.data_ptr<int>(),
        /*permutedIdxToExpandedIdx=*/nullptr, permutedIdxToTokenIdx.data_ptr<int>(), expertWeightsPtr, args.topk_ids,
        numTokensPerExpert.data_ptr<int>(), ctaIdxXyToBatchIdx.data_ptr<int>(), ctaIdxXyToMnLimit.data_ptr<int>(),
        numNonExitingCtas.data_ptr<int>(), args.mDtypeElt, /*useRoutingScalesOnInput=*/false,
        /*useDeepSeekFp8=*/false, methodType, stream, dtypeRoutingLogits, args.mDtypeBias);

    at::Tensor const output = outTensor.has_value()
        ? outTensor.value()
        : at::Tensor(at::detail::empty_cuda(
            {args.num_tokens, args.hidden_size}, at::ScalarType::BFloat16, device, /*memory_format_opt=*/std::nullopt));

    workspace.total_num_padded_tokens = totalNumPaddedTokens.data_ptr<int>();
    workspace.total_max_padded_tokens = std::max(maxNumPaddedTokensGemm1, maxNumPaddedTokensGemm2);
    workspace.routing_expert_indexes = expertIndexes.data_ptr<int>();
    workspace.permuted_idx_size = totalNumPaddedTokens.data_ptr<int>();
    workspace.expanded_idx_to_permuted_idx = expandedIdxToPermutedIdx.data_ptr<int>();
    workspace.permuted_idx_to_token_idx = permutedIdxToTokenIdx.data_ptr<int>();
    workspace.expert_weights = expertWeightsPtr;
    workspace.cta_idx_xy_to_batch_idx = ctaIdxXyToBatchIdx.data_ptr<int>();
    workspace.cta_idx_xy_to_mn_limit = ctaIdxXyToMnLimit.data_ptr<int>();
    workspace.num_non_exiting_ctas = numNonExitingCtas.data_ptr<int>();
    workspace.gemm1_output = gemm1Output.data_ptr();
    workspace.gemm1_output_scale = nullptr;
    workspace.gemm2_output = gemm2Output.data_ptr();
    workspace.gemm2_output_scale = nullptr;
    args.output = output.data_ptr();
    args.output_scale = nullptr;
    args.do_finalize = doFinalize;

    auto const workspaceSizes = moeRunner.getWorkspaceSizeInBytes(args, moeConfigIndex);
    at::Tensor const workspaceFc1 = at::detail::empty_cuda(
        {std::get<0>(workspaceSizes)}, at::ScalarType::Char, device, /*memory_format_opt=*/std::nullopt);
    at::Tensor const workspaceFc2 = at::detail::empty_cuda(
        {std::get<1>(workspaceSizes)}, at::ScalarType::Char, device, /*memory_format_opt=*/std::nullopt);
    workspace.bmm1_workspace = workspaceFc1.data_ptr();
    workspace.bmm2_workspace = workspaceFc2.data_ptr();
    moeRunner.run(args, workspace, device.index(), stream, moeConfigIndex);

    if (!doFinalize)
    {
        // With caller-supplied top-k the routing kernel reads topk_weights and leaves expertWeights unwritten.
        // Copy rather than return topk_weights: an op output must not alias its inputs.
        if (hasTopK)
        {
            expertWeights.copy_(topkWeights.value());
        }
        return {gemm2Output, expertWeights, expandedIdxToPermutedIdx};
    }
    return {output};
}
} // namespace

//! \brief The BF16 x BF16 TRTLLM-Gen MoE runner, wrapped in a Torch custom class for the autotuner.
class Bf16MoeRunner : public torch::CustomClassHolder
{
public:
    explicit Bf16MoeRunner(int64_t actType)
        // Tiles the BF16 x BF16 cubins ship: low-latency 8/16/32 and 2-CTA high-throughput 64/128.
        : mSupportedTileN{8, 16, 32, 64, 128}
        , mActType{static_cast<tensorrt_llm::kernels::ActType>(actType)}
    {
        for (int32_t const tileN : mSupportedTileN)
        {
            mRunners.emplace(tileN,
                std::make_unique<MoeRunnerType>(btg::Dtype::Bfloat16, btg::Dtype::Bfloat16,
                    /*useDeepSeekFp8=*/false, /*tileTokensDim=*/tileN, mActType));
        }
    }

    //! \brief Returns the (tileN, config) pairs valid for the given problem.
    [[nodiscard]] std::vector<std::vector<int64_t>> getValidConfigs(
        int64_t topK, int64_t hiddenSize, int64_t intermediateSize, int64_t numLocalExperts, int64_t numTokens) const
    {
        auto const chosen = computeSelectedTileN(mSupportedTileN, numTokens, topK, numLocalExperts);
        std::vector<std::vector<int64_t>> tactics;
        for (auto const& [tileN, runner] : mRunners)
        {
            if (chosen.find(tileN) == chosen.end())
            {
                continue;
            }
            for (auto const config :
                runner->getValidConfigIndices(topK, hiddenSize, intermediateSize, numLocalExperts, numTokens))
            {
                tactics.push_back({tileN, config});
            }
        }
        return tactics;
    }

    [[nodiscard]] std::vector<torch::Tensor> run(torch::optional<torch::Tensor> const& routingLogits,
        torch::optional<torch::Tensor> const& routingBias, torch::Tensor const& hiddenStates,
        torch::Tensor const& gemm1Weights, torch::Tensor const& gemm2Weights, int64_t const numExperts,
        int64_t const topK, std::optional<int64_t> const nGroup, std::optional<int64_t> const topkGroup,
        int64_t const intermediateSize, int64_t const localExpertOffset, int64_t const localNumExperts,
        std::optional<double> const routedScalingFactor, int64_t const routingMethodType, bool const doFinalize,
        std::vector<int64_t> moeConfigIndex, torch::optional<torch::Tensor> const& topkWeights,
        torch::optional<torch::Tensor> const& topkIds, torch::optional<torch::Tensor> const& output = torch::nullopt)
    {
        TORCH_CHECK(moeConfigIndex.size() == 2, "moeConfigIndex must be a (tileN, config) pair.");
        int64_t tileN = moeConfigIndex[0];
        int64_t config = moeConfigIndex[1];
        bool const useDefaultConfig = tileN == -1 || config == -1;
        if (useDefaultConfig)
        {
            // No autotuner: pick the tile from the average expert load.
            int64_t const numTokens = hiddenStates.sizes()[0];
            float const avgTokensPerExpert = static_cast<float>(numTokens * topK) / localNumExperts;
            tileN = std::clamp(nextPowerOfTwo(avgTokensPerExpert), mSupportedTileN.front(), mSupportedTileN.back());
        }
        auto const it = mRunners.find(static_cast<int32_t>(tileN));
        TORCH_CHECK(it != mRunners.end(), "No BF16 TRTLLM-Gen MoE runner for tileN=", tileN, ".");
        MoeRunnerType& runner = *it->second;
        if (useDefaultConfig)
        {
            config = runner.getDefaultValidConfigIndex(
                topK, hiddenStates.sizes()[1], intermediateSize, localNumExperts, hiddenStates.sizes()[0]);
        }
        return runBf16Moe(routingLogits, routingBias, hiddenStates, gemm1Weights, gemm2Weights, numExperts, topK,
            nGroup, topkGroup, intermediateSize, localExpertOffset, localNumExperts, routedScalingFactor, tileN,
            routingMethodType, tensorrt_llm::kernels::isGatedActType(mActType), doFinalize, runner, config, topkWeights,
            topkIds, output);
    }

private:
    std::vector<int32_t> const mSupportedTileN;
    tensorrt_llm::kernels::ActType const mActType;
    std::map<int32_t, std::unique_ptr<MoeRunnerType>> mRunners;
};

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.class_<tensorrt_llm::torch_ext::Bf16MoeRunner>("Bf16MoERunner")
        .def(torch::init<int64_t>())
        .def("get_valid_configs", &tensorrt_llm::torch_ext::Bf16MoeRunner::getValidConfigs)
        .def("run_moe", &tensorrt_llm::torch_ext::Bf16MoeRunner::run);
}
