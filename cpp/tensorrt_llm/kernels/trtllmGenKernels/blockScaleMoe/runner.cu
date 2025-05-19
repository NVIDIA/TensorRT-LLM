/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.  All rights reserved.
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

#include "DevKernel.h"
#include "RoutingKernel.h"
#include "runner.h"
#include "tensorrt_llm/kernels/trtllmGenKernels/batchedGemm/KernelRunner.h"
#include "tensorrt_llm/kernels/trtllmGenKernels/batchedGemm/trtllmGen_bmm_export/trtllm/gen/DtypeDecl.h"
#include "tensorrt_llm/kernels/trtllmGenKernels/batchedGemm/trtllmGen_bmm_export/trtllm/gen/SfLayoutDecl.h"
#include <iostream>
#include <tensorrt_llm/common/assert.h>

namespace tensorrt_llm
{
namespace kernels
{
namespace trtllmGenFp8BlockScaleMoe
{

namespace tg = trtllm::gen;

namespace Routing
{
namespace
{
inline int32_t computeLog2(int32_t val, std::string const& name = "")
{
    int32_t n = val;
    int32_t out = 0;
    while (n >>= 1)
    {
        ++out;
    }
    TLLM_CHECK_WITH_INFO((1 << out) == val, "Expected %s to be a power of 2, got %d", name.c_str(), val);
    return out;
}
} // namespace

Runner::Runner() {}

void Runner::run(void* routingLogits, void* routingBias, int32_t numTokens, int32_t numExperts, int32_t topK,
    int32_t nGroup, int32_t topkGroup, int32_t localExpertOffset, int32_t localNumExperts, float routedScalingFactor,
    int32_t* routingExpertIndexes, int32_t* expertCountHistogram, int32_t* permutedIdxSize,
    int32_t* expandedIdxToPermutedIdx, int32_t* permutedIdxToExpandedIdx, int32_t* permutedIdxToTokenIdx,
    void* expertWeights, int32_t* numTokensPerExpert, int32_t* ctaIdxXyToBatchIdx, int32_t* ctaIdxXyToMnLimit,
    int32_t* numNonExitingCtas, tg::Dtype dtypeElt, bool useRoutingScalesOnInput, bool useDeepSeekFp8,
    cudaStream_t stream)
{
    if (topK == 8)
    {
        // FIXME: hardcoded for now
        int32_t tileN = 8;

        moe::dev::routing::Data routingData;
        routingData.mDtypeElt = dtypeElt; // no-op for now as hidden_state is not input
        routingData.mDtypeExpW = tg::Dtype::Bfloat16;
        routingData.mUsePdl = true;

        // output:
        routingData.mPtrExpertIdx = routingExpertIndexes;
        routingData.mPtrExpertCounts = expertCountHistogram;
        routingData.mPtrPermutedIdxSize = permutedIdxSize;
        routingData.mPtrExpandedIdxToPermutedIdx = expandedIdxToPermutedIdx;
        routingData.mPtrPermutedIdxToExpandedIdx = permutedIdxToExpandedIdx;
        routingData.mPtrPermutedIdxToTokenIdx = permutedIdxToTokenIdx;
        routingData.mPtrNumTokensPerExpert = numTokensPerExpert;
        routingData.mPtrExpertWeights = expertWeights;

        routingData.mPtrCtaIdxXyToBatchIdx = ctaIdxXyToBatchIdx;
        routingData.mPtrCtaIdxXyToMnLimit = ctaIdxXyToMnLimit;
        routingData.mPtrNumNonExitingCtas = numNonExitingCtas;
        routingData.mAllToAllRouteAct = false;

        // input:
        // routingData.mPtrRoutingWeights = args.mRoutingWeights;  // routing weights (don't need if not using gemm)
        routingData.mPtrRoutingBias = routingBias;
        routingData.mPtrScores = reinterpret_cast<float*>(routingLogits);
        // routingData.mPtrIn = args.mInputActs;
        routingData.mNumTokens = numTokens;
        // routingData.mHiddenDim = args.mHiddenDim;
        routingData.mNumExperts = numExperts;
        routingData.mNumExpertGroups = nGroup;
        routingData.mNumLimitedGroups = topkGroup;
        routingData.mTopK = topK;
        routingData.mPaddingLog2 = computeLog2(tileN);
        routingData.mLocalExpertsStartIdx = localExpertOffset;
        routingData.mLocalExpertsStrideLog2 = 0;
        routingData.mNumLocalExperts = localNumExperts;
        routingData.mRouteScale = routedScalingFactor;
        routingData.mUseRoutingSoftmax = false;
        moe::dev::routing::run(routingData, stream);
    }
    else if (topK == 1)
    {
        // FIXME: hardcoded for now
        int32_t tileN = 8;

        moe::dev::routingLlama4::Data routingData;
        // routingData.mDtypeElt = dtypeElt; // no-op for now as hidden_state is not input
        routingData.mDtypeExpW = tg::Dtype::Bfloat16;
        routingData.mUsePdl = true;

        // output:
        routingData.mPtrExpertIdx = routingExpertIndexes;
        routingData.mPtrExpertCounts = expertCountHistogram;
        routingData.mPtrPermutedIdxSize = permutedIdxSize;
        routingData.mPtrExpandedIdxToPermutedIdx = expandedIdxToPermutedIdx;
        // routingData.mPtrPermutedIdxToExpandedIdx = permuted_idx_to_expanded_idx;
        routingData.mPtrPermutedIdxToTokenIdx = permutedIdxToTokenIdx;
        // routingData.mPtrNumTokensPerExpert = num_tokens_per_expert;
        routingData.mPtrExpertWeights = expertWeights;

        routingData.mPtrCtaIdxXyToBatchIdx = ctaIdxXyToBatchIdx;
        routingData.mPtrCtaIdxXyToMnLimit = ctaIdxXyToMnLimit;
        routingData.mPtrNumNonExitingCtas = numNonExitingCtas;
        // routingData.mAllToAllRouteAct = false;

        // input:
        // routingData.mPtrRoutingWeights = args.mRoutingWeights;  // routing weights (don't need if not using gemm)
        // routingData.mPtrRoutingBias = routingBias;
        routingData.mPtrScores = routingLogits;
        // routingData.mPtrIn = args.mInputActs;
        routingData.mNumTokens = numTokens;
        // routingData.mHiddenDim = args.mHiddenDim;
        routingData.mNumExperts = numExperts;
        // routingData.mNumExpertGroups = n_group;
        // routingData.mNumLimitedGroups = topk_group;
        routingData.mTopK = topK;
        routingData.mPaddingLog2 = computeLog2(tileN);
        routingData.mLocalExpertsStartIdx = localExpertOffset;
        routingData.mLocalExpertsStrideLog2 = 0;
        routingData.mNumLocalExperts = localNumExperts;
        // routingData.mRouteScale = routed_scaling_factor;
        // routingData.mUseRoutingSoftmax = false;
        moe::dev::routingLlama4::run(routingData, stream);
    }
    else
    {
        TLLM_CHECK_WITH_INFO(false, "top_k can only be 1 or 8.");
    }
}
} // namespace Routing

namespace PermuteGemm1
{

tensorrt_llm::kernels::TrtllmGenBatchedGemmRunnerOptions getOptions(
    trtllm::gen::Dtype dtypeElt, int32_t tileTokensDim, bool useDeepSeekFp8)
{
    tensorrt_llm::kernels::TrtllmGenBatchedGemmRunnerOptions options = {.eltType = dtypeElt,
        .outputType = dtypeElt,
        .deepSeekFp8 = useDeepSeekFp8,
        .fusedAct = !useDeepSeekFp8,
        .routeAct = true,
        .staticBatch = false,
        .transposeMmaOutput = true,
        .tileSize = tileTokensDim,
        .epilogueTileM = useDeepSeekFp8 ? 64 : 128};
    return options;
}

Runner::Runner(trtllm::gen::Dtype dtypeElt, bool useDeepSeekFp8)
    : mDtypeElt(dtypeElt)
    , mRunner(tensorrt_llm::kernels::TrtllmGenBatchedGemmRunner(getOptions(mDtypeElt, mTileTokensDim, useDeepSeekFp8)))
{
}

void Runner::run(void* hiddenState, void* hiddenStateScale, void* weights, void* weightsScale, void* expertWeights,
    float* outputScalesScalar, float* outputScalesGateScalar, void* output, void* outputScale, int32_t topK,
    int32_t hiddenSize, int32_t intermediateSize, int32_t numExperts, int32_t numTokens, int32_t* permutedIdxToTokenIdx,
    int32_t* ptrNumNonExitingCtas, int32_t* ptrTotalNumPaddedTokens, int32_t* ptrCtaIdxXyToBatchIdx,
    int32_t* ptrCtaIdxXyToMnLimit, void* bmm1Workspace, bool useRoutingScalesOnInput, int device, cudaStream_t stream)
{
    auto maxNumCtasInBatchDim = Routing::getMaxNumCtasInBatchDim(numTokens, topK, numExperts, mTileTokensDim);
    mRunner.run(numTokens, 2 * intermediateSize, hiddenSize, {}, numTokens, numExperts, maxNumCtasInBatchDim,
        hiddenState, hiddenStateScale, weights, weightsScale, expertWeights, /* perTokensSfB */ nullptr,
        outputScalesScalar, outputScalesGateScalar, output, outputScale, permutedIdxToTokenIdx, ptrTotalNumPaddedTokens,
        ptrCtaIdxXyToBatchIdx, ptrCtaIdxXyToMnLimit, ptrNumNonExitingCtas, bmm1Workspace, stream, device);
}

size_t Runner::getWorkspaceSizeInBytes(
    int32_t topK, int32_t hiddenSize, int32_t intermediateSize, int32_t numExperts, int32_t numTokens)
{
    auto maxNumCtasInBatchDim = Routing::getMaxNumCtasInBatchDim(numTokens, topK, numExperts, mTileTokensDim);
    return mRunner.getWorkspaceSizeInBytes(
        numTokens, 2 * intermediateSize, hiddenSize, {}, numTokens, numExperts, maxNumCtasInBatchDim);
}
} // namespace PermuteGemm1

namespace Gemm2
{
tensorrt_llm::kernels::TrtllmGenBatchedGemmRunnerOptions getOptions(
    trtllm::gen::Dtype dtypeElt, trtllm::gen::Dtype dtypeOut, int32_t tileTokensDim, bool useDeepSeekFp8)
{
    tensorrt_llm::kernels::TrtllmGenBatchedGemmRunnerOptions options = {.eltType = dtypeElt,
        .outputType = dtypeOut,
        .deepSeekFp8 = useDeepSeekFp8,
        .fusedAct = false,
        .routeAct = false,
        .staticBatch = false,
        .transposeMmaOutput = true,
        .tileSize = tileTokensDim,
        .epilogueTileM = useDeepSeekFp8 ? 64 : 128};
    return options;
}

Runner::Runner(tg::Dtype dtypeElt, tg::Dtype outputDtype, bool useDeepSeekFp8)
    : mDtypeElt(dtypeElt)
    , mOutputDtype(outputDtype)
    , mRunner(tensorrt_llm::kernels::TrtllmGenBatchedGemmRunner(
          getOptions(mDtypeElt, mOutputDtype, mTileTokensDim, useDeepSeekFp8)))
{
}

void Runner::run(void* permutedHiddenState, void* permutedHiddenStateScale, void* weights, void* weightsScale,
    float* outputScalesScalar, void* output, void* outputScale, int32_t topK, int32_t hiddenSize,
    int32_t intermediateSize, int32_t numExperts, int32_t numTokens, int32_t* ptrNumNonExitingCtas,
    int32_t* ptrTotalNumPaddedTokens, int32_t* ptrCtaIdxXyToBatchIdx, int32_t* ptrCtaIdxXyToMnLimit,
    void* bmm2Workspace, int device, cudaStream_t stream)
{
    auto maxNumCtasInBatchDim = Routing::getMaxNumCtasInBatchDim(numTokens, topK, numExperts, mTileTokensDim);
    mRunner.run(numTokens, hiddenSize, intermediateSize, {}, numTokens, numExperts, maxNumCtasInBatchDim,
        permutedHiddenState, permutedHiddenStateScale, weights, weightsScale, /* perTokensSfA */ nullptr,
        /* perTokensSfB */ nullptr, outputScalesScalar, /* outputScalesGateScalar */ nullptr, output, outputScale,
        /* permutedIdxToTokenIdx */ nullptr, ptrTotalNumPaddedTokens, ptrCtaIdxXyToBatchIdx, ptrCtaIdxXyToMnLimit,
        ptrNumNonExitingCtas, bmm2Workspace, stream, device);
}

size_t Runner::getWorkspaceSizeInBytes(
    int32_t topK, int32_t hiddenSize, int32_t intermediateSize, int32_t numExperts, int32_t numTokens)
{
    auto maxNumCtasInBatchDim = Routing::getMaxNumCtasInBatchDim(numTokens, topK, numExperts, mTileTokensDim);
    return mRunner.getWorkspaceSizeInBytes(
        numTokens, hiddenSize, intermediateSize, {}, numTokens, numExperts, maxNumCtasInBatchDim);
}
} // namespace Gemm2

namespace MoE
{
Runner::Runner(trtllm::gen::Dtype dtypeElt, bool useDeepSeekFp8)
    : mPermuteGemm1(PermuteGemm1::Runner(dtypeElt, useDeepSeekFp8))
    , mGemm2(Gemm2::Runner(dtypeElt, tg::Dtype::Bfloat16, useDeepSeekFp8))
{
}

void Runner::setOpsData(MoERunnerArgs const& args, MoEWorkspace const& workspace,
    moe::dev::convertsf::Data& convertSfData, moe::dev::activation::Data& activationData,
    moe::dev::finalize::Data& finalizeData)
{
    // Setup sf conversion data if needed
    convertSfData.inSfPtr = args.hidden_states_scale;
    convertSfData.outSfPtr = workspace.hidden_states_scale_linear;
    convertSfData.hiddenDimSf = args.hidden_size / 16;
    convertSfData.numTokens = args.num_tokens;
    convertSfData.sfLayoutSrc = tg::SfLayout::R128c4;
    convertSfData.sfLayoutDst = tg::SfLayout::Linear;
    convertSfData.mUsePdl = true;

    // Setup activation data
    activationData.mDtypeElt = args.mDtypeElt;
    activationData.mUsePdl = true;
    activationData.mUseDeepSeekFp8 = true;
    activationData.inPtr = workspace.gemm1_output;
    activationData.outPtr = workspace.activation_output;
    activationData.inDqSfsPtr = workspace.gemm1_output_scale;
    activationData.outDqSfsPtr = workspace.activation_output_scale;
    activationData.innerDim = args.intermediate_size * 2;
    activationData.topK = args.top_k;
    activationData.numTokens = args.num_tokens;
    activationData.expandedIdxToPermutedIdx = workspace.expanded_idx_to_permuted_idx;

    activationData.totalNumPaddedTokens = workspace.total_num_padded_tokens;

    // Setup finalize data
    finalizeData.mDtypeElt = args.mDtypeOut;
    finalizeData.mDtypeExpW = args.mDtypeExpW;
    finalizeData.mUsePdl = true;
    finalizeData.mUseDeepSeekFp8 = false;
    finalizeData.inPtr = workspace.gemm2_output;
    finalizeData.outPtr = args.output;
    finalizeData.inDqSfsPtr = workspace.gemm2_output_scale;
    finalizeData.outDqSfsPtr = args.output_scale;
    if (args.mUseRoutingScalesOnInput)
    {
        finalizeData.expertWeightsPtr = nullptr;
    }
    else
    {
        finalizeData.expertWeightsPtr = workspace.expert_weights;
    }
    finalizeData.expandedIdxToPermutedIdx = workspace.expanded_idx_to_permuted_idx;
    finalizeData.numTokens = args.num_tokens;
    finalizeData.numExperts = args.num_experts;
    finalizeData.topK = args.top_k;
    finalizeData.hiddenDim = args.hidden_size;
    finalizeData.totalNumPaddedTokens = workspace.total_num_padded_tokens;
}

std::tuple<int32_t, int32_t> Runner::getWorkspaceSizeInBytes(MoERunnerArgs const& args)
{
    auto workspace_size_fc1 = static_cast<int32_t>(mPermuteGemm1.getWorkspaceSizeInBytes(
        args.top_k, args.hidden_size, args.intermediate_size, args.local_num_experts, args.num_tokens));
    auto workspace_size_fc2 = static_cast<int32_t>(mGemm2.getWorkspaceSizeInBytes(
        args.top_k, args.hidden_size, args.intermediate_size, args.local_num_experts, args.num_tokens));
    return std::make_tuple(workspace_size_fc1, workspace_size_fc2);
}

void Runner::run(MoERunnerArgs const& args, MoEWorkspace const& workspace, int device, cudaStream_t stream)
{
    // Setup all operation data
    moe::dev::activation::Data activationData;
    moe::dev::finalize::Data finalizeData;
    moe::dev::convertsf::Data convertSfData;

    setOpsData(args, workspace, convertSfData, activationData, finalizeData);

    void* hidden_states_scale_linear{args.hidden_states_scale};

    mPermuteGemm1.run(args.hidden_states, hidden_states_scale_linear, args.gemm1_weights, args.gemm1_weights_scale,
        workspace.expert_weights, args.output1_scales_scalar, args.output1_scales_gate_scalar, workspace.gemm1_output,
        workspace.gemm1_output_scale, args.top_k, args.hidden_size, args.intermediate_size, args.local_num_experts,
        args.num_tokens, workspace.permuted_idx_to_token_idx, workspace.num_non_exiting_ctas,
        workspace.total_num_padded_tokens, workspace.cta_idx_xy_to_batch_idx, workspace.cta_idx_xy_to_mn_limit,
        workspace.bmm1_workspace, args.mUseRoutingScalesOnInput, device, stream);

    // We do not fuse activation with FC1 for DeepSeek FP8 due to the weights shuffling constraint.
    void* gemm2_input = workspace.gemm1_output;
    void* gemm2_input_scale = workspace.gemm1_output_scale;
    // We do activation only for DeepSeek FP8, as cubins do not have fused activation.
    if (args.mDtypeElt == tg::Dtype::E4m3 && args.mUseDeepSeekFp8)
    {
        // Run activation
        moe::dev::activation::run(activationData, stream);
        gemm2_input = workspace.activation_output;
        gemm2_input_scale = workspace.activation_output_scale;
    }

    // Run gemm2
    mGemm2.run(gemm2_input, gemm2_input_scale, args.gemm2_weights, args.gemm2_weights_scale, args.output2_scales_scalar,
        workspace.gemm2_output, workspace.gemm2_output_scale, args.top_k, args.hidden_size, args.intermediate_size,
        args.local_num_experts, args.num_tokens, workspace.num_non_exiting_ctas, workspace.total_num_padded_tokens,
        workspace.cta_idx_xy_to_batch_idx, workspace.cta_idx_xy_to_mn_limit, workspace.bmm2_workspace, device, stream);

    // Run finalize
    moe::dev::finalize::run(finalizeData, stream);
}
} // namespace MoE

} // namespace trtllmGenFp8BlockScaleMoe
} // namespace kernels
} // namespace tensorrt_llm
