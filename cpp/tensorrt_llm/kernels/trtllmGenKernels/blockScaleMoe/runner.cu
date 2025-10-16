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

namespace btg = batchedGemm::trtllm::gen;

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
    if ((1 << out) != val)
    {
        out = -1;
    }
    return out;
}
} // namespace

Runner::Runner() {}

Runner::Runner(int32_t tileTokensDim)
    : mTileTokensDim(tileTokensDim)
{
}

void Runner::run(void* routingLogits, void* routingBias, int32_t numTokens, int32_t numExperts, int32_t topK,
    int32_t nGroup, int32_t topkGroup, int32_t localExpertOffset, int32_t localNumExperts, float routedScalingFactor,
    int32_t* routingExpertIndexes, int32_t* expertCountHistogram, int32_t* permutedIdxSize,
    int32_t* expandedIdxToPermutedIdx, int32_t* permutedIdxToExpandedIdx, int32_t* permutedIdxToTokenIdx,
    void* expertWeights, int32_t* expertIds, int32_t* numTokensPerExpert, int32_t* ctaIdxXyToBatchIdx,
    int32_t* ctaIdxXyToMnLimit, int32_t* numNonExitingCtas, btg::Dtype dtypeElt, bool useRoutingScalesOnInput,
    bool useDeepSeekFp8, RoutingMethodType routingMethodType, cudaStream_t stream)
{
    if (routingMethodType == RoutingMethodType::DeepSeekV3)
    {
        TLLM_CHECK_WITH_INFO(topK <= 8, "For DeepSeek routing method, must have topK <= 8");
        TLLM_CHECK_WITH_INFO(topkGroup <= 4, "For DeepSeek routing method, must have topkGroup <= 4");
        moe::dev::routing::routingDeepSeek::Data routingData;
        routingData.mDtypeExpW = btg::Dtype::Bfloat16;
        routingData.mUsePdl = true;

        // output:
        routingData.mPtrTopKPacked = routingExpertIndexes;
        routingData.mPtrExpertCounts = expertCountHistogram;
        routingData.mPtrPermutedIdxSize = permutedIdxSize;
        routingData.mPtrExpandedIdxToPermutedIdx = expandedIdxToPermutedIdx;
        routingData.mPtrPermutedIdxToTokenIdx = permutedIdxToTokenIdx;
        routingData.mPtrTopKWeights = expertWeights;

        routingData.mPtrCtaIdxXyToBatchIdx = ctaIdxXyToBatchIdx;
        routingData.mPtrCtaIdxXyToMnLimit = ctaIdxXyToMnLimit;
        routingData.mPtrNumNonExitingCtas = numNonExitingCtas;

        // input:
        routingData.mPtrRoutingBias = routingBias;
        // Pass-through raw pointer; kernels will cast to the proper InputT based on routing method
        routingData.mPtrScores = expertIds == nullptr ? routingLogits : nullptr;
        routingData.mPtrTopKIds = expertIds;
        routingData.mNumTokens = numTokens;
        routingData.mNumExperts = numExperts;
        routingData.mNumExpertGroups = nGroup;
        routingData.mNumLimitedGroups = topkGroup;
        routingData.mTopK = topK;
        routingData.mPaddingLog2 = computeLog2(mTileTokensDim);
        routingData.mTileTokensDim = mTileTokensDim;
        routingData.mLocalExpertsStartIdx = localExpertOffset;
        routingData.mLocalExpertsStrideLog2 = 0;
        routingData.mNumLocalExperts = localNumExperts;
        routingData.mRouteScale = routedScalingFactor;
        routingData.mUseRoutingSoftmax = false;
        moe::dev::routing::routingDeepSeek::run(routingData, stream);
    }
    else if (routingMethodType == RoutingMethodType::Llama4)
    {
        TLLM_CHECK_WITH_INFO(topK == 1, "For Llama routing method, must have topK == 1");
        if (nGroup > 0 || topkGroup > 0)
        {
            TLLM_LOG_WARNING("For Llama routing method, nGroup/topkGroup is ignored, got %d/%d.", nGroup, topkGroup);
        }
        moe::dev::routing::routingLlama4::Data routingData;
        routingData.mDtypeExpW = btg::Dtype::Bfloat16;
        routingData.mUsePdl = true;

        // output:
        routingData.mPtrTopKPacked = routingExpertIndexes;
        routingData.mPtrExpertCounts = expertCountHistogram;
        routingData.mPtrPermutedIdxSize = permutedIdxSize;
        routingData.mPtrExpandedIdxToPermutedIdx = expandedIdxToPermutedIdx;
        routingData.mPtrPermutedIdxToTokenIdx = permutedIdxToTokenIdx;
        routingData.mPtrTopKWeights = expertWeights;

        routingData.mPtrCtaIdxXyToBatchIdx = ctaIdxXyToBatchIdx;
        routingData.mPtrCtaIdxXyToMnLimit = ctaIdxXyToMnLimit;
        routingData.mPtrNumNonExitingCtas = numNonExitingCtas;
        // routingData.mAllToAllRouteAct = false;

        // input:
        // routingData.mPtrRoutingWeights = args.mRoutingWeights;  // routing weights (don't need if not using gemm)
        // routingData.mPtrRoutingBias = routingBias;

        // Pass-through raw pointer; kernels will cast to the proper InputT based on routing method
        routingData.mPtrScores = expertIds == nullptr ? routingLogits : nullptr;
        routingData.mPtrTopKIds = expertIds;
        // routingData.mPtrIn = args.mInputActs;
        routingData.mNumTokens = numTokens;
        // routingData.mHiddenDim = args.mHiddenDim;
        routingData.mNumExperts = numExperts;
        // routingData.mNumExpertGroups = nGroup;
        // routingData.mNumLimitedGroups =topkGroup;
        routingData.mTopK = topK;
        routingData.mPaddingLog2 = computeLog2(mTileTokensDim);
        routingData.mTileTokensDim = mTileTokensDim;
        routingData.mLocalExpertsStartIdx = localExpertOffset;
        routingData.mLocalExpertsStrideLog2 = 0;
        routingData.mNumLocalExperts = localNumExperts;
        // routingData.mRouteScale = routed_scaling_factor;
        // routingData.mUseRoutingSoftmax = false;
        moe::dev::routing::routingLlama4::run(routingData, stream);
    }
    else if (routingMethodType == RoutingMethodType::Renormalize /* default */
        || routingMethodType == RoutingMethodType::RenormalizeNaive /* Softmax -> TopK */)
    {
        moe::dev::routing::routingRenormalize::Data routingData;

        //
        // Config
        //

        routingData.mDtypeExpW = btg::Dtype::Bfloat16;
        // routingData.mDtypeElt = dtypeElt; // no-op for now as hidden_state is not input
        routingData.mUsePdl = true;
        routingData.mDoSoftmaxBeforeTopK = routingMethodType == RoutingMethodType::RenormalizeNaive;
        routingData.mNormTopkProb = routingMethodType == RoutingMethodType::RenormalizeNaive;

        // Pass-through raw pointer; kernels will cast to the proper InputT based on routing method
        routingData.mPtrScores = expertIds == nullptr ? routingLogits : nullptr;
        //
        // Outputs
        //
        routingData.mPtrTopKPacked = routingExpertIndexes;
        routingData.mPtrExpertCounts = expertCountHistogram;
        routingData.mPtrPermutedIdxSize = permutedIdxSize;
        routingData.mPtrExpandedIdxToPermutedIdx = expandedIdxToPermutedIdx;
        routingData.mPtrPermutedIdxToTokenIdx = permutedIdxToTokenIdx;
        routingData.mPtrTopKWeights = expertWeights;
        routingData.mPtrTopKIds = expertIds;
        //
        // Grouped Gemm Launch Config Buffers
        //
        routingData.mPtrCtaIdxXyToBatchIdx = ctaIdxXyToBatchIdx;
        routingData.mPtrCtaIdxXyToMnLimit = ctaIdxXyToMnLimit;
        routingData.mPtrNumNonExitingCtas = numNonExitingCtas;

        //
        // Inputs
        //
        routingData.mNumTokens = numTokens;
        routingData.mNumExperts = numExperts;
        routingData.mTopK = topK;
        routingData.mPaddingLog2 = computeLog2(mTileTokensDim);
        routingData.mTileTokensDim = mTileTokensDim;
        routingData.mLocalExpertsStartIdx = localExpertOffset;
        routingData.mLocalExpertsStrideLog2 = 0;
        routingData.mNumLocalExperts = localNumExperts;

        moe::dev::routing::routingRenormalize::run(routingData, stream);
    }
    else
    {
        TLLM_CHECK_WITH_INFO(false, "Unimplemented routing method %s of enum %d",
            serializeMoeRoutingMethodType(routingMethodType).c_str(), (int) routingMethodType);
    }
}
} // namespace Routing

namespace PermuteGemm1
{

tensorrt_llm::kernels::TrtllmGenBatchedGemmRunnerOptions getOptions(
    btg::Dtype dtypeAct, btg::Dtype dtypeWeights, int32_t tileTokensDim, bool useDeepSeekFp8, ActType actType)
{
    tensorrt_llm::kernels::TrtllmGenBatchedGemmRunnerOptions options
        = {// Swap A and B dtypes because transposeMmaOutput is hardcoded to true
            .dtypeA = dtypeWeights,
            .dtypeB = dtypeAct,
            .dtypeC = dtypeAct,
            .actType = actType,
            .deepSeekFp8 = useDeepSeekFp8,
            .fusedAct = !useDeepSeekFp8,
            .routeAct = true,
            .staticBatch = false,
            .transposeMmaOutput = true,
            .tileSize = tileTokensDim,
            .epilogueTileM = useDeepSeekFp8 ? 64 : 128};
    return options;
}

Runner::Runner(btg::Dtype dtypeAct, btg::Dtype dtypeWeights, bool useDeepSeekFp8, int tileTokensDim, ActType actType)
    : mDtypeAct(dtypeAct)
    , mDtypeWeights(dtypeWeights)
    , mTileTokensDim(tileTokensDim)
    , mRunner(tensorrt_llm::kernels::TrtllmGenBatchedGemmRunner(
          getOptions(mDtypeAct, mDtypeWeights, mTileTokensDim, useDeepSeekFp8, actType)))
{
}

void Runner::run(void* hiddenState, void* hiddenStateScale, void* weights, void* weightsScale, void* expertWeights,
    float* outputScalesScalar, float* outputScalesGateScalar, float* ptrBias, float* ptrAlpha, float* ptrBeta,
    float* ptrClampLimit, void* output, void* outputScale, int32_t topK, int32_t hiddenSize, int32_t intermediateSize,
    int32_t numExperts, int32_t numTokens, int32_t* permutedIdxToTokenIdx, int32_t* ptrNumNonExitingCtas,
    int32_t* ptrTotalNumPaddedTokens, int32_t* ptrCtaIdxXyToBatchIdx, int32_t* ptrCtaIdxXyToMnLimit,
    void* bmm1Workspace, bool useRoutingScalesOnInput, int device, cudaStream_t stream, int32_t configIndex)
{
    auto maxNumCtasInBatchDim = Routing::getMaxNumCtasInBatchDim(numTokens, topK, numExperts, mTileTokensDim);
    mRunner.run(numTokens, 2 * intermediateSize, hiddenSize, {}, numTokens, numExperts, maxNumCtasInBatchDim,
        hiddenState, hiddenStateScale, weights, weightsScale, expertWeights, /* perTokensSfB */ nullptr,
        outputScalesScalar, outputScalesGateScalar, ptrBias, ptrAlpha, ptrBeta, ptrClampLimit, output, outputScale,
        permutedIdxToTokenIdx, ptrTotalNumPaddedTokens, ptrCtaIdxXyToBatchIdx, ptrCtaIdxXyToMnLimit,
        ptrNumNonExitingCtas, bmm1Workspace, stream, device, configIndex);
}

size_t Runner::getWorkspaceSizeInBytes(int32_t topK, int32_t hiddenSize, int32_t intermediateSize, int32_t numExperts,
    int32_t numTokens, int32_t configIndex) const
{
    auto maxNumCtasInBatchDim = Routing::getMaxNumCtasInBatchDim(numTokens, topK, numExperts, mTileTokensDim);
    return mRunner.getWorkspaceSizeInBytes(
        numTokens, 2 * intermediateSize, hiddenSize, {}, numTokens, numExperts, maxNumCtasInBatchDim, configIndex);
}

int32_t Runner::getDefaultValidConfigIndex(
    int32_t topK, int32_t hiddenSize, int32_t intermediateSize, int32_t numExperts, int32_t numTokens) const
{
    auto maxNumCtasInBatchDim = Routing::getMaxNumCtasInBatchDim(numTokens, topK, numExperts, mTileTokensDim);
    return mRunner.getDefaultValidConfigIndex(
        numTokens, 2 * intermediateSize, hiddenSize, {}, numTokens, numExperts, maxNumCtasInBatchDim);
}

bool Runner::isValidConfigIndex(int32_t configIndex, int32_t topK, int32_t hiddenSize, int32_t intermediateSize,
    int32_t numExperts, int32_t numTokens) const
{
    auto maxNumCtasInBatchDim = Routing::getMaxNumCtasInBatchDim(numTokens, topK, numExperts, mTileTokensDim);

    auto const isValid = mRunner.isValidConfigIndex(
        configIndex, numTokens, 2 * intermediateSize, hiddenSize, {}, numTokens, numExperts, maxNumCtasInBatchDim);

    return isValid;
}

std::vector<int64_t> Runner::getPassingConfigIndices() const
{
    return mRunner.getPassingConfigIndices();
}

} // namespace PermuteGemm1

namespace Gemm2
{
tensorrt_llm::kernels::TrtllmGenBatchedGemmRunnerOptions getOptions(
    btg::Dtype dtypeAct, btg::Dtype dtypeWeights, btg::Dtype dtypeOut, int32_t tileTokensDim, bool useDeepSeekFp8)
{
    tensorrt_llm::kernels::TrtllmGenBatchedGemmRunnerOptions options
        = {// Swap A and B dtypes because transposeMmaOutput is hardcoded to true
            .dtypeA = dtypeWeights,
            .dtypeB = dtypeAct,
            .dtypeC = dtypeOut,
            .deepSeekFp8 = useDeepSeekFp8,
            .fusedAct = false,
            .routeAct = false,
            .staticBatch = false,
            .transposeMmaOutput = true,
            .tileSize = tileTokensDim,
            .epilogueTileM = useDeepSeekFp8 ? 64 : 128};
    return options;
}

Runner::Runner(
    btg::Dtype dtypeAct, btg::Dtype dtypeWeights, btg::Dtype dtypeOut, bool useDeepSeekFp8, int tileTokensDim)
    : mDtypeAct(dtypeAct)
    , mDtypeWeights(dtypeWeights)
    , mDtypeOut(dtypeOut)
    , mTileTokensDim(tileTokensDim)
    , mRunner(tensorrt_llm::kernels::TrtllmGenBatchedGemmRunner(
          getOptions(dtypeAct, dtypeWeights, dtypeOut, tileTokensDim, useDeepSeekFp8)))
{
}

void Runner::run(void* permutedHiddenState, void* permutedHiddenStateScale, void* weights, void* weightsScale,
    float* outputScalesScalar, float* ptrBias, void* output, void* outputScale, int32_t topK, int32_t hiddenSize,
    int32_t intermediateSize, int32_t numExperts, int32_t numTokens, int32_t* ptrNumNonExitingCtas,
    int32_t* ptrTotalNumPaddedTokens, int32_t* ptrCtaIdxXyToBatchIdx, int32_t* ptrCtaIdxXyToMnLimit,
    void* bmm2Workspace, int device, cudaStream_t stream, int32_t configIndex)
{
    auto maxNumCtasInBatchDim = Routing::getMaxNumCtasInBatchDim(numTokens, topK, numExperts, mTileTokensDim);
    mRunner.run(numTokens, hiddenSize, intermediateSize, {}, numTokens, numExperts, maxNumCtasInBatchDim,
        permutedHiddenState, permutedHiddenStateScale, weights, weightsScale, /* perTokensSfA */ nullptr,
        /* perTokensSfB */ nullptr, outputScalesScalar, /* outputScalesGateScalar */ nullptr, ptrBias,
        /* ptrAlpha */ nullptr, /* ptrBeta */ nullptr, /* clampLimit */ nullptr, output, outputScale,
        /* permutedIdxToTokenIdx */ nullptr, ptrTotalNumPaddedTokens, ptrCtaIdxXyToBatchIdx, ptrCtaIdxXyToMnLimit,
        ptrNumNonExitingCtas, bmm2Workspace, stream, device, configIndex);
}

size_t Runner::getWorkspaceSizeInBytes(int32_t topK, int32_t hiddenSize, int32_t intermediateSize, int32_t numExperts,
    int32_t numTokens, int32_t configIndex) const
{
    auto maxNumCtasInBatchDim = Routing::getMaxNumCtasInBatchDim(numTokens, topK, numExperts, mTileTokensDim);
    return mRunner.getWorkspaceSizeInBytes(
        numTokens, hiddenSize, intermediateSize, {}, numTokens, numExperts, maxNumCtasInBatchDim, configIndex);
}

int32_t Runner::getDefaultValidConfigIndex(
    int32_t topK, int32_t hiddenSize, int32_t intermediateSize, int32_t numExperts, int32_t numTokens) const
{
    auto maxNumCtasInBatchDim = Routing::getMaxNumCtasInBatchDim(numTokens, topK, numExperts, mTileTokensDim);
    return mRunner.getDefaultValidConfigIndex(
        numTokens, hiddenSize, intermediateSize, {}, numTokens, numExperts, maxNumCtasInBatchDim);
}

bool Runner::isValidConfigIndex(int32_t configIndex, int32_t topK, int32_t hiddenSize, int32_t intermediateSize,
    int32_t numExperts, int32_t numTokens) const
{

    auto const maxNumCtasInBatchDim = Routing::getMaxNumCtasInBatchDim(numTokens, topK, numExperts, mTileTokensDim);

    auto const isValid = mRunner.isValidConfigIndex(
        configIndex, numTokens, hiddenSize, intermediateSize, {}, numTokens, numExperts, maxNumCtasInBatchDim);

    return isValid;
}

std::vector<int64_t> Runner::getPassingConfigIndices() const
{
    return mRunner.getPassingConfigIndices();
}

} // namespace Gemm2

namespace MoE
{
Runner::Runner(
    btg::Dtype dtypeAct, btg::Dtype dtypeWeights, bool useDeepSeekFp8, int32_t tileTokensDim, ActType actType)
    : mPermuteGemm1(PermuteGemm1::Runner(dtypeAct, dtypeWeights, useDeepSeekFp8, tileTokensDim, actType))
    , mGemm2(Gemm2::Runner(dtypeAct, dtypeWeights, btg::Dtype::Bfloat16, useDeepSeekFp8, tileTokensDim))
{
    auto const& gemm1PassingIndices = mPermuteGemm1.getPassingConfigIndices();
    auto const& gemm2PassingIndices = mGemm2.getPassingConfigIndices();

    auto const totalPassingIndices = gemm1PassingIndices.size() * gemm2PassingIndices.size();
    mPassingConfigs.reserve(totalPassingIndices);

    for (auto const& indexGemm1 : gemm1PassingIndices)
    {
        for (auto const& indexGemm2 : gemm2PassingIndices)
        {
            mPassingConfigs.push_back(MoEConfig{indexGemm1, indexGemm2});
        }
    }

    TLLM_CHECK_WITH_INFO(!mPassingConfigs.empty(), "No compatible configs found for the fp8 block scale MoE runner.");
}

Runner::Runner(btg::Dtype dtypeElt, bool useDeepSeekFp8, int32_t tileTokensDim)
    : Runner(dtypeElt, dtypeElt, useDeepSeekFp8, tileTokensDim)
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
    convertSfData.sfLayoutSrc = btg::SfLayout::R128c4;
    convertSfData.sfLayoutDst = btg::SfLayout::Linear;
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

    if (args.do_finalize)
    {
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
        // We want to fuse unpadding into the finalize kernel, so we need to use the output hidden size.
        finalizeData.hiddenDim = args.hidden_size_output.value_or(args.hidden_size);
        finalizeData.hiddenDimPadded = args.hidden_size;
        finalizeData.totalNumPaddedTokens = workspace.total_num_padded_tokens;
    }
}

std::tuple<int32_t, int32_t> Runner::getWorkspaceSizeInBytes(MoERunnerArgs const& args, int64_t configIndex) const
{
    auto const& config = mPassingConfigs[configIndex];

    auto workspace_size_fc1 = static_cast<int32_t>(mPermuteGemm1.getWorkspaceSizeInBytes(args.top_k, args.hidden_size,
        args.intermediate_size, args.local_num_experts, args.num_tokens, config.gemm1Config));
    auto workspace_size_fc2 = static_cast<int32_t>(mGemm2.getWorkspaceSizeInBytes(args.top_k, args.hidden_size,
        args.intermediate_size, args.local_num_experts, args.num_tokens, config.gemm2Config));
    return std::make_tuple(workspace_size_fc1, workspace_size_fc2);
}

std::vector<int64_t> Runner::getValidConfigIndices(
    int32_t topK, int32_t hiddenSize, int32_t intermediateSize, int32_t numLocalExperts, int32_t numTokens) const
{
    std::vector<int64_t> validIndices;

    for (int i = 0; i < mPassingConfigs.size(); ++i)
    {
        auto const& config = mPassingConfigs[i];

        if (mPermuteGemm1.isValidConfigIndex(
                config.gemm1Config, topK, hiddenSize, intermediateSize, numLocalExperts, numTokens)
            && mGemm2.isValidConfigIndex(
                config.gemm2Config, topK, hiddenSize, intermediateSize, numLocalExperts, numTokens))
        {
            validIndices.push_back(i);
        }
    }

    return validIndices;
}

int64_t Runner::getDefaultValidConfigIndex(
    int32_t topK, int32_t hiddenSize, int32_t intermediateSize, int32_t numLocalExperts, int32_t numTokens) const
{

    int32_t indexGemm1
        = mPermuteGemm1.getDefaultValidConfigIndex(topK, hiddenSize, intermediateSize, numLocalExperts, numTokens);
    int32_t indexGemm2
        = mGemm2.getDefaultValidConfigIndex(topK, hiddenSize, intermediateSize, numLocalExperts, numTokens);

    auto it = std::find_if(mPassingConfigs.begin(), mPassingConfigs.end(),
        [indexGemm1, indexGemm2](MoEConfig cfg)
        { return (cfg.gemm1Config == indexGemm1 && cfg.gemm2Config == indexGemm2); });
    TLLM_CHECK_WITH_INFO(it != mPassingConfigs.end(), "No compatible configs found for the block scale MoE runner.");
    return std::distance(mPassingConfigs.begin(), it);
}

void Runner::run(
    MoERunnerArgs const& args, MoEWorkspace const& workspace, int device, cudaStream_t stream, int64_t configIndex)
{
    // Setup all operation data
    moe::dev::activation::Data activationData;
    moe::dev::finalize::Data finalizeData;
    moe::dev::convertsf::Data convertSfData;
    sync_check_cuda_error(stream);
    setOpsData(args, workspace, convertSfData, activationData, finalizeData);

    void* hidden_states_scale_linear{args.hidden_states_scale};

    auto const& config = mPassingConfigs[configIndex];

    mPermuteGemm1.run(args.hidden_states, hidden_states_scale_linear, args.gemm1_weights, args.gemm1_weights_scale,
        workspace.expert_weights, args.output1_scales_scalar, args.output1_scales_gate_scalar, args.gemm1_bias,
        args.gemm1_alpha, args.gemm1_beta, args.gemm1_clamp_limit, workspace.gemm1_output, workspace.gemm1_output_scale,
        args.top_k, args.hidden_size, args.intermediate_size, args.local_num_experts, args.num_tokens,
        workspace.permuted_idx_to_token_idx, workspace.num_non_exiting_ctas, workspace.total_num_padded_tokens,
        workspace.cta_idx_xy_to_batch_idx, workspace.cta_idx_xy_to_mn_limit, workspace.bmm1_workspace,
        args.mUseRoutingScalesOnInput, device, stream, config.gemm1Config);

    // We do not fuse activation with FC1 for DeepSeek FP8 due to the weights shuffling constraint.
    void* gemm2_input = workspace.gemm1_output;
    void* gemm2_input_scale = workspace.gemm1_output_scale;
    // We do activation only for DeepSeek FP8, as cubins do not have fused activation.
    if (args.mDtypeElt == btg::Dtype::E4m3 && args.mUseDeepSeekFp8)
    {
        // Run activation
        moe::dev::activation::run(activationData, stream);
        gemm2_input = workspace.activation_output;
        gemm2_input_scale = workspace.activation_output_scale;
    }

    // Run gemm2
    mGemm2.run(gemm2_input, gemm2_input_scale, args.gemm2_weights, args.gemm2_weights_scale, args.output2_scales_scalar,
        args.gemm2_bias, workspace.gemm2_output, workspace.gemm2_output_scale, args.top_k, args.hidden_size,
        args.intermediate_size, args.local_num_experts, args.num_tokens, workspace.num_non_exiting_ctas,
        workspace.total_num_padded_tokens, workspace.cta_idx_xy_to_batch_idx, workspace.cta_idx_xy_to_mn_limit,
        workspace.bmm2_workspace, device, stream, config.gemm2Config);

    // Run finalize
    if (args.do_finalize)
    {
        // Run finalize
        moe::dev::finalize::run(finalizeData, stream);
        sync_check_cuda_error(stream);
    }
}
} // namespace MoE

} // namespace trtllmGenFp8BlockScaleMoe
} // namespace kernels
} // namespace tensorrt_llm
