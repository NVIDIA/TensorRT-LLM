/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "gemmCommon.h"
#include "gemmList.h"
#include "runner.h"
#include "trtllmGenSrc/DevKernel.h"
#include "trtllmGenSrc/RoutingKernel.h"
#include <iostream>

namespace tensorrt_llm
{
namespace kernels
{
namespace trtllmGenFp8BlockScaleMoe
{

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
    TLLM_CHECK_ERROR((1 << out) == val, "Expected ", name, " to be a power of 2, got ", val);
    return out;
}
} // namespace

Runner::Runner() {}

void Runner::run(float* routingLogits, void* routingBias, int32_t num_tokens, int32_t num_experts, int32_t top_k,
    int32_t n_group, int32_t topk_group, float routed_scaling_factor, int32_t* routingExpertIndexes,
    int32_t* permuted_idx_size, int32_t* expanded_idx_to_permuted_idx, int32_t* permuted_idx_to_expanded_idx,
    int32_t* permuted_idx_to_token_idx, void* expert_weights, uint8_t* num_tokens_per_expert, cudaStream_t stream)
{
    // TODO: remove this once we have a way to get the tileN
    int32_t tileN = 128;
    // int32_t tileN = Gemm1::getTileN();

    moe::dev::routing::Data routingData;
    routingData.mDtypeElt = tg::Dtype::E4m3; // no-op for now as hidden_state is not input
    routingData.mDtypeExpW = tg::Dtype::Bfloat16;
    routingData.mUsePdl = false;

    // output:
    routingData.mPtrExpertIdx = routingExpertIndexes;
    routingData.mPtrPermutedIdxSize = permuted_idx_size;
    routingData.mPtrExpandedIdxToPermutedIdx = expanded_idx_to_permuted_idx;
    routingData.mPtrPermutedIdxToExpandedIdx = permuted_idx_to_expanded_idx;
    routingData.mPtrPermutedIdxToTokenIdx = permuted_idx_to_token_idx;
    routingData.mPtrNumTokensPerExpert = num_tokens_per_expert;
    routingData.mPtrExpertWeights = expert_weights;
    // input:
    // routingData.mPtrRoutingWeights = args.mRoutingWeights;  // routing weights (don't need if not using gemm)
    routingData.mPtrRoutingBias = routingBias;
    routingData.mPtrScores = routingLogits;
    // routingData.mPtrIn = args.mInputActs;
    routingData.mNumTokens = num_tokens;
    // routingData.mHiddenDim = args.mHiddenDim;
    routingData.mNumExperts = num_experts;
    routingData.mNumExpertGroups = n_group;
    routingData.mNumLimitedGroups = topk_group;
    routingData.mTopK = top_k;
    routingData.mPaddingLog2 = computeLog2(tileN);
    routingData.mLocalExpertsStartIdx = 0;
    routingData.mLocalExpertsSrideLog2 = 0;
    routingData.mNumLocalExperts = num_experts;
    routingData.mRouteScale = routed_scaling_factor;
    routingData.mUseRoutingSoftmax = false;
    moe::dev::routing::run(routingData, stream);
}
} // namespace Routing

namespace PermuteGemm1
{
Runner::Runner() {}

void Runner::run(void* hidden_state, float* hidden_state_scale, void* weight, float* weight_scale, void* output,
    float* output_scale, int32_t hidden_size, int32_t intermediate_size, int32_t num_experts, int32_t num_tokens,
    int32_t* num_tokens_per_expert, int32_t* permuted_idx_to_token_idx, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(gemmList.size() == 1, "Currently only one kernel is supported");
    auto const& kernelInfo = gemmList[0];

    gemmCommon::MyOptions options;
    options.mBatchM = false;
    options.mTransposeMmaOutput = true;
    options.mBatchedM = {};
    options.mBatchedN = std::vector(num_tokens_per_expert, num_tokens_per_expert + num_experts);
    options.mNumTokens = num_tokens;
    options.mNumBatches = (int) options.mBatchedN.size();
    options.mM = 2 * intermediate_size;
    options.mN = 256; // A default value in GemmOptions.h that is not supposed to be used. Same as trtllm-gen behavior.
    options.mK = hidden_size;
    options.mClusterDimX = 1;
    options.mClusterDimY = 1;
    options.mClusterDimZ = 1;
    options.mAllReduceAlgo = gemmCommon::gemm::AllReduceAlgo::None;
    options.mSplitK = gemmCommon::gemm::SplitK::None;
    gemmCommon::copyKernelInfoToOptions(kernelInfo, options);
    gemmCommon::batchedGemm::checkAndUpdateGemmOptions(options, true, false, false);

    gemmCommon::BatchedGemmData batchedGemmData;
    gemmCommon::setSingleBatchedGemmData(weight, hidden_state, output, nullptr, nullptr, weight_scale,
        hidden_state_scale, output_scale, permuted_idx_to_token_idx, nullptr, nullptr, options, batchedGemmData);

    gemmCommon::launchGemmFromData(kernelInfo, options, batchedGemmData, stream);
}
} // namespace PermuteGemm1

namespace Gemm2
{
Runner::Runner(tg::Dtype outputDtype)
    : mOutputDtype(outputDtype)
{
}

void Runner::run(void* permuted_hidden_state, float* permuted_hidden_state_scale, void* weight, float* weight_scale,
    void* output, float* output_scale, int32_t hidden_size, int32_t intermediate_size, int32_t num_experts,
    int32_t* num_tokens_per_expert, cudaStream_t stream)
{
    std::vector<int32_t> selectedIndex;
    for (size_t i = 0; i < gemmList.size(); i++)
    {
        if (gemmList[i].dtypeC == mOutputDtype)
        {
            selectedIndex.push_back(i);
        }
    }
    TLLM_CHECK_WITH_INFO(selectedIndex.size() != 0, "No kernel found for the given output type");
    TLLM_CHECK_WITH_INFO(selectedIndex.size() == 1, "Multiple kernels found for the given output type");
    auto const& kernelInfo = gemmList[*selectedIndex.begin()];

    gemmCommon::MyOptions options;
    options.mBatchM = false;
    options.mTransposeMmaOutput = true;
    options.mBatchedM = {};
    options.mBatchedN = std::vector(num_tokens_per_expert, num_tokens_per_expert + num_experts);
    options.mNumTokens = -1; // not used
    options.mNumBatches = (int) options.mBatchedN.size();
    options.mM = hidden_size;
    options.mN = 256; // A default value in GemmOptions.h that is not supposed to be used. Same as trtllm-gen behavior.
    options.mK = intermediate_size;
    options.mClusterDimX = 1;
    options.mClusterDimY = 1;
    options.mClusterDimZ = 1;
    options.mAllReduceAlgo = gemmCommon::gemm::AllReduceAlgo::None;
    options.mSplitK = gemmCommon::gemm::SplitK::None;
    gemmCommon::copyKernelInfoToOptions(kernelInfo, options);
    gemmCommon::batchedGemm::checkAndUpdateGemmOptions(options, true, false, false);

    gemmCommon::BatchedGemmData batchedGemmData;
    gemmCommon::setSingleBatchedGemmData(weight, permuted_hidden_state, output, nullptr, nullptr, weight_scale,
        permuted_hidden_state_scale, output_scale, nullptr, nullptr, nullptr, options, batchedGemmData);

    gemmCommon::launchGemmFromData(kernelInfo, options, batchedGemmData, stream);
}
} // namespace Gemm2

namespace MoE
{
Runner::Runner() {}

void Runner::setOpsData(MoERunnerArgs const& args, MoEWorkspace const& workspace,
    moe::dev::activation::Data& activationData, moe::dev::finalize::Data& finalizeData)
{
    // Setup activation data
    activationData.mDtypeElt = args.mDtypeElt;
    activationData.mUsePdl = false;
    activationData.mUseDeepSeekFp8 = true;
    activationData.inPtr = workspace.gemm1_output;
    activationData.outPtr = workspace.activation_output;
    activationData.inDqSfsPtr = workspace.gemm1_output_scale;
    activationData.outDqSfsPtr = workspace.activation_output_scale;
    activationData.permutedIdxToExpandedIdx = workspace.permuted_idx_to_expanded_idx;
    activationData.innerDim = args.intermediate_size * 2;
    activationData.outerDim = workspace.total_num_padded_tokens;
    activationData.totalNumPaddedTokens = workspace.total_num_padded_tokens;

    // Setup finalize data
    finalizeData.mDtypeElt = args.mDtypeOut;
    finalizeData.mDtypeExpW = args.mDtypeExpW;
    finalizeData.mUsePdl = false;
    finalizeData.mUseDeepSeekFp8 = true;
    finalizeData.inPtr = workspace.gemm2_output;
    finalizeData.outPtr = args.output;
    finalizeData.inDqSfsPtr = workspace.gemm2_output_scale;
    finalizeData.outDqSfsPtr = args.output_scale;
    finalizeData.expertWeightsPtr = workspace.expert_weights;
    finalizeData.expandedIdxToPermutedIdx = workspace.expanded_idx_to_permuted_idx;
    finalizeData.numTokens = args.num_tokens;
    finalizeData.numExperts = args.num_experts;
    finalizeData.topK = args.top_k;
    finalizeData.hiddenDim = args.hidden_size;
    finalizeData.totalNumPaddedTokens = workspace.total_num_padded_tokens;
}

void Runner::run(MoERunnerArgs const& args, MoEWorkspace const& workspace, cudaStream_t stream)
{
    // Setup all operation data
    moe::dev::activation::Data activationData;
    moe::dev::finalize::Data finalizeData;

    setOpsData(args, workspace, activationData, finalizeData);

    assert(workspace.ProjUpTileN == 128);
    // Calling routing outside to properly allocate workspace
    // moe::dev::routing::run(routingData, stream);

    PermuteGemm1::Runner permuteGemm1;
    permuteGemm1.run(args.hidden_states, args.hidden_states_scale, args.gemm1_weights, args.gemm1_weights_scale,
        workspace.gemm1_output, workspace.gemm1_output_scale, args.hidden_size, args.intermediate_size,
        args.num_experts, args.num_tokens, workspace.num_tokens_per_expert, workspace.permuted_idx_to_token_idx,
        stream);

    // Run activation
    moe::dev::activation::run(activationData, stream);

    // Run gemm2
    Gemm2::Runner gemm2(tg::Dtype::Bfloat16);
    gemm2.run(workspace.activation_output, workspace.activation_output_scale, args.gemm2_weights,
        args.gemm2_weights_scale, workspace.gemm2_output, workspace.gemm2_output_scale, args.hidden_size,
        args.intermediate_size, args.num_experts, workspace.num_tokens_per_expert, stream);

    // Run finalize
    moe::dev::finalize::run(finalizeData, stream);
}
} // namespace MoE

} // namespace trtllmGenFp8BlockScaleMoe
} // namespace kernels
} // namespace tensorrt_llm
