/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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
    int32_t n_group, int32_t topk_group, int32_t local_expert_offset, int32_t local_num_experts,
    float routed_scaling_factor, int32_t* routingExpertIndexes, int32_t* expertCountHistogram,
    int32_t* permuted_idx_size, int32_t* expanded_idx_to_permuted_idx, int32_t* permuted_idx_to_expanded_idx,
    int32_t* permuted_idx_to_token_idx, void* expert_weights, int32_t* num_tokens_per_expert,
    int32_t* cta_idx_xy_to_batch_idx, int32_t* cta_idx_xy_to_mn_limit, int32_t* num_non_exiting_ctas,
    tg::Dtype dtypeElt, cudaStream_t stream)
{
    std::vector<int32_t> selectedIndex;
    for (size_t i = 0; i < PermuteGemm1::gemmList.size(); i++)
    {
        if (PermuteGemm1::gemmList[i].dtypeElt == dtypeElt)
        {
            selectedIndex.push_back(i);
        }
    }
    TLLM_CHECK_WITH_INFO(selectedIndex.size() != 0, "No kernel found for the given element type");
    TLLM_CHECK_WITH_INFO(selectedIndex.size() == 1, "Multiple kernels found for the given element type");
    auto const& kernelInfo = PermuteGemm1::gemmList[*selectedIndex.begin()];
    int32_t tileN = kernelInfo.tileN;

    moe::dev::routing::Data routingData;
    routingData.mDtypeElt = dtypeElt; // no-op for now as hidden_state is not input
    routingData.mDtypeExpW = tg::Dtype::Bfloat16;
    routingData.mUsePdl = true;

    // output:
    routingData.mPtrExpertIdx = routingExpertIndexes;
    routingData.mPtrExpertCounts = expertCountHistogram;
    routingData.mPtrPermutedIdxSize = permuted_idx_size;
    routingData.mPtrExpandedIdxToPermutedIdx = expanded_idx_to_permuted_idx;
    routingData.mPtrPermutedIdxToExpandedIdx = permuted_idx_to_expanded_idx;
    routingData.mPtrPermutedIdxToTokenIdx = permuted_idx_to_token_idx;
    routingData.mPtrNumTokensPerExpert = num_tokens_per_expert;
    routingData.mPtrExpertWeights = expert_weights;

    routingData.mPtrCtaIdxXyToBatchIdx = cta_idx_xy_to_batch_idx;
    routingData.mPtrCtaIdxXyToMnLimit = cta_idx_xy_to_mn_limit;
    routingData.mPtrNumNonExitingCtas = num_non_exiting_ctas;
    routingData.mAllToAllRouteAct = false;

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
    routingData.mLocalExpertsStartIdx = local_expert_offset;
    routingData.mLocalExpertsStrideLog2 = 0;
    routingData.mNumLocalExperts = local_num_experts;
    routingData.mRouteScale = routed_scaling_factor;
    routingData.mUseRoutingSoftmax = false;
    moe::dev::routing::run(routingData, stream);
}
} // namespace Routing

namespace PermuteGemm1
{
Runner::Runner(trtllm::gen::Dtype dtypeElt)
    : mDtypeElt(dtypeElt)
{
}

void Runner::run(void* hidden_state, void* hidden_state_scale, void* weight, void* weight_scale,
    float* output_scales_scalar, float* output_scales_gate_scalar, void* output, void* output_scale, int32_t top_k,
    int32_t hidden_size, int32_t intermediate_size, int32_t num_experts, int32_t num_tokens,
    int32_t* permuted_idx_to_token_idx, int32_t* ptr_num_non_exiting_ctas, int32_t* ptr_total_num_padded_tokens,
    int32_t* ptr_cta_idx_xy_to_batch_idx, int32_t* ptr_cta_idx_xy_to_mn_limit, cudaStream_t stream)
{
    std::vector<int32_t> selectedIndex;
    for (size_t i = 0; i < gemmList.size(); i++)
    {
        if (gemmList[i].dtypeElt == mDtypeElt)
        {
            selectedIndex.push_back(i);
        }
    }
    TLLM_CHECK_WITH_INFO(selectedIndex.size() != 0, "No kernel found for the given element type");
    TLLM_CHECK_WITH_INFO(selectedIndex.size() == 1, "Multiple kernels found for the given element type");
    auto const& kernelInfo = gemmList[*selectedIndex.begin()];

    gemmCommon::MyOptions options;
    options.mTopK = top_k;
    options.mBatchM = false;
    options.mTransposeMmaOutput = true;
    options.mNumTokens = num_tokens;
    options.mNumExperts = num_experts;
    options.mM = 2 * intermediate_size;
    options.mN = 256; // A default value in GemmOptions.h that is not supposed to be used. Same as trtllm-gen behavior.
    options.mK = hidden_size;
    options.mClusterDimX = 1;
    options.mClusterDimY = 1;
    options.mClusterDimZ = 1;
    options.mAllReduceAlgo = gemmCommon::gemm::AllReduceAlgo::None;
    options.mSplitK = gemmCommon::gemm::SplitK::None;
    options.mPtrNumNonExitingCtas = ptr_num_non_exiting_ctas;
    options.mPtrTotalNumPaddedTokens = ptr_total_num_padded_tokens;
    options.mPtrCtaIdxXyToBatchIdx = ptr_cta_idx_xy_to_batch_idx;
    options.mPtrCtaIdxXyToMnLimit = ptr_cta_idx_xy_to_mn_limit;
    options.mSfLayoutB = tg::SfLayout::Linear;
    options.mSfLayoutC = tg::SfLayout::Linear;
    options.mUseCustomLowLatencyImpl = false;
    options.mAllToAllRouteAct = false;
    options.mIsStaticBatch = false;
    options.mBatchedN = std::vector(num_experts, -1);
    gemmCommon::copyKernelInfoToOptions(kernelInfo, options);
    gemmCommon::batchedGemm::checkAndUpdateGemmOptions(options, true, false, false);

    gemmCommon::BatchedGemmData batchedGemmData;
    auto max_num_padded_tokens = Routing::getMaxPermutedPaddedCount(num_tokens, top_k, num_experts, kernelInfo.tileN);
    gemmCommon::setSingleBatchedGemmData(weight, hidden_state, output, output_scales_scalar, output_scales_gate_scalar,
        reinterpret_cast<float*>(weight_scale), reinterpret_cast<float*>(hidden_state_scale),
        reinterpret_cast<float*>(output_scale),
        // FIXME: we pass the same scaling factors in one case for dsfp8 and in the other case for fp4
        // We should pass them once only and decide on the case inside of the setSingleBatchedGemmData
        weight_scale, hidden_state_scale, output_scale, permuted_idx_to_token_idx, nullptr, nullptr, 1, options,
        batchedGemmData, max_num_padded_tokens);

    gemmCommon::launchGemmFromData(kernelInfo, options, batchedGemmData, stream, /*usePDL*/ false);
}
} // namespace PermuteGemm1

namespace Gemm2
{
Runner::Runner(tg::Dtype dtypeElt, tg::Dtype outputDtype)
    : mDtypeElt(dtypeElt)
    , mOutputDtype(outputDtype)
{
}

void Runner::run(void* permuted_hidden_state, void* permuted_hidden_state_scale, void* weight, void* weight_scale,
    float* output_scales_scalar, void* output, void* output_scale, int32_t top_k, int32_t hidden_size,
    int32_t intermediate_size, int32_t num_experts, int32_t num_tokens, int32_t* ptr_num_non_exiting_ctas,
    int32_t* ptr_total_num_padded_tokens, int32_t* ptr_cta_idx_xy_to_batch_idx, int32_t* ptr_cta_idx_xy_to_mn_limit,
    cudaStream_t stream)
{
    std::vector<int32_t> selectedIndex;
    for (size_t i = 0; i < gemmList.size(); i++)
    {
        if (gemmList[i].dtypeElt == mDtypeElt && gemmList[i].dtypeC == mOutputDtype)
        {
            selectedIndex.push_back(i);
        }
    }
    TLLM_CHECK_WITH_INFO(selectedIndex.size() != 0, "No kernel found for the given element and output types");
    TLLM_CHECK_WITH_INFO(selectedIndex.size() == 1, "Multiple kernels found for the given element and output types");
    auto const& kernelInfo = gemmList[*selectedIndex.begin()];

    gemmCommon::MyOptions options;
    options.mTopK = top_k;
    options.mBatchM = false;
    options.mTransposeMmaOutput = true;
    options.mNumExperts = num_experts;
    options.mNumTokens = num_tokens;
    options.mM = hidden_size;
    options.mN = 256; // A default value in GemmOptions.h that is not supposed to be used. Same as trtllm-gen behavior.
    options.mK = intermediate_size;
    options.mClusterDimX = 1;
    options.mClusterDimY = 1;
    options.mClusterDimZ = 1;
    options.mAllReduceAlgo = gemmCommon::gemm::AllReduceAlgo::None;
    options.mSplitK = gemmCommon::gemm::SplitK::None;
    options.mPtrNumNonExitingCtas = ptr_num_non_exiting_ctas;
    options.mPtrTotalNumPaddedTokens = ptr_total_num_padded_tokens;
    options.mPtrCtaIdxXyToBatchIdx = ptr_cta_idx_xy_to_batch_idx;
    options.mPtrCtaIdxXyToMnLimit = ptr_cta_idx_xy_to_mn_limit;
    options.mSfLayoutB = tg::SfLayout::Linear;
    options.mSfLayoutC = tg::SfLayout::Linear;
    options.mUseCustomLowLatencyImpl = false;
    options.mAllToAllRouteAct = false;
    options.mIsStaticBatch = false;
    options.mBatchedN = std::vector(num_experts, -1);
    gemmCommon::copyKernelInfoToOptions(kernelInfo, options);
    gemmCommon::batchedGemm::checkAndUpdateGemmOptions(options, true, false, false);

    gemmCommon::BatchedGemmData batchedGemmData;
    auto max_num_padded_tokens = Routing::getMaxPermutedPaddedCount(num_tokens, top_k, num_experts, kernelInfo.tileN);
    gemmCommon::setSingleBatchedGemmData(weight, permuted_hidden_state, output, output_scales_scalar, nullptr,
        reinterpret_cast<float*>(weight_scale), reinterpret_cast<float*>(permuted_hidden_state_scale),
        reinterpret_cast<float*>(output_scale), weight_scale, permuted_hidden_state_scale, output_scale, nullptr,
        nullptr, nullptr, 1, options, batchedGemmData, max_num_padded_tokens);

    gemmCommon::launchGemmFromData(kernelInfo, options, batchedGemmData, stream);
}
} // namespace Gemm2

namespace MoE
{
Runner::Runner() {}

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
    moe::dev::convertsf::Data convertSfData;

    setOpsData(args, workspace, convertSfData, activationData, finalizeData);

    void* hidden_states_scale_linear{args.hidden_states_scale};

    PermuteGemm1::Runner permuteGemm1(args.mDtypeElt);
    permuteGemm1.run(args.hidden_states, hidden_states_scale_linear, args.gemm1_weights, args.gemm1_weights_scale,
        args.output1_scales_scalar, args.output1_scales_gate_scalar, workspace.gemm1_output,
        workspace.gemm1_output_scale, args.top_k, args.hidden_size, args.intermediate_size, args.local_num_experts,
        args.num_tokens, workspace.permuted_idx_to_token_idx, workspace.num_non_exiting_ctas,
        workspace.total_num_padded_tokens, workspace.cta_idx_xy_to_batch_idx, workspace.cta_idx_xy_to_mn_limit, stream);

    // We do not fuse activation with FC1 for DeepSeek FP8 due to the weights shuffling constraint.
    void* gemm2_input = workspace.gemm1_output;
    void* gemm2_input_scale = workspace.gemm1_output_scale;
    if (args.mDtypeElt == tg::Dtype::E4m3)
    {
        // Run activation
        moe::dev::activation::run(activationData, stream);
        gemm2_input = workspace.activation_output;
        gemm2_input_scale = workspace.activation_output_scale;
    }

    // Run gemm2
    Gemm2::Runner gemm2(args.mDtypeElt, tg::Dtype::Bfloat16);
    gemm2.run(gemm2_input, gemm2_input_scale, args.gemm2_weights, args.gemm2_weights_scale, args.output2_scales_scalar,
        workspace.gemm2_output, workspace.gemm2_output_scale, args.top_k, args.hidden_size, args.intermediate_size,
        args.local_num_experts, args.num_tokens, workspace.num_non_exiting_ctas, workspace.total_num_padded_tokens,
        workspace.cta_idx_xy_to_batch_idx, workspace.cta_idx_xy_to_mn_limit, stream);

    // Run finalize
    moe::dev::finalize::run(finalizeData, stream);
}
} // namespace MoE

} // namespace trtllmGenFp8BlockScaleMoe
} // namespace kernels
} // namespace tensorrt_llm
