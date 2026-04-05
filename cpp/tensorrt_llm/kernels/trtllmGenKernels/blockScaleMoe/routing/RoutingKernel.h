/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include "IntFastDiv.h"
#include "tensorrt_llm/kernels/trtllmGenKernels/batchedGemm/trtllmGen_bmm_export/trtllm/gen/DtypeDecl.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <tensorrt_llm/common/cudaUtils.h>

namespace moe::dev
{

namespace routing
{

namespace tg = batchedGemm::trtllm::gen;

template <typename DataType>
struct PackedScoreIdx
{
    DataType score;
    int16_t idx;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct DataBase
{
    bool mUsePdl{false};

    // optional: only used as an intermediate buffer when the number of tokens is large.
    // dim: max([2*NumThreads] = [512], mNumExperts*2)
    int32_t* mPtrExpertCounts{nullptr};
    // optional: if `nullptr`, it is not filled
    // dim: [1]
    int32_t* mPtrPermutedIdxSize{nullptr};
    // optional: if `nullptr`, it is not filled
    // dim: [mNumTokens * mTopK]
    int32_t* mPtrExpandedIdxToPermutedIdx{nullptr};
    // optional: if `nullptr`, it is not filled
    // dim: [mTileTokensDim * mTopK + (mNumExperts × mTileTokensDim) - mNumExperts]
    int32_t* mPtrPermutedIdxToExpandedIdx{nullptr};
    // optional: if `nullptr`, it is not filled
    // dim: [mTileTokensDim * mTopK + (mNumExperts × mTileTokensDim) - mNumExperts]
    // Note: this array (mPtrPermutedIdxToTokenIdx) is uninitialized
    // Any out-of-bounds values are undefined.
    int32_t* mPtrPermutedIdxToTokenIdx{nullptr};

    // optional: if `nullptr`, it is not filled
    // dim: [mNumTokens, mTopK]
    // When mPtrTopKIds is provided, mPtrTopKWeights must be also provided as inputs.
    // Otherwise, mPtrTopKWeights is the output scores of the topK experts.
    void* mPtrTopKWeights{nullptr};
    // optional: if `nullptr`, it is not filled
    // dim: [mNumTokens, mTopK]
    // mPtrTopKIds[i] is the index of the expert for the i-th token in the top-k experts
    // Together with mPtrTopKWeights, they form the top-k experts for each token
    int32_t* mPtrTopKIds{nullptr};

    // optional: if `nullptr`, scores are used directly as input.
    // If it is given, it must represent a packed value s.t. the most significant
    // 16/32 bits represent the score without sigmoid activation and
    // the least significant 16 bits represent the index of the chosen expert (unsigned).
    // note: this is required if the number of tokens is large.
    // dim: [mNumTokens, mTopK]
    void* mPtrTopKPacked{nullptr};
    // optional: if `nullptr`, `mPtrTopKPacked` must be provided.
    // If it is given, it represents the scores without sigmoid activation for
    // each token and expert.
    // note: if it is provided, we always re-compute the top1 scores
    // dim: [mNumTokens, mNumExperts]
    void const* mPtrScores{nullptr};

    //
    // Grouped Gemm Launch Config Buffers
    //
    int32_t* mPtrCtaIdxXyToBatchIdx{nullptr};
    int32_t* mPtrCtaIdxXyToMnLimit{nullptr};
    int32_t* mPtrNumNonExitingCtas{nullptr};

    //
    // Metadata
    //
    int32_t mNumTokens;
    int32_t mNumExperts;
    int32_t mTopK;
    // Cluster-wide tile size in token dimension.
    int32_t mTileTokensDim;
    // log2() of the padding size in cluster-wide tile.
    int32_t mPaddingLog2;

    /// For expert parallelization
    int32_t mLocalExpertsStartIdx;
    int32_t mLocalExpertsStrideLog2;
    int32_t mNumLocalExperts;
};

template <typename InputT_, typename OutputT_, int MaxNumExperts_, int MaxNumTopExperts_>
struct KernelParamsBase
{
    using InputT = InputT_;
    using OutputT = OutputT_;
    static constexpr int MaxNumExperts = MaxNumExperts_;
    static constexpr int MaxNumTopExperts = MaxNumTopExperts_;

    bool mUsePdl = false;
    bool mIsPow2 = false;

    // Public pointer members
    int32_t* mPtrExpertCounts = nullptr;
    int32_t* mPtrPermutedIdxSize = nullptr;
    int32_t* mPtrExpandedIdxToPermutedIdx = nullptr;
    int32_t* mPtrPermutedIdxToExpandedIdx = nullptr;
    int32_t* mPtrPermutedIdxToTokenIdx = nullptr;
    int32_t* mPtrCtaIdxXyToBatchIdx = nullptr;
    int32_t* mPtrCtaIdxXyToMnLimit = nullptr;
    int32_t* mPtrNumNonExitingCtas = nullptr;
    OutputT* mPtrTopKWeights = nullptr;
    int32_t* mPtrTopKIds = nullptr;
    InputT const* mPtrScores = nullptr;

    // Public scalar members
    int32_t mNumTokens = 0;
    int32_t mNumExperts = 0;

    int32_t mPaddingLog2 = -1;
    int32_t mTileTokensDim = 0;
    int32_t mLocalExpertsStartIdx = 0;
    int32_t mLocalExpertsStrideLog2 = 0;
    int32_t mNumLocalExperts = 0;

    // Public initialization function - make it a template to accept different Data types
    template <typename DataType>
    void setBaseParams(DataType const& data)
    {
        mUsePdl = data.mUsePdl;
        mIsPow2 = data.mPaddingLog2 > 0;
        mPtrExpertCounts = data.mPtrExpertCounts;
        mPtrPermutedIdxSize = data.mPtrPermutedIdxSize;
        mPtrExpandedIdxToPermutedIdx = data.mPtrExpandedIdxToPermutedIdx;
        mPtrPermutedIdxToExpandedIdx = data.mPtrPermutedIdxToExpandedIdx;
        mPtrPermutedIdxToTokenIdx = data.mPtrPermutedIdxToTokenIdx;
        mPtrCtaIdxXyToBatchIdx = data.mPtrCtaIdxXyToBatchIdx;
        mPtrCtaIdxXyToMnLimit = data.mPtrCtaIdxXyToMnLimit;
        mPtrNumNonExitingCtas = data.mPtrNumNonExitingCtas;
        mPtrTopKWeights = static_cast<OutputT*>(data.mPtrTopKWeights);
        mPtrTopKIds = static_cast<int32_t*>(data.mPtrTopKIds);
        mPtrScores = (InputT const*) data.mPtrScores;

        mNumTokens = data.mNumTokens;
        mNumExperts = data.mNumExperts;

        mPaddingLog2 = data.mPaddingLog2;
        mTileTokensDim = data.mTileTokensDim;
        mLocalExpertsStartIdx = data.mLocalExpertsStartIdx;
        mLocalExpertsStrideLog2 = data.mLocalExpertsStrideLog2;
        mNumLocalExperts = data.mNumLocalExperts;
    }
};

namespace routingDeepSeek
{

////////////////////////////////////////////////////////////////////////////////////////////////////
struct Data : public DataBase
{
    tg::Dtype mDtypeOutput{tg::Dtype::Bfloat16};

    //
    // Grouped Gemm Launch Config Buffers
    //
    void const* mPtrRoutingBias;
    // Dtype of the routing bias buffer (Bfloat16 or Fp32).
    tg::Dtype mDtypeBias{tg::Dtype::Bfloat16};

    int32_t mHiddenDim; // not used
    int32_t mNumExpertGroups;
    int32_t mNumLimitedGroups;

    float mRouteScale;
    bool mUseRoutingSoftmax;
};

template <typename InputT_, typename OutputT_, int MaxNumExperts_, int MaxNumTopExperts_, bool UseGroups_>
struct KernelParams : public KernelParamsBase<InputT_, OutputT_, MaxNumExperts_, MaxNumTopExperts_>
{
    using InputT = InputT_;
    using OutputT = OutputT_;

    static constexpr bool UseGroups = UseGroups_;

    PackedScoreIdx<OutputT>* mPtrTopKPacked = nullptr;

    // OutputT* mPtrTopKWeightsFull = nullptr;
    // Note: this variable(mPtrTopKWeightsFull) might need to be added back for the low-latency kernels for MoE in
    // tllm-gen in the future

    // Type-erased bias pointer — supports both float and bfloat16 without conversion.
    void const* mPtrRoutingBias = nullptr;
    tg::Dtype mDtypeBias = tg::Dtype::Bfloat16;

    int32_t mNumExpertGroups = 0;
    int32_t mNumExpertsPerGroup = 0;
    int32_t mNumLimitedGroups = 0;

    trtllm::dev::IntFastDiv mTopK;
    float mRouteScale = 0.f;

    static KernelParams setKernelParams(Data const& data)
    {
        KernelParams params;
        params.setBaseParams(data);

        params.mPtrTopKPacked = (PackedScoreIdx<OutputT>*) data.mPtrTopKPacked;

        params.mPtrRoutingBias = data.mPtrRoutingBias;
        params.mDtypeBias = data.mDtypeBias;

        params.mNumExpertGroups = data.mNumExpertGroups;
        params.mNumExpertsPerGroup = data.mNumExperts / data.mNumExpertGroups;
        params.mNumLimitedGroups = data.mNumLimitedGroups;
        params.mTopK = trtllm::dev::IntFastDiv(data.mTopK);
        params.mRouteScale = data.mRouteScale;

        return params;
    }
};

void run(Data& data, void* stream);

} // namespace routingDeepSeek

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace routingLlama4
{

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Data : public DataBase
{
    tg::Dtype mDtypeOutput{tg::Dtype::Bfloat16};
};

template <typename InputT_, typename OutputT_, int MaxNumExperts_, int MaxNumTopExperts_>
struct KernelParams : public KernelParamsBase<InputT_, OutputT_, MaxNumExperts_, MaxNumTopExperts_>
{
    using InputT = InputT_;
    using OutputT = OutputT_;

    PackedScoreIdx<OutputT>* mPtrTopKPacked = nullptr;

    int32_t mTopK;

    static KernelParams setKernelParams(Data const& data)
    {
        KernelParams params;
        params.setBaseParams(data);

        params.mPtrTopKPacked = (PackedScoreIdx<OutputT>*) data.mPtrTopKPacked;
        params.mTopK = data.mTopK;
        return params;
    }
};

void run(Data const& data, void* stream);

} // namespace routingLlama4

////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
// Routing preprocess/postprocess policy type enums.
// These are used to select the compile-time policy at dispatch time.
////////////////////////////////////////////////////////////////////////////////////////////////////

enum class RoutingPreprocessType
{
    None,        // No preprocessing before topK
    Softmax,     // Apply softmax on all expert scores before topK
    Sigmoid,     // Apply sigmoid(score) for topK selection (Cohere-style, no bias)
    SigmoidBias, // Apply sigmoid(score) + bias for topK selection (DeepSeek-style)
};

enum class RoutingPostprocessType
{
    None,               // No postprocessing after topK
    Softmax,            // Apply softmax on top-K scores
    SumNormalize,       // Normalize top-K scores by their sum
    ScaledSumNormalize, // Recover sigmoid scores, normalize by sum and scale (DeepSeek-style)
};

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace routingCustom
{

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Data : public DataBase
{
    tg::Dtype mDtypeOutput{tg::Dtype::Fp32};    // OutputT: expert weights dtype (typically Bfloat16)
    tg::Dtype mDtypeInput{tg::Dtype::Bfloat16}; // InputT: routing logits dtype (Bfloat16 or Fp32)

    RoutingPreprocessType mPreprocessType{RoutingPreprocessType::None};
    RoutingPostprocessType mPostprocessType{RoutingPostprocessType::Softmax};
    bool mNormTopkProb{true}; // Default value is true for Qwen3 model

    // Optional: per-expert routing bias (used by SigmoidBias preprocess).
    void const* mPtrRoutingBias{nullptr};
    // Dtype of the routing bias buffer (Bfloat16 or Fp32). Used to read mPtrRoutingBias correctly.
    tg::Dtype mDtypeBias{tg::Dtype::Bfloat16};
    // Optional: scaling factor applied to final scores (used by ScaledSumNormalize postprocess).
    float mRouteScale{1.0f};
    // Optional: epsilon added to the sum before division to prevent division by zero.
    // MiniMax2 uses 1e-20f; DeepSeek uses 0.0f (no epsilon).
    float mSumEpsilon{0.0f};
};

template <typename InputT_, typename OutputT_, int MaxNumExperts_, int MaxNumTopExperts_, typename ExpertSelectPolicy_>
struct KernelParams : public KernelParamsBase<InputT_, OutputT_, MaxNumExperts_, MaxNumTopExperts_>
{
    using InputT = InputT_;
    using OutputT = OutputT_;
    using ExpertSelectPolicy = ExpertSelectPolicy_;

    // Expert select policy params — empty structs have zero register cost.
    using ExpertSelectParams = typename ExpertSelectPolicy::template Params<OutputT>;

    PackedScoreIdx<OutputT>* mPtrTopKPacked = nullptr;

    int32_t mTopK = 0;

    ExpertSelectParams mExpertSelectParams;

    static KernelParams setKernelParams(Data const& data)
    {
        KernelParams params;
        params.setBaseParams(data);

        params.mPtrTopKPacked = (PackedScoreIdx<OutputT>*) data.mPtrTopKPacked;
        params.mTopK = data.mTopK;

        // Policy populates only the fields it needs from Data.
        params.mExpertSelectParams.set(data);
        return params;
    }
};

void run(Data const& data, void* stream);

} // namespace routingCustom

////////////////////////////////////////////////////////////////////////////////////////////////////
// Shared utility for post-topK pipeline when mPtrTopKIds != nullptr.
// All routing methods (Custom, DeepSeek, Llama4) use the same workflow in this case:
// 1. Reset expert counts
// 2. Run histogram kernel
// 3. Run offsets kernel
// Since the kernels are shared and we don't need routing-method-specific logic,
// we can use routingCustom's launch mechanism.
//
// This function works with any Data type that inherits from DataBase.
// Implementation is in RoutingFromTopKIds.cu
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename DataType>
void runPostTopKPipeline(DataType const& data, uint32_t numThreadsHist, void* stream);

////////////////////////////////////////////////////////////////////////////////////////////////////
} // namespace routing
} // namespace moe::dev
