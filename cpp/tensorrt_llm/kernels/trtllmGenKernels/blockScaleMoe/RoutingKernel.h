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
    // dim: [mNumTokens * mTopK + (mNumExperts << mPaddingLog2) - mNumExperts]
    // Note: this array (mPtrPermutedIdxToTokenIdx) is uninitialized
    // Any out-of-bounds values are undefined.
    int32_t* mPtrPermutedIdxToTokenIdx{nullptr};
    // optional: if `nullptr`, it is not filled
    // dim: [mNumTokens, mTopK]
    void* mPtrExpertWeights{nullptr};
    // optional: if `nullptr`, scores are used directly as input.
    // If it is given, it must represent a packed value s.t. the most significant
    // 16/32 bits represent the score without sigmoid activation and
    // the least significant 16 bits represent the index of the chosen expert (unsigned).
    // note: this is required if the number of tokens is large.
    // dim: [mNumTokens, mTopK]
    void* mPtrExpertIdx{nullptr};
    // optional: if `nullptr`, `mPtrExpertIdx` must be provided.
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
    int32_t mPaddingLog2;

    /// For expert parallelization
    int32_t mLocalExpertsStartIdx;
    int32_t mLocalExpertsStrideLog2;
    int32_t mNumLocalExperts;
};

template <typename InputT_, typename OutputT_, bool UsePdl_>
struct KernelParamsBase
{
    using InputT = InputT_;
    using OutputT = OutputT_;
    static constexpr bool UsePdl = UsePdl_;

    // Public pointer members
    int32_t* mPtrExpertCounts = nullptr;
    int32_t* mPtrPermutedIdxSize = nullptr;
    int32_t* mPtrExpandedIdxToPermutedIdx = nullptr;
    int32_t* mPtrPermutedIdxToTokenIdx = nullptr;
    int32_t* mPtrCtaIdxXyToBatchIdx = nullptr;
    int32_t* mPtrCtaIdxXyToMnLimit = nullptr;
    int32_t* mPtrNumNonExitingCtas = nullptr;
    OutputT* mPtrExpertWeights = nullptr;
    InputT const* mPtrScores = nullptr;

    // Public scalar members
    int32_t mNumTokens = 0;
    int32_t mNumExperts = 0;

    int32_t mPaddingLog2 = 0;
    int32_t mLocalExpertsStartIdx = 0;
    int32_t mLocalExpertsStrideLog2 = 0;
    int32_t mNumLocalExperts = 0;

    // Public initialization function - make it a template to accept different Data types
    template <typename DataType>
    void setBaseParams(DataType const& data)
    {
        mPtrExpertCounts = data.mPtrExpertCounts;
        mPtrPermutedIdxSize = data.mPtrPermutedIdxSize;
        mPtrExpandedIdxToPermutedIdx = data.mPtrExpandedIdxToPermutedIdx;
        mPtrPermutedIdxToTokenIdx = data.mPtrPermutedIdxToTokenIdx;
        mPtrCtaIdxXyToBatchIdx = data.mPtrCtaIdxXyToBatchIdx;
        mPtrCtaIdxXyToMnLimit = data.mPtrCtaIdxXyToMnLimit;
        mPtrNumNonExitingCtas = data.mPtrNumNonExitingCtas;
        mPtrExpertWeights = static_cast<OutputT*>(data.mPtrExpertWeights);
        mPtrScores = (InputT const*) data.mPtrScores;

        mNumTokens = data.mNumTokens;
        mNumExperts = data.mNumExperts;

        mPaddingLog2 = data.mPaddingLog2;
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
    tg::Dtype mDtypeExpW{tg::Dtype::Bfloat16};

    //
    // Grouped Gemm Launch Config Buffers
    //
    void const* mPtrRoutingBias;

    int32_t mHiddenDim; // not used
    int32_t mNumExpertGroups;
    int32_t mNumLimitedGroups;

    float mRouteScale;
    bool mUseRoutingSoftmax;
};

template <typename InputT_, typename OutputT_, bool UseGroups_, bool UsePdl_>
struct KernelParams : public KernelParamsBase<InputT_, OutputT_, UsePdl_>
{
    using InputT = InputT_;
    using OutputT = OutputT_;

    static constexpr bool UseGroups = UseGroups_;

    PackedScoreIdx<OutputT>* mPtrExpertIdx = nullptr;

    // OutputT* mPtrExpertWeightsFull = nullptr;
    // Note: this variable(mPtrExpertWeightsFull) might need to be added back for the low-latency kernels for MoE in
    // tllm-gen in the future

    OutputT const* mPtrRoutingBias = nullptr;

    int32_t mNumExpertGroups = 0;
    int32_t mNumExpertsPerGroup = 0;
    int32_t mNumLimitedGroups = 0;

    trtllm::dev::IntFastDiv mTopK;
    float mRouteScale = 0.f;

    static KernelParams setKernelParams(Data const& data)
    {
        KernelParams params;
        params.setBaseParams(data);

        params.mPtrExpertIdx = (PackedScoreIdx<OutputT>*) data.mPtrExpertIdx;

        // params.mPtrExpertWeightsFull = static_cast<OutputT*>(data.mPtrExpertWeightsFull);
        params.mPtrRoutingBias = static_cast<OutputT const*>(data.mPtrRoutingBias);

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
    tg::Dtype mDtypeExpW{tg::Dtype::Bfloat16};
};

template <typename InputT_, typename OutputT_, bool UsePdl_>
struct KernelParams : public KernelParamsBase<InputT_, OutputT_, UsePdl_>
{
    using InputT = InputT_;
    using OutputT = OutputT_;

    PackedScoreIdx<OutputT>* mPtrExpertIdx = nullptr;

    int32_t mTopK;

    static KernelParams setKernelParams(Data const& data)
    {
        KernelParams params;
        params.setBaseParams(data);

        params.mPtrExpertIdx = (PackedScoreIdx<OutputT>*) data.mPtrExpertIdx;
        params.mTopK = data.mTopK;
        return params;
    }
};

void run(Data const& data, void* stream);

} // namespace routingLlama4

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace routingRenormalize
{

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Data : public DataBase
{
    tg::Dtype mDtypeExpW{tg::Dtype::Fp32};
    tg::Dtype mDtypeElt{tg::Dtype::Bfloat16};

    bool mDoSoftmaxBeforeTopK{false};
    bool mNormTopkProb{true}; // Default value is true for Qwen3 model
};

template <typename InputT_, typename OutputT_, bool DoSoftmaxBeforeTopK_, bool UsePdl_>
struct KernelParams : public KernelParamsBase<InputT_, OutputT_, UsePdl_>
{
    using InputT = InputT_;
    using OutputT = OutputT_;

    static constexpr bool DoSoftmaxBeforeTopK = DoSoftmaxBeforeTopK_;

    PackedScoreIdx<OutputT>* mPtrExpertIdx = nullptr;

    int32_t mTopK = 0;

    bool mNormTopkProb = true;

    static KernelParams setKernelParams(Data const& data)
    {
        KernelParams params;
        params.setBaseParams(data);

        params.mPtrExpertIdx = (PackedScoreIdx<OutputT>*) data.mPtrExpertIdx;
        params.mNormTopkProb = data.mNormTopkProb;
        params.mTopK = data.mTopK;
        return params;
    }
};

void run(Data const& data, void* stream);

} // namespace routingRenormalize

////////////////////////////////////////////////////////////////////////////////////////////////////
} // namespace routing
} // namespace moe::dev
