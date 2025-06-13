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

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace routing
{

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace tg = batchedGemm::trtllm::gen;

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Data
{
    tg::Dtype mDtypeExpW{tg::Dtype::Bfloat16};
    bool mUsePdl{false};

    // note: at least one of the optional outputs below must be provided
    // note: if one of the indexes using "PermutedIdx" is provided,
    // then `mPtrExpertIdx` and `mPtrPermutedIdxSize` must be provided
    // optional: if `nullptr`, it is not filled
    // dim: [mNumTokens, mTopK]
    int32_t* mPtrExpertIdx{nullptr};
    // optional: only used as an intermediate buffer when the number of tokens is large.
    // dim: [2*NumThreads] = [512]
    int32_t* mPtrExpertCounts{nullptr};
    // optional: if `nullptr`, it is not filled
    // dim: [1]
    int32_t* mPtrPermutedIdxSize{nullptr};
    // optional: if `nullptr`, it is not filled
    // dim: [mNumTokens * mTopK]
    int32_t* mPtrExpandedIdxToPermutedIdx{nullptr};
    // optional: if `nullptr`, it is not filled
    // dim: [mNumTokens * mTopK + (mNumExperts << mPaddingLog2) - mNumExperts]
    int32_t* mPtrPermutedIdxToTokenIdx{nullptr};
    // optional: if `nullptr`, it is not filled
    // dim: [mNumLocalExperts * (2 ^ mLocalExpertsStrideLog2), mNumTokens]
    void* mPtrExpertWeightsFull{nullptr};
    // optional: if `nullptr`, it is not filled
    // dim: [mNumTokens, mTopK]
    void* mPtrExpertWeights{nullptr};

    //
    // Grouped Gemm Launch Config Buffers
    //
    int32_t* mPtrCtaIdxXyToBatchIdx{nullptr};
    int32_t* mPtrCtaIdxXyToMnLimit{nullptr};
    int32_t* mPtrNumNonExitingCtas{nullptr};
    // mPtrPermutedIdxSize is ptrTotalNumPaddedTokens
    bool mAllToAllRouteAct{false};

    void const* mPtrRoutingWeights;
    void const* mPtrRoutingBias;
    void const* mPtrIn;
    float* mPtrScores;

    int32_t mNumTokens;
    int32_t mHiddenDim;
    int32_t mNumExperts;
    int32_t mNumExpertGroups;
    int32_t mNumLimitedGroups;
    int32_t mTopK;
    int32_t mPaddingLog2;
    int32_t mLocalExpertsStartIdx;
    int32_t mLocalExpertsStrideLog2;
    int32_t mNumLocalExperts;
    float mRouteScale;
    bool mUseRoutingSoftmax;

    int32_t* mPtrNumTokensPerExpert{nullptr};
    int32_t* mPtrPermutedIdxToExpandedIdx{nullptr};
};

template <typename TypeExpW_, bool UseGroups_, bool UsePdl_>
struct KernelParams
{
    using TypeExpW = TypeExpW_;
    static constexpr bool UseGroups = UseGroups_;
    static constexpr bool UsePdl = UsePdl_;

    int32_t* mPtrExpertIdx;
    int32_t* mPtrExpertCounts;
    int32_t* mPtrPermutedIdxSize;
    int32_t* mPtrExpandedIdxToPermutedIdx;
    int32_t* mPtrPermutedIdxToTokenIdx;
    int32_t* mPtrPermutedIdxToExpandedIdx;
    int32_t* mPtrNumTokensPerExpert;

    int32_t* mPtrCtaIdxXyToBatchIdx;
    int32_t* mPtrCtaIdxXyToMnLimit;
    int32_t* mPtrNumNonExitingCtas;

    TypeExpW* mPtrExpertWeightsFull;
    TypeExpW* mPtrExpertWeights;
    TypeExpW const* mPtrRoutingWeights;
    TypeExpW const* mPtrRoutingBias;
    float const* mPtrScores;

    int32_t mHiddenDim;
    int32_t mNumExperts;
    int32_t mNumExpertGroups;
    int32_t mNumExpertsPerGroup;
    int32_t mNumLimitedGroups;
    trtllm::dev::IntFastDiv mTopK;
    int32_t mPaddingLog2;
    int32_t mLocalExpertsStartIdx;
    int32_t mLocalExpertsStrideLog2;
    int32_t mNumLocalExperts;
    int32_t mNumTokens;
    float mRouteScale;
    bool mAllToAllRouteAct;

    static KernelParams setKernelParams(Data const& data)
    {
        KernelParams params;

        params.mPtrExpertIdx = data.mPtrExpertIdx;
        params.mPtrExpertCounts = data.mPtrExpertCounts;
        params.mPtrPermutedIdxSize = data.mPtrPermutedIdxSize;
        params.mPtrExpandedIdxToPermutedIdx = data.mPtrExpandedIdxToPermutedIdx;
        params.mPtrPermutedIdxToTokenIdx = data.mPtrPermutedIdxToTokenIdx;
        params.mPtrPermutedIdxToExpandedIdx = data.mPtrPermutedIdxToExpandedIdx;
        params.mPtrNumTokensPerExpert = data.mPtrNumTokensPerExpert;

        params.mPtrCtaIdxXyToBatchIdx = data.mPtrCtaIdxXyToBatchIdx;
        params.mPtrCtaIdxXyToMnLimit = data.mPtrCtaIdxXyToMnLimit;
        params.mPtrNumNonExitingCtas = data.mPtrNumNonExitingCtas;

        params.mPtrExpertWeightsFull = (TypeExpW*) data.mPtrExpertWeightsFull;
        params.mPtrExpertWeights = (TypeExpW*) data.mPtrExpertWeights;
        params.mPtrRoutingWeights = (TypeExpW*) data.mPtrRoutingWeights;
        params.mPtrRoutingBias = (TypeExpW*) data.mPtrRoutingBias;
        params.mPtrScores = data.mPtrScores;

        params.mHiddenDim = data.mHiddenDim;
        params.mNumExperts = data.mNumExperts;
        params.mNumExpertGroups = data.mNumExpertGroups;
        params.mNumExpertsPerGroup = data.mNumExperts / data.mNumExpertGroups;
        params.mNumLimitedGroups = data.mNumLimitedGroups;
        params.mTopK = trtllm::dev::IntFastDiv(data.mTopK);
        params.mPaddingLog2 = data.mPaddingLog2;
        params.mLocalExpertsStartIdx = data.mLocalExpertsStartIdx;
        params.mLocalExpertsStrideLog2 = data.mLocalExpertsStrideLog2;
        params.mNumLocalExperts = data.mNumLocalExperts;
        params.mNumTokens = data.mNumTokens;
        params.mRouteScale = data.mRouteScale;
        params.mAllToAllRouteAct = data.mAllToAllRouteAct;
        return params;
    }
};

void run(Data const& data, void* stream);

} // namespace routing

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace routingLlama4
{

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace tg = batchedGemm::trtllm::gen;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename TypeExpW>
struct PackedScoreIdx
{
    TypeExpW score;
    int16_t idx;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Data
{
    tg::Dtype mDtypeExpW{tg::Dtype::Bfloat16};
    bool mUsePdl{false};

    // optional: if `nullptr`, `mPtrExpertIdx` must be provided.
    // If it is given, it represents the scores without sigmoid activation for
    // each token and expert.
    // note: if it is provided, we always re-compute the top1 scores
    // dim: [mNumTokens, mNumExperts]
    void const* mPtrScores{nullptr};
    // optional: if `nullptr`, scores are used directly as input.
    // If it is given, it must represent a packed value s.t. the most significant
    // 16/32 bits represent the score without sigmoid activation and
    // the least significant 16 bits represent the index of the chosen expert (unsigned).
    // note: this is required if the number of tokens is large.
    // dim: [mNumTokens, mTopK]
    void* mPtrExpertIdx{nullptr};

    // note: at least one of the optional outputs below must be provided
    // optional: only used as an intermediate buffer when the number of tokens is large.
    // dim: [2, mNumExperts]
    int32_t* mPtrExpertCounts{nullptr};
    // dim: [1]
    int32_t* mPtrPermutedIdxSize{nullptr};
    // optional: if `nullptr`, it is not filled
    // dim: [mNumTokens * mTopK]
    int32_t* mPtrExpandedIdxToPermutedIdx{nullptr};
    // optional: if `nullptr`, it is not filled
    // dim: [mNumTokens * mTopK + (mNumExperts << mPaddingLog2) - mNumExperts]
    int32_t* mPtrPermutedIdxToTokenIdx{nullptr};
    // optional: if `nullptr`, it is not filled
    // dim: [mNumTokens, mTopK]
    void* mPtrExpertWeights{nullptr};
    //
    // Grouped Gemm Launch Config Buffers
    //
    int32_t* mPtrCtaIdxXyToBatchIdx{nullptr};
    int32_t* mPtrCtaIdxXyToMnLimit{nullptr};
    int32_t* mPtrNumNonExitingCtas{nullptr};

    int32_t mNumTokens;
    int32_t mNumExperts;
    int32_t mTopK;
    int32_t mPaddingLog2;
    int32_t mLocalExpertsStartIdx;
    int32_t mLocalExpertsStrideLog2;
    int32_t mNumLocalExperts;
};

template <typename TypeExpW_, bool UsePdl_>
struct KernelParams
{
    using TypeExpW = TypeExpW_;
    static constexpr bool UsePdl = UsePdl_;

    PackedScoreIdx<TypeExpW>* mPtrExpertIdx;
    TypeExpW const* mPtrScores;
    int32_t* mPtrExpertCounts;
    int32_t* mPtrPermutedIdxSize;
    int32_t* mPtrExpandedIdxToPermutedIdx;
    int32_t* mPtrPermutedIdxToTokenIdx;
    int32_t* mPtrCtaIdxXyToBatchIdx;
    int32_t* mPtrCtaIdxXyToMnLimit;
    int32_t* mPtrNumNonExitingCtas;
    TypeExpW* mPtrExpertWeights;

    int32_t mNumTokens;
    int32_t mNumExperts;
    int32_t mPaddingLog2;
    int32_t mLocalExpertsStartIdx;
    int32_t mLocalExpertsStrideLog2;
    int32_t mNumLocalExperts;

    static KernelParams setKernelParams(Data const& data)
    {
        KernelParams params;

        params.mPtrExpertIdx = (PackedScoreIdx<TypeExpW>*) data.mPtrExpertIdx;
        params.mPtrScores = (TypeExpW const*) data.mPtrScores;
        params.mPtrExpertCounts = data.mPtrExpertCounts;
        params.mPtrPermutedIdxSize = data.mPtrPermutedIdxSize;
        params.mPtrExpandedIdxToPermutedIdx = data.mPtrExpandedIdxToPermutedIdx;
        params.mPtrPermutedIdxToTokenIdx = data.mPtrPermutedIdxToTokenIdx;
        params.mPtrCtaIdxXyToBatchIdx = data.mPtrCtaIdxXyToBatchIdx;
        params.mPtrCtaIdxXyToMnLimit = data.mPtrCtaIdxXyToMnLimit;
        params.mPtrNumNonExitingCtas = data.mPtrNumNonExitingCtas;
        params.mPtrExpertWeights = (TypeExpW*) data.mPtrExpertWeights;

        params.mNumTokens = data.mNumTokens;
        params.mNumExperts = data.mNumExperts;
        params.mPaddingLog2 = data.mPaddingLog2;
        params.mLocalExpertsStartIdx = data.mLocalExpertsStartIdx;
        params.mLocalExpertsStrideLog2 = data.mLocalExpertsStrideLog2;
        params.mNumLocalExperts = data.mNumLocalExperts;

        return params;
    }
};

void run(Data const& data, void* stream);

} // namespace routingLlama4

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace routingQwen3
{

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace tg = batchedGemm::trtllm::gen;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename TypeExpW>
struct PackedScoreIdx
{
    TypeExpW score;
    int16_t idx; // @TODO: Might use int8_t as the number of experts is 128
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Data
{
    tg::Dtype mDtypeExpW{tg::Dtype::Fp32};
    tg::Dtype mDtypeElt{tg::Dtype::Bfloat16};
    bool mUsePdl{false};
    bool mDoSoftmaxBeforeTopK{false};
    bool mNormTopkProb{true}; // Default value is true for Qwen3 model
    // optional: if `nullptr`, `mPtrExpertIdx` must be provided.
    // If it is given, it represents the scores without sigmoid activation for
    // each token and expert.
    // note: if it is provided, we always re-compute the top1 scores
    // dim: [mNumTokens, mNumExperts]
    void const* mPtrScores{nullptr};
    // optional: if `nullptr`, scores are used directly as input.
    // If it is given, it must represent a packed value s.t. the most significant
    // 16/32 bits represent the score without sigmoid activation and
    // the least significant 16 bits represent the index of the chosen expert (unsigned).
    // note: this is required if the number of tokens is large.
    // dim: [mNumTokens, mTopK]
    void* mPtrExpertIdx{nullptr};

    // note: at least one of the optional outputs below must be provided
    // optional: only used as an intermediate buffer when the number of tokens is large.
    // dim: [2, mNumExperts]
    int32_t* mPtrExpertCounts{nullptr};
    // dim: [1]
    int32_t* mPtrPermutedIdxSize{nullptr};
    // optional: if `nullptr`, it is not filled
    // dim: [mNumTokens * mTopK]
    int32_t* mPtrExpandedIdxToPermutedIdx{nullptr};
    // optional: if `nullptr`, it is not filled
    // dim: [mNumTokens * mTopK + (mNumExperts << mPaddingLog2) - mNumExperts]
    int32_t* mPtrPermutedIdxToTokenIdx{nullptr};
    // optional: if `nullptr`, it is not filled
    // dim: [mNumTokens, mTopK]
    void* mPtrExpertWeights{nullptr};
    //
    // Grouped Gemm Launch Config Buffers
    //
    int32_t* mPtrCtaIdxXyToBatchIdx{nullptr};
    int32_t* mPtrCtaIdxXyToMnLimit{nullptr};
    int32_t* mPtrNumNonExitingCtas{nullptr};

    int32_t mNumTokens;
    int32_t mNumExperts;
    int32_t mTopK;
    int32_t mPaddingLog2;
    int32_t mLocalExpertsStartIdx;
    int32_t mLocalExpertsStrideLog2;
    int32_t mNumLocalExperts;
};

template <typename Type_, typename TypeExpW_, bool UsePdl_>
struct KernelParams
{
    using Type = Type_;
    using TypeExpW = TypeExpW_;
    static constexpr bool UsePdl = UsePdl_;
    bool mNormTopkProb = true;
    PackedScoreIdx<TypeExpW>* mPtrExpertIdx;
    TypeExpW const* mPtrScores;
    int32_t* mPtrExpertCounts;
    int32_t* mPtrPermutedIdxSize;
    int32_t* mPtrExpandedIdxToPermutedIdx;
    int32_t* mPtrPermutedIdxToTokenIdx;
    int32_t* mPtrCtaIdxXyToBatchIdx;
    int32_t* mPtrCtaIdxXyToMnLimit;
    int32_t* mPtrNumNonExitingCtas;
    TypeExpW* mPtrExpertWeights;

    int32_t mNumTokens;
    int32_t mNumExperts;
    int32_t mPaddingLog2;
    int32_t mLocalExpertsStartIdx;
    int32_t mLocalExpertsStrideLog2;
    int32_t mNumLocalExperts;

    static KernelParams setKernelParams(Data const& data)
    {
        KernelParams params;
        params.mNormTopkProb = data.mNormTopkProb;
        params.mPtrExpertIdx = (PackedScoreIdx<TypeExpW>*) data.mPtrExpertIdx;
        params.mPtrScores = (TypeExpW const*) data.mPtrScores;
        params.mPtrExpertCounts = data.mPtrExpertCounts;
        params.mPtrPermutedIdxSize = data.mPtrPermutedIdxSize;
        params.mPtrExpandedIdxToPermutedIdx = data.mPtrExpandedIdxToPermutedIdx;
        params.mPtrPermutedIdxToTokenIdx = data.mPtrPermutedIdxToTokenIdx;
        params.mPtrCtaIdxXyToBatchIdx = data.mPtrCtaIdxXyToBatchIdx;
        params.mPtrCtaIdxXyToMnLimit = data.mPtrCtaIdxXyToMnLimit;
        params.mPtrNumNonExitingCtas = data.mPtrNumNonExitingCtas;
        params.mPtrExpertWeights = (TypeExpW*) data.mPtrExpertWeights;

        params.mNumTokens = data.mNumTokens;
        params.mNumExperts = data.mNumExperts;
        params.mPaddingLog2 = data.mPaddingLog2;
        params.mLocalExpertsStartIdx = data.mLocalExpertsStartIdx;
        params.mLocalExpertsStrideLog2 = data.mLocalExpertsStrideLog2;
        params.mNumLocalExperts = data.mNumLocalExperts;

        return params;
    }
};

void run(Data const& data, void* stream);

} // namespace routingQwen3

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace moe::dev
