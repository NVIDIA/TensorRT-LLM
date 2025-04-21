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

#pragma once

#include "Dtype.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

namespace moe::dev
{

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace routing
{

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace tg = trtllm::gen;

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Data
{
    tg::Dtype mDtypeElt{tg::Dtype::Bfloat16};
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

template <typename Type_, typename TypeExpW_, bool UsePdl_>
struct KernelParams
{
    using Type = Type_;
    using TypeExpW = TypeExpW_;
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
    Type const* mPtrIn;
    float* mPtrScores;

    int32_t mHiddenDim;
    int32_t mNumExperts;
    int32_t mNumExpertGroups;
    int32_t mNumExpertsPerGroup;
    int32_t mNumLimitedGroups;
    int32_t mTopK;
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
        params.mPtrIn = (Type*) data.mPtrIn;
        params.mPtrScores = data.mPtrScores;

        params.mHiddenDim = data.mHiddenDim;
        params.mNumExperts = data.mNumExperts;
        params.mNumExpertGroups = data.mNumExpertGroups;
        params.mNumExpertsPerGroup = data.mNumExperts / data.mNumExpertGroups;
        params.mNumLimitedGroups = data.mNumLimitedGroups;
        params.mTopK = data.mTopK;
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

} // namespace moe::dev
