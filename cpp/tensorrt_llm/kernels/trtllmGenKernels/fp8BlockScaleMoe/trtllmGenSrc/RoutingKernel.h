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

#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_size.h>
#include <cutlass/numeric_types.h>

#include "Dtype.h"

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
    // dim: [mNumLocalExperts * (2 ^ mLocalExpertsSrideLog2), mNumTokens]
    void* mPtrExpertWeightsFull{nullptr};
    // optional: if `nullptr`, it is not filled
    // dim: [mNumTokens, mTopK]
    void* mPtrExpertWeights{nullptr};
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
    int32_t mLocalExpertsSrideLog2;
    int32_t mNumLocalExperts;
    float mRouteScale;
    bool mUseRoutingSoftmax;

    uint8_t* mPtrNumTokensPerExpert{nullptr};
    int32_t* mPtrPermutedIdxToExpandedIdx{nullptr};
};

template <typename Type_, typename TypeExpW_, bool UsePdl_>
struct KernelParams
{
    using Type = Type_;
    using TypeExpW = TypeExpW_;
    static constexpr bool UsePdl = UsePdl_;

    int32_t* mPtrExpertIdx;
    int32_t* mPtrPermutedIdxSize;
    int32_t* mPtrExpandedIdxToPermutedIdx;
    int32_t* mPtrPermutedIdxToTokenIdx;
    int32_t* mPtrPermutedIdxToExpandedIdx;
    uint8_t* mPtrNumTokensPerExpert;
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
    int32_t mLocalExpertsSrideLog2;
    int32_t mNumLocalExperts;
    float mRouteScale;

    static KernelParams setKernelParams(Data const& data)
    {
        KernelParams params;

        params.mPtrExpertIdx = data.mPtrExpertIdx;
        params.mPtrPermutedIdxSize = data.mPtrPermutedIdxSize;
        params.mPtrExpandedIdxToPermutedIdx = data.mPtrExpandedIdxToPermutedIdx;
        params.mPtrPermutedIdxToTokenIdx = data.mPtrPermutedIdxToTokenIdx;
        params.mPtrPermutedIdxToExpandedIdx = data.mPtrPermutedIdxToExpandedIdx;
        params.mPtrNumTokensPerExpert = data.mPtrNumTokensPerExpert;
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
        params.mLocalExpertsSrideLog2 = data.mLocalExpertsSrideLog2;
        params.mNumLocalExperts = data.mNumLocalExperts;
        params.mRouteScale = data.mRouteScale;

        return params;
    }
};

void run(Data const& data, void* stream);

} // namespace routing

} // namespace moe::dev
