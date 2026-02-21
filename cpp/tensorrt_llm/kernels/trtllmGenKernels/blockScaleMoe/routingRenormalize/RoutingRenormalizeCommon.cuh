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

#include "../RoutingKernel.cuh"

namespace moe::dev::routing
{
namespace routingRenormalize
{
////////////////////////////////////////////////////////////////////////////////////////////////////

static constexpr int NumExperts128Experts = 128;
static constexpr int NumExperts512Experts = 512;
static constexpr int MaxSupportedExperts = 2048;

static constexpr int NumTop8Experts = 8;
static constexpr int NumTop16Experts = 16;
static constexpr int MaxSupportedTopExperts = 32;

static constexpr int NumThreads = 1024;
static constexpr int NumWarps = NumThreads / WarpSize;

static constexpr int MaxNumTokensSingleCluster = NumBlocksPerCluster * NumThreads;
static constexpr int MaxNumTokensSingleClusterScores = NumBlocksPerCluster * NumWarps;

static constexpr int BlockKernelMaxNumTokens = 4;

template <typename DataType, typename InputType, int VecSize, int K, bool DoSoftmaxBeforeTopK>
__forceinline__ __device__ void routingTopKExperts(cg::thread_block_tile<WarpSize> const& warp,
    DataType (&score)[VecSize], int32_t (&idx)[VecSize], DataType (&warpTopKScore)[K], int32_t (&warpTopKExpertIdx)[K],
    int32_t const laneIdx, int32_t const numExperts, int32_t topK, InputType const* ptrScores, bool const normTopkProb,
    bool const applySoftmaxAfterTopK = true)
{
    DataType minScore = DataType{-INFINITY};

    for (int i = 0; i < VecSize; i++)
    {
        auto expertIdx = i * WarpSize + laneIdx;
        auto newScore = expertIdx < numExperts ? static_cast<DataType>(ptrScores[expertIdx]) : minScore;
        score[i] = newScore;
        idx[i] = expertIdx;
    }
    if constexpr (DoSoftmaxBeforeTopK)
    {
        calcSoftmax(warp, score);
    }

    // Get the top-k scores and their corresponding expert indices
    topk::reduceTopK(warp, warpTopKScore, warpTopKExpertIdx, score, idx, minScore, topK);

    // Normalize the scores
    if constexpr (DoSoftmaxBeforeTopK)
    {
        float sum = float{1.f};
        if (normTopkProb)
        {
            sum = static_cast<float>(laneIdx < topK ? warpTopKScore[laneIdx] : 0);
            sum = cg::reduce(warp, sum, cg::plus<float>());
        }
        if (laneIdx < topK)
        {
            warpTopKScore[laneIdx] = warpTopKScore[laneIdx] / sum;
        }
    }
    else
    {
        if (applySoftmaxAfterTopK)
        {
            auto softmaxScore = calcSoftmax(warp, laneIdx < topK ? warpTopKScore[laneIdx] : minScore, laneIdx, topK);
            if (laneIdx < topK)
            {
                warpTopKScore[laneIdx] = softmaxScore;
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int32_t constexpr getMaxNumExperts(int32_t numExperts)
{
    if (numExperts <= NumExperts128Experts)
    {
        return NumExperts128Experts;
    }
    else if (numExperts <= NumExperts512Experts)
    {
        return NumExperts512Experts;
    }
    else if (numExperts <= MaxSupportedExperts)
    {
        return MaxSupportedExperts;
    }
    else
    {
        TLLM_LOG_ERROR("Unsupported numExperts");
        return 0;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper macro: dispatch on topK tier for a given numExperts tier.
#define LAUNCH_ROUTING_WITH_TOPK(                                                                                      \
    data, coopLaunch, kernel, numBlocks, numThreads, smemSize, stream, extraFlag1, numExperts)                         \
    if (data.mTopK <= NumTop8Experts)                                                                                  \
    {                                                                                                                  \
        LAUNCH_ROUTING_WITH_NUM_EXPERTS(data, coopLaunch, kernel, numBlocks, numThreads, smemSize, stream, extraFlag1, \
            numExperts, NumTop8Experts);                                                                               \
    }                                                                                                                  \
    else if (data.mTopK <= NumTop16Experts)                                                                            \
    {                                                                                                                  \
        LAUNCH_ROUTING_WITH_NUM_EXPERTS(data, coopLaunch, kernel, numBlocks, numThreads, smemSize, stream, extraFlag1, \
            numExperts, NumTop16Experts);                                                                              \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        LAUNCH_ROUTING_WITH_NUM_EXPERTS(data, coopLaunch, kernel, numBlocks, numThreads, smemSize, stream, extraFlag1, \
            numExperts, MaxSupportedTopExperts);                                                                       \
    }

#define LAUNCH_ROUTING_RENORNALIZE(data, coopLaunch, kernel, numBlocks, numThreads, smemSize, stream, extraFlag1)      \
    if (data.mNumExperts <= NumExperts128Experts)                                                                      \
    {                                                                                                                  \
        LAUNCH_ROUTING_WITH_TOPK(                                                                                      \
            data, coopLaunch, kernel, numBlocks, numThreads, smemSize, stream, extraFlag1, NumExperts128Experts);      \
    }                                                                                                                  \
    else if (data.mNumExperts <= NumExperts512Experts)                                                                 \
    {                                                                                                                  \
        LAUNCH_ROUTING_WITH_TOPK(                                                                                      \
            data, coopLaunch, kernel, numBlocks, numThreads, smemSize, stream, extraFlag1, NumExperts512Experts);      \
    }                                                                                                                  \
    else if (data.mNumExperts <= MaxSupportedExperts)                                                                  \
    {                                                                                                                  \
        LAUNCH_ROUTING_WITH_TOPK(                                                                                      \
            data, coopLaunch, kernel, numBlocks, numThreads, smemSize, stream, extraFlag1, MaxSupportedExperts);       \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        TLLM_LOG_ERROR("Unsupported numExperts");                                                                      \
    }

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace routingRenormalize
} // namespace moe::dev::routing
