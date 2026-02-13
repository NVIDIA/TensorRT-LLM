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
namespace routingDeepSeek
{

////////////////////////////////////////////////////////////////////////////////////////////////////
static constexpr int NumNemotronExperts = 512;
static constexpr int NumKimiK2Experts = 384;
static constexpr int NumDeepseekExperts = 256;
static constexpr int MaxSupportedExpertCount = std::max({NumNemotronExperts, NumKimiK2Experts, NumDeepseekExperts});
static constexpr int NumTopGroupScores = 2;
static constexpr int MaxNumTopGroups = 4;
static constexpr int MaxNumGroups = 8;

static constexpr int NumTop8Experts = 8;
static constexpr int NumTop22Experts = 22;
static constexpr int MaxSupportedTopExperts = 32;

////////////////////////////////////////////////////////////////////////////////////////////////////

int constexpr getMaxNumExperts(int32_t numExperts)
{
    if (numExperts <= topk::MaxNumExpertsUnit)
    {
        return topk::MaxNumExpertsUnit;
    }
    else if (numExperts <= NumDeepseekExperts)
    {
        return NumDeepseekExperts;
    }
    else if (numExperts <= NumKimiK2Experts)
    {
        return NumKimiK2Experts;
    }
    else if (numExperts <= NumNemotronExperts)
    {
        return NumNemotronExperts;
    }
    else
    {
        TLLM_LOG_ERROR("Unsupported numExperts");
        return 0;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Helper macro: dispatch on topK tier for a given numExperts tier.
#define LAUNCH_DEEPSEEK_WITH_TOPK(                                                                                     \
    data, coopLaunch, kernel, numBlocks, numThreads, smemSize, stream, extraFlag1, forceFloatInput, numExperts)        \
    if (data.mTopK <= NumTop8Experts)                                                                                  \
    {                                                                                                                  \
        LAUNCH_ROUTING_WITH_NUM_EXPERTS_FORCE_FLOAT_INPUT(data, coopLaunch, kernel, numBlocks, numThreads, smemSize,   \
            stream, extraFlag1, forceFloatInput, numExperts, NumTop8Experts);                                          \
    }                                                                                                                  \
    else if (data.mTopK <= NumTop22Experts)                                                                            \
    {                                                                                                                  \
        LAUNCH_ROUTING_WITH_NUM_EXPERTS_FORCE_FLOAT_INPUT(data, coopLaunch, kernel, numBlocks, numThreads, smemSize,   \
            stream, extraFlag1, forceFloatInput, numExperts, NumTop22Experts);                                         \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        LAUNCH_ROUTING_WITH_NUM_EXPERTS_FORCE_FLOAT_INPUT(data, coopLaunch, kernel, numBlocks, numThreads, smemSize,   \
            stream, extraFlag1, forceFloatInput, numExperts, MaxSupportedTopExperts);                                  \
    }

#define LAUNCH_ROUTING_DEEPSEEK(                                                                                       \
    data, coopLaunch, kernel, numBlocks, numThreads, smemSize, stream, extraFlag1, forceFloatInput)                    \
    if (data.mNumExperts <= topk::MaxNumExpertsUnit)                                                                   \
    {                                                                                                                  \
        LAUNCH_DEEPSEEK_WITH_TOPK(data, coopLaunch, kernel, numBlocks, numThreads, smemSize, stream, extraFlag1,       \
            forceFloatInput, topk::MaxNumExpertsUnit);                                                                 \
    }                                                                                                                  \
    else if (data.mNumExperts <= NumDeepseekExperts)                                                                   \
    {                                                                                                                  \
        LAUNCH_DEEPSEEK_WITH_TOPK(data, coopLaunch, kernel, numBlocks, numThreads, smemSize, stream, extraFlag1,       \
            forceFloatInput, NumDeepseekExperts);                                                                      \
    }                                                                                                                  \
    else if (data.mNumExperts <= NumKimiK2Experts)                                                                     \
    {                                                                                                                  \
        LAUNCH_DEEPSEEK_WITH_TOPK(data, coopLaunch, kernel, numBlocks, numThreads, smemSize, stream, extraFlag1,       \
            forceFloatInput, NumKimiK2Experts);                                                                        \
    }                                                                                                                  \
    else if (data.mNumExperts <= NumNemotronExperts)                                                                   \
    {                                                                                                                  \
        LAUNCH_DEEPSEEK_WITH_TOPK(data, coopLaunch, kernel, numBlocks, numThreads, smemSize, stream, extraFlag1,       \
            forceFloatInput, NumNemotronExperts);                                                                      \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        TLLM_LOG_ERROR("Unsupported numExperts");                                                                      \
    }

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace routingDeepSeek
} // namespace moe::dev::routing
