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

#include "routingDeepSeek/RoutingDeepSeekCommon.cuh"

namespace moe::dev::routing
{
namespace routingDeepSeek
{

////////////////////////////////////////////////////////////////////////////////////////////////////
// Forward declarations for split-compiled launch wrappers.
void launchMainKernel(Data& data, int numBlocks, int numThreadsMain, void* stream);
void launchInitExpertCounts(Data& data, int numThreadsHist, void* stream);
void launchClusterKernel(Data& data, int numThreadsHist, void* stream);
void launchCoopKernel(Data& data, int numBlocksCoop, int numThreadsHist, void* stream);
void launchHistogramKernel(Data& data, int numBlocksHistogram, int numThreadsHist, void* stream);
void launchOffsetsKernel(Data& data, int numBlocksOffsets, int numThreadsHist, void* stream);

////////////////////////////////////////////////////////////////////////////////////////////////////

void run(Data& data, void* stream)
{
    TLLM_CHECK_WITH_INFO(data.mPtrTopKPacked != nullptr || data.mPtrScores != nullptr || data.mPtrTopKIds != nullptr,
        "Routing kernel requires at least one input parameter");
    if (data.mPtrTopKIds != nullptr)
    {
        TLLM_CHECK_WITH_INFO(data.mPtrTopKWeights != nullptr,
            "When mPtrTopKIds is provided, mPtrTopKWeights must also be provided for DeepSeek routing.");
    }
    if (data.mPtrExpandedIdxToPermutedIdx != nullptr || data.mPtrPermutedIdxToExpandedIdx != nullptr
        || data.mPtrPermutedIdxToTokenIdx != nullptr)
        TLLM_CHECK_WITH_INFO(
            (data.mPtrTopKPacked != nullptr || data.mPtrTopKIds != nullptr) && data.mPtrPermutedIdxSize,
            "If permuted index is required, `mPtrTopKPacked` or `mPtrTopKIds` is also required");
    TLLM_CHECK_WITH_INFO(!data.mUseRoutingSoftmax, "Routing with softmax not implemented yet");
    int const numBlocks = data.mNumTokens;
    int const numThreadsHist = getMaxNumExperts(data.mNumExperts);

    bool const useSingleCluster = data.mNumTokens <= 1024;
    if (!useSingleCluster)
    {
        // Reset the global histograms (not used in single-cluster code path).
        // Cover both for the cooperative and two-kernel code paths.
        TLLM_CHECK_WITH_INFO(
            data.mPtrExpertCounts != nullptr, "When #tokens is large, `mPtrExpertCounts` is a required input.");
    }
    else
    {
        data.mPtrExpertCounts = nullptr; // Set it to nullptr for single-cluster code path, as it won't be used
    }

    // Number of blocks we can use in the cooperative kernel
    // The number of blocks must be:
    //   >= ⌈(numTokens * topK) / (MaxExpandedIdxPerThread * NumThreads)⌉
    //   <= numSms, assuming an occupancy of 1 block/SM
    //
    // If too small for the given numTokens, fall back to the less performant two-step method.
    //
    // The upper bound is a strict requirement. The number of blocks should be determined by querying
    // the device properties, or conservatively low.
    static int const smCount = tensorrt_llm::common::getMultiProcessorCount();
    // WAR: Reserve 8 SMs for overlapping kernels.
    int const numBlocksCoop = smCount - 8;

    // Maximum number of tokens supported by the kernel using a cooperative launch.
    int const maxTokensCoop = (numBlocksCoop * numThreadsHist * 64) / data.mTopK;
    if (data.mPtrTopKIds == nullptr)
    {
        TLLM_CHECK_WITH_INFO(data.mNumExperts >= MaxSupportedTopExperts,
            "Routing kernel expects %d to be at most #experts %d", MaxSupportedTopExperts, data.mNumExperts);
        TLLM_CHECK_WITH_INFO(data.mNumExperts <= MaxSupportedExpertCount,
            "Routing kernel expects #experts %d  <= #threads %d", data.mNumExperts, MaxSupportedExpertCount);
        TLLM_CHECK_WITH_INFO(data.mTopK <= MaxSupportedTopExperts, "Routing kernel expects topK experts <= %d, got %d",
            MaxSupportedTopExperts, data.mTopK);

        // Routing needs to be executed - validate routing kernel constraints
        if (data.mNumExpertGroups > 1)
        {
            // Note: Routing-specific constraints (experts per group, topK limits) are checked when routing is actually
            // needed (data.mPtrTopKIds == nullptr)
            TLLM_CHECK_WITH_INFO(data.mNumExpertGroups <= MaxNumGroups,
                "Routing kernel expects #expert groups %d to be <= max groups %d", data.mNumExpertGroups, MaxNumGroups);
            TLLM_CHECK_WITH_INFO(data.mNumExperts % data.mNumExpertGroups == 0,
                "Routing kernel expects #experts %d to be a multiple of #expert groups %d", data.mNumExperts,
                data.mNumExpertGroups);
            TLLM_CHECK_WITH_INFO(data.mNumExperts / data.mNumExpertGroups <= WarpSize,
                "Routing kernel expects #experts per group <= warp size (%d), got %d experts / %d groups = %d experts "
                "per group",
                WarpSize, data.mNumExperts, data.mNumExpertGroups, data.mNumExperts / data.mNumExpertGroups);
            TLLM_CHECK_WITH_INFO(data.mNumLimitedGroups <= MaxNumTopGroups,
                "Routing kernel expects <= %d top groups, got %d", MaxNumTopGroups, data.mNumLimitedGroups);

            TLLM_CHECK_WITH_INFO(data.mNumExpertGroups >= data.mNumLimitedGroups,
                "Routing kernel expects top groups %d to be limited by #expert groups %d", data.mNumLimitedGroups,
                data.mNumExpertGroups);
            TLLM_CHECK_WITH_INFO(data.mNumExperts % 4 == 0, "Routing kernel expects #experts %d to be a multiple of 4.",
                data.mNumExperts);
        }

        int const numThreadsMain = max(data.mNumExpertGroups * WarpSize, getMaxNumExperts(data.mNumExperts));
        launchMainKernel(data, numBlocks, numThreadsMain, stream);
    }
    else
    {
        // Reset the global histograms.
        launchInitExpertCounts(data, numThreadsHist, stream);
    }

    if (data.mPtrPermutedIdxSize != nullptr)
    {
        if (useSingleCluster)
        {
            launchClusterKernel(data, numThreadsHist, stream);
        }
        else if (data.mNumTokens <= maxTokensCoop)
        {
            launchCoopKernel(data, numBlocksCoop, numThreadsHist, stream);
        }
        else
        {
            const int32_t expandedIdxSize = data.mNumTokens * data.mTopK;
            const int32_t histogramEltsPerBlock = 8 * numThreadsHist;
            const int32_t offsetEltsPerBlock = NumEltsPerOffsetTilePerThread * numThreadsHist;

            // Limit grid size (both kernels use a grid-stride loop).
            const int32_t maxNumBlocks = 1024;

            int const numBlocksHistogram
                = std::min((expandedIdxSize + histogramEltsPerBlock - 1) / histogramEltsPerBlock, maxNumBlocks);
            int const numBlocksOffsets
                = std::min((expandedIdxSize + offsetEltsPerBlock - 1) / offsetEltsPerBlock, maxNumBlocks);

            launchHistogramKernel(data, numBlocksHistogram, numThreadsHist, stream);
            launchOffsetsKernel(data, numBlocksOffsets, numThreadsHist, stream);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace routingDeepSeek
} // namespace moe::dev::routing
