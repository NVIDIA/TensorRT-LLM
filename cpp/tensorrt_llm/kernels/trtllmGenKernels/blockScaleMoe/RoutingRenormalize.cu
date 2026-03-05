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
#include "routingRenormalize/RoutingRenormalizeCommon.cuh"

namespace moe::dev::routing
{
namespace routingRenormalize
{

////////////////////////////////////////////////////////////////////////////////////////////////////
// Forward declarations of per-kernel launch wrappers (defined in routingRenormalize/*.cu).
void launchBlockKernel(Data const& data, uint32_t numThreadsHist, void* stream);
void launchClusterKernel(Data const& data, void* stream);
void launchHistogramScoresKernel(Data const& data, uint32_t maxNumBlocks, uint32_t numThreadsHist, void* stream);
void launchInitExpertCounts(Data const& data, uint32_t numThreadsHist, void* stream);
void launchHistogramKernel(Data const& data, int numBlocksHistogram, uint32_t numThreadsHist, void* stream);
void launchOffsetsKernel(Data const& data, int numBlocksOffsets, uint32_t numThreadsHist, void* stream);

////////////////////////////////////////////////////////////////////////////////////////////////////
void run(Data const& data, void* stream)
{
    TLLM_CHECK_WITH_INFO(data.mPtrTopKPacked != nullptr || data.mPtrScores != nullptr || data.mPtrTopKIds != nullptr,
        "Routing kernel requires at least one input parameter");
    if (data.mPtrTopKIds != nullptr)
    {
        TLLM_CHECK_WITH_INFO(data.mPtrTopKWeights != nullptr,
            "When mPtrTopKIds is provided, mPtrTopKWeights must also be provided for Renormalize routing.");
    }
    TLLM_CHECK_WITH_INFO(data.mPtrPermutedIdxSize != nullptr && data.mPtrCtaIdxXyToBatchIdx != nullptr
            && data.mPtrCtaIdxXyToMnLimit != nullptr && data.mPtrNumNonExitingCtas != nullptr,
        "Llama4 routing kernel expects permuted idx and grouped Gemm launch config buffers");
    TLLM_CHECK_WITH_INFO(data.mTopK <= MaxSupportedTopExperts, "Routing kernel expects topK experts <= %d, got %d",
        MaxSupportedTopExperts, data.mTopK);
    TLLM_CHECK_WITH_INFO(data.mNumExperts <= MaxSupportedExperts,
        "Routing kernel expects #experts %d to be no more than %d", data.mNumExperts, MaxSupportedExperts);
    // similar check
    TLLM_CHECK_WITH_INFO(
        data.mNumExperts % 4 == 0, "Routing kernel expects #experts %d to be a multiple of 4.", data.mNumExperts);

    bool const useSingleBlock = data.mNumTokens <= BlockKernelMaxNumTokens;

    bool const useSingleCluster = data.mNumTokens <= ((data.mPtrScores != nullptr || data.mPtrTopKIds != nullptr)
                                          ? MaxNumTokensSingleClusterScores
                                          : MaxNumTokensSingleCluster);

    if (!useSingleCluster && !useSingleBlock)
    {
        TLLM_CHECK_WITH_INFO((data.mPtrTopKPacked != nullptr || data.mPtrTopKIds != nullptr),
            "When #tokens is large, `mPtrTopKPacked` or `mPtrTopKIds` is a required input.");
        TLLM_CHECK_WITH_INFO(
            data.mPtrExpertCounts != nullptr, "When #tokens is large, `mPtrExpertCounts` is a required input.");
    }
    uint32_t const numThreadsHist = min(1024, getMaxNumExperts(data.mNumExperts));
    if (useSingleBlock)
    {
        //@TODO: For now we use the single block kernel for cases with token number no larger than 4.
        // We will future tune this threshold based on the performance.
        launchBlockKernel(data, numThreadsHist, stream);
    }
    else if (useSingleCluster)
    {
        launchClusterKernel(data, stream);
    }
    else
    {
        uint32_t const expandedIdxSize = data.mNumTokens * data.mTopK;
        uint32_t const histogramEltsPerBlock = 8 * numThreadsHist;
        uint32_t const offsetEltsPerBlock = NumEltsPerOffsetTilePerThread * numThreadsHist;

        // Limit grid size (all kernels use a grid-stride loop).
        uint32_t const maxNumBlocks = 1024;

        int const numBlocksHistogram
            = std::min((expandedIdxSize + histogramEltsPerBlock - 1) / histogramEltsPerBlock, maxNumBlocks);
        int const numBlocksOffsets
            = std::min((expandedIdxSize + offsetEltsPerBlock - 1) / offsetEltsPerBlock, maxNumBlocks);

        if (data.mPtrScores != nullptr && data.mPtrTopKIds == nullptr)
        {
            launchHistogramScoresKernel(data, maxNumBlocks, numThreadsHist, stream);
        }
        else
        {
            // Reset the global histograms.
            launchInitExpertCounts(data, numThreadsHist, stream);
        }
        launchHistogramKernel(data, numBlocksHistogram, numThreadsHist, stream);
        launchOffsetsKernel(data, numBlocksOffsets, numThreadsHist, stream);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace routingRenormalize
} // namespace moe::dev::routing
