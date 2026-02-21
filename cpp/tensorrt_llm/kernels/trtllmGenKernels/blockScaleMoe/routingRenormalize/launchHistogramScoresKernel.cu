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
#include "RoutingRenormalizeCommon.cuh"

namespace moe::dev::routing
{
namespace routingRenormalize
{

////////////////////////////////////////////////////////////////////////////////////////////////////

// this kernel is needed in case we have scores as input for the histogram kernel
template <typename KernelParams>
__global__ void __launch_bounds__(KernelParams::MaxNumExperts <= 1024 ? KernelParams::MaxNumExperts : 1024)
    routingIndicesHistogramScoresKernel(KernelParams params)
{
    using OutputT = typename KernelParams::OutputT;
    using InputT = typename KernelParams::InputT;
    using BaseType = std::conditional_t<KernelParams::DoSoftmaxBeforeTopK, float, InputT>;
    // Cap actual thread count at 1024 when MaxNumExperts > 1024.
    static constexpr int NumThreadsBlock = KernelParams::MaxNumExperts <= 1024 ? KernelParams::MaxNumExperts : 1024;

    // VecSize stays based on MaxNumExperts — each warp still processes all experts for one token.
    static constexpr int VecSize = KernelParams::MaxNumExperts / WarpSize;

    int32_t const laneIdx = cutlass::arch::LaneId();
    int32_t const warpIdx = threadIdx.x / WarpSize;
    // Use NumThreadsBlock (actual thread count) for grid-stride warp/thread addressing
    int32_t const globalWarpIdx = blockIdx.x * NumThreadsBlock / WarpSize + warpIdx;
    int32_t const globalWarpStride = gridDim.x * NumThreadsBlock / WarpSize;
    BaseType minScore = BaseType{-INFINITY};
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<WarpSize>(block);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    // Wait on primary grid.
    if constexpr (KernelParams::UsePdl)
    {
        cudaGridDependencySynchronize();
    }
#endif // if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))

    // initialize the mPtrExpertCounts — use NumThreadsBlock for grid-stride
    int32_t expertCountsNum = 2 * params.mNumExperts;
    int32_t globalThreadIdx = blockIdx.x * NumThreadsBlock + threadIdx.x;
    int32_t globalThreadStride = gridDim.x * NumThreadsBlock;
    initArr(globalThreadIdx, expertCountsNum, globalThreadStride, params.mPtrExpertCounts, 0);

    // in this case, each warp represents a token, and we use a grid-stride loop
    // over all warps/tokens
    BaseType allScores[VecSize];
    int32_t allExpertIdx[VecSize];
    BaseType warpTopKScore[KernelParams::MaxNumTopExperts];
    int32_t warpTopKExpertIdx[KernelParams::MaxNumTopExperts];
    for (int tokenIdx = globalWarpIdx; tokenIdx < params.mNumTokens; tokenIdx += globalWarpStride)
    {
        auto scoreOffset = tokenIdx * params.mNumExperts;

        routingTopKExperts<BaseType, InputT, VecSize, KernelParams::MaxNumTopExperts,
            KernelParams::DoSoftmaxBeforeTopK>(warp, allScores, allExpertIdx, warpTopKScore, warpTopKExpertIdx, laneIdx,
            params.mNumExperts, params.mTopK, params.mPtrScores + scoreOffset, params.mNormTopkProb);

        if (laneIdx < params.mTopK)
        {
            PackedScoreIdx<OutputT> packedScore{
                static_cast<OutputT>(warpTopKScore[laneIdx]), static_cast<int16_t>(warpTopKExpertIdx[laneIdx])};
            params.mPtrTopKPacked[tokenIdx * params.mTopK + laneIdx] = packedScore;
        }
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    // Trigger secondary kernel AFTER writing all packed scores, so the next kernel
    // (routingIndicesHistogramKernel) sees the completed mPtrTopKPacked writes.
    if constexpr (KernelParams::UsePdl)
    {
        cudaTriggerProgrammaticLaunchCompletion();
    }
#endif // if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void launchHistogramScoresKernel(Data const& data, uint32_t maxNumBlocks, uint32_t numThreadsHist, void* stream)
{
    LAUNCH_ROUTING_RENORNALIZE(data, false, routingIndicesHistogramScoresKernel, maxNumBlocks, numThreadsHist,
        /*smemSize=*/0, // No dynamic smem
        stream, data.mDoSoftmaxBeforeTopK);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace routingRenormalize
} // namespace moe::dev::routing
