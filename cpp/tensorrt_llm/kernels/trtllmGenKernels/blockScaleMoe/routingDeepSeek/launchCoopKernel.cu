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
#include "RoutingDeepSeekCommon.cuh"

namespace moe::dev::routing
{
namespace routingDeepSeek
{

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KernelParams>
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
__global__ void __launch_bounds__(KernelParams::MaxNumExperts) routingIndicesCoopKernel(KernelParams params)
{
    // number of experts is bounded by number of threads
    int constexpr NumThreads = KernelParams::MaxNumExperts;
    __shared__ int32_t __attribute((aligned(128))) smemExpertCount[NumThreads];
    __shared__ int32_t __attribute((aligned(128))) smemExpertOffset[NumThreads];
    // needed for the exclusive sum of token offsets
    using Scan = cub::BlockScan<int32_t, NumThreads, cub::BLOCK_SCAN_WARP_SCANS>;
    __shared__ typename Scan::TempStorage tempStorage;
    // 64 elements -> 128+ registers. Above that we may start to see spilling to local memory.
    static constexpr int MaxExpandedIdxPerThread = 64;

    // Initialize grid.
    cg::grid_group grid = cg::this_grid();
    // Note: the following is more efficient than grid.block_index() because we don't use y and z.
    int32_t const gridBlockIdx = blockIdx.x;
    int32_t const gridThreadIdx = NumThreads * gridBlockIdx + threadIdx.x;
    int32_t const numBlocks = gridDim.x;
    int32_t const numThreadsPerGrid = numBlocks * NumThreads;

    int32_t const warpIdx = __shfl_sync(0xffffffff, threadIdx.x / WarpSize, 0);

    auto expandedIdxSize = params.mNumTokens * params.mTopK;

    // pre-fill the counts with 0
    smemExpertCount[threadIdx.x] = 0;
    __syncthreads();

    // then wait on primary grid
    if constexpr (KernelParams::UsePdl)
    {
        cudaGridDependencySynchronize();
    }

    // each thread keeps has some number of "expanded indexes" assigned to it
    // for each of these, we keep the associated expert and offset within expert in registers
    int32_t expertIndexes[MaxExpandedIdxPerThread];
    int32_t expertOffsets[MaxExpandedIdxPerThread];
    auto localExpertExtent = params.mNumLocalExperts << params.mLocalExpertsStrideLog2;
    // In order to avoid a serialization LDG-ATOMS-LDG-ATOMS-..., we skip multiple iterations at a
    // time, and branch between a fast path without bound checks and a slow path with bound checks.
    int constexpr IterStride = 4;
    static_assert(MaxExpandedIdxPerThread % IterStride == 0);

    // Define a lambda to avoid code duplication in both branches.
    auto loopBody = [&](int ii, int expandedIdx)
    {
        int32_t expertIdx
            = params.mPtrTopKIds != nullptr ? params.mPtrTopKIds[expandedIdx] : params.mPtrTopKPacked[expandedIdx].idx;
        expertIndexes[ii] = expertIdx;
        // check whether this expert is local to our GPU at all and ignore if not
        auto localExpertIdx = expertIdx - params.mLocalExpertsStartIdx;
        auto isLocalExpert = localExpertIdx >= 0 && localExpertIdx < localExpertExtent
            && (localExpertIdx & params.mLocalExpertsStrideLog2) == 0;
        expertOffsets[ii] = isLocalExpert ? atomicAdd(smemExpertCount + expertIdx, 1) : 0;
    };

#pragma unroll
    for (int32_t ii0 = 0; ii0 < MaxExpandedIdxPerThread; ii0 += IterStride)
    {
        // Whether it's safe to do multiple iterations without bound checks.
        bool const takeFastPath = (ii0 + IterStride) * numThreadsPerGrid <= expandedIdxSize;
        if (takeFastPath)
        {
#pragma unroll
            for (int32_t jj = 0; jj < IterStride; jj++)
            {
                int const ii = ii0 + jj;
                auto expandedIdx = static_cast<int32_t>(gridThreadIdx) + ii * numThreadsPerGrid;
                loopBody(ii, expandedIdx);
            }
        }
        else
        {
            bool doBreak = false;
#pragma unroll
            for (int32_t jj = 0; jj < IterStride; jj++)
            {
                int const ii = ii0 + jj;
                auto expandedIdx = static_cast<int32_t>(gridThreadIdx) + ii * numThreadsPerGrid;
                if (expandedIdx >= expandedIdxSize)
                {
                    doBreak = true;
                    break;
                }
                loopBody(ii, expandedIdx);
            }
            if (doBreak)
            {
                break;
            }
        }
    }

    // Make histogram (token counts per expert) available to all threads in the block.
    __syncthreads();

    //
    // Each thread now represents one expert
    //

    // Add the local bin count to the common bin count and get a per-CTA offset.
    int32_t const localExpertCount = smemExpertCount[threadIdx.x];

    int32_t blockExpertOffset = 0;
    if (threadIdx.x < params.mNumExperts)
    {
        blockExpertOffset = atomicAdd(&params.mPtrExpertCounts[threadIdx.x], localExpertCount);
    }

    // Sync to wait for completion of the histogram reduction.
    grid.sync();

    // Get total count for this expert.
    int32_t count = (threadIdx.x < params.mNumExperts) ? params.mPtrExpertCounts[threadIdx.x] : 0;

    // Note: the scan is redundant in all CTAs, but doing it in only 1 CTA would be worse for latency.

    // Compute the runtime config for projections
    // Whether or not an expert is local is taken into account when smemExpertCount is computed
    // so we do not need to take it into account here.

    int32_t numCta;
    if constexpr (KernelParams::isPow2)
    {
        numCta = divUpLog2<int32_t>(count, params.mPaddingLog2);
    }
    else
    {
        numCta = divUpTileN<int32_t>(count, params.mTileTokensDim);
    }

    int32_t ctaOffset;
    int32_t numNonExitingCtas;
    Scan(tempStorage).ExclusiveSum(numCta, ctaOffset, numNonExitingCtas);

    for (int32_t cta = gridBlockIdx; cta < numCta; cta += numBlocks)
    {
        const int32_t localExpertIdx = (threadIdx.x - params.mLocalExpertsStartIdx) >> params.mLocalExpertsStrideLog2;
        params.mPtrCtaIdxXyToBatchIdx[ctaOffset + cta] = localExpertIdx;
        int32_t mnLimit1;
        int32_t mnLimit2;
        if constexpr (KernelParams::isPow2)
        {
            mnLimit1 = mulLog2<int32_t>(ctaOffset + cta + 1, params.mPaddingLog2);
            mnLimit2 = mulLog2<int32_t>(ctaOffset, params.mPaddingLog2) + count;
        }
        else
        {
            mnLimit1 = mulTileN<int32_t>(ctaOffset + cta + 1, params.mTileTokensDim);
            mnLimit2 = mulTileN<int32_t>(ctaOffset, params.mTileTokensDim) + count;
        }
        params.mPtrCtaIdxXyToMnLimit[ctaOffset + cta] = min(mnLimit1, mnLimit2);
    }

    // get the padded offset associated with this expert
    int32_t offset;
    if constexpr (KernelParams::isPow2)
    {
        offset = mulLog2<int32_t>(ctaOffset, params.mPaddingLog2);
    }
    else
    {
        offset = mulTileN<int32_t>(ctaOffset, params.mTileTokensDim);
    }
    int32_t permutedIdxSize;
    if constexpr (KernelParams::isPow2)
    {
        permutedIdxSize = mulLog2<int32_t>(numNonExitingCtas, params.mPaddingLog2);
    }
    else
    {
        permutedIdxSize = mulTileN<int32_t>(numNonExitingCtas, params.mTileTokensDim);
    }

    // write out padded count
    if (gridBlockIdx == 0 && warpIdx == NumThreads / WarpSize - 1 && cute::elect_one_sync())
    {
        params.mPtrPermutedIdxSize[0] = permutedIdxSize;
        params.mPtrNumNonExitingCtas[0] = numNonExitingCtas;
    }

    // write expert offsets to shared
    smemExpertOffset[threadIdx.x] = offset + blockExpertOffset;

    // make expert offsets available to all threads
    __syncthreads();

    // trigger the secondary kernel when using PDL
    // We can't do it earlier because FC1 depends on the mPtrCtaIdxXyToBatchIdx,
    // mPtrCtaIdxXyToMnLimit, mPtrNumNonExitingCtas and mPtrTotalNumPaddedTokens
    // TODO: this is not sufficient to ensure visibility in the next kernel!
    if constexpr (KernelParams::UsePdl)
    {
        cudaTriggerProgrammaticLaunchCompletion();
    }

// each thread has the same "expanded indexes" assigned to it as above
// at this point, we know the final offsets of experts and the offsets within
// experts, which allows writing the final index values
#pragma unroll
    for (int32_t ii = 0; ii < MaxExpandedIdxPerThread; ++ii)
    {
        auto expandedIdx = static_cast<int32_t>(gridThreadIdx) + ii * numThreadsPerGrid;
        if (expandedIdx >= expandedIdxSize)
        {
            break;
        }
        auto expertIdx = expertIndexes[ii];
        // check whether this expert is local to our GPU at all
        auto localExpertIdx = static_cast<int32_t>(expertIdx) - params.mLocalExpertsStartIdx;
        auto isLocalExpert = localExpertIdx >= 0 && localExpertIdx < localExpertExtent
            && (localExpertIdx & params.mLocalExpertsStrideLog2) == 0;
        auto tokenIdx = expandedIdx / params.mTopK;
        auto permutedIdx = isLocalExpert ? int32_t{smemExpertOffset[expertIdx]} + expertOffsets[ii] : int32_t{-1};
        if (params.mPtrExpandedIdxToPermutedIdx != nullptr)
        {
            params.mPtrExpandedIdxToPermutedIdx[expandedIdx] = permutedIdx;
        }
        if (params.mPtrPermutedIdxToExpandedIdx != nullptr && isLocalExpert)
        {
            params.mPtrPermutedIdxToExpandedIdx[permutedIdx] = expandedIdx;
        }
        if (params.mPtrPermutedIdxToTokenIdx != nullptr && isLocalExpert)
        {
            params.mPtrPermutedIdxToTokenIdx[permutedIdx] = tokenIdx;
        }
    }
}
#else
__global__ void routingIndicesCoopKernel(KernelParams params)
{
    assert(false && "routingIndicesCoopKernel is only supported on SM90+ architectures");
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

void launchCoopKernel(Data& data, int numBlocksCoop, int numThreadsHist, void* stream)
{
    LAUNCH_ROUTING_DEEPSEEK(data,
        /*coopLaunch=*/true, routingIndicesCoopKernel, numBlocksCoop, numThreadsHist,
        /*smemSize=*/0, // No dynamic smem
        stream, data.mNumExpertGroups > 1, /*forceFloatInput=*/true);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace routingDeepSeek
} // namespace moe::dev::routing
