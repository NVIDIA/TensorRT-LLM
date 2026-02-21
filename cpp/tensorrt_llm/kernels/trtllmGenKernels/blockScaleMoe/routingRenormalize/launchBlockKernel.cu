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

template <typename KernelParams>
__global__ void __launch_bounds__(KernelParams::MaxNumExperts <= 1024 ? KernelParams::MaxNumExperts : 1024)
    routingIndicesBlockKernel(KernelParams params)
{
    // types used in this kernel
    using OutputT = typename KernelParams::OutputT;
    using InputT = typename KernelParams::InputT;
    using BaseType = std::conditional_t<KernelParams::DoSoftmaxBeforeTopK, float, InputT>;
    using TypePacked = PackedScoreIdx<BaseType>;
    static constexpr int MaxNumExperts = KernelParams::MaxNumExperts;
    // When MaxNumExperts > 1024, cap actual thread count at 1024 and let each thread handle
    // multiple experts. This is needed because CUDA blocks support at most 1024 threads.
    static constexpr int NumThreadsBlock = MaxNumExperts <= 1024 ? MaxNumExperts : 1024;
    static constexpr int ExpertsPerThread = MaxNumExperts / NumThreadsBlock;
    static_assert(MaxNumExperts % NumThreadsBlock == 0, "MaxNumExperts must be a multiple of NumThreadsBlock");

    int32_t const warpIdx = __shfl_sync(0xffffffff, threadIdx.x / WarpSize, 0);
    int32_t const laneIdx = cutlass::arch::LaneId();
    auto scoreOffset = warpIdx * params.mNumExperts;
    bool validToken = warpIdx < params.mNumTokens;

    static constexpr int VecSize = KernelParams::MaxNumExperts / WarpSize;
    static constexpr int totalExpertCounts = BlockKernelMaxNumTokens * MaxNumExperts;
    __shared__ int8_t __attribute((aligned(128))) smemOffset[totalExpertCounts];
    __shared__ int8_t __attribute((aligned(128))) smemKIdx[totalExpertCounts];

    using Scan = cub::BlockScan<int32_t, NumThreadsBlock>;
    __shared__ typename Scan::TempStorage tempStorage;

    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<WarpSize>(block);

    for (int i = threadIdx.x; i < totalExpertCounts; i += blockDim.x)
    {
        smemOffset[i] = int8_t{-1};
        smemKIdx[i] = int8_t{-1};
    }
    __syncthreads();

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    // then wait on primary grid
    if constexpr (KernelParams::UsePdl)
    {
        cudaGridDependencySynchronize();
    }
#endif

    if (params.mPtrTopKIds != nullptr)
    {
        if (validToken)
        {
            if (laneIdx < params.mTopK)
            {
                auto expertIdx = params.mPtrTopKIds[warpIdx * params.mTopK + laneIdx];
                if (expertIdx != -1)
                {
                    int offset = warpIdx * MaxNumExperts + expertIdx;
                    smemKIdx[offset] = static_cast<int8_t>(laneIdx);
                }
                else
                {
                    params.mPtrExpandedIdxToPermutedIdx[warpIdx * params.mTopK + laneIdx] = int32_t{-1};
                }
            }
        }
    }
    else if (params.mPtrScores != nullptr)
    {
        // in this case, each warp represents a token
        BaseType score[VecSize];
        int32_t idx[VecSize];

        BaseType warpTopKScore[KernelParams::MaxNumTopExperts];
        int32_t warpTopKExpertIdx[KernelParams::MaxNumTopExperts];

        BaseType minScore = BaseType{-INFINITY};
        if (validToken)
        {
            routingTopKExperts<BaseType, InputT, VecSize, KernelParams::MaxNumTopExperts,
                KernelParams::DoSoftmaxBeforeTopK>(warp, score, idx, warpTopKScore, warpTopKExpertIdx, laneIdx,
                params.mNumExperts, params.mTopK, params.mPtrScores + scoreOffset, params.mNormTopkProb);

            if (laneIdx < params.mTopK)
            {
                int offset = warpIdx * MaxNumExperts + warpTopKExpertIdx[laneIdx];
                smemKIdx[offset] = static_cast<int8_t>(laneIdx);
                if (params.mPtrTopKWeights != nullptr)
                {
                    params.mPtrTopKWeights[warpIdx * params.mTopK + laneIdx] = OutputT{warpTopKScore[laneIdx]};
                }
            }
        } // end if (validToken)
    }
    __syncthreads();

    // Each thread handles ExpertsPerThread contiguous experts.
    // Thread i handles experts [i * ExpertsPerThread, (i+1) * ExpertsPerThread).
    // Contiguous assignment ensures prefix sum ordering is correct.
    int accExpertCount[ExpertsPerThread];
#pragma unroll
    for (int e = 0; e < ExpertsPerThread; e++)
    {
        int expert = threadIdx.x * ExpertsPerThread + e;
        auto localExpIdx = expert - params.mLocalExpertsStartIdx;
        auto isLocal = localExpIdx >= 0 && localExpIdx < params.mNumLocalExperts
            && (localExpIdx & params.mLocalExpertsStrideLog2) == 0;

        // Get the count of each expert and the offset for each token
        accExpertCount[e] = 0;
        if (isLocal)
        {
            int offset = expert;
            for (int j = 0; j < BlockKernelMaxNumTokens; j++)
            {
                if (smemKIdx[offset] >= 0)
                {
                    smemOffset[offset] = static_cast<int8_t>(accExpertCount[e]);
                    accExpertCount[e]++;
                }
                offset += MaxNumExperts;
            }
        }
    }
    __syncthreads();

    // Get the number of CTAs and the offset for each CTA.
    // Use cub::BlockScan's array overload: each thread holds ExpertsPerThread items,
    // and ExclusiveSum computes the prefix sum across all NumThreadsBlock * ExpertsPerThread
    // items in thread order — exactly matching our contiguous expert assignment.
    int32_t numCtaPerExpert[ExpertsPerThread];
#pragma unroll
    for (int e = 0; e < ExpertsPerThread; e++)
    {
        if constexpr (KernelParams::isPow2)
        {
            numCtaPerExpert[e] = divUpLog2<int32_t>(accExpertCount[e], params.mPaddingLog2);
        }
        else
        {
            numCtaPerExpert[e] = divUpTileN<int32_t>(accExpertCount[e], params.mTileTokensDim);
        }
    }
    int32_t ctaOffsetPerExpert[ExpertsPerThread];
    int32_t numNonExitingCtas;
    Scan(tempStorage).ExclusiveSum(numCtaPerExpert, ctaOffsetPerExpert, numNonExitingCtas);

    // Compute padded expert scan counts (same array-overload pattern)
    int32_t tmpCountPerExpert[ExpertsPerThread];
#pragma unroll
    for (int e = 0; e < ExpertsPerThread; e++)
    {
        if constexpr (KernelParams::isPow2)
        {
            tmpCountPerExpert[e] = divUpMulLog2<int32_t>(accExpertCount[e], params.mPaddingLog2);
        }
        else
        {
            tmpCountPerExpert[e] = divUpMulTileN<int32_t>(accExpertCount[e], params.mTileTokensDim);
        }
    }
    int32_t expertScanCountsPerExpert[ExpertsPerThread];
    Scan(tempStorage).ExclusiveSum(tmpCountPerExpert, expertScanCountsPerExpert);
    __syncthreads();

    // Write CTA configs for each expert this thread handles
#pragma unroll
    for (int e = 0; e < ExpertsPerThread; e++)
    {
        int expert = threadIdx.x * ExpertsPerThread + e;
        auto localExpIdx = expert - params.mLocalExpertsStartIdx;
        auto isLocal = localExpIdx >= 0 && localExpIdx < params.mNumLocalExperts
            && (localExpIdx & params.mLocalExpertsStrideLog2) == 0;

        if (isLocal)
        {
            for (int cta = 0; cta < numCtaPerExpert[e]; ++cta)
            {
                int32_t const mappedLocalIdx
                    = (expert - params.mLocalExpertsStartIdx) >> params.mLocalExpertsStrideLog2;
                params.mPtrCtaIdxXyToBatchIdx[ctaOffsetPerExpert[e] + cta] = mappedLocalIdx;
                int32_t mnLimit1;
                int32_t mnLimit2;
                if constexpr (KernelParams::isPow2)
                {
                    mnLimit1 = mulLog2<int32_t>(ctaOffsetPerExpert[e] + cta + 1, params.mPaddingLog2);
                    mnLimit2 = mulLog2<int32_t>(ctaOffsetPerExpert[e], params.mPaddingLog2) + accExpertCount[e];
                }
                else
                {
                    mnLimit1 = mulTileN<int32_t>(ctaOffsetPerExpert[e] + cta + 1, params.mTileTokensDim);
                    mnLimit2 = mulTileN<int32_t>(ctaOffsetPerExpert[e], params.mTileTokensDim) + accExpertCount[e];
                }
                params.mPtrCtaIdxXyToMnLimit[ctaOffsetPerExpert[e] + cta] = min(mnLimit1, mnLimit2);
            }
        }
    }

    // at this point, we can write out padded count
    if (threadIdx.x == 0)
    {
        int32_t permutedIdxSize;
        if constexpr (KernelParams::isPow2)
        {
            permutedIdxSize = mulLog2<int32_t>(numNonExitingCtas, params.mPaddingLog2);
        }
        else
        {
            permutedIdxSize = mulTileN<int32_t>(numNonExitingCtas, params.mTileTokensDim);
        }
        params.mPtrPermutedIdxSize[0] = permutedIdxSize;
        params.mPtrNumNonExitingCtas[0] = numNonExitingCtas;
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    // we can trigger the next kernel at this point
    if constexpr (KernelParams::UsePdl)
    {
        cudaTriggerProgrammaticLaunchCompletion();
    }
#endif

    for (int tokenIdx = 0; tokenIdx < params.mNumTokens; tokenIdx++)
    {
#pragma unroll
        for (int e = 0; e < ExpertsPerThread; e++)
        {
            int expert = threadIdx.x * ExpertsPerThread + e;
            int offset = tokenIdx * MaxNumExperts + expert;
            if (smemKIdx[offset] >= 0)
            {
                auto localExpIdx = expert - params.mLocalExpertsStartIdx;
                auto isLocal = localExpIdx >= 0 && localExpIdx < params.mNumLocalExperts
                    && (localExpIdx & params.mLocalExpertsStrideLog2) == 0;

                int const expandedIdx = tokenIdx * params.mTopK + smemKIdx[offset];
                int const offsetWithinExpert = static_cast<int>(smemOffset[offset]);
                int const offsetForExpert = expertScanCountsPerExpert[e];
                int const permutedIdx = isLocal ? offsetForExpert + offsetWithinExpert : int32_t{-1};

                if (params.mPtrExpandedIdxToPermutedIdx != nullptr)
                {
                    params.mPtrExpandedIdxToPermutedIdx[expandedIdx] = permutedIdx;
                }
                if (params.mPtrPermutedIdxToExpandedIdx != nullptr && isLocal)
                {
                    params.mPtrPermutedIdxToExpandedIdx[permutedIdx] = expandedIdx;
                }
                if (params.mPtrPermutedIdxToTokenIdx != nullptr && isLocal)
                {
                    params.mPtrPermutedIdxToTokenIdx[permutedIdx] = tokenIdx;
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void launchBlockKernel(Data const& data, uint32_t numThreadsHist, void* stream)
{
    LAUNCH_ROUTING_RENORNALIZE(data, false, routingIndicesBlockKernel, 1, numThreadsHist,
        /*smemSize=*/0, // No dynamic smem
        stream, data.mDoSoftmaxBeforeTopK);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace routingRenormalize
} // namespace moe::dev::routing
