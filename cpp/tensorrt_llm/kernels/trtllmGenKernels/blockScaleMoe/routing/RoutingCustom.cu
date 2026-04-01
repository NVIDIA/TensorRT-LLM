/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
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

// Custom routing: entry point, kernel definitions, and launch wrappers.
//
// Kernel inventory:
//   1. routingIndicesBlockKernel      — single-block fused kernel (≤4 tokens)
//  1b. routingIndicesDynBlockKernel   — dynamic-block fused kernel (≤16 tokens, ≤512 experts)
//   2. routingIndicesClusterKernel    — single-cluster fused kernel (≤256 tokens, SM90+)
//   3. routingIndicesHistogramScoresKernel — TopK + histogram from raw scores
//   4. routingIndicesCoopKernel       — cooperative histogram + offsets (defined in RoutingKernel.cuh)
//   5. routingInitExpertCounts        — zero expert counts (defined in RoutingKernel.cuh)
//   6. routingIndicesHistogramKernel  — histogram from packed TopK (defined in RoutingKernel.cuh)
//   7. routingIndicesOffsetsKernel    — prefix-scan + permutation (defined in RoutingKernel.cuh)

#include "RoutingCustomPolicy.cuh"

namespace moe::dev::routing
{
namespace routingCustom
{

////////////////////////////////////////////////////////////////////////////////////////////////////
// Dual warp-level exclusive prefix scan over NumExpertWarps * 32 values.
// Scans val1 and val2 simultaneously while sharing the same two __syncthreads() barriers,
// reducing 4 barriers (two separate scans) to 2.
////////////////////////////////////////////////////////////////////////////////////////////////////
template <int NumExpertWarps>
__device__ __forceinline__ void warpExclusiveScan(int32_t val1, int32_t val2, int32_t laneIdx, int32_t warpIdx,
    int32_t* warpTotals1, int32_t* warpTotals2, int32_t& prefix1, int32_t& prefix2, int32_t& totalSum1)
{
    static_assert(NumExpertWarps <= WarpSize, "NumExpertWarps must fit in one warp for the cross-warp scan");

    int32_t inc1 = val1, inc2 = val2;
#pragma unroll
    for (int j = 1; j < WarpSize; j *= 2)
    {
        int32_t n1 = __shfl_up_sync(0xffffffff, inc1, j);
        int32_t n2 = __shfl_up_sync(0xffffffff, inc2, j);
        if (laneIdx >= j)
        {
            inc1 += n1;
            inc2 += n2;
        }
    }

    if (warpIdx < NumExpertWarps && laneIdx == WarpSize - 1)
    {
        warpTotals1[warpIdx] = inc1;
        warpTotals2[warpIdx] = inc2;
    }
    __syncthreads();

    if (warpIdx == 0)
    {
        int32_t wt1 = (laneIdx < NumExpertWarps) ? warpTotals1[laneIdx] : 0;
        int32_t wt2 = (laneIdx < NumExpertWarps) ? warpTotals2[laneIdx] : 0;
#pragma unroll
        for (int j = 1; j < NumExpertWarps; j *= 2)
        {
            int32_t n1 = __shfl_up_sync(0xffffffff, wt1, j);
            int32_t n2 = __shfl_up_sync(0xffffffff, wt2, j);
            if (laneIdx >= j)
            {
                wt1 += n1;
                wt2 += n2;
            }
        }
        if (laneIdx < NumExpertWarps)
        {
            warpTotals1[laneIdx] = wt1;
            warpTotals2[laneIdx] = wt2;
        }
    }
    __syncthreads();

    totalSum1 = warpTotals1[NumExpertWarps - 1];
    int32_t wp1 = (warpIdx > 0 && warpIdx < NumExpertWarps) ? warpTotals1[warpIdx - 1] : 0;
    int32_t wp2 = (warpIdx > 0 && warpIdx < NumExpertWarps) ? warpTotals2[warpIdx - 1] : 0;
    prefix1 = inc1 - val1 + wp1;
    prefix2 = inc2 - val2 + wp2;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// 1. Block kernel — single-block fused kernel for ≤4 tokens.
//    Fuses TopK, histogram, prefix-scan, and permutation in one block.
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KernelParams>
__global__ void __launch_bounds__(KernelParams::MaxNumExperts <= 1024 ? KernelParams::MaxNumExperts : 1024)
    routingIndicesBlockKernel(KernelParams params)
{
    // types used in this kernel
    using OutputT = typename KernelParams::OutputT;
    using InputT = typename KernelParams::InputT;
    using BaseType = typename KernelParams::ExpertSelectPolicy::template BaseType<InputT>;
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
    if (params.mUsePdl)
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
                if (expertIdx > -1 && expertIdx < params.mNumExperts)
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
        BaseType warpTopKScore[KernelParams::MaxNumTopExperts];
        int32_t warpTopKExpertIdx[KernelParams::MaxNumTopExperts];

        if (validToken)
        {
            KernelParams::ExpertSelectPolicy::template apply<BaseType, InputT, VecSize, KernelParams::MaxNumTopExperts>(
                warp, warpTopKScore, warpTopKExpertIdx, laneIdx, params.mNumExperts, params.mTopK,
                params.mPtrScores + scoreOffset, params);

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
    else if (params.mPtrTopKPacked != nullptr)
    {
        if (validToken)
        {
            if (laneIdx < params.mTopK)
            {
                auto const expandedIdx = warpIdx * params.mTopK + laneIdx;
                auto const scoreIdx = params.mPtrTopKPacked[expandedIdx];
                int const expertIdx = static_cast<int>(scoreIdx.idx);
                if (expertIdx >= 0 && expertIdx < params.mNumExperts)
                {
                    int const offset = warpIdx * MaxNumExperts + expertIdx;
                    smemKIdx[offset] = static_cast<int8_t>(laneIdx);
                    if (params.mPtrTopKWeights != nullptr)
                    {
                        params.mPtrTopKWeights[expandedIdx] = static_cast<OutputT>(scoreIdx.score);
                    }
                }
                else if (params.mPtrExpandedIdxToPermutedIdx != nullptr)
                {
                    params.mPtrExpandedIdxToPermutedIdx[expandedIdx] = int32_t{-1};
                }
            }
        }
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
            && (localExpIdx & ((1 << params.mLocalExpertsStrideLog2) - 1)) == 0;

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
        if (params.mIsPow2)
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
    __syncthreads(); // Required barrier before reusing TempStorage for the next BlockScan

    // Compute padded expert scan counts (same array-overload pattern)
    int32_t tmpCountPerExpert[ExpertsPerThread];
#pragma unroll
    for (int e = 0; e < ExpertsPerThread; e++)
    {
        if (params.mIsPow2)
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
            && (localExpIdx & ((1 << params.mLocalExpertsStrideLog2) - 1)) == 0;

        if (isLocal)
        {
            for (int cta = 0; cta < numCtaPerExpert[e]; ++cta)
            {
                int32_t const mappedLocalIdx
                    = (expert - params.mLocalExpertsStartIdx) >> params.mLocalExpertsStrideLog2;
                params.mPtrCtaIdxXyToBatchIdx[ctaOffsetPerExpert[e] + cta] = mappedLocalIdx;
                int32_t mnLimit1;
                int32_t mnLimit2;
                if (params.mIsPow2)
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

    if (threadIdx.x == 0)
    {
        int32_t permutedIdxSize;
        if (params.mIsPow2)
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
                    && (localExpIdx & ((1 << params.mLocalExpertsStrideLog2) - 1)) == 0;

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

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    // Trigger the secondary kernel AFTER all global memory writes (including permutation indices).
    // The downstream kernels depend on all routing outputs being visible.
    if (params.mUsePdl)
    {
        cudaTriggerProgrammaticLaunchCompletion();
    }
#endif
}

void launchBlockKernel(Data const& data, uint32_t numThreadsHist, void* stream)
{
    LAUNCH_ROUTING_CUSTOM(data, false, routingIndicesBlockKernel, 1, numThreadsHist,
        /*smemSize=*/0, // No dynamic smem
        stream);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// 1b. Dynamic-block kernel — single-block with dynamic thread count and dynamic shared memory.
//
//     Compared to routingIndicesBlockKernel (which fixes blockDim = MaxExperts):
//       1. Thread count = min(max(numTokens*32, MaxExperts), 1024) so each token
//          gets its own warp — eliminates the Phase-1 TopK batch loop for small batches.
//       2. Warp-level Hillis-Steele scan replaces CUB BlockScan, fusing two scans
//          into one (2 barriers instead of 4) with no compile-time thread count dependency.
//       3. Dynamic shared memory enables flexible token counts (up to 16).
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KernelParams>
__global__ void routingIndicesDynBlockKernel(KernelParams params)
{
    using OutputT = typename KernelParams::OutputT;
    using InputT = typename KernelParams::InputT;
    using BaseType = typename KernelParams::ExpertSelectPolicy::template BaseType<InputT>;
    using TypePacked = PackedScoreIdx<BaseType>;
    static constexpr int MaxNumExperts = KernelParams::MaxNumExperts;
    static constexpr int NumThreadsExperts = MaxNumExperts <= 1024 ? MaxNumExperts : 1024;
    static constexpr int ExpertsPerThread = MaxNumExperts / NumThreadsExperts;
    static constexpr int NumExpertWarps = NumThreadsExperts / WarpSize;
    static constexpr int VecSize = MaxNumExperts / WarpSize;

    static_assert(MaxNumExperts % WarpSize == 0);
    static_assert(MaxNumExperts % NumThreadsExperts == 0);

    int32_t const warpIdx = __shfl_sync(0xffffffff, threadIdx.x / WarpSize, 0);
    int32_t const laneIdx = cutlass::arch::LaneId();
    int32_t const numWarps = blockDim.x / WarpSize;

    extern __shared__ char dynSmem[];
    int const numSlots = params.mNumTokens * MaxNumExperts;
    int8_t* smemKIdx = reinterpret_cast<int8_t*>(dynSmem);
    int16_t* smemOffset = reinterpret_cast<int16_t*>(dynSmem + numSlots);
    char* warpBase = dynSmem + numSlots + numSlots * 2;
    warpBase = reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(warpBase) + 127) & ~127);
    int32_t* warpTotals = reinterpret_cast<int32_t*>(warpBase);
    int32_t* warpTotals2 = warpTotals + NumExpertWarps;

    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<WarpSize>(block);

    for (int i = threadIdx.x; i < numSlots; i += blockDim.x)
    {
        smemKIdx[i] = int8_t{-1};
    }
    __syncthreads();

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    if (params.mUsePdl)
    {
        cudaGridDependencySynchronize();
    }
#endif

    // Phase 1: TopK — one warp per token (loop only when numTokens > numWarps)
    for (int tokenIdx = warpIdx; tokenIdx < params.mNumTokens; tokenIdx += numWarps)
    {
        if (params.mPtrTopKIds != nullptr)
        {
            if (laneIdx < params.mTopK)
            {
                auto expertIdx = params.mPtrTopKIds[tokenIdx * params.mTopK + laneIdx];
                if (expertIdx > -1 && expertIdx < params.mNumExperts)
                {
                    smemKIdx[tokenIdx * MaxNumExperts + expertIdx] = static_cast<int8_t>(laneIdx);
                }
                else if (params.mPtrExpandedIdxToPermutedIdx != nullptr)
                {
                    params.mPtrExpandedIdxToPermutedIdx[tokenIdx * params.mTopK + laneIdx] = int32_t{-1};
                }
            }
        }
        else if (params.mPtrScores != nullptr)
        {
            BaseType warpTopKScore[KernelParams::MaxNumTopExperts];
            int32_t warpTopKExpertIdx[KernelParams::MaxNumTopExperts];

            auto scoreOff = tokenIdx * params.mNumExperts;
            KernelParams::ExpertSelectPolicy::template apply<BaseType, InputT, VecSize, KernelParams::MaxNumTopExperts>(
                warp, warpTopKScore, warpTopKExpertIdx, laneIdx, params.mNumExperts, params.mTopK,
                params.mPtrScores + scoreOff, params);

            if (laneIdx < params.mTopK)
            {
                smemKIdx[tokenIdx * MaxNumExperts + warpTopKExpertIdx[laneIdx]] = static_cast<int8_t>(laneIdx);
                if (params.mPtrTopKWeights != nullptr)
                {
                    params.mPtrTopKWeights[tokenIdx * params.mTopK + laneIdx] = OutputT{warpTopKScore[laneIdx]};
                }
            }
        }
        else if (params.mPtrTopKPacked != nullptr)
        {
            if (laneIdx < params.mTopK)
            {
                auto expandedIdx = tokenIdx * params.mTopK + laneIdx;
                auto scoreIdx = params.mPtrTopKPacked[expandedIdx];
                int const expertIdx = static_cast<int>(scoreIdx.idx);
                if (expertIdx >= 0 && expertIdx < params.mNumExperts)
                {
                    smemKIdx[tokenIdx * MaxNumExperts + expertIdx] = static_cast<int8_t>(laneIdx);
                    if (params.mPtrTopKWeights != nullptr)
                    {
                        params.mPtrTopKWeights[expandedIdx] = static_cast<OutputT>(scoreIdx.score);
                    }
                }
                else if (params.mPtrExpandedIdxToPermutedIdx != nullptr)
                {
                    params.mPtrExpandedIdxToPermutedIdx[expandedIdx] = int32_t{-1};
                }
            }
        }
    }
    __syncthreads();

    // Phase 2: Histogram
    int accExpertCount[ExpertsPerThread];
    if (threadIdx.x < NumThreadsExperts)
    {
#pragma unroll
        for (int e = 0; e < ExpertsPerThread; e++)
        {
            int expert = threadIdx.x * ExpertsPerThread + e;
            auto localExpIdx = expert - params.mLocalExpertsStartIdx;
            auto isLocal = localExpIdx >= 0 && localExpIdx < params.mNumLocalExperts
                && (localExpIdx & ((1 << params.mLocalExpertsStrideLog2) - 1)) == 0;
            accExpertCount[e] = 0;
            if (isLocal)
            {
                int offset = expert;
                for (int j = 0; j < params.mNumTokens; j++)
                {
                    if (smemKIdx[offset] >= 0)
                    {
                        smemOffset[offset] = static_cast<int16_t>(accExpertCount[e]);
                        accExpertCount[e]++;
                    }
                    offset += MaxNumExperts;
                }
            }
        }
    }
    else
    {
#pragma unroll
        for (int e = 0; e < ExpertsPerThread; e++)
        {
            accExpertCount[e] = 0;
        }
    }

    // Phase 3: Prefix-scan (merged dual warp-level scan, 2 barriers instead of 4)
    int32_t numCtaPerExpert[ExpertsPerThread];
    int32_t tmpCountPerExpert[ExpertsPerThread];
    int32_t ctaOffsetPerExpert[ExpertsPerThread];
    int32_t expertScanCountsPerExpert[ExpertsPerThread];
    int32_t numNonExitingCtas;
    {
#pragma unroll
        for (int e = 0; e < ExpertsPerThread; e++)
        {
            if (threadIdx.x < NumThreadsExperts)
            {
                if (params.mIsPow2)
                {
                    numCtaPerExpert[e] = divUpLog2<int32_t>(accExpertCount[e], params.mPaddingLog2);
                    tmpCountPerExpert[e] = divUpMulLog2<int32_t>(accExpertCount[e], params.mPaddingLog2);
                }
                else
                {
                    numCtaPerExpert[e] = divUpTileN<int32_t>(accExpertCount[e], params.mTileTokensDim);
                    tmpCountPerExpert[e] = divUpMulTileN<int32_t>(accExpertCount[e], params.mTileTokensDim);
                }
            }
            else
            {
                numCtaPerExpert[e] = 0;
                tmpCountPerExpert[e] = 0;
            }
        }

        int32_t localPrefix1[ExpertsPerThread], localPrefix2[ExpertsPerThread];
        int32_t threadTotal1 = 0, threadTotal2 = 0;
#pragma unroll
        for (int e = 0; e < ExpertsPerThread; e++)
        {
            localPrefix1[e] = threadTotal1;
            localPrefix2[e] = threadTotal2;
            threadTotal1 += numCtaPerExpert[e];
            threadTotal2 += tmpCountPerExpert[e];
        }

        int32_t threadPrefix1, threadPrefix2;
        warpExclusiveScan<NumExpertWarps>(threadTotal1, threadTotal2, laneIdx, warpIdx, warpTotals, warpTotals2,
            threadPrefix1, threadPrefix2, numNonExitingCtas);

#pragma unroll
        for (int e = 0; e < ExpertsPerThread; e++)
        {
            ctaOffsetPerExpert[e] = threadPrefix1 + localPrefix1[e];
            expertScanCountsPerExpert[e] = threadPrefix2 + localPrefix2[e];
        }
    }

    // Phase 4: CTA configs
    if (threadIdx.x < NumThreadsExperts)
    {
#pragma unroll
        for (int e = 0; e < ExpertsPerThread; e++)
        {
            int expert = threadIdx.x * ExpertsPerThread + e;
            auto localExpIdx = expert - params.mLocalExpertsStartIdx;
            auto isLocal = localExpIdx >= 0 && localExpIdx < params.mNumLocalExperts
                && (localExpIdx & ((1 << params.mLocalExpertsStrideLog2) - 1)) == 0;
            if (isLocal)
            {
                for (int cta = 0; cta < numCtaPerExpert[e]; ++cta)
                {
                    int32_t const mappedLocalIdx
                        = (expert - params.mLocalExpertsStartIdx) >> params.mLocalExpertsStrideLog2;
                    params.mPtrCtaIdxXyToBatchIdx[ctaOffsetPerExpert[e] + cta] = mappedLocalIdx;
                    int32_t mnLimit1, mnLimit2;
                    if (params.mIsPow2)
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
    }

    if (threadIdx.x == 0)
    {
        int32_t permutedIdxSize;
        if (params.mIsPow2)
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
    if (params.mUsePdl)
    {
        cudaTriggerProgrammaticLaunchCompletion();
    }
#endif

    // Phase 5: Permutation
    if (threadIdx.x < NumThreadsExperts)
    {
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
                        && (localExpIdx & ((1 << params.mLocalExpertsStrideLog2) - 1)) == 0;

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
}

void launchDynBlockKernel(Data const& data, uint32_t numThreadsHist, void* stream)
{
    int32_t const maxExperts = queryDispatchedMaxExperts(data);
    int const numSlots = data.mNumTokens * maxExperts;
    int const smemSize
        = numSlots + numSlots * 2 + 128 + 2 * (maxExperts / WarpSize) * static_cast<int>(sizeof(int32_t));
    int const threads = std::min(std::max(data.mNumTokens * static_cast<int>(WarpSize), maxExperts), 1024);

    LAUNCH_ROUTING_CUSTOM(data, false, routingIndicesDynBlockKernel, 1, threads, smemSize, stream);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// 2. Cluster kernel — single-cluster fused kernel for ≤256 tokens (SM90+).
//    Uses distributed shared memory across 8 blocks in a cluster.
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KernelParams>
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
__global__ void __cluster_dims__(NumBlocksPerCluster, 1, 1) __launch_bounds__(NumThreads)
    routingIndicesClusterKernel(KernelParams params)
{
    using OutputT = typename KernelParams::OutputT;
    using InputT = typename KernelParams::InputT;
    using BaseType = typename KernelParams::ExpertSelectPolicy::template BaseType<InputT>;
    using TypePacked = PackedScoreIdx<BaseType>;
    static constexpr int VecSize = KernelParams::MaxNumExperts / WarpSize;

    __shared__ TypePacked __attribute((aligned(128))) smemPackedScoreIdx[NumWarps * KernelParams::MaxNumTopExperts];

    uint32_t const clusterBlockRank = blockIdx.x;
    int32_t const warpIdx = __shfl_sync(0xffffffff, threadIdx.x / WarpSize, 0);
    int32_t const laneIdx = cutlass::arch::LaneId();
    auto warpTokenIdx = clusterBlockRank * NumWarps + warpIdx;
    auto scoreOffset = warpTokenIdx * params.mNumExperts;
    bool validToken = warpTokenIdx < params.mNumTokens;
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<WarpSize>(block);

    if (params.mUsePdl)
    {
        cudaGridDependencySynchronize();
    }

    if (params.mPtrScores != nullptr)
    {
        BaseType warpTopKScore[KernelParams::MaxNumTopExperts];
        int32_t warpTopKExpertIdx[KernelParams::MaxNumTopExperts];
        if (validToken)
        {
            KernelParams::ExpertSelectPolicy::template apply<BaseType, InputT, VecSize, KernelParams::MaxNumTopExperts>(
                warp, warpTopKScore, warpTopKExpertIdx, laneIdx, params.mNumExperts, params.mTopK,
                params.mPtrScores + scoreOffset, params);
            if (laneIdx < params.mTopK)
            {
                smemPackedScoreIdx[warpIdx * params.mTopK + laneIdx]
                    = TypePacked{warpTopKScore[laneIdx], static_cast<int16_t>(warpTopKExpertIdx[laneIdx])};
            }
        }
    }

    __cluster_barrier_arrive();
    __cluster_barrier_wait();

    if (params.mPtrScores != nullptr)
    {
        routingPermutation<KernelParams, BaseType, NumThreads, NumWarps, KernelParams::MaxNumTopExperts,
            /*LoadExpertIdxFromGlobal=*/false>(params, smemPackedScoreIdx, warpIdx, clusterBlockRank);
    }
    else
    {
        routingPermutation<KernelParams, BaseType, NumThreads, NumWarps, KernelParams::MaxNumTopExperts,
            /*LoadExpertIdxFromGlobal=*/true>(params, smemPackedScoreIdx, warpIdx, clusterBlockRank);
    }
}
#else
__global__ void __launch_bounds__(NumThreads) routingIndicesClusterKernel(KernelParams /* params */)
{
    assert(false && "routingIndicesClusterKernel is only supported on SM90+ architectures");
}
#endif // if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))

void launchClusterKernel(Data const& data, void* stream)
{
    LAUNCH_ROUTING_CUSTOM(data, false, routingIndicesClusterKernel, NumBlocksPerCluster, NumThreads,
        /*smemSize=*/0, // No dynamic smem
        stream);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// 3. HistogramScores kernel — computes TopK from raw scores and initializes expert counts.
//    Used as step 1 of the multi-kernel pipeline when input is raw logits.
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KernelParams>
__global__ void __launch_bounds__(KernelParams::MaxNumExperts <= 1024 ? KernelParams::MaxNumExperts : 1024)
    routingIndicesHistogramScoresKernel(KernelParams params)
{
    using OutputT = typename KernelParams::OutputT;
    using InputT = typename KernelParams::InputT;
    using BaseType = typename KernelParams::ExpertSelectPolicy::template BaseType<InputT>;
    // Cap actual thread count at 1024 when MaxNumExperts > 1024.
    static constexpr int NumThreadsBlock = KernelParams::MaxNumExperts <= 1024 ? KernelParams::MaxNumExperts : 1024;

    // VecSize stays based on MaxNumExperts — each warp still processes all experts for one token.
    static constexpr int VecSize = KernelParams::MaxNumExperts / WarpSize;

    int32_t const laneIdx = cutlass::arch::LaneId();
    int32_t const warpIdx = threadIdx.x / WarpSize;
    // Use NumThreadsBlock (actual thread count) for grid-stride warp/thread addressing
    int32_t const globalWarpIdx = blockIdx.x * NumThreadsBlock / WarpSize + warpIdx;
    int32_t const globalWarpStride = gridDim.x * NumThreadsBlock / WarpSize;
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<WarpSize>(block);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    // Wait on primary grid.
    if (params.mUsePdl)
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
    BaseType warpTopKScore[KernelParams::MaxNumTopExperts];
    int32_t warpTopKExpertIdx[KernelParams::MaxNumTopExperts];
    for (int tokenIdx = globalWarpIdx; tokenIdx < params.mNumTokens; tokenIdx += globalWarpStride)
    {
        auto scoreOffset = tokenIdx * params.mNumExperts;

        KernelParams::ExpertSelectPolicy::template apply<BaseType, InputT, VecSize, KernelParams::MaxNumTopExperts>(
            warp, warpTopKScore, warpTopKExpertIdx, laneIdx, params.mNumExperts, params.mTopK,
            params.mPtrScores + scoreOffset, params);

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
    if (params.mUsePdl)
    {
        cudaTriggerProgrammaticLaunchCompletion();
    }
#endif // if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
}

static void launchHistogramScoresKernel(Data const& data, uint32_t maxNumBlocks, uint32_t numThreadsHist, void* stream)
{
    LAUNCH_ROUTING_CUSTOM(data, false, routingIndicesHistogramScoresKernel, maxNumBlocks, numThreadsHist,
        /*smemSize=*/0, // No dynamic smem
        stream);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// 4. Coop kernel — cooperative histogram + offsets via grid-sync.
//
////////////////////////////////////////////////////////////////////////////////////////////////////

void launchCoopKernel(Data const& data, int numBlocksCoop, uint32_t numThreadsHist, void* stream)
{
    if (data.mNumExperts <= NumExperts128Experts)
    {
        LAUNCH_ROUTING_WITH_POLICIES(data, /*coopLaunch=*/true, routingIndicesCoopKernel, numBlocksCoop, numThreadsHist,
            /*smemSize=*/0, stream, NoOpPreprocess, NoOpPostprocess, NumExperts128Experts, NumTop8Experts);
    }
    else if (data.mNumExperts <= NumExperts160Experts)
    {
        LAUNCH_ROUTING_WITH_POLICIES(data, /*coopLaunch=*/true, routingIndicesCoopKernel, numBlocksCoop, numThreadsHist,
            /*smemSize=*/0, stream, NoOpPreprocess, NoOpPostprocess, NumExperts160Experts, NumTop8Experts);
    }
    else if (data.mNumExperts <= NumExperts256Experts)
    {
        LAUNCH_ROUTING_WITH_POLICIES(data, /*coopLaunch=*/true, routingIndicesCoopKernel, numBlocksCoop, numThreadsHist,
            /*smemSize=*/0, stream, NoOpPreprocess, NoOpPostprocess, NumExperts256Experts, NumTop8Experts);
    }
    else if (data.mNumExperts <= NumExperts384Experts)
    {
        LAUNCH_ROUTING_WITH_POLICIES(data, /*coopLaunch=*/true, routingIndicesCoopKernel, numBlocksCoop, numThreadsHist,
            /*smemSize=*/0, stream, NoOpPreprocess, NoOpPostprocess, NumExperts384Experts, NumTop8Experts);
    }
    else if (data.mNumExperts <= NumExperts512Experts)
    {
        LAUNCH_ROUTING_WITH_POLICIES(data, /*coopLaunch=*/true, routingIndicesCoopKernel, numBlocksCoop, numThreadsHist,
            /*smemSize=*/0, stream, NoOpPreprocess, NoOpPostprocess, NumExperts512Experts, NumTop8Experts);
    }
    else if (data.mNumExperts <= NumExperts576Experts)
    {
        LAUNCH_ROUTING_WITH_POLICIES(data, /*coopLaunch=*/true, routingIndicesCoopKernel, numBlocksCoop, numThreadsHist,
            /*smemSize=*/0, stream, NoOpPreprocess, NoOpPostprocess, NumExperts576Experts, NumTop8Experts);
    }
    else
    {
        TLLM_LOG_ERROR("Coop kernel does not support numExperts > %d", NumExperts576Experts);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// 5-7. Launch wrappers for shared kernels (defined in RoutingKernel.cuh):
//      - InitExpertCounts (zero expert counts)
//      - Histogram kernel (histogram from packed TopK)
//      - Offsets kernel (prefix-scan + permutation)
//
////////////////////////////////////////////////////////////////////////////////////////////////////

void launchInitExpertCounts(Data const& data, uint32_t numThreadsHist, void* stream)
{
    LAUNCH_ROUTING_CUSTOM_NO_POLICY(data, false, routingInitExpertCounts,
        (2 * data.mNumExperts - 1) / numThreadsHist + 1, numThreadsHist,
        /*smemSize=*/0, // No dynamic smem
        stream);
}

void launchHistogramKernel(Data const& data, int numBlocksHistogram, uint32_t numThreadsHist, void* stream)
{
    LAUNCH_ROUTING_CUSTOM_NO_POLICY(data, false, routingIndicesHistogramKernel, numBlocksHistogram, numThreadsHist,
        /*smemSize=*/0, // No dynamic smem
        stream);
}

void launchOffsetsKernel(Data const& data, int numBlocksOffsets, uint32_t numThreadsHist, void* stream)
{
    LAUNCH_ROUTING_CUSTOM_NO_POLICY(data, false, routingIndicesOffsetsKernel, numBlocksOffsets, numThreadsHist,
        /*smemSize=*/0, // No dynamic smem
        stream);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Entry point
//
////////////////////////////////////////////////////////////////////////////////////////////////////

void run(Data const& data, void* stream)
{
    TLLM_CHECK_WITH_INFO(data.mPtrTopKPacked != nullptr || data.mPtrScores != nullptr || data.mPtrTopKIds != nullptr,
        "Routing kernel requires at least one input parameter");

    // When topK is already computed (mPtrTopKIds or mPtrTopKPacked without scores),
    // delegate to the shared post-topK pipeline which handles all path selection
    // (single-block, single-cluster, coop, multi-kernel) automatically.
    // No routing-method-specific logic needed.
    if (data.mPtrTopKIds != nullptr || (data.mPtrTopKPacked != nullptr && data.mPtrScores == nullptr))
    {
        if (data.mPtrTopKIds != nullptr)
        {
            TLLM_CHECK_WITH_INFO(data.mPtrTopKWeights != nullptr,
                "When mPtrTopKIds is provided, mPtrTopKWeights must also be provided for custom routing.");
        }
        uint32_t const numThreadsHist = min(1024, getMaxNumExperts(data.mNumExperts));
        runPostTopKPipeline(data, numThreadsHist, stream);
        return;
    }

    // After this point, input is mPtrScores (raw logits that need topK computation).
    TLLM_CHECK_WITH_INFO(data.mPtrScores != nullptr, "Expected mPtrScores to be non-null at this point.");
    TLLM_CHECK_WITH_INFO(data.mPtrPermutedIdxSize != nullptr && data.mPtrCtaIdxXyToBatchIdx != nullptr
            && data.mPtrCtaIdxXyToMnLimit != nullptr && data.mPtrNumNonExitingCtas != nullptr,
        "Custom routing kernel expects permuted idx and grouped Gemm launch config buffers");
    TLLM_CHECK_WITH_INFO(data.mTopK <= MaxSupportedTopExperts, "Routing kernel expects topK experts <= %d, got %d",
        MaxSupportedTopExperts, data.mTopK);
    TLLM_CHECK_WITH_INFO(data.mNumExperts <= MaxSupportedExperts,
        "Routing kernel expects #experts %d to be no more than %d", data.mNumExperts, MaxSupportedExperts);
    TLLM_CHECK_WITH_INFO(
        data.mNumExperts % 4 == 0, "Routing kernel expects #experts %d to be a multiple of 4.", data.mNumExperts);

    static int const smMajor = tensorrt_llm::common::getSMVersion() / 10;

    bool const useStaticBlock = data.mNumTokens <= BlockKernelMaxNumTokens;
    bool const useDynBlock = !useStaticBlock && data.mNumTokens <= DynBlockKernelMaxNumTokens
        && data.mNumExperts <= DynBlockKernelMaxNumExperts;
    bool const useSingleBlock = useStaticBlock || useDynBlock;
    bool const useSingleCluster = (smMajor >= 9) && (data.mNumTokens <= MaxNumTokensSingleClusterScores);

    if (!useSingleCluster && !useSingleBlock)
    {
        TLLM_CHECK_WITH_INFO(
            data.mPtrTopKPacked != nullptr, "When #tokens is large, `mPtrTopKPacked` is a required input.");
        TLLM_CHECK_WITH_INFO(
            data.mPtrExpertCounts != nullptr, "When #tokens is large, `mPtrExpertCounts` is a required input.");
    }

    uint32_t const numThreadsHist = min(1024, getMaxNumExperts(data.mNumExperts));

    // Last routing kernel: disable programmaticStreamSerializationAllowed so GEMM waits.
    Data lastKernelData = data;
    lastKernelData.mPdlAllowOverlap = false;

    if (useDynBlock)
    {
        launchDynBlockKernel(lastKernelData, numThreadsHist, stream);
    }
    else if (useStaticBlock)
    {
        launchBlockKernel(lastKernelData, numThreadsHist, stream);
    }
    else if (useSingleCluster)
    {
        launchClusterKernel(lastKernelData, stream);
    }
    else
    {
        uint32_t const maxNumBlocks = 1024;

        launchHistogramScoresKernel(data, maxNumBlocks, numThreadsHist, stream);

        bool const canUseCoop = (smMajor >= 9) && (data.mNumExperts <= 1024) && (data.mPtrPermutedIdxSize != nullptr);
        bool useCoop = false;
        int numBlocksCoop = 0;

        if (canUseCoop)
        {
            static int const smCount = tensorrt_llm::common::getMultiProcessorCount();
            numBlocksCoop = smCount - 8;
            int const maxTokensCoop = (numBlocksCoop * numThreadsHist * 64) / data.mTopK;
            useCoop = (data.mNumTokens <= maxTokensCoop);
        }

        if (useCoop)
        {
            launchInitExpertCounts(data, numThreadsHist, stream);
            launchCoopKernel(lastKernelData, numBlocksCoop, numThreadsHist, stream);
        }
        else
        {
            uint32_t const expandedIdxSize = data.mNumTokens * data.mTopK;
            uint32_t const histogramEltsPerBlock = 8 * numThreadsHist;
            uint32_t const offsetEltsPerBlock = NumEltsPerOffsetTilePerThread * numThreadsHist;

            int const numBlocksHistogram
                = std::min((expandedIdxSize + histogramEltsPerBlock - 1) / histogramEltsPerBlock, maxNumBlocks);
            int const numBlocksOffsets
                = std::min((expandedIdxSize + offsetEltsPerBlock - 1) / offsetEltsPerBlock, maxNumBlocks);

            launchHistogramKernel(data, numBlocksHistogram, numThreadsHist, stream);
            launchOffsetsKernel(lastKernelData, numBlocksOffsets, numThreadsHist, stream);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace routingCustom
} // namespace moe::dev::routing
