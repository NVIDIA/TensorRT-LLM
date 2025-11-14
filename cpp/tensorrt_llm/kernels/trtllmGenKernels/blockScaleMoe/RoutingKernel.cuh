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

#include "DevKernel.h"
#include "RoutingKernel.h"
#include "RoutingKernelTopK.cuh"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/cub.cuh>

#include <cute/arch/cluster_sm90.hpp>
#include <cutlass/arch/arch.h>

#include <type_traits>

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace moe::dev
{

////////////////////////////////////////////////////////////////////////////////////////////////////
namespace routing
{

namespace cg = cooperative_groups;

////////////////////////////////////////////////////////////////////////////////////////////////////

static constexpr int WarpSize = 32;
static constexpr int NumBlocksPerCluster = 8;
// Performance tuning knob.
static constexpr int NumEltsPerOffsetTilePerThread = 8;

////////////////////////////////////////////////////////////////////////////////////////////////////

static __device__ inline float sigmoid_accurate(float x)
{
    return 0.5f * tanhf(0.5f * x) + 0.5f;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
__host__ __device__ constexpr T mulLog2(T a, T bLog2)
{
    return a << bLog2;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
__host__ __device__ constexpr T divUpLog2(T a, T bLog2)
{
    return ((a + (1 << bLog2) - 1) >> bLog2);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
__host__ __device__ constexpr T divUpMulLog2(T a, T bLog2)
{
    return mulLog2<T>(divUpLog2<T>(a, bLog2), bLog2);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
__host__ __device__ constexpr T mulTileN(T a, T tileN)
{
    return a * tileN;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
__host__ __device__ constexpr T divUpTileN(T a, T tileN)
{
    return (a + tileN - 1) / tileN;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
__host__ __device__ constexpr T divUpMulTileN(T a, T tileN)
{
    return divUpTileN(a, tileN) * tileN;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__host__ __device__ constexpr int32_t getBits(int32_t value, int idx)
{
    int mask = idx == 0 ? 0x000000FF : idx == 1 ? 0x0000FF00 : idx == 2 ? 0x00FF0000 : 0xFF000000;
    return (value & mask) >> (idx * 8);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool IsZero = false>
__host__ __device__ constexpr void setBits(int32_t& value, int32_t newBits, int idx)
{
    if constexpr (!IsZero)
    {
        int mask = idx == 0 ? 0xFFFFFF00 : idx == 1 ? 0xFFFF00FF : idx == 2 ? 0xFF00FFFF : 0x00FFFFFF;
        value &= mask;
    }
    value |= (newBits << (idx * 8));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename DataType>
__device__ void initArr(int startIdx, int numElts, int stride, DataType* arr, DataType value)
{
    if (arr != nullptr)
    {
        for (int i = startIdx; i < numElts; i += stride)
        {
            arr[i] = value;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename DataType, int VecSize>
__device__ void calcSoftmax(cg::thread_block_tile<WarpSize> const& warp, DataType (&scores)[VecSize])
{
    // Compute in float to support half/bfloat16 inputs safely.
    float maxScore = -INFINITY;
    float sumScore = 0.f;
    // Get the max score for each token
#pragma unroll
    for (int i = 0; i < VecSize; ++i)
    {
        float si = static_cast<float>(scores[i]);
        maxScore = si >= maxScore ? si : maxScore;
    }
    maxScore = cg::reduce(warp, maxScore, cg::greater<float>());

    // Get the summation of scores for each token
#pragma unroll
    for (int i = 0; i < VecSize; ++i)
    {
        float si = static_cast<float>(scores[i]);
        float e = expf(si - maxScore);
        scores[i] = static_cast<DataType>(e);
        sumScore += e;
    }
    sumScore = cg::reduce(warp, sumScore, cg::plus<float>());

    // Normalize the scores
#pragma unroll
    for (int i = 0; i < VecSize; ++i)
    {
        float si = static_cast<float>(scores[i]) / sumScore;
        scores[i] = static_cast<DataType>(si);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename DataType>
__device__ DataType calcSoftmax(
    cg::thread_block_tile<WarpSize> const& warp, DataType score, int32_t laneIdx, int32_t NumTopExperts)
{
    DataType maxScore = DataType{-INFINITY};
    if (laneIdx < NumTopExperts)
    {
        maxScore = score >= maxScore ? score : maxScore;
    }
    maxScore = cg::reduce(warp, maxScore, cg::greater<DataType>());

    float sumScore = float{0.f};
    float newScore;
    // Get the summation of scores for each token
    if (laneIdx < NumTopExperts)
    {
        newScore = static_cast<float>(score) - static_cast<float>(maxScore);
        newScore = static_cast<float>(exp(newScore));
        sumScore += newScore;
    }
    sumScore = cg::reduce(warp, sumScore, cg::plus<float>());

    if (laneIdx < NumTopExperts)
    {
        score = static_cast<DataType>(newScore / sumScore);
    }

    return score;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename KernelParams, typename BaseType, int NumThreads, int NumWarps, int MaxNumTopExperts,
    bool LoadExpertIdxFromGlobal = false>
__device__ void routingPermutation(KernelParams params, PackedScoreIdx<BaseType>* smemPackedScoreIdx,
    int32_t const warpIdx, uint32_t const clusterBlockRank)
{

    using OutputT = typename KernelParams::OutputT;
    using TypePacked = PackedScoreIdx<BaseType>;

    static constexpr int MaxNumTokensSingleCluster = NumBlocksPerCluster * NumThreads;
    // Number of threads in the cluster.
    static constexpr int NumThreadsPerCluster = NumThreads * NumBlocksPerCluster;
    // same as max num tokens
    static constexpr int MaxExpandedIdxPerThread
        = (MaxNumTokensSingleCluster * MaxNumTopExperts + NumThreadsPerCluster - 1) / NumThreadsPerCluster;

    // Needed for the exclusive sum of token offsets.
    // Note: the scan might include more bins than needed, with bin counts of 0 to pad
    using Scan = cub::BlockScan<int32_t, NumThreads, cub::BLOCK_SCAN_WARP_SCANS>;
    __shared__ typename Scan::TempStorage tempStorage;

    uint32_t const clusterThreadIdx = NumThreads * clusterBlockRank + threadIdx.x;
    auto expandedIdxSize = params.mNumTokens * params.mTopK;

    // number of experts is bounded by number of threads
    __shared__ int32_t __attribute((aligned(128))) smemExpertCount[NumThreads];
    __shared__ int32_t __attribute((aligned(128))) smemExpertOffset[NumThreads];

    // pre-fill the counts with 0
    if (threadIdx.x < params.mNumExperts)
    {
        smemExpertCount[threadIdx.x] = 0;
    }
    __syncthreads();

    // each thread keeps some number of "expanded indexes" assigned to it
    // note that expanded indexes simply represent tokens here.
    // for each of these, we keep the associated expert and offset within expert in registers
    int32_t expertIndexes[MaxExpandedIdxPerThread];
    int32_t expertOffsets[MaxExpandedIdxPerThread];
    auto localExpertExtent = params.mNumLocalExperts << params.mLocalExpertsStrideLog2;

    // In order to avoid a serialization LDG-ATOMS-LDG-ATOMS-..., we skip multiple iterations at a
    // time, and branch between a fast path without bound checks and a slow path with bound checks.
    // TODO(mjoux): potentially add this back for perf tuning
    // int constexpr IterStride = 4;
    // static_assert(MaxExpandedIdxPerThread % IterStride == 0);

    // Define a lambda to avoid code duplication in both branches.
    auto loopBody = [&](int ii, int expandedIdx)
    {
        TypePacked scoreIdx;
        if constexpr (LoadExpertIdxFromGlobal)
        {
            if (params.mPtrTopKIds != nullptr)
            {
                scoreIdx = TypePacked{static_cast<BaseType>(params.mPtrTopKWeights[expandedIdx]),
                    static_cast<int16_t>(params.mPtrTopKIds[expandedIdx])};
            }
            else
            {
                scoreIdx = TypePacked{static_cast<BaseType>(params.mPtrTopKPacked[expandedIdx].score),
                    static_cast<int16_t>(params.mPtrTopKPacked[expandedIdx].idx)};
            }
        }
        else
        {
            TypePacked const* remoteSmem
                = cg::cluster_group::map_shared_rank(smemPackedScoreIdx, expandedIdx / (NumWarps * params.mTopK));
            scoreIdx = remoteSmem[expandedIdx % (NumWarps * params.mTopK)];
        }

        expertIndexes[ii] = scoreIdx.idx;
        // check whether this expert is local to our GPU at all and ignore if not
        auto localExpertIdx = scoreIdx.idx - params.mLocalExpertsStartIdx;
        auto isLocalExpert = localExpertIdx >= 0 && localExpertIdx < localExpertExtent
            && (localExpertIdx & params.mLocalExpertsStrideLog2) == 0;
        expertOffsets[ii] = isLocalExpert ? atomicAdd(smemExpertCount + scoreIdx.idx, 1) : 0;
        if (params.mPtrTopKWeights != nullptr && params.mPtrTopKIds == nullptr)
        {
            params.mPtrTopKWeights[expandedIdx] = OutputT{scoreIdx.score};
        }
    };

    int constexpr IterStride = 4;
#pragma unroll
    for (int32_t ii0 = 0; ii0 < MaxExpandedIdxPerThread; ii0 += IterStride)
    {
        // Whether it's safe to do multiple iterations without bound checks.
        bool const takeFastPath = (ii0 + IterStride) * NumThreadsPerCluster <= expandedIdxSize;
        if (takeFastPath)
        {
#pragma unroll
            for (int32_t jj = 0; jj < IterStride; jj++)
            {
                int const ii = ii0 + jj;
                auto expandedIdx = static_cast<int32_t>(clusterThreadIdx) + ii * NumThreadsPerCluster;
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
                auto expandedIdx = static_cast<int32_t>(clusterThreadIdx) + ii * NumThreadsPerCluster;
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
    // Make local histogram (token counts per expert) available to all threads in the cluster.
    __cluster_barrier_arrive();
    __cluster_barrier_wait();

    //
    // Each thread now represents one expert
    //

    // Total number of tokens for this expert.
    int32_t count = 0;
    // Per-expert offset for this block.
    int32_t blockExpertOffset = 0;

    if (threadIdx.x < params.mNumExperts)
    {
        // Get the histogram bin from each rank for this expert.
        int32_t expertCounts[NumBlocksPerCluster];
#pragma unroll
        for (int rank = 0; rank < NumBlocksPerCluster; rank++)
        {
            int32_t const* remoteSmem = cg::cluster_group::map_shared_rank(smemExpertCount, rank);
            expertCounts[rank] = rank * NumWarps < params.mNumTokens ? remoteSmem[threadIdx.x] : 0;
        }

        // Compute an exclusive prefix sum of the block-local count.
#pragma unroll
        for (int rank = 0; rank < NumBlocksPerCluster; rank++)
        {
            if (rank == clusterBlockRank)
            {
                blockExpertOffset = count;
            }
            count += expertCounts[rank];
        }
    }

    // Arrive: we do not access distributed shared memory after this point.
    __cluster_barrier_arrive();

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

    if (threadIdx.x < params.mNumExperts)
    {
        // Strided loop to share this work between blocks.
        for (int32_t cta = clusterBlockRank; cta < numCta; cta += NumBlocksPerCluster)
        {
            const int32_t localExpertIdx
                = (threadIdx.x - params.mLocalExpertsStartIdx) >> params.mLocalExpertsStrideLog2;
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

        // write expert offsets to shared
        smemExpertOffset[threadIdx.x] = offset + blockExpertOffset;
    }

    // write out padded count
    if (clusterBlockRank == 0 && warpIdx == NumWarps - 1 && cute::elect_one_sync())
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

    // make expert offsets available to all threads
    __syncthreads();

    // Wait: we cannot exit while other blocks may be accessing the current block's shared memory.
    // Note: I observed a perf benefit to doing this before the final loop so the compiler can
    // implement break with EXIT.
    __cluster_barrier_wait();

    // trigger the secondary kernel when using PDL
    // We can't do it earlier because FC1 depends on the mPtrCtaIdxXyToBatchIdx,
    // mPtrCtaIdxXyToMnLimit, mPtrNumNonExitingCtas and mPtrTotalNumPaddedTokens
    // TODO: this is not sufficient to ensure visibility in the next kernel!
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    if constexpr (KernelParams::UsePdl)
    {
        cudaTriggerProgrammaticLaunchCompletion();
    }
#endif

    // each thread has the same "expanded indexes" assigned to it as above
    // at this point, we know the final offsets of experts and the offsets within
    // experts, which allows writing the final index values

#pragma unroll
    for (int32_t ii = 0; ii < MaxExpandedIdxPerThread; ++ii)
    {
        auto expandedIdx = static_cast<int32_t>(clusterThreadIdx) + ii * NumThreadsPerCluster;
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
        if (params.mPtrPermutedIdxToTokenIdx != nullptr && isLocalExpert)
        {
            params.mPtrPermutedIdxToTokenIdx[permutedIdx] = tokenIdx;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Two-step approach (if number of tokens exceed limits of what cluster / cooperative launch
// variants can handle): in order to minimize the amount of data to exchange through global memory,
// we will compute the local histograms in smem twice: the first kernel will get us the total number
// of tokens per expert. The second kernel will use the smem and L2 atomics to get corresponding
// element and tile offsets.
//
// Note: the histogram calculation could also be fused with routingMainKernel, but this might be
// inefficient if we have one CTA per token doing a single global atomic.
template <typename KernelParams>
__global__ void __launch_bounds__(KernelParams::MaxNumExperts) routingIndicesHistogramKernel(KernelParams params)
{
    using OutputT = typename KernelParams::OutputT;

    // number of experts is bounded by number of threads
    __shared__ int32_t __attribute((aligned(128))) smemExpertCount[KernelParams::MaxNumExperts];

    // For unrolling.
    uint32_t constexpr NumEltsPerThread = 8;

    // Pre-fill the counts with 0
    if (threadIdx.x < params.mNumExperts)
    {
        smemExpertCount[threadIdx.x] = 0;
    }
    __syncthreads();

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    // Wait on primary grid and trigger secondary kernel.
    if constexpr (KernelParams::UsePdl)
    {
        cudaGridDependencySynchronize();
        cudaTriggerProgrammaticLaunchCompletion();
    }
#endif // if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))

    uint32_t const expandedIdxSize = params.mNumTokens * params.mTopK;
    uint32_t const localExpertExtent = params.mNumLocalExperts << params.mLocalExpertsStrideLog2;

    uint32_t const gridBlockOffset = blockIdx.x * KernelParams::MaxNumExperts;
    uint32_t const gridStride = gridDim.x * KernelParams::MaxNumExperts;

    // Define a lambda to avoid code duplication in branches.
    auto loopBody = [&](int expandedIdx)
    {
        PackedScoreIdx<OutputT> scoreIdx;
        int idx;
        if (params.mPtrTopKIds != nullptr)
        {
            idx = params.mPtrTopKIds[expandedIdx];
        }
        else
        {
            // If params.mPtrTopKIds != nullptr, we don't need to store the weights
            if (params.mPtrTopKWeights != nullptr)
            {
                scoreIdx = params.mPtrTopKPacked[expandedIdx];
                idx = scoreIdx.idx;
                params.mPtrTopKWeights[expandedIdx] = static_cast<OutputT>(scoreIdx.score);
            }
        }
        // check whether this expert is local to our GPU at all and ignore if not
        auto localExpertIdx = idx - params.mLocalExpertsStartIdx;
        auto isLocalExpert = localExpertIdx >= 0 && localExpertIdx < localExpertExtent
            && (localExpertIdx & params.mLocalExpertsStrideLog2) == 0;
        if (isLocalExpert)
        {
            atomicAdd(&smemExpertCount[idx], 1);
        }
    };

    // Grid-stride loop.
    for (uint32_t expandedIdx0 = gridBlockOffset * NumEltsPerThread; expandedIdx0 < expandedIdxSize;
         expandedIdx0 += gridStride * NumEltsPerThread)
    {
        // Fast path if bound checks aren't necessary
        if (expandedIdx0 + NumEltsPerThread * KernelParams::MaxNumExperts <= expandedIdxSize)
        {
#pragma unroll
            for (uint32_t ii = 0; ii < NumEltsPerThread; ii++)
            {
                uint32_t expandedIdx = expandedIdx0 + ii * KernelParams::MaxNumExperts + threadIdx.x;
                loopBody(expandedIdx);
            }
        }
        else
        {
            for (uint32_t expandedIdx = expandedIdx0 + threadIdx.x; expandedIdx < expandedIdxSize;
                 expandedIdx += KernelParams::MaxNumExperts)
            {
                loopBody(expandedIdx);
            }
        }
    }
    __syncthreads();

    //
    // Each thread now represents one expert
    //

    // Reduce histograms with atomics.
    if (threadIdx.x < params.mNumExperts)
    {
        int32_t const localExpertCount = smemExpertCount[threadIdx.x];
        atomicAdd(&params.mPtrExpertCounts[threadIdx.x], localExpertCount);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KernelParams>
__global__ void __launch_bounds__(KernelParams::MaxNumExperts) routingIndicesOffsetsKernel(KernelParams params)
{
    using OutputT = typename KernelParams::OutputT;

    // number of experts is bounded by number of threads
    __shared__ int32_t __attribute((aligned(128))) smemExpertOffset[KernelParams::MaxNumExperts];
    __shared__ int32_t __attribute((aligned(128))) smemExpertCount[KernelParams::MaxNumExperts];
    __shared__ int32_t __attribute((aligned(128))) smemExpertTileOffset[KernelParams::MaxNumExperts];
    // needed for the exclusive sum of token offsets
    using Scan = cub::BlockScan<int32_t, KernelParams::MaxNumExperts, cub::BLOCK_SCAN_WARP_SCANS>;
    __shared__ typename Scan::TempStorage tempStorage;
    static constexpr int MaxExpandedIdxPerThread = NumEltsPerOffsetTilePerThread;
    static constexpr int MaxExpandedIdxPerBlock = KernelParams::MaxNumExperts * MaxExpandedIdxPerThread;

    int32_t const warpIdx = __shfl_sync(0xffffffff, threadIdx.x / WarpSize, 0);

    uint32_t const expandedIdxSize = params.mNumTokens * params.mTopK;
    uint32_t const numTiles = (expandedIdxSize + MaxExpandedIdxPerBlock - 1) / (MaxExpandedIdxPerBlock);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    // Wait on primary grid.
    if constexpr (KernelParams::UsePdl)
    {
        cudaGridDependencySynchronize();
    }
#endif // if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))

    // The expert offsets are common to all tiles of all blocks.
    // Load the histogram, scan it and write offsets to shared memory.
    // Note: the scan is redundant in all CTAs. Would it make sense to use an intermediate kernel for
    // the scan, with PDL?

    //
    // Each thread represents one expert.
    //

    // Get total count for this expert.
    int32_t count = (threadIdx.x < params.mNumExperts) ? params.mPtrExpertCounts[threadIdx.x] : 0;

    // Compute the runtime config for projections
    // Whether or not an expert is local is taken into account when the histogram is computed
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

    if (threadIdx.x < params.mNumExperts)
    {
        // Get the padded offset associated with this expert
        int32_t offset;
        if constexpr (KernelParams::isPow2)
        {
            offset = mulLog2<int32_t>(ctaOffset, params.mPaddingLog2);
        }
        else
        {
            offset = mulTileN<int32_t>(ctaOffset, params.mTileTokensDim);
        }

        // Write expert offsets to shared
        smemExpertOffset[threadIdx.x] = offset;
    }

    // Sync to make expert offsets available to all threads.
    __syncthreads();

    // The first block writes out padded count
    if (blockIdx.x == 0 && warpIdx == KernelParams::MaxNumExperts / WarpSize - 1 && cute::elect_one_sync())
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

    if (threadIdx.x < params.mNumExperts)
    {
        // Strided loop to share this work between blocks.
        for (int32_t cta = blockIdx.x; cta < numCta; cta += gridDim.x)
        {
            const int32_t localExpertIdx
                = (threadIdx.x - params.mLocalExpertsStartIdx) >> params.mLocalExpertsStrideLog2;
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
    }

    //
    // Now loop on indices and compute offsets.
    //

    // Grid-stride loop on 1D "tiles" of input indices.
    for (uint32_t tileIdx = blockIdx.x; tileIdx < numTiles; tileIdx += gridDim.x)
    {
        if (tileIdx > 0)
        {
            // Sync for safe reuse of smem buffers.
            __syncthreads();
        }

        // Pre-fill the counts with 0
        if (threadIdx.x < params.mNumExperts)
        {
            smemExpertCount[threadIdx.x] = 0;
        }
        __syncthreads();

        // each thread keeps has some number of "expanded indexes" assigned to it
        // for each of these, we keep the associated expert and offset within expert in registers
        int32_t expertIndexes[MaxExpandedIdxPerThread];
        int32_t expertOffsets[MaxExpandedIdxPerThread];
        auto localExpertExtent = params.mNumLocalExperts << params.mLocalExpertsStrideLog2;

        // Define a lambda to avoid code duplication in branches.
        auto loopBody = [&](int ii, int expandedIdx)
        {
            expertIndexes[ii]
                = params.mPtrTopKIds ? params.mPtrTopKIds[expandedIdx] : params.mPtrTopKPacked[expandedIdx].idx;
            // check whether this expert is local to our GPU at all and ignore if not
            auto localExpertIdx = expertIndexes[ii] - params.mLocalExpertsStartIdx;
            auto isLocalExpert = localExpertIdx >= 0 && localExpertIdx < localExpertExtent
                && (localExpertIdx & params.mLocalExpertsStrideLog2) == 0;
            expertOffsets[ii] = isLocalExpert ? atomicAdd(smemExpertCount + expertIndexes[ii], 1) : 0;
        };

        // For all tiles but the last, all indices are in bounds.
        if (tileIdx < numTiles - 1)
        {
#pragma unroll
            for (int32_t ii = 0; ii < MaxExpandedIdxPerThread; ii += 1)
            {
                auto expandedIdx = tileIdx * MaxExpandedIdxPerBlock + ii * KernelParams::MaxNumExperts + threadIdx.x;
                loopBody(ii, expandedIdx);
            }
        }
        else
        {
            // For the last tile, we need to exit the loop when out of bounds.
            // In order to avoid a serialization LDG-ATOMS-LDG-ATOMS-..., we skip multiple iterations at a
            // time, and branch between a fast path without bound checks and a slow path with bound checks
            int constexpr IterStride = 4;
            static_assert(MaxExpandedIdxPerThread % IterStride == 0);

#pragma unroll
            for (int32_t ii0 = 0; ii0 < MaxExpandedIdxPerThread; ii0 += IterStride)
            {
                // Whether it's safe to do multiple iterations without bound checks.
                bool const takeFastPath
                    = tileIdx * MaxExpandedIdxPerBlock + (ii0 + IterStride) * KernelParams::MaxNumExperts
                    <= expandedIdxSize;
                if (takeFastPath)
                {
#pragma unroll
                    for (int32_t jj = 0; jj < IterStride; jj++)
                    {
                        int const ii = ii0 + jj;
                        auto expandedIdx
                            = tileIdx * MaxExpandedIdxPerBlock + ii * KernelParams::MaxNumExperts + threadIdx.x;
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
                        auto expandedIdx
                            = tileIdx * MaxExpandedIdxPerBlock + ii * KernelParams::MaxNumExperts + threadIdx.x;
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
        }

        // Make local histogram (token counts per expert) available to all threads in the block.
        __syncthreads();

        //
        // Each thread now represents one expert
        //

        if (threadIdx.x < params.mNumExperts)
        {
            // Add the local bin count to the common bin count and get a per-CTA offset. We use the second
            // half of the histogram buffer for this histogram, because the first half already holds the
            // reduced histogram from the previous kernel.
            int32_t const localExpertCount = smemExpertCount[threadIdx.x];
            int32_t const tileExpertOffset
                = atomicAdd(&params.mPtrExpertCounts[params.mNumExperts + threadIdx.x], localExpertCount);

            // Make per-expert tile offsets available to all threads in the block.
            smemExpertTileOffset[threadIdx.x] = tileExpertOffset + smemExpertOffset[threadIdx.x];
        }
        __syncthreads();

        // Add tile offset and element offset and write to global memory.
        auto storeLoopBody = [&](int ii, int expandedIdx)
        {
            int32_t expertIdx = expertIndexes[ii];
            // check whether this expert is local to our GPU at all
            auto localExpertIdx = static_cast<int32_t>(expertIdx) - params.mLocalExpertsStartIdx;
            auto isLocalExpert = localExpertIdx >= 0 && localExpertIdx < localExpertExtent
                && (localExpertIdx & params.mLocalExpertsStrideLog2) == 0;
            auto tokenIdx = expandedIdx / params.mTopK;
            auto permutedIdx = isLocalExpert ? (expertOffsets[ii] + smemExpertTileOffset[expertIdx]) : int32_t{-1};
            if (params.mPtrExpandedIdxToPermutedIdx != nullptr)
            {
                params.mPtrExpandedIdxToPermutedIdx[expandedIdx] = permutedIdx;
            }
            if (params.mPtrPermutedIdxToTokenIdx != nullptr && isLocalExpert)
            {
                params.mPtrPermutedIdxToTokenIdx[permutedIdx] = tokenIdx;
            }
        };
        // Bound checks only in last tile.
        if (tileIdx < numTiles - 1)
        {
#pragma unroll
            for (int32_t ii = 0; ii < MaxExpandedIdxPerThread; ii += 1)
            {
                auto expandedIdx = tileIdx * MaxExpandedIdxPerBlock + ii * KernelParams::MaxNumExperts + threadIdx.x;
                storeLoopBody(ii, expandedIdx);
            }
        }
        else
        {
#pragma unroll
            for (int32_t ii = 0; ii < MaxExpandedIdxPerThread; ii += 1)
            {
                auto expandedIdx = tileIdx * MaxExpandedIdxPerBlock + ii * KernelParams::MaxNumExperts + threadIdx.x;
                if (expandedIdx >= expandedIdxSize)
                {
                    break;
                }
                storeLoopBody(ii, expandedIdx);
            }
        }
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    // Trigger secondary kernel.
    // Note: this does not guarantee the visibility of prior writes unless the consumer executes a
    // dependency sync.
    if constexpr (KernelParams::UsePdl)
    {
        cudaTriggerProgrammaticLaunchCompletion();
    }
#endif // if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KernelParams>
__global__ void __launch_bounds__(KernelParams::MaxNumExperts) routingInitExpertCounts(KernelParams params)
{
    // initialize the mPtrExpertCounts
    int32_t expertCountsNum = 2 * params.mNumExperts;
    int32_t globalThreadIdx = blockIdx.x * KernelParams::MaxNumExperts + threadIdx.x;
    int32_t globalThreadStride = gridDim.x * KernelParams::MaxNumExperts;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    // Wait on primary grid.
    if constexpr (KernelParams::UsePdl)
    {
        cudaGridDependencySynchronize();
    }
#endif // if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))

    initArr(globalThreadIdx, expertCountsNum, globalThreadStride, params.mPtrExpertCounts, 0);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    // Wait on primary grid.
    if constexpr (KernelParams::UsePdl)
    {
        cudaTriggerProgrammaticLaunchCompletion();
    }
#endif // if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
}
} // namespace routing
} // namespace moe::dev
