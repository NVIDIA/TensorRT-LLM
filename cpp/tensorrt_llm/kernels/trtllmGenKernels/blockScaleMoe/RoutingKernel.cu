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

#include "DevKernel.h"
#include "RoutingKernel.h"

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

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace tg = batchedGemm::trtllm::gen;
namespace cg = cooperative_groups;

////////////////////////////////////////////////////////////////////////////////////////////////////

static constexpr int NumThreads = 256;
static constexpr int NumBlocksPerCluster = 8;
static constexpr int WarpSize = 32;
static constexpr int NumWarps = NumThreads / WarpSize;
static constexpr int NumTopGroupScores = 2;
static constexpr int MaxNumTopExperts = 8;
static constexpr int MaxNumTopGroups = 4;

// Performance tuning knob.
static constexpr int NumEltsPerOffsetTilePerThread = 8;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000 && defined(__CUDA_ARCH_FEAT_SM100_ALL))
#define TLLM_GEN_ENABLE_FAST_REDUX
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename TypeExpW_>
struct TopKRedType
{
    using TypeExpW = TypeExpW_;
    static_assert(std::is_same_v<TypeExpW, float> || std::is_same_v<TypeExpW, cutlass::bfloat16_t>,
        "Top K reduction only implemented for float and Bf16");
    using TypeCmp = std::conditional_t<sizeof(TypeExpW) >= 4, double, float>;
    static constexpr int64_t Mask64 = 0x000000000000FFFF;
    static constexpr int32_t Mask32 = 0x0000FFFF;

    TypeCmp compVal;

    static __host__ __device__ inline TypeCmp makeCmpVal(TypeExpW val, int32_t idx = 0)
    {
        auto cmpVal = TypeCmp{val};
        TypeCmp cmpValWithIdx;
        if constexpr (sizeof(TypeExpW) >= 4)
        {
            auto cmpValIdx64 = reinterpret_cast<int64_t&>(cmpVal) | (Mask64& int64_t{idx});
            cmpValWithIdx = reinterpret_cast<TypeCmp&>(cmpValIdx64);
        }
        else
        {
            auto cmpValIdx32 = reinterpret_cast<int32_t&>(cmpVal) | (Mask32 & idx);
            cmpValWithIdx = reinterpret_cast<TypeCmp&>(cmpValIdx32);
        }
        return cmpValWithIdx;
    }

    static __host__ __device__ inline void unpack(TypeExpW& val, int32_t& idx, TypeCmp cmp)
    {
        if constexpr (sizeof(TypeExpW) >= 4)
        {
            idx = static_cast<int32_t>(reinterpret_cast<int64_t&>(cmp) & Mask64);
            auto val64 = reinterpret_cast<int64_t&>(cmp) & ~Mask64;
            val = static_cast<float>(reinterpret_cast<double&>(val64));
        }
        else
        {
            idx = reinterpret_cast<int32_t&>(cmp) & Mask32;
            auto val32 = reinterpret_cast<int32_t&>(cmp) >> 16;
            val = TypeExpW::bitcast(reinterpret_cast<uint16_t&>(val32));
        }
    }

    __host__ __device__ TopKRedType() = default;

    __host__ __device__ TopKRedType(TypeExpW val, int32_t idx)
        : compVal(makeCmpVal(val, idx))
    {
    }

    __host__ __device__ operator TypeCmp() const noexcept
    {
        return compVal;
    }

    __device__ inline TypeCmp reduce(cg::thread_block_tile<WarpSize> const& warp)
    {
#if defined(TLLM_GEN_ENABLE_FAST_REDUX)
        static constexpr bool UseCg = false;
#else
        static constexpr bool UseCg = true;
#endif
        if constexpr (UseCg || sizeof(TypeExpW) >= 4)
        {
            return cg::reduce(warp, compVal, cg::greater<TypeCmp>{});
        }
        else
        {
            float result;
            asm("redux.sync.max.f32 %0, %1, 0xffffffff;\n" : "=f"(result) : "f"(compVal));
            return result;
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

static __device__ inline float sigmoid_accurate(float x)
{
    return 0.5f * tanhf(0.5f * x) + 0.5f;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int K_, bool Enable_>
struct TopKIdx
{
    // by default, empty
};

template <int K_>
struct TopKIdx<K_, true>
{
    static constexpr int K = K_;
    int32_t val[K];
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int K, typename Type>
__device__ void reduceTopK(cg::thread_block_tile<WarpSize> const& warp, Type (&out)[K], int32_t (&outIdx)[K],
    Type value, int32_t idx, Type minValue)
{
    static_assert(K > 0, "Top K must have K > 0");
    static_assert(K < WarpSize, "Top K must have K < WarpSize");
    using RedType = TopKRedType<Type>;
    RedType topK{value, idx};
    typename RedType::TypeCmp packedMax{};
#pragma unroll
    for (int kk = 0; kk < K; ++kk)
    {
        topK = kk > 0 && packedMax == topK.compVal ? RedType{minValue, idx} : topK;
        // get the next largest value
        packedMax = topK.reduce(warp);
        RedType::unpack(out[kk], outIdx[kk], packedMax);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

#define TOPK_SWAP(I, J)                                                                                                \
    {                                                                                                                  \
        auto pairMin = min(topK[I].compVal, topK[J].compVal);                                                          \
        auto pairMax = max(topK[I].compVal, topK[J].compVal);                                                          \
        topK[I].compVal = pairMax;                                                                                     \
        topK[J].compVal = pairMin;                                                                                     \
    }

template <int K, typename Type, int N, bool IsSorted = false>
__device__ void reduceTopK(cg::thread_block_tile<WarpSize> const& warp, Type (&out)[K], int32_t (&outIdx)[K],
    Type (&value)[N], int32_t (&idx)[N], Type minValue)
{
    static_assert(K > 0, "Top K must have K > 0");
    static_assert(K < WarpSize, "Top K must have K < WarpSize");
    static_assert(N > 0, "Top K must have N > 1");
    static_assert(N <= K, "Top K must have N < K");
    using RedType = TopKRedType<Type>;
    RedType topK[N];
#pragma unroll
    for (int nn = 0; nn < N; ++nn)
        topK[nn] = RedType{value[nn], idx[nn]};
    if constexpr (!IsSorted)
    {
        static_assert(N <= 4, "Unsorted topK expects N <= 4");
        TOPK_SWAP(0, 2);
        TOPK_SWAP(1, 3);

        TOPK_SWAP(0, 1);
        TOPK_SWAP(2, 3);

        TOPK_SWAP(1, 2);
    }
    typename RedType::TypeCmp packedMax{};
#pragma unroll
    for (int kk = 0; kk < K; ++kk)
    {
        bool update = kk > 0 && packedMax == topK[0].compVal;
#pragma unroll
        for (int nn = 0; nn < N; ++nn)
        {
            topK[nn] = update && nn == N - 1 ? RedType{minValue, idx[nn]} : update ? topK[nn + 1] : topK[nn];
        }
        // get the next largest value
        packedMax = topK[0].reduce(warp);
        RedType::unpack(out[kk], outIdx[kk], packedMax);
    }
};

#undef TOPK_SWAP

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

template <typename KernelParams>
__global__ void routingMainKernel(KernelParams params)
{
    // declare types required for reductions
    using TypeExpW = typename KernelParams::TypeExpW;

    // declare shared memory structure
    // number of experts is bounded by number of threads
    __shared__ float __attribute((aligned(128))) smemScoreSigmoid[NumThreads];
    __shared__ float __attribute((aligned(128))) smemScoreBias[NumThreads];
    // number of expert groups is bounded by number of warps
    __shared__ float __attribute((aligned(128))) smemGroupScores[NumWarps];

    // needed for warp reduce
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<WarpSize>(block);
    // for the final reduction of weight norm, only some lanes need to participate
    int32_t laneIdx = threadIdx.x % WarpSize;
    int32_t warpIdx = __shfl_sync(0xffffffff, threadIdx.x / WarpSize, 0);
    // warps outside the range of expert groups do not participate
    if constexpr (KernelParams::UseGroups)
    {
        if (warpIdx >= params.mNumExpertGroups)
        {
            return;
        }
    }

    // note that for invalid scores, we simply use a negative value:
    // they work well even with the compacted format used in topK, and
    // sigmoid / bias activated scores cannot be negative
    static constexpr float invalidScoreFloat = -1.F;
    const TypeExpW invalidScore = TypeExpW{invalidScoreFloat};

    // load bias already; each warp represents one expert group
    auto threadExpert = threadIdx.x;
    bool expertSelected = threadExpert < params.mNumExperts;
    if constexpr (KernelParams::UseGroups)
    {
        threadExpert = warpIdx * params.mNumExpertsPerGroup + laneIdx;
        expertSelected = laneIdx < params.mNumExpertsPerGroup;
    }
    auto scoreIdx = int64_t{blockIdx.x} * int64_t{params.mNumExperts} + threadExpert;
    auto biasVal = expertSelected ? params.mPtrRoutingBias[threadExpert] : invalidScore;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    // trigger the secondary kernel when using PDL, then wait on primary
    if constexpr (KernelParams::UsePdl)
    {
        cudaTriggerProgrammaticLaunchCompletion();
        cudaGridDependencySynchronize();
    }
#endif

    // get our assigned thread score; each warp represents one expert group
    float score = expertSelected ? params.mPtrScores[scoreIdx] : invalidScoreFloat;
    // get the sigmoid score
    // note that for invalid values, we simply use a negative value:
    // sigmoig scores are always strictly positive
    auto scoreSigmoid = sigmoid_accurate(score);
    // write the sigmoid score to shared for later use
    if (expertSelected)
    {
        smemScoreSigmoid[threadExpert] = scoreSigmoid;
    }
    // get the score with bias
    // note that with invalid values, because sigmoid is < 1 and bias is -1,
    // we must get a negative value, which is smaller than any valid value
    auto scoreBias = float{scoreSigmoid + float{biasVal}};
    if (expertSelected)
    {
        smemScoreBias[threadExpert] = scoreBias;
    }

    // registers for top group score reduction
    float topExpGroupScores[NumTopGroupScores];
    [[maybe_unused]] int32_t topExpGroupIdx[NumTopGroupScores];
    float topGroups[MaxNumTopGroups]; // bound of params.mNumLimitedGroups
    int32_t topGroupIdx[MaxNumTopGroups];
    float expertScoreGroup[MaxNumTopGroups];
    int32_t expertIdxGroup[MaxNumTopGroups];
    float topScores[MaxNumTopExperts]; // bound of params.mTopK
    int32_t topExperts[MaxNumTopExperts];

    if constexpr (KernelParams::UseGroups)
    {
        reduceTopK(warp, topExpGroupScores, topExpGroupIdx, scoreBias, threadExpert,
            /* minValue */ invalidScoreFloat);

        // get the final group score and write it to shared
        if (cute::elect_one_sync())
        {
            auto groupScore = topExpGroupScores[0] + topExpGroupScores[1];
            smemGroupScores[warpIdx] = groupScore;
        }
    }

    // make group scores available to all warps
    __syncthreads();

    auto localExpertExtent = params.mNumLocalExperts << params.mLocalExpertsStrideLog2;
    if (warpIdx == 0)
    {
        // a single warp performs the selection of top groups, and goes on to select the final experts
        if constexpr (KernelParams::UseGroups)
        {
            float groupScore = laneIdx < params.mNumExpertGroups ? smemGroupScores[laneIdx] : invalidScoreFloat;

            reduceTopK(warp, topGroups, topGroupIdx, groupScore, laneIdx,
                /* minValue */ invalidScoreFloat);

            // final expert selection: get relevant indexes and scores from shared

#pragma unroll
            for (int ii = 0; ii < MaxNumTopGroups; ++ii)
            { // bound of params.mNumLimitedGroups
                auto groupIdx = topGroupIdx[ii];
                expertIdxGroup[ii] = groupIdx * params.mNumExpertsPerGroup + laneIdx;
                // note: expertSelected implies laneIdx < params.mNumExpertsPerGroup.
                // we have params.mNumExpertsPerGroup == params.mNumExperts / params.mNumExpertGroups,
                // thus groupIdx <= params.mNumExpertGroups - 1 =>
                // groupIdx * params.mNumExpertsPerGroup <= params.mNumExperts - params.mNumExpertsPerGroup
                // => expertIdxGroup[ii] < params.mNumExperts <= NumThreads,
                // so the access is safe here
                expertScoreGroup[ii] = groupIdx < params.mNumExpertGroups && expertSelected
                    ? smemScoreBias[expertIdxGroup[ii]]
                    : invalidScoreFloat;
            }
        }
        else
        {
            // without groups, each thread just takes `MaxNumTopGroups` experts

#pragma unroll
            for (int ii = 0; ii < MaxNumTopGroups; ++ii)
            {
                auto expertIdx = ii * WarpSize + laneIdx;
                expertIdxGroup[ii] = expertIdx;
                expertScoreGroup[ii] = expertIdx < params.mNumExperts ? smemScoreBias[expertIdx] : invalidScoreFloat;
            }
        }

        reduceTopK(warp, topScores, topExperts, expertScoreGroup, expertIdxGroup,
            /* minValue */ invalidScoreFloat);

        // determine our lane's expert index and write to output
        int32_t expertIdx = 0;
#pragma unroll
        for (int ii = 0; ii < MaxNumTopExperts; ++ii)
        { // bound of params.mTopK
            expertIdx = laneIdx == ii ? topExperts[ii] : expertIdx;
        }
        // determine whether our expert is local to this GPU
        auto localExpertIdx = expertIdx - params.mLocalExpertsStartIdx;
        auto isLocalExpert = localExpertIdx >= 0 && localExpertIdx < localExpertExtent
            && (localExpertIdx & params.mLocalExpertsStrideLog2) == 0;

        // write expert idx out already
        auto idxTopK = blockIdx.x * params.mTopK + laneIdx;
        if (laneIdx < params.mTopK && params.mPtrExpertIdx != nullptr)
        {
            params.mPtrExpertIdx[idxTopK] = expertIdx;
        }
        float scoreNorm = laneIdx < params.mTopK ? smemScoreSigmoid[expertIdx] : 0.F;
        auto redNorm = cg::reduce(warp, scoreNorm, cg::plus<float>{});
        auto finalScore = TypeExpW{scoreNorm * params.mRouteScale / redNorm};
        if (laneIdx < params.mTopK && params.mPtrExpertWeights != nullptr)
        {
            params.mPtrExpertWeights[idxTopK] = finalScore;
        }
        if (laneIdx < params.mTopK && params.mPtrExpertWeightsFull != nullptr && isLocalExpert)
        {
            auto idxWeightsFull = localExpertIdx * gridDim.x + blockIdx.x;
            params.mPtrExpertWeightsFull[idxWeightsFull] = finalScore;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KernelParams>
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
__global__ void __cluster_dims__(NumBlocksPerCluster, 1, 1) __launch_bounds__(NumThreads)
    routingIndicesClusterKernel(KernelParams params)
{
    // number of experts is bounded by number of threads
    __shared__ int32_t __attribute((aligned(128))) smemExpertCount[NumThreads];
    __shared__ int32_t __attribute((aligned(128))) smemExpertOffset[NumThreads];
    // needed for the exclusive sum of token offsets
    using Scan = cub::BlockScan<int32_t, NumThreads, cub::BLOCK_SCAN_WARP_SCANS>;
    __shared__ typename Scan::TempStorage tempStorage;
    // Number of threads in the cluster.
    static constexpr int NumThreadsPerCluster = NumThreads * NumBlocksPerCluster;
    // If the number of tokens is bounded by 16384, then the total number of indexes
    // is bounded by 16384 * TopK.
    // TODO: if we only use this kernel up to 1024 tokens, we could use 1024 here.
    static constexpr int MaxExpandedIdxPerThread
        = (16384 * MaxNumTopExperts + NumThreadsPerCluster - 1) / NumThreadsPerCluster;

    // Initialize cluster.
    int32_t const clusterBlockRank = blockIdx.x;
    int32_t const clusterThreadIdx = NumThreads * clusterBlockRank + threadIdx.x;

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
        int32_t expertIdx = params.mPtrExpertIdx[expandedIdx];
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
    cg::cluster_group::sync();

    //
    // Each thread now represents one expert
    //

    // Get the histogram bin from each rank for this expert.
    int32_t expertCounts[NumBlocksPerCluster];
#pragma unroll
    for (int rank = 0; rank < NumBlocksPerCluster; rank++)
    {
        int32_t const* remoteSmem = cg::cluster_group::map_shared_rank(smemExpertCount, rank);
        expertCounts[rank] = remoteSmem[threadIdx.x];
    }

    // Compute an exclusive prefix sum of the block-local count.
    // Each block only needs the count up to its rank, and the total count.
    int32_t count = 0;
    int32_t blockExpertOffset = 0;
#pragma unroll
    for (int rank = 0; rank < NumBlocksPerCluster; rank++)
    {
        if (rank == clusterBlockRank)
        {
            blockExpertOffset = count;
        }
        count += expertCounts[rank];
    }

    // Arrive: we do not access distributed shared memory after this point.
    __cluster_barrier_arrive();

    // Compute the runtime config for projections
    // Whether or not an expert is local is taken into account when smemExpertCount is computed
    // so we do not need to take it into account here.
    const int32_t numCta = divUpLog2<int32_t>(count, params.mPaddingLog2);
    int32_t ctaOffset;
    int32_t numNonExitingCtas;
    Scan(tempStorage).ExclusiveSum(numCta, ctaOffset, numNonExitingCtas);

    // Strided loop to share this work between blocks.
    int32_t tokensPerTile = params.mAllToAllRouteAct ? params.mNumTokens : count;
    for (int32_t cta = clusterBlockRank; cta < numCta; cta += NumBlocksPerCluster)
    {
        const int32_t localExpertIdx = (threadIdx.x - params.mLocalExpertsStartIdx) >> params.mLocalExpertsStrideLog2;
        params.mPtrCtaIdxXyToBatchIdx[ctaOffset + cta] = localExpertIdx;
        params.mPtrCtaIdxXyToMnLimit[ctaOffset + cta] = min(mulLog2<int32_t>(ctaOffset + cta + 1, params.mPaddingLog2),
            mulLog2<int32_t>(ctaOffset, params.mPaddingLog2) + tokensPerTile);
    }

    // get the padded offset associated with this expert
    const int32_t offset = mulLog2<int32_t>(ctaOffset, params.mPaddingLog2);
    const int32_t permutedIdxSize = mulLog2<int32_t>(numNonExitingCtas, params.mPaddingLog2);

    // write out padded count
    if (clusterBlockRank == 0 && warpIdx == NumWarps - 1 && cute::elect_one_sync())
    {
        params.mPtrPermutedIdxSize[0] = permutedIdxSize;
        params.mPtrNumNonExitingCtas[0] = numNonExitingCtas;
    }

    // write expert offsets to shared
    smemExpertOffset[threadIdx.x] = offset + blockExpertOffset;

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

    // TODO: disable PDL for now to avoid race condition in FC1
    if constexpr (KernelParams::UsePdl)
    {
        // cudaTriggerProgrammaticLaunchCompletion();
    }

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
#else
__global__ void routingIndicesClusterKernel(KernelParams params)
{
    assert(false && "routingIndicesClusterKernel is only supported on SM90+ architectures");
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KernelParams>
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
__global__ void __launch_bounds__(NumThreads) routingIndicesCoopKernel(KernelParams params)
{
    // number of experts is bounded by number of threads
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
        int32_t expertIdx = params.mPtrExpertIdx[expandedIdx];
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
    int32_t const blockExpertOffset = atomicAdd(&params.mPtrExpertCounts[threadIdx.x], localExpertCount);

    // Sync to wait for completion of the histogram reduction.
    grid.sync();

    // Get total count for this expert.
    int32_t count = params.mPtrExpertCounts[threadIdx.x];

    // Note: the scan is redundant in all CTAs, but doing it in only 1 CTA would be worse for latency.

    // Compute the runtime config for projections
    // Whether or not an expert is local is taken into account when smemExpertCount is computed
    // so we do not need to take it into account here.
    const int32_t numCta = divUpLog2<int32_t>(count, params.mPaddingLog2);
    int32_t ctaOffset;
    int32_t numNonExitingCtas;
    Scan(tempStorage).ExclusiveSum(numCta, ctaOffset, numNonExitingCtas);

    // Strided loop to share this work between blocks.
    int32_t tokensPerTile = params.mAllToAllRouteAct ? params.mNumTokens : count;
    for (int32_t cta = gridBlockIdx; cta < numCta; cta += numBlocks)
    {
        const int32_t localExpertIdx = (threadIdx.x - params.mLocalExpertsStartIdx) >> params.mLocalExpertsStrideLog2;
        params.mPtrCtaIdxXyToBatchIdx[ctaOffset + cta] = localExpertIdx;
        params.mPtrCtaIdxXyToMnLimit[ctaOffset + cta] = min(mulLog2<int32_t>(ctaOffset + cta + 1, params.mPaddingLog2),
            mulLog2<int32_t>(ctaOffset, params.mPaddingLog2) + tokensPerTile);
    }

    // get the padded offset associated with this expert
    const int32_t offset = mulLog2<int32_t>(ctaOffset, params.mPaddingLog2);
    const int32_t permutedIdxSize = mulLog2<int32_t>(numNonExitingCtas, params.mPaddingLog2);

    // write out padded count
    if (gridBlockIdx == 0 && warpIdx == NumWarps - 1 && cute::elect_one_sync())
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

// Two-step approach (if number of tokens exceed limits of what cluster / cooperative launch
// variants can handle): in order to minimize the amount of data to exchange through global memory,
// we will compute the local histograms in smem twice: the first kernel will get us the total number
// of tokens per expert. The second kernel will use the smem and L2 atomics to get corresponding
// element and tile offsets.
//
// Note: the histogram calculation could also be fused with routingMainKernel, but this might be
// inefficient if we have one CTA per token doing a single global atomic.

template <typename KernelParams>
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
__global__ void __launch_bounds__(NumThreads) routingIndicesHistogramKernel(KernelParams params)
{
    // number of experts is bounded by number of threads
    __shared__ int32_t __attribute((aligned(128))) smemExpertCount[NumThreads];

    // For unrolling.
    int32_t constexpr NumEltsPerThread = 8;

    // Pre-fill the counts with 0
    smemExpertCount[threadIdx.x] = 0;
    __syncthreads();

    // Wait on primary grid and trigger secondary kernel.
    if constexpr (KernelParams::UsePdl)
    {
        cudaGridDependencySynchronize();
        cudaTriggerProgrammaticLaunchCompletion();
    }

    int32_t const expandedIdxSize = params.mNumTokens * params.mTopK;
    int32_t const localExpertExtent = params.mNumLocalExperts << params.mLocalExpertsStrideLog2;

    int32_t const gridBlockOffset = blockIdx.x * NumThreads;
    int32_t const gridStride = gridDim.x * NumThreads;

    // Define a lambda to avoid code duplication in branches.
    auto loopBody = [&](int expandedIdx)
    {
        int32_t expertIdx = params.mPtrExpertIdx[expandedIdx];
        // check whether this expert is local to our GPU at all and ignore if not
        auto localExpertIdx = expertIdx - params.mLocalExpertsStartIdx;
        auto isLocalExpert = localExpertIdx >= 0 && localExpertIdx < localExpertExtent
            && (localExpertIdx & params.mLocalExpertsStrideLog2) == 0;
        if (isLocalExpert)
        {
            atomicAdd(&smemExpertCount[expertIdx], 1);
        }
    };

    // Grid-stride loop.
    for (int32_t expandedIdx0 = gridBlockOffset * NumEltsPerThread; expandedIdx0 < expandedIdxSize;
         expandedIdx0 += gridStride * NumEltsPerThread)
    {
        // Fast path if bound checks aren't necessary
        if (expandedIdx0 + NumEltsPerThread * NumThreads <= expandedIdxSize)
        {
#pragma unroll
            for (int32_t ii = 0; ii < NumEltsPerThread; ii++)
            {
                int32_t expandedIdx = expandedIdx0 + ii * NumThreads + threadIdx.x;
                loopBody(expandedIdx);
            }
        }
        else
        {
            for (int32_t expandedIdx = expandedIdx0 + threadIdx.x; expandedIdx < expandedIdxSize;
                 expandedIdx += NumThreads)
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
    int32_t const localExpertCount = smemExpertCount[threadIdx.x];
    atomicAdd(&params.mPtrExpertCounts[threadIdx.x], localExpertCount);
}
#else
__global__ void routingIndicesHistogramKernel(KernelParams params)
{
    assert(false && "routingIndicesHistogramKernel is only supported on SM90+ architectures");
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KernelParams>
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
__global__ void __launch_bounds__(NumThreads) routingIndicesOffsetsKernel(KernelParams params)
{
    // number of experts is bounded by number of threads
    __shared__ int32_t __attribute((aligned(128))) smemExpertOffset[NumThreads];
    __shared__ int32_t __attribute((aligned(128))) smemExpertCount[NumThreads];
    __shared__ int32_t __attribute((aligned(128))) smemExpertTileOffset[NumThreads];
    // needed for the exclusive sum of token offsets
    using Scan = cub::BlockScan<int32_t, NumThreads, cub::BLOCK_SCAN_WARP_SCANS>;
    __shared__ typename Scan::TempStorage tempStorage;
    static constexpr int MaxExpandedIdxPerThread = NumEltsPerOffsetTilePerThread;
    static constexpr int MaxExpandedIdxPerBlock = NumThreads * MaxExpandedIdxPerThread;

    int32_t const warpIdx = __shfl_sync(0xffffffff, threadIdx.x / WarpSize, 0);

    int32_t const expandedIdxSize = params.mNumTokens * params.mTopK;
    int32_t const numTiles = (expandedIdxSize + MaxExpandedIdxPerBlock - 1) / (MaxExpandedIdxPerBlock);

    // Wait on primary grid.
    if constexpr (KernelParams::UsePdl)
    {
        cudaGridDependencySynchronize();
    }

    // The expert offsets are common to all tiles of all blocks.
    // Load the histogram, scan it and write offsets to shared memory.
    // Note: the scan is redundant in all CTAs. Would it make sense to use an intermediate kernel for
    // the scan, with PDL?

    // Each thread represents one expert. Get total count for this expert.
    int32_t count = params.mPtrExpertCounts[threadIdx.x];

    // Compute the runtime config for projections
    // Whether or not an expert is local is taken into account when the histogram is computed
    // so we do not need to take it into account here.
    const int32_t numCta = divUpLog2<int32_t>(count, params.mPaddingLog2);
    int32_t ctaOffset;
    int32_t numNonExitingCtas;
    Scan(tempStorage).ExclusiveSum(numCta, ctaOffset, numNonExitingCtas);

    // Get the padded offset associated with this expert
    const int32_t offset = mulLog2<int32_t>(ctaOffset, params.mPaddingLog2);
    const int32_t permutedIdxSize = mulLog2<int32_t>(numNonExitingCtas, params.mPaddingLog2);

    // Write expert offsets to shared
    smemExpertOffset[threadIdx.x] = offset;
    // Sync to make expert offsets available to all threads.
    __syncthreads();

    // The first block writes out padded count
    if (blockIdx.x == 0 && warpIdx == NumWarps - 1 && cute::elect_one_sync())
    {
        params.mPtrPermutedIdxSize[0] = permutedIdxSize;
        params.mPtrNumNonExitingCtas[0] = numNonExitingCtas;
    }

    // Strided loop to share this work between blocks.
    int32_t tokensPerTile = params.mAllToAllRouteAct ? params.mNumTokens : count;
    for (int32_t cta = blockIdx.x; cta < numCta; cta += gridDim.x)
    {
        const int32_t localExpertIdx = (threadIdx.x - params.mLocalExpertsStartIdx) >> params.mLocalExpertsStrideLog2;
        params.mPtrCtaIdxXyToBatchIdx[ctaOffset + cta] = localExpertIdx;
        params.mPtrCtaIdxXyToMnLimit[ctaOffset + cta] = min(mulLog2<int32_t>(ctaOffset + cta + 1, params.mPaddingLog2),
            mulLog2<int32_t>(ctaOffset, params.mPaddingLog2) + tokensPerTile);
    }

    //
    // Now loop on indices and compute offsets.
    //

    // Grid-stride loop on 1D "tiles" of input indices.
    for (int32_t tileIdx = blockIdx.x; tileIdx < numTiles; tileIdx += gridDim.x)
    {
        if (tileIdx > 0)
        {
            // Sync for safe reuse of smem buffers.
            __syncthreads();
        }

        // Pre-fill the counts with 0
        smemExpertCount[threadIdx.x] = 0;
        __syncthreads();

        // each thread keeps has some number of "expanded indexes" assigned to it
        // for each of these, we keep the associated expert and offset within expert in registers
        int32_t expertIndexes[MaxExpandedIdxPerThread];
        int32_t expertOffsets[MaxExpandedIdxPerThread];
        auto localExpertExtent = params.mNumLocalExperts << params.mLocalExpertsStrideLog2;

        // Define a lambda to avoid code duplication in branches.
        auto loopBody = [&](int ii, int expandedIdx)
        {
            int32_t expertIdx = params.mPtrExpertIdx[expandedIdx];
            expertIndexes[ii] = expertIdx;
            // check whether this expert is local to our GPU at all and ignore if not
            auto localExpertIdx = expertIdx - params.mLocalExpertsStartIdx;
            auto isLocalExpert = localExpertIdx >= 0 && localExpertIdx < localExpertExtent
                && (localExpertIdx & params.mLocalExpertsStrideLog2) == 0;
            expertOffsets[ii] = isLocalExpert ? atomicAdd(smemExpertCount + expertIdx, 1) : 0;
        };

        // For all tiles but the last, all indices are in bounds.
        if (tileIdx < numTiles - 1)
        {
#pragma unroll
            for (int32_t ii = 0; ii < MaxExpandedIdxPerThread; ii += 1)
            {
                auto expandedIdx = tileIdx * MaxExpandedIdxPerBlock + ii * NumThreads + threadIdx.x;
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
                    = tileIdx * MaxExpandedIdxPerBlock + (ii0 + IterStride) * NumThreads <= expandedIdxSize;
                if (takeFastPath)
                {
#pragma unroll
                    for (int32_t jj = 0; jj < IterStride; jj++)
                    {
                        int const ii = ii0 + jj;
                        auto expandedIdx = tileIdx * MaxExpandedIdxPerBlock + ii * NumThreads + threadIdx.x;
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
                        auto expandedIdx = tileIdx * MaxExpandedIdxPerBlock + ii * NumThreads + threadIdx.x;
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

        // Each thread now represents one expert

        // Add the local bin count to the common bin count and get a per-CTA offset. We use the second
        // half of the histogram buffer for this histogram, because the first half already holds the
        // reduced histogram from the previous kernel.
        int32_t const localExpertCount = smemExpertCount[threadIdx.x];
        int32_t const tileExpertOffset
            = atomicAdd(&params.mPtrExpertCounts[NumThreads + threadIdx.x], localExpertCount);

        // Make per-expert tile offsets available to all threads in the block.
        smemExpertTileOffset[threadIdx.x] = tileExpertOffset + smemExpertOffset[threadIdx.x];
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
                auto expandedIdx = tileIdx * MaxExpandedIdxPerBlock + ii * NumThreads + threadIdx.x;
                storeLoopBody(ii, expandedIdx);
            }
        }
        else
        {
#pragma unroll
            for (int32_t ii = 0; ii < MaxExpandedIdxPerThread; ii += 1)
            {
                auto expandedIdx = tileIdx * MaxExpandedIdxPerBlock + ii * NumThreads + threadIdx.x;
                if (expandedIdx >= expandedIdxSize)
                {
                    break;
                }
                storeLoopBody(ii, expandedIdx);
            }
        }
    }

    // Trigger secondary kernel.
    // Note: this does not guarantee the visibility of prior writes unless the consumer executes a
    // dependency sync.
    if constexpr (KernelParams::UsePdl)
    {
        cudaTriggerProgrammaticLaunchCompletion();
    }
}
#else
__global__ void routingIndicesOffsetsKernel(KernelParams params)
{
    assert(false && "routingIndicesOffsetsKernel is only supported on SM90+ architectures");
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

void run(Data const& data, void* stream)
{
    TLLM_CHECK_WITH_INFO(data.mPtrExpertIdx != nullptr || data.mPtrPermutedIdxSize != nullptr
            || data.mPtrExpertWeightsFull != nullptr || data.mPtrExpertWeights != nullptr,
        "Routing kernel requires at least one output parameter");
    if (data.mPtrExpandedIdxToPermutedIdx != nullptr || data.mPtrPermutedIdxToTokenIdx != nullptr)
        TLLM_CHECK_WITH_INFO(data.mPtrExpertIdx != nullptr && data.mPtrPermutedIdxSize,
            "If permuted index is required, `mPtrExpertIdx` is also required");
    TLLM_CHECK_WITH_INFO(!data.mUseRoutingSoftmax, "Routing with softmax not implemented yet");
    TLLM_CHECK_WITH_INFO(data.mNumLimitedGroups <= MaxNumTopGroups, "Routing kernel expects <= %d top groups, got %d",
        MaxNumTopGroups, data.mNumLimitedGroups);
    TLLM_CHECK_WITH_INFO(data.mTopK <= MaxNumTopExperts, "Routing kernel expects topK experts <= %d, got %d",
        MaxNumTopExperts, data.mTopK);
    TLLM_CHECK_WITH_INFO(data.mTopK <= WarpSize, "Routing kernel expects top K <= warp size, got %d", data.mTopK);
    TLLM_CHECK_WITH_INFO(data.mTopK * data.mNumLimitedGroups <= WarpSize,
        "Routing kernel expects top K * top groups <= warp size (for now), got %d * %d", data.mTopK,
        data.mNumLimitedGroups);
    TLLM_CHECK_WITH_INFO(data.mNumExperts >= MaxNumTopExperts, "Routing kernel expects %d to be at most #experts %d",
        MaxNumTopExperts, data.mNumExperts);
    TLLM_CHECK_WITH_INFO(data.mNumExperts <= NumThreads, "Routing kernel expects #experts %d  <= #threads %d",
        data.mNumExperts, NumThreads);
    TLLM_CHECK_WITH_INFO(data.mNumExpertGroups >= data.mNumLimitedGroups,
        "Routing kernel expects top groups %d to be limited by #expert groups %d", data.mNumLimitedGroups,
        data.mNumExpertGroups);
    if (data.mNumExpertGroups > 1)
    {
        TLLM_CHECK_WITH_INFO(data.mNumExpertGroups <= NumWarps,
            "Routing kernel expects #experts groups %d to be <= #warps %d", data.mNumExpertGroups, NumWarps);
        TLLM_CHECK_WITH_INFO(data.mNumExperts % data.mNumExpertGroups == 0,
            "Routing kernel expects #experts %d to be a multiple of #expert groups %d", data.mNumExperts,
            data.mNumExpertGroups);
        TLLM_CHECK_WITH_INFO(data.mNumExperts / data.mNumExpertGroups <= WarpSize,
            "Routing kernel expects #experts per group <= warp size, got %d", data.mNumExperts / data.mNumExpertGroups);
    }
    else
    {
        TLLM_CHECK_WITH_INFO(data.mNumExperts <= WarpSize * MaxNumTopGroups,
            "Routing kernel expects #experts %d <= WarpSize * MaxNumTopGroups %d", data.mNumExperts,
            WarpSize * MaxNumTopGroups);
        TLLM_CHECK_WITH_INFO(
            data.mTopK <= NumWarps, "Routing kernel expects top K %d to be <= #warps %d", data.mTopK, NumWarps);
    }
    TLLM_CHECK_WITH_INFO(
        data.mNumExperts % 4 == 0, "Routing kernel expects #experts %d to be a multiple of 4.", data.mNumExperts);
    TLLM_CHECK_WITH_INFO(data.mPaddingLog2 < 8, "Routing kernel expects padding log2 < 8, got %d", data.mPaddingLog2);
    int const numBlocks = data.mNumTokens;

    if (data.mPtrExpertWeightsFull != nullptr)
    {
        auto localExpertExtent = data.mNumLocalExperts << data.mLocalExpertsStrideLog2;
        // note: we set a value of 0 here, s.t. even if the routing happens,
        // it will be ignored / not given any weight
        TLLM_CUDA_CHECK(cudaMemsetAsync(
            data.mPtrExpertWeightsFull, 0, localExpertExtent * data.mNumTokens * sizeof(float), (cudaStream_t) stream));
    }

    /*  disable memset(-1) for permuted_idx_to_token_idx for performance
        if (data.mPtrPermutedIdxToTokenIdx != nullptr)
        {
            // need to set all values to -1 before running the kernel
            auto maxPermutedSize
                = data.mNumTokens * data.mTopK + (data.mNumExperts << data.mPaddingLog2) - data.mNumExperts;
            // note that a value of -1 per byte works for any size of signed integer
            // to set each full value to the logical value -1
            TLLM_CUDA_CHECK(cudaMemsetAsync(data.mPtrPermutedIdxToTokenIdx, -1,
                static_cast<size_t>(maxPermutedSize) * sizeof(int32_t), (cudaStream_t) stream));
        }
    */

    bool const useSingleCluster = data.mNumTokens <= 1024;
    if (!useSingleCluster)
    {
        // Reset the global histograms (not used in single-cluster code path).
        // Cover both for the cooperative and two-kernel code paths.
        TLLM_CUDA_CHECK(cudaMemsetAsync(
            data.mPtrExpertCounts, 0, static_cast<size_t>(2 * NumThreads) * sizeof(int32_t), (cudaStream_t) stream));
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
    // /!\ The following number is not portable!! (but works on H100 and B200)
    int const numBlocksCoop = 128;

    // Maximum number of tokens supported by the kernel using a cooperative launch.
    int const maxTokensCoop = (numBlocksCoop * NumThreads * 64) / data.mTopK;
    LAUNCH_EXPW_ONLY_GROUPS(data,
        /*coopLaunch=*/false, routingMainKernel, numBlocks, NumThreads,
        /*smemSize=*/0, // No dynamic smem
        stream);

    if (data.mPtrPermutedIdxSize != nullptr)
    {
        if (useSingleCluster)
        {
            LAUNCH_EXPW_ONLY_GROUPS(data,
                /*coopLaunch=*/false, routingIndicesClusterKernel, NumBlocksPerCluster, NumThreads,
                /*smemSize=*/0, // No dynamic smem
                stream);
        }
        else if (data.mNumTokens <= maxTokensCoop)
        {
            LAUNCH_EXPW_ONLY_GROUPS(data,
                /*coopLaunch=*/true, routingIndicesCoopKernel, numBlocksCoop, NumThreads,
                /*smemSize=*/0, // No dynamic smem
                stream);
        }
        else
        {
            const int32_t expandedIdxSize = data.mNumTokens * data.mTopK;

            const int32_t histogramEltsPerBlock = 8 * NumThreads;
            const int32_t offsetEltsPerBlock = NumEltsPerOffsetTilePerThread * NumThreads;

            // Limit grid size (both kernels use a grid-stride loop).
            const int32_t maxNumBlocks = 1024;

            int const numBlocksHistogram
                = std::min((expandedIdxSize + histogramEltsPerBlock - 1) / histogramEltsPerBlock, maxNumBlocks);
            int const numBlocksOffsets
                = std::min((expandedIdxSize + offsetEltsPerBlock - 1) / offsetEltsPerBlock, maxNumBlocks);

            LAUNCH_EXPW_ONLY_GROUPS(data,
                /*coopLaunch=*/false, routingIndicesHistogramKernel, numBlocksHistogram, NumThreads,
                /*smemSize=*/0, // No dynamic smem
                stream);
            LAUNCH_EXPW_ONLY_GROUPS(data,
                /*coopLaunch=*/false, routingIndicesOffsetsKernel, numBlocksOffsets, NumThreads,
                /*smemSize=*/0, // No dynamic smem
                stream);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace routing

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace routingLlama4
{

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace tg = batchedGemm::trtllm::gen;
namespace cg = cooperative_groups;

////////////////////////////////////////////////////////////////////////////////////////////////////

static constexpr int NumThreads = 1024;
static constexpr int NumThreadsHist = 256;
static constexpr int NumBlocksPerCluster = 8;
static constexpr int WarpSize = 32;
static constexpr int NumWarps = NumThreads / WarpSize;
static constexpr int NumWarpsHist = NumThreadsHist / WarpSize;
static constexpr int NumTopExperts = 1;
static constexpr int MaxNumExperts = 128;
static constexpr int MaxNumTokensSingleCluster = NumBlocksPerCluster * NumThreads;
static constexpr int MaxNumTokensSingleClusterScores = NumBlocksPerCluster * NumWarps;
static constexpr int WarpKernelSmemStride = 33;
// with further optimization to `routingIndicesWarpKernel`, this limit may
// increase. For now, it is a good cut-off point for when the block-wise
// operations are more efficient end-to-end.
static constexpr int WarpKernelMaxNumTokens = 4;

// Performance tuning knob.
static constexpr int NumEltsPerOffsetTilePerThread = 8;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000 && defined(__CUDA_ARCH_FEAT_SM100_ALL))
#define TLLM_GEN_ENABLE_FAST_REDUX
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename TypeExpW_>
struct TopKRedType
{
    using TypeExpW = TypeExpW_;
    static_assert(std::is_same_v<TypeExpW, float> || std::is_same_v<TypeExpW, cutlass::bfloat16_t>,
        "Top K reduction only implemented for float and Bf16");
    using TypeCmp = std::conditional_t<sizeof(TypeExpW) >= 4, double, float>;
    static constexpr int64_t Mask64 = 0x000000000000FFFF;
    static constexpr int32_t Mask32 = 0x0000FFFF;

    TypeCmp compVal;

    static __host__ __device__ inline TypeCmp makeCmpVal(TypeExpW val, int32_t idx = 0)
    {
        auto cmpVal = TypeCmp{val};
        TypeCmp cmpValWithIdx;
        if constexpr (sizeof(TypeExpW) >= 4)
        {
            auto cmpValIdx64 = reinterpret_cast<int64_t&>(cmpVal) | (Mask64& int64_t{idx});
            cmpValWithIdx = reinterpret_cast<TypeCmp&>(cmpValIdx64);
        }
        else
        {
            auto cmpValIdx32 = reinterpret_cast<int32_t&>(cmpVal) | (Mask32 & idx);
            cmpValWithIdx = reinterpret_cast<TypeCmp&>(cmpValIdx32);
        }
        return cmpValWithIdx;
    }

    static __host__ __device__ inline void unpack(TypeExpW& val, int32_t& idx, TypeCmp cmp)
    {
        if constexpr (sizeof(TypeExpW) >= 4)
        {
            idx = static_cast<int32_t>(reinterpret_cast<int64_t&>(cmp) & Mask64);
            auto val64 = reinterpret_cast<int64_t&>(cmp) & ~Mask64;
            val = static_cast<float>(reinterpret_cast<double&>(val64));
        }
        else
        {
            idx = reinterpret_cast<int32_t&>(cmp) & Mask32;
            auto val32 = reinterpret_cast<int32_t&>(cmp) >> 16;
            val = TypeExpW::bitcast(reinterpret_cast<uint16_t&>(val32));
        }
    }

    __host__ __device__ TopKRedType() = default;

    __host__ __device__ TopKRedType(TypeExpW val, int32_t idx)
        : compVal(makeCmpVal(val, idx))
    {
    }

    __host__ __device__ operator TypeCmp() const noexcept
    {
        return compVal;
    }

    __device__ inline TypeCmp reduce(cg::thread_block_tile<WarpSize> const& warp)
    {
#if defined(TLLM_GEN_ENABLE_FAST_REDUX)
        static constexpr bool UseCg = false;
#else
        static constexpr bool UseCg = true;
#endif
        if constexpr (UseCg || sizeof(TypeExpW) >= 4)
        {
            return cg::reduce(warp, compVal, cg::greater<TypeCmp>{});
        }
        else
        {
            float result;
            asm("redux.sync.max.f32 %0, %1, 0xffffffff;\n" : "=f"(result) : "f"(compVal));
            return result;
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

static __device__ inline float sigmoid_accurate(float x)
{
    return 0.5f * tanhf(0.5f * x) + 0.5f;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int K_, bool Enable_>
struct TopKIdx
{
    // by default, empty
};

template <int K_>
struct TopKIdx<K_, true>
{
    static constexpr int K = K_;
    int32_t val[K];
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int K, typename Type>
__device__ void reduceTopK(cg::thread_block_tile<WarpSize> const& warp, Type (&out)[K], int32_t (&outIdx)[K],
    Type value, int32_t idx, Type minValue)
{
    static_assert(K > 0, "Top K must have K > 0");
    static_assert(K < WarpSize, "Top K must have K < WarpSize");
    using RedType = TopKRedType<Type>;
    RedType topK{value, idx};
    typename RedType::TypeCmp packedMax{};
#pragma unroll
    for (int kk = 0; kk < K; ++kk)
    {
        topK = kk > 0 && packedMax == topK.compVal ? RedType{minValue, idx} : topK;
        // get the next largest value
        packedMax = topK.reduce(warp);
        RedType::unpack(out[kk], outIdx[kk], packedMax);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

#define TOPK_SWAP(I, J)                                                                                                \
    {                                                                                                                  \
        auto pairMin = min(topK[I].compVal, topK[J].compVal);                                                          \
        auto pairMax = max(topK[I].compVal, topK[J].compVal);                                                          \
        topK[I].compVal = pairMax;                                                                                     \
        topK[J].compVal = pairMin;                                                                                     \
    }

template <int K, typename Type, int N, bool IsSorted = false>
__device__ void reduceTopK(cg::thread_block_tile<WarpSize> const& warp, Type (&out)[K], int32_t (&outIdx)[K],
    Type (&value)[N], int32_t (&idx)[N], Type minValue)
{
    static_assert(K > 0, "Top K must have K > 0");
    static_assert(K < WarpSize, "Top K must have K < WarpSize");
    static_assert(N > 0, "Top K must have N > 1");
    static_assert(N <= K, "Top K must have N < K");
    using RedType = TopKRedType<Type>;
    RedType topK[N];
#pragma unroll
    for (int nn = 0; nn < N; ++nn)
        topK[nn] = RedType{value[nn], idx[nn]};
    if constexpr (!IsSorted)
    {
        TOPK_SWAP(0, 2);
        TOPK_SWAP(1, 3);

        TOPK_SWAP(0, 1);
        TOPK_SWAP(2, 3);

        TOPK_SWAP(1, 2);
    }
    typename RedType::TypeCmp packedMax{};
#pragma unroll
    for (int kk = 0; kk < K; ++kk)
    {
        bool update = kk > 0 && packedMax == topK[0].compVal;
#pragma unroll
        for (int nn = 0; nn < N; ++nn)
        {
            topK[nn] = update && nn == N - 1 ? RedType{minValue, idx[nn]} : update ? topK[nn + 1] : topK[nn];
        }
        // get the next largest value
        packedMax = topK[0].reduce(warp);
        RedType::unpack(out[kk], outIdx[kk], packedMax);
    }
};

#undef TOPK_SWAP

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

template <typename KernelParams>
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
__global__ void __launch_bounds__(WarpSize) routingIndicesWarpKernel(KernelParams params)
{
    // types used in this kernel
    using TypeExpW = typename KernelParams::TypeExpW;
    using TypePacked = PackedScoreIdx<TypeExpW>;
    // use the default cub warp-scan, with shfl
    using Scan = cub::WarpScan<int32_t>;
    __shared__ typename Scan::TempStorage tempStorage;

    // each thread encodes 4 experts in one `int32_t`. The assumption is that
    // we don't have more than 127 tokens, but `WarpKernelMaxNumTokens` must be
    // smaller than that because other approaches will be more efficient for
    // 127 tokens.
    static constexpr int ExpertsPerThread = sizeof(int32_t);
    static_assert(WarpKernelMaxNumTokens <= 127);
    // this is a full table of which token is routed to which expert.
    // the assumption here is that there are no more than 128 experts.
    // we use a stride of 33 instead of 32 to avoid shared memory bank conflicts.
    __shared__ int32_t __attribute((aligned(128)))
    smemExpertTokenCountFull[WarpKernelMaxNumTokens][WarpKernelSmemStride];
    static_assert(WarpKernelSmemStride == WarpSize + 1);
    static_assert(MaxNumExperts / sizeof(int32_t) <= WarpSize);

    // values needed for the top-1 reduction, if required
    TypeExpW minScore = TypeExpW{-INFINITY};
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<WarpSize>(block);

#pragma unroll
    for (int tokenIdx = 0; tokenIdx < WarpKernelMaxNumTokens; ++tokenIdx)
    {
        // reset full shared memory field to 0
        smemExpertTokenCountFull[tokenIdx][threadIdx.x] = 0;
    }
    __syncwarp();

    // then wait on primary grid
    if constexpr (KernelParams::UsePdl)
    {
        cudaGridDependencySynchronize();
    }

    if (params.mPtrScores != nullptr)
    {
        // if we use `mPtrScores` as input, we need to perform the top-1 reduction
        // for each token, we load the scores then use `reduceTopK` for this.
        // each thread works on 4 experts, so a local reduction is done before
        for (int tokenIdx = 0; tokenIdx < params.mNumTokens; ++tokenIdx)
        {
            auto scoreOffset = tokenIdx * params.mNumExperts;
            // local reduction to get the best score for our 4 experts
            TypeExpW maxScore = minScore;
            int32_t maxExpertIdx{-1};
#pragma unroll
            for (int ii = 0; ii < ExpertsPerThread; ++ii)
            {
                auto expertIdx = ii * WarpSize + threadIdx.x;
                auto newScore = expertIdx < params.mNumExperts ? params.mPtrScores[scoreOffset + expertIdx] : minScore;
                // note: use `>=` s.t. highest index always wins, just like in `reduceTopK`
                maxExpertIdx = newScore >= maxScore ? expertIdx : maxExpertIdx;
                maxScore = newScore >= maxScore ? newScore : maxScore;
            }
            int32_t warpMaxExpertIdx[NumTopExperts];
            TypeExpW warpMaxScore[NumTopExperts];
            // warp-wide reduction to get the best score for all experts
            reduceTopK(warp, warpMaxScore, warpMaxExpertIdx, maxScore, maxExpertIdx, minScore);
            if (cute::elect_one_sync())
            {
                // one thread updates the count linking token to chosen expert
                auto expertTokenCount = 0;
                setBits</* IsZero= */ true>(expertTokenCount, 1, warpMaxExpertIdx[0] % ExpertsPerThread);
                smemExpertTokenCountFull[tokenIdx][warpMaxExpertIdx[0] / ExpertsPerThread] = expertTokenCount;
                // we also compute the final score here and write it out if required
                auto finalScore = TypeExpW{sigmoid_accurate(float{warpMaxScore[0]})};
                if (params.mPtrExpertWeights != nullptr)
                {
                    params.mPtrExpertWeights[tokenIdx] = finalScore;
                }
            }
        }
    }
    else
    {
        // if we do not have `mPtrScores` as input, we expect that `mPtrExpertWeights`
        // contains the top-1 packed score and index already.
        // Each thread represents a token here, and we extract the relevant score
        // The assumption is that the #tokens is limited by warp-size
        static_assert(WarpKernelMaxNumTokens <= WarpSize);
        TypePacked scoreIdx = threadIdx.x < params.mNumTokens ? params.mPtrExpertIdx[threadIdx.x] : TypePacked{};
        int32_t expertTokenCount = 0;
        setBits</* IsZero= */ true>(expertTokenCount, 1, scoreIdx.idx % ExpertsPerThread);
        if (threadIdx.x < params.mNumTokens)
        {
            smemExpertTokenCountFull[threadIdx.x][scoreIdx.idx / ExpertsPerThread] = expertTokenCount;
        }
        // we also compute the final score here and write it out if required
        auto finalScore = TypeExpW{sigmoid_accurate(float{scoreIdx.score})};
        if (params.mPtrExpertWeights != nullptr && threadIdx.x < params.mNumTokens)
        {
            params.mPtrExpertWeights[threadIdx.x] = finalScore;
        }
    }

    // make the full table available to all threads
    __syncwarp();

    // at this point, each thread keeps a count of its 4 assigned experts in
    // `expertCount`, as well as the offsets for all tokens w.r.t. these 4 experts
    // in `expertOffset`.
    int32_t expertCount = 0;
    int32_t expertOffset[WarpKernelMaxNumTokens + 1];
#pragma unroll
    for (int tokenIdx = 0; tokenIdx < WarpKernelMaxNumTokens + 1; ++tokenIdx)
    {
        if (tokenIdx > params.mNumTokens)
            break;
        // simple reduction for `expertCount`, and scan for `expertOffset`
        auto expertTokenCount = tokenIdx < params.mNumTokens ? smemExpertTokenCountFull[tokenIdx][threadIdx.x] : 0;
        expertOffset[tokenIdx] = expertCount;
        expertCount += expertTokenCount;
    }

    // at this point, we are ready for the scan across all experts to get the
    // thread-wise offsets across experts
    // first, we need to reduce across our 4 experts into `numCta`
    int32_t numCta = 0;
#pragma unroll
    for (int ii = 0; ii < ExpertsPerThread; ++ii)
    {
        auto count = getBits(expertCount, ii);
        numCta += divUpLog2<int32_t>(count, params.mPaddingLog2);
    }
    // second, we perform the exclusive sum across the warp
    int32_t ctaOffset;
    int32_t numNonExitingCtas;
    Scan(tempStorage).ExclusiveSum(numCta, ctaOffset, numNonExitingCtas);

    // finally, we perform a scan across our local experts, starting with the
    // warp-wide scan result (`ctaOffset`)
    auto ctaOffsetExp = ctaOffset;
#pragma unroll
    for (int ii = 0; ii < ExpertsPerThread; ++ii)
    {
        auto count = getBits(expertCount, ii);
        auto finalNumCta = divUpLog2<int32_t>(count, params.mPaddingLog2);
        auto expertIdx = threadIdx.x * ExpertsPerThread + ii;
        // during the scan for expert offsets, we can already write out
        // both `mPtrCtaIdxXyToBatchIdx` and `mPtrCtaIdxXyToMnLimit`
        for (int cta = 0; cta < finalNumCta; ++cta)
        {
            params.mPtrCtaIdxXyToBatchIdx[ctaOffsetExp + cta] = expertIdx;
            params.mPtrCtaIdxXyToMnLimit[ctaOffsetExp + cta]
                = min(mulLog2<int32_t>(ctaOffsetExp + cta + 1, params.mPaddingLog2),
                    mulLog2<int32_t>(ctaOffsetExp, params.mPaddingLog2) + count);
        }
        ctaOffsetExp += finalNumCta;
    }

    // at this point, we can write out padded count from the warp-aggregate
    if (cute::elect_one_sync())
    {
        const int32_t permutedIdxSize = mulLog2<int32_t>(numNonExitingCtas, params.mPaddingLog2);
        params.mPtrPermutedIdxSize[0] = permutedIdxSize;
        params.mPtrNumNonExitingCtas[0] = numNonExitingCtas;
    }

#if !defined(PDL_PROFILE) || PDL_PROFILE == 0
    // we can trigger the next kernel at this point
    if constexpr (KernelParams::UsePdl)
    {
        cudaTriggerProgrammaticLaunchCompletion();
    }
#endif

    // at this point, all values for offsets are ready, except the final offsets
    // within the padded index (`permutedIdx`)
    // for this, we perform a scan similar to the one directly after the warp-scan:
    // here, we keep the local offset for each of the thread's experts in a field
    // of registers
    auto localExpertExtent = params.mNumLocalExperts << params.mLocalExpertsStrideLog2;
    int32_t finalExpertOffset[ExpertsPerThread];
    finalExpertOffset[0] = mulLog2<int32_t>(ctaOffset, params.mPaddingLog2);
#pragma unroll
    for (int ii = 1; ii < ExpertsPerThread; ++ii)
    {
        finalExpertOffset[ii]
            = finalExpertOffset[ii - 1] + divUpMulLog2<int32_t>(getBits(expertCount, ii - 1), params.mPaddingLog2);
    }

#pragma unroll
    for (int tokenIdx = 0; tokenIdx < WarpKernelMaxNumTokens; ++tokenIdx)
    {
        // at this point, we can calculate the final index:
        // we simply loop over all tokens, and all experts assigned to this thread.
        // For each pair, we determine whether that token was routed to that expert
        // based on whether the offset for that token changed.
        // we can then easily compute the final `expertIdx` and `permutedIdx` relative
        // to this token and expert, and write them out.
        if (tokenIdx >= params.mNumTokens)
            break;

#pragma unroll
        for (int ii = 0; ii < ExpertsPerThread; ++ii)
        {
            // determine whether the offset for this expert and token changes
            auto localOffsetToken = getBits(expertOffset[tokenIdx], ii);
            auto isTokenRouted = getBits(expertOffset[tokenIdx + 1], ii) > localOffsetToken;
            // the expert index of this expert
            auto expertIdx = threadIdx.x * ExpertsPerThread + ii;
            auto localExpertIdx = static_cast<int32_t>(expertIdx) - params.mLocalExpertsStartIdx;
            auto isLocalExpert = localExpertIdx >= 0 && localExpertIdx < localExpertExtent
                && (localExpertIdx & params.mLocalExpertsStrideLog2) == 0;
            // the permuted index: we add the local offset relative to this expert and token
            // to the global offset from the scan for this expert
            auto permutedIdx = isLocalExpert ? finalExpertOffset[ii] + localOffsetToken : int32_t{-1};
            // write out `mPtrExpandedIdxToPermutedIdx` if required
            if (params.mPtrExpandedIdxToPermutedIdx != nullptr && isTokenRouted)
            {
                params.mPtrExpandedIdxToPermutedIdx[tokenIdx] = permutedIdx;
            }
            // write out `mPtrPermutedIdxToTokenIdx` if required
            if (params.mPtrPermutedIdxToTokenIdx != nullptr && isLocalExpert && isTokenRouted)
            {
                params.mPtrPermutedIdxToTokenIdx[permutedIdx] = tokenIdx;
            }
        }
    }
}
#else
__global__ void routingIndicesWarpKernel(KernelParams params)
{
    assert(false && "routingIndicesWarpKernel is only supported on SM90+ architectures");
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KernelParams>
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
__global__ void __cluster_dims__(NumBlocksPerCluster, 1, 1) __launch_bounds__(NumThreads)
    routingIndicesClusterKernel(KernelParams params)
{
    // number of experts is bounded by number of threads
    __shared__ int32_t __attribute((aligned(128))) smemExpertCount[NumThreads];
    __shared__ int32_t __attribute((aligned(128))) smemExpertOffset[NumThreads];
    // number of tokens/expanded idx is bounded by total number of warps
    using TypeExpW = typename KernelParams::TypeExpW;
    using TypePacked = PackedScoreIdx<TypeExpW>;
    __shared__ TypePacked __attribute((aligned(128))) smemPackedScoreIdx[NumWarps];
    // Needed for the exclusive sum of token offsets.
    // Note: the scan might include more bins than needed, with bin counts of 0 to pad
    using Scan = cub::BlockScan<int32_t, NumThreads, cub::BLOCK_SCAN_WARP_SCANS>;
    __shared__ typename Scan::TempStorage tempStorage;
    // Number of threads in the cluster.
    static constexpr int NumThreadsPerCluster = NumThreads * NumBlocksPerCluster;
    // same as max num tokens
    static constexpr int MaxExpandedIdxPerThread
        = (MaxNumTokensSingleCluster * NumTopExperts + NumThreadsPerCluster - 1) / NumThreadsPerCluster;

    uint32_t const clusterBlockRank = blockIdx.x;
    uint32_t const clusterThreadIdx = NumThreads * clusterBlockRank + threadIdx.x;

    int32_t const warpIdx = __shfl_sync(0xffffffff, threadIdx.x / WarpSize, 0);
    int32_t const laneIdx = cutlass::arch::LaneId();

    auto expandedIdxSize = params.mNumTokens * NumTopExperts;
    // TODO(mjoux): expand to more tokens (possibly)
    auto warpTokenIdx = clusterBlockRank * NumWarps + warpIdx;
    auto scoreOffset = warpTokenIdx * params.mNumExperts;
    bool validToken = warpTokenIdx < params.mNumTokens;
    TypeExpW minScore = TypeExpW{-INFINITY};

    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<WarpSize>(block);

    // pre-fill the counts with 0
    if (threadIdx.x < params.mNumExperts)
    {
        smemExpertCount[threadIdx.x] = 0;
    }
    __syncthreads();

    // then wait on primary grid
    if constexpr (KernelParams::UsePdl)
    {
        cudaGridDependencySynchronize();
    }

    if (params.mPtrScores != nullptr)
    {
        TypeExpW maxScore = minScore;
        int32_t maxExpertIdx{-1};
        // in this case, each warp represents a token
        // we then exchange all token max scores, s.t. afterwards, each thread
        // represents a token
        if (validToken)
        {
#pragma unroll
            for (int i = 0; i < MaxNumExperts / WarpSize; ++i)
            {
                auto expertIdx = i * WarpSize + laneIdx;
                auto newScore = expertIdx < params.mNumExperts ? params.mPtrScores[scoreOffset + expertIdx] : minScore;
                // note: use `>=` s.t. highest index always wins, just like in `reduceTopK`
                maxExpertIdx = newScore >= maxScore ? expertIdx : maxExpertIdx;
                maxScore = newScore >= maxScore ? newScore : maxScore;
            }
            int32_t warpMaxExpertIdx[NumTopExperts];
            TypeExpW warpMaxScore[NumTopExperts];
            reduceTopK(warp, warpMaxScore, warpMaxExpertIdx, maxScore, maxExpertIdx, minScore);
            if (cute::elect_one_sync())
            {
                TypePacked packedScore{warpMaxScore[0], static_cast<int16_t>(warpMaxExpertIdx[0])};
                smemPackedScoreIdx[warpIdx] = packedScore;
            }
        }
        // make packed scores available to all threads in cluster
        __cluster_barrier_arrive();
        __cluster_barrier_wait();
    }

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
        if (params.mPtrScores != nullptr)
        {
            TypePacked const* remoteSmem
                = cg::cluster_group::map_shared_rank(smemPackedScoreIdx, expandedIdx / NumWarps);
            scoreIdx = remoteSmem[expandedIdx % NumWarps];
        }
        else
        {
            scoreIdx = params.mPtrExpertIdx[expandedIdx];
        }
        expertIndexes[ii] = scoreIdx.idx;
        // check whether this expert is local to our GPU at all and ignore if not
        auto localExpertIdx = scoreIdx.idx - params.mLocalExpertsStartIdx;
        auto isLocalExpert = localExpertIdx >= 0 && localExpertIdx < localExpertExtent
            && (localExpertIdx & params.mLocalExpertsStrideLog2) == 0;
        expertOffsets[ii] = isLocalExpert ? atomicAdd(smemExpertCount + scoreIdx.idx, 1) : 0;
        auto finalScore = TypeExpW{sigmoid_accurate(float{scoreIdx.score})};
        if (params.mPtrExpertWeights != nullptr)
        {
            params.mPtrExpertWeights[expandedIdx] = finalScore;
        }
    };

    if (clusterThreadIdx < expandedIdxSize)
    {
        loopBody(0, clusterThreadIdx);
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
    // Weather or not an expert is local is taken into account when smemExpertCount is computed
    // so we do not need to take it into account here.
    const int32_t numCta = divUpLog2<int32_t>(count, params.mPaddingLog2);
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
            params.mPtrCtaIdxXyToMnLimit[ctaOffset + cta]
                = min(mulLog2<int32_t>(ctaOffset + cta + 1, params.mPaddingLog2),
                    mulLog2<int32_t>(ctaOffset, params.mPaddingLog2) + count);
        }

        // get the padded offset associated with this expert
        const int32_t offset = mulLog2<int32_t>(ctaOffset, params.mPaddingLog2);

        // write expert offsets to shared
        smemExpertOffset[threadIdx.x] = offset + blockExpertOffset;
    }

    // write out padded count
    if (clusterBlockRank == 0 && warpIdx == NumWarps - 1 && cute::elect_one_sync())
    {
        const int32_t permutedIdxSize = mulLog2<int32_t>(numNonExitingCtas, params.mPaddingLog2);
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
#if !defined(PDL_PROFILE) || PDL_PROFILE == 0
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
        auto tokenIdx = expandedIdx / NumTopExperts;
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
#else
__global__ void routingIndicesClusterKernel(KernelParams params)
{
    assert(false && "routingIndicesClusterKernel is only supported on SM90+ architectures");
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

// this kernel is needed in case we have scores as input for the histogram kernel
template <typename KernelParams>
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
__global__ void __launch_bounds__(NumThreadsHist) routingIndicesHistogramScoresKernel(KernelParams params)
{
    using TypeExpW = typename KernelParams::TypeExpW;
    using TypeExpWVec = std::conditional_t<sizeof(TypeExpW) == 2, float2, float4>;
    using TypePacked = PackedScoreIdx<TypeExpW>;
    static constexpr int VecSize = MaxNumExperts / WarpSize;
    // we assume that #experts is a multiple of 4, so VecSize must be 4.
    static_assert(VecSize == 4);

    int32_t const laneIdx = cutlass::arch::LaneId();
    int32_t const warpIdx = threadIdx.x / WarpSize;
    int32_t const globalWarpIdx = blockIdx.x * NumWarpsHist + warpIdx;
    int32_t const globalWarpStride = gridDim.x * NumWarpsHist;
    TypeExpW minScore = TypeExpW{-INFINITY};
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<WarpSize>(block);

    // Wait on primary grid and trigger secondary kernel.
    if constexpr (KernelParams::UsePdl)
    {
        cudaGridDependencySynchronize();
        cudaTriggerProgrammaticLaunchCompletion();
    }

    // in this case, each warp represents a token, and we use a grid-stride loop
    // over all warps/tokens
    for (int tokenIdx = globalWarpIdx; tokenIdx < params.mNumTokens; tokenIdx += globalWarpStride)
    {
        TypeExpW maxScore = minScore;
        int32_t maxExpertIdx{-1};
        auto scoreOffset = (tokenIdx * params.mNumExperts) / VecSize + laneIdx;

        TypeExpW allScores[VecSize];
        auto* ptrAllScores = reinterpret_cast<TypeExpWVec const*>(params.mPtrScores);
        *reinterpret_cast<TypeExpWVec*>(allScores) = ptrAllScores[scoreOffset];

#pragma unroll
        for (int i = 0; i < VecSize; ++i)
        {
            auto expertIdx = laneIdx * VecSize + i;
            auto newScore = expertIdx < params.mNumExperts ? allScores[i] : minScore;
            // note: use `>=` s.t. highest index always wins, just like in `reduceTopK`
            maxExpertIdx = newScore >= maxScore ? expertIdx : maxExpertIdx;
            maxScore = newScore >= maxScore ? newScore : maxScore;
        }
        int32_t warpMaxExpertIdx[NumTopExperts];
        TypeExpW warpMaxScore[NumTopExperts];
        reduceTopK(warp, warpMaxScore, warpMaxExpertIdx, maxScore, maxExpertIdx, minScore);
        if (cute::elect_one_sync())
        {
            TypePacked packedScore{warpMaxScore[0], static_cast<int16_t>(warpMaxExpertIdx[0])};
            params.mPtrExpertIdx[tokenIdx] = packedScore;
        }
    }
}
#else
__global__ void routingIndicesHistogramScoresKernel(KernelParams params)
{
    assert(false && "routingIndicesHistogramScoresKernel is only supported on SM90+ architectures");
}
#endif

// Two-step approach (if number of tokens exceed limits of what cluster / cooperative launch
// variants can handle): in order to minimize the amount of data to exchange through global memory,
// we will compute the local histograms in smem twice: the first kernel will get us the total number
// of tokens per expert. The second kernel will use the smem and L2 atomics to get corresponding
// element and tile offsets.
//
// Note: the histogram calculation could also be fused with routingMainKernel, but this might be
// inefficient if we have one CTA per token doing a single global atomic.
template <typename KernelParams>
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
__global__ void __launch_bounds__(NumThreadsHist) routingIndicesHistogramKernel(KernelParams params)
{
    using TypeExpW = typename KernelParams::TypeExpW;
    using TypePacked = PackedScoreIdx<TypeExpW>;
    // number of experts is bounded by number of threads
    __shared__ int32_t __attribute((aligned(128))) smemExpertCount[NumThreadsHist];

    // For unrolling.
    uint32_t constexpr NumEltsPerThread = 8;

    // Pre-fill the counts with 0
    if (threadIdx.x < params.mNumExperts)
    {
        smemExpertCount[threadIdx.x] = 0;
    }
    __syncthreads();

    // Wait on primary grid and trigger secondary kernel.
    if constexpr (KernelParams::UsePdl)
    {
        cudaGridDependencySynchronize();
        cudaTriggerProgrammaticLaunchCompletion();
    }

    uint32_t const expandedIdxSize = params.mNumTokens * NumTopExperts;
    uint32_t const localExpertExtent = params.mNumLocalExperts << params.mLocalExpertsStrideLog2;

    uint32_t const gridBlockOffset = blockIdx.x * NumThreadsHist;
    uint32_t const gridStride = gridDim.x * NumThreadsHist;

    // Define a lambda to avoid code duplication in branches.
    auto loopBody = [&](int expandedIdx)
    {
        TypePacked scoreIdx = params.mPtrExpertIdx[expandedIdx];
        // check whether this expert is local to our GPU at all and ignore if not
        auto localExpertIdx = scoreIdx.idx - params.mLocalExpertsStartIdx;
        auto isLocalExpert = localExpertIdx >= 0 && localExpertIdx < localExpertExtent
            && (localExpertIdx & params.mLocalExpertsStrideLog2) == 0;
        if (isLocalExpert)
        {
            atomicAdd(&smemExpertCount[scoreIdx.idx], 1);
        }
        auto finalScore = TypeExpW{sigmoid_accurate(float{scoreIdx.score})};
        if (params.mPtrExpertWeights != nullptr)
        {
            params.mPtrExpertWeights[expandedIdx] = finalScore;
        }
    };

    // Grid-stride loop.
    for (uint32_t expandedIdx0 = gridBlockOffset * NumEltsPerThread; expandedIdx0 < expandedIdxSize;
         expandedIdx0 += gridStride * NumEltsPerThread)
    {
        // Fast path if bound checks aren't necessary
        if (expandedIdx0 + NumEltsPerThread * NumThreadsHist <= expandedIdxSize)
        {
#pragma unroll
            for (uint32_t ii = 0; ii < NumEltsPerThread; ii++)
            {
                uint32_t expandedIdx = expandedIdx0 + ii * NumThreadsHist + threadIdx.x;
                loopBody(expandedIdx);
            }
        }
        else
        {
            for (uint32_t expandedIdx = expandedIdx0 + threadIdx.x; expandedIdx < expandedIdxSize;
                 expandedIdx += NumThreadsHist)
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
#else
__global__ void routingIndicesHistogramKernel(KernelParams params)
{
    assert(false && "routingIndicesHistogramKernel is only supported on SM90+ architectures");
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KernelParams>
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
__global__ void __launch_bounds__(NumThreadsHist) routingIndicesOffsetsKernel(KernelParams params)
{
    using TypeExpW = typename KernelParams::TypeExpW;
    using TypePacked = PackedScoreIdx<TypeExpW>;
    // number of experts is bounded by number of threads
    __shared__ int32_t __attribute((aligned(128))) smemExpertOffset[NumThreadsHist];
    __shared__ int32_t __attribute((aligned(128))) smemExpertCount[NumThreadsHist];
    __shared__ int32_t __attribute((aligned(128))) smemExpertTileOffset[NumThreadsHist];
    // needed for the exclusive sum of token offsets
    using Scan = cub::BlockScan<int32_t, NumThreadsHist, cub::BLOCK_SCAN_WARP_SCANS>;
    __shared__ typename Scan::TempStorage tempStorage;
    static constexpr int MaxExpandedIdxPerThread = NumEltsPerOffsetTilePerThread;
    static constexpr int MaxExpandedIdxPerBlock = NumThreadsHist * MaxExpandedIdxPerThread;

    int32_t const warpIdx = __shfl_sync(0xffffffff, threadIdx.x / WarpSize, 0);

    uint32_t const expandedIdxSize = params.mNumTokens * NumTopExperts;
    uint32_t const numTiles = (expandedIdxSize + MaxExpandedIdxPerBlock - 1) / (MaxExpandedIdxPerBlock);

    // Wait on primary grid.
    if constexpr (KernelParams::UsePdl)
    {
        cudaGridDependencySynchronize();
    }

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
    // Weather or not an expert is local is taken into account when the histogram is computed
    // so we do not need to take it into account here.
    const int32_t numCta = divUpLog2<int32_t>(count, params.mPaddingLog2);
    int32_t ctaOffset;
    int32_t numNonExitingCtas;
    Scan(tempStorage).ExclusiveSum(numCta, ctaOffset, numNonExitingCtas);

    if (threadIdx.x < params.mNumExperts)
    {
        // Get the padded offset associated with this expert
        const int32_t offset = mulLog2<int32_t>(ctaOffset, params.mPaddingLog2);

        // Write expert offsets to shared
        smemExpertOffset[threadIdx.x] = offset;
    }

    // Sync to make expert offsets available to all threads.
    __syncthreads();

    // The first block writes out padded count
    if (blockIdx.x == 0 && warpIdx == NumWarpsHist - 1 && cute::elect_one_sync())
    {
        const int32_t permutedIdxSize = mulLog2<int32_t>(numNonExitingCtas, params.mPaddingLog2);
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
            params.mPtrCtaIdxXyToMnLimit[ctaOffset + cta]
                = min(mulLog2<int32_t>(ctaOffset + cta + 1, params.mPaddingLog2),
                    mulLog2<int32_t>(ctaOffset, params.mPaddingLog2) + count);
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
            TypePacked scoreIdx = params.mPtrExpertIdx[expandedIdx];
            expertIndexes[ii] = scoreIdx.idx;
            // check whether this expert is local to our GPU at all and ignore if not
            auto localExpertIdx = scoreIdx.idx - params.mLocalExpertsStartIdx;
            auto isLocalExpert = localExpertIdx >= 0 && localExpertIdx < localExpertExtent
                && (localExpertIdx & params.mLocalExpertsStrideLog2) == 0;
            expertOffsets[ii] = isLocalExpert ? atomicAdd(smemExpertCount + scoreIdx.idx, 1) : 0;
        };

        // For all tiles but the last, all indices are in bounds.
        if (tileIdx < numTiles - 1)
        {
#pragma unroll
            for (int32_t ii = 0; ii < MaxExpandedIdxPerThread; ii += 1)
            {
                auto expandedIdx = tileIdx * MaxExpandedIdxPerBlock + ii * NumThreadsHist + threadIdx.x;
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
                    = tileIdx * MaxExpandedIdxPerBlock + (ii0 + IterStride) * NumThreadsHist <= expandedIdxSize;
                if (takeFastPath)
                {
#pragma unroll
                    for (int32_t jj = 0; jj < IterStride; jj++)
                    {
                        int const ii = ii0 + jj;
                        auto expandedIdx = tileIdx * MaxExpandedIdxPerBlock + ii * NumThreadsHist + threadIdx.x;
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
                        auto expandedIdx = tileIdx * MaxExpandedIdxPerBlock + ii * NumThreadsHist + threadIdx.x;
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
            auto tokenIdx = expandedIdx / NumTopExperts;
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
                auto expandedIdx = tileIdx * MaxExpandedIdxPerBlock + ii * NumThreadsHist + threadIdx.x;
                storeLoopBody(ii, expandedIdx);
            }
        }
        else
        {
#pragma unroll
            for (int32_t ii = 0; ii < MaxExpandedIdxPerThread; ii += 1)
            {
                auto expandedIdx = tileIdx * MaxExpandedIdxPerBlock + ii * NumThreadsHist + threadIdx.x;
                if (expandedIdx >= expandedIdxSize)
                {
                    break;
                }
                storeLoopBody(ii, expandedIdx);
            }
        }
    }

// Trigger secondary kernel.
// Note: this does not guarantee the visibility of prior writes unless the consumer executes a
// dependency sync.
#if !defined(PDL_PROFILE) || PDL_PROFILE == 0
    if constexpr (KernelParams::UsePdl)
    {
        cudaTriggerProgrammaticLaunchCompletion();
    }
#endif
}
#else
__global__ void routingIndicesOffsetsKernel(KernelParams params)
{
    assert(false && "routingIndicesOffsetsKernel is only supported on SM90+ architectures");
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

void run(Data const& data, void* stream)
{
    TLLM_CHECK_WITH_INFO(data.mPtrExpertIdx != nullptr || data.mPtrScores != nullptr,
        "Routing kernel requires at least one input parameter");
    TLLM_CHECK_WITH_INFO(data.mPtrPermutedIdxSize != nullptr && data.mPtrCtaIdxXyToBatchIdx != nullptr
            && data.mPtrCtaIdxXyToMnLimit != nullptr && data.mPtrNumNonExitingCtas != nullptr,
        "Llama4 routing kernel expects permuted idx and grouped Gemm launch config buffers");
    TLLM_CHECK_WITH_INFO(
        data.mTopK == NumTopExperts, "Routing kernel expects %d topK experts (for now)", NumTopExperts);
    TLLM_CHECK_WITH_INFO(data.mNumExperts <= MaxNumExperts,
        "Routing kernel expects #experts %d to be at most max #experts %d", data.mNumExperts, MaxNumExperts);
    static_assert(MaxNumExperts <= NumThreads, "#experts must be bounded by #threads");
    static_assert(MaxNumExperts <= NumThreadsHist, "#experts must be bounded by #threads");
    TLLM_CHECK_WITH_INFO(
        data.mNumExperts % 4 == 0, "Routing kernel expects #experts %d to be a multiple of 4.", data.mNumExperts);
    TLLM_CHECK_WITH_INFO(data.mPaddingLog2 < 8, "Routing kernel expects padding log2 < 8, got %d", data.mPaddingLog2);

    if (data.mPtrPermutedIdxToTokenIdx != nullptr)
    {
        // need to set all values to -1 before running the kernel
        auto maxPermutedSize
            = data.mNumTokens * data.mTopK + (data.mNumExperts << data.mPaddingLog2) - data.mNumExperts;
        // note that a value of -1 per byte works for any size of signed integer
        // to set each full value to the logical value -1
        TLLM_CUDA_CHECK(cudaMemsetAsync(data.mPtrPermutedIdxToTokenIdx, -1,
            static_cast<size_t>(maxPermutedSize) * sizeof(int32_t), (cudaStream_t) stream));
    }

    bool const useSingleWarp = (data.mPtrScores == nullptr && data.mNumTokens <= WarpKernelMaxNumTokens)
        || data.mNumTokens < WarpKernelMaxNumTokens;
    bool const useSingleCluster
        = data.mNumTokens <= (data.mPtrScores != nullptr ? MaxNumTokensSingleClusterScores : MaxNumTokensSingleCluster);
    if (!useSingleCluster)
    {
        TLLM_CHECK_WITH_INFO(
            data.mPtrExpertIdx != nullptr, "When #tokens is large, `mPtrExpertIdx` is a required input.");
        TLLM_CHECK_WITH_INFO(
            data.mPtrExpertCounts != nullptr, "When #tokens is large, `mPtrExpertCounts` is a required input.");
        // Reset the global histograms (not used in single-cluster code path).
        TLLM_CUDA_CHECK(cudaMemsetAsync(data.mPtrExpertCounts, 0,
            static_cast<size_t>(2 * data.mNumExperts) * sizeof(int32_t), (cudaStream_t) stream));
    }

    if (useSingleWarp)
    {
        LAUNCH_EXPW_ONLY(data,
            /*coopLaunch=*/false, routingIndicesWarpKernel, 1, WarpSize,
            /*smemSize=*/0, // No dynamic smem
            stream);
    }
    else if (useSingleCluster)
    {
        LAUNCH_EXPW_ONLY(data,
            /*coopLaunch=*/false, routingIndicesClusterKernel, NumBlocksPerCluster, NumThreads,
            /*smemSize=*/0, // No dynamic smem
            stream);
    }
    else
    {
        const uint32_t expandedIdxSize = data.mNumTokens * NumTopExperts;

        const uint32_t histogramEltsPerBlock = 8 * NumThreadsHist;
        const uint32_t offsetEltsPerBlock = NumEltsPerOffsetTilePerThread * NumThreadsHist;

        // Limit grid size (all kernels use a grid-stride loop).
        const uint32_t maxNumBlocks = 1024;

        int const numBlocksHistogram
            = std::min((expandedIdxSize + histogramEltsPerBlock - 1) / histogramEltsPerBlock, maxNumBlocks);
        int const numBlocksOffsets
            = std::min((expandedIdxSize + offsetEltsPerBlock - 1) / offsetEltsPerBlock, maxNumBlocks);

        if (data.mPtrScores != nullptr)
        {
            LAUNCH_EXPW_ONLY(data,
                /*coopLaunch=*/false, routingIndicesHistogramScoresKernel, maxNumBlocks, NumThreadsHist,
                /*smemSize=*/0, // No dynamic smem
                stream);
        }
        LAUNCH_EXPW_ONLY(data,
            /*coopLaunch=*/false, routingIndicesHistogramKernel, numBlocksHistogram, NumThreadsHist,
            /*smemSize=*/0, // No dynamic smem
            stream);
        LAUNCH_EXPW_ONLY(data,
            /*coopLaunch=*/false, routingIndicesOffsetsKernel, numBlocksOffsets, NumThreadsHist,
            /*smemSize=*/0, // No dynamic smem
            stream);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace routingLlama4

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace routingQwen3
{

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace cg = cooperative_groups;

////////////////////////////////////////////////////////////////////////////////////////////////////

static constexpr int NumThreads = 1024;
static constexpr int NumThreadsHist = 256;
static constexpr int NumBlocksPerCluster = 8;
static constexpr int WarpSize = 32;
static constexpr int NumWarps = NumThreads / WarpSize;
static constexpr int NumWarpsHist = NumThreadsHist / WarpSize;
static constexpr int NumTopExperts = 8;
static constexpr int MaxNumExperts = 128;
static constexpr int MaxNumTokensSingleCluster = NumBlocksPerCluster * NumThreads;
static constexpr int MaxNumTokensSingleClusterScores = NumBlocksPerCluster * NumWarps;

// Performance tuning knob.
static constexpr int NumEltsPerOffsetTilePerThread = 8;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000 && defined(__CUDA_ARCH_FEAT_SM100_ALL))
#define TLLM_GEN_ENABLE_FAST_REDUX
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename TypeExpW_>
struct TopKRedType
{
    using TypeExpW = TypeExpW_;
    static_assert(std::is_same_v<TypeExpW, float> || std::is_same_v<TypeExpW, cutlass::bfloat16_t>,
        "Top K reduction only implemented for float and Bf16");
    using TypeCmp = std::conditional_t<sizeof(TypeExpW) >= 4, double, float>;
    static constexpr int64_t Mask64 = 0x000000000000FFFF;
    static constexpr int32_t Mask32 = 0x0000FFFF;

    TypeCmp compVal;

    static __host__ __device__ inline TypeCmp makeCmpVal(TypeExpW val, int32_t idx = 0)
    {
        auto cmpVal = TypeCmp{val};
        TypeCmp cmpValWithIdx;
        if constexpr (sizeof(TypeExpW) >= 4)
        {
            auto cmpValIdx64 = reinterpret_cast<int64_t&>(cmpVal) | (Mask64& int64_t{idx});
            cmpValWithIdx = reinterpret_cast<TypeCmp&>(cmpValIdx64);
        }
        else
        {
            auto cmpValIdx32 = reinterpret_cast<int32_t&>(cmpVal) | (Mask32 & idx);
            cmpValWithIdx = reinterpret_cast<TypeCmp&>(cmpValIdx32);
        }
        return cmpValWithIdx;
    }

    static __host__ __device__ inline void unpack(TypeExpW& val, int32_t& idx, TypeCmp cmp)
    {
        if constexpr (sizeof(TypeExpW) >= 4)
        {
            idx = static_cast<int32_t>(reinterpret_cast<int64_t&>(cmp) & Mask64);
            auto val64 = reinterpret_cast<int64_t&>(cmp) & ~Mask64;
            val = static_cast<float>(reinterpret_cast<double&>(val64));
        }
        else
        {
            idx = reinterpret_cast<int32_t&>(cmp) & Mask32;
            auto val32 = reinterpret_cast<int32_t&>(cmp) >> 16;
            val = TypeExpW::bitcast(reinterpret_cast<uint16_t&>(val32));
        }
    }

    __host__ __device__ TopKRedType() = default;

    __host__ __device__ TopKRedType(TypeExpW val, int32_t idx)
        : compVal(makeCmpVal(val, idx))
    {
    }

    __host__ __device__ operator TypeCmp() const noexcept
    {
        return compVal;
    }

    __device__ inline TypeCmp reduce(cg::thread_block_tile<WarpSize> const& warp)
    {
#if defined(TLLM_GEN_ENABLE_FAST_REDUX)
        static constexpr bool UseCg = false;
#else
        static constexpr bool UseCg = true;
#endif
        if constexpr (UseCg || sizeof(TypeExpW) >= 4)
        {
            return cg::reduce(warp, compVal, cg::greater<TypeCmp>{});
        }
        else
        {
            float result;
            asm("redux.sync.max.f32 %0, %1, 0xffffffff;\n" : "=f"(result) : "f"(compVal));
            return result;
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int K_, bool Enable_>
struct TopKIdx
{
    // by default, empty
};

template <int K_>
struct TopKIdx<K_, true>
{
    static constexpr int K = K_;
    int32_t val[K];
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int K, typename Type>
__device__ void reduceTopK(cg::thread_block_tile<WarpSize> const& warp, Type (&out)[K], int32_t (&outIdx)[K],
    Type value, int32_t idx, Type minValue)
{
    static_assert(K > 0, "Top K must have K > 0");
    static_assert(K < WarpSize, "Top K must have K < WarpSize");
    using RedType = TopKRedType<Type>;
    RedType topK{value, idx};
    typename RedType::TypeCmp packedMax{};
#pragma unroll
    for (int kk = 0; kk < K; ++kk)
    {
        topK = kk > 0 && packedMax == topK.compVal ? RedType{minValue, idx} : topK;
        // get the next largest value
        packedMax = topK.reduce(warp);
        RedType::unpack(out[kk], outIdx[kk], packedMax);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

#define TOPK_SWAP(I, J)                                                                                                \
    {                                                                                                                  \
        auto pairMin = min(topK[I].compVal, topK[J].compVal);                                                          \
        auto pairMax = max(topK[I].compVal, topK[J].compVal);                                                          \
        topK[I].compVal = pairMax;                                                                                     \
        topK[J].compVal = pairMin;                                                                                     \
    }

template <int K, typename Type, int N, bool IsSorted = false>
__device__ void reduceTopK(cg::thread_block_tile<WarpSize> const& warp, Type (&out)[K], int32_t (&outIdx)[K],
    Type (&value)[N], int32_t (&idx)[N], Type minValue)
{
    static_assert(K > 0, "Top K must have K > 0");
    static_assert(K < WarpSize, "Top K must have K < WarpSize");
    static_assert(N > 0, "Top K must have N > 1");
    // static_assert(N <= K, "Top K must have N < K");
    using RedType = TopKRedType<Type>;
    RedType topK[N];
#pragma unroll
    for (int nn = 0; nn < N; ++nn)
    {
        topK[nn] = RedType{value[nn], idx[nn]};
    }

    if constexpr (!IsSorted)
    {
        TOPK_SWAP(0, 2);
        TOPK_SWAP(1, 3);

        TOPK_SWAP(0, 1);
        TOPK_SWAP(2, 3);

        TOPK_SWAP(1, 2);
    }
    typename RedType::TypeCmp packedMax{};
#pragma unroll
    for (int kk = 0; kk < K; ++kk)
    {
        bool update = kk > 0 && packedMax == topK[0].compVal;
#pragma unroll
        for (int nn = 0; nn < N; ++nn)
        {
            topK[nn] = update && nn == N - 1 ? RedType{minValue, idx[nn]} : update ? topK[nn + 1] : topK[nn];
        }
        // get the next largest value
        packedMax = topK[0].reduce(warp);
        RedType::unpack(out[kk], outIdx[kk], packedMax);
    }
};

#undef TOPK_SWAP

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

template <typename TypeExpW, int VecSize>
__device__ void calcSoftmax(cg::thread_block_tile<WarpSize> const& warp, TypeExpW (&scores)[VecSize])
{
    TypeExpW maxScore = TypeExpW{-INFINITY};
    TypeExpW sumScore = TypeExpW{0.f};

    // Get the max score for each token
    for (int i = 0; i < VecSize; ++i)
    {
        maxScore = scores[i] >= maxScore ? scores[i] : maxScore;
    }
    maxScore = cg::reduce(warp, maxScore, cg::greater<TypeExpW>());

    // Get the summation of scores for each token
#pragma unroll
    for (int i = 0; i < VecSize; ++i)
    {
        scores[i] = static_cast<TypeExpW>(exp(scores[i] - maxScore));
        sumScore += scores[i];
    }
    sumScore = cg::reduce(warp, sumScore, cg::plus<TypeExpW>());

    // Normalize the scores
#pragma unroll
    for (int i = 0; i < VecSize; ++i)
    {
        scores[i] = static_cast<TypeExpW>(scores[i] / sumScore);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename TypeExpW>
__device__ TypeExpW calcSoftmax(
    cg::thread_block_tile<WarpSize> const& warp, TypeExpW score, int32_t laneIdx, int32_t NumTopExperts)
{
    TypeExpW maxScore = TypeExpW{-INFINITY};
    if (laneIdx < NumTopExperts)
    {
        maxScore = score >= maxScore ? score : maxScore;
    }
    maxScore = cg::reduce(warp, maxScore, cg::greater<TypeExpW>());

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
        score = static_cast<TypeExpW>(newScore / sumScore);
    }

    return score;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KernelParams, bool DoSoftmaxBeforeTopK = false>
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
__global__ void __cluster_dims__(NumBlocksPerCluster, 1, 1) __launch_bounds__(NumThreads)
    routingIndicesClusterKernel(KernelParams params)
{
    // number of experts is bounded by number of threads
    __shared__ int32_t __attribute((aligned(128))) smemExpertCount[NumThreads];
    __shared__ int32_t __attribute((aligned(128))) smemExpertOffset[NumThreads];
    // number of tokens/expanded idx is bounded by total number of warps
    using TypeExpW = typename KernelParams::TypeExpW;

    using BaseType = std::conditional_t<DoSoftmaxBeforeTopK, float, TypeExpW>;
    using TypePacked = PackedScoreIdx<BaseType>;

    __shared__ TypePacked __attribute((aligned(128))) smemPackedScoreIdx[NumWarps * NumTopExperts];
    // Needed for the exclusive sum of token offsets.
    // Note: the scan might include more bins than needed, with bin counts of 0 to pad
    using Scan = cub::BlockScan<int32_t, NumThreads, cub::BLOCK_SCAN_WARP_SCANS>;
    __shared__ typename Scan::TempStorage tempStorage;
    // Number of threads in the cluster.
    static constexpr int NumThreadsPerCluster = NumThreads * NumBlocksPerCluster;
    // same as max num tokens*num top experts
    static constexpr int MaxExpandedIdxPerThread
        = (MaxNumTokensSingleCluster * NumTopExperts + NumThreadsPerCluster - 1) / NumThreadsPerCluster;

    uint32_t const clusterBlockRank = blockIdx.x;
    uint32_t const clusterThreadIdx = NumThreads * clusterBlockRank + threadIdx.x;

    int32_t const warpIdx = __shfl_sync(0xffffffff, threadIdx.x / WarpSize, 0);
    int32_t const laneIdx = cutlass::arch::LaneId();

    auto expandedIdxSize = params.mNumTokens * NumTopExperts;
    auto warpTokenIdx = clusterBlockRank * NumWarps + warpIdx;
    auto scoreOffset = warpTokenIdx * params.mNumExperts;
    bool validToken = warpTokenIdx < params.mNumTokens;

    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<WarpSize>(block);

    // pre-fill the counts with 0
    if (threadIdx.x < params.mNumExperts)
    {
        smemExpertCount[threadIdx.x] = 0;
    }
    __syncthreads();

    // then wait on primary grid
    if constexpr (KernelParams::UsePdl)
    {
        cudaGridDependencySynchronize();
    }

    // initialize the mPtrPermutedIdxToTokenIdx
    if (params.mPtrPermutedIdxToTokenIdx != nullptr)
    {
        int32_t permIdxToTokenIdxNum
            = (params.mNumTokens * NumTopExperts + (params.mNumExperts << params.mPaddingLog2) - params.mNumExperts);
        for (int32_t i = clusterThreadIdx; i < permIdxToTokenIdxNum; i += NumThreadsPerCluster)
        {
            params.mPtrPermutedIdxToTokenIdx[i] = -1;
        }
        // A cluster synchronization is performed prior to setting mPtrPermutedIdxToTokenIdx at the end of the kernel.
        // Don't need to use __threadfence() here.
    }

    if (params.mPtrScores != nullptr)
    {
        // in this case, each warp represents a token
        BaseType score[MaxNumExperts / WarpSize];
        int32_t idx[MaxNumExperts / WarpSize];

        BaseType warpTopKScore[NumTopExperts];
        int32_t warpTopKExpertIdx[NumTopExperts];

        BaseType minScore = BaseType{-INFINITY};
        if (validToken)
        {
            for (int i = 0; i < MaxNumExperts / WarpSize; i++)
            {
                auto expertIdx = i * WarpSize + laneIdx;
                auto newScore = expertIdx < params.mNumExperts
                    ? static_cast<BaseType>(params.mPtrScores[scoreOffset + expertIdx])
                    : minScore;
                score[i] = newScore;
                idx[i] = expertIdx;
            }

            if constexpr (DoSoftmaxBeforeTopK)
            {
                calcSoftmax(warp, score);
            }

            // Get the top-k scores and their corresponding expert indices
            reduceTopK(warp, warpTopKScore, warpTopKExpertIdx, score, idx, minScore);

            // Normalize the scores
            if constexpr (DoSoftmaxBeforeTopK)
            {
                float sum = float{1.f};
                if (params.mNormTopkProb)
                {
                    sum = static_cast<float>(laneIdx < NumTopExperts ? warpTopKScore[laneIdx] : 0);
                    sum = cg::reduce(warp, sum, cg::plus<float>());
                }
                if (laneIdx < NumTopExperts)
                {
                    warpTopKScore[laneIdx] = warpTopKScore[laneIdx] / sum;
                    smemPackedScoreIdx[warpIdx * NumTopExperts + laneIdx]
                        = TypePacked{warpTopKScore[laneIdx], static_cast<int16_t>(warpTopKExpertIdx[laneIdx])};
                }
            }
            else
            {
                auto score = calcSoftmax(
                    warp, laneIdx < NumTopExperts ? warpTopKScore[laneIdx] : minScore, laneIdx, NumTopExperts);
                if (laneIdx < NumTopExperts)
                {
                    warpTopKScore[laneIdx] = score;
                    smemPackedScoreIdx[warpIdx * NumTopExperts + laneIdx]
                        = TypePacked{warpTopKScore[laneIdx], static_cast<int16_t>(warpTopKExpertIdx[laneIdx])};
                }
            }
        } // end if (validToken)

        // make packed scores available to all threads in cluster
        __cluster_barrier_arrive();
        __cluster_barrier_wait();
    }

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
        if (params.mPtrScores != nullptr)
        {
            TypePacked const* remoteSmem
                = cg::cluster_group::map_shared_rank(smemPackedScoreIdx, expandedIdx / (NumWarps * NumTopExperts));
            scoreIdx = remoteSmem[expandedIdx % (NumWarps * NumTopExperts)];
        }
        else
        {
            scoreIdx = TypePacked{static_cast<BaseType>(params.mPtrExpertIdx[expandedIdx].score),
                static_cast<int16_t>(params.mPtrExpertIdx[expandedIdx].idx)};
        }
        expertIndexes[ii] = scoreIdx.idx;
        // check whether this expert is local to our GPU at all and ignore if not
        auto localExpertIdx = scoreIdx.idx - params.mLocalExpertsStartIdx;
        auto isLocalExpert = localExpertIdx >= 0 && localExpertIdx < localExpertExtent
            && (localExpertIdx & params.mLocalExpertsStrideLog2) == 0;
        expertOffsets[ii] = isLocalExpert ? atomicAdd(smemExpertCount + scoreIdx.idx, 1) : 0;
        if (params.mPtrExpertWeights != nullptr)
        {
            params.mPtrExpertWeights[expandedIdx] = static_cast<TypeExpW>(scoreIdx.score);
        }
    };

    if (clusterThreadIdx < expandedIdxSize)
    {
        loopBody(0, clusterThreadIdx);
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
    const int32_t numCta = divUpLog2<int32_t>(count, params.mPaddingLog2);
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
            params.mPtrCtaIdxXyToMnLimit[ctaOffset + cta]
                = min(mulLog2<int32_t>(ctaOffset + cta + 1, params.mPaddingLog2),
                    mulLog2<int32_t>(ctaOffset, params.mPaddingLog2) + count);
        }

        // get the padded offset associated with this expert
        const int32_t offset = mulLog2<int32_t>(ctaOffset, params.mPaddingLog2);

        // write expert offsets to shared
        smemExpertOffset[threadIdx.x] = offset + blockExpertOffset;
    }

    // write out padded count
    if (clusterBlockRank == 0 && warpIdx == NumWarps - 1 && cute::elect_one_sync())
    {
        const int32_t permutedIdxSize = mulLog2<int32_t>(numNonExitingCtas, params.mPaddingLog2);
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
#if !defined(PDL_PROFILE) || PDL_PROFILE == 0
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
        auto tokenIdx = expandedIdx / NumTopExperts;
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
#else
__global__ void __launch_bounds__(NumThreads) routingIndicesClusterKernel(KernelParams /* params */)
{
    assert(false && "routingIndicesClusterKernel is only supported on SM90+ architectures");
}
#endif // if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
////////////////////////////////////////////////////////////////////////////////////////////////////

// this kernel is needed in case we have scores as input for the histogram kernel
template <typename KernelParams, bool DoSoftmaxBeforeTopK = true>
__global__ void __launch_bounds__(NumThreadsHist) routingIndicesHistogramScoresKernel(KernelParams params)
{
    using TypeExpW = typename KernelParams::TypeExpW;

    using BaseType = std::conditional_t<DoSoftmaxBeforeTopK, float, TypeExpW>;

    static constexpr int VecSize = MaxNumExperts / WarpSize;
    // we assume that #experts is a multiple of 4, so VecSize must be 4.
    static_assert(VecSize == 4);

    int32_t const laneIdx = cutlass::arch::LaneId();
    int32_t const warpIdx = threadIdx.x / WarpSize;
    int32_t const globalWarpIdx = blockIdx.x * NumWarpsHist + warpIdx;
    int32_t const globalWarpStride = gridDim.x * NumWarpsHist;
    BaseType minScore = BaseType{-INFINITY};
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<WarpSize>(block);

    // Wait on primary grid.
    if constexpr (KernelParams::UsePdl)
    {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
        cudaGridDependencySynchronize();
#endif // if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    }

    // initialize the mPtrPermutedIdxToTokenIdx
    int32_t globalThreadIdx = globalWarpIdx * WarpSize + laneIdx;
    int32_t globalThreadStride = globalWarpStride * WarpSize;
    if (params.mPtrPermutedIdxToTokenIdx != nullptr)
    {
        int32_t permIdxToTokenIdxNum
            = (params.mNumTokens * NumTopExperts + (params.mNumExperts << params.mPaddingLog2) - params.mNumExperts);
        for (int32_t i = globalThreadIdx; i < permIdxToTokenIdxNum; i += globalThreadStride)
        {
            params.mPtrPermutedIdxToTokenIdx[i] = -1;
        }
    }

    // initialize the mPtrExpertCounts
    if (params.mPtrExpertCounts != nullptr)
    {
        int32_t expertCountsNum = 2 * params.mNumExperts;
        for (int32_t i = globalThreadIdx; i < expertCountsNum; i += globalThreadStride)
        {
            params.mPtrExpertCounts[i] = 0;
        }
    }

    // Trigger secondary kernel.
    if constexpr (KernelParams::UsePdl)
    {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
        cudaTriggerProgrammaticLaunchCompletion();
#endif // if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    }

    // in this case, each warp represents a token, and we use a grid-stride loop
    // over all warps/tokens
    for (int tokenIdx = globalWarpIdx; tokenIdx < params.mNumTokens; tokenIdx += globalWarpStride)
    {
        auto scoreOffset = tokenIdx * params.mNumExperts;
        BaseType allScores[VecSize];
        int32_t allExpertIdx[VecSize];
        BaseType warpTopKScore[NumTopExperts];
        int32_t warpTopKExpertIdx[NumTopExperts];

        //@TODO：optimize this part with vectorized loading

#pragma unroll
        for (int i = 0; i < VecSize; ++i)
        {
            auto expertIdx = i * WarpSize + laneIdx;
            auto newScore = expertIdx < params.mNumExperts
                ? static_cast<BaseType>(params.mPtrScores[scoreOffset + expertIdx])
                : minScore;
            allScores[i] = newScore;
            allExpertIdx[i] = expertIdx;
        }

        if constexpr (DoSoftmaxBeforeTopK)
        {
            calcSoftmax(warp, allScores);
        }

        // Get the top-k scores and their corresponding expert indices
        reduceTopK(warp, warpTopKScore, warpTopKExpertIdx, allScores, allExpertIdx, minScore);
        __syncwarp(); //@TODO: check the synchronization

        // Normalize the scores
        if constexpr (DoSoftmaxBeforeTopK)
        {
            float sum = float{1.f};
            if (params.mNormTopkProb)
            {
                sum = static_cast<float>(laneIdx < NumTopExperts ? warpTopKScore[laneIdx] : 0);
                sum = cg::reduce(warp, sum, cg::plus<float>());
            }
            if (laneIdx < NumTopExperts)
            {
                warpTopKScore[laneIdx] = warpTopKScore[laneIdx] / sum;
            }
        }
        else
        {
            auto score = laneIdx < NumTopExperts ? warpTopKScore[laneIdx] : minScore;
            score = calcSoftmax(warp, score, laneIdx, NumTopExperts);
            if (laneIdx < NumTopExperts)
            {
                warpTopKScore[laneIdx] = score;
            }
        }
        for (int i = laneIdx; i < NumTopExperts; i += WarpSize)
        {
            PackedScoreIdx<TypeExpW> packedScore{
                static_cast<TypeExpW>(warpTopKScore[i]), static_cast<int16_t>(warpTopKExpertIdx[i])};
            params.mPtrExpertIdx[tokenIdx * NumTopExperts + i] = packedScore;
        }
    }
}

// Two-step approach (if number of tokens exceed limits of what cluster / cooperative launch
// variants can handle): in order to minimize the amount of data to exchange through global memory,
// we will compute the local histograms in smem twice: the first kernel will get us the total number
// of tokens per expert. The second kernel will use the smem and L2 atomics to get corresponding
// element and tile offsets.
//
// Note: the histogram calculation could also be fused with routingMainKernel, but this might be
// inefficient if we have one CTA per token doing a single global atomic.
template <typename KernelParams>
__global__ void __launch_bounds__(NumThreadsHist) routingIndicesHistogramKernel(KernelParams params)
{
    using TypeExpW = typename KernelParams::TypeExpW;

    using TypePacked = PackedScoreIdx<float>;
    // number of experts is bounded by number of threads
    __shared__ int32_t __attribute((aligned(128))) smemExpertCount[NumThreadsHist];

    // For unrolling.
    uint32_t constexpr NumEltsPerThread = 8;

    // Pre-fill the counts with 0
    if (threadIdx.x < params.mNumExperts)
    {
        smemExpertCount[threadIdx.x] = 0;
    }
    __syncthreads();

    // Wait on primary grid and trigger secondary kernel.
    if constexpr (KernelParams::UsePdl)
    {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
        cudaGridDependencySynchronize();
        cudaTriggerProgrammaticLaunchCompletion();
#endif // if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    }

    uint32_t const expandedIdxSize = params.mNumTokens * NumTopExperts;
    uint32_t const localExpertExtent = params.mNumLocalExperts << params.mLocalExpertsStrideLog2;

    uint32_t const gridBlockOffset = blockIdx.x * NumThreadsHist;
    uint32_t const gridStride = gridDim.x * NumThreadsHist;

    // Define a lambda to avoid code duplication in branches.
    auto loopBody = [&](int expandedIdx)
    {
        PackedScoreIdx<TypeExpW> scoreIdx = params.mPtrExpertIdx[expandedIdx];
        // check whether this expert is local to our GPU at all and ignore if not
        auto localExpertIdx = scoreIdx.idx - params.mLocalExpertsStartIdx;
        auto isLocalExpert = localExpertIdx >= 0 && localExpertIdx < localExpertExtent
            && (localExpertIdx & params.mLocalExpertsStrideLog2) == 0;
        if (isLocalExpert)
        {
            atomicAdd(&smemExpertCount[scoreIdx.idx], 1);
        }

        if (params.mPtrExpertWeights != nullptr)
        {
            params.mPtrExpertWeights[expandedIdx] = static_cast<TypeExpW>(scoreIdx.score);
        }
    };

    // Grid-stride loop.
    for (uint32_t expandedIdx0 = gridBlockOffset * NumEltsPerThread; expandedIdx0 < expandedIdxSize;
         expandedIdx0 += gridStride * NumEltsPerThread)
    {
        // Fast path if bound checks aren't necessary
        if (expandedIdx0 + NumEltsPerThread * NumThreadsHist <= expandedIdxSize)
        {
#pragma unroll
            for (uint32_t ii = 0; ii < NumEltsPerThread; ii++)
            {
                uint32_t expandedIdx = expandedIdx0 + ii * NumThreadsHist + threadIdx.x;
                loopBody(expandedIdx);
            }
        }
        else
        {
            for (uint32_t expandedIdx = expandedIdx0 + threadIdx.x; expandedIdx < expandedIdxSize;
                 expandedIdx += NumThreadsHist)
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
__global__ void __launch_bounds__(NumThreadsHist) routingIndicesOffsetsKernel(KernelParams params)
{
    using TypeExpW = typename KernelParams::TypeExpW;
    using TypePacked = PackedScoreIdx<TypeExpW>;

    // number of experts is bounded by number of threads
    __shared__ int32_t __attribute((aligned(128))) smemExpertOffset[NumThreadsHist];
    __shared__ int32_t __attribute((aligned(128))) smemExpertCount[NumThreadsHist];
    __shared__ int32_t __attribute((aligned(128))) smemExpertTileOffset[NumThreadsHist];
    // needed for the exclusive sum of token offsets
    using Scan = cub::BlockScan<int32_t, NumThreadsHist, cub::BLOCK_SCAN_WARP_SCANS>;
    __shared__ typename Scan::TempStorage tempStorage;
    static constexpr int MaxExpandedIdxPerThread = NumEltsPerOffsetTilePerThread;
    static constexpr int MaxExpandedIdxPerBlock = NumThreadsHist * MaxExpandedIdxPerThread;

    int32_t const warpIdx = __shfl_sync(0xffffffff, threadIdx.x / WarpSize, 0);

    uint32_t const expandedIdxSize = params.mNumTokens * NumTopExperts;
    uint32_t const numTiles = (expandedIdxSize + MaxExpandedIdxPerBlock - 1) / (MaxExpandedIdxPerBlock);

    // Wait on primary grid.
    if constexpr (KernelParams::UsePdl)
    {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
        cudaGridDependencySynchronize();
#endif // if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    }

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
    const int32_t numCta = divUpLog2<int32_t>(count, params.mPaddingLog2);
    int32_t ctaOffset;
    int32_t numNonExitingCtas;
    Scan(tempStorage).ExclusiveSum(numCta, ctaOffset, numNonExitingCtas);

    if (threadIdx.x < params.mNumExperts)
    {
        // Get the padded offset associated with this expert
        const int32_t offset = mulLog2<int32_t>(ctaOffset, params.mPaddingLog2);

        // Write expert offsets to shared
        smemExpertOffset[threadIdx.x] = offset;
    }

    // Sync to make expert offsets available to all threads.
    __syncthreads();

    // The first block writes out padded count
    if (blockIdx.x == 0 && warpIdx == NumWarpsHist - 1 && cute::elect_one_sync())
    {
        const int32_t permutedIdxSize = mulLog2<int32_t>(numNonExitingCtas, params.mPaddingLog2);
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
            params.mPtrCtaIdxXyToMnLimit[ctaOffset + cta]
                = min(mulLog2<int32_t>(ctaOffset + cta + 1, params.mPaddingLog2),
                    mulLog2<int32_t>(ctaOffset, params.mPaddingLog2) + count);
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
            PackedScoreIdx<TypeExpW> scoreIdx = params.mPtrExpertIdx[expandedIdx];
            expertIndexes[ii] = scoreIdx.idx;
            // check whether this expert is local to our GPU at all and ignore if not
            auto localExpertIdx = scoreIdx.idx - params.mLocalExpertsStartIdx;
            auto isLocalExpert = localExpertIdx >= 0 && localExpertIdx < localExpertExtent
                && (localExpertIdx & params.mLocalExpertsStrideLog2) == 0;
            expertOffsets[ii] = isLocalExpert ? atomicAdd(smemExpertCount + scoreIdx.idx, 1) : 0;
        };

        // For all tiles but the last, all indices are in bounds.
        if (tileIdx < numTiles - 1)
        {
#pragma unroll
            for (int32_t ii = 0; ii < MaxExpandedIdxPerThread; ii += 1)
            {
                auto expandedIdx = tileIdx * MaxExpandedIdxPerBlock + ii * NumThreadsHist + threadIdx.x;
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
                    = tileIdx * MaxExpandedIdxPerBlock + (ii0 + IterStride) * NumThreadsHist <= expandedIdxSize;
                if (takeFastPath)
                {
#pragma unroll
                    for (int32_t jj = 0; jj < IterStride; jj++)
                    {
                        int const ii = ii0 + jj;
                        auto expandedIdx = tileIdx * MaxExpandedIdxPerBlock + ii * NumThreadsHist + threadIdx.x;
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
                        auto expandedIdx = tileIdx * MaxExpandedIdxPerBlock + ii * NumThreadsHist + threadIdx.x;
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
            auto tokenIdx = expandedIdx / NumTopExperts;
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
                auto expandedIdx = tileIdx * MaxExpandedIdxPerBlock + ii * NumThreadsHist + threadIdx.x;
                storeLoopBody(ii, expandedIdx);
            }
        }
        else
        {
#pragma unroll
            for (int32_t ii = 0; ii < MaxExpandedIdxPerThread; ii += 1)
            {
                auto expandedIdx = tileIdx * MaxExpandedIdxPerBlock + ii * NumThreadsHist + threadIdx.x;
                if (expandedIdx >= expandedIdxSize)
                {
                    break;
                }
                storeLoopBody(ii, expandedIdx);
            }
        }
    }

// Trigger secondary kernel.
// Note: this does not guarantee the visibility of prior writes unless the consumer executes a
// dependency sync.
#if !defined(PDL_PROFILE) || PDL_PROFILE == 0
    if constexpr (KernelParams::UsePdl)
    {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
        cudaTriggerProgrammaticLaunchCompletion();
#endif // if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    }
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void run(Data const& data, void* stream)
{
    TLLM_CHECK_WITH_INFO(data.mPtrExpertIdx != nullptr || data.mPtrScores != nullptr,
        "Routing kernel requires at least one input parameter");
    TLLM_CHECK_WITH_INFO(data.mPtrPermutedIdxSize != nullptr && data.mPtrCtaIdxXyToBatchIdx != nullptr
            && data.mPtrCtaIdxXyToMnLimit != nullptr && data.mPtrNumNonExitingCtas != nullptr,
        "Llama4 routing kernel expects permuted idx and grouped Gemm launch config buffers");
    TLLM_CHECK_WITH_INFO(
        data.mTopK == NumTopExperts, "Routing kernel expects %d topK experts (for now)", NumTopExperts);
    TLLM_CHECK_WITH_INFO(data.mNumExperts <= MaxNumExperts,
        "Routing kernel expects #experts %d to be at most max #experts %d", data.mNumExperts, MaxNumExperts);
    static_assert(MaxNumExperts <= NumThreads, "#experts must be bounded by #threads");
    static_assert(MaxNumExperts <= NumThreadsHist, "#experts must be bounded by #threads");
    TLLM_CHECK_WITH_INFO(
        data.mNumExperts % 4 == 0, "Routing kernel expects #experts %d to be a multiple of 4.", data.mNumExperts);
    TLLM_CHECK_WITH_INFO(data.mPaddingLog2 < 8, "Routing kernel expects padding log2 < 8, got %d", data.mPaddingLog2);

    bool const useSingleCluster
        = data.mNumTokens <= (data.mPtrScores != nullptr ? MaxNumTokensSingleClusterScores : MaxNumTokensSingleCluster);
    if (!useSingleCluster)
    {
        TLLM_CHECK_WITH_INFO(
            data.mPtrExpertIdx != nullptr, "When #tokens is large, `mPtrExpertIdx` is a required input.");
        TLLM_CHECK_WITH_INFO(
            data.mPtrExpertCounts != nullptr, "When #tokens is large, `mPtrExpertCounts` is a required input.");
    }

    if (useSingleCluster)
    {
        LAUNCH_EXPW_QWEN3(data, false, routingIndicesClusterKernel, NumBlocksPerCluster, NumThreads,
            /*smemSize=*/0, // No dynamic smem
            stream);
    }
    else
    {
        uint32_t const expandedIdxSize = data.mNumTokens * NumTopExperts;

        uint32_t const histogramEltsPerBlock = 8 * NumThreadsHist;
        uint32_t const offsetEltsPerBlock = NumEltsPerOffsetTilePerThread * NumThreadsHist;

        // Limit grid size (all kernels use a grid-stride loop).
        uint32_t const maxNumBlocks = 1024;

        int const numBlocksHistogram
            = std::min((expandedIdxSize + histogramEltsPerBlock - 1) / histogramEltsPerBlock, maxNumBlocks);
        int const numBlocksOffsets
            = std::min((expandedIdxSize + offsetEltsPerBlock - 1) / offsetEltsPerBlock, maxNumBlocks);

        if (data.mPtrScores != nullptr)
        {
            LAUNCH_EXPW_QWEN3(data, false, routingIndicesHistogramScoresKernel, maxNumBlocks, NumThreadsHist,
                /*smemSize=*/0, // No dynamic smem
                stream);
        }
        LAUNCH_EXPW_ONLY_QWEN3(data, false, routingIndicesHistogramKernel, numBlocksHistogram, NumThreadsHist,
            /*smemSize=*/0, // No dynamic smem
            stream);
        LAUNCH_EXPW_ONLY_QWEN3(data, false, routingIndicesOffsetsKernel, numBlocksOffsets, NumThreadsHist,
            /*smemSize=*/0, // No dynamic smem
            stream);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace routingQwen3

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace moe::dev
