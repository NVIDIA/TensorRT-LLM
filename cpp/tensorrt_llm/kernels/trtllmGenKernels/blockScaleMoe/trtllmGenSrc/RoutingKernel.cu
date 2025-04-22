/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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
//// FIX
#include "macros.h" // #include <utils/macros.h>

#include "Utils.h"  // #include <trtllm/dev/Utils.h>

// #include "trtllmGenSrc/gen/GenCtx.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/cub.cuh>

#include <cute/arch/cluster_sm90.hpp>

#include <type_traits>

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace moe::dev
{

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace routing
{

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace tg = trtllm::gen;
namespace cg = cooperative_groups;

////////////////////////////////////////////////////////////////////////////////////////////////////

static constexpr int NumThreads = 256;
static constexpr int NumBlocksPerCluster = 8;
static constexpr int NumThreadsGemm = 128;
static constexpr int WarpSize = 32;
static constexpr int NumWarps = NumThreads / WarpSize;
static constexpr int NumTopGroups = 4;
static constexpr int NumTopGroupScores = 2;
static constexpr int NumTopExperts = 8;

// Performance tuning knob.
static constexpr int NumEltsPerOffsetTilePerThread = 8;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000 && defined(__CUDA_ARCH_FEAT_SM100_ALL))
#define TLLM_GEN_ENABLE_FAST_REDUX
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

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

static __device__ inline float tanh_fast(float x)
{
    float res;
    asm volatile("{ tanh.approx.f32 %0, %1; }\n" : "=f"(res) : "f"(x));
    return res;
}

static __device__ inline float sigmoid_fast(float x)
{
    return 0.5f * tanh_fast(0.5f * x) + 0.5f;
}

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

template <typename KernelParams>
__global__ void routingKernelGemm(KernelParams params)
{
    // naive Gemm, to be replaced by performant kernel
    using Type = typename KernelParams::Type;
    using TypeExpW = typename KernelParams::TypeExpW;
    // each thread has space for the dot product of each expert here
    extern __shared__ char __attribute((aligned(128))) smemBase[];
    auto* smemDotPartial = reinterpret_cast<float*>(smemBase);
    static constexpr int SmemStride = NumThreadsGemm + 1;

    auto tokenOff = int64_t{blockIdx.x} * int64_t{params.mHiddenDim};

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    // immediately trigger the secondary kernel when using PDL
    if constexpr (KernelParams::UsePdl)
    {
        cudaTriggerProgrammaticLaunchCompletion();
    }
#endif

    // dot product for all experts
    // entire block must go into this loop
    for (int32_t dd = threadIdx.x; dd < params.mHiddenDim; dd += NumThreadsGemm)
    {
        Type act = params.mPtrIn[tokenOff + dd];

        for (int32_t expertIdx = 0; expertIdx < params.mNumExperts; ++expertIdx)
        {
            auto weightOff = int64_t{expertIdx} * int64_t{params.mHiddenDim};
            TypeExpW weight = params.mPtrRoutingWeights[weightOff + dd];
            auto val = float{act} * float{weight};
            if (dd == threadIdx.x)
            {
                smemDotPartial[expertIdx * SmemStride + threadIdx.x] = val;
            }
            else
            {
                smemDotPartial[expertIdx * SmemStride + threadIdx.x] += val;
            }
        }
    }
    // make all partial dot products available to all threads
    __syncthreads();

    // finalize dot product and write to output
    for (int32_t expertIdx = threadIdx.x; expertIdx < params.mNumExperts; expertIdx += NumThreadsGemm)
    {
        float dot = 0.F;
        for (int32_t ii = 0; ii < NumThreadsGemm; ++ii)
        {
            dot += smemDotPartial[expertIdx * SmemStride + ii];
        }
        params.mPtrScores[int64_t{blockIdx.x} * int64_t{params.mNumExperts} + expertIdx] = dot;
    }
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
    if (warpIdx >= params.mNumExpertGroups)
    {
        return;
    }

    // note that for invalid scores, we simply use a negative value:
    // they work well even with the compacted format used in topK, and
    // sigmoid / bias activated scores cannot be negative
    static constexpr float invalidScoreFloat = -1.F;
    const TypeExpW invalidScore = TypeExpW{invalidScoreFloat};

    // load bias already; each warp represents one expert group
    auto threadExpert = warpIdx * params.mNumExpertsPerGroup + laneIdx;
    auto expertSelected = laneIdx < params.mNumExpertsPerGroup;
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
    // TODO: verify bf16 scoreBias accuracy before changing it back to bf16
    // auto scoreBias = TypeExpW{scoreSigmoid + float{biasVal}}; // TypeExpW is bf16
    auto scoreBias = float{scoreSigmoid + float{biasVal}};
    if (expertSelected)
    {
        smemScoreBias[threadExpert] = scoreBias;
    }

    // registers for top group score reduction
    float topExpGroupScores[NumTopGroupScores];
    [[maybe_unused]] int32_t topExpGroupIdx[NumTopGroupScores];
    reduceTopK(warp, topExpGroupScores, topExpGroupIdx, scoreBias, threadExpert,
        /* minValue */ invalidScoreFloat);

    // get the final group score and write it to shared
    if (cute::elect_one_sync())
    {
        auto groupScore = topExpGroupScores[0] + topExpGroupScores[1];
        smemGroupScores[warpIdx] = groupScore;
    }

    // make group scores available to all warps
    __syncthreads();

    float topGroups[NumTopGroups]; // params.mNumLimitedGroups
    int32_t topGroupIdx[NumTopGroups];
    float expertScoreGroup[NumTopGroups];
    int32_t expertIdxGroup[NumTopGroups];
    float topScores[NumTopExperts]; // params.mTopK
    int32_t topExperts[NumTopExperts];
    auto localExpertExtent = params.mNumLocalExperts << params.mLocalExpertsStrideLog2;
    if (warpIdx == 0)
    {
        // a single warp performs the selection of top groups, and goes on to select the final experts
        float groupScore = laneIdx < params.mNumExpertGroups ? smemGroupScores[laneIdx] : float{};

        reduceTopK(warp, topGroups, topGroupIdx, groupScore, laneIdx,
            /* minValue */ invalidScoreFloat);

        // final expert selection: get relevant indexes and scores from shared

#pragma unroll
        for (int ii = 0; ii < NumTopGroups; ++ii)
        { // params.mNumLimitedGroups
            auto groupIdx = topGroupIdx[ii];
            expertIdxGroup[ii] = groupIdx * params.mNumExpertsPerGroup + laneIdx;
            expertScoreGroup[ii] = expertSelected ? smemScoreBias[expertIdxGroup[ii]] : invalidScoreFloat;
        }

        reduceTopK(warp, topScores, topExperts, expertScoreGroup, expertIdxGroup,
            /* minValue */ invalidScoreFloat);

        // determine our lane's expert index and write to output
        int32_t expertIdx = 0;
#pragma unroll
        for (int ii = 0; ii < NumTopExperts; ++ii)
        { // params.mTopK
            expertIdx = laneIdx == ii ? topExperts[ii] : expertIdx;
        }
        // determine whether our expert is local to this GPU
        auto localExpertIdx = expertIdx - params.mLocalExpertsStartIdx;
        auto isLocalExpert = localExpertIdx >= 0 && localExpertIdx < localExpertExtent
            && (localExpertIdx & params.mLocalExpertsStrideLog2) == 0;

        // write expert idx out already
        auto idxTopK = blockIdx.x * NumTopExperts + laneIdx; // params.mTopK
        if (laneIdx < NumTopExperts && params.mPtrExpertIdx != nullptr)
        {                                                    // params.mTopK
            params.mPtrExpertIdx[idxTopK] = expertIdx;
        }
        float scoreNorm = laneIdx < NumTopExperts ? smemScoreSigmoid[expertIdx] : 0.F;
        auto redNorm = cg::reduce(warp, scoreNorm, cg::plus<float>{});
        auto finalScore = TypeExpW{scoreNorm * params.mRouteScale / redNorm};
        if (laneIdx < NumTopExperts && params.mPtrExpertWeights != nullptr)
        { // params.mTopK
            params.mPtrExpertWeights[idxTopK] = finalScore;
        }
        if (laneIdx < NumTopExperts && params.mPtrExpertWeightsFull != nullptr && isLocalExpert)
        { // params.mTopK
            auto idxWeightsFull = localExpertIdx * gridDim.x + blockIdx.x;
            params.mPtrExpertWeightsFull[idxWeightsFull] = finalScore;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

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
        = (16384 * NumTopExperts + NumThreadsPerCluster - 1) / NumThreadsPerCluster;

    // Initialize cluster.
    uint32_t const clusterBlockRank = blockIdx.x;
    uint32_t const clusterThreadIdx = NumThreads * clusterBlockRank + threadIdx.x;

    int32_t const warpIdx = __shfl_sync(0xffffffff, threadIdx.x / WarpSize, 0);

    auto expandedIdxSize = params.mNumTokens * NumTopExperts;

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
    // Note (lsugy): I observed a perf benefit to doing this before the final loop so the compiler can
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

template <typename KernelParams>
__global__ void __launch_bounds__(NumThreads) routingIndicesCoopKernel(KernelParams params)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
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
    uint32_t const gridBlockIdx = blockIdx.x;
    uint32_t const gridThreadIdx = NumThreads * gridBlockIdx + threadIdx.x;
    uint32_t const numBlocks = gridDim.x;
    uint32_t const numThreadsPerGrid = numBlocks * NumThreads;

    int32_t const warpIdx = __shfl_sync(0xffffffff, threadIdx.x / WarpSize, 0);

    auto expandedIdxSize = params.mNumTokens * NumTopExperts;

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
#else
    assert(false && "routingIndicesCoopKernel is only supported on SM90+ architectures");
#endif
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
__global__ void __launch_bounds__(NumThreads) routingIndicesHistogramKernel(KernelParams params)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    // number of experts is bounded by number of threads
    __shared__ int32_t __attribute((aligned(128))) smemExpertCount[NumThreads];

    // For unrolling.
    uint32_t constexpr NumEltsPerThread = 8;

    // Pre-fill the counts with 0
    smemExpertCount[threadIdx.x] = 0;
    __syncthreads();

    // Wait on primary grid and trigger secondary kernel.
    if constexpr (KernelParams::UsePdl)
    {
        cudaGridDependencySynchronize();
        cudaTriggerProgrammaticLaunchCompletion();
    }

    uint32_t const expandedIdxSize = params.mNumTokens * NumTopExperts;
    uint32_t const localExpertExtent = params.mNumLocalExperts << params.mLocalExpertsStrideLog2;

    uint32_t const gridBlockOffset = blockIdx.x * NumThreads;
    uint32_t const gridStride = gridDim.x * NumThreads;

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
    for (uint32_t expandedIdx0 = gridBlockOffset * NumEltsPerThread; expandedIdx0 < expandedIdxSize;
         expandedIdx0 += gridStride * NumEltsPerThread)
    {
        // Fast path if bound checks aren't necessary
        if (expandedIdx0 + NumEltsPerThread * NumThreads <= expandedIdxSize)
        {
#pragma unroll
            for (uint32_t ii = 0; ii < NumEltsPerThread; ii++)
            {
                uint32_t expandedIdx = expandedIdx0 + ii * NumThreads + threadIdx.x;
                loopBody(expandedIdx);
            }
        }
        else
        {
            for (uint32_t expandedIdx = expandedIdx0 + threadIdx.x; expandedIdx < expandedIdxSize;
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
#else
    assert(false && "routingIndicesHistogramKernel is only supported on SM90+ architectures");
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KernelParams>
__global__ void __launch_bounds__(NumThreads) routingIndicesOffsetsKernel(KernelParams params)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
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
    for (uint32_t tileIdx = blockIdx.x; tileIdx < numTiles; tileIdx += gridDim.x)
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
#else
    assert(false && "routingIndicesOffsetsKernel is only supported on SM90+ architectures");
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void run(Data const& data, void* stream)
{
    TLLM_CHECK_ERROR(data.mPtrExpertIdx != nullptr || data.mPtrPermutedIdxSize != nullptr
            || data.mPtrExpertWeightsFull != nullptr || data.mPtrExpertWeights != nullptr,
        "Routing kernel requires at least one output parameter");
    if (data.mPtrExpandedIdxToPermutedIdx != nullptr || data.mPtrPermutedIdxToTokenIdx != nullptr)
        TLLM_CHECK_ERROR(data.mPtrExpertIdx != nullptr && data.mPtrPermutedIdxSize,
            "If permuted index is required, `mPtrExpertIdx` is also required");
    TLLM_CHECK_ERROR(!data.mUseRoutingSoftmax, "Routing with softmax not implemented yet");
    TLLM_CHECK_ERROR(
        data.mNumLimitedGroups == NumTopGroups, "Routing kernel expects ", NumTopGroups, " groups (for now)");
    TLLM_CHECK_ERROR(data.mTopK == NumTopExperts, "Routing kernel expects ", NumTopExperts, " topK experts (for now)");
    TLLM_CHECK_ERROR(data.mTopK <= WarpSize, "Routing kernel expects top K <= warp size, got ", data.mTopK);
    TLLM_CHECK_ERROR(data.mTopK * data.mNumLimitedGroups <= WarpSize,
        "Routing kernel expects top K * top groups <= warp size (for now), got ", data.mTopK, " * ",
        data.mNumLimitedGroups);
    TLLM_CHECK_ERROR(data.mNumExperts >= NumTopExperts, "Routing kernel expects ", NumTopExperts,
        " to be at most #experts ", data.mNumExperts);
    TLLM_CHECK_ERROR(data.mNumExperts <= NumThreads, "Routing kernel expects #experts ", data.mNumExperts,
        " <= #threads ", NumThreads);
    TLLM_CHECK_ERROR(data.mNumExpertGroups <= NumWarps, "Routing kernel expects #experts groups ",
        data.mNumExpertGroups, " to be <= #warps", NumWarps);
    TLLM_CHECK_ERROR(data.mNumExperts % data.mNumExpertGroups == 0, "Routing kernel expects #experts ",
        data.mNumExperts, " to be a multiple of #expert groups ", data.mNumExpertGroups);
    TLLM_CHECK_ERROR(data.mNumExperts / data.mNumExpertGroups <= WarpSize,
        "Routing kernel expects #experts per group <= warp size, got ", data.mNumExperts / data.mNumExpertGroups);
    TLLM_CHECK_ERROR(
        data.mNumExperts % 4 == 0, "Routing kernel expects #experts ", data.mNumExperts, " to be a multiple of 4.");
    TLLM_CHECK_ERROR(data.mPaddingLog2 < 8, "Routing kernel expects padding log2 < 8, got ", data.mPaddingLog2);
    int const numBlocks = data.mNumTokens;

    if (data.mPtrExpertWeightsFull != nullptr)
    {
        auto localExpertExtent = data.mNumLocalExperts << data.mLocalExpertsStrideLog2;
        // note: we set a value of 0 here, s.t. even if the routing happens,
        // it will be ignored / not given any weight
        TLLM_CHECK_CUDA(cudaMemsetAsync(
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
            TLLM_CHECK_CUDA(cudaMemsetAsync(data.mPtrPermutedIdxToTokenIdx, -1,
                static_cast<size_t>(maxPermutedSize) * sizeof(int32_t), (cudaStream_t) stream));
        }
    */

    bool const useSingleCluster = data.mNumTokens <= 1024;
    if (!useSingleCluster)
    {
        // Reset the global histograms (not used in single-cluster code path).
        // Cover both for the cooperative and two-kernel code paths.
        TLLM_CHECK_CUDA(cudaMemsetAsync(
            data.mPtrExpertCounts, 0, static_cast<size_t>(2 * NumThreads) * sizeof(int32_t), (cudaStream_t) stream));
    }

    // Number of blocks we can use in the cooperative kernel
    // The number of blocks must be:
    //   >= ⌈(numTokens * NumTopExperts) / (MaxExpandedIdxPerThread * NumThreads)⌉
    //   <= numSms, assuming an occupancy of 1 block/SM
    //
    // If too small for the given numTokens, fall back to the less performant two-step method.
    //
    // The upper bound is a strict requirement. The number of blocks should be determined by querying
    // the device properties, or conservatively low.
    // /!\ The following number is not portable!! (but works on H100 and B200)
    int const numBlocksCoop = 128;

    // Maximum number of tokens supported by the kernel using a cooperative launch.
    int const maxTokensCoop = (numBlocksCoop * NumThreads * 64) / NumTopExperts;
    LAUNCH_EXPW_ONLY(data,
        /*coopLaunch=*/false, routingMainKernel, numBlocks, NumThreads,
        /*smemSize=*/0, // No dynamic smem
        stream);

    if (data.mPtrPermutedIdxSize != nullptr)
    {
        if (useSingleCluster)
        {
            LAUNCH_EXPW_ONLY(data,
                /*coopLaunch=*/false, routingIndicesClusterKernel, NumBlocksPerCluster, NumThreads,
                /*smemSize=*/0, // No dynamic smem
                stream);
        }
        else if (data.mNumTokens <= maxTokensCoop)
        {
            LAUNCH_EXPW_ONLY(data,
                /*coopLaunch=*/true, routingIndicesCoopKernel, numBlocksCoop, NumThreads,
                /*smemSize=*/0, // No dynamic smem
                stream);
        }
        else
        {
            const uint32_t expandedIdxSize = data.mNumTokens * NumTopExperts;

            const uint32_t histogramEltsPerBlock = 8 * NumThreads;
            const uint32_t offsetEltsPerBlock = NumEltsPerOffsetTilePerThread * NumThreads;

            // Limit grid size (both kernels use a grid-stride loop).
            const uint32_t maxNumBlocks = 1024;

            int const numBlocksHistogram
                = std::min((expandedIdxSize + histogramEltsPerBlock - 1) / histogramEltsPerBlock, maxNumBlocks);
            int const numBlocksOffsets
                = std::min((expandedIdxSize + offsetEltsPerBlock - 1) / offsetEltsPerBlock, maxNumBlocks);

            LAUNCH_EXPW_ONLY(data,
                /*coopLaunch=*/false, routingIndicesHistogramKernel, numBlocksHistogram, NumThreads,
                /*smemSize=*/0, // No dynamic smem
                stream);
            LAUNCH_EXPW_ONLY(data,
                /*coopLaunch=*/false, routingIndicesOffsetsKernel, numBlocksOffsets, NumThreads,
                /*smemSize=*/0, // No dynamic smem
                stream);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace routing

} // namespace moe::dev
