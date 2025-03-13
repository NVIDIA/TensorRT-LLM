/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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
static constexpr int NumThreadsGemm = 128;
static constexpr int WarpSize = 32;
static constexpr int NumWarps = NumThreads / WarpSize;
static constexpr int NumTopGroups = 4;
static constexpr int NumTopGroupScores = 2;
static constexpr int NumTopExperts = 8;

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

size_t getSmemSize(Data const& data)
{
    auto bytesPerExpW = tg::dtypeGetNumBits(data.mDtypeExpW) / 8 /* bits */;
    auto sizeScoreSigmoid = data.mNumExperts * sizeof(float) + 127;
    auto sizeGroupScores = NumWarps * bytesPerExpW + 127;
    auto sizeTopGroupIdx = data.mNumLimitedGroups * sizeof(int32_t) + 127;
    auto sizeTopExpertsIdx = data.mNumLimitedGroups * data.mTopK * sizeof(int32_t) + 127;
    auto sizeTopExpertsScores = data.mNumLimitedGroups * data.mTopK * bytesPerExpW;
    return sizeScoreSigmoid + sizeGroupScores + sizeTopGroupIdx + sizeTopExpertsIdx + sizeTopExpertsScores;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ inline uintptr_t smemRoundUp(uintptr_t addr)
{
    return 128 * ((addr + 127) / 128);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__host__ __device__ constexpr uint8_t divUpMulLog2(uint8_t a, uint8_t bLog2)
{
    return ((a + (1 << bLog2) - 1) >> bLog2) << bLog2;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KernelParams>
__global__ void routingKernel(KernelParams params)
{
    // declare types required for reductions
    using TypeExpW = typename KernelParams::TypeExpW;

    // declare shared memory structure
    // number of experts is bounded by number of threads
    __shared__ float __attribute((aligned(128))) smemScoreSigmoid[NumThreads];
    __shared__ TypeExpW __attribute((aligned(128))) smemScoreBias[NumThreads];
    // number of expert groups is bounded by number of warps
    __shared__ TypeExpW __attribute((aligned(128))) smemGroupScores[NumWarps];
    __shared__ TypeExpW __attribute((aligned(128)))
    smemTopExpertsScores[NumTopGroups * NumTopExperts]; // params.mNumLimitedGroups * params.mTopK
    // number of experts is bounded by number of threads
    // note: we assume number of tokens bounded by 256, so the 4 counts fit in uint32_t
    __shared__ uint32_t __attribute((aligned(128))) smemExpertCount[NumThreads / 4];
    __shared__ int16_t __attribute((aligned(128))) smemExpertOffset[NumThreads];
    // needed for the exclusive sum of token offsets
    using Scan = cub::BlockScan<int16_t, NumThreads, cub::BLOCK_SCAN_WARP_SCANS>;
    __shared__ typename Scan::TempStorage tempStorage;
    // if the number of tokens is bounded by 256, then the total number of indexes
    // is bounded by 256 * TopK
    static constexpr int MaxExpandedIdxPerThread = (256 * NumTopExperts) / NumThreads; // params.mTopK

    // needed for warp reduce
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<WarpSize>(block);
    // for the final reduction of weight norm, only some lanes need to participate
    int32_t laneIdx = threadIdx.x % WarpSize;
    int32_t warpIdx = __shfl_sync(0xffffffff, threadIdx.x / WarpSize, 0);
    // warps outside the range of expert groups do not participate
    if (warpIdx >= params.mNumExpertGroups)
        return;

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
    auto scoreSigmoid = sigmoid_fast(score);
    // write the sigmoid score to shared for later use
    if (expertSelected)
        smemScoreSigmoid[threadExpert] = scoreSigmoid;
    // get the score with bias
    // note that with invalid values, because sigmoid is < 1 and bias is -1,
    // we must get a negative value, which is smaller than any valid value
    auto scoreBias = TypeExpW{scoreSigmoid + float{biasVal}};
    if (expertSelected)
        smemScoreBias[threadExpert] = scoreBias;

    // registers for top group score reduction
    TypeExpW topExpGroupScores[NumTopGroupScores];
    [[maybe_unused]] int32_t topExpGroupIdx[NumTopGroupScores];
    reduceTopK(warp, topExpGroupScores, topExpGroupIdx, scoreBias, threadExpert,
        /* minValue */ invalidScore);

    // get the final group score and write it to shared
    if (cute::elect_one_sync())
    {
        auto groupScore = topExpGroupScores[0] + topExpGroupScores[1];
        smemGroupScores[warpIdx] = groupScore;
    }

    // make group scores available to all warps
    __syncthreads();

    TypeExpW topGroups[NumTopGroups]; // params.mNumLimitedGroups
    int32_t topGroupIdx[NumTopGroups];
    TypeExpW expertScoreGroup[NumTopGroups];
    int32_t expertIdxGroup[NumTopGroups];
    TypeExpW topScores[NumTopExperts]; // params.mTopK
    int32_t topExperts[NumTopExperts];
    auto localExpertExtent = params.mNumLocalExperts << params.mLocalExpertsSrideLog2;
    if (warpIdx == 0)
    {
        // a single warp performs the selection of top groups, and goes on to select the final experts
        TypeExpW groupScore = laneIdx < params.mNumExpertGroups ? smemGroupScores[laneIdx] : TypeExpW{};

        reduceTopK(warp, topGroups, topGroupIdx, groupScore, laneIdx,
            /* minValue */ invalidScore);

        // final expert selection: get relevant indexes and scores from shared

#pragma unroll
        for (int ii = 0; ii < NumTopGroups; ++ii)
        { // params.mNumLimitedGroups
            auto groupIdx = topGroupIdx[ii];
            expertIdxGroup[ii] = groupIdx * params.mNumExpertsPerGroup + laneIdx;
            expertScoreGroup[ii] = expertSelected ? smemScoreBias[expertIdxGroup[ii]] : invalidScore;
        }

        reduceTopK(warp, topScores, topExperts, expertScoreGroup, expertIdxGroup,
            /* minValue */ invalidScore);

        // determine our lane's expert index and write to output
        // TODO(mjoux) this can be slightly optimized using PRMT
        int32_t expertIdx = 0;
#pragma unroll
        for (int ii = 0; ii < NumTopExperts; ++ii)
        { // params.mTopK
            expertIdx = laneIdx == ii ? topExperts[ii] : expertIdx;
        }
        // determine whether our expert is local to this GPU
        auto localExpertIdx = expertIdx - params.mLocalExpertsStartIdx;
        auto isLocalExpert = localExpertIdx >= 0 && localExpertIdx < localExpertExtent
            && (localExpertIdx & params.mLocalExpertsSrideLog2) == 0;

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

    // if no other index required, we can stop here
    if (params.mPtrPermutedIdxSize == nullptr)
        return;

    // note : the grid dimension is the number of tokens
    auto expandedIdxSize = gridDim.x * NumTopExperts; // params.mTopK

    // pre-fill the counts with 0
    // note that we assume that at least NumThreads / 4 threads are running here,
    // thus the number of groups must be >= NumWarps / 4
    if (threadIdx.x < NumThreads / 4)
        smemExpertCount[threadIdx.x] = 0;

    // sync the grid to be able to read the expert index from all other blocks
    // note: this also syncs the block, so shared memory writes above are available
    // to the whole block
    auto grid = cg::this_grid();
    grid.sync();

    // only a single block continues computation of the indexes
    if (blockIdx.x > 0)
        return;

    // each thread keeps has some number of "expanded indexes" assigned to it
    // for each of these, we keep the associated expert and offset within expert in registers
    uint8_t expertIndexes[MaxExpandedIdxPerThread];
    uint8_t expertOffsets[MaxExpandedIdxPerThread];
#pragma unroll
    for (int32_t ii = 0; ii < MaxExpandedIdxPerThread; ++ii)
    {
        auto expandedIdx = static_cast<int32_t>(threadIdx.x) + ii * NumThreads;
        if (expandedIdx >= expandedIdxSize)
            break;
        int32_t expertIdx = params.mPtrExpertIdx[expandedIdx];
        expertIndexes[ii] = static_cast<uint8_t>(expertIdx);
        // check whether this expert is local to our GPU at all and ignore if not
        auto localExpertIdx = expertIdx - params.mLocalExpertsStartIdx;
        auto isLocalExpert = localExpertIdx >= 0 && localExpertIdx < localExpertExtent
            && (localExpertIdx & params.mLocalExpertsSrideLog2) == 0;
        if (!isLocalExpert)
            continue;

        // encode the value 1 at the right position in uint32_t value
        auto byteIdx = expertIdx % 4;
        auto byteShift = byteIdx * 8;
        auto val = uint32_t{1} << byteShift;
        auto prevVal = atomicAdd(smemExpertCount + expertIdx / 4, val);
        // get the previous count at the right byte position
        prevVal >>= byteShift;
        expertOffsets[ii] = static_cast<uint8_t>(prevVal);
    }

    // make histogram (tokens counts per expert) available to all threads
    __syncthreads();

    // TODO(mjoux) potentially better to perform this scan with a single warp
    // rather than the whole block (above and below parts should still be handled
    // with full block, or we wouldn't be able to handle many tokens at all)

    // each thread now represents one expert
    // get count associated with this expert
    auto count4 = smemExpertCount[threadIdx.x / 4];
    auto byteIdx = threadIdx.x % 4;
    count4 >>= (byteIdx * 8);
    auto count = static_cast<uint8_t>(count4);

    // Each 4th thread writes four 8-bit counts:
    if (params.mPtrNumTokensPerExpert != nullptr)
    {
        if ((threadIdx.x % 4) == 0 && threadIdx.x / 4 < (params.mNumExperts + 3) / 4)
        {
            auto count4 = smemExpertCount[threadIdx.x / 4];
            // Write all 4 counts at once
            reinterpret_cast<uint32_t*>(params.mPtrNumTokensPerExpert)[threadIdx.x / 4] = count4;
        }
    }

    // get the padded count associated with this expert
    auto numTokensPerExpertPadded = static_cast<int16_t>(divUpMulLog2(count, params.mPaddingLog2));
    int16_t blockAggregate;
    int16_t offset;
    // exclusive sum across all threads to get the final expert offsets, and full padded count
    Scan(tempStorage).ExclusiveSum(numTokensPerExpertPadded, offset, blockAggregate);

    // write out padded count
    if (/* blockIdx.x == 0 && */ warpIdx == NumWarps - 1 && cute::elect_one_sync())
        params.mPtrPermutedIdxSize[0] = int32_t{blockAggregate};

    // write expert offsets to shared
    smemExpertOffset[threadIdx.x] = offset;

    // make expert offsets available to all threads
    __syncthreads();

// each thread has the same "expanded indexes" assigned to it as above
// at this point, we know the final offsets of experts and the offsets within
// experts, which allows writing the final index values
#pragma unroll
    for (int32_t ii = 0; ii < MaxExpandedIdxPerThread; ++ii)
    {
        auto expandedIdx = static_cast<int32_t>(threadIdx.x) + ii * NumThreads;
        if (expandedIdx >= expandedIdxSize)
            break;
        auto expertIdx = expertIndexes[ii];
        // check whether this expert is local to our GPU at all
        auto localExpertIdx = static_cast<int32_t>(expertIdx) - params.mLocalExpertsStartIdx;
        auto isLocalExpert = localExpertIdx >= 0 && localExpertIdx < localExpertExtent
            && (localExpertIdx & params.mLocalExpertsSrideLog2) == 0;
        auto tokenIdx = expandedIdx / NumTopExperts; // params.mTopK
        auto permutedIdx = isLocalExpert
            ? int32_t{smemExpertOffset[expertIdx]} + static_cast<int32_t>(expertOffsets[ii])
            : int32_t{-1};
        if (params.mPtrExpandedIdxToPermutedIdx != nullptr)
            params.mPtrExpandedIdxToPermutedIdx[expandedIdx] = permutedIdx;
        if (params.mPtrPermutedIdxToExpandedIdx != nullptr)
            params.mPtrPermutedIdxToExpandedIdx[permutedIdx] = expandedIdx;
        if (params.mPtrPermutedIdxToTokenIdx != nullptr && isLocalExpert)
            params.mPtrPermutedIdxToTokenIdx[permutedIdx] = tokenIdx;
    }
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
    TLLM_CHECK_ERROR(data.mNumExpertGroups <= NumWarps && data.mNumExpertGroups >= NumWarps / 4,
        "Routing kernel expects #experts groups ", data.mNumExpertGroups, " to be <= #warps and >= #warps / 4",
        NumWarps);
    TLLM_CHECK_ERROR(data.mNumExperts % data.mNumExpertGroups == 0, "Routing kernel expects #experts ",
        data.mNumExperts, " to be a multiple of #expert groups ", data.mNumExpertGroups);
    TLLM_CHECK_ERROR(data.mNumExperts / data.mNumExpertGroups <= WarpSize,
        "Routing kernel expects #experts per group <= warp size, got ", data.mNumExperts / data.mNumExpertGroups);
    TLLM_CHECK_ERROR(
        data.mNumExperts % 4 == 0, "Routing kernel expects #experts ", data.mNumExperts, " to be a multiple of 4.");
    TLLM_CHECK_ERROR(data.mPaddingLog2 < 8, "Routing kernel expects padding log2 < 8, got ", data.mPaddingLog2);

    if (data.mPtrPermutedIdxSize != nullptr)
    {
        TLLM_CHECK_ERROR(data.mNumTokens < 256, "Routing kernel expects #tokens < 256, got ", data.mNumTokens);
    }

    int64_t const numEltBytes = tg::dtypeGetNumBits(data.mDtypeElt) / 8 /* bits */;
    int64_t const numExpWBytes = tg::dtypeGetNumBits(data.mDtypeExpW) / 8 /* bits */;
    TLLM_CHECK_ERROR(numExpWBytes % numEltBytes == 0,
        "Routing kernel expects weights type size to be a multiple of input type size");
    // TLLM_CHECK_ERROR((data.mHiddenDim * numEltBytes) % 16 == 0,
    //                 "Routing kernel expects hiddenDim to be aligned to 16 bytes");
    int const numBlocks = data.mNumTokens;

    if (data.mPtrExpertWeightsFull != nullptr)
    {
        auto localExpertExtent = data.mNumLocalExperts << data.mLocalExpertsSrideLog2;
        // note: we set a value of 0 here, s.t. even if the routing happens,
        // it will be ignored / not given any weight
        TLLM_CHECK_CUDA(cudaMemsetAsync(
            data.mPtrExpertWeightsFull, 0, localExpertExtent * data.mNumTokens * sizeof(float), (cudaStream_t) stream));
    }

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

    if (data.mPtrPermutedIdxToExpandedIdx != nullptr)
    {
        // need to set all values to -1 before running the kernel
        auto maxPermutedSize
            = data.mNumTokens * data.mTopK + (data.mNumExperts << data.mPaddingLog2) - data.mNumExperts;
        // note that a value of -1 per byte works for any size of signed integer
        // to set each full value to the logical value -1
        TLLM_CHECK_CUDA(cudaMemsetAsync(data.mPtrPermutedIdxToExpandedIdx, -1,
            static_cast<size_t>(maxPermutedSize) * sizeof(int32_t), (cudaStream_t) stream));
    }

    // TODO: enable GEMM kernel for pre-routing
    // auto smemGemm = data.mNumExperts * (NumThreadsGemm + 1) * sizeof(float);
    // LAUNCH_EXPW(data, routingKernelGemm, numBlocks, NumThreadsGemm, smemGemm, stream);

    auto isCooperative = data.mPtrPermutedIdxSize != nullptr;
    LAUNCH_EXPW_ONLY(data, isCooperative, routingKernel, numBlocks, NumThreads, getSmemSize(data), stream);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace routing

} // namespace moe::dev
