/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "cutlass/numeric_types.h"
#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/kernels/renormMoeRoutingKernels.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <math.h>

namespace cg = cooperative_groups;
using namespace tensorrt_llm::common;

namespace tensorrt_llm::kernels
{

static constexpr int BLOCK_SIZE = 1024;
static constexpr int WARP_SIZE = 32;
static constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;

namespace reduce_topk
{
template <typename T_>
struct TopKRedType
{
    using T = T_;
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, cutlass::bfloat16_t>,
        "Top K reduction only implemented for float and Bf16");
    using TypeCmp = std::conditional_t<sizeof(T) >= 4, double, float>;
    static constexpr int64_t Mask64 = 0x000000000000FFFF;
    static constexpr int32_t Mask32 = 0x0000FFFF;

    TypeCmp compVal;

    static __host__ __device__ inline TypeCmp makeCmpVal(T val, int32_t idx = 0)
    {
        auto cmpVal = TypeCmp{val};
        TypeCmp cmpValWithIdx;
        if constexpr (sizeof(T) >= 4)
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

    static __host__ __device__ inline void unpack(T& val, int32_t& idx, TypeCmp cmp)
    {
        if constexpr (sizeof(T) >= 4)
        {
            idx = static_cast<int32_t>(reinterpret_cast<int64_t&>(cmp) & Mask64);
            auto val64 = reinterpret_cast<int64_t&>(cmp) & ~Mask64;
            val = static_cast<float>(reinterpret_cast<double&>(val64));
        }
        else
        {
            idx = reinterpret_cast<int32_t&>(cmp) & Mask32;
            auto val32 = reinterpret_cast<int32_t&>(cmp) >> 16;
            val = T::bitcast(reinterpret_cast<uint16_t&>(val32));
        }
    }

    __host__ __device__ TopKRedType() = default;

    __host__ __device__ TopKRedType(T val, int32_t idx)
        : compVal(makeCmpVal(val, idx))
    {
    }

    __host__ __device__ operator TypeCmp() const noexcept
    {
        return compVal;
    }

    __device__ inline TypeCmp reduce(cg::thread_block_tile<WARP_SIZE> const& warp)
    {
#if defined(TLLM_GEN_ENABLE_FAST_REDUX)
        static constexpr bool UseCg = false;
#else
        static constexpr bool UseCg = true;
#endif
        if constexpr (UseCg || sizeof(T) >= 4)
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
__device__ void reduceTopK(cg::thread_block_tile<WARP_SIZE> const& warp, Type (&out)[K], int32_t (&outIdx)[K],
    Type value, int32_t idx, Type minValue)
{
    static_assert(K > 0, "Top K must have K > 0");
    static_assert(K < WARP_SIZE, "Top K must have K < WARP_SIZE");
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
__device__ void reduceTopK(cg::thread_block_tile<WARP_SIZE> const& warp, Type (&out)[K], int32_t (&outIdx)[K],
    Type (&value)[N], int32_t (&idx)[N], Type minValue)
{
    static_assert(K > 0, "Top K must have K > 0");
    static_assert(K < WARP_SIZE, "Top K must have K < WARP_SIZE");
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

} // end of namespace reduce_topk

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
__device__ T calcSoftmax(cg::thread_block_tile<WARP_SIZE> const& warp, T score, int32_t laneIdx, int32_t NumTopExperts)
{
    T maxScore = T{-INFINITY};
    if (laneIdx < NumTopExperts)
    {
        maxScore = score >= maxScore ? score : maxScore;
    }
    maxScore = cg::reduce(warp, maxScore, cg::greater<T>());

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
        score = static_cast<T>(newScore / sumScore);
    }

    return score;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename InputT, typename OutputT, typename IdxT, int MaxNumExperts, int MaxNumTopExperts>
__global__ void renormMoeRoutingKernel(InputT* routerLogits, OutputT* topkValues, IdxT* topkIndices,
    int32_t const numTokens, int32_t const numExperts, int32_t const topK)
{

    uint32_t const blockRank = blockIdx.x;
    uint32_t const tIdx = BLOCK_SIZE * blockRank + threadIdx.x;
    uint32_t const warpIdx = tIdx / WARP_SIZE;
    uint32_t const laneIdx = tIdx % WARP_SIZE;
    uint32_t const warpNum = gridDim.x * WARPS_PER_BLOCK;
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<WARP_SIZE>(block);

    InputT minScore = InputT{-INFINITY};
    for (uint32_t tokenId = warpIdx; tokenId < numTokens; tokenId += warpNum)
    {
        auto scoreOffset = tokenId * numExperts;
        auto outputOffset = tokenId * topK;
        InputT inputScore[MaxNumExperts / WARP_SIZE];
        IdxT inputIndex[MaxNumExperts / WARP_SIZE];

        InputT warpTopKScore[MaxNumTopExperts];
        IdxT warpTopKExpertIdx[MaxNumTopExperts];

        // Load scores and indices for this warp
        for (uint32_t i = 0; i < MaxNumExperts / WARP_SIZE; ++i)
        {
            auto expertIdx = i * WARP_SIZE + laneIdx;
            inputScore[i]
                = expertIdx < numExperts ? static_cast<InputT>(routerLogits[scoreOffset + expertIdx]) : minScore;
            inputIndex[i] = expertIdx;
        }

        // Reduce topK scores and indices for this warp
        reduce_topk::reduceTopK(warp, warpTopKScore, warpTopKExpertIdx, inputScore, inputIndex, minScore);

        // Perform softmax on topK scores
        auto score = calcSoftmax(warp,
            laneIdx < topK ? static_cast<float>(warpTopKScore[laneIdx]) : static_cast<float>(minScore), laneIdx, topK);
        if (laneIdx < topK)
        {
            topkValues[outputOffset + laneIdx] = static_cast<OutputT>(score);
            topkIndices[outputOffset + laneIdx] = warpTopKExpertIdx[laneIdx];
        }
    } // end for tokenId
}

template <typename InputT, typename OutputT, typename IdxT>
void invokeRenormMoeRouting(InputT* routerLogits, OutputT* topkValues, IdxT* topkIndices, int64_t const numTokens,
    int64_t const numExperts, int64_t const topK, cudaStream_t const stream)
{

    const uint32_t maxNumBlocks = 1024;
    const uint32_t numBlocks = std::min(static_cast<uint32_t>((numTokens - 1) / WARPS_PER_BLOCK + 1), maxNumBlocks);

    constexpr uint32_t maxNumExperts = 128;
    constexpr uint32_t maxNumTopExperts = 8;
    auto* kernelInstance = &renormMoeRoutingKernel<InputT, OutputT, IdxT, maxNumExperts, maxNumTopExperts>;

    if (topK <= 4)
    {
        kernelInstance = &renormMoeRoutingKernel<InputT, OutputT, IdxT, maxNumExperts, 4>;
    }
    else if (topK <= 2)
    {
        kernelInstance = &renormMoeRoutingKernel<InputT, OutputT, IdxT, maxNumExperts, 2>;
    }
    dim3 renormMoeRoutingGridDim(numBlocks);
    dim3 renormMoeRoutingBlockDim(BLOCK_SIZE);
    cudaLaunchConfig_t config;
    config.gridDim = renormMoeRoutingGridDim;
    config.blockDim = renormMoeRoutingBlockDim;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
    config.numAttrs = 1;
    config.attrs = attrs;
    cudaLaunchKernelEx(&config, kernelInstance, routerLogits, topkValues, topkIndices, static_cast<int32_t>(numTokens),
        static_cast<int32_t>(numExperts), static_cast<int32_t>(topK));
    sync_check_cuda_error(stream);
}

#define INSTANTIATE_RENORM_MOE_ROUTING(InputT, OutputT, IdxT)                                                          \
    template void invokeRenormMoeRouting<InputT, OutputT, IdxT>(InputT * routerLogits, OutputT * topkValues,           \
        IdxT * topkIndices, int64_t const numTokens, int64_t const numExperts, int64_t const topK,                     \
        cudaStream_t const stream);

INSTANTIATE_RENORM_MOE_ROUTING(float, float, int32_t);
#ifdef ENABLE_BF16
INSTANTIATE_RENORM_MOE_ROUTING(cutlass::bfloat16_t, float, int32_t);
#endif

} // namespace tensorrt_llm::kernels
