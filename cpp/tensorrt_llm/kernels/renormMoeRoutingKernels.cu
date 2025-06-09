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

#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/kernels/renormMoeRoutingKernels.h"
#include <climits> // For INT_MAX
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/cub.cuh>
#include <cuda/std/limits> // For numeric_limits
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

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000 && defined(__CUDA_ARCH_FEAT_SM100_ALL))
#define TLLM_GEN_ENABLE_FAST_REDUX
#endif

template <typename T_>
struct TopKRedType
{
    using T = T_;
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, half> || std::is_same_v<T, __nv_bfloat16>,
        "Top K reduction only implemented for float, float16 and bfloat16");

    using TypeCmp = std::conditional_t<sizeof(T) == 4, uint64_t, uint32_t>;
    using IdxT = std::conditional_t<sizeof(T) == 4, int32_t, int16_t>;
    static constexpr int moveBits = (sizeof(T) == 4) ? 32 : 16;
    static constexpr int maxIdx = 65535;
    TypeCmp compValIdx;

    static __host__ __device__ inline TypeCmp makeCmpVal(T val, int32_t idx = 0)
    {
        auto valueBits = cub::Traits<T>::TwiddleIn(reinterpret_cast<typename cub::Traits<T>::UnsignedBits&>(val));
        TypeCmp compactTmp = reinterpret_cast<TypeCmp&>(valueBits);
        compactTmp = (compactTmp << moveBits) | (0xFFFF & (maxIdx - idx));
        // Use 65535 minus idx to give higher priority to elements with smaller indices.
        return compactTmp;
    }

    static __host__ __device__ void unpack(T& value, int32_t& index, TypeCmp cmp)
    {
        // Since “65535-idx” is always smaller than 65536 and positive, we can directly use it as the lower 16 bits
        index = maxIdx - static_cast<int32_t>((cmp & 0xFFFF));

        auto compactTmp = cmp >> moveBits;
        auto valueBits
            = cub::Traits<T>::TwiddleOut(reinterpret_cast<typename cub::Traits<T>::UnsignedBits&>(compactTmp));
        value = reinterpret_cast<T&>(valueBits);
    }

    __host__ __device__ TopKRedType() = default;

    __host__ __device__ TopKRedType(T val, int32_t idx)
        : compValIdx(makeCmpVal(val, idx))
    {
    }

    __host__ __device__ operator TypeCmp() const noexcept
    {
        return compValIdx;
    }

    __device__ inline TypeCmp reduce(cg::thread_block_tile<WARP_SIZE> const& warp)
    {
#if defined(TLLM_GEN_ENABLE_FAST_REDUX)
        static constexpr bool UseCg = false;
#else
        static constexpr bool UseCg = true;
#endif
        if constexpr (UseCg || sizeof(TypeCmp) == 8)
        {
            return cg::reduce(warp, compValIdx, cg::greater<TypeCmp>{});
        }
        else
        {
            TypeCmp result;
            asm("redux.sync.max.u32 %0, %1, 0xffffffff;\n" : "=r"(result) : "r"(compValIdx));
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

#define TOPK_SWAP(I, J)                                                                                                \
    {                                                                                                                  \
        auto pairMin = min(topK[I].compValIdx, topK[J].compValIdx);                                                    \
        auto pairMax = max(topK[I].compValIdx, topK[J].compValIdx);                                                    \
        topK[I].compValIdx = pairMax;                                                                                  \
        topK[J].compValIdx = pairMin;                                                                                  \
    }

template <int N, typename RedType>
struct Sort;

template <typename RedType>
struct Sort<1, RedType>
{
    static __device__ void run(RedType* topK) {}
};

template <typename RedType>
struct Sort<2, RedType>
{
    static __device__ void run(RedType* topK)
    {
        TOPK_SWAP(0, 1);
    }
};

template <typename RedType>
struct Sort<3, RedType>
{
    static __device__ void run(RedType* topK)
    {
        TOPK_SWAP(0, 1);
        TOPK_SWAP(1, 2);
        TOPK_SWAP(0, 1);
    }
};

template <typename RedType>
struct Sort<4, RedType>
{
    static __device__ void run(RedType* topK)
    {
        TOPK_SWAP(0, 2);
        TOPK_SWAP(1, 3);
        TOPK_SWAP(0, 1);
        TOPK_SWAP(2, 3);
        TOPK_SWAP(1, 2);
    }
};

template <int K, typename Type, int N, bool IsSorted = false>
__device__ void reduceTopK(cg::thread_block_tile<WARP_SIZE> const& warp, Type (&out)[K], int32_t (&outIdx)[K],
    Type (&value)[N], int32_t (&idx)[N], Type minValue)
{
    static_assert(K > 0, "Top K must have K > 0");
    static_assert(K < WARP_SIZE, "Top K must have K < WARP_SIZE");
    static_assert(N > 0, "Top K must have N > 0");
    static_assert(N < 5, "Only support candidates number less than or equal to 128");
    using RedType = TopKRedType<Type>;
    RedType topK[N];
#pragma unroll
    for (int nn = 0; nn < N; ++nn)
    {
        topK[nn] = RedType{value[nn], idx[nn]};
    }

    if constexpr (!IsSorted)
    {
        Sort<N, RedType>::run(topK);
    }
    typename RedType::TypeCmp packedMax{};
#pragma unroll
    for (int kk = 0; kk < K; ++kk)
    {
        bool update = kk > 0 && packedMax == topK[0].compValIdx;
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

int nextPowerOfTwo(int num)
{
    if (num <= 0)
    {
        return 1; // Handle invalid input
    }
    int power = 1;
    while (power < num)
    {
        // Check for overflow before shifting
        if (power > INT_MAX / 2)
        {
            return power;
        }
        power <<= 1;
    }
    return power;
}

#define CASE(MAX_NUM_EXPERTS)                                                                                          \
    case MAX_NUM_EXPERTS:                                                                                              \
        switch (maxNumTopExperts)                                                                                      \
        {                                                                                                              \
        case 1: kernelInstance = &renormMoeRoutingKernel<InputT, OutputT, IdxT, MAX_NUM_EXPERTS, 1>; break;            \
        case 2: kernelInstance = &renormMoeRoutingKernel<InputT, OutputT, IdxT, MAX_NUM_EXPERTS, 2>; break;            \
        case 4: kernelInstance = &renormMoeRoutingKernel<InputT, OutputT, IdxT, MAX_NUM_EXPERTS, 4>; break;            \
        case 8: kernelInstance = &renormMoeRoutingKernel<InputT, OutputT, IdxT, MAX_NUM_EXPERTS, 8>; break;            \
        default: kernelInstance = nullptr; break;                                                                      \
        }                                                                                                              \
        break;

template <typename InputT, typename OutputT, typename IdxT>
void invokeRenormMoeRouting(InputT* routerLogits, OutputT* topkValues, IdxT* topkIndices, int64_t const numTokens,
    int64_t const numExperts, int64_t const topK, cudaStream_t const stream)
{

    const uint32_t maxNumBlocks = 1024;
    const uint32_t numBlocks = std::min(static_cast<uint32_t>((numTokens - 1) / WARPS_PER_BLOCK + 1), maxNumBlocks);

    uint32_t maxNumExperts = nextPowerOfTwo(numExperts) < 32 ? 32 : nextPowerOfTwo(numExperts);
    uint32_t maxNumTopExperts = nextPowerOfTwo(topK);

    auto* kernelInstance = &renormMoeRoutingKernel<InputT, OutputT, IdxT, 128, 8>;

    switch (maxNumExperts)
    {
        CASE(32)
        CASE(64)
        CASE(96)
        CASE(128)
    default: kernelInstance = nullptr; break;
    }

    if (kernelInstance == nullptr)
    {
        TLLM_CHECK_WITH_INFO(kernelInstance != nullptr, "Can not find corresponding kernel instance.");
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
INSTANTIATE_RENORM_MOE_ROUTING(half, float, int32_t);
#ifdef ENABLE_BF16
INSTANTIATE_RENORM_MOE_ROUTING(__nv_bfloat16, float, int32_t);
#endif

} // namespace tensorrt_llm::kernels
