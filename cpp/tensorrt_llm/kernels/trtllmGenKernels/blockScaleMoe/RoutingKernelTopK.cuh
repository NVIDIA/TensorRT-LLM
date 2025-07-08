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

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/cub.cuh>

namespace moe::dev::routing
{

namespace topk
{

////////////////////////////////////////////////////////////////////////////////////////////////////
namespace cg = cooperative_groups;

////////////////////////////////////////////////////////////////////////////////////////////////////

static constexpr int WarpSize = 32;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename TypeExpW_>
struct TopKRedType
{
    using TypeExpW = TypeExpW_;
    static_assert(
        std::is_same_v<TypeExpW, float> || std::is_same_v<TypeExpW, half> || std::is_same_v<TypeExpW, __nv_bfloat16>,
        "Top K reduction only implemented for float, float16 and bfloat16");

    using TypeCmp = std::conditional_t<sizeof(TypeExpW) == 4, uint64_t, uint32_t>;
    using IdxT = std::conditional_t<sizeof(TypeExpW) == 4, int32_t, int16_t>;
    static constexpr int moveBits = (sizeof(TypeExpW) == 4) ? 32 : 16;
    static constexpr int maxIdx = 65535;

    TypeCmp compVal;

    static __host__ __device__ inline TypeCmp makeCmpVal(TypeExpW val, int32_t idx = 0)
    {
        auto valueBits
            = cub::Traits<TypeExpW>::TwiddleIn(reinterpret_cast<typename cub::Traits<TypeExpW>::UnsignedBits&>(val));
        TypeCmp compactTmp = reinterpret_cast<TypeCmp&>(valueBits);
        compactTmp = (compactTmp << moveBits) | (0xFFFF & (maxIdx - idx));
        // Use 65535 minus idx to give higher priority to elements with smaller indices.
        return compactTmp;
    }

    static __host__ __device__ inline void unpack(TypeExpW& value, int32_t& index, TypeCmp cmp)
    {
        // Since idx is always smaller than 65536 and positive, we can directly use it as the lower 16
        // bits
        index = maxIdx - static_cast<int32_t>(cmp & 0xFFFF);

        auto compactTmp = cmp >> moveBits;
        auto valueBits = cub::Traits<TypeExpW>::TwiddleOut(
            reinterpret_cast<typename cub::Traits<TypeExpW>::UnsignedBits&>(compactTmp));
        value = reinterpret_cast<TypeExpW&>(valueBits);
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
#if defined(TLLM_GEN_HAS_FAST_REDUX)
        static constexpr bool UseCg = false;
#else
        static constexpr bool UseCg = true;
#endif
        if constexpr (UseCg || sizeof(TypeCmp) == 8)
        {
            return cg::reduce(warp, compVal, cg::greater<TypeCmp>{});
        }
        else
        {
            TypeCmp result;
            asm("redux.sync.max.u32 %0, %1, 0xffffffff;\n" : "=r"(result) : "r"(compVal));
            return result;
        }
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

////////////////////////////////////////////////////////////////////////////////////////////////////

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

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int K, typename Type>
__forceinline__ __device__ void reduceTopK(cg::thread_block_tile<WarpSize> const& warp, Type (&out)[K],
    int32_t (&outIdx)[K], Type value, int32_t idx, Type const minValue)
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

template <int K, typename Type, int N>
__forceinline__ __device__ void reduceTopK(cg::thread_block_tile<WarpSize> const& warp, Type (&out)[K],
    int32_t (&outIdx)[K], Type (&value)[N], int32_t (&idx)[N], Type const minValue)
{
    static_assert(K > 0, "Top K must have K > 0");
    static_assert(K < WarpSize, "Top K must have K < WarpSize");
    static_assert(N > 0, "Top K must have N > 0");
    static_assert(N < 5, "Only support candidates number less than or equal to 128");
    using RedType = TopKRedType<Type>;
    RedType topK[N];
#pragma unroll
    for (int nn = 0; nn < N; ++nn)
    {
        topK[nn] = RedType{value[nn], idx[nn]};
    }

    Sort<N, RedType>::run(topK);

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
} // namespace topk
} // namespace moe::dev::routing
