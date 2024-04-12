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

#pragma once
#include "tensorrt_llm/kernels/weightOnlyBatchedGemv/common.h"

namespace tensorrt_llm
{
namespace kernels
{
namespace weight_only
{
struct FP16DetailsA
{
    using Type = half;
    using Type2 = half2;
    static constexpr int kElemBits = 16;
};

struct BF16DetailsA
{
    using Type = __nv_bfloat16;
    using Type2 = __nv_bfloat162;
    static constexpr int kElemBits = 16;
};

struct Int8DetailsW
{
    static constexpr int kElemBits = 8;
};

struct Int4DetailsW
{
    static constexpr int kElemBits = 4;
};

template <typename TypeDetailsA, typename TypeDetailsW>
struct ColumnMajor
{
    using DetailsA = TypeDetailsA;
    using DetailsW = TypeDetailsW;
    using AccessTypeA = float4;
    using AccessTypeW = int;
    static constexpr int kAccessSize = 128;
    static constexpr int kStepK = kAccessSize / TypeDetailsA::kElemBits;
    static constexpr int kTileSize = 64;
    static constexpr int kInterleave = 1;

    struct Mapper
    {
        __device__ __forceinline__ int operator()(int i)
        {
            return i;
        }
    };
};

template <typename TypeDetailsA, typename TypeDetailsW>
struct ColumnMajorInterleaved
{
    using DetailsA = TypeDetailsA;
    using DetailsW = TypeDetailsW;
    using AccessTypeA = float4;
    using AccessTypeW = int4;
    static constexpr int kAccessSize = 128;
    static constexpr int kStepK = kAccessSize / TypeDetailsW::kElemBits;
    static constexpr int kTileSize = 64;
    static constexpr int kInterleave = 128 * 8 / (kTileSize * TypeDetailsW::kElemBits);

    struct Mapper
    {
        __device__ __forceinline__ int operator()(int i)
        {
            return (i % 8) / 2 * kInterleave * 2 + i % 2 + i / 8 * 2;
        }
    };
};

template <typename TypeDetailsA_, typename TypeDetailsW_, template <typename, typename> class LayoutDeatils_,
    bool UseInterleavedConverter>
struct KernelDetails
{
    using TypeDetailsA = TypeDetailsA_;
    using TypeDetailsW = TypeDetailsW_;
    using LayoutDeatils = LayoutDeatils_<TypeDetailsA, TypeDetailsW>;
    using AccessTypeA = typename LayoutDeatils::AccessTypeA;
    using AccessTypeW = typename LayoutDeatils::AccessTypeW;
    static constexpr int kWarpSize = 32;
    static constexpr int kStepK = LayoutDeatils::kStepK;
    static constexpr int kAccessNumA = kStepK * TypeDetailsA::kElemBits / (sizeof(AccessTypeA) * 8);
    static constexpr int kAccessNumW = kStepK * TypeDetailsW::kElemBits / (sizeof(AccessTypeW) * 8);
    static constexpr int kInterleave = LayoutDeatils::kInterleave;
    static constexpr int kThreadsPerInterleavedTile = LayoutDeatils::kTileSize / kStepK;
    static constexpr int kElemsPerByteW = 8 / TypeDetailsW::kElemBits;
    static constexpr bool kUseInterleavedConverter = UseInterleavedConverter;
};
} // namespace weight_only
} // namespace kernels
} // namespace tensorrt_llm
