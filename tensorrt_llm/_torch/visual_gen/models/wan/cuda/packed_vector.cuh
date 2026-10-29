/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

// Packed vector helpers (runtime-compiled copy of tensorrt_llm/common/packedVector.cuh).

#pragma once

#include <cuda/std/cstddef>

namespace vaewino
{

namespace common
{

__host__ __device__ inline constexpr unsigned largestPow2Divisor(unsigned v)
{
    return v ? (v & (~v + 1)) : 1;
}

// Simple packed vector type for loading N elements of type T.
// Alignment is the largest power-of-2 factor of the total byte size,
// so it is always valid even when N * sizeof(T) is not a power of 2 (e.g. 3 x 2 = 6).
template <typename T, int N = sizeof(float4) / sizeof(T)>
struct alignas(largestPow2Divisor(N * sizeof(T))) PackedAligned
{
    static_assert(N > 0, "PackedAligned instantiated with N == 0");
    T val[N];
    static constexpr int count = N;
    using dtype = T;
};

template <class load_t>
__device__ __forceinline__ auto make_zeros() -> load_t
{
    load_t ret{};
#pragma unroll
    for (int i = 0; i < ret.count; i++)
        ret.val[i] = typename load_t::dtype{};
    return ret;
}

} // namespace common

} // namespace vaewino
