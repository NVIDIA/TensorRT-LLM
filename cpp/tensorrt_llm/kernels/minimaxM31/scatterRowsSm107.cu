/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/kernels/minimaxM31/scatterRowsSm107.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM31
{
namespace
{

constexpr int kThreads = 256;
constexpr int kChunk = 16384; // elements per CTA

template <bool kVec>
__global__ void __launch_bounds__(kThreads) scatterRowsKernel(float* __restrict__ dst, int64_t dstSlotStride,
    int64_t dstRowStride, float const* __restrict__ src, int64_t const* __restrict__ slotIds, int32_t K, int32_t v)
{
    int32_t const row = blockIdx.y; // i * K + k
    int32_t const i = row / K;
    int32_t const k = row % K;
    float* d = dst + slotIds[i] * dstSlotStride + static_cast<int64_t>(k) * dstRowStride;
    float const* s = src + static_cast<int64_t>(row) * v;
    int32_t const begin = blockIdx.x * kChunk;
    int32_t const end = begin + min(v - begin, kChunk);
    if constexpr (kVec)
    {
        for (int32_t j = begin / 4 + threadIdx.x; j < end / 4; j += kThreads)
        {
            reinterpret_cast<float4*>(d)[j] = reinterpret_cast<float4 const*>(s)[j];
        }
    }
    else
    {
        for (int64_t j = static_cast<int64_t>(begin) + threadIdx.x; j < end; j += kThreads)
        {
            d[j] = s[j];
        }
    }
}

} // namespace

void invokeScatterRows(float* dst, int64_t dstSlotStride, int64_t dstRowStride, float const* src,
    int64_t const* slotIds, int32_t n, int32_t K, int32_t v, cudaStream_t stream)
{
    dim3 const grid((v - 1) / kChunk + 1, n * K);
    bool const vec = v % 4 == 0 && dstSlotStride % 4 == 0 && dstRowStride % 4 == 0
        && reinterpret_cast<uintptr_t>(dst) % 16 == 0 && reinterpret_cast<uintptr_t>(src) % 16 == 0;
    if (vec)
    {
        scatterRowsKernel<true><<<grid, kThreads, 0, stream>>>(dst, dstSlotStride, dstRowStride, src, slotIds, K, v);
    }
    else
    {
        scatterRowsKernel<false><<<grid, kThreads, 0, stream>>>(dst, dstSlotStride, dstRowStride, src, slotIds, K, v);
    }
}

} // namespace minimaxM31
} // namespace kernels

TRTLLM_NAMESPACE_END
