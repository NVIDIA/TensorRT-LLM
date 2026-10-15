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

#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/GatherScatterKernel.h"

namespace tensorrt_llm::executor::kv_cache::bounce
{

namespace
{

// One thread block per buffer. Within a block, threads stride over the buffer. When src, dst
// and size are all 16-byte aligned, a vectorized uint4 path gives coalesced 16-byte stores.
__global__ void batchedCopyKernel(
    std::uint64_t const* srcs, std::uint64_t const* dsts, std::uint32_t const* sizes, std::uint32_t n)
{
    std::uint32_t const buf = blockIdx.x;
    if (buf >= n)
    {
        return;
    }
    auto const* src = reinterpret_cast<char const*>(srcs[buf]);
    auto* dst = reinterpret_cast<char*>(dsts[buf]);
    std::uint32_t const size = sizes[buf];

    auto const addrMask = reinterpret_cast<std::uintptr_t>(src) | reinterpret_cast<std::uintptr_t>(dst)
        | static_cast<std::uintptr_t>(size);
    if ((addrMask & 0xFU) == 0U)
    {
        auto const* s4 = reinterpret_cast<uint4 const*>(src);
        auto* d4 = reinterpret_cast<uint4*>(dst);
        std::uint32_t const n4 = size >> 4U;
        for (std::uint32_t i = threadIdx.x; i < n4; i += blockDim.x)
        {
            d4[i] = s4[i];
        }
    }
    else
    {
        for (std::uint32_t i = threadIdx.x; i < size; i += blockDim.x)
        {
            dst[i] = src[i];
        }
    }
}

} // namespace

cudaError_t launchBatchedCopy(std::uint64_t const* srcs, std::uint64_t const* dsts, std::uint32_t const* sizes,
    std::uint32_t n, cudaStream_t stream)
{
    if (n == 0)
    {
        return cudaSuccess;
    }
    constexpr int kThreads = 256;
    batchedCopyKernel<<<n, kThreads, 0, stream>>>(srcs, dsts, sizes, n);
    return cudaGetLastError();
}

} // namespace tensorrt_llm::executor::kv_cache::bounce
