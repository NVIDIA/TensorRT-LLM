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

#include "tensorrt_llm/executor/cache_transmission/nixl_utils/direct/DirectPeerCopy.h"

namespace tensorrt_llm::executor::kv_cache::direct
{

namespace
{

constexpr int kUnroll = 4;

__device__ __forceinline__ uint4 loadStreaming(uint4 const* p)
{
    // Read-once source: cache-streaming load, so the copied bytes do not displace other cached data.
    return __ldcs(p);
}

// A fixed grid of CTAs walks the work units (host-split pieces of at most unitBytes) grid-stride. Each CTA copies
// one unit with all its threads: kUnroll independent 16-byte loads in flight per thread before the stores, so a
// few dozen CTAs keep enough bytes in flight to fill the NVLink write path without taking the whole GPU.
// The next unit's plan entry is loaded before the current unit is copied, so its latency (the plan is in mapped host
// memory) hides behind the copy.
__global__ void __launch_bounds__(1024) directCopyKernel(std::uint64_t const* __restrict__ srcs,
    std::uint64_t const* __restrict__ dsts, std::uint32_t const* __restrict__ sizes, std::uint32_t n)
{
    std::uint32_t e = blockIdx.x;
    if (e >= n)
    {
        return;
    }
    std::uint64_t nextSrc = srcs[e];
    std::uint64_t nextDst = dsts[e];
    std::uint32_t nextSize = sizes[e];
    for (; e < n; e += gridDim.x)
    {
        auto const* src = reinterpret_cast<char const*>(nextSrc);
        auto* dst = reinterpret_cast<char*>(nextDst);
        std::uint32_t const size = nextSize;
        std::uint32_t const en = e + gridDim.x;
        if (en < n)
        {
            nextSrc = srcs[en];
            nextDst = dsts[en];
            nextSize = sizes[en];
        }
        auto const mask = reinterpret_cast<std::uintptr_t>(src) | reinterpret_cast<std::uintptr_t>(dst)
            | static_cast<std::uintptr_t>(size);
        if ((mask & 0xFU) == 0U)
        {
            auto const* s4 = reinterpret_cast<uint4 const*>(src);
            auto* d4 = reinterpret_cast<uint4*>(dst);
            std::uint32_t const n4 = size >> 4U;
            std::uint32_t const stride = blockDim.x;
            std::uint32_t i = threadIdx.x;
            for (; i + (kUnroll - 1) * stride < n4; i += kUnroll * stride)
            {
                uint4 v[kUnroll];
#pragma unroll
                for (int u = 0; u < kUnroll; ++u)
                {
                    v[u] = loadStreaming(s4 + i + u * stride);
                }
#pragma unroll
                for (int u = 0; u < kUnroll; ++u)
                {
                    d4[i + u * stride] = v[u];
                }
            }
            for (; i < n4; i += stride)
            {
                d4[i] = loadStreaming(s4 + i);
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
}

// Small-unit variant: one warp per plan entry (grid-stride over warps), for plans whose units are a few KiB each
// (per-layer head slices of KV blocks). Every warp keeps its own entry in flight, so the plan-entry latency and the
// tail of each copy overlap across 32x more entries than with one CTA per entry.
__global__ void __launch_bounds__(1024) directCopyWarpKernel(std::uint64_t const* __restrict__ srcs,
    std::uint64_t const* __restrict__ dsts, std::uint32_t const* __restrict__ sizes, std::uint32_t n)
{
    std::uint32_t const lane = threadIdx.x & 31U;
    std::uint32_t const warps = (gridDim.x * blockDim.x) >> 5U;
    std::uint32_t e = (blockIdx.x * blockDim.x + threadIdx.x) >> 5U;
    if (e >= n)
    {
        return;
    }
    std::uint64_t nextSrc = srcs[e];
    std::uint64_t nextDst = dsts[e];
    std::uint32_t nextSize = sizes[e];
    for (; e < n; e += warps)
    {
        auto const* src = reinterpret_cast<char const*>(nextSrc);
        auto* dst = reinterpret_cast<char*>(nextDst);
        std::uint32_t const size = nextSize;
        std::uint32_t const en = e + warps;
        if (en < n)
        {
            nextSrc = srcs[en];
            nextDst = dsts[en];
            nextSize = sizes[en];
        }
        auto const mask = reinterpret_cast<std::uintptr_t>(src) | reinterpret_cast<std::uintptr_t>(dst)
            | static_cast<std::uintptr_t>(size);
        if ((mask & 0xFU) == 0U)
        {
            auto const* s4 = reinterpret_cast<uint4 const*>(src);
            auto* d4 = reinterpret_cast<uint4*>(dst);
            std::uint32_t const n4 = size >> 4U;
            std::uint32_t i = lane;
            for (; i + (kUnroll - 1) * 32U < n4; i += kUnroll * 32U)
            {
                uint4 v[kUnroll];
#pragma unroll
                for (int u = 0; u < kUnroll; ++u)
                {
                    v[u] = loadStreaming(s4 + i + u * 32U);
                }
#pragma unroll
                for (int u = 0; u < kUnroll; ++u)
                {
                    d4[i + u * 32U] = v[u];
                }
            }
            for (; i < n4; i += 32U)
            {
                d4[i] = loadStreaming(s4 + i);
            }
        }
        else
        {
            for (std::uint32_t i = lane; i < size; i += 32U)
            {
                dst[i] = src[i];
            }
        }
    }
}

} // namespace

cudaError_t launchDirectCopyWarpPerEntry(std::uint64_t const* srcs, std::uint64_t const* dsts,
    std::uint32_t const* sizes, std::uint32_t n, std::uint32_t maxCtas, std::uint32_t threads, cudaStream_t stream)
{
    if (n == 0)
    {
        return cudaSuccess;
    }
    std::uint32_t const warpsPerCta = threads / 32U;
    std::uint32_t const needCtas = (n + warpsPerCta - 1) / warpsPerCta;
    std::uint32_t const grid = needCtas < maxCtas ? needCtas : maxCtas;
    directCopyWarpKernel<<<grid, threads, 0, stream>>>(srcs, dsts, sizes, n);
    return cudaGetLastError();
}

cudaError_t launchDirectCopy(std::uint64_t const* srcs, std::uint64_t const* dsts, std::uint32_t const* sizes,
    std::uint32_t n, std::uint32_t maxCtas, std::uint32_t threads, cudaStream_t stream)
{
    if (n == 0)
    {
        return cudaSuccess;
    }
    std::uint32_t const grid = n < maxCtas ? n : maxCtas;
    directCopyKernel<<<grid, threads, 0, stream>>>(srcs, dsts, sizes, n);
    return cudaGetLastError();
}

} // namespace tensorrt_llm::executor::kv_cache::direct
