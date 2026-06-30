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

#include <cub/device/device_memcpy.cuh>

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

// ---- cub::DeviceMemcpy::Batched backend (opt-in) -------------------------------------------------
// Each `srcs[i]`/`dsts[i]` is a 64-bit DEVICE address; reinterpret the uint64 arrays as arrays of
// void* (8 bytes each on LP64) so cub sees per-buffer source/dest pointers. `sizes` (uint32) is the
// per-buffer byte count. cub is tuned to load-balance across wildly varying buffer sizes — the point
// of trying it for the many-small-desc gather/scatter.
cudaError_t batchedCopyCubTempBytes(std::uint32_t maxN, std::size_t& outBytes)
{
    outBytes = 0;
    if (maxN == 0)
    {
        return cudaSuccess;
    }
    // Size query: with d_temp_storage == nullptr cub only computes the workspace size from the buffer
    // count (it does not dereference the iterators), so typed null iterators are fine here.
    void* const* nullBufs = nullptr;
    std::uint32_t const* nullSizes = nullptr;
    return cub::DeviceMemcpy::Batched(nullptr, outBytes, nullBufs, nullBufs, nullSizes, maxN);
}

cudaError_t launchBatchedCopyCub(std::uint64_t const* srcs, std::uint64_t const* dsts, std::uint32_t const* sizes,
    std::uint32_t n, cudaStream_t stream, void* dTemp, std::size_t tempBytes)
{
    if (n == 0)
    {
        return cudaSuccess;
    }
    auto const* inBufs = reinterpret_cast<void* const*>(srcs);  // each uint64 element == a device address
    auto const* outBufs = reinterpret_cast<void* const*>(dsts); // cub writes the bulk bytes into *outBufs[i]
    std::size_t tb = tempBytes;                                 // cub takes temp_storage_bytes by reference
    return cub::DeviceMemcpy::Batched(dTemp, tb, inBufs, outBufs, sizes, n, stream);
}

} // namespace tensorrt_llm::executor::kv_cache::bounce
