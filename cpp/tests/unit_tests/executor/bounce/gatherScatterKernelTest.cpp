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

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

namespace b = tensorrt_llm::executor::kv_cache::bounce;

#define CUDA_OK(call) ASSERT_EQ((call), cudaSuccess) << "CUDA call failed: " #call

namespace
{
std::uint64_t alignUp(std::uint64_t value, std::uint64_t align)
{
    return (value + align - 1) / align * align;
}

// Distinct per-(buffer,byte) pattern so any mis-routing/overlap is caught.
unsigned char pattern(std::size_t buf, std::size_t idx)
{
    return static_cast<unsigned char>((buf * 131 + idx * 7 + 13) & 0xFF);
}
} // namespace

// Gather scattered src buffers into a packed slot, then scatter the slot back out to fresh
// dst buffers; verify every byte survives the round trip. Mixes 16B-aligned sizes (uint4 path)
// and unaligned sizes (byte path).
TEST(GatherScatterKernel, GatherThenScatterRoundTrip)
{
    int devs = 0;
    CUDA_OK(cudaGetDeviceCount(&devs));
    if (devs == 0)
    {
        GTEST_SKIP() << "no CUDA device";
    }

    std::vector<std::uint32_t> sizes{64, 100, 256, 16, 4096, 17, 1, 32, 3, 512};
    auto const n = static_cast<std::uint32_t>(sizes.size());

    // 256-align each buffer's offset so device addresses hit the vectorized path when the size
    // is also a multiple of 16; unaligned sizes still exercise the byte path.
    std::vector<std::uint64_t> off(n);
    std::uint64_t cur = 0;
    for (std::uint32_t i = 0; i < n; ++i)
    {
        off[i] = cur;
        cur = alignUp(cur + sizes[i], 256);
    }
    std::uint64_t const totalAligned = cur;

    // Host reference: fill src region with the pattern.
    std::vector<unsigned char> srcHost(totalAligned, 0);
    for (std::uint32_t i = 0; i < n; ++i)
    {
        for (std::uint32_t j = 0; j < sizes[i]; ++j)
        {
            srcHost[off[i] + j] = pattern(i, j);
        }
    }

    // Device buffers: src region, packed slot, dst region.
    void *dSrc = nullptr, *dSlot = nullptr, *dDst = nullptr;
    CUDA_OK(cudaMalloc(&dSrc, totalAligned));
    CUDA_OK(cudaMalloc(&dSlot, totalAligned));
    CUDA_OK(cudaMalloc(&dDst, totalAligned));
    CUDA_OK(cudaMemcpy(dSrc, srcHost.data(), totalAligned, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemset(dDst, 0, totalAligned));

    auto base = [](void* p, std::uint64_t o) { return reinterpret_cast<std::uint64_t>(static_cast<char*>(p) + o); };

    // Plan arrays (host) -> device. Gather: src region -> slot. Scatter: slot -> dst region.
    std::vector<std::uint64_t> gatherSrc(n), gatherDst(n), scatterSrc(n), scatterDst(n);
    for (std::uint32_t i = 0; i < n; ++i)
    {
        gatherSrc[i] = base(dSrc, off[i]);
        gatherDst[i] = base(dSlot, off[i]);
        scatterSrc[i] = base(dSlot, off[i]);
        scatterDst[i] = base(dDst, off[i]);
    }

    auto toDev = [&](std::vector<std::uint64_t> const& h) -> std::uint64_t*
    {
        std::uint64_t* d = nullptr;
        EXPECT_EQ(cudaMalloc(reinterpret_cast<void**>(&d), h.size() * sizeof(std::uint64_t)), cudaSuccess);
        EXPECT_EQ(cudaMemcpy(d, h.data(), h.size() * sizeof(std::uint64_t), cudaMemcpyHostToDevice), cudaSuccess);
        return d;
    };
    std::uint64_t* dGatherSrc = toDev(gatherSrc);
    std::uint64_t* dGatherDst = toDev(gatherDst);
    std::uint64_t* dScatterSrc = toDev(scatterSrc);
    std::uint64_t* dScatterDst = toDev(scatterDst);
    std::uint32_t* dSizes = nullptr;
    CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&dSizes), n * sizeof(std::uint32_t)));
    CUDA_OK(cudaMemcpy(dSizes, sizes.data(), n * sizeof(std::uint32_t), cudaMemcpyHostToDevice));

    cudaStream_t stream{};
    CUDA_OK(cudaStreamCreate(&stream));
    CUDA_OK(b::launchBatchedCopy(dGatherSrc, dGatherDst, dSizes, n, stream));   // gather
    CUDA_OK(b::launchBatchedCopy(dScatterSrc, dScatterDst, dSizes, n, stream)); // scatter
    CUDA_OK(cudaStreamSynchronize(stream));

    std::vector<unsigned char> dstHost(totalAligned, 0xEE);
    CUDA_OK(cudaMemcpy(dstHost.data(), dDst, totalAligned, cudaMemcpyDeviceToHost));

    for (std::uint32_t i = 0; i < n; ++i)
    {
        for (std::uint32_t j = 0; j < sizes[i]; ++j)
        {
            ASSERT_EQ(dstHost[off[i] + j], pattern(i, j)) << "mismatch buf=" << i << " byte=" << j;
        }
    }

    cudaFree(dSrc);
    cudaFree(dSlot);
    cudaFree(dDst);
    cudaFree(dGatherSrc);
    cudaFree(dGatherDst);
    cudaFree(dScatterSrc);
    cudaFree(dScatterDst);
    cudaFree(dSizes);
    cudaStreamDestroy(stream);
}

TEST(GatherScatterKernel, ZeroBuffersIsNoop)
{
    cudaStream_t stream{};
    CUDA_OK(cudaStreamCreate(&stream));
    EXPECT_EQ(b::launchBatchedCopy(nullptr, nullptr, nullptr, 0, stream), cudaSuccess);
    CUDA_OK(cudaStreamSynchronize(stream));
    cudaStreamDestroy(stream);
}

namespace
{
// Driver for the two opt-in copy backends: gather (src->slot) + scatter (slot->dst) over n buffers
// of mixed sizes, asserting byte-exact. `useCub` selects cub::DeviceMemcpy::Batched over the custom
// kernel; `mappedPlan` puts the [srcs|dsts|sizes] plan arrays in MAPPED host memory (the zero-copy-
// args path) instead of device memory. Same round trip as GatherThenScatterRoundTrip.
void runBackendRoundTrip(bool useCub, bool mappedPlan)
{
    int devs = 0;
    CUDA_OK(cudaGetDeviceCount(&devs));
    if (devs == 0)
    {
        GTEST_SKIP() << "no CUDA device";
    }
    std::vector<std::uint32_t> sizes{64, 100, 256, 16, 4096, 17, 1, 32, 3, 512};
    auto const n = static_cast<std::uint32_t>(sizes.size());
    std::vector<std::uint64_t> off(n);
    std::uint64_t cur = 0;
    for (std::uint32_t i = 0; i < n; ++i)
    {
        off[i] = cur;
        cur = alignUp(cur + sizes[i], 256);
    }
    std::uint64_t const total = cur;
    std::vector<unsigned char> srcHost(total, 0);
    for (std::uint32_t i = 0; i < n; ++i)
        for (std::uint32_t j = 0; j < sizes[i]; ++j)
            srcHost[off[i] + j] = pattern(i, j);

    void *dSrc = nullptr, *dSlot = nullptr, *dDst = nullptr;
    CUDA_OK(cudaMalloc(&dSrc, total));
    CUDA_OK(cudaMalloc(&dSlot, total));
    CUDA_OK(cudaMalloc(&dDst, total));
    CUDA_OK(cudaMemcpy(dSrc, srcHost.data(), total, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemset(dDst, 0, total));
    auto base = [](void* p, std::uint64_t o) { return reinterpret_cast<std::uint64_t>(static_cast<char*>(p) + o); };
    std::vector<std::uint64_t> gSrc(n), gDst(n), sSrc(n), sDst(n);
    for (std::uint32_t i = 0; i < n; ++i)
    {
        gSrc[i] = base(dSrc, off[i]);
        gDst[i] = base(dSlot, off[i]);
        sSrc[i] = base(dSlot, off[i]);
        sDst[i] = base(dDst, off[i]);
    }

    // Lay a plan array on device (cudaMalloc + H2D) or in mapped host (zero-copy alias).
    std::vector<void*> devBufs;     // device-path allocations, freed below
    std::vector<void*> mappedHosts; // mapped-path host allocations, freed below
    auto place = [&](void const* data, std::size_t bytes) -> void*
    {
        if (mappedPlan)
        {
            void* hp = nullptr;
            EXPECT_EQ(cudaHostAlloc(&hp, bytes, cudaHostAllocMapped), cudaSuccess);
            std::memcpy(hp, data, bytes);
            mappedHosts.push_back(hp);
            void* dp = nullptr;
            EXPECT_EQ(cudaHostGetDevicePointer(&dp, hp, 0), cudaSuccess); // device-accessible alias
            return dp;
        }
        void* d = nullptr;
        EXPECT_EQ(cudaMalloc(&d, bytes), cudaSuccess);
        EXPECT_EQ(cudaMemcpy(d, data, bytes, cudaMemcpyHostToDevice), cudaSuccess);
        devBufs.push_back(d);
        return d;
    };
    auto* dgSrc = static_cast<std::uint64_t*>(place(gSrc.data(), n * sizeof(std::uint64_t)));
    auto* dgDst = static_cast<std::uint64_t*>(place(gDst.data(), n * sizeof(std::uint64_t)));
    auto* dsSrc = static_cast<std::uint64_t*>(place(sSrc.data(), n * sizeof(std::uint64_t)));
    auto* dsDst = static_cast<std::uint64_t*>(place(sDst.data(), n * sizeof(std::uint64_t)));
    auto* dSizes = static_cast<std::uint32_t*>(place(sizes.data(), n * sizeof(std::uint32_t)));

    cudaStream_t stream{};
    CUDA_OK(cudaStreamCreate(&stream));
    void* dTemp = nullptr;
    std::size_t tmpBytes = 0;
    if (useCub)
    {
        CUDA_OK(b::batchedCopyCubTempBytes(n, tmpBytes));
        if (tmpBytes > 0)
        {
            CUDA_OK(cudaMalloc(&dTemp, tmpBytes));
        }
    }
    auto gather = [&]
    {
        return useCub ? b::launchBatchedCopyCub(dgSrc, dgDst, dSizes, n, stream, dTemp, tmpBytes)
                      : b::launchBatchedCopy(dgSrc, dgDst, dSizes, n, stream);
    };
    auto scatter = [&]
    {
        return useCub ? b::launchBatchedCopyCub(dsSrc, dsDst, dSizes, n, stream, dTemp, tmpBytes)
                      : b::launchBatchedCopy(dsSrc, dsDst, dSizes, n, stream);
    };
    CUDA_OK(gather());
    CUDA_OK(scatter());
    CUDA_OK(cudaStreamSynchronize(stream));

    std::vector<unsigned char> dstHost(total, 0xEE);
    CUDA_OK(cudaMemcpy(dstHost.data(), dDst, total, cudaMemcpyDeviceToHost));
    for (std::uint32_t i = 0; i < n; ++i)
        for (std::uint32_t j = 0; j < sizes[i]; ++j)
            ASSERT_EQ(dstHost[off[i] + j], pattern(i, j)) << "mismatch buf=" << i << " byte=" << j;

    cudaFree(dSrc);
    cudaFree(dSlot);
    cudaFree(dDst);
    for (void* p : devBufs)
        cudaFree(p);
    for (void* p : mappedHosts)
        cudaFreeHost(p);
    if (dTemp != nullptr)
        cudaFree(dTemp);
    cudaStreamDestroy(stream);
}
} // namespace

// cub::DeviceMemcpy::Batched backend (TRTLLM_NIXL_BOUNCE_CUB_COPY).
TEST(GatherScatterKernel, CubBatchedCopyRoundTrip)
{
    runBackendRoundTrip(/*useCub=*/true, /*mappedPlan=*/false);
}

// Zero-copy plan args: kernel reads [srcs|dsts|sizes] from mapped host (TRTLLM_NIXL_BOUNCE_ZEROCOPY_ARGS).
TEST(GatherScatterKernel, ZeroCopyPlanArgsRoundTrip)
{
    runBackendRoundTrip(/*useCub=*/false, /*mappedPlan=*/true);
}

// Both options together.
TEST(GatherScatterKernel, CubPlusZeroCopyPlanArgsRoundTrip)
{
    runBackendRoundTrip(/*useCub=*/true, /*mappedPlan=*/true);
}
