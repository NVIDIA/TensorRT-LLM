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

#include "bounceTestUtils.h"

#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/GatherScatterKernel.h"

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

namespace b = tensorrt_llm::executor::kv_cache::bounce;

#define CUDA_OK(call) ASSERT_EQ((call), cudaSuccess) << "CUDA call failed: " #call

using bounce_test::alignUp;
using bounce_test::hasCuda;

namespace
{
// Distinct per-(buffer,byte) pattern so any mis-routing/overlap is caught.
unsigned char pattern(std::size_t buf, std::size_t idx)
{
    return static_cast<unsigned char>((buf * 131 + idx * 7 + 13) & 0xFF);
}
} // namespace

TEST(GatherScatterKernel, ZeroBuffersIsNoop)
{
    if (!hasCuda())
    {
        GTEST_SKIP() << "no CUDA device";
    }
    cudaStream_t stream{};
    CUDA_OK(cudaStreamCreate(&stream));
    EXPECT_EQ(b::launchBatchedCopy(nullptr, nullptr, nullptr, 0, stream), cudaSuccess);
    CUDA_OK(cudaStreamSynchronize(stream));
    cudaStreamDestroy(stream);
}

namespace
{
// Round-trip driver shared by both plan-placement tests: gather scattered source buffers into one
// packed arena region, then scatter that region into fresh dst buffers, asserting byte-exact.
// Mixes 16B-aligned sizes (uint4 path) and unaligned sizes (byte path). `mappedPlan` puts the
// [srcs|dsts|sizes] plan arrays in MAPPED host memory (the zero-copy-args path) instead of device
// memory.
void runRoundTrip(bool mappedPlan)
{
    if (!hasCuda())
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

    void *dSrc = nullptr, *dPackedRegion = nullptr, *dDst = nullptr;
    CUDA_OK(cudaMalloc(&dSrc, total));
    CUDA_OK(cudaMalloc(&dPackedRegion, total));
    CUDA_OK(cudaMalloc(&dDst, total));
    CUDA_OK(cudaMemcpy(dSrc, srcHost.data(), total, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemset(dDst, 0, total));
    auto base = [](void* p, std::uint64_t o) { return reinterpret_cast<std::uint64_t>(static_cast<char*>(p) + o); };
    std::vector<std::uint64_t> gSrc(n), gDst(n), sSrc(n), sDst(n);
    for (std::uint32_t i = 0; i < n; ++i)
    {
        gSrc[i] = base(dSrc, off[i]);
        gDst[i] = base(dPackedRegion, off[i]);
        sSrc[i] = base(dPackedRegion, off[i]);
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
    CUDA_OK(b::launchBatchedCopy(dgSrc, dgDst, dSizes, n, stream)); // gather
    CUDA_OK(b::launchBatchedCopy(dsSrc, dsDst, dSizes, n, stream)); // scatter
    CUDA_OK(cudaStreamSynchronize(stream));

    std::vector<unsigned char> dstHost(total, 0xEE);
    CUDA_OK(cudaMemcpy(dstHost.data(), dDst, total, cudaMemcpyDeviceToHost));
    for (std::uint32_t i = 0; i < n; ++i)
        for (std::uint32_t j = 0; j < sizes[i]; ++j)
            ASSERT_EQ(dstHost[off[i] + j], pattern(i, j)) << "mismatch buf=" << i << " byte=" << j;

    cudaFree(dSrc);
    cudaFree(dPackedRegion);
    cudaFree(dDst);
    for (void* p : devBufs)
        cudaFree(p);
    for (void* p : mappedHosts)
        cudaFreeHost(p);
    cudaStreamDestroy(stream);
}
} // namespace

// Plan arrays staged in device memory (TRTLLM_NIXL_BOUNCE_USE_ZERO_COPY_ARGUMENTS=0).
TEST(GatherScatterKernel, GatherThenScatterRoundTrip)
{
    runRoundTrip(/*mappedPlan=*/false);
}

// Zero-copy plan args: kernel reads [srcs|dsts|sizes] from mapped host
// (TRTLLM_NIXL_BOUNCE_USE_ZERO_COPY_ARGUMENTS=1, the default).
TEST(GatherScatterKernel, ZeroCopyPlanArgsRoundTrip)
{
    runRoundTrip(/*mappedPlan=*/true);
}
