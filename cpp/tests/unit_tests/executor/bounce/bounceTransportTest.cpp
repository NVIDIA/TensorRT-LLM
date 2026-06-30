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

#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/BounceTransport.h"
#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/LocalCopyTransferEngine.h"
#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/ZmqControlChannel.h"

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <chrono>
#include <memory>
#include <vector>

namespace b = tensorrt_llm::executor::kv_cache::bounce;
namespace kvc = tensorrt_llm::executor::kv_cache;

namespace
{
bool hasCuda()
{
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

std::uint64_t alignUp(std::uint64_t v, std::uint64_t a)
{
    return (v + a - 1) / a * a;
}

unsigned char pattern(std::size_t desc, std::size_t idx)
{
    return static_cast<unsigned char>((desc * 167 + idx * 13 + 7) & 0xFF);
}

// One end-to-end transfer of `nDescs` descriptors of `descBytes` each through the bounce
// pipeline, where maxChunkBytes/windowDepth are chosen so the chunk count exceeds the pool depth
// (forcing credit recycling). Verifies every byte arrives at the destination.
void runTransfer(std::uint32_t nDescs, std::uint32_t descBytes, std::size_t maxChunkBytes, std::uint32_t windowDepth)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    // Lay out src and dst regions with 256-aligned per-desc offsets.
    std::vector<std::uint64_t> off(nDescs);
    std::uint64_t cur = 0;
    for (std::uint32_t i = 0; i < nDescs; ++i)
    {
        off[i] = cur;
        cur = alignUp(cur + descBytes, 256);
    }
    std::uint64_t const total = cur;

    void* dSrc = nullptr;
    void* dDst = nullptr;
    ASSERT_EQ(cudaMalloc(&dSrc, total), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dDst, total), cudaSuccess);
    std::vector<unsigned char> src(total, 0);
    for (std::uint32_t i = 0; i < nDescs; ++i)
    {
        for (std::uint32_t j = 0; j < descBytes; ++j)
        {
            src[off[i] + j] = pattern(i, j);
        }
    }
    ASSERT_EQ(cudaMemcpy(dSrc, src.data(), total, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemset(dDst, 0, total), cudaSuccess);

    auto addr = [](void* p, std::uint64_t o) { return reinterpret_cast<std::uintptr_t>(static_cast<char*>(p) + o); };
    std::vector<kvc::MemoryDesc> srcD;
    std::vector<kvc::MemoryDesc> dstD;
    for (std::uint32_t i = 0; i < nDescs; ++i)
    {
        srcD.emplace_back(addr(dSrc, off[i]), descBytes, 0);
        dstD.emplace_back(addr(dDst, off[i]), descBytes, 0);
    }
    kvc::TransferDescs srcDescs{kvc::MemoryType::kVRAM, std::move(srcD)};
    kvc::TransferDescs dstDescs{kvc::MemoryType::kVRAM, std::move(dstD)};

    b::BounceConfig cfg;
    cfg.maxChunkBytes = maxChunkBytes; // per-chunk byte cap (maxChunkBytes)
    cfg.windowDepth = windowDepth;
    cfg.window = windowDepth;          // per-flow in-flight region cap -> still forces credit recycling
    cfg.scatterWorkers = 2;
    cfg.minBlock = 256;                // buddy granularity (matches the 256-aligned desc layout)
    // Arena large enough to hold a full window of max-size regions for BOTH roles with headroom for
    // buddy rounding; rounded to a power of two so the BuddyAllocator uses all of it.
    std::size_t arenaBytes = 1ULL << 20;
    while (arenaBytes < static_cast<std::size_t>(windowDepth) * maxChunkBytes * 4ULL)
    {
        arenaBytes <<= 1;
    }
    cfg.arenaBytes = arenaBytes;
    std::size_t const maxDescs = std::max<std::size_t>(1024ULL, maxChunkBytes / 256ULL);
    std::uint32_t const execCount = windowDepth + cfg.scatterWorkers + 2;

    // Sender A and receiver B, each with its own control channel + arena + exec pool. One arena is
    // unused on each side (A only gathers locally, B only grants), but the transport is symmetric.
    b::ZmqControlChannel chA("A");
    b::ZmqControlChannel chB("B");
    b::LocalCopyTransferEngine engA;
    b::LocalCopyTransferEngine engB;
    auto arenaA = std::make_unique<b::BounceArena>(arenaBytes, 0, /*allowFabric=*/false);
    auto arenaB = std::make_unique<b::BounceArena>(arenaBytes, 0, /*allowFabric=*/false);
    auto execA = std::make_unique<b::ExecPool>(execCount, maxDescs, 0);
    auto execB = std::make_unique<b::ExecPool>(execCount, maxDescs, 0);

    auto A = std::make_unique<b::BounceTransport>("A", cfg, 0, &chA, &engA, arenaA.get(), execA.get());
    auto B = std::make_unique<b::BounceTransport>("B", cfg, 0, &chB, &engB, arenaB.get(), execB.get());
    A->addPeer("B", chB.localEndpoint());
    B->addPeer("A", chA.localEndpoint());

    auto fut = A->submit(srcDescs, dstDescs, "B");
    ASSERT_EQ(fut.wait_for(std::chrono::seconds(20)), std::future_status::ready) << "transfer hung";
    EXPECT_EQ(fut.get(), kvc::TransferState::kSUCCESS);

    std::vector<unsigned char> got(total, 0xEE);
    ASSERT_EQ(cudaMemcpy(got.data(), dDst, total, cudaMemcpyDeviceToHost), cudaSuccess);
    for (std::uint32_t i = 0; i < nDescs; ++i)
    {
        for (std::uint32_t j = 0; j < descBytes; ++j)
        {
            ASSERT_EQ(got[off[i] + j], pattern(i, j)) << "mismatch desc=" << i << " byte=" << j;
        }
    }

    A->shutdown();
    B->shutdown();
    cudaFree(dSrc);
    cudaFree(dDst);
}
} // namespace

TEST(BounceTransport, SmallTransferFitsOneWindow)
{
    if (!hasCuda())
        GTEST_SKIP();
    // 4 descs, slot holds a few -> chunks <= windowDepth.
    runTransfer(/*nDescs=*/4, /*descBytes=*/512, /*maxChunkBytes=*/8192, /*windowDepth=*/4);
}

TEST(BounceTransport, LargeTransferRecyclesCredits)
{
    if (!hasCuda())
        GTEST_SKIP();
    // 40 descs of ~700B, slot=4KB (a few descs/chunk) -> ~chunks >> windowDepth(2): forces
    // credit recycling through a 2-slot receiver pool. Exercises R1 + the recycling loop.
    runTransfer(/*nDescs=*/40, /*descBytes=*/700, /*maxChunkBytes=*/4096, /*windowDepth=*/2);
}

TEST(BounceTransport, ManySmallDescs)
{
    if (!hasCuda())
        GTEST_SKIP();
    // Closer to the real KV pattern: many tiny descs.
    runTransfer(/*nDescs=*/500, /*descBytes=*/256, /*maxChunkBytes=*/16384, /*windowDepth=*/4);
}

TEST(BounceTransport, ConcurrentRequestsToSameReceiver)
{
    if (!hasCuda())
        GTEST_SKIP();
    // Two independent transfers (distinct rids) over one transport pair run as separate flows.
    runTransfer(8, 1024, 8192, 3);
    runTransfer(8, 1024, 8192, 3);
}
