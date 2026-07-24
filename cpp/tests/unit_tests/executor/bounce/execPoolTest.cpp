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

#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/ExecPool.h"

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <cstring>
#include <set>
#include <vector>

namespace b = tensorrt_llm::executor::kv_cache::bounce;

namespace
{
bool hasCuda()
{
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}
} // namespace

TEST(ExecPool, AcquireReleaseExhaustion)
{
    if (!hasCuda())
        GTEST_SKIP();
    b::ExecPool pool(/*count=*/3, /*maxDescsPerChunk=*/64, /*dev=*/0);
    EXPECT_EQ(pool.size(), 3u);
    EXPECT_EQ(pool.freeCount(), 3u);

    std::vector<b::ExecCtx*> held;
    std::set<std::uint32_t> ids;
    std::set<void*> scratch;
    for (int i = 0; i < 3; ++i)
    {
        auto* c = pool.tryAcquire();
        ASSERT_NE(c, nullptr);
        EXPECT_NE(c->stream, nullptr);
        EXPECT_NE(c->event, nullptr);
        EXPECT_NE(c->scratch, nullptr);
        EXPECT_NE(c->hostPinned, nullptr);
        ids.insert(c->id);
        scratch.insert(c->scratch);
        held.push_back(c);
    }
    EXPECT_EQ(ids.size(), 3u);             // distinct contexts
    EXPECT_EQ(scratch.size(), 3u);         // distinct scratch buffers
    EXPECT_EQ(pool.freeCount(), 0u);
    EXPECT_EQ(pool.tryAcquire(), nullptr); // exhausted -> non-blocking nullptr

    pool.release(held.back());
    held.pop_back();
    EXPECT_EQ(pool.freeCount(), 1u);
    auto* again = pool.tryAcquire(); // the freed one is reusable
    ASSERT_NE(again, nullptr);
    pool.release(again);
    for (auto* c : held)
    {
        pool.release(c);
    }
    EXPECT_EQ(pool.freeCount(), 3u);
}

TEST(ExecPool, StreamScratchAreUsable)
{
    if (!hasCuda())
        GTEST_SKIP();
    b::ExecPool pool(1, 64, 0);
    auto* c = pool.tryAcquire();
    ASSERT_NE(c, nullptr);
    // Use the ctx's hostPinned -> scratch H2D on its stream, then record/poll its event.
    std::vector<unsigned char> in(128, 0xCD);
    ASSERT_LE(128u, c->scratchBytes);
    std::memcpy(c->hostPinned, in.data(), 128);
    ASSERT_EQ(cudaMemcpyAsync(c->scratch, c->hostPinned, 128, cudaMemcpyHostToDevice, c->stream), cudaSuccess);
    ASSERT_EQ(cudaEventRecord(c->event, c->stream), cudaSuccess);
    ASSERT_EQ(cudaEventSynchronize(c->event), cudaSuccess);
    std::vector<unsigned char> out(128, 0);
    ASSERT_EQ(cudaMemcpy(out.data(), c->scratch, 128, cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(in, out);
    pool.release(c);
}
