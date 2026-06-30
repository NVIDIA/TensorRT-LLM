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

// GPU test for BounceArena: the one shared bounce data buffer. Verifies allocation, the
// offset->address mapping the scheduler/transport rely on, and that the buffer is real device
// memory (writable/readable). Fabric is force-disabled so this runs on any CUDA device (CI/x86).

#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/BounceArena.h"

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

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

TEST(BounceArena, BaseAndOffsetMapping)
{
    if (!hasCuda())
        GTEST_SKIP();
    constexpr std::size_t kBytes = 4ULL << 20; // 4 MiB
    b::BounceArena arena(kBytes, 0, /*allowFabric=*/false);
    EXPECT_NE(arena.base(), nullptr);
    EXPECT_EQ(arena.bytes(), kBytes);
    EXPECT_EQ(arena.baseAddr(), reinterpret_cast<std::uint64_t>(arena.base()));
    EXPECT_FALSE(arena.isFabric()); // forced off
    // at(offset) is exactly base + offset (what Grant.addr == baseAddr + offset relies on).
    EXPECT_EQ(arena.at(0), arena.base());
    EXPECT_EQ(arena.at(1024), static_cast<char*>(arena.base()) + 1024);
    EXPECT_EQ(reinterpret_cast<std::uint64_t>(arena.at(4096)), arena.baseAddr() + 4096);
}

TEST(BounceArena, BufferIsUsableDeviceMemory)
{
    if (!hasCuda())
        GTEST_SKIP();
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    constexpr std::size_t kBytes = 1ULL << 20;
    b::BounceArena arena(kBytes, 0, /*allowFabric=*/false);

    // Write a pattern at a non-zero offset region and read it back through at().
    constexpr std::uint64_t kOffset = 64 * 1024;
    constexpr std::size_t kLen = 4096;
    std::vector<unsigned char> in(kLen);
    for (std::size_t i = 0; i < kLen; ++i)
    {
        in[i] = static_cast<unsigned char>((i * 31 + 7) & 0xFF);
    }
    ASSERT_EQ(cudaMemcpy(arena.at(kOffset), in.data(), kLen, cudaMemcpyHostToDevice), cudaSuccess);
    std::vector<unsigned char> out(kLen, 0);
    ASSERT_EQ(cudaMemcpy(out.data(), arena.at(kOffset), kLen, cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(in, out);
}
