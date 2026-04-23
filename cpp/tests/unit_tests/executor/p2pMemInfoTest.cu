/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/executor/cache_transmission/nixl_utils/p2pTransferAgent.h"
#include <gtest/gtest.h>

using namespace tensorrt_llm::executor::kv_cache;

namespace
{

P2pMemChunk makeChunk(uint64_t offset, uint64_t size, uint8_t handleByte)
{
    P2pMemChunk c{};
    c.virtAddrOffset = offset;
    c.size = size;
    for (size_t i = 0; i < sizeof(c.fabricHandle); ++i)
    {
        c.fabricHandle[i] = static_cast<uint8_t>(handleByte + i);
    }
    return c;
}

P2pMemPool makePool(int32_t deviceId, uint64_t base, uint64_t total, std::vector<P2pMemChunk> chunks)
{
    P2pMemPool p{};
    p.deviceId = deviceId;
    p.poolBaseAddr = base;
    p.poolTotalSize = total;
    p.registeredAddr = base + 0x1000;
    p.registeredSize = total - 0x2000;
    p.mappedOffset = 0;
    p.mappedSize = total;
    p.chunks = std::move(chunks);
    return p;
}

} // anonymous namespace

TEST(P2pMemInfoTest, RoundTripEmptyUnsupported)
{
    P2pMemInfo info;
    info.supported = false;
    info.handleType = VmmHandleType::kNone;

    auto blob = info.serialize();
    auto decoded = P2pMemInfo::deserialize(blob);
    ASSERT_TRUE(decoded.has_value());
    EXPECT_FALSE(decoded->supported);
    EXPECT_EQ(decoded->handleType, VmmHandleType::kNone);
    EXPECT_TRUE(decoded->udsPath.empty());
    EXPECT_TRUE(decoded->pools.empty());
}

TEST(P2pMemInfoTest, RoundTripFabricWithChunks)
{
    P2pMemInfo info;
    info.supported = true;
    info.handleType = VmmHandleType::kFabric;
    info.pools.push_back(
        makePool(0, 0x70000000ULL, 0x10000ULL, {makeChunk(0, 0x2000, 0x11), makeChunk(0x2000, 0x2000, 0x22)}));
    info.pools.push_back(makePool(1, 0x80000000ULL, 0x20000ULL, {makeChunk(0, 0x4000, 0x33)}));

    auto blob = info.serialize();
    auto decoded = P2pMemInfo::deserialize(blob);
    ASSERT_TRUE(decoded.has_value());
    EXPECT_TRUE(decoded->supported);
    EXPECT_EQ(decoded->handleType, VmmHandleType::kFabric);
    EXPECT_TRUE(decoded->udsPath.empty());
    ASSERT_EQ(decoded->pools.size(), 2u);

    EXPECT_EQ(decoded->pools[0].deviceId, 0);
    EXPECT_EQ(decoded->pools[0].poolBaseAddr, 0x70000000ULL);
    ASSERT_EQ(decoded->pools[0].chunks.size(), 2u);
    EXPECT_EQ(decoded->pools[0].chunks[0].virtAddrOffset, 0u);
    EXPECT_EQ(decoded->pools[0].chunks[0].size, 0x2000u);
    EXPECT_EQ(decoded->pools[0].chunks[0].fabricHandle[0], 0x11);
    EXPECT_EQ(decoded->pools[0].chunks[1].fabricHandle[0], 0x22);
    EXPECT_EQ(decoded->pools[1].chunks[0].fabricHandle[0], 0x33);
}

TEST(P2pMemInfoTest, RoundTripPosixFdWithUdsPath)
{
    P2pMemInfo info;
    info.supported = true;
    info.handleType = VmmHandleType::kPosixFd;
    info.udsPath = "/tmp/trt_llm_p2p_fd_12345_67890.sock";
    info.pools.push_back(makePool(0, 0x100000ULL, 0x4000ULL, {makeChunk(0, 0x2000, 0xAB)}));

    auto blob = info.serialize();
    auto decoded = P2pMemInfo::deserialize(blob);
    ASSERT_TRUE(decoded.has_value());
    EXPECT_EQ(decoded->handleType, VmmHandleType::kPosixFd);
    EXPECT_EQ(decoded->udsPath, "/tmp/trt_llm_p2p_fd_12345_67890.sock");
    ASSERT_EQ(decoded->pools.size(), 1u);
    EXPECT_EQ(decoded->pools[0].chunks[0].fabricHandle[0], 0xAB);
}

TEST(P2pMemInfoTest, DeserializeRejectsBadMagic)
{
    // 4 bytes of junk magic — deserialize should return nullopt.
    std::string bad(128, '\0');
    bad[0] = 'X';
    bad[1] = 'X';
    bad[2] = 'X';
    bad[3] = 'X';
    auto decoded = P2pMemInfo::deserialize(bad);
    EXPECT_FALSE(decoded.has_value());
}

TEST(P2pMemInfoTest, DeserializeRejectsTooShort)
{
    // Anything shorter than the minimum fixed header cannot encode a valid P2pMemInfo.
    std::string bad(4, '\0');
    auto decoded = P2pMemInfo::deserialize(bad);
    EXPECT_FALSE(decoded.has_value());
}
