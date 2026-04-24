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

#include "tensorrt_llm/common/tllmException.h"
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

// ============================================================================
// P2pRemoteMappingRegistry::translate — address lookup for per-segment bucketing
// ============================================================================
//
// These tests drive translate() directly against hand-built RemoteP2pMapping values.
// No CUDA / NIXL setup is needed because translate is pure arithmetic over the mapping's
// recorded address ranges. This is specifically the decision point used by the mixed
// P2P + NIXL routing path in NixlTransferAgent::submitTransferRequests: segments for
// which translate returns a non-null local ptr go P2P; nullptr goes NIXL.

namespace
{
// Build one RemoteP2pPoolMapping. Layout:
//   remoteBase ... remoteBase+registeredOffset ... +registeredSize ... +totalSize
//                  ^ remoteRegisteredAddr
//   localVirtAddr points to the start of the locally-reserved VA matching
//   remoteBase + remoteMappedOffset. For these tests remoteMappedOffset == registeredOffset
//   i.e. the mapped region starts exactly at the registered region (typical import layout).
RemoteP2pPoolMapping makePoolMapping(
    uint64_t remoteBase, uint64_t totalSize, uint64_t registeredOffset, uint64_t registeredSize, uintptr_t localVa)
{
    RemoteP2pPoolMapping m{};
    m.remoteBaseAddr = remoteBase;
    m.totalSize = totalSize;
    m.remoteRegisteredAddr = remoteBase + registeredOffset;
    m.registeredSize = registeredSize;
    m.remoteMappedOffset = registeredOffset; // mapped region aligned with registered
    m.mappedSize = registeredSize;
    m.localVirtAddr = static_cast<CUdeviceptr>(localVa);
    return m;
}
} // namespace

TEST(P2pTranslateTest, MappedAddressReturnsLocalPtr)
{
    RemoteP2pMapping mapping;
    mapping.pools.push_back(makePoolMapping(
        /*remoteBase=*/0x1000'0000, /*totalSize=*/0x10000,
        /*registeredOffset=*/0x1000, /*registeredSize=*/0xE000, /*localVa=*/0xAAAA'0000));

    // Segment at start of registered range -> localVa + 0.
    void* p = P2pRemoteMappingRegistry::translate(mapping, 0x1000'1000, 0x100);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(p), 0xAAAA'0000u);

    // Segment in the middle -> localVa + (offsetFromPoolBase - mappedOffset).
    p = P2pRemoteMappingRegistry::translate(mapping, 0x1000'5000, 0x100);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(p), 0xAAAA'4000u);
}

TEST(P2pTranslateTest, UnmappedAddressReturnsNull)
{
    RemoteP2pMapping mapping;
    mapping.pools.push_back(makePoolMapping(
        /*remoteBase=*/0x1000'0000, /*totalSize=*/0x10000,
        /*registeredOffset=*/0x1000, /*registeredSize=*/0xE000, /*localVa=*/0xAAAA'0000));

    // Below the pool range entirely.
    EXPECT_EQ(P2pRemoteMappingRegistry::translate(mapping, 0x0F00'0000, 0x100), nullptr);
    // Above the pool range entirely.
    EXPECT_EQ(P2pRemoteMappingRegistry::translate(mapping, 0x2000'0000, 0x100), nullptr);
}

TEST(P2pTranslateTest, AddrInPoolButBeforeRegisteredThrows)
{
    // Addr falls inside the pool's totalSize but BEFORE the registered region.
    // translate treats this as a caller bug and throws (not fallback-able to NIXL).
    RemoteP2pMapping mapping;
    mapping.pools.push_back(
        makePoolMapping(0x1000'0000, 0x10000, /*registeredOffset=*/0x1000, /*registeredSize=*/0xE000, 0xAAAA'0000));
    EXPECT_THROW(
        (void) P2pRemoteMappingRegistry::translate(mapping, 0x1000'0500, 0x100), tensorrt_llm::common::TllmException);
}

TEST(P2pTranslateTest, TransferRangeExceedsRegisteredThrows)
{
    // Addr is inside registered, but addr+size crosses the registered end.
    RemoteP2pMapping mapping;
    mapping.pools.push_back(makePoolMapping(0x1000'0000, 0x10000, 0x1000, 0xE000, 0xAAAA'0000));
    uint64_t registeredEnd = 0x1000'0000 + 0x1000 + 0xE000; // 0x1000F000
    uint64_t addr = registeredEnd - 0x80;                   // 128B before end
    EXPECT_THROW((void) P2pRemoteMappingRegistry::translate(mapping, addr, 0x200), tensorrt_llm::common::TllmException);
}

// THE mixed-scenario test: mapping has pool 0 (imported), but the second pool failed to
// import and was skipped -> not present in mapping.pools. Addresses in pool 0's registered
// range translate; addresses that would have been in pool 1 return nullptr (because pool 1
// doesn't exist from translate's perspective). This is exactly the state that drives
// NixlTransferAgent::submitTransferRequests to route some segments via P2P and others via
// NIXL.
TEST(P2pTranslateTest, PartialMappingMixedSegments)
{
    RemoteP2pMapping mapping;
    // Pool 0: successfully imported at remote [0x1000'0000, 0x1001'0000).
    mapping.pools.push_back(makePoolMapping(/*remoteBase=*/0x1000'0000, /*totalSize=*/0x10000,
        /*registeredOffset=*/0x1000, /*registeredSize=*/0xE000, /*localVa=*/0xAAAA'0000));

    // Pool 1 would have lived at remote [0x2000'0000, 0x2001'0000) but import failed,
    // so it's NOT in mapping.pools.

    struct Segment
    {
        uint64_t remoteAddr;
        size_t size;
        bool expectMapped;
        uintptr_t expectLocalAddr; // only valid when expectMapped
    };

    // Use 32B transfers so nothing crosses registered boundaries.
    Segment segs[] = {
        {0x1000'1000, 0x20, true, 0xAAAA'0000}, // pool 0 start
        {0x1000'2000, 0x20, true, 0xAAAA'1000}, // pool 0 middle
        {0x1000'E000, 0x20, true, 0xAAAA'D000}, // pool 0 near end
        {0x2000'1000, 0x20, false, 0},          // pool 1 (failed) — would translate if imported
        {0x2000'5000, 0x20, false, 0},          // pool 1 (failed)
        {0x3000'0000, 0x20, false, 0},          // completely unknown pool
    };

    int mapped = 0, unmapped = 0;
    for (auto const& s : segs)
    {
        void* p = P2pRemoteMappingRegistry::translate(mapping, s.remoteAddr, s.size);
        if (s.expectMapped)
        {
            ASSERT_NE(p, nullptr) << "remoteAddr=0x" << std::hex << s.remoteAddr;
            EXPECT_EQ(reinterpret_cast<uintptr_t>(p), s.expectLocalAddr) << std::hex << s.remoteAddr;
            ++mapped;
        }
        else
        {
            EXPECT_EQ(p, nullptr) << "remoteAddr=0x" << std::hex << s.remoteAddr;
            ++unmapped;
        }
    }

    EXPECT_EQ(mapped, 3);
    EXPECT_EQ(unmapped, 3);
}

// Multi-pool mapping where both pools were imported successfully — addresses in either
// translate to the correct local VA. Safety check that translate walks all pools, not
// just the first one.
TEST(P2pTranslateTest, MultiPoolAllMapped)
{
    RemoteP2pMapping mapping;
    mapping.pools.push_back(makePoolMapping(0x1000'0000, 0x10000, 0x1000, 0xE000, 0xAAAA'0000));
    mapping.pools.push_back(makePoolMapping(0x2000'0000, 0x20000, 0x0, 0x20000, 0xBBBB'0000));

    void* p0 = P2pRemoteMappingRegistry::translate(mapping, 0x1000'2000, 0x20);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(p0), 0xAAAA'1000u);

    void* p1 = P2pRemoteMappingRegistry::translate(mapping, 0x2001'0000, 0x20);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(p1), 0xBBBC'0000u); // 0xBBBB'0000 + 0x1'0000
}
