/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/executor/transferAgent.h"

#include <gtest/gtest.h>

using namespace tensorrt_llm::executor::kv_cache;

// ── AgentDesc serialization tests (no backend needed) ──

TEST(AgentDescTest, SerializeDeserializeEmpty)
{
    std::string nixlBlob = "some_nixl_metadata_blob";
    AgentDesc original{nixlBlob};
    auto serialized = original.serialize();

    auto decoded = AgentDesc::deserialize(serialized);
    EXPECT_EQ(decoded.getBackendAgentDesc(), nixlBlob);
    EXPECT_TRUE(decoded.getVramRegions().empty());
}

TEST(AgentDescTest, SerializeDeserializeWithRegions)
{
    std::string nixlBlob = "nixl_blob_data_here";
    std::vector<VramRegionMeta> regions{
        {0x7f0000000000UL, 67108864, 2097152},   // 64MB pool, 2MB chunks
        {0x7f0004000000UL, 134217728, 33554432}, // 128MB pool, 32MB chunks
    };
    AgentDesc original{nixlBlob, regions};
    auto serialized = original.serialize();

    auto decoded = AgentDesc::deserialize(serialized);
    EXPECT_EQ(decoded.getBackendAgentDesc(), nixlBlob);
    ASSERT_EQ(decoded.getVramRegions().size(), 2);
    EXPECT_EQ(decoded.getVramRegions()[0].baseAddr, 0x7f0000000000UL);
    EXPECT_EQ(decoded.getVramRegions()[0].totalLen, 67108864);
    EXPECT_EQ(decoded.getVramRegions()[0].chunkSize, 2097152);
    EXPECT_EQ(decoded.getVramRegions()[1].baseAddr, 0x7f0004000000UL);
    EXPECT_EQ(decoded.getVramRegions()[1].totalLen, 134217728);
    EXPECT_EQ(decoded.getVramRegions()[1].chunkSize, 33554432);
}

TEST(AgentDescTest, SerializeDeserializeBinaryBlob)
{
    // NIXL blob can contain arbitrary binary data including null bytes
    std::string nixlBlob(256, '\0');
    for (size_t i = 0; i < nixlBlob.size(); ++i)
    {
        nixlBlob[i] = static_cast<char>(i);
    }
    std::vector<VramRegionMeta> regions{{0xdeadbeefUL, 4096, 4096}};
    AgentDesc original{nixlBlob, regions};
    auto serialized = original.serialize();

    auto decoded = AgentDesc::deserialize(serialized);
    EXPECT_EQ(decoded.getBackendAgentDesc(), nixlBlob);
    ASSERT_EQ(decoded.getVramRegions().size(), 1);
    EXPECT_EQ(decoded.getVramRegions()[0].baseAddr, 0xdeadbeefUL);
}

TEST(AgentDescTest, SerializeDeserializeUnalignedBase)
{
    // Simulate an unaligned VMM pool base (what happens with cuMemAddressReserve alignment=0)
    uintptr_t unalignedBase = 0x7f0000200000UL; // 2MB aligned but not 32MB aligned
    size_t chunkSize = 33554432;                // 32MB
    size_t totalLen = chunkSize * 4;            // 128MB pool

    std::vector<VramRegionMeta> regions{{unalignedBase, totalLen, chunkSize}};
    AgentDesc original{"blob", regions};
    auto serialized = original.serialize();

    auto decoded = AgentDesc::deserialize(serialized);
    ASSERT_EQ(decoded.getVramRegions().size(), 1);
    auto const& r = decoded.getVramRegions()[0];
    EXPECT_EQ(r.baseAddr, unalignedBase);
    EXPECT_EQ(r.totalLen, totalLen);
    EXPECT_EQ(r.chunkSize, chunkSize);

    // Verify chunk boundary calculation: (addr - base) % chunkSize
    uintptr_t addr = unalignedBase + chunkSize + 100;
    size_t offsetInChunk = (addr - r.baseAddr) % r.chunkSize;
    EXPECT_EQ(offsetInChunk, 100);

    // addr % chunkSize would give the WRONG answer for unaligned base
    size_t wrongOffset = addr % chunkSize;
    // wrongOffset != 100 in general for unaligned base
    // (only equal if base happens to be chunk-aligned)
    EXPECT_NE(wrongOffset, offsetInChunk);
}

TEST(AgentDescTest, DeserializeTruncatedData)
{
    // Serialize a valid AgentDesc, then truncate the data
    std::string nixlBlob = "nixl_blob_data";
    std::vector<VramRegionMeta> regions{{0x1000UL, 4096, 4096}};
    AgentDesc original{nixlBlob, regions};
    auto serialized = original.serialize();

    // Truncate to half the data
    std::string truncated = serialized.substr(0, serialized.size() / 2);
    EXPECT_THROW(AgentDesc::deserialize(truncated), std::exception);
}

// ── VmmDescSplitter tests (backend-agnostic, no NIXL dependency) ──

TEST(VmmDescSplitterTest, LookupChunkInfoHit)
{
    VramRegionMap regionMap;
    regionMap[0x100000] = {0x300000, 0x100000}; // base=1MB, len=3MB, chunk=1MB
    auto [chunkSize, base] = VmmDescSplitter::lookupChunkInfo(0x200000, regionMap);
    EXPECT_EQ(chunkSize, 0x100000);
    EXPECT_EQ(base, 0x100000);
}

TEST(VmmDescSplitterTest, LookupChunkInfoMiss)
{
    VramRegionMap regionMap;
    regionMap[0x100000] = {0x100000, 0x100000};
    // Address outside the region
    auto [chunkSize, base] = VmmDescSplitter::lookupChunkInfo(0x300000, regionMap);
    EXPECT_EQ(chunkSize, 0u);
    EXPECT_EQ(base, 0u);
}

TEST(VmmDescSplitterTest, LookupChunkInfoMultipleRegions)
{
    VramRegionMap regionMap;
    regionMap[0x100000] = {0x200000, 0x100000}; // region1: [0x100000, 0x300000)
    regionMap[0x400000] = {0x200000, 0x80000};  // region2: [0x400000, 0x600000)

    auto [cs1, b1] = VmmDescSplitter::lookupChunkInfo(0x150000, regionMap);
    EXPECT_EQ(cs1, 0x100000);
    EXPECT_EQ(b1, 0x100000);

    auto [cs2, b2] = VmmDescSplitter::lookupChunkInfo(0x500000, regionMap);
    EXPECT_EQ(cs2, 0x80000);
    EXPECT_EQ(b2, 0x400000);

    // Gap between regions
    auto [cs3, b3] = VmmDescSplitter::lookupChunkInfo(0x350000, regionMap);
    EXPECT_EQ(cs3, 0u);
}

TEST(VmmDescSplitterTest, SplitDescsAlignedBase)
{
    // Pool base is chunk-aligned: base=0x200000, chunkSize=0x100000 (1MB)
    VramRegionMap regionMap;
    regionMap[0x200000] = {0x400000, 0x100000}; // 4MB pool, 1MB chunks

    // A 2.5MB descriptor spanning 3 chunks
    std::vector<MemoryDesc> descs{{0x200000, 0x280000, 0}};
    MemoryDescs input{MemoryType::kVRAM, descs};

    auto result = VmmDescSplitter::splitDescsWithRegionMap(input, regionMap);
    auto const& out = result.getDescs();

    ASSERT_EQ(out.size(), 3);
    EXPECT_EQ(out[0].getAddr(), 0x200000);
    EXPECT_EQ(out[0].getLen(), 0x100000); // first full chunk
    EXPECT_EQ(out[1].getAddr(), 0x300000);
    EXPECT_EQ(out[1].getLen(), 0x100000); // second full chunk
    EXPECT_EQ(out[2].getAddr(), 0x400000);
    EXPECT_EQ(out[2].getLen(), 0x80000);  // remaining 0.5MB
}

TEST(VmmDescSplitterTest, SplitDescsUnalignedBase)
{
    // Pool base is NOT chunk-aligned: base=0x200000 (2MB), chunkSize=0x2000000 (32MB)
    uintptr_t base = 0x200000;
    size_t chunkSize = 0x2000000;
    VramRegionMap regionMap;
    regionMap[base] = {chunkSize * 3, chunkSize};

    // Descriptor starting at base, spanning 2.5 chunks
    size_t descLen = chunkSize * 2 + chunkSize / 2;
    std::vector<MemoryDesc> descs{{base, descLen, 0}};
    MemoryDescs input{MemoryType::kVRAM, descs};

    auto result = VmmDescSplitter::splitDescsWithRegionMap(input, regionMap);
    auto const& out = result.getDescs();

    ASSERT_EQ(out.size(), 3);
    EXPECT_EQ(out[0].getAddr(), base);
    EXPECT_EQ(out[0].getLen(), chunkSize);
    EXPECT_EQ(out[1].getAddr(), base + chunkSize);
    EXPECT_EQ(out[1].getLen(), chunkSize);
    EXPECT_EQ(out[2].getAddr(), base + chunkSize * 2);
    EXPECT_EQ(out[2].getLen(), chunkSize / 2);
}

TEST(VmmDescSplitterTest, SplitDescsNonVmm)
{
    VramRegionMap regionMap;
    // Address not in any region → no split
    std::vector<MemoryDesc> descs{{0x1000, 4096, 0}};
    MemoryDescs input{MemoryType::kVRAM, descs};

    auto result = VmmDescSplitter::splitDescsWithRegionMap(input, regionMap);
    ASSERT_EQ(result.getDescs().size(), 1);
    EXPECT_EQ(result.getDescs()[0].getLen(), 4096);
}

TEST(VmmDescSplitterTest, SplitDescsNonVramPassthrough)
{
    VramRegionMap regionMap;
    regionMap[0x100000] = {0x200000, 0x100000};
    // DRAM descs should pass through unchanged regardless of region map
    std::vector<MemoryDesc> descs{{0x100000, 0x200000, 0}};
    MemoryDescs input{MemoryType::kDRAM, descs};

    auto result = VmmDescSplitter::splitDescsWithRegionMap(input, regionMap);
    ASSERT_EQ(result.getDescs().size(), 1);
    EXPECT_EQ(result.getDescs()[0].getLen(), 0x200000);
}

TEST(VmmDescSplitterTest, SplitTransferDescsDifferentChunkSizes)
{
    // src: 1MB chunks, dst: 512KB chunks
    VramRegionMap localMap, remoteMap;
    localMap[0x100000] = {0x400000, 0x100000}; // 4MB, 1MB chunks
    remoteMap[0x800000] = {0x400000, 0x80000}; // 4MB, 512KB chunks

    // Transfer 2MB from src to dst
    std::vector<MemoryDesc> srcDescs{{0x100000, 0x200000, 0}};
    std::vector<MemoryDesc> dstDescs{{0x800000, 0x200000, 0}};
    MemoryDescs srcInput{MemoryType::kVRAM, srcDescs};
    MemoryDescs dstInput{MemoryType::kVRAM, dstDescs};

    auto [splitSrc, splitDst] = VmmDescSplitter::splitAndCoalesceTransferDescs(srcInput, dstInput, localMap, remoteMap);

    // dst has smaller chunks (512KB), so we get 4 pieces: 512K, 512K, 512K, 512K
    ASSERT_EQ(splitSrc.getDescs().size(), 4);
    ASSERT_EQ(splitDst.getDescs().size(), 4);
    for (size_t i = 0; i < 4; ++i)
    {
        EXPECT_EQ(splitSrc.getDescs()[i].getLen(), 0x80000);
        EXPECT_EQ(splitDst.getDescs()[i].getLen(), 0x80000);
        EXPECT_EQ(splitSrc.getDescs()[i].getAddr(), 0x100000 + i * 0x80000);
        EXPECT_EQ(splitDst.getDescs()[i].getAddr(), 0x800000 + i * 0x80000);
    }
}

TEST(VmmDescSplitterTest, SplitTransferDescsUnalignedBothSides)
{
    // Both src and dst have unaligned bases with different chunk sizes
    uintptr_t srcBase = 0x200000;  // 2MB (not 4MB aligned)
    uintptr_t dstBase = 0x1800000; // 24MB (not 32MB aligned)
    size_t srcChunk = 0x400000;    // 4MB
    size_t dstChunk = 0x200000;    // 2MB

    VramRegionMap localMap, remoteMap;
    localMap[srcBase] = {srcChunk * 4, srcChunk};
    remoteMap[dstBase] = {dstChunk * 8, dstChunk};

    // Transfer exactly 4MB
    std::vector<MemoryDesc> srcDescs{{srcBase, 0x400000, 0}};
    std::vector<MemoryDesc> dstDescs{{dstBase, 0x400000, 0}};
    MemoryDescs srcInput{MemoryType::kVRAM, srcDescs};
    MemoryDescs dstInput{MemoryType::kVRAM, dstDescs};

    auto [splitSrc, splitDst] = VmmDescSplitter::splitAndCoalesceTransferDescs(srcInput, dstInput, localMap, remoteMap);

    // src has 4MB chunks starting at srcBase → 1 piece from src side
    // dst has 2MB chunks starting at dstBase → 2 pieces from dst side
    // min produces 2 pieces: 2MB, 2MB
    ASSERT_EQ(splitSrc.getDescs().size(), 2);
    ASSERT_EQ(splitDst.getDescs().size(), 2);
    EXPECT_EQ(splitSrc.getDescs()[0].getLen(), 0x200000);
    EXPECT_EQ(splitSrc.getDescs()[1].getLen(), 0x200000);
}

TEST(VmmDescSplitterTest, SplitTransferDescsNoDstRegion)
{
    // src has VMM chunks, dst has no VMM info (empty remote map) → only src boundaries
    VramRegionMap localMap;
    VramRegionMap emptyRemoteMap;
    localMap[0x100000] = {0x400000, 0x100000}; // 4MB, 1MB chunks

    std::vector<MemoryDesc> srcDescs{{0x100000, 0x200000, 0}};
    std::vector<MemoryDesc> dstDescs{{0x900000, 0x200000, 0}};
    MemoryDescs srcInput{MemoryType::kVRAM, srcDescs};
    MemoryDescs dstInput{MemoryType::kVRAM, dstDescs};

    auto [splitSrc, splitDst]
        = VmmDescSplitter::splitAndCoalesceTransferDescs(srcInput, dstInput, localMap, emptyRemoteMap);

    // Only src boundaries: 2 pieces of 1MB each
    ASSERT_EQ(splitSrc.getDescs().size(), 2);
    ASSERT_EQ(splitDst.getDescs().size(), 2);
    EXPECT_EQ(splitSrc.getDescs()[0].getLen(), 0x100000);
    EXPECT_EQ(splitSrc.getDescs()[1].getLen(), 0x100000);
    EXPECT_EQ(splitDst.getDescs()[0].getLen(), 0x100000);
    EXPECT_EQ(splitDst.getDescs()[1].getLen(), 0x100000);
}
