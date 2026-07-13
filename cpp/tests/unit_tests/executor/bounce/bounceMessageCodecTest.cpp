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

#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/BounceMessage.h"

#include <gtest/gtest.h>

#include <cstring>

namespace b = tensorrt_llm::executor::kv_cache::bounce;

namespace
{
b::BounceMsgHeader decodeOk(std::string const& blob, b::BounceMsgType expectType)
{
    EXPECT_TRUE(b::hasBounceMagic(blob));
    b::BounceMsgHeader h{};
    EXPECT_TRUE(b::decodeHeader(blob, h));
    EXPECT_EQ(h.msgType, static_cast<std::uint16_t>(expectType));
    return h;
}
} // namespace

TEST(BounceMessageCodec, WantCarriesChunkSizesAndEndpoint)
{
    std::vector<std::uint32_t> sizes{4096, 8192, 1024};
    std::string const ep = "tcp://10.0.0.3:5170";
    auto blob = b::encodeWant(/*rid=*/42, sizes, ep);
    auto h = decodeOk(blob, b::BounceMsgType::kWANT);
    EXPECT_EQ(h.requestId, 42u);
    EXPECT_EQ(h.count, 3u);
    EXPECT_EQ(h.aux, 3u); // numChunks
    std::vector<std::uint32_t> out;
    std::string outEp;
    ASSERT_TRUE(b::decodeWant(blob, h, out, outEp));
    EXPECT_EQ(out, sizes);
    EXPECT_EQ(outEp, ep);
}

TEST(BounceMessageCodec, CancelIsEmptyWantThatStillCarriesEndpoint)
{
    std::string const ep = "tcp://127.0.0.1:9999";
    auto blob = b::encodeCancel(42, ep); // same wire form as an empty-chunk WANT
    auto h = decodeOk(blob, b::BounceMsgType::kWANT);
    EXPECT_EQ(h.count, 0u);              // no chunks -> cancel
    EXPECT_EQ(h.aux, 0u);
    std::vector<std::uint32_t> out;
    std::string outEp;
    ASSERT_TRUE(b::decodeWant(blob, h, out, outEp));
    EXPECT_TRUE(out.empty());
    EXPECT_TRUE(b::isCancelWant(out)); // decoded WANT is recognized as a cancel
    EXPECT_EQ(outEp, ep);              // endpoint still travels so the receiver can bootstrap even on a bare cancel

    // A real (non-empty) WANT is NOT a cancel.
    std::vector<std::uint32_t> real;
    std::string ep2;
    auto wblob = b::encodeWant(43, std::vector<std::uint32_t>{4096}, ep);
    auto wh = decodeOk(wblob, b::BounceMsgType::kWANT);
    ASSERT_TRUE(b::decodeWant(wblob, wh, real, ep2));
    EXPECT_FALSE(b::isCancelWant(real));
}

TEST(BounceMessageCodec, WantEmptyEndpointRoundTrips)
{
    auto blob = b::encodeWant(7, std::vector<std::uint32_t>{16}, "");
    auto h = decodeOk(blob, b::BounceMsgType::kWANT);
    std::vector<std::uint32_t> out;
    std::string outEp;
    ASSERT_TRUE(b::decodeWant(blob, h, out, outEp));
    EXPECT_EQ(out, (std::vector<std::uint32_t>{16}));
    EXPECT_TRUE(outEp.empty());
}

TEST(BounceMessageCodec, GrantRoundTrip)
{
    std::vector<b::BounceCreditEntry> credits{{0x3000, 256, 1, 3}, {0x7000, 512, 1, 7}};
    auto blob = b::encodeGrant(/*rid=*/99, credits);
    auto h = decodeOk(blob, b::BounceMsgType::kGRANT);
    EXPECT_EQ(h.requestId, 99u);
    std::vector<b::BounceCreditEntry> out;
    ASSERT_TRUE(b::decodeCredits(blob, h, out));
    ASSERT_EQ(out.size(), 2u);
    EXPECT_EQ(out[0].regionHandle, 3u);
    EXPECT_EQ(out[0].devId, 1u);
    EXPECT_EQ(out[1].addr, 0x7000u);
    EXPECT_EQ(out[1].len, 512u);
}

TEST(BounceMessageCodec, DataRoundTrip)
{
    // {bounceOffset, dstAddr, dstStride, bounceStride, pieceSize, count}: one plain extent + one
    // 3-piece strided run.
    std::vector<b::BounceScatterRun> entries{{0, 0xD000, 0, 0, 128, 1}, {128, 0xE000, 4096, 128, 64, 3}};
    auto blob = b::encodeData(/*rid=*/7, /*chunk=*/2, /*numChunks=*/5, /*regionHandle=*/4, entries);
    auto h = decodeOk(blob, b::BounceMsgType::kDATA);
    EXPECT_EQ(h.requestId, 7u);
    EXPECT_EQ(h.chunkIdx, 2u);
    EXPECT_EQ(h.numChunks, 5u);
    EXPECT_EQ(h.regionHandle, 4u);
    std::vector<b::BounceScatterRun> out;
    ASSERT_TRUE(b::decodeScatter(blob, h, out));
    ASSERT_EQ(out.size(), 2u);
    EXPECT_EQ(out[0].dstAddr, 0xD000u);
    EXPECT_EQ(out[0].count, 1u);
    EXPECT_EQ(out[1].bounceOffset, 128u);
    EXPECT_EQ(out[1].dstStride, 4096u);
    EXPECT_EQ(out[1].bounceStride, 128u);
    EXPECT_EQ(out[1].pieceSize, 64u);
    EXPECT_EQ(out[1].count, 3u);
}

TEST(BounceMessageCodec, AckRoundTrip)
{
    auto blob = b::encodeAck(/*rid=*/7, /*chunk=*/3, /*regionHandle=*/9);
    auto h = decodeOk(blob, b::BounceMsgType::kACK);
    EXPECT_EQ(h.requestId, 7u);
    EXPECT_EQ(h.chunkIdx, 3u);
    EXPECT_EQ(h.regionHandle, 9u);
}

TEST(BounceMessageCodec, EmptyEntriesSerialize)
{
    auto blob = b::encodeGrant(1, {});
    auto h = decodeOk(blob, b::BounceMsgType::kGRANT);
    EXPECT_EQ(h.count, 0u);
    EXPECT_EQ(h.payloadBytes, 0u);
    std::vector<b::BounceCreditEntry> out;
    ASSERT_TRUE(b::decodeCredits(blob, h, out));
    EXPECT_TRUE(out.empty());
}

TEST(BounceMessageCodec, LargeScatterCountRoundTrips)
{
    // A single DATA chunk can carry many small descriptors (the exact case bounce optimizes).
    // `count`/`payloadBytes` are 32-bit, so a count well past 65535 must round-trip intact —
    // guards against any accidental 16-bit truncation of the entry count.
    constexpr std::uint32_t kN = 100000; // 100k * 36B = 3.6 MB payload
    std::vector<b::BounceScatterRun> entries(kN);
    for (std::uint32_t i = 0; i < kN; ++i)
    {
        entries[i] = {static_cast<std::uint64_t>(i) * 64, 0x10000000ULL + i, 0, 0, 64, 1};
    }
    auto blob = b::encodeData(/*rid=*/1, /*chunk=*/0, /*numChunks=*/1, /*regionHandle=*/0, entries);
    auto h = decodeOk(blob, b::BounceMsgType::kDATA);
    EXPECT_EQ(h.count, kN);
    EXPECT_EQ(h.payloadBytes, kN * sizeof(b::BounceScatterRun));
    std::vector<b::BounceScatterRun> out;
    ASSERT_TRUE(b::decodeScatter(blob, h, out));
    ASSERT_EQ(out.size(), kN);
    // Spot-check the boundaries and the >16-bit index region.
    EXPECT_EQ(out[0].dstAddr, 0x10000000ULL);
    EXPECT_EQ(out[65535].bounceOffset, static_cast<std::uint64_t>(65535) * 64);
    EXPECT_EQ(out[kN - 1].dstAddr, 0x10000000ULL + (kN - 1));
    EXPECT_EQ(out[kN - 1].count, 1u);
}

TEST(BounceMessageCodec, LargeCreditCountRoundTrips)
{
    constexpr std::uint32_t kN = 70000; // > 65535 credits in one GRANT
    std::vector<b::BounceCreditEntry> credits(kN);
    for (std::uint32_t i = 0; i < kN; ++i)
    {
        credits[i] = {0x2000ULL + i, 4096, 0, i}; // {addr, len, devId, regionHandle}
    }
    auto blob = b::encodeGrant(/*rid=*/5, credits);
    auto h = decodeOk(blob, b::BounceMsgType::kGRANT);
    EXPECT_EQ(h.count, kN);
    EXPECT_EQ(h.payloadBytes, kN * sizeof(b::BounceCreditEntry));
    std::vector<b::BounceCreditEntry> out;
    ASSERT_TRUE(b::decodeCredits(blob, h, out));
    ASSERT_EQ(out.size(), kN);
    EXPECT_EQ(out[kN - 1].regionHandle, kN - 1);
    EXPECT_EQ(out[kN - 1].addr, 0x2000ULL + (kN - 1));
}

TEST(BounceMessageCodec, ShortBlobRejected)
{
    std::string tiny(8, '\0');
    b::BounceMsgHeader h{};
    EXPECT_FALSE(b::decodeHeader(tiny, h));
    EXPECT_FALSE(b::hasBounceMagic(std::string(2, '\0')));
}

TEST(BounceMessageCodec, BadMagicRejected)
{
    auto blob = b::encodeAck(1, 0, 0);
    blob[0] = 'X'; // corrupt magic
    b::BounceMsgHeader h{};
    EXPECT_FALSE(b::decodeHeader(blob, h));
}

TEST(BounceMessageCodec, InflatedPayloadBytesRejected)
{
    auto blob = b::encodeAck(1, 0, 0);
    // Corrupt payloadBytes to a value the blob can't satisfy.
    std::uint32_t inflated = 4096;
    std::memcpy(blob.data() + offsetof(b::BounceMsgHeader, payloadBytes), &inflated, sizeof(inflated));
    b::BounceMsgHeader h{};
    EXPECT_FALSE(b::decodeHeader(blob, h));
}

TEST(BounceMessageCodec, CrossTypeDecodeMismatchRejected)
{
    // A WANT blob (4-byte u32 chunk sizes) decoded as credits (24-byte) -> count*24 != payloadBytes.
    auto blob = b::encodeWant(1, std::vector<std::uint32_t>{4096, 8192}, "tcp://x:1");
    b::BounceMsgHeader h{};
    ASSERT_TRUE(b::decodeHeader(blob, h));
    std::vector<b::BounceCreditEntry> wrong;
    EXPECT_FALSE(b::decodeCredits(blob, h, wrong));
}
