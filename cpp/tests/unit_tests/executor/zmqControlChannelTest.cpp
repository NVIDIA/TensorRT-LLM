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

#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/ZmqControlChannel.h"
#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/BounceMessage.h"

#include <gtest/gtest.h>

#include <chrono>
#include <future>
#include <memory>
#include <string>
#include <thread>

namespace b = tensorrt_llm::executor::kv_cache::bounce;

namespace
{
// recv with a few retries to absorb zmq connection-establishment latency.
bool recvRetry(b::ZmqControlChannel& ch, std::string& peer, std::string& blob, int totalMs)
{
    int waited = 0;
    while (waited < totalMs)
    {
        if (ch.recv(peer, blob, 100))
        {
            return true;
        }
        waited += 100;
    }
    return false;
}
} // namespace

TEST(ZmqControlChannel, BidirectionalMessagesOverTcp)
{
    b::ZmqControlChannel a("agentA");
    b::ZmqControlChannel bch("agentB");
    a.addPeer("agentB", bch.localEndpoint());
    bch.addPeer("agentA", a.localEndpoint());

    // A -> B : WANT(rid=42, chunk sizes [4096, 8192, 1024])
    a.sendTo("agentB", b::encodeWant(42, std::vector<std::uint32_t>{4096, 8192, 1024}, a.localEndpoint()));
    std::string from, blob;
    ASSERT_TRUE(recvRetry(bch, from, blob, 4000)) << "B never received A's WANT";
    EXPECT_EQ(from, "agentA");
    b::BounceMsgHeader h{};
    ASSERT_TRUE(b::decodeHeader(blob, h));
    EXPECT_EQ(h.msgType, static_cast<std::uint16_t>(b::BounceMsgType::kWANT));
    EXPECT_EQ(h.requestId, 42u);
    EXPECT_EQ(h.aux, 3u); // numChunks
    std::vector<std::uint32_t> sizes;
    std::string ep;
    ASSERT_TRUE(b::decodeWant(blob, h, sizes, ep));
    EXPECT_EQ(sizes, (std::vector<std::uint32_t>{4096, 8192, 1024}));
    EXPECT_EQ(ep, a.localEndpoint());

    // B -> A : GRANT(rid=42, {region handle 3 @ 0x3000, len 4096}) -- {addr, len, devId, regionHandle}
    bch.sendTo("agentA", b::encodeGrant(42, {{0x3000, 4096, 0, 3}}));
    ASSERT_TRUE(recvRetry(a, from, blob, 4000)) << "A never received B's GRANT";
    EXPECT_EQ(from, "agentB");
    ASSERT_TRUE(b::decodeHeader(blob, h));
    EXPECT_EQ(h.msgType, static_cast<std::uint16_t>(b::BounceMsgType::kGRANT));
    std::vector<b::BounceCreditEntry> credits;
    ASSERT_TRUE(b::decodeCredits(blob, h, credits));
    ASSERT_EQ(credits.size(), 1u);
    EXPECT_EQ(credits[0].regionHandle, 3u);
    EXPECT_EQ(credits[0].addr, 0x3000u);
}

TEST(ZmqControlChannel, RecvTimesOutWhenIdle)
{
    b::ZmqControlChannel a("idleA");
    std::string peer, blob;
    EXPECT_FALSE(a.recv(peer, blob, 50)); // nothing sent -> timeout, no spurious message
}

TEST(ZmqControlChannel, ManyMessagesPreserveOrderAndContent)
{
    b::ZmqControlChannel a("ordA");
    b::ZmqControlChannel bch("ordB");
    a.addPeer("ordB", bch.localEndpoint());

    constexpr int kN = 50;
    for (int i = 0; i < kN; ++i)
    {
        a.sendTo("ordB", b::encodeAck(/*rid=*/static_cast<std::uint64_t>(i), /*chunk=*/i, /*regionHandle=*/i));
    }
    // Per-peer FIFO: messages from a single DEALER arrive in order.
    for (int i = 0; i < kN; ++i)
    {
        std::string from, blob;
        ASSERT_TRUE(recvRetry(bch, from, blob, 4000)) << "missing msg " << i;
        b::BounceMsgHeader h{};
        ASSERT_TRUE(b::decodeHeader(blob, h));
        EXPECT_EQ(h.requestId, static_cast<std::uint64_t>(i));
        EXPECT_EQ(h.chunkIdx, static_cast<std::uint32_t>(i));
    }
}

TEST(ZmqControlChannel, SendToDoesNotBlockWhenPeerQueueFull)
{
    // sendTo() runs on the reactor IO thread (and under the dealer mutex that also gates submit()),
    // so it must NEVER block. Blast far more messages than any HWM/TCP buffer can hold to a peer that
    // never reads: a blocking send would wedge the caller forever (the reactor stall the design
    // forbids); the non-blocking send must drop and let the loop finish promptly.
    auto a = std::make_shared<b::ZmqControlChannel>("floodA");
    b::ZmqControlChannel bch("floodB"); // bound ROUTER, but we deliberately never recv() on it
    a->addPeer("floodB", bch.localEndpoint());

    std::promise<void> done;
    auto fut = done.get_future();
    // Capture `a` by value (shared_ptr) so that if the bug is present and this thread stays wedged in
    // sendTo when we give up and detach below, the channel outlives this scope (no use-after-free).
    std::thread flood(
        [a, &done]
        {
            for (int i = 0; i < 200000; ++i)
            {
                a->sendTo("floodB", b::encodeAck(static_cast<std::uint64_t>(i), i, i));
            }
            done.set_value();
        });

    bool const finished = fut.wait_for(std::chrono::seconds(10)) == std::future_status::ready;
    EXPECT_TRUE(finished) << "sendTo blocked when the peer's queue filled (it must drop, not block)";
    if (finished)
    {
        flood.join();
    }
    else
    {
        flood.detach(); // bug present: leave the wedged thread; the shared_ptr keeps `a` alive
    }
}

TEST(ZmqControlChannel, BidirectionalMessagesOverIPv6)
{
    // Binding an IPv6 endpoint requires ZMQ_IPV6 on the ROUTER, and connecting to one requires it on
    // the DEALER (both off by default). Bind the IPv6 loopback; skip cleanly where IPv6 is unavailable.
    std::unique_ptr<b::ZmqControlChannel> a;
    std::unique_ptr<b::ZmqControlChannel> bch;
    try
    {
        a = std::make_unique<b::ZmqControlChannel>("v6A", "tcp://[::1]:*");
        bch = std::make_unique<b::ZmqControlChannel>("v6B", "tcp://[::1]:*");
    }
    catch (zmq::error_t const& e)
    {
        GTEST_SKIP() << "IPv6 unavailable: " << e.what();
    }
    // The bound endpoint must be IPv6 (bracketed) — proves the IPv6 bind actually took.
    ASSERT_NE(a->localEndpoint().find('['), std::string::npos) << "expected IPv6 endpoint, got " << a->localEndpoint();
    a->addPeer("v6B", bch->localEndpoint());

    a->sendTo("v6B", b::encodeWant(7, std::vector<std::uint32_t>{2048}, a->localEndpoint()));
    std::string from, blob;
    ASSERT_TRUE(recvRetry(*bch, from, blob, 4000)) << "B never received A's WANT over IPv6";
    EXPECT_EQ(from, "v6A");
    b::BounceMsgHeader h{};
    ASSERT_TRUE(b::decodeHeader(blob, h));
    std::vector<std::uint32_t> sizes;
    std::string ep;
    ASSERT_TRUE(b::decodeWant(blob, h, sizes, ep));
    EXPECT_EQ(sizes, (std::vector<std::uint32_t>{2048}));
    EXPECT_EQ(ep, a->localEndpoint());
}
