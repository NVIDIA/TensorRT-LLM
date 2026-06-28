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

#include "tensorrt_llm/common/logger.h"

#include <array>
#include <chrono>
#include <utility>

namespace tensorrt_llm::executor::kv_cache::bounce
{

namespace
{
// Per-peer outbound queue cap (messages). With the non-blocking send in sendTo(), hitting this cap
// DROPS the message instead of blocking the caller. Control messages are tiny (tens to a few hundred
// bytes), so this bounds a stalled peer's queue to a few MB while sitting far above any legitimate
// in-flight burst (per-flow window x concurrent flows) — drops only happen under genuine peer stall.
constexpr int kSendHwm = 1 << 16;
} // namespace

ZmqControlChannel::ZmqControlChannel(std::string selfName, std::string const& bindAddr)
    : mSelfName(std::move(selfName))
    , mCtx(/*io_threads=*/1)
    , mRouter(mCtx, zmq::socket_type::router)
{
    // Identify ourselves to peers' ROUTERs and fail fast rather than queue forever if a peer
    // is unreachable (ROUTER drops messages to unknown/again-full peers by default).
    mRouter.set(zmq::sockopt::routing_id, mSelfName);
    mRouter.set(zmq::sockopt::linger, 0);
    // zmq disables IPv6 on a socket by default, so binding an IPv6 address would fail. Enable it when
    // the bind address is IPv6 (brackets, e.g. "tcp://[::1]:*"). Mirrors ucx_utils. (Default ctor arg
    // is the IPv4 loopback, so tests are unaffected.)
    if (bindAddr.find('[') != std::string::npos)
    {
        mRouter.set(zmq::sockopt::ipv6, 1);
    }
    mRouter.bind(bindAddr);
    mEndpoint = mRouter.get(zmq::sockopt::last_endpoint);
}

ZmqControlChannel::~ZmqControlChannel() = default;

std::string ZmqControlChannel::localEndpoint() const
{
    return mEndpoint;
}

void ZmqControlChannel::addPeer(std::string const& peer, std::string const& endpoint)
{
    std::lock_guard<std::mutex> lk(mMu);
    if (mDealers.find(peer) != mDealers.end())
    {
        return; // idempotent
    }
    zmq::socket_t dealer(mCtx, zmq::socket_type::dealer);
    dealer.set(zmq::sockopt::routing_id, mSelfName); // so peer's ROUTER sees us by name
    dealer.set(zmq::sockopt::linger, 0);
    dealer.set(zmq::sockopt::sndhwm, kSendHwm);      // bound the queue; full -> sendTo drops (never blocks)
    // Enable IPv6 unconditionally (off by default in zmq) so a DEALER can connect to an IPv6 peer
    // endpoint; harmless when the endpoint is IPv4. Mirrors ucx_utils' connect socket.
    dealer.set(zmq::sockopt::ipv6, 1);
    dealer.connect(endpoint);
    mDealers.emplace(peer, std::move(dealer));
}

void ZmqControlChannel::sendTo(std::string const& peer, std::string const& blob)
{
    std::lock_guard<std::mutex> lk(mMu);
    auto it = mDealers.find(peer);
    if (it == mDealers.end())
    {
        TLLM_LOG_WARNING(
            "ZmqControlChannel(%s): sendTo unknown peer %s (call addPeer first)", mSelfName.c_str(), peer.c_str());
        return;
    }
    zmq::message_t msg(blob.data(), blob.size());
    try
    {
        // DEALER -> peer ROUTER; the peer receives [our routing id, blob]. NON-BLOCKING: this runs on
        // the IO thread (and under mMu, which also gates submit()), so a blocking send to a stalled /
        // unreachable peer whose queue is full (kSendHwm + TCP buffers) would wedge the whole reactor
        // — exactly the hang the design forbids. With dontwait a full queue returns an empty result
        // (EAGAIN) instead; we DROP the message. A dropped control message degrades the affected
        // request to a leaseTimeout FAILURE — never a hang or data corruption (DESIGN.md §10).
        auto const sent = it->second.send(msg, zmq::send_flags::dontwait);
        if (!sent.has_value())
        {
            TLLM_LOG_WARNING("ZmqControlChannel(%s): send to %s dropped (queue full / peer stalled)", mSelfName.c_str(),
                peer.c_str());
        }
    }
    catch (zmq::error_t const& e)
    {
        TLLM_LOG_WARNING("ZmqControlChannel(%s): send to %s failed: %s", mSelfName.c_str(), peer.c_str(), e.what());
    }
}

bool ZmqControlChannel::recv(std::string& outPeer, std::string& outBlob, int timeoutMs)
{
    std::array<zmq::pollitem_t, 1> items{{{mRouter.handle(), 0, ZMQ_POLLIN, 0}}};
    zmq::poll(items.data(), items.size(), std::chrono::milliseconds(timeoutMs));
    if ((items[0].revents & ZMQ_POLLIN) == 0)
    {
        return false;
    }
    // A DEALER->ROUTER message arrives as [identity, body].
    zmq::message_t idFrame;
    auto r1 = mRouter.recv(idFrame, zmq::recv_flags::none);
    if (!r1.has_value())
    {
        return false;
    }
    zmq::message_t bodyFrame;
    auto r2 = mRouter.recv(bodyFrame, zmq::recv_flags::none);
    if (!r2.has_value())
    {
        return false;
    }
    outPeer.assign(static_cast<char const*>(idFrame.data()), idFrame.size());
    outBlob.assign(static_cast<char const*>(bodyFrame.data()), bodyFrame.size());
    // A well-formed message is exactly [identity, body]. If a malformed peer sent extra frames,
    // drain them so they don't desync the NEXT recv() (which would then read this message's leftover
    // frame as an identity). We accept [identity, body] and discard any trailing parts.
    while (bodyFrame.more())
    {
        zmq::message_t extra;
        auto re = mRouter.recv(extra, zmq::recv_flags::none);
        if (!re.has_value())
        {
            break;
        }
        bodyFrame.swap(extra); // advance the "more" flag to the just-read frame
    }
    return true;
}

} // namespace tensorrt_llm::executor::kv_cache::bounce
