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

#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/BounceNvtx.h"

#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/tllmException.h"

#include <array>
#include <chrono>
#include <exception>
#include <future>
#include <utility>

namespace tensorrt_llm::executor::kv_cache::bounce
{

namespace
{
// Per-peer outbound queue cap (messages). With the non-blocking send in sendTo(), hitting this cap
// DROPS the message instead of blocking the caller. Control messages are tiny (tens to a few hundred
// bytes), so this bounds a stalled peer's queue to a few MB while sitting far above any legitimate
// in-flight burst (per-request chunk cap times concurrent flows), so drops indicate peer stall.
constexpr int kSendHwm = 1 << 16;
// Bound the application-side command queue as well. Otherwise a stalled outbound thread could
// consume unbounded host memory before the DEALER's own high-water mark is reached.
constexpr std::size_t kCommandHwm = 1 << 16;
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
    // Accept a reconnecting peer that reuses an existing routing id. Our peers' DEALERs identify by a
    // fixed agent name, so when a peer is forgotten (removePeer drops its DEALER) and later comes back
    // — same agent name = same identity — it reconnects with the SAME routing id. Without handover the
    // ROUTER REJECTS the new connection while the old one is still being reaped and SILENTLY DROPS its
    // messages (a forgotten-then-readded peer's WANTs vanish until request timeout). HANDOVER hands the
    // identity to the new connection instead. (Loopback reconnect makes this race easy to hit.)
    mRouter.set(zmq::sockopt::router_handover, 1);
    // zmq disables IPv6 on a socket by default, so binding an IPv6 address would fail. Enable it when
    // the bind address is IPv6 (brackets, e.g. "tcp://[::1]:*"). Mirrors ucx_utils. (Default ctor arg
    // is the IPv4 loopback, so tests are unaffected.)
    if (bindAddr.find('[') != std::string::npos)
    {
        mRouter.set(zmq::sockopt::ipv6, 1);
    }
    mRouter.bind(bindAddr);
    mEndpoint = mRouter.get(zmq::sockopt::last_endpoint);
    mOutboundThread = std::thread(&ZmqControlChannel::outboundLoop, this);
}

ZmqControlChannel::~ZmqControlChannel()
{
    {
        std::lock_guard<std::mutex> lk(mQueueMu);
        mStopping = true;
        mOutbound.push_back({OutboundOp::kStop});
    }
    mQueueCv.notify_one();
    if (mOutboundThread.joinable())
    {
        mOutboundThread.join();
    }
}

std::string ZmqControlChannel::localEndpoint() const
{
    return mEndpoint;
}

bool ZmqControlChannel::addPeer(std::string const& peer, std::string const& endpoint)
{
    auto done = std::make_shared<std::promise<bool>>();
    auto result = done->get_future();
    {
        std::lock_guard<std::mutex> lk(mQueueMu);
        if (mStopping)
        {
            TLLM_THROW("ZmqControlChannel(%s): cannot add peer %s during shutdown", mSelfName.c_str(), peer.c_str());
        }
        mOutbound.push_back({OutboundOp::kAddPeer, peer, endpoint, std::move(done), {}});
    }
    mQueueCv.notify_one();
    return result.get();
}

void ZmqControlChannel::removePeer(std::string const& peer)
{
    auto done = std::make_shared<std::promise<void>>();
    auto result = done->get_future();
    {
        std::lock_guard<std::mutex> lk(mQueueMu);
        if (mStopping)
        {
            return;
        }
        mOutbound.push_back({OutboundOp::kRemovePeer, peer, {}, {}, std::move(done)});
    }
    mQueueCv.notify_one();
    result.get();
}

void ZmqControlChannel::sendTo(std::string const& peer, std::string const& blob)
{
    {
        std::lock_guard<std::mutex> lk(mQueueMu);
        if (mStopping)
        {
            return;
        }
        if (mOutbound.size() >= kCommandHwm)
        {
            if (!mQueueFullWarned.exchange(true))
            {
                TLLM_LOG_WARNING(
                    "ZmqControlChannel(%s): outbound command queue full; dropping sends", mSelfName.c_str());
            }
            return;
        }
        mOutbound.push_back({OutboundOp::kSend, peer, blob, {}, {}});
    }
    mQueueCv.notify_one();
}

void ZmqControlChannel::outboundLoop()
{
    while (true)
    {
        OutboundCommand cmd;
        {
            std::unique_lock<std::mutex> lk(mQueueMu);
            mQueueCv.wait(lk, [this] { return !mOutbound.empty(); });
            cmd = std::move(mOutbound.front());
            mOutbound.pop_front();
        }

        if (cmd.op == OutboundOp::kStop)
        {
            mDealers.clear();
            return;
        }
        if (cmd.op == OutboundOp::kAddPeer)
        {
            try
            {
                if (mDealers.find(cmd.peer) == mDealers.end())
                {
                    if (cmd.payload.empty())
                    {
                        TLLM_THROW("ZmqControlChannel(%s): peer %s requires a non-empty endpoint", mSelfName.c_str(),
                            cmd.peer.c_str());
                    }
                    zmq::socket_t dealer(mCtx, zmq::socket_type::dealer);
                    dealer.set(zmq::sockopt::routing_id, mSelfName);
                    dealer.set(zmq::sockopt::linger, 0);
                    dealer.set(zmq::sockopt::sndhwm, kSendHwm);
                    dealer.set(zmq::sockopt::ipv6, 1);
                    dealer.connect(cmd.payload);
                    mDealers.emplace(cmd.peer, std::move(dealer));
                }
                cmd.addDone->set_value(true);
            }
            catch (...)
            {
                cmd.addDone->set_exception(std::current_exception());
            }
            continue;
        }
        if (cmd.op == OutboundOp::kRemovePeer)
        {
            try
            {
                // Erasing closes the DEALER on the same thread that created and used it.
                mDealers.erase(cmd.peer);
                cmd.removeDone->set_value();
            }
            catch (...)
            {
                cmd.removeDone->set_exception(std::current_exception());
            }
            continue;
        }

        BounceNvtxScope sendScope(kNvtxZmqSend, "zmqSend bytes=%zu", cmd.payload.size());
        auto it = mDealers.find(cmd.peer);
        if (it == mDealers.end())
        {
            TLLM_LOG_WARNING("ZmqControlChannel(%s): sendTo unknown peer %s (call addPeer first)", mSelfName.c_str(),
                cmd.peer.c_str());
            continue;
        }
        zmq::message_t msg(cmd.payload.data(), cmd.payload.size());
        try
        {
            // DEALER -> peer ROUTER; the peer receives [our routing id, blob]. NON-BLOCKING: a
            // full or stalled peer queue drops this message, allowing request timeout to fail only
            // the affected transfer instead of wedging the sole outbound owner.
            auto const sent = it->second.send(msg, zmq::send_flags::dontwait);
            if (!sent.has_value())
            {
                TLLM_LOG_WARNING("ZmqControlChannel(%s): send to %s dropped (queue full / peer stalled)",
                    mSelfName.c_str(), cmd.peer.c_str());
            }
        }
        catch (zmq::error_t const& e)
        {
            TLLM_LOG_WARNING(
                "ZmqControlChannel(%s): send to %s failed: %s", mSelfName.c_str(), cmd.peer.c_str(), e.what());
        }
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
    // A DEALER->ROUTER message arrives as [identity, body]. The span excludes the poll wait above
    // (that's idle time, not work): it measures the frame reads + the blob copy out to the caller.
    BounceNvtxScope recvScope(kNvtxZmqRecv, "zmqRecv");
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
