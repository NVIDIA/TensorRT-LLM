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

#include <algorithm>
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
// Alternate bounded command processing with receive polling so a burst of scatter-worker ACKs
// cannot starve inbound credits on the same control thread.
constexpr std::size_t kCommandBatch = 256;
} // namespace

ZmqControlChannel::ZmqControlChannel(std::string selfName, std::string const& bindAddr)
    : mSelfName(std::move(selfName))
    , mCtx(/*io_threads=*/1)
{
    std::promise<std::string> ready;
    auto endpoint = ready.get_future();
    mControlThread = std::thread(&ZmqControlChannel::controlLoop, this, bindAddr, std::move(ready));
    try
    {
        mEndpoint = endpoint.get();
    }
    catch (...)
    {
        if (mControlThread.joinable())
        {
            mControlThread.join();
        }
        throw;
    }
}

ZmqControlChannel::~ZmqControlChannel()
{
    {
        std::lock_guard<std::mutex> lk(mQueueMu);
        mStopping = true;
        mOutbound.push_back({OutboundOp::kStop});
    }
    if (mControlThread.joinable())
    {
        mControlThread.join();
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
}

void ZmqControlChannel::controlLoop(std::string bindAddr, std::promise<std::string> ready)
{
    bool readySignaled = false;
    try
    {
        zmq::socket_t router(mCtx, zmq::socket_type::router);
        // Identify ourselves to peers' ROUTERs and fail fast rather than queue forever if a peer
        // is unreachable (ROUTER drops messages to unknown/again-full peers by default).
        router.set(zmq::sockopt::routing_id, mSelfName);
        router.set(zmq::sockopt::linger, 0);
        // Accept a reconnecting peer that reuses an existing routing id. Our peers' DEALERs identify
        // by a fixed agent name, so hand the identity to the new connection while the old one closes.
        router.set(zmq::sockopt::router_handover, 1);
        if (bindAddr.find('[') != std::string::npos)
        {
            router.set(zmq::sockopt::ipv6, 1);
        }
        router.bind(bindAddr);
        ready.set_value(router.get(zmq::sockopt::last_endpoint));
        readySignaled = true;

        while (true)
        {
            std::deque<OutboundCommand> commands;
            {
                std::lock_guard<std::mutex> lk(mQueueMu);
                auto const count = std::min(kCommandBatch, mOutbound.size());
                for (std::size_t i = 0; i < count; ++i)
                {
                    commands.push_back(std::move(mOutbound.front()));
                    mOutbound.pop_front();
                }
            }

            for (auto& cmd : commands)
            {
                if (cmd.op == OutboundOp::kStop)
                {
                    mDealers.clear();
                    mControlStopped.store(true, std::memory_order_release);
                    mInboundCv.notify_all();
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
                                TLLM_THROW("ZmqControlChannel(%s): peer %s requires a non-empty endpoint",
                                    mSelfName.c_str(), cmd.peer.c_str());
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
                    TLLM_LOG_WARNING("ZmqControlChannel(%s): sendTo unknown peer %s (call addPeer first)",
                        mSelfName.c_str(), cmd.peer.c_str());
                    continue;
                }
                zmq::message_t msg(cmd.payload.data(), cmd.payload.size());
                try
                {
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

            std::array<zmq::pollitem_t, 1> items{{{router.handle(), 0, ZMQ_POLLIN, 0}}};
            auto const pollTimeout = commands.empty() ? std::chrono::milliseconds(1) : std::chrono::milliseconds(0);
            zmq::poll(items.data(), items.size(), pollTimeout);
            if ((items[0].revents & ZMQ_POLLIN) == 0)
            {
                continue;
            }

            BounceNvtxScope recvScope(kNvtxZmqRecv, "zmqRecv");
            zmq::message_t idFrame;
            auto const r1 = router.recv(idFrame, zmq::recv_flags::none);
            if (!r1.has_value())
            {
                continue;
            }
            zmq::message_t bodyFrame;
            auto const r2 = router.recv(bodyFrame, zmq::recv_flags::none);
            if (!r2.has_value())
            {
                continue;
            }
            std::string peer(static_cast<char const*>(idFrame.data()), idFrame.size());
            std::string blob(static_cast<char const*>(bodyFrame.data()), bodyFrame.size());
            while (bodyFrame.more())
            {
                zmq::message_t extra;
                auto const more = router.recv(extra, zmq::recv_flags::none);
                if (!more.has_value())
                {
                    break;
                }
                bodyFrame.swap(extra);
            }
            {
                std::lock_guard<std::mutex> lk(mInboundMu);
                mInbound.emplace_back(std::move(peer), std::move(blob));
            }
            mInboundCv.notify_one();
        }
    }
    catch (...)
    {
        auto const error = std::current_exception();
        if (!readySignaled)
        {
            ready.set_exception(error);
        }
        else
        {
            try
            {
                std::rethrow_exception(error);
            }
            catch (std::exception const& e)
            {
                TLLM_LOG_ERROR("ZmqControlChannel(%s): control thread failed: %s", mSelfName.c_str(), e.what());
            }
        }
        mDealers.clear();
        std::deque<OutboundCommand> pending;
        {
            std::lock_guard<std::mutex> lk(mQueueMu);
            mStopping = true;
            pending.swap(mOutbound);
        }
        for (auto& cmd : pending)
        {
            if (cmd.addDone)
            {
                cmd.addDone->set_exception(error);
            }
            if (cmd.removeDone)
            {
                cmd.removeDone->set_exception(error);
            }
        }
        mControlStopped.store(true, std::memory_order_release);
        mInboundCv.notify_all();
    }
}

bool ZmqControlChannel::recv(std::string& outPeer, std::string& outBlob, int timeoutMs)
{
    std::unique_lock<std::mutex> lk(mInboundMu);
    auto const ready = [this] { return !mInbound.empty() || mControlStopped.load(std::memory_order_acquire); };
    if (mInbound.empty() && timeoutMs > 0)
    {
        mInboundCv.wait_for(lk, std::chrono::milliseconds(timeoutMs), ready);
    }
    if (mInbound.empty())
    {
        return false;
    }
    outPeer = std::move(mInbound.front().first);
    outBlob = std::move(mInbound.front().second);
    mInbound.pop_front();
    return true;
}

} // namespace tensorrt_llm::executor::kv_cache::bounce
