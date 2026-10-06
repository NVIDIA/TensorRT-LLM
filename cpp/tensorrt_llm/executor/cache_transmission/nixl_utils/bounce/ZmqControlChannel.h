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

#pragma once

#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/ControlChannel.h"

#include <zmq.hpp>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <future>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>

namespace tensorrt_llm::executor::kv_cache::bounce
{

// ============================================================================
// ZmqControlChannel — zmq implementation of ControlChannel
// ----------------------------------------------------------------------------
// Topology (one-way, symmetric peers)
//   Each agent owns:
//     - ONE ROUTER socket, bound to a tcp endpoint, ZMQ_ROUTING_ID = self name.
//       It only RECEIVES. A peer's DEALER connects here; on recv the ROUTER yields
//       [senderName, blob] (senderName = the peer DEALER's routing id).
//     - ONE DEALER per peer, ZMQ_ROUTING_ID = self name, connected to that peer's
//       ROUTER endpoint. It only SENDS. The peer's ROUTER then sees us by name.
//   So "A -> B" travels A's DEALER(B) -> B's ROUTER; "B -> A" travels B's DEALER(A)
//   -> A's ROUTER. No request/reply coupling — purely async one-way, which is all
//   the credit protocol needs.
//
// Bootstrap
//   localEndpoint() returns the ROUTER's bound address (auto-port via "tcp://host:*"
//   + ZMQ_LAST_ENDPOINT). Endpoints are exchanged out-of-band (NIXL agent metadata)
//   and registered with addPeer() before the first sendTo().
//
// Threading
//   Every ROUTER and DEALER is created, used and destroyed by ONE dedicated control
//   thread. Public addPeer/removePeer calls enqueue synchronous commands; sendTo is
//   fire-and-forget and only enqueues the blob. The control thread places received
//   messages on an application queue for recv(). This is stricter than serializing
//   calls with a mutex: zmq sockets are not thread-safe, so creating a socket on the
//   transfer-agent thread and later using it on the reactor thread is also invalid.
// ============================================================================
class ZmqControlChannel : public ControlChannel
{
public:
    /// @param selfName    this agent's name (becomes the zmq routing id).
    /// @param bindAddr    address pattern to bind the ROUTER ("tcp://127.0.0.1:*" by default,
    ///                    "*" picks a free port; query the chosen one via localEndpoint()).
    explicit ZmqControlChannel(std::string selfName, std::string const& bindAddr = "tcp://127.0.0.1:*");
    ~ZmqControlChannel() override;

    [[nodiscard]] std::string localEndpoint() const override;
    bool addPeer(std::string const& peer, std::string const& endpoint) override;
    void removePeer(std::string const& peer) override;
    void sendTo(std::string const& peer, std::string const& blob) override;
    [[nodiscard]] bool recv(std::string& outPeer, std::string& outBlob, int timeoutMs) override;

private:
    enum class OutboundOp
    {
        kAddPeer,
        kRemovePeer,
        kSend,
        kStop,
    };

    struct OutboundCommand
    {
        OutboundOp op{};
        std::string peer;
        std::string payload;
        std::shared_ptr<std::promise<bool>> addDone;
        std::shared_ptr<std::promise<void>> removeDone;
    };

    void controlLoop(std::string bindAddr, std::promise<std::string> ready);

    std::string mSelfName;
    zmq::context_t mCtx;
    std::string mEndpoint;

    std::mutex mQueueMu;
    std::deque<OutboundCommand> mOutbound;
    bool mStopping{false};
    std::atomic<bool> mQueueFullWarned{false};

    std::mutex mInboundMu;
    std::condition_variable mInboundCv;
    std::deque<std::pair<std::string, std::string>> mInbound;
    std::atomic<bool> mControlStopped{false};

    // Control-thread-only. The thread clears this map before it exits so every DEALER is destroyed
    // by its sole owning thread. The ROUTER is a local in controlLoop for the same reason.
    std::unordered_map<std::string, zmq::socket_t> mDealers;
    std::thread mControlThread;
};

} // namespace tensorrt_llm::executor::kv_cache::bounce
