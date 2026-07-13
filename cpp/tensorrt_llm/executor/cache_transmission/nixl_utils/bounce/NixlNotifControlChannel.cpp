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

#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/NixlNotifControlChannel.h"

#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/BounceMessage.h"
#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/BounceNvtx.h"

#include "tensorrt_llm/common/logger.h"

#include "nixl.h"

#include <chrono>
#include <thread>

namespace tensorrt_llm::executor::kv_cache::bounce
{

NixlNotifControlChannel::NixlNotifControlChannel(nixlAgent* agent, std::string selfName)
    : mAgent(agent)
    , mSelfName(std::move(selfName))
{
}

std::string NixlNotifControlChannel::localEndpoint() const
{
    // The "endpoint" IS this agent's serialized NIXL metadata: the receiver loadRemoteMD()s it in
    // addPeer (WANT self-bootstrap) so it can genNotif back. Fetched lazily and cached — conn info
    // is stable once the agent's backend is up.
    std::lock_guard<std::mutex> lk(mMdMu);
    if (mLocalMd.empty())
    {
        nixl_blob_t blob;
        nixl_status_t const st = mAgent->getLocalMD(blob);
        if (st != NIXL_SUCCESS)
        {
            TLLM_LOG_WARNING("NixlNotifControlChannel(%s): getLocalMD failed: %s (reverse control path will not "
                             "bootstrap; requests to us will lease-timeout)",
                mSelfName.c_str(), nixlEnumStrings::statusStr(st).c_str());
            return {};
        }
        mLocalMd = std::move(blob);
    }
    return mLocalMd;
}

void NixlNotifControlChannel::addPeer(std::string const& peer, std::string const& endpoint)
{
    {
        std::lock_guard<std::mutex> lk(mMu);
        if (mPeers.find(peer) != mPeers.end())
        {
            return; // idempotent
        }
    }
    // `endpoint` is the peer's serialized agent metadata (see localEndpoint). Loading it gives this
    // agent the connection info needed to genNotif the peer. On the KV-sender side the peer is
    // usually already loaded (registrar loadRemoteAgent) — loadRemoteMD is idempotent there.
    if (endpoint.empty())
    {
        TLLM_LOG_WARNING(
            "NixlNotifControlChannel(%s): addPeer(%s) with empty metadata; dropping", mSelfName.c_str(), peer.c_str());
        return;
    }
    std::string loadedName;
    nixl_status_t const st = mAgent->loadRemoteMD(endpoint, loadedName);
    if (st != NIXL_SUCCESS)
    {
        TLLM_LOG_WARNING("NixlNotifControlChannel(%s): loadRemoteMD for peer %s failed: %s", mSelfName.c_str(),
            peer.c_str(), nixlEnumStrings::statusStr(st).c_str());
        return;
    }
    if (loadedName != peer)
    {
        // The MD names a different agent than the control-plane peer id — a protocol mixup (e.g. a
        // zmq endpoint string arrived instead of MD because the peers disagree on the channel type).
        TLLM_LOG_WARNING("NixlNotifControlChannel(%s): addPeer(%s) metadata names agent '%s'; dropping",
            mSelfName.c_str(), peer.c_str(), loadedName.c_str());
        return;
    }
    std::lock_guard<std::mutex> lk(mMu);
    mPeers.insert(peer);
}

void NixlNotifControlChannel::removePeer(std::string const& peer)
{
    // Only forget the channel-level peer entry. The agent-level remote MD is owned by the
    // transfer-agent lifecycle (invalidateRemoteAgent) — invalidating it here would yank the DATA
    // plane's connection out from under in-flight transfers.
    std::lock_guard<std::mutex> lk(mMu);
    mPeers.erase(peer);
}

void NixlNotifControlChannel::sendTo(std::string const& peer, std::string const& blob)
{
    // Same fire-and-forget semantics as the zmq channel: a failed send is DROPPED (warned) and the
    // affected request degrades to a leaseTimeout FAILURE — never a hang. genNotif is safe from any
    // thread (the agent is created with thread-safe sync; the data plane already relies on this).
    BounceNvtxScope sendScope(kNvtxZmqSend, "notifSend bytes=%zu", blob.size());
    nixl_status_t const st = mAgent->genNotif(peer, blob);
    if (st != NIXL_SUCCESS)
    {
        TLLM_LOG_WARNING("NixlNotifControlChannel(%s): genNotif to %s (%zu B) failed: %s", mSelfName.c_str(),
            peer.c_str(), blob.size(), nixlEnumStrings::statusStr(st).c_str());
    }
}

bool NixlNotifControlChannel::drainNotifs()
{
    nixl_notifs_t notifs;
    nixl_status_t const st = mAgent->getNotifs(notifs);
    if (st != NIXL_SUCCESS)
    {
        TLLM_LOG_WARNING(
            "NixlNotifControlChannel(%s): getNotifs failed: %s", mSelfName.c_str(), nixlEnumStrings::statusStr(st).c_str());
        return false;
    }
    bool any = false;
    std::lock_guard<std::mutex> lk(mMu);
    for (auto& [peer, blobs] : notifs)
    {
        for (auto& blob : blobs)
        {
            any = true;
            if (hasBounceMagic(blob))
            {
                mInbox.emplace_back(peer, std::move(blob));
            }
            else
            {
                // getNotifs is a per-agent singleton queue; this channel owns it while enabled. A
                // foreign notification means some other component (e.g. the C++ transceiver's sync
                // messages) also uses agent notifications — that combination is unsupported with
                // TRTLLM_NIXL_BOUNCE_NIXL_CONTROL and the message cannot be re-queued; drop loudly.
                TLLM_LOG_WARNING("NixlNotifControlChannel(%s): dropping non-bounce notification from %s (%zu B)",
                    mSelfName.c_str(), peer.c_str(), blob.size());
            }
        }
    }
    return any;
}

bool NixlNotifControlChannel::recv(std::string& outPeer, std::string& outBlob, int timeoutMs)
{
    auto const deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
    for (;;)
    {
        {
            std::lock_guard<std::mutex> lk(mMu);
            if (!mInbox.empty())
            {
                outPeer = std::move(mInbox.front().first);
                outBlob = std::move(mInbox.front().second);
                mInbox.pop_front();
                return true;
            }
        }
        if (drainNotifs())
        {
            continue; // something arrived; loop back to pop (may have been a dropped foreign blob)
        }
        if (timeoutMs <= 0 || std::chrono::steady_clock::now() >= deadline)
        {
            return false;
        }
        // getNotifs has no waitable fd -> short-sleep poll until the deadline. Only the reactor's
        // IDLE path passes a non-zero (1ms) timeout, so this loop is bounded and low-rate.
        std::this_thread::sleep_for(std::chrono::microseconds(20));
    }
}

} // namespace tensorrt_llm::executor::kv_cache::bounce
