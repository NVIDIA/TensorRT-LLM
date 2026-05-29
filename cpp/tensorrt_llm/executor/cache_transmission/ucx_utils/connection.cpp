/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "ucxCacheCommunicator.h"
#if ENABLE_UCX

#include "tensorrt_llm/batch_manager/dataTransceiver.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/tllmException.h"
#include "tensorrt_llm/executor/cache_transmission/ucx_utils/connection.h"

#include <chrono>
#include <cstdint>

namespace tensorrt_llm::executor::kv_cache
{

// Using declarations to shorten the code
using RequestSpecificException = tensorrt_llm::common::RequestSpecificException;
using RequestErrorCode = tensorrt_llm::common::RequestErrorCode;

UcxConnection::UcxConnection(ConnectionIdType connectionId, std::shared_ptr<ucxx::Endpoint> endpoint,
    UcxConnectionManager* manager, bool fromRequester)
    : mConnectionId(connectionId)
    , mEndpoint(std::move(endpoint))
    , mManager(manager)
    , mFromRequester(fromRequester)
{

    try
    {
        if (mFromRequester)
        {

            // since the tag don't contain the information of the connection id or mConnectionIdInPeer, we need to
            // lock the mutex ,to ensure only one tagRecv is called in the same time.
            std::shared_ptr<ucxx::Request> recvRequest
                = mEndpoint->tagRecv(reinterpret_cast<void*>(&mConnectionIdInPeer), sizeof(mConnectionIdInPeer),
                    ucxx::Tag(ResponserTag), ucxx::TagMaskFull);
            while (!recvRequest->isCompleted())
                ;

            recvRequest->checkError();

            auto sendTag = ucxx::Tag(mConnectionIdInPeer << 32 | (RequesterTag & 0xFFFFFFFF));
            std::shared_ptr<ucxx::Request> sendRequest
                = mEndpoint->tagSend(reinterpret_cast<void*>(&mConnectionId), sizeof(mConnectionId), sendTag);
            while (!sendRequest->isCompleted())
                ;
            sendRequest->checkError();
        }
        else
        {

            // Since Responder may recv from multiple Requesters, we need to send the mConnectionId to the Reqester
            // first and use ConnectionId as the tag to recv the mConnectionIdInPeer from the Requester
            std::shared_ptr<ucxx::Request> sendRequest = mEndpoint->tagSend(
                reinterpret_cast<void*>(&mConnectionId), sizeof(mConnectionId), ucxx::Tag(ResponserTag));
            while (!sendRequest->isCompleted())
                ;
            sendRequest->checkError();

            auto recvTag = ucxx::Tag(mConnectionId << 32 | (RequesterTag & 0xFFFFFFFF));
            std::shared_ptr<ucxx::Request> recvRequest = mEndpoint->tagRecv(
                reinterpret_cast<void*>(&mConnectionIdInPeer), sizeof(mConnectionIdInPeer), recvTag, ucxx::TagMaskFull);
            while (!recvRequest->isCompleted())
                ;
            recvRequest->checkError();
        }
    }
    catch (std::exception const& e)
    {
        std::string error = std::string("Error in UcxConnection constructor for rank ")
            + std::to_string(mManager->getRank()) + ": " + e.what();
        TLLM_THROW(error);
    }

    mSendTagPrefix = mConnectionIdInPeer;
    mRecvTagPrefix = mConnectionId;

    TLLM_LOG_DEBUG(mManager->getRank(),
        "UcxConnection::UcxConnection, mConnectionId: %lu, mConnectionIdInPeer: %lu,fromRequester: %d", mConnectionId,
        mConnectionIdInPeer, mFromRequester);
}

UcxConnection::~UcxConnection()
{

    TLLM_LOG_DEBUG(mManager->getRank(),
        "UcxConnection::~UcxConnection, mConnectionId: %lu, mConnectionIdInPeer: %lu,fromRequester: %d", mConnectionId,
        mConnectionIdInPeer, mFromRequester);
    // TODO: how to close the endpoint safely?
}

void UcxConnection::sendConnectionId(DataContext const& ctx, void const* data, size_t size) const
{
    TLLM_LOG_DEBUG(mManager->getRank(),
        "start UcxConnection::sendConnectionId , mConnectionId: %lu, mConnectionIdInPeer: %lu,fromRequester: %d",
        mConnectionId, mConnectionIdInPeer, mFromRequester);

    std::promise<void> promise;

    std::future<void> future = promise.get_future();
    auto completionCallback = [&](ucs_status_t, ucxx::RequestCallbackUserData) -> void { promise.set_value(); };

    uint64_t tag = ((mSendTagPrefix & 0xFFFFFFFF) << 32)
        | static_cast<uint64_t>(tensorrt_llm::batch_manager::TransceiverTag::kID_TAG);
    std::vector<char> buffer(size + sizeof(mConnectionId));
    memcpy(buffer.data(), data, size);
    memcpy(buffer.data() + size, &mConnectionIdInPeer, sizeof(mConnectionIdInPeer));
    auto req = mEndpoint->tagSend(buffer.data(), buffer.size(), ucxx::Tag(tag), false, completionCallback);
    if (!req->isCompleted())
    {
        future.get();
    }
    TLLM_CHECK_WITH_INFO(req->isCompleted(), "sendConnectionId should be completed");
    req->checkError();
    TLLM_LOG_DEBUG(mManager->getRank(),
        "end UcxConnection::sendConnectionId , mConnectionId: %lu, mConnectionIdInPeer: %lu,fromRequester: %d",
        mConnectionId, mConnectionIdInPeer, mFromRequester);
}

void UcxConnection::send(DataContext const& ctx, void const* data, size_t size) const
{
    if (ctx.getTag() == tensorrt_llm::batch_manager::TransceiverTag::kID_TAG)
    {
        sendConnectionId(ctx, data, size);
        return;
    }
    uint64_t sendTag = ((mSendTagPrefix & 0xFFFFFFFF) << 32) | (static_cast<uint64_t>(ctx.getTag()) & (0xFFFFFFFF));
    // nvbugs/6104831 diagnostic: trace UCX send entry. Unlike NIXL, UCX has no
    // separate notifySyncMessage call — the tagSend completion IS the
    // synchronization. The wedge would manifest as a hung future.get() below;
    // the SEND-WAIT heartbeat records (connId, wireSendTag) every 5s while
    // waiting so we can pinpoint exactly which transfer is stuck.
    TLLM_LOG_INFO("[DIAG-UCX-SEND-ENTRY] connId=%lu peerConnId=%lu fromRequester=%d ctxTag=%d wireSendTag=%lu size=%zu",
        mConnectionId, mConnectionIdInPeer, mFromRequester, ctx.getTag(), sendTag, size);
    TLLM_LOG_DEBUG(mManager->getRank(),
        "start UcxConnection::send , mConnectionId: %lu, mConnectionIdInPeer: %lu,fromRequester: %d", mConnectionId,
        mConnectionIdInPeer, mFromRequester);

    TLLM_CHECK_WITH_INFO((mEndpoint), "sendBuffer called without established communicator channel.");
    std::promise<void> promise;
    std::future<void> future = promise.get_future();
    auto completionCallback = [&](ucs_status_t, ucxx::RequestCallbackUserData) -> void { promise.set_value(); };

    auto req = mEndpoint->tagSend(const_cast<void*>(data), size, ucxx::Tag(sendTag), false, completionCallback);
    if (!req->isCompleted())
    {
        // nvbugs/6104831 diagnostic: heartbeat-wrap the blocking future.get() so
        // a UCX wedge (tagSend never completes) produces periodic markers
        // naming the specific (connId, wireSendTag) pair stuck. Same blocking
        // semantics as plain future.get() — no timeout introduced; truly
        // wedged calls still wedge forever, just visibly.
        static constexpr int64_t kDiagHeartbeatSec = 5;
        std::uint64_t waitCycles = 0;
        while (future.wait_for(std::chrono::seconds(kDiagHeartbeatSec)) == std::future_status::timeout)
        {
            ++waitCycles;
            TLLM_LOG_INFO(
                "[DIAG-UCX-SEND-WAIT] connId=%lu peerConnId=%lu fromRequester=%d ctxTag=%d wireSendTag=%lu "
                "waitedSec=%lu",
                mConnectionId, mConnectionIdInPeer, mFromRequester, ctx.getTag(), sendTag,
                static_cast<unsigned long>(waitCycles * kDiagHeartbeatSec));
        }
        future.get();
    }
    TLLM_CHECK_WITH_INFO(req->isCompleted(), "send should be completed");
    // throw if there is error
    req->checkError();

    TLLM_LOG_INFO("[DIAG-UCX-SEND-DONE] connId=%lu peerConnId=%lu fromRequester=%d ctxTag=%d wireSendTag=%lu",
        mConnectionId, mConnectionIdInPeer, mFromRequester, ctx.getTag(), sendTag);
    TLLM_LOG_DEBUG(mManager->getRank(),
        "end UcxConnection::send , mConnectionId: %lu, mConnectionIdInPeer: %lu,fromRequester: %d", mConnectionId,
        mConnectionIdInPeer, mFromRequester);
}

void UcxConnection::recv(DataContext const& ctx, void* data, size_t size) const
{
    uint64_t recvTag = ((mRecvTagPrefix & 0xFFFFFFFF) << 32) | (static_cast<uint64_t>(ctx.getTag()) & (0xFFFFFFFF));
    // nvbugs/6104831 diagnostic: trace UCX recv entry. Pairs with peer's
    // SEND-ENTRY on matching wireSendTag (== wireRecvTag here, by ucxx::Tag
    // matching). If RECV-WAIT heartbeats appear without a peer SEND-ENTRY for
    // the same tag, the peer never tried to send; if both ENTRY log lines
    // appear but neither DONE, UCX is not progressing the transfer.
    TLLM_LOG_INFO("[DIAG-UCX-RECV-ENTRY] connId=%lu peerConnId=%lu fromRequester=%d ctxTag=%d wireRecvTag=%lu size=%zu",
        mConnectionId, mConnectionIdInPeer, mFromRequester, ctx.getTag(), recvTag, size);
    // Guard to ensure CUDA context is initialized for UCX ops
    TLLM_LOG_DEBUG(mManager->getRank(),
        "start UcxConnection::recv , mConnectionId: %lu, mConnectionIdInPeer: %lu,fromRequester: %d", mConnectionId,
        mConnectionIdInPeer, mFromRequester);
    TLLM_CHECK_WITH_INFO((mEndpoint), "recvBuffer called without established communicator channel.");
    std::promise<void> promise;
    std::future<void> future = promise.get_future();
    auto completionCallback = [&](ucs_status_t, ucxx::RequestCallbackUserData) -> void { promise.set_value(); };
    auto req = mEndpoint->tagRecv(data, size, ucxx::Tag(recvTag), ucxx::TagMaskFull, false, completionCallback);
    if (!req->isCompleted())
    {
        // nvbugs/6104831 diagnostic: heartbeat-wrap the blocking future.get()
        // so a UCX wedge (tagRecv never matches a send) produces periodic
        // markers naming the specific (connId, wireRecvTag) pair stuck. Same
        // blocking semantics; no timeout introduced.
        static constexpr int64_t kDiagHeartbeatSec = 5;
        std::uint64_t waitCycles = 0;
        while (future.wait_for(std::chrono::seconds(kDiagHeartbeatSec)) == std::future_status::timeout)
        {
            ++waitCycles;
            TLLM_LOG_INFO(
                "[DIAG-UCX-RECV-WAIT] connId=%lu peerConnId=%lu fromRequester=%d ctxTag=%d wireRecvTag=%lu "
                "waitedSec=%lu",
                mConnectionId, mConnectionIdInPeer, mFromRequester, ctx.getTag(), recvTag,
                static_cast<unsigned long>(waitCycles * kDiagHeartbeatSec));
        }
        future.get();
    }
    TLLM_CHECK_WITH_INFO(req->isCompleted(), "recv should be completed");
    // throw if there is error
    req->checkError();

    TLLM_LOG_INFO("[DIAG-UCX-RECV-DONE] connId=%lu peerConnId=%lu fromRequester=%d ctxTag=%d wireRecvTag=%lu",
        mConnectionId, mConnectionIdInPeer, mFromRequester, ctx.getTag(), recvTag);
    TLLM_LOG_DEBUG(mManager->getRank(),
        "end UcxConnection::recv , mConnectionId: %lu, mConnectionIdInPeer: %lu,fromRequester: %d", mConnectionId,
        mConnectionIdInPeer, mFromRequester);
}

} // namespace tensorrt_llm::executor::kv_cache

#endif
