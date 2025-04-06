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

#include "tensorrt_llm/batch_manager/dataTransceiverImpl.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/executor/cache_transmission/ucx_utils/connection.h"

namespace tensorrt_llm::executor::kv_cache
{

UcxConnection::UcxConnection(ConnectionIdType connectionId, std::shared_ptr<ucxx::Endpoint> endpoint,
    UcxConnectionManager* manager, bool fromRequester)
    : mConnectionId(connectionId)
    , mEndpoint(std::move(endpoint))
    , mManager(manager)
    , mFromRequester(fromRequester)
{

    if (mFromRequester)
    {
        std::shared_ptr<ucxx::Request> request
            = mEndpoint->streamSend(reinterpret_cast<void*>(&mConnectionId), sizeof(mConnectionId), false);
        while (!request->isCompleted())
            ;
        request->checkError();

        request
            = mEndpoint->streamRecv(reinterpret_cast<void*>(&mConnectionIdInPeer), sizeof(mConnectionIdInPeer), false);
        while (!request->isCompleted())
            ;
        request->checkError();
    }
    else
    {
        std::shared_ptr<ucxx::Request> request
            = mEndpoint->streamRecv(reinterpret_cast<void*>(&mConnectionIdInPeer), sizeof(mConnectionIdInPeer), false);
        while (!request->isCompleted())
            ;
        request->checkError();
        request = mEndpoint->streamSend(reinterpret_cast<void*>(&mConnectionId), sizeof(mConnectionId), false);
        while (!request->isCompleted())
            ;
        request->checkError();
    }

    mSendTagPrefix = mConnectionIdInPeer;
    mRecvTagPrefix = mConnectionId;

    TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(),
        "UcxConnection::UcxConnection, mConnectionId: %lu, mConnectionIdInPeer: %lu,fromRequester: %d", mConnectionId,
        mConnectionIdInPeer, mFromRequester);
}

UcxConnection::~UcxConnection()
{

    TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(),
        "UcxConnection::~UcxConnection, mConnectionId: %lu, mConnectionIdInPeer: %lu,fromRequester: %d", mConnectionId,
        mConnectionIdInPeer, mFromRequester);
    // TODO: how to close the endpoint safely?
}

void UcxConnection::sendConnectionId(DataContext const& ctx, void const* data, size_t size) const
{
    TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(),
        "start UcxConnection::sendConnectionId , mConnectionId: %lu, mConnectionIdInPeer: %lu,fromRequester: %d",
        mConnectionId, mConnectionIdInPeer, mFromRequester);

    std::promise<void> promise;

    std::future<void> future = promise.get_future();
    auto completionCallback = [&](ucs_status_t, ucxx::RequestCallbackUserData) -> void { promise.set_value(); };

    uint64_t tag
        = ((mSendTagPrefix & 0xFFFFFFFF) << 32) | static_cast<uint64_t>(batch_manager::TransceiverTag::kID_TAG);
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
    TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(),
        "end UcxConnection::sendConnectionId , mConnectionId: %lu, mConnectionIdInPeer: %lu,fromRequester: %d",
        mConnectionId, mConnectionIdInPeer, mFromRequester);
}

void UcxConnection::send(DataContext const& ctx, void const* data, size_t size) const
{
    if (ctx.getTag() == batch_manager::TransceiverTag::kID_TAG)
    {
        sendConnectionId(ctx, data, size);
        return;
    }
    TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(),
        "start UcxConnection::send , mConnectionId: %lu, mConnectionIdInPeer: %lu,fromRequester: %d", mConnectionId,
        mConnectionIdInPeer, mFromRequester);

    TLLM_CHECK_WITH_INFO((mEndpoint), "sendBuffer called without established communicator channel.");
    std::promise<void> promise;
    std::future<void> future = promise.get_future();
    auto completionCallback = [&](ucs_status_t, ucxx::RequestCallbackUserData) -> void { promise.set_value(); };
    uint64_t sendTag = ((mSendTagPrefix & 0xFFFFFFFF) << 32) | (static_cast<uint64_t>(ctx.getTag()) & (0xFFFFFFFF));

    auto req = mEndpoint->tagSend(const_cast<void*>(data), size, ucxx::Tag(sendTag), false, completionCallback);
    if (!req->isCompleted())
    {
        future.get();
    }
    TLLM_CHECK_WITH_INFO(req->isCompleted(), "send should be completed");
    // throw if there is error
    req->checkError();
    TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(),
        "end UcxConnection::send , mConnectionId: %lu, mConnectionIdInPeer: %lu,fromRequester: %d", mConnectionId,
        mConnectionIdInPeer, mFromRequester);
}

void UcxConnection::recv(DataContext const& ctx, void* data, size_t size) const
{
    // Guard to ensure CUDA context is initialized for UCX ops
    TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(),
        "start UcxConnection::recv , mConnectionId: %lu, mConnectionIdInPeer: %lu,fromRequester: %d", mConnectionId,
        mConnectionIdInPeer, mFromRequester);
    TLLM_CHECK_WITH_INFO((mEndpoint), "recvBuffer called without established communicator channel.");
    std::promise<void> promise;
    std::future<void> future = promise.get_future();
    auto completionCallback = [&](ucs_status_t, ucxx::RequestCallbackUserData) -> void { promise.set_value(); };
    uint64_t recvTag = ((mRecvTagPrefix & 0xFFFFFFFF) << 32) | (static_cast<uint64_t>(ctx.getTag()) & (0xFFFFFFFF));
    auto req = mEndpoint->tagRecv(data, size, ucxx::Tag(recvTag), ucxx::TagMaskFull, false, completionCallback);
    if (!req->isCompleted())
    {
        future.get();
    }
    TLLM_CHECK_WITH_INFO(req->isCompleted(), "recv should be completed");
    // throw if there is error
    req->checkError();
    TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(),
        "end UcxConnection::recv , mConnectionId: %lu, mConnectionIdInPeer: %lu,fromRequester: %d", mConnectionId,
        mConnectionIdInPeer, mFromRequester);
}

} // namespace tensorrt_llm::executor::kv_cache

#endif
