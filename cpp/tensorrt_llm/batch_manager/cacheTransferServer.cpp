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

#include "cacheTransferServer.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/executor/dataTransceiverState.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"

namespace tensorrt_llm::batch_manager
{

TransferTagServer::TransferTagServer()
{
    mThread = std::thread(&TransferTagServer::loop, this);
    pthread_setname_np(mThread.native_handle(), "TransferTagServer");
}

TransferTagServer::~TransferTagServer()
{
    stop();
    if (mThread.joinable())
    {
        try
        {
            mThread.join();
        }
        catch (std::exception const& e)
        {
            TLLM_LOG_ERROR("TransferTagServer destructor: failed to join thread: %s", e.what());
        }
    }
}

void TransferTagServer::loop()
{
    mContext = std::make_unique<zmq::context_t>(1);
    mSocket = std::make_unique<zmq::socket_t>(*mContext, zmq::socket_type::router);

    mSocket->bind("tcp://*:0");
    mSocket->set(zmq::sockopt::linger, 0);
    std::string boundEndpoint = mSocket->get(zmq::sockopt::last_endpoint);
    size_t portPos = boundEndpoint.find_last_of(':');
    std::string port = boundEndpoint.substr(portPos + 1);
    mEndpoint = "tcp://" + executor::getLocalIp(mpi::MpiComm::world().getRank()) + ":" + port;
    TLLM_LOG_INFO("[Rank %d] Cache transfer server started on %s", mpi::MpiComm::world().getRank(), mEndpoint.c_str());

    {
        std::lock_guard<std::mutex> lock(mReadyMutex);
        mReady = true;
    }
    mReadyCv.notify_all();

    while (mRunning)
    {
        try
        {
            zmq::pollitem_t items[] = {{static_cast<void*>(*mSocket), 0, ZMQ_POLLIN, 0}};
            zmq::poll(items, 1, 100);

            if (items[0].revents & ZMQ_POLLIN)
            {
                handleRequest();
            }
        }
        catch (std::exception const& e)
        {
            if (mRunning)
            {
                TLLM_LOG_ERROR("TransferTagServer loop exception: %s", e.what());
            }
        }
        catch (...)
        {
            if (mRunning)
            {
                TLLM_LOG_ERROR("TransferTagServer loop unknown exception");
            }
        }
    }

    try
    {
        mSocket->set(zmq::sockopt::linger, 0);
        mSocket.reset();
        mContext.reset();
    }
    catch (std::exception const& e)
    {
        TLLM_LOG_ERROR("TransferTagServer: failed to close socket: %s", e.what());
    }
}

void TransferTagServer::handleRequest()
{
    zmq::message_t identity;
    zmq::message_t request;
    auto result = mSocket->recv(identity, zmq::recv_flags::none);
    TLLM_CHECK_WITH_INFO(result.has_value(), "Failed to receive identity");
    TLLM_CHECK_WITH_INFO(mSocket->get(zmq::sockopt::rcvmore), "Expected more messages");

    result = mSocket->recv(request, zmq::recv_flags::none);
    TLLM_CHECK_WITH_INFO(result.has_value(), "Failed to receive request");
    TLLM_CHECK_WITH_INFO(request.size() == sizeof(TransferTagRequest), "Invalid request size");

    TransferTagRequest req = *reinterpret_cast<TransferTagRequest*>(request.data());
    zmq::message_t response;

    if (req.type == TransferTagRequestType::kGetTransferTag)
    {
        auto key = std::make_pair(
            req.payload.getTransferTag.receiverTransferId, req.payload.getTransferTag.receiverServerUuid);
        auto it = mTransferTagRefCount.find(key);
        uint64_t transferTag;
        if (it == mTransferTagRefCount.end())
        {
            transferTag = TransferTagGenerator::get();
            it = mTransferTagRefCount
                     .emplace(key, std::make_pair(req.payload.getTransferTag.expectedRefCount, transferTag))
                     .first;
        }
        else
        {
            transferTag = it->second.second;
        }
        response.rebuild(&transferTag, sizeof(transferTag));
    }
    else if (req.type == TransferTagRequestType::kReleaseTransferTag)
    {
        auto key = std::make_pair(
            req.payload.releaseTransferTag.receiverTransferId, req.payload.releaseTransferTag.receiverServerUuid);
        auto it = mTransferTagRefCount.find(key);
        TLLM_CHECK_WITH_INFO(it != mTransferTagRefCount.end(), "Unique ID not found");
        it->second.first--;
        if (it->second.first == 0)
        {
            TransferTagGenerator::release(it->second.second);
            mTransferTagRefCount.erase(it);
        }
        response.rebuild(0);
    }

    mSocket->send(identity, zmq::send_flags::sndmore);
    mSocket->send(response, zmq::send_flags::none);
}

} // namespace tensorrt_llm::batch_manager
