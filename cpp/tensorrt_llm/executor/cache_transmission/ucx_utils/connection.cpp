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

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/executor/cache_transmission/ucx_utils/connection.h"

namespace tensorrt_llm::executor::kv_cache
{

UcxConnection::UcxConnection(
    uint64_t connectionId, std::shared_ptr<ucxx::Endpoint> endpoint, UcxConnectionManager* manager)
    : mConnectionId(connectionId)
    , mLocalGID(manager->getLocalGID())
    , mRemoteGID(std::numeric_limits<uint64_t>::max())
    , mEndpoint(endpoint)
    , mManager(manager)
{
    if (mEndpoint)
    {
        initializeEndpointTag();
    }
}

// void UcxConnection::initialize(UcxConnectionManager* manager)
// {
//     std::cout << "UcxConnection::initialize" << std::endl;
//     mManager = manager;
//     if (mManager)
//     {
//         mLocalGID = mManager->getLocalGID();
//         TLLM_LOG_DEBUG("UcxConnection::initialize | rank %d | local gid: %lu", mManager->getRank(), mLocalGID);
//     }
// }

void UcxConnection::sendGID()
{
    if (mLocalGID == std::numeric_limits<uint64_t>::max())
    {
        throw std::runtime_error("UcxConnection::sendGID | mLocalGID is not set");
    }
    std::shared_ptr<ucxx::Request> request
        = mEndpoint->streamSend(reinterpret_cast<void*>(&mLocalGID), sizeof(mLocalGID), false);
    while (request->isCompleted() == false)
        ;
    request->checkError();
}

void UcxConnection::initializeEndpointTag(int maxTryTimes)
{
    // [FIXME] IP exchange seems to be more robust to ensure tag establishment,
    // i.e. different peers can use the same port number to connect with self worker
    // which results in identical tag if only self / peer port is used.

    // knowing that ucxx::Tag is uint64_t
    ucxx::Tag localPort{0};
    ucxx::Tag remotePort{0};
    char lIpStr[INET6_ADDRSTRLEN];
    char rIpStr[INET6_ADDRSTRLEN];
    char portStr[INET6_ADDRSTRLEN];

    ucp_ep_attr_t ep_attr;
    ep_attr.field_mask = UCP_EP_ATTR_FIELD_LOCAL_SOCKADDR | UCP_EP_ATTR_FIELD_REMOTE_SOCKADDR;

    ucs_status_t status = ucp_ep_query(mEndpoint->getHandle(), &ep_attr);
    if (status == UCS_OK)
    {

        std::cout << " initializeEndpointTag" << std::endl;
        ucxx::utils::sockaddr_get_ip_port_str(&ep_attr.remote_sockaddr, rIpStr, portStr, INET6_ADDRSTRLEN);
        remotePort = static_cast<ucxx::Tag>(std::stoull(portStr));

        ucxx::utils::sockaddr_get_ip_port_str(&ep_attr.local_sockaddr, lIpStr, portStr, INET6_ADDRSTRLEN);
        localPort = static_cast<ucxx::Tag>(std::stoull(portStr));

        // Network port value is defined to fit in 16 bit.
        // sendTag format : [remotePort localPort remotIP]
        mSendTag = static_cast<ucxx::Tag>((localPort << (16 + 32)) | (remotePort << 32) | std::stoull(lIpStr));
        TLLM_LOG_DEBUG("UcxConnection::initializeEndpointTag | localGID %d | sendTag: %lu", mLocalGID, mSendTag);
        mRecvTag = static_cast<ucxx::Tag>((remotePort << (16 + 32)) | (localPort << 32) | std::stoull(rIpStr));
        TLLM_LOG_DEBUG("UcxConnection::initializeEndpointTag | localGID %d | recvTag: %lu", mLocalGID, mRecvTag);
    }
    else
    {
        // [FIXME] better message
        if (status == UCS_ERR_NOT_CONNECTED && maxTryTimes > 0)
        {
            TLLM_LOG_WARNING("UCX connection has not been established yet. wait 100 ms before retrying. maxTryTimes:%d",
                maxTryTimes);
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            initializeEndpointTag(maxTryTimes - 1);
        }
        else
        {
            TLLM_LOG_WARNING("UCX data transceiver is not created by connecting to a socket address.");
        }
    }
}

void UcxConnection::send(DataContext const& ctx, void const* data, size_t size) const
{
    // Guard to ensure CUDA context is initialized for UCX ops
    TLLM_CUDA_CHECK(cudaFree(0));
    TLLM_CHECK_WITH_INFO((mEndpoint), "sendBuffer called without established communicator channel.");
    TLLM_LOG_DEBUG("UcxConnection::send | rank %d | sendTag: %lu | remote gid: %lu", mLocalGID, mSendTag, mRemoteGID);
    auto completionCallback = [this](ucs_status_t, ucxx::RequestCallbackUserData) -> void { mCv.notify_all(); };
    auto req = mEndpoint->tagSend(const_cast<void*>(data), size, mSendTag, false, completionCallback);
    std::unique_lock<std::mutex> lk(mMtx);
    mCv.wait(lk, [&req]() { return req->isCompleted(); });
    TLLM_LOG_DEBUG("UcxConnection::send | rank %d | sendTag: %lu | remote gid: %lu", mLocalGID, mSendTag, mRemoteGID);
    // throw if there is error
    req->checkError();
}

void UcxConnection::recv(DataContext const& ctx, void* data, size_t size) const
{
    // Guard to ensure CUDA context is initialized for UCX ops
    TLLM_CUDA_CHECK(cudaFree(0));
    TLLM_CHECK_WITH_INFO((mEndpoint), "recvBuffer called without established communicator channel.");
    TLLM_LOG_DEBUG("UcxConnection::recv | rank %d | tagReceiveFrom: %lu", mLocalGID, mRecvTag);
    auto completionCallback = [this](ucs_status_t, ucxx::RequestCallbackUserData) -> void { mCv.notify_all(); };
    auto req = mEndpoint->tagRecv(data, size, mRecvTag, ucxx::TagMaskFull, false, completionCallback);
    std::unique_lock<std::mutex> lk(mMtx);
    mCv.wait(lk, [&req]() { return req->isCompleted(); });
    TLLM_LOG_DEBUG("UcxConnection::recv | rank %d | recvTag: %lu | remote gid: %lu", mLocalGID, mRecvTag, mRemoteGID);
    // throw if there is error
    req->checkError();
}


} // namespace tensorrt_llm::executor::kv_cache

#endif
