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

#include "connection.h"
#include "tensorrt_llm/common/logger.h"
#include <map>
#include <memory>
#include <sstream>
#include <string>

namespace tensorrt_llm::executor::kv_cache
{

NixlConnection::NixlConnection(
    nixlAgent* agent, std::string const& remote_agent, int rank, int device_id, NixlConnectionManager* manager)
    : agent_(agent)
    , remote_agent_(remote_agent)
    , rank_(rank)
    , device_id_(device_id)
    , mManager(manager)
{
    TLLM_LOG_DEBUG("NIXL connection created with remote agent: %s from rank %d", remote_agent_.c_str(), rank_);
}

void NixlConnection::setRecvBufferInfo(uint64_t request_id, std::string basicDescStr)
{
    mRequestToRecvBufferInfo[request_id] = std::move(basicDescStr);
}

void NixlConnection::sendRequestInfo(std::string serializedInfo, runtime::ITensor::SharedPtr recvBuffer)
{
    // Serialize the buffer descriptor
    nixlBasicDesc bufferDesc(reinterpret_cast<uintptr_t>(recvBuffer->data()), recvBuffer->getSizeInBytes(), device_id_);
    TLLM_LOG_DEBUG(rank_, "Sent buffer descriptor (addr: %p, size: %zu, devid: %d)", (void*) bufferDesc.addr,
        bufferDesc.len, bufferDesc.devId);
    std::string bufferDescStr = bufferDesc.serialize();

    // Create a notification message that includes both the request info and buffer descriptor
    std::string notifMsg = serializedInfo + "$" + bufferDescStr; // TODO: check with TRTLLM team

    // Use genNotif to send the request info and buffer descriptor
    agent_->genNotif(remote_agent_.c_str(), notifMsg);
}

void NixlConnection::send(DataContext const& ctx, void const* data, size_t size) const
{
    // Create descriptor list for transfer
    TLLM_LOG_DEBUG("NIXL: enter send data to agent %s", remote_agent_.c_str());
    nixl_xfer_dlist_t desc_list(VRAM_SEG);
    auto mTag = ctx.getTag();
    auto request_id = mTag;
    auto remote_agent = remote_agent_;

    // Convert mTag to string
    std::string mTagStr = std::to_string(mTag);

    // For now, we are using a dummy descriptor list
    nixl_xfer_dlist_t remote_desc_list(VRAM_SEG);

    TLLM_LOG_DEBUG("NIXL: send data to agent %s, request_id %d", remote_agent_.c_str(), request_id);
    auto bufferDescStr = mRequestToRecvBufferInfo.at(request_id);
    TLLM_LOG_DEBUG("NIXL: bufferDescStr %s", bufferDescStr.c_str());

    auto bufferDesc = nixlBasicDesc(bufferDescStr);
    bufferDesc.len = size; // tmp
    remote_desc_list.addDesc(bufferDesc);

    if (TLLM_UNLIKELY(remote_desc_list.descCount() == 0))
    {
        throw std::runtime_error("NIXL remote descriptor list is empty");
    }

    // Create and add local descriptor with correct device ID
    nixlBasicDesc desc(reinterpret_cast<uintptr_t>(data), size, device_id_);
    desc_list.addDesc(desc);
    // Create transfer request
    nixlXferReqH* req_handle = nullptr;
    nixl_status_t status = agent_->createXferReq(NIXL_WRITE, desc_list, remote_desc_list, remote_agent, req_handle);
    if (TLLM_UNLIKELY(status != NIXL_SUCCESS))
    {
        throw std::runtime_error("NIXL create transfer request failed");
    }
    // Submit transfer request
    nixl_status_t xfer_state = agent_->postXferReq(req_handle);
    while (xfer_state != NIXL_SUCCESS)
    {
        xfer_state = agent_->getXferStatus(req_handle);
        if (xfer_state != NIXL_SUCCESS && xfer_state != NIXL_IN_PROG)
        {
            throw std::runtime_error("NIXL post transfer request failed ");
        }
    }

    agent_->genNotif(remote_agent.c_str(), mTagStr.c_str());
    agent_->releaseXferReq(req_handle);
    if (TLLM_UNLIKELY(xfer_state != NIXL_SUCCESS))
    {
        throw std::runtime_error("NIXL send failed");
    }
    TLLM_LOG_DEBUG("NIXL: exit send data to agent %s", remote_agent_.c_str());
}

bool NixlConnection::check_xfer_done(std::string const& remote_agent, std::string const& mTagStr) const
{
    // Update notifications through manager
    mManager->updateNotifications();

    // Check in manager's unhandled notifications
    std::lock_guard<std::mutex> lock(mManager->mNotificationMutex);
    auto it = mManager->mUnhandledNotifications.find(remote_agent);
    if (it != mManager->mUnhandledNotifications.end())
    {
        auto& notif_list = it->second;
        auto msg_it = std::find(notif_list.begin(), notif_list.end(), mTagStr);
        if (msg_it != notif_list.end())
        {
            notif_list.erase(msg_it);
            if (notif_list.empty())
            {
                mManager->mUnhandledNotifications.erase(it);
            }
            return true;
        }
    }
    return false;
}

void NixlConnection::recv(DataContext const& ctx, void* data, size_t size) const
{
    TLLM_LOG_DEBUG("NIXL: enter recv data from agent %s", remote_agent_.c_str());
    auto mTag = ctx.getTag();
    auto remote_agent = remote_agent_;

    // Convert mTag to string
    std::string mTagStr = std::to_string(mTag);
    // Loop until the desired notification is received
    while (!check_xfer_done(remote_agent, mTagStr))
    {
        continue;
    }
    TLLM_LOG_DEBUG("NIXL: exit recv data from agent %s", remote_agent_.c_str());
}

void NixlConnection::setRemoteMD(std::string const& remote_metadata)
{
    std::string remoteMD;
    nixl_status_t status = agent_->loadRemoteMD(remote_metadata, remoteMD);
    if (status != NIXL_SUCCESS)
    {
        TLLM_LOG_ERROR("Failed to load remote metadata for agent: %s", remote_metadata.c_str());
    }
    else
    {
        TLLM_LOG_DEBUG("Loaded remote metadata for agent: %s", remoteMD.c_str());
    }
}

int NixlConnection::getRank() const
{
    return rank_;
}

NixlConnectionManager::NixlConnectionManager(tensorrt_llm::mpi::MpiComm const* comm)
    : mComm(comm)
    , registration_list_(VRAM_SEG)
{
    nixlAgentConfig config(true); // Use progress thread
    int rank = mComm->getRank();
    agent_name_ = "Nixl_agent_" + std::to_string(rank);

    agent_ = new nixlAgent(agent_name_, config);
    if (!agent_)
    {
        throw std::runtime_error("Failed to create NIXL agent");
    }
    nixl_b_params_t init1;
    nixl_mem_list_t mems1;
    nixl_status_t status = agent_->getPluginParams("UCX", mems1, init1);

    status = agent_->createBackend("UCX", init1, backend_);
    if (status != NIXL_SUCCESS)
    {
        throw std::runtime_error("Failed to create NIXL backend");
    }
    registration_list_ = nixl_reg_dlist_t(VRAM_SEG);
}

NixlConnectionManager::~NixlConnectionManager()
{
    // Deregister buffers if any were registered
    if (has_registered_buffers_)
    {
        try
        {
            deregisterBuffers();
        }
        catch (std::exception const& e)
        {
            TLLM_LOG_ERROR("Failed to deregister buffers during cleanup: %s", e.what());
        }
    }

    if (agent_)
    {
        delete agent_;
        agent_ = nullptr;
    }
}

Connection const* NixlConnectionManager::recvRequestInfo(std::string& retSerializedInfo, std::string& retBufferDescStr)
{
    bool received = false;
    std::string serializedInfo;
    TLLM_LOG_DEBUG("NIXL: recvRequestInfo: waiting for notification");
    while (!received)
    {
        // Update notifications from agent
        updateNotifications();

        // Check in unhandled notifications
        std::unique_lock<std::mutex> lock(mNotificationMutex);
        for (auto& [agent, notif_list] : mUnhandledNotifications)
        {
            if (!notif_list.empty())
            {
                auto notif = notif_list.front();
                notif_list.erase(notif_list.begin());
                if (notif_list.empty())
                {
                    mUnhandledNotifications.erase(agent);
                }

                // Parse notification
                std::istringstream iss(notif);
                std::string serializedInfo, bufferDescStr;
                std::getline(iss, serializedInfo, '$'); // TODO: check with TRTLLM team
                std::getline(iss, bufferDescStr);
                retBufferDescStr = bufferDescStr;
                retSerializedInfo = serializedInfo;

                TLLM_LOG_DEBUG("NIXL: return connection for agent %s", agent.c_str());
                auto it_connection = connections_by_name_.find(agent);
                if (it_connection != connections_by_name_.end())
                {
                    Connection const* connection = it_connection->second.get();
                    // TLLM_LOG_DEBUG("NIXL: get connection for agent %s with rank %d", agent.c_str(),
                    // connection->getRank());
                    return connection;
                }
                else
                {
                    throw std::runtime_error("NIXL: connection not found for agent " + agent);
                }
            }
        }
        lock.unlock();
    }
    return nullptr;
}

Connection const* NixlConnectionManager::recvConnect(DataContext const& ctx, void* data, size_t size)
{
    // Return a pointer to the NixlConnection object
    return connections_.at(ctx.getTag()).get();
}

std::vector<Connection const*> NixlConnectionManager::getConnections(CommState const& state)
{
    std::vector<Connection const*> connections;
    for (auto& [rank, connection] : connections_)
    {
        connections.push_back(connection.get());
    }
    return connections;
}

std::unique_ptr<ConnectionManager> makeNixlConnectionManager(mpi::MpiComm const* comm)
{
    return std::make_unique<NixlConnectionManager>(comm);
}

void NixlConnectionManager::setRecvBufferInfo(int rank, uint64_t request_id, std::string basicDescStr)
{
    // Assuming you want to set the buffer info for all connections
    // TODO: chose rank
    connections_.at(rank)->setRecvBufferInfo(request_id, basicDescStr);
    TLLM_LOG_DEBUG("NIXL: set recv buffer info for rank %d, request_id %d, basicDescStr %s", rank, request_id,
        basicDescStr.c_str());
}

void NixlConnectionManager::sendRequestInfo(
    int rank, std::string serializedInfo, runtime::ITensor::SharedPtr recvBuffer)
{
    connections_.at(rank)->sendRequestInfo(serializedInfo, recvBuffer);
}

void NixlConnectionManager::setRemoteMD(std::string const& remote_metadata, int rank)
{
    TLLM_LOG_DEBUG("NIXL: enter set remote metadata for agent %s", agent_name_.c_str());
    std::string remoteMD;
    nixl_status_t status = agent_->loadRemoteMD(remote_metadata, remoteMD);
    if (TLLM_UNLIKELY(remoteMD.empty()))
    {
        TLLM_LOG_ERROR("Failed to load remote metadata for agent %s: %s", agent_name_.c_str(), remoteMD.c_str());
    }
    else
    {
        TLLM_LOG_DEBUG("Loaded remote metadata for agent %s: %s", agent_name_.c_str(), remoteMD.c_str());
    }
    // TODO: check if rank already exists

    auto connection = std::make_shared<NixlConnection>(agent_, remoteMD, rank, device_id_, this);
    connections_.insert({rank, connection});
    connections_by_name_.insert({remoteMD, connection});

    TLLM_LOG_DEBUG(
        "NIXL: save in both maps remote metadata for agent %s: %s", remoteMD.c_str(), remote_metadata.c_str());
}

std::string NixlConnectionManager::getLocalMD() const
{
    std::string local_md_blob;
    nixl_status_t status = agent_->getLocalMD(local_md_blob);
    if (status != NIXL_SUCCESS)
    {
        throw std::runtime_error("Failed to get local metadata");
    }
    return local_md_blob;
}

void NixlConnectionManager::registerBuffers(std::vector<BufferInfo> const& buffers)
{
    TLLM_CHECK(agent_ != nullptr);
    TLLM_CHECK(backend_ != nullptr);

    auto device_id = buffers[0].deviceId;
    device_id_ = device_id;

    // Create a new descriptor list for registration
    auto current_registration_list = nixl_reg_dlist_t(VRAM_SEG);

    // Add each buffer to the registration list
    for (auto const& buffer : buffers)
    {
        nixlBlobDesc desc(reinterpret_cast<uintptr_t>(buffer.data), buffer.size, buffer.deviceId);
        current_registration_list.addDesc(desc);
        registration_list_.addDesc(desc);
    }

    // Register the descriptor list with the NIXL agent
    nixl_opt_args_t extra_params;
    extra_params.backends.push_back(backend_);
    nixl_status_t status = agent_->registerMem(current_registration_list, &extra_params);
    if (TLLM_UNLIKELY(status != NIXL_SUCCESS))
    {
        throw std::runtime_error("Failed to register memory with NIXL agent " + std::to_string(status));
    }
    has_registered_buffers_ = true;
}

void NixlConnectionManager::deregisterBuffers()
{
    if (TLLM_UNLIKELY(!has_registered_buffers_))
    {
        return;
    }

    TLLM_CHECK(agent_ != nullptr);
    TLLM_CHECK(backend_ != nullptr);

    // Deregister the memory through the agent
    nixl_opt_args_t extra_params;
    extra_params.backends.push_back(backend_);
    nixl_status_t status = agent_->deregisterMem(registration_list_, &extra_params);
    if (TLLM_UNLIKELY(status != NIXL_SUCCESS))
    {
        throw std::runtime_error("Failed to deregister memory with NIXL agent");
    }
    has_registered_buffers_ = false;
}

void NixlConnectionManager::updateNotifications()
{
    std::lock_guard<std::mutex> lock(mNotificationMutex);
    nixl_notifs_t notif_map;
    nixl_status_t status = agent_->getNotifs(notif_map);
    if (status != NIXL_SUCCESS)
    {
        TLLM_LOG_ERROR("Failed to get notifications from agent");
        return;
    }

    // Merge new notifications with existing ones
    for (auto const& [agent, notifs] : notif_map)
    {
        auto& existing_notifs = mUnhandledNotifications[agent];
        existing_notifs.insert(existing_notifs.end(), notifs.begin(), notifs.end());
    }
}

CommState const& NixlConnectionManager::getCommState() const
{
    // TODO: implement
    return mCommState;
}

} // namespace tensorrt_llm::executor::kv_cache
