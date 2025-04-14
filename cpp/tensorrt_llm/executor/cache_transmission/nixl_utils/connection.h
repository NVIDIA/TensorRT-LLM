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

#pragma once

#include "nixl.h"
#include "tensorrt_llm/executor/cacheCommunicator.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace tensorrt_llm::executor::kv_cache
{

// Forward declaration
class ConnectionManager;
class NixlConnectionManager;

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
#endif

extern "C"
{
    [[nodiscard]] std::unique_ptr<ConnectionManager> makeNixlConnectionManager(mpi::MpiComm const* comm);
}

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

class NixlConnection : public Connection
{
public:
    explicit NixlConnection(
        nixlAgent* agent, std::string const& remote_agent, int rank, int device_id, NixlConnectionManager* manager);
    ~NixlConnection() override = default;

    void send(DataContext const& ctx, void const* data, size_t size) const override;
    void recv(DataContext const& ctx, void* data, size_t size) const override;

    void sendRequestInfo(std::string serializedInfo, runtime::ITensor::SharedPtr recvBuffer);
    std::string recvRequestInfo(std::string& retBufferDescStr);

    bool isThreadSafe() const noexcept override
    {
        return true;
    }

    void setRecvBufferInfo(uint64_t request_id, std::string basicDescStr);
    std::string getLocalMD() const;
    void setRemoteMD(std::string const& remote_metadata);

    // Add the declaration for check_xfer_done
    bool check_xfer_done(std::string const& remote_agent, std::string const& mTagStr) const;
    int getRank() const;

private:
    nixlAgent* agent_;
    std::string remote_agent_;
    int rank_;
    int device_id_;
    std::unordered_map<uint64_t, std::string> mRequestToRecvBufferInfo;
    // Add pointer to connection manager
    NixlConnectionManager* mManager;
};

class NixlConnectionManager : public ConnectionManager
{
    // Make NixlConnection a friend to access private members
    friend class NixlConnection;

public:
    NixlConnectionManager(tensorrt_llm::mpi::MpiComm const* comm);
    ~NixlConnectionManager() override;

    Connection const* recvConnect(DataContext const& ctx, void* data, size_t size) override;
    Connection const* recvRequestInfo(std::string& serializedInfo, std::string& retBufferDescStr);
    void sendRequestInfo(int rank, std::string serializedInfo, runtime::ITensor::SharedPtr recvBuffer);
    std::vector<Connection const*> getConnections(CommState const& state) override;
    void setRecvBufferInfo(int rank, uint64_t request_id, std::string basicDescStr);
    void setRemoteMD(std::string const& remote_metadata, int rank);
    std::string getLocalMD() const;
    [[nodiscard]] CommState const& getCommState() const override;

    // New API for registering multiple buffers at once
    struct BufferInfo
    {
        void* data;
        size_t size;
        int deviceId; // CUDA device ID where the memory was allocated
    };

    void registerBuffers(std::vector<BufferInfo> const& buffers);
    void deregisterBuffers();

    // Make updateNotifications public so NixlConnection can access it
    void updateNotifications();

private:
    mpi::MpiComm const* mComm;
    // TODO: implement comm state
    CommState mCommState;
    nixlAgent* agent_;
    nixlBackendH* backend_;
    std::string agent_name_;
    std::map<int, std::shared_ptr<NixlConnection>> connections_;
    std::map<std::string, std::shared_ptr<NixlConnection>> connections_by_name_;
    // Store the registration list for later deregistration
    nixl_reg_dlist_t registration_list_;
    bool has_registered_buffers_{false};
    int device_id_;
    // Add database for unhandled notifications per agent
    std::map<std::string, std::vector<std::string>> mUnhandledNotifications;
    // Add mutex to protect the notifications database
    mutable std::mutex mNotificationMutex;
};

} // namespace tensorrt_llm::executor::kv_cache
