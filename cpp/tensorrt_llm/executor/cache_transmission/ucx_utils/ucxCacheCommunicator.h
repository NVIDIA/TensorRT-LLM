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

#include "tensorrt_llm/executor/cacheCommunicator.h"
#include "tensorrt_llm/executor/dataTransceiverState.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h" //TODO: remove when progressing to standalone UCX stack
#include "ucxx/api.h"
#include "ucxx/utils/sockaddr.h"
#include "ucxx/utils/ucx.h"
#include <cstdint>
#include <future>
#if __linux__
#include <arpa/inet.h>
#include <ifaddrs.h>
#endif
#include "tensorrt_llm/executor/cache_transmission/ucx_utils/connection.h"

#include <future>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <zmq.hpp>

namespace tensorrt_llm::executor::kv_cache
{

class UcxConnectionManager : public ConnectionManager, public std::enable_shared_from_this<UcxConnectionManager>
{
private:
    std::shared_ptr<ucxx::Context> mUcxCtx;
    std::vector<std::shared_ptr<ucxx::Worker>> mWorkersPool;
    std::string mWorkerAddress;
    std::map<UcxConnection::ConnectionIdType, std::shared_ptr<UcxConnection>> mConnections;
    std::map<UcxConnection::ConnectionIdType, std::future<void>> mConnectionFutures;
    std::mutex mConnectionsMutex;
    std::mutex mConnectionFuturesMutex;
    std::unordered_map<std::string, uint64_t> mAddressToConnectionId;
    std::mutex mAddressToConnectionIdMutex;
    CommState mCommState;
    int mDevice;
    int mRank;
    int mWorldSize;
    std::atomic<UcxConnection::ConnectionIdType> mConnectionIdCounter{1};
    zmq::context_t mZmqContext;
    zmq::socket_t mZmqRepSocket;
    std::string mZmqRepEndpoint;
    std::thread mZmqRepThread;

    UcxConnection::ConnectionIdType getNewConnectionId(std::shared_ptr<ucxx::Endpoint> const& newEp);
    UcxConnection::ConnectionIdType addConnection(std::string const& ip, uint16_t port);

public:
    explicit UcxConnectionManager();
    ~UcxConnectionManager();

    // Factory function
    [[nodiscard]] static std::unique_ptr<tensorrt_llm::executor::kv_cache::UcxConnectionManager> create()
    {
        return std::make_unique<UcxConnectionManager>();
    }

    void addConnection(std::string const& workerAddress);
    Connection const* recvConnect(DataContext const& ctx, void* data, size_t size) override;
    std::vector<Connection const*> getConnections(CommState const& state) override;
    [[nodiscard]] CommState const& getCommState() const override;

    [[nodiscard]] int getRank() const
    {
        return mRank;
    }
};

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
#endif

extern "C"
{
    [[nodiscard]] std::unique_ptr<ConnectionManager> makeUcxConnectionManager();
}

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

} // namespace tensorrt_llm::executor::kv_cache
