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
#include "tensorrt_llm/runtime/utils/mpiUtils.h" //TODO: remove when progressing to standalone UCX stack

#include "ucxx/api.h"
#include "ucxx/utils/sockaddr.h"
#include "ucxx/utils/ucx.h"
#include <cstdint>
#if __linux__
#include <arpa/inet.h>
#include <ifaddrs.h>
#endif
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/executor/cache_transmission/ucx_utils/connection.h"
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace tensorrt_llm::executor::kv_cache
{

class UcxConnectionManager : public ConnectionManager, public std::enable_shared_from_this<UcxConnectionManager>
{
private:
    mpi::MpiComm const* mComm;
    std::shared_ptr<ucxx::Context> mUcxCtx;
    std::vector<std::shared_ptr<ucxx::Worker>> mWorkersPool;
    std::map<uint64_t, std::shared_ptr<UcxConnection>> mConnections;
    std::shared_ptr<ucxx::Listener> mListener;
    std::mutex mGIDToConnectionIdMutex;
    std::map<uint64_t, uint64_t> mGIDToConnectionId;
    std::mutex mPendingGIDFuturesMutex;
    std::queue<std::shared_ptr<std::future<void>>> mPendingGIDFutures;

    uint64_t getNewConnectionId(std::shared_ptr<ucxx::Endpoint> newEp);
    uint64_t addConnection(std::string ip, uint16_t port);
    // void initializeConnections();
    void updateGIDToConnectionIdMap(std::shared_ptr<ucxx::Request> request, uint64_t* gid, uint64_t connectionId);

public:
    explicit UcxConnectionManager(tensorrt_llm::mpi::MpiComm const* comm);
    ~UcxConnectionManager();

    // Factory function
    static std::unique_ptr<UcxConnectionManager> create(tensorrt_llm::mpi::MpiComm const* comm)
    {
        return std::make_unique<UcxConnectionManager>(comm);
    }

    [[nodiscard]] uint64_t getLocalGID() const;

    uint64_t addConnection(ucp_conn_request_h connRequest);
    Connection const* recvConnect(DataContext const& ctx, void* data, size_t size) override;
    std::vector<Connection const*> getConnections(CommState const& state) override;
};

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
#endif

extern "C"
{
    [[nodiscard]] std::unique_ptr<ConnectionManager> makeUcxConnectionManager(tensorrt_llm::mpi::MpiComm const* comm);
}

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

} // namespace tensorrt_llm::executor::kv_cache
