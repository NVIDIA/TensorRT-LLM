/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include "tensorrt_llm/common/envUtils.h"
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
    std::map<std::string, uint64_t> mIpToMpiRank;
    std::map<uint64_t, uint64_t> mMpiRankToConnectionId;

    uint64_t getNewConnectionId(std::shared_ptr<ucxx::Endpoint> newEp);
    void addConnection(std::string ip, uint16_t port);

public:
    explicit UcxConnectionManager(mpi::MpiComm const* comm);
    void addConnection(ucp_conn_request_h connRequest);
    Connection const* recvConnect(DataContext const& ctx, void* data, size_t size) override;
    std::vector<Connection const*> getConnections(CommState const& state) override;
};

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
#endif

extern "C"
{
    std::unique_ptr<ConnectionManager> makeUcxConnectionManager(mpi::MpiComm const* comm);
}

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

} // namespace tensorrt_llm::executor::kv_cache
