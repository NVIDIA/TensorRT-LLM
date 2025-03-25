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
#include "tensorrt_llm/executor/cacheCommunicator.h"
#include <memory>

namespace tensorrt_llm::executor::kv_cache
{

class UcxConnectionManager;

class UcxConnection : public Connection
{
public:
    UcxConnection() = default;
    explicit UcxConnection(
        uint64_t connectionId, std::shared_ptr<ucxx::Endpoint> endpoint, UcxConnectionManager* manager);
    // UcxConnection(UcxConnection&& other);
    void sendGID();
    void send(DataContext const& ctx, void const* data, size_t size) const override;
    void recv(DataContext const& ctx, void* data, size_t size) const override;
    // void initialize(UcxConnectionManager* manager);
    friend class UcxConnectionManager;

private:
    void initializeEndpointTag(int maxTryTimes = 10);
    /*
        static constexpr ucxx::Tag kID_TAG{1};
        static constexpr ucxx::Tag kDATA_TAG{2};

        // Set of bit map used to represent UCXComm flags
        ucxx::Tag mInfoTag{1};

        static constexpr ucxx::TagMask mEndpointMask{(((uint64_t) 1 << (32 + 1)) - 1) << 32};
        static constexpr ucxx::TagMask mRequestIdMask{(((uint64_t) 1 << (16 + 1)) - 1) << 16};
        static constexpr ucxx::TagMask mFlagMask{(((uint64_t) 1 << (16 + 1)) - 1)};
    */

    // a send tag is defined as:
    // | local port (16 bits) | remote port (16 bits) | truncated request id (16 bits) | UCXComm flags (16 bits) |
    // a recv tag is defined as:
    // | remote port (16 bits) | local port (16 bits) | truncated request id (16 bits) | UCXComm flags (16 bits) |
    ucxx::Tag mSendTag{0};
    ucxx::Tag mRecvTag{0};

    mutable std::mutex mMtx;
    mutable std::condition_variable mCv;
    uint64_t mConnectionId;
    uint64_t mLocalGID;
    uint64_t mRemoteGID;
    std::shared_ptr<ucxx::Endpoint> mEndpoint;
    UcxConnectionManager* mManager;
};

// class UcxServer
// {
// public:
//     UcxServer(uint16_t listenerPort = 0);
//     [[nodiscard]] ConnectionManager const& getConnectionManager() const;

// private:
//     struct NetINfoT
//     {
//         std::string interface;
//         std::string ipv4;
//         std::string ipv6;
//         int idx = -1;
//     };

//     static void listenerCallback(ucp_conn_request_h conn_request, void* arg);
//     void startListener(uint16_t listenerPort);

//     UcxConnectionManager& mManager;
//     std::unordered_map<std::string, NetINfoT> mNetInfoMap;
//     std::shared_ptr<ucxx::Context> mContext;
//     std::shared_ptr<ucxx::Worker> mWorker;
//     std::shared_ptr<ucxx::Listener> mListener;
// };

// class UcxClient
// {
// public:
//     UcxClient();
//     void connect(std::string const& ip, std::uint16_t port);

//     [[nodiscard]] ConnectionManager const& getConnectionManager() const
//     {
//         return mManager;
//     }

// private:
//     void initSelfIps();

//     UcxConnectionManager mManager;
//     std::shared_ptr<ucxx::Context> mContext;
//     std::shared_ptr<ucxx::Worker> mWorker;
//     std::unordered_set<std::string> mSelfIps;
// };

} // namespace tensorrt_llm::executor::kv_cache
