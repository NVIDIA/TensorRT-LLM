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
#include <future>
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
    using ConnectionIdType = uint64_t;

    UcxConnection() = default;
    explicit UcxConnection(ConnectionIdType connectionId, std::shared_ptr<ucxx::Endpoint> endpoint,
        UcxConnectionManager* manager, bool fromRequester);
    ~UcxConnection();
    void sendConnectionId(DataContext const& ctx, void const* data, size_t size) const;
    void send(DataContext const& ctx, void const* data, size_t size) const override;
    void recv(DataContext const& ctx, void* data, size_t size) const override;
    friend class UcxConnectionManager;

private:
    constexpr static uint64_t ResponserTag = 0xF1;
    constexpr static uint64_t RequesterTag = 0xF2;
    uint64_t mSendTagPrefix{0};
    uint64_t mRecvTagPrefix{0};

    ConnectionIdType mConnectionId;
    ConnectionIdType mConnectionIdInPeer;
    std::shared_ptr<ucxx::Endpoint> mEndpoint;
    UcxConnectionManager* mManager;
    bool mFromRequester;

    bool isFromRequester() const
    {
        return mFromRequester;
    }
};

} // namespace tensorrt_llm::executor::kv_cache
