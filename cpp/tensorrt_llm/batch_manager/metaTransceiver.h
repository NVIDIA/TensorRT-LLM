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
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"

#include <string>
#include <unordered_map>
#include <vector>
#include <zmq.hpp>

namespace tensorrt_llm::batch_manager
{

class MetaTransceiver
{
public:
    MetaTransceiver(bool isServer, std::optional<std::string> endpoint = std::nullopt)
        : mContext(1)
        , mIsServer(isServer)
    {
        if (mIsServer)
        {
            mSocket = zmq::socket_t(mContext, zmq::socket_type::sub);
            if (endpoint)
            {
                mSocket.set(zmq::sockopt::subscribe, "");
                mSocket.bind(endpoint.value());
                mEndpoint = endpoint.value();
            }
            else
            {
                mSocket.bind("tcp://127.0.0.1:*");
                mEndpoint = mSocket.get(zmq::sockopt::last_endpoint);
            }
            TLLM_LOG_INFO("MetaTransceiver bound to %s", mEndpoint.c_str());
        }
        else
        {
            mSocket = zmq::socket_t(mContext, zmq::socket_type::pub);
        }
    }

    ~MetaTransceiver()
    {
        mSocket.close();
        mContext.close();
    }

    void connect(std::string const& endpoint)
    {
        mEndpoint = endpoint;
        TLLM_CHECK_WITH_INFO(!mIsServer, "Only client can connect to an endpoint");
        mSocket.connect(endpoint);
    }

    void send(void const* data, size_t size);

    void recv(void* data, size_t size);

    std::string getEndpoint() const
    {
        return mEndpoint;
    }

private:
    bool mIsServer;
    zmq::context_t mContext;
    zmq::socket_t mSocket;
    std::string mEndpoint;
};
} // namespace tensorrt_llm::batch_manager
