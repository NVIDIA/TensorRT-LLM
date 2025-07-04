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

#include "metaTransceiver.h"
#include <zmq.hpp>

namespace tensorrt_llm::batch_manager
{
void MetaTransceiver::send(void const* data, size_t size)
{
    zmq::message_t message(data, size);
    mSocket.send(message, zmq::send_flags::none);
    TLLM_LOG_DEBUG("Sent message to IP %s", mEndpoint.c_str());
}

void MetaTransceiver::recv(void* data, size_t size)
{
    zmq::message_t message;
    mSocket.recv(message, zmq::recv_flags::none);
    TLLM_CHECK_WITH_INFO(message.size() == size, "Received message size does not match");
    memcpy(data, message.data(), size);
    TLLM_LOG_DEBUG("Received message from IP %s", mEndpoint.c_str());
}
} // namespace tensorrt_llm::batch_manager
