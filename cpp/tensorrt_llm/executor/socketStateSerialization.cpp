/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/executor/dataTransceiverState.h"
#include "tensorrt_llm/executor/serialization.h"
#include "tensorrt_llm/executor/serializeUtils.h"

#include <utility>

namespace tensorrt_llm::executor
{

namespace su = serialize_utils;

// SocketState
kv_cache::SocketState Serialization::deserializeSocketState(std::istream& is)
{
    auto port = su::deserialize<decltype(kv_cache::SocketState::mPort)>(is);
    auto ip = su::deserialize<decltype(kv_cache::SocketState::mIp)>(is);
    return kv_cache::SocketState{port, std::move(ip)};
}

void Serialization::serialize(kv_cache::SocketState const& state, std::ostream& os)
{
    su::serialize(state.mPort, os);
    su::serialize(state.mIp, os);
}

size_t Serialization::serializedSize(kv_cache::SocketState const& state)
{
    size_t totalSize = 0;
    totalSize += su::serializedSize(state.mPort);
    totalSize += su::serializedSize(state.mIp);
    return totalSize;
}

} // namespace tensorrt_llm::executor
