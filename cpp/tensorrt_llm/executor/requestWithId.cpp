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

#include "requestWithId.h"
#include "tensorrt_llm/executor/serializeUtils.h"

#include <istream>
#include <ostream>
#include <sstream>

using namespace tensorrt_llm::executor;
namespace su = tensorrt_llm::executor::serialize_utils;

std::vector<char> tensorrt_llm::executor::RequestWithId::serializeReqWithIds(
    std::vector<RequestWithId> const& reqWithIds)
{
    // Compute the size of serialized buffer
    size_t totalSize = 0;
    totalSize += sizeof(size_t);
    for (auto const& reqWithId : reqWithIds)
    {
        totalSize += su::serializedSize(reqWithId.id);
        totalSize += su::serializedSize(reqWithId.req);
        totalSize += su::serializedSize(reqWithId.childReqIds);
        totalSize += su::serializedSize(static_cast<uint64_t>(reqWithId.queuedStart.time_since_epoch().count()));
    }

    std::vector<char> buffer(totalSize);
    std::stringbuf strbuf{std::ios_base::out | std::ios_base::in};
    strbuf.pubsetbuf(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    std::ostream ostream{&strbuf};

    su::serialize(reqWithIds.size(), ostream);
    for (auto const& reqWithId : reqWithIds)
    {
        su::serialize(reqWithId.id, ostream);
        su::serialize(reqWithId.req, ostream);
        su::serialize(reqWithId.childReqIds, ostream);
        su::serialize(static_cast<uint64_t>(reqWithId.queuedStart.time_since_epoch().count()), ostream);
    }
    return buffer;
}

std::vector<RequestWithId> tensorrt_llm::executor::RequestWithId::deserializeReqWithIds(std::vector<char>& buffer)
{
    std::vector<RequestWithId> reqWithIds;
    su::VectorWrapBuf<char> strbuf{buffer};
    std::istream istream{&strbuf};
    auto numReq = su::deserialize<size_t>(istream);
    for (size_t req = 0; req < numReq; ++req)
    {
        auto const id = su::deserialize<IdType>(istream);
        auto const request = Serialization::deserializeRequest(istream);
        auto const childReqIds = su::deserialize<std::vector<IdType>>(istream);
        auto const queuedStart = std::chrono::steady_clock::time_point{
            std::chrono::steady_clock::duration{su::deserialize<uint64_t>(istream)}};
        reqWithIds.emplace_back(RequestWithId{request, id, childReqIds, queuedStart});
    }
    return reqWithIds;
}
