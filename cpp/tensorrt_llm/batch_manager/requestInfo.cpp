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

#include "tensorrt_llm/batch_manager/dataTransceiver.h"

#include <utility>

namespace tensorrt_llm::batch_manager
{

RequestInfo::RequestInfo(LlmRequest::RequestIdType requestId, executor::DataTransceiverState transState)
    : mRequestId{requestId}
    , mTransState{std::move(transState)}
{
}

RequestInfo::RequestInfo(LlmRequest::RequestIdType requestId, executor::DataTransceiverState transState,
    int32_t indexFromEnd, BlockKey const& lastBlockKey)
    : mRequestId{requestId}
    , mIndexFromEnd{indexFromEnd}
    , mLastBlockKey{lastBlockKey}
    , mTransState{std::move(transState)}
{
}

bool RequestInfo::operator==(RequestInfo const& rhs) const
{
    return mRequestId == rhs.mRequestId && mIndexFromEnd == rhs.mIndexFromEnd && mLastBlockKey == rhs.mLastBlockKey
        && mIsArbitraryTransfer == rhs.mIsArbitraryTransfer && mTransState == rhs.mTransState;
}

LlmRequest::RequestIdType RequestInfo::getRequestId() const noexcept
{
    return mRequestId;
}

executor::DataTransceiverState const& RequestInfo::getTransState() const noexcept
{
    return mTransState;
}

void RequestInfo::serialize(RequestInfo const& requestInfo, std::ostream& os)
{
    namespace su = executor::serialize_utils;
    su::serialize(requestInfo.mRequestId, os);
    su::serialize(requestInfo.mIndexFromEnd, os);
    su::serialize(requestInfo.mLastBlockKey, os);
    su::serialize(requestInfo.mIsArbitraryTransfer, os);
    su::serialize(requestInfo.mTransState, os);
}

RequestInfo RequestInfo::deserialize(std::istream& is)
{
    namespace su = executor::serialize_utils;
    auto requestId = su::deserialize<decltype(mRequestId)>(is);
    auto indexFromEnd = su::deserialize<decltype(mIndexFromEnd)>(is);
    auto lastBlockKey = su::deserialize<decltype(mLastBlockKey)>(is);
    auto isArbitraryTransfer = su::deserialize<decltype(mIsArbitraryTransfer)>(is);
    auto transState = su::deserialize<decltype(mTransState)>(is);
    auto requestInfo = RequestInfo{requestId, std::move(transState), indexFromEnd, lastBlockKey};
    requestInfo.setIsArbitraryTransfer(isArbitraryTransfer);
    return requestInfo;
}

std::size_t RequestInfo::serializedSize(RequestInfo const& requestInfo)
{
    namespace su = executor::serialize_utils;
    std::size_t totalSize = 0;
    totalSize += su::serializedSize(requestInfo.mRequestId);
    totalSize += su::serializedSize(requestInfo.mIndexFromEnd);
    totalSize += su::serializedSize(requestInfo.mLastBlockKey);
    totalSize += su::serializedSize(requestInfo.mIsArbitraryTransfer);
    totalSize += su::serializedSize(requestInfo.mTransState);
    return totalSize;
}

} // namespace tensorrt_llm::batch_manager
