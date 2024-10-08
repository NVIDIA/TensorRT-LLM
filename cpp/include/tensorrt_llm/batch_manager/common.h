/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "tensorrt_llm/runtime/common.h"
#include <cstdint>
#include <list>
#include <memory>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tensorrt_llm::executor
{
class RequestWithId;
}

namespace tensorrt_llm::batch_manager
{
class LlmRequest;

using RequestList = std::list<std::shared_ptr<LlmRequest>>;
using RequestIdType = std::uint64_t;
using RequestVector = std::vector<std::shared_ptr<LlmRequest>>;
using ReqIdsSet = std::unordered_set<RequestIdType>;

class ScheduledRequests
{
public:
    /// @brief context phase requests (for decoder-only models) or encoder phase requests (for encoder-decoder models
    /// and encoder-only models)
    RequestVector contextRequests;

    /// @brief generation phase requests (for decoder-only models) or empty for others
    RequestVector generationRequests;

    ScheduledRequests() = default;

    explicit ScheduledRequests(RequestVector contextRequests, RequestVector generationRequests)
        : contextRequests{std::move(contextRequests)}
        , generationRequests{std::move(generationRequests)}
    {
    }

    [[nodiscard]] bool empty() const
    {
        return contextRequests.empty() && generationRequests.empty();
    }

    [[nodiscard]] std::size_t size() const
    {
        return contextRequests.size() + generationRequests.size();
    }
};

class BatchState
{
public:
    BatchState() = default;

    BatchState(runtime::SizeType32 numCtxRequests, runtime::SizeType32 numGenRequests, runtime::SizeType32 numTokens,
        runtime::SizeType32 maxKvCacheLength)
        : mNumCtxRequests{numCtxRequests}
        , mNumGenRequests{numGenRequests}
        , mNumTokens{numTokens}
        , mMaxKvCacheLength{maxKvCacheLength}
    {
    }

    bool isAnyContext() const
    {
        return mNumCtxRequests > 0;
    }

    bool operator==(BatchState const& other) const
    {
        return mNumCtxRequests == other.mNumCtxRequests && mNumGenRequests == other.mNumGenRequests
            && mNumTokens == other.mNumTokens && mMaxKvCacheLength == other.mMaxKvCacheLength;
    }

    size_t hash() const
    {
        size_t h1 = std::hash<runtime::SizeType32>{}(mNumCtxRequests);
        size_t h2 = std::hash<runtime::SizeType32>{}(mNumGenRequests);
        size_t h3 = std::hash<runtime::SizeType32>{}(mNumTokens);
        size_t h4 = std::hash<runtime::SizeType32>{}(mMaxKvCacheLength);
        return h1 ^ h2 ^ h3 ^ h4;
    }

    runtime::SizeType32 mNumCtxRequests;
    runtime::SizeType32 mNumGenRequests;
    runtime::SizeType32 mNumTokens;
    runtime::SizeType32 mMaxKvCacheLength;
};

struct BatchStateHash
{
    size_t operator()(BatchState const& bs) const
    {
        return bs.hash();
    }
};

} // namespace tensorrt_llm::batch_manager
