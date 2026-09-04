/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/runtime/common.h"

namespace tensorrt_llm::runtime
{

class SpeculativeDecodingModule
{
public:
    explicit SpeculativeDecodingModule(SizeType32 maxDraftPathLen, SizeType32 maxDecodingDraftTokens) noexcept
        : mMaxDraftPathLen(maxDraftPathLen)
        , mMaxDecodingDraftTokens(maxDecodingDraftTokens)
    {
    }

    explicit SpeculativeDecodingModule() noexcept
        : SpeculativeDecodingModule(0, 0)
    {
    }

    virtual ~SpeculativeDecodingModule() = default;

    SpeculativeDecodingModule(SpeculativeDecodingModule const& o) = default;
    SpeculativeDecodingModule& operator=(SpeculativeDecodingModule const& o) = default;

    /// @return max number of draft tokens that can be accepted by one step of the decoder
    [[nodiscard]] SizeType32 getMaxDraftPathLen() const noexcept
    {
        return mMaxDraftPathLen;
    }

    /// @return max number of tokens that a request can grow in one step of the decoder
    /// @details one more than draft path len for prediction from primary head
    [[nodiscard]] SizeType32 getMaxPathLen() const noexcept
    {
        return getMaxDraftPathLen() + 1;
    }

    /// @return max number of draft tokens processed by one step of the decoder
    [[nodiscard]] SizeType32 getMaxDecodingDraftTokens() const noexcept
    {
        return mMaxDecodingDraftTokens;
    }

    /// @return max number of tokens processed by one step of the decoder
    /// @details one more than decoding draft tokens for prediction from primary head
    [[nodiscard]] SizeType32 getMaxDecodingTokens() const noexcept
    {
        return getMaxDecodingDraftTokens() + 1;
    }

private:
    SizeType32 mMaxDraftPathLen;        // max length per path (or ray/branch)
    SizeType32 mMaxDecodingDraftTokens; // max combined length of all paths (or rays/branches)
};
} // namespace tensorrt_llm::runtime
