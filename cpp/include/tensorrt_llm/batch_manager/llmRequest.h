/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/batch_manager/BatchManager.h"
#include "tensorrt_llm/runtime/samplingConfig.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace tensorrt_llm::batch_manager
{

enum LlmRequestState_t
{
    REQUEST_STATE_UNKNOWN = 0,
    REQUEST_STATE_CONTEXT_INIT = 1,
    REQUEST_STATE_GENERATION_IN_PROGRESS = 2,
    REQUEST_STATE_GENERATION_COMPLETE = 3
};

class LlmRequest
{
public:
    using BeamTokens = std::vector<std::vector<int32_t>>;
    using SizeType = runtime::SizeType;

    LlmRequest(uint64_t requestId, int32_t maxNewTokens, std::shared_ptr<std::vector<int32_t>> input_tokens,
        runtime::SamplingConfig samplingConfig, bool isStreaming, std::optional<SizeType> endId = std::nullopt,
        std::optional<SizeType> padId = std::nullopt)
        : mRequestId(requestId)
        , mMaxNewTokens(maxNewTokens)
        , mSamplingConfig(samplingConfig)
        , mPromptLen(input_tokens->size())
        , mNumGeneratedTokens(0)
        , mState(REQUEST_STATE_CONTEXT_INIT)
        , mIsStreaming(isStreaming)
        , mEndId(endId)
        , mPadId(padId)
    {
        // Scatter the input tokens to other beam
        mTokens = std::make_shared<BeamTokens>(mSamplingConfig.beamWidth, *input_tokens);
    }

    uint64_t mRequestId;
    int32_t mMaxNewTokens;
    // Tokens [beam_size, mPromptLen + mNumGeneratedTokens]
    std::shared_ptr<BeamTokens> mTokens;
    runtime::SamplingConfig mSamplingConfig;
    int32_t mPromptLen;
    int32_t mNumGeneratedTokens;
    LlmRequestState_t mState;
    bool mIsStreaming;
    std::optional<SizeType> mEndId;
    std::optional<SizeType> mPadId;

    ~LlmRequest() {}
};

} // namespace tensorrt_llm::batch_manager
