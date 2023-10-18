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

#include "tensorrt_llm/runtime/samplingConfig.h"

#include <assert.h>
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
    using SizeType = runtime::SizeType;
    using TokenIdType = runtime::TokenIdType;
    using RequestIdType = std::uint64_t;
    using BeamTokens = std::vector<std::vector<TokenIdType>>;

    LlmRequest(RequestIdType requestId, SizeType maxNewTokens, std::shared_ptr<std::vector<TokenIdType>> input_tokens,
        runtime::SamplingConfig samplingConfig, bool isStreaming, std::optional<SizeType> endId = std::nullopt,
        std::optional<SizeType> padId = std::nullopt)
        : mRequestId(requestId)
        , mPromptLen(input_tokens->size())
        , mMaxNewTokens(maxNewTokens)
        , mSamplingConfig(samplingConfig)
        , mState(REQUEST_STATE_CONTEXT_INIT)
        , mIsStreaming(isStreaming)
        , mEndId(endId)
        , mPadId(padId)
        , mBatchSlot(-1)
        , mNumGeneratedTokens(0)
    {
        mMaxSentTokenPos = mPromptLen - 1;
        // Scatter the input tokens to other beam
        mTokens = std::make_shared<BeamTokens>(mSamplingConfig.beamWidth, *input_tokens);
    }

    /// @brief Get total number of tokens for this req (prompt + generated)
    /// @param beam The beam index
    /// @return  The number of tokens
    SizeType getNumTokens(SizeType beam) const
    {
        return mTokens->at(beam).size();
    }

    /// @brief Get a token at a given position and beam index
    /// @param beam  The beam index
    /// @param pos The position of the token relative to beginning of the prompt
    /// @return  The token index
    TokenIdType getToken(SizeType beam, SizeType pos) const
    {
        return mTokens->at(beam).at(pos);
    }

    /// @brief Get the tokens at a given beam index
    /// @param beam  The beam index
    /// @return  A vector of tokens for this beam index, includes the prompt
    std::vector<TokenIdType> getTokens(SizeType beam) const
    {
        return mTokens->at(beam);
    }

    /// @brief Get the number of generated tokens
    /// @return  The number of generated tokens (doesn't include the prompt tokens)
    SizeType getNumGeneratedTokens() const
    {
        return mNumGeneratedTokens;
    }

    /// @brief Add new generated tokens to the vector of tokens
    /// @param beamTokens A vector containing the tokens to add for each beam index
    ///                   beamTokens is expected to be of size beamWidth
    void addNewTokens(const std::vector<TokenIdType>& beamTokens)
    {
        assert(mSamplingConfig.beamWidth == beamTokens.size());
        for (std::size_t beam = 0; beam < beamTokens.size(); ++beam)
        {
            mTokens->at(beam).push_back(beamTokens[beam]);
        }
        mNumGeneratedTokens++;
    }

    /// @brief Sets the generated tokens for all beams. Erases all previous generated tokens.
    /// @param generatedBeamTokens The generated tokens for all beams (vector of vector of tokens)
    void setGeneratedTokens(const BeamTokens& generatedBeamTokens)
    {
        assert(generatedBeamTokens.size() == mSamplingConfig.beamWidth);
        for (std::size_t beam = 0; beam < generatedBeamTokens.size(); ++beam)
        {
            auto& beamTokens = (*mTokens)[beam];
            beamTokens.resize(mPromptLen);
            beamTokens.insert(beamTokens.end(), generatedBeamTokens[beam].begin(), generatedBeamTokens[beam].end());
        }
        mNumGeneratedTokens = generatedBeamTokens.at(0).size();
    }

    /// @brief Pause a request by moving the generated tokens to the prompt
    /// @param maxInputLen The maximum prompt len.
    void pause(SizeType maxInputLen)
    {
        // TODO: For beamWidth > 1, we would need to support swapping to avoid
        // recomputing from the start
        // As a temporary solution, we currently reset the tokens to the prompt
        if (mSamplingConfig.beamWidth > 1)
        {
            for (auto& beamTokens : *mTokens)
            {
                beamTokens.resize(mPromptLen);
            }
        }
        else
        {
            SizeType newPromptLen = std::min(maxInputLen, mPromptLen + mNumGeneratedTokens);
            for (auto& beamTokens : *mTokens)
            {
                beamTokens.resize(newPromptLen);
            }
            mMaxNewTokens -= (newPromptLen - mPromptLen);
            mPromptLen = newPromptLen;
        }
        mNumGeneratedTokens = 0;
        mState = REQUEST_STATE_CONTEXT_INIT;
        mBatchSlot = -1;
    }

    /// @brief Get the maximum position of the tokens returned to the client. Use to ensure we don't return to client
    /// duplicated token positions.
    /// @return The maximum position of the tokens sent to the client
    SizeType getMaxSentTokenPos() const
    {
        return mMaxSentTokenPos;
    }

    /// @brief Sets the maximum position of the tokens returned to the client. Use to ensure we don't return to client
    /// duplicated token positions.
    /// @param pos The maximum position
    void setMaxSentTokenPos(SizeType pos)
    {
        mMaxSentTokenPos = pos;
    }

    RequestIdType mRequestId;
    SizeType mPromptLen;
    SizeType mMaxNewTokens;
    // Tokens [beam_size, mPromptLen + mNumGeneratedTokens]
    runtime::SamplingConfig mSamplingConfig;
    LlmRequestState_t mState;
    bool mIsStreaming;
    std::optional<SizeType> mEndId;
    std::optional<SizeType> mPadId;
    SizeType mBatchSlot;

private:
    std::shared_ptr<BeamTokens> mTokens;
    SizeType mNumGeneratedTokens;
    SizeType mMaxSentTokenPos;
};

} // namespace tensorrt_llm::batch_manager
