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

#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/types.h"

#include <memory>
#include <optional>
#include <utility>

namespace tensorrt_llm::executor
{

LookaheadDecodingConfig::LookaheadDecodingConfig(
    SizeType32 windowSize, SizeType32 ngramSize, SizeType32 verificationSetSize)
    : mWindowSize(windowSize)
    , mNgramSize(ngramSize)
    , mVerificationSetSize(verificationSetSize)
{
    TLLM_CHECK_WITH_INFO(mNgramSize >= 1, "ngramSize requires >= 1");
    TLLM_CHECK_WITH_INFO(mWindowSize >= 1, "windowSize requires >= 1");
    TLLM_CHECK_WITH_INFO(
        mNgramSize == 1 ? mVerificationSetSize == 0 : true, "ngramSize=1 requires verificationSetSize=0");
    TLLM_CHECK_WITH_INFO(mNgramSize == 1 ? mWindowSize == 1 : true, "ngramSize=1 requires windowSize=1");
    TLLM_CHECK_WITH_INFO(mVerificationSetSize >= 0, "verificationSetSize requires >=0");
}

bool LookaheadDecodingConfig::operator==(LookaheadDecodingConfig const& other) const
{
    return mNgramSize == other.mNgramSize && mWindowSize == other.mWindowSize
        && mVerificationSetSize == other.mVerificationSetSize;
}

std::tuple<SizeType32 const, SizeType32 const, SizeType32 const> LookaheadDecodingConfig::get() const
{
    return std::make_tuple(mWindowSize, mNgramSize, mVerificationSetSize);
}

SizeType32 LookaheadDecodingConfig::getNgramSize() const
{
    return mNgramSize;
}

SizeType32 LookaheadDecodingConfig::getWindowSize() const
{
    return mWindowSize;
}

SizeType32 LookaheadDecodingConfig::getVerificationSetSize() const
{
    return mVerificationSetSize;
}

bool LookaheadDecodingConfig::isLegal(
    SizeType32 windowSize, SizeType32 ngramSize, SizeType32 verificationSetSize) noexcept
{
    bool result = true;
    result &= ngramSize >= 1;
    result &= windowSize >= 1;
    result &= ngramSize == 1 ? windowSize == 1 : true;
    result &= ngramSize == 1 ? verificationSetSize == 0 : true;
    result &= verificationSetSize >= 0;
    return result;
}

std::tuple<SizeType32, SizeType32, SizeType32, SizeType32> LookaheadDecodingConfig::calculateSpeculativeResource() const
{
    return calculateSpeculativeResourceTuple(mWindowSize, mNgramSize, mVerificationSetSize);
}

std::tuple<SizeType32, SizeType32, SizeType32, SizeType32> LookaheadDecodingConfig::calculateSpeculativeResourceTuple(
    SizeType32 windowSize, SizeType32 ngramSize, SizeType32 verificationSetSize)
{
    SizeType32 maxPathLen = ngramSize;
    SizeType32 maxDraftTokens =                                     //
        ((ngramSize == 1) ? 0 : (ngramSize - 2))                    // lookahead Window first
        + (windowSize - 1 + verificationSetSize) * (ngramSize - 1); // lookahead Window rest and guess tokens
    SizeType32 maxDecodingTokens = maxDraftTokens + 1;              // + golden Token
    SizeType32 maxDraftPathLen = ngramSize - 1;
    return std::make_tuple(maxDecodingTokens, maxPathLen, maxDraftTokens, maxDraftPathLen);
}

bool LookaheadDecodingConfig::isLE(LookaheadDecodingConfig const& that) const
{
    return mWindowSize <= that.mWindowSize && mNgramSize <= that.mNgramSize
        && mVerificationSetSize <= that.mVerificationSetSize;
}

DecodingConfig::DecodingConfig(
    std::optional<DecodingMode> decodingMode, std::optional<LookaheadDecodingConfig> lookaheadDecodingConfig)
    : mDecodingMode{decodingMode}
    , mLookaheadDecodingConfig{lookaheadDecodingConfig}
{
    if (mLookaheadDecodingConfig)
    {
        TLLM_CHECK_WITH_INFO(mDecodingMode, "LookaheadDecodingConfig is set, but DecodingMode is not set");
        TLLM_CHECK_WITH_INFO(mDecodingMode.value().isLookahead(),
            "LookaheadDecodingConfig is set, but DecodingMode is not set to Lookahead");
    }
}

bool DecodingConfig::operator==(DecodingConfig const& other) const
{
    return mDecodingMode == other.mDecodingMode && mLookaheadDecodingConfig == other.mLookaheadDecodingConfig;
}

std::optional<DecodingMode> DecodingConfig::getDecodingMode() const
{
    return mDecodingMode;
}

void DecodingConfig::setDecodingMode(DecodingMode const& decodingMode)
{
    if (decodingMode.isMedusa() || decodingMode.isLookahead() || decodingMode.isExplicitDraftTokens())
    {
        TLLM_THROW(
            "Decoding mode must not be set with `setDecodingMode` for Medusa, Lookahead or explicit draft tokens. "
            "Please, use setters for the respective configs or set decoding mode at the DecodingConfig constructor");
    }
    mDecodingMode = decodingMode;
}

std::optional<LookaheadDecodingConfig> DecodingConfig::getLookaheadDecodingConfig() const
{
    return mLookaheadDecodingConfig;
}

void DecodingConfig::setLookaheadDecodingConfig(LookaheadDecodingConfig const& lookaheadDecodingConfig)
{
    mLookaheadDecodingConfig = lookaheadDecodingConfig;
    mDecodingMode = DecodingMode::Lookahead();
}

SizeType32 DecodingConfig::getLookaheadDecodingMaxNumRequest() const
{
    return mLookaheadDecodingMaxNumRequest;
}

void DecodingConfig::enableSeamlessLookaheadDecoding()
{
    mDecodingMode = DecodingMode::Lookahead();
    if (!mLookaheadDecodingConfig.has_value())
    {
        mLookaheadDecodingConfig = executor::LookaheadDecodingConfig();
    }
}

} // namespace tensorrt_llm::executor
