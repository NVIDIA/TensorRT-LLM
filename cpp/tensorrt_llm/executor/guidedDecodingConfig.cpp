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

#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/executor/executor.h"

#include <optional>

namespace tensorrt_llm::executor
{

GuidedDecodingConfig::GuidedDecodingConfig(GuidedDecodingConfig::GuidedDecodingBackend backend,
    std::optional<std::vector<std::string>> encodedVocab, std::optional<std::string> tokenizerStr,
    std::optional<std::vector<TokenIdType>> stopTokenIds)
    : mBackend(backend)
    , mEncodedVocab(std::move(encodedVocab))
    , mTokenizerStr(std::move(tokenizerStr))
    , mStopTokenIds(std::move(stopTokenIds))
{
}

bool GuidedDecodingConfig::operator==(GuidedDecodingConfig const& other) const
{
    return mBackend == other.mBackend && mEncodedVocab == other.mEncodedVocab && mTokenizerStr == other.mTokenizerStr
        && mStopTokenIds == other.mStopTokenIds;
}

void GuidedDecodingConfig::setBackend(GuidedDecodingConfig::GuidedDecodingBackend const& backend)
{
    mBackend = backend;
}

GuidedDecodingConfig::GuidedDecodingBackend GuidedDecodingConfig::getBackend() const
{
    return mBackend;
}

void GuidedDecodingConfig::setEncodedVocab(std::vector<std::string> const& encodedVocab)
{
    mEncodedVocab = encodedVocab;
}

std::optional<std::vector<std::string>> GuidedDecodingConfig::getEncodedVocab() const
{
    return mEncodedVocab;
}

void GuidedDecodingConfig::setTokenizerStr(std::string const& tokenizerStr)
{
    mTokenizerStr = tokenizerStr;
}

std::optional<std::string> GuidedDecodingConfig::getTokenizerStr() const
{
    return mTokenizerStr;
}

void GuidedDecodingConfig::setStopTokenIds(std::vector<TokenIdType> const& stopTokenIds)
{
    mStopTokenIds = stopTokenIds;
}

std::optional<std::vector<TokenIdType>> GuidedDecodingConfig::getStopTokenIds() const
{
    return mStopTokenIds;
}

void GuidedDecodingConfig::validate() const
{
    if (mBackend == GuidedDecodingBackend::kXGRAMMAR)
    {
        TLLM_CHECK_WITH_INFO(mEncodedVocab, "Guided decoding is enabled with xgrammar, but EncodedVocab is not set");
        if (!mStopTokenIds)
        {
            TLLM_LOG_WARNING(
                "Guided decoding is enabled with xgrammar, but StopTokenIds is not set. The mismatch of stop token ids "
                "between requests and xgrammar may cause xgrammar execution error.");
        }
    }
}

} // namespace tensorrt_llm::executor
