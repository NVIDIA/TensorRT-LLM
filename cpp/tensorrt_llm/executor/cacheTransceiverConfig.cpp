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

namespace tensorrt_llm::executor
{

CacheTransceiverConfig::CacheTransceiverConfig(std::optional<BackendType> backendType,
    std::optional<size_t> maxNumTokens, std::optional<std::chrono::milliseconds> kvTransferTimeoutMs)
    : mBackendType(backendType)
    , mMaxTokensInBuffer(maxNumTokens)
    , mKvTransferTimeoutMs(kvTransferTimeoutMs)
{
}

bool CacheTransceiverConfig::operator==(CacheTransceiverConfig const& other) const
{
    return mMaxTokensInBuffer == other.mMaxTokensInBuffer && mBackendType == other.mBackendType
        && mKvTransferTimeoutMs == other.mKvTransferTimeoutMs;
}

void CacheTransceiverConfig::setBackendType(std::optional<BackendType> backendType)
{
    mBackendType = backendType;
}

void CacheTransceiverConfig::setMaxTokensInBuffer(std::optional<size_t> maxTokensInBuffer)
{
    mMaxTokensInBuffer = maxTokensInBuffer;
}

std::optional<CacheTransceiverConfig::BackendType> CacheTransceiverConfig::getBackendType() const
{
    return mBackendType;
}

std::optional<size_t> CacheTransceiverConfig::getMaxTokensInBuffer() const
{
    return mMaxTokensInBuffer;
}

void CacheTransceiverConfig::setKvTransferTimeoutMs(std::optional<std::chrono::milliseconds> kvTransferTimeoutMs)
{
    mKvTransferTimeoutMs = kvTransferTimeoutMs;
}

std::optional<std::chrono::milliseconds> CacheTransceiverConfig::getKvTransferTimeoutMs() const
{
    return mKvTransferTimeoutMs;
}

} // namespace tensorrt_llm::executor
