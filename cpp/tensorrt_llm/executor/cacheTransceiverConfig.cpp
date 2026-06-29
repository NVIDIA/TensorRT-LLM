/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    std::optional<size_t> maxNumTokens, std::optional<int> kvTransferTimeoutMs,
    std::optional<int> kvTransferSenderFutureTimeoutMs, std::optional<int> kvTransferPollIntervalMs)
    : mBackendType(backendType)
    , mMaxTokensInBuffer(maxNumTokens)
    , mKvTransferTimeoutMs(kvTransferTimeoutMs)
    , mKvTransferSenderFutureTimeoutMs(kvTransferSenderFutureTimeoutMs)
{
    setKvTransferPollIntervalMs(kvTransferPollIntervalMs);
}

bool CacheTransceiverConfig::operator==(CacheTransceiverConfig const& other) const
{
    return mMaxTokensInBuffer == other.mMaxTokensInBuffer && mBackendType == other.mBackendType
        && mKvTransferTimeoutMs == other.mKvTransferTimeoutMs
        && mKvTransferSenderFutureTimeoutMs == other.mKvTransferSenderFutureTimeoutMs
        && mKvTransferPollIntervalMs == other.mKvTransferPollIntervalMs;
}

void CacheTransceiverConfig::setBackendType(std::optional<BackendType> backendType)
{
    mBackendType = backendType;
}

void CacheTransceiverConfig::setMaxTokensInBuffer(std::optional<size_t> maxTokensInBuffer)
{
    mMaxTokensInBuffer = maxTokensInBuffer;
}

void CacheTransceiverConfig::setKvTransferTimeoutMs(std::optional<int> kvTransferTimeoutMs)
{
    if (kvTransferTimeoutMs.has_value() && kvTransferTimeoutMs.value() <= 0)
    {
        TLLM_THROW("kvTransferTimeoutMs must be positive");
    }
    mKvTransferTimeoutMs = kvTransferTimeoutMs;
}

void CacheTransceiverConfig::setKvTransferSenderFutureTimeoutMs(std::optional<int> kvTransferSenderFutureTimeoutMs)
{
    if (kvTransferSenderFutureTimeoutMs.has_value() && kvTransferSenderFutureTimeoutMs.value() <= 0)
    {
        TLLM_THROW("kvTransferSenderFutureTimeoutMs must be positive");
    }
    mKvTransferSenderFutureTimeoutMs = kvTransferSenderFutureTimeoutMs;
}

void CacheTransceiverConfig::setKvTransferPollIntervalMs(std::optional<int> kvTransferPollIntervalMs)
{
    if (kvTransferPollIntervalMs.has_value() && kvTransferPollIntervalMs.value() <= 0)
    {
        TLLM_THROW("kvTransferPollIntervalMs must be positive");
    }
    mKvTransferPollIntervalMs = kvTransferPollIntervalMs;
}

std::optional<CacheTransceiverConfig::BackendType> CacheTransceiverConfig::getBackendType() const
{
    return mBackendType;
}

std::optional<size_t> CacheTransceiverConfig::getMaxTokensInBuffer() const
{
    return mMaxTokensInBuffer;
}

std::optional<int> CacheTransceiverConfig::getKvTransferTimeoutMs() const
{
    return mKvTransferTimeoutMs;
}

std::optional<int> CacheTransceiverConfig::getKvTransferSenderFutureTimeoutMs() const
{
    return mKvTransferSenderFutureTimeoutMs;
}

std::optional<int> CacheTransceiverConfig::getKvTransferPollIntervalMs() const
{
    return mKvTransferPollIntervalMs;
}
} // namespace tensorrt_llm::executor
