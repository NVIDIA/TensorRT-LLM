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

CacheTransceiverConfig::CacheTransceiverConfig(
    bool enableCacheTransceiver, std::optional<CommType> commType, std::optional<size_t> maxNumTokens)
    : mEnableCacheTransceiver(enableCacheTransceiver)
    , mCommType(commType)
    , mMaxNumTokens(maxNumTokens)
{
}

bool CacheTransceiverConfig::operator==(CacheTransceiverConfig const& other) const
{
    return mMaxNumTokens == other.mMaxNumTokens;
}

bool CacheTransceiverConfig::getEnableCacheTransceiver() const
{
    return mEnableCacheTransceiver;
}

void CacheTransceiverConfig::setCommType(std::optional<CommType> commType)
{
    mCommType = commType;
}

std::optional<CacheTransceiverConfig::CommType> CacheTransceiverConfig::getCommType() const
{
    return mCommType;
}

void CacheTransceiverConfig::setEnableCacheTransceiver(bool enableCacheTransceiver)
{
    mEnableCacheTransceiver = enableCacheTransceiver;
}

std::optional<size_t> CacheTransceiverConfig::getMaxNumTokens() const
{
    return mMaxNumTokens;
}

void CacheTransceiverConfig::setMaxNumTokens(size_t maxNumTokens)
{
    mMaxNumTokens = maxNumTokens;
}

} // namespace tensorrt_llm::executor
