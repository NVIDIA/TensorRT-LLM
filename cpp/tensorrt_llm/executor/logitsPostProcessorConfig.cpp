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

namespace tensorrt_llm::executor
{

LogitsPostProcessorConfig::LogitsPostProcessorConfig(std::optional<LogitsPostProcessorMap> processorMap,
    std::optional<LogitsPostProcessorBatched> processorBatched, bool replicate, bool returnsLogProbs)
    : mProcessorMap(std::move(processorMap))
    , mProcessorBatched(std::move(processorBatched))
    , mReplicate(replicate)
    , mReturnsLogProbs(returnsLogProbs)
{
}

std::optional<LogitsPostProcessorMap> LogitsPostProcessorConfig::getProcessorMap() const
{
    return mProcessorMap;
}

std::optional<LogitsPostProcessorBatched> LogitsPostProcessorConfig::getProcessorBatched() const
{
    return mProcessorBatched;
}

bool LogitsPostProcessorConfig::getReplicate() const
{
    return mReplicate;
}

bool LogitsPostProcessorConfig::getReturnsLogProbs() const
{
    return mReturnsLogProbs;
}

void LogitsPostProcessorConfig::setProcessorMap(LogitsPostProcessorMap const& processorMap)
{
    mProcessorMap = processorMap;
}

void LogitsPostProcessorConfig::setProcessorBatched(LogitsPostProcessorBatched const& processorBatched)
{
    mProcessorBatched = processorBatched;
}

void LogitsPostProcessorConfig::setReplicate(bool replicate)
{
    mReplicate = replicate;
}

void LogitsPostProcessorConfig::setReturnsLogProbs(bool returnsLogProbs)
{
    mReturnsLogProbs = returnsLogProbs;
}

} // namespace tensorrt_llm::executor
