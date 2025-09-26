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

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/types.h"
#include <filesystem>

#include <utility>

namespace tensorrt_llm::executor
{
ParallelConfig::ParallelConfig(CommunicationType commType, CommunicationMode commMode,
    std::optional<std::vector<SizeType32>> deviceIds, std::optional<std::vector<SizeType32>> participantIds,
    std::optional<OrchestratorConfig> const& orchestratorConfig, std::optional<SizeType32> numNodes)
    : mCommType(commType)
    , mCommMode(commMode)
    , mDeviceIds(std::move(deviceIds))
    , mParticipantIds(std::move(participantIds))
    , mOrchestratorConfig(orchestratorConfig)
    , mNumNodes(numNodes)
{
    if (mDeviceIds)
    {
        TLLM_CHECK(!mDeviceIds.value().empty());
    }
}

CommunicationType ParallelConfig::getCommunicationType() const
{
    return mCommType;
}

CommunicationMode ParallelConfig::getCommunicationMode() const
{
    return mCommMode;
}

std::optional<std::vector<SizeType32>> ParallelConfig::getDeviceIds() const
{
    return mDeviceIds;
}

std::optional<std::vector<SizeType32>> ParallelConfig::getParticipantIds() const
{
    return mParticipantIds;
}

std::optional<OrchestratorConfig> ParallelConfig::getOrchestratorConfig() const
{
    return mOrchestratorConfig;
}

std::optional<SizeType32> ParallelConfig::getNumNodes() const
{
    return mNumNodes;
}

void ParallelConfig::setCommunicationType(CommunicationType type)
{
    mCommType = type;
}

void ParallelConfig::setCommunicationMode(CommunicationMode mode)
{
    mCommMode = mode;
}

void ParallelConfig::setDeviceIds(std::vector<SizeType32> const& deviceIds)
{
    TLLM_CHECK(!deviceIds.empty());
    mDeviceIds = deviceIds;
}

void ParallelConfig::setParticipantIds(std::vector<SizeType32> const& participantIds)
{
    mParticipantIds = participantIds;
}

void ParallelConfig::setOrchestratorConfig(OrchestratorConfig const& orchestratorConfig)
{
    mOrchestratorConfig = orchestratorConfig;
}

void ParallelConfig::setNumNodes(SizeType32 numNodes)
{
    mNumNodes = numNodes;
}

} // namespace tensorrt_llm::executor
