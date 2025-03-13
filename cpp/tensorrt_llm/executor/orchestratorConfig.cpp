/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/types.h"
#include <filesystem>

#include <utility>

namespace tensorrt_llm::executor
{
OrchestratorConfig::OrchestratorConfig(bool isOrchestrator, std::string workerExecutablePath,
    std::shared_ptr<mpi::MpiComm> orchLeaderComm, bool spawnProcesses)
    : mIsOrchestrator(isOrchestrator)
    , mWorkerExecutablePath(std::move(workerExecutablePath))
    , mOrchLeaderComm(std::move(orchLeaderComm))
    , mSpawnProcesses(spawnProcesses)
{
    if (spawnProcesses)
    {
        TLLM_CHECK_WITH_INFO(std::filesystem::exists(mWorkerExecutablePath), "Worker executable at %s does not exist.",
            mWorkerExecutablePath.c_str());
    }
}

bool OrchestratorConfig::getIsOrchestrator() const
{
    return mIsOrchestrator;
}

std::string OrchestratorConfig::getWorkerExecutablePath() const
{
    return mWorkerExecutablePath;
}

std::shared_ptr<mpi::MpiComm> OrchestratorConfig::getOrchLeaderComm() const
{
    return mOrchLeaderComm;
}

bool OrchestratorConfig::getSpawnProcesses() const
{
    return mSpawnProcesses;
}

void OrchestratorConfig::setIsOrchestrator(bool isOrchestrator)
{
    mIsOrchestrator = isOrchestrator;
}

void OrchestratorConfig::setWorkerExecutablePath(std::string const& workerExecutablePath)
{
    mWorkerExecutablePath = workerExecutablePath;
}

void OrchestratorConfig::setOrchLeaderComm(std::shared_ptr<mpi::MpiComm> const& orchLeaderComm)
{
    mOrchLeaderComm = orchLeaderComm;
}

void OrchestratorConfig::setSpawnProcesses(bool spawnProcesses)
{
    mSpawnProcesses = spawnProcesses;
}

} // namespace tensorrt_llm::executor
