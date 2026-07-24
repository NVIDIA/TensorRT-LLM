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

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/serialization.h"
#include "tensorrt_llm/plugins/api/tllmPlugin.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include <csignal>

namespace tle = tensorrt_llm::executor;

int main(int argc, char* argv[])
{
#if ENABLE_MULTI_DEVICE

    if (std::getenv("FORCE_NCCL_ALL_REDUCE_STRATEGY") != nullptr)
    {
        TLLM_LOG_INFO("FORCE_NCCL_ALL_REDUCE_STRATEGY env variable detected in worker");
    }

    // Register the TRT-LLM plugins
    initTrtLlmPlugins();

    tensorrt_llm::mpi::initialize(tensorrt_llm::mpi::MpiThreadSupport::THREAD_MULTIPLE, true);

    MPI_Comm parentComm;
    MPI_Comm_get_parent(&parentComm);
    if (parentComm == MPI_COMM_NULL)
    {
        TLLM_LOG_ERROR("TRT-LLM worker has no parent!");
        return -1;
    }

    int size;
    MPI_Comm_remote_size(parentComm, &size);
    if (size != 1)
    {
        TLLM_LOG_ERROR("Parent size is %d, must be 1", size);
        return -1;
    }

    // Since parentComm is an intercommunicator, input root
    // is the rank of the parent process in his group
    // (always 0 as the parent size is checked before)

    // Receive from the parent the executor configuration
    int64_t bufferSize;
    MPICHECK(MPI_Bcast(&bufferSize, 1, MPI_INT64_T, 0, parentComm));
    std::vector<char> buffer(bufferSize);
    MPICHECK(MPI_Bcast(buffer.data(), bufferSize, MPI_CHAR, 0, parentComm));
    std::istringstream is(std::string(buffer.begin(), buffer.end()));
    auto modelPath = tle::Serialization::deserializeString(is);
    auto modelType = tle::Serialization::deserializeModelType(is);
    auto executorConfig = tle::Serialization::deserializeExecutorConfig(is);

    // Create the orchestrator config for workers
    auto orchLeaderComm = std::make_shared<tensorrt_llm::mpi::MpiComm>(parentComm, true);
    auto parallelConfig = executorConfig.getParallelConfig();
    TLLM_CHECK_WITH_INFO(parallelConfig.has_value(), "Parallel config should have a value.");
    TLLM_CHECK_WITH_INFO(
        parallelConfig.value().getOrchestratorConfig().has_value(), "Orchestrator config should have a value.");
    auto orchConfig = parallelConfig.value().getOrchestratorConfig().value();
    TLLM_CHECK_WITH_INFO(parallelConfig.has_value(), "Parallel config should have a value.");
    auto newOrchConfig = tle::OrchestratorConfig(false, orchConfig.getWorkerExecutablePath(), orchLeaderComm);
    parallelConfig.value().setOrchestratorConfig(newOrchConfig);
    executorConfig.setParallelConfig(parallelConfig.value());
    // In orchestrator mode, the spawned threads will wait for termination signal from orchestrator
    auto executor = tle::Executor(modelPath, modelType, executorConfig);

    // Wait for all workers to have created their instances
    MPI_Barrier(parentComm);
    TLLM_LOG_INFO("Executor instance created by worker");

#endif // ENABLE_MULTI_DEVICE

    return 0;
}
