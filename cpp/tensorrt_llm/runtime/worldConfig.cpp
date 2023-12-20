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

#include "tensorrt_llm/runtime/worldConfig.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/stringUtils.h"
#include "tensorrt_llm/runtime/tllmLogger.h"
#include "tensorrt_llm/runtime/utils/multiDeviceUtils.h"

#include <algorithm>
#include <csignal>
#include <cstdlib>
#include <mpi.h>
#include <mutex>
#include <numeric>

using namespace tensorrt_llm::runtime;
namespace tc = tensorrt_llm::common;

namespace
{

bool mpiInitialized = false;
std::mutex mpiMutex;

void initMpi(nvinfer1::ILogger& logger, int threadMode = MPI_THREAD_FUNNELED)
{
    std::lock_guard<std::mutex> lk(mpiMutex);
    if (mpiInitialized)
    {
        return;
    }

    int initialized = 0;
    TLLM_MPI_CHECK(MPI_Initialized(&initialized));
    if (!initialized)
    {
        logger.log(
            nvinfer1::ILogger::Severity::kINFO, tc::fmtstr("Initializing MPI with thread mode %d", threadMode).c_str());
        int providedMode;
        TLLM_MPI_CHECK(MPI_Init_thread(nullptr, nullptr, threadMode, &providedMode));
        TLLM_CHECK_WITH_INFO(providedMode >= threadMode, "MPI_Init_thread failed");
        std::atexit([]() { MPI_Finalize(); });

        auto previousHandler = std::signal(SIGABRT, [](int signal) { MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE); });
        TLLM_CHECK_WITH_INFO(previousHandler != SIG_ERR, "Signal handler setup failed");
    }

    mpiInitialized = true;
}

} // namespace

bool WorldConfig::validConfig(nvinfer1::ILogger& logger, SizeType tensorParallelism, SizeType pipelineParallelism)
{
    initMpi(logger);

    int mpiSize;
    TLLM_MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &mpiSize));
    return mpiSize == tensorParallelism * pipelineParallelism;
}

WorldConfig WorldConfig::mpi(nvinfer1::ILogger& logger, SizeType gpusPerNode, std::optional<SizeType> tensorParallelism,
    std::optional<SizeType> pipelineParallelism, std::optional<std::vector<SizeType>> userSpecifiedDeviceIds)
{
    initMpi(logger);

    int mpiSize, mpiRank;
    TLLM_MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &mpiSize));
    TLLM_MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank));
    logger.log(nvinfer1::ILogger::Severity::kINFO, tc::fmtstr("MPI size: %d, rank: %d", mpiSize, mpiRank).c_str());

    auto pp = pipelineParallelism.value_or(1);
    auto tp = tensorParallelism.value_or(mpiSize / pp);
    TLLM_CHECK(mpiSize == tp * pp);
    // Pass the user-specified device lists to the WorldConfig. Otherwise create a default list of device ids.
    std::vector<SizeType> deviceIds(mpiSize);
    std::iota(deviceIds.begin(), deviceIds.end(), 0);

    if (userSpecifiedDeviceIds)
    {
        TLLM_CHECK(static_cast<SizeType>(userSpecifiedDeviceIds.value().size()) == tp * pp);
        deviceIds = userSpecifiedDeviceIds.value();

        // For user provided device list, verify:
        // 1) total number is smaller than the total cuda-visible device counts
        // 2) all deviceIds is within the range
        // 3) All ids are unique
        // 4) if the deviceIds are contiguous, and throw a warning if not
        TLLM_CHECK((gpusPerNode >= static_cast<SizeType>(deviceIds.size()))
            && (gpusPerNode > *std::max_element(deviceIds.begin(), deviceIds.end()))
            && *std::min_element(deviceIds.begin(), deviceIds.end()) >= 0);
        gpusPerNode = deviceIds.size();
        auto it = std::unique(deviceIds.begin(), deviceIds.end());
        TLLM_CHECK(std::distance(deviceIds.begin(), it) == gpusPerNode);
        std::sort(deviceIds.begin(), deviceIds.end());

        // If the deviceIds are not contiguous, throw a warning
        bool isContiguous = true;
        for (SizeType i = 1; i < static_cast<SizeType>(deviceIds.size()); ++i)
        {
            if (deviceIds[i] != deviceIds[i - 1] + 1)
            {
                isContiguous = false;
                break;
            }
        }
        if (!isContiguous)
        {
            logger.log(nvinfer1::ILogger::Severity::kWARNING, "The user specified device IDs are not contiguous!");
        }

        std::stringstream ss;
        ss << "Using user-specificed devices: [";
        for (auto& id : deviceIds)
        {
            ss << id << ",";
        }
        ss << "]";
        logger.log(nvinfer1::ILogger::Severity::kINFO, ss.str().c_str());
    }
    return WorldConfig{tp, pp, mpiRank, gpusPerNode, deviceIds};
}

WorldConfig WorldConfig::mpi(SizeType gpusPerNode, std::optional<SizeType> tensorParallelism,
    std::optional<SizeType> pipelineParallelism, std::optional<std::vector<SizeType>> userSpecifiedDeviceIds)
{
    TllmLogger logger{};
    return mpi(logger, gpusPerNode, tensorParallelism, pipelineParallelism, userSpecifiedDeviceIds);
}

std::vector<SizeType> WorldConfig::getPipelineParallelGroup() const
{
    auto const pp = getPipelineParallelism();
    auto const tp = getTensorParallelism();
    auto const worldSize = getSize();
    std::vector<SizeType> group;
    group.reserve(pp);
    for (SizeType idx = getTensorParallelRank(); idx < worldSize; idx += tp)
    {
        group.push_back(idx);
    }
    return group;
}
