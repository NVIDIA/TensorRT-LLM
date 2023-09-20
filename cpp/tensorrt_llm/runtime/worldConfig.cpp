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

#include <cstdlib>
#include <mpi.h>

using namespace tensorrt_llm::runtime;
namespace tc = tensorrt_llm::common;

#define TLLM_MPI_CHECK(cmd, logger)                                                                                    \
    do                                                                                                                 \
    {                                                                                                                  \
        auto e = cmd;                                                                                                  \
        if (e != MPI_SUCCESS)                                                                                          \
        {                                                                                                              \
            logger.log(nvinfer1::ILogger::Severity::kERROR,                                                            \
                tc::fmtstr("Failed: MPI error %s:%d '%d'", __FILE__, __LINE__, e).c_str());                            \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

namespace
{

bool mpiInitialized = false;

void initMpi(nvinfer1::ILogger& logger, int threadMode = MPI_THREAD_FUNNELED)
{
    if (mpiInitialized)
    {
        return;
    }

    int initialized = 0;
    TLLM_MPI_CHECK(MPI_Initialized(&initialized), logger);
    if (!initialized)
    {
        logger.log(
            nvinfer1::ILogger::Severity::kINFO, tc::fmtstr("Initializing MPI with thread mode %d", threadMode).c_str());
        int providedMode;
        TLLM_MPI_CHECK(MPI_Init_thread(nullptr, nullptr, threadMode, &providedMode), logger);
        TLLM_CHECK_WITH_INFO(providedMode >= threadMode, "MPI_Init_thread failed");
        std::atexit([]() { MPI_Finalize(); });
    }

    mpiInitialized = true;
}

} // namespace

WorldConfig WorldConfig::mpi(nvinfer1::ILogger& logger, SizeType gpusPerNode)
{
    initMpi(logger);

    int mpiSize, mpiRank;
    TLLM_MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &mpiSize), logger);
    TLLM_MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank), logger);
    logger.log(nvinfer1::ILogger::Severity::kINFO, tc::fmtstr("MPI size: %d, rank: %d", mpiSize, mpiRank).c_str());
    return WorldConfig{mpiSize, mpiRank, gpusPerNode};
}

WorldConfig WorldConfig::mpi(SizeType gpusPerNode)
{
    TllmLogger logger{};
    return mpi(logger, gpusPerNode);
}
