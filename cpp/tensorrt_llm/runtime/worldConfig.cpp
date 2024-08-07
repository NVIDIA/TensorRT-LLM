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

#include "tensorrt_llm/runtime/worldConfig.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/mpiUtils.h"
#include "tensorrt_llm/common/stringUtils.h"

#include <algorithm>
#include <numeric>
#include <set>

using namespace tensorrt_llm::runtime;
namespace tc = tensorrt_llm::common;

WorldConfig::WorldConfig(SizeType32 tensorParallelism, SizeType32 pipelineParallelism, SizeType32 rank,
    SizeType32 gpusPerNode, std::optional<std::vector<SizeType32>> const& deviceIds)
    : mTensorParallelism{tensorParallelism}
    , mPipelineParallelism{pipelineParallelism}
    , mRank{rank}
    , mGpusPerNode{gpusPerNode}
    , mDeviceIds{deviceIds.value_or(std::vector<SizeType32>(mGpusPerNode))}
{
#if ENABLE_MULTI_DEVICE
    auto const numDevices = mDeviceIds.size();
    TLLM_CHECK(numDevices > 0);

    if (!deviceIds.has_value())
    {
        mDeviceIds.resize(mGpusPerNode);
        std::iota(mDeviceIds.begin(), mDeviceIds.end(), 0);
    }
    else
    {
        // total number is at most mGpusPerNode
        TLLM_CHECK_WITH_INFO(static_cast<SizeType32>(numDevices) <= mGpusPerNode,
            "Number of device IDs %zu is greater than GPUs per node %d", numDevices, mGpusPerNode);

        // all deviceIds is within the range
        TLLM_CHECK(*std::max_element(mDeviceIds.begin(), mDeviceIds.end()) < mGpusPerNode);
        TLLM_CHECK(*std::min_element(mDeviceIds.begin(), mDeviceIds.end()) >= 0);

        // all ids are unique
        std::set<SizeType32> const deviceIdSet(mDeviceIds.begin(), mDeviceIds.end());
        TLLM_CHECK_WITH_INFO(
            deviceIdSet.size() == numDevices, "Device IDs are not unique %zu != %zu", deviceIdSet.size(), numDevices);

        // log a warning if device ids are not contiguous
        if (std::adjacent_find(deviceIdSet.begin(), deviceIdSet.end(), [](auto x, auto y) { return y - x != 1; })
            != deviceIdSet.end())
        {
            TLLM_LOG_WARNING("The user specified device IDs are not contiguous!");
        }
        TLLM_LOG_INFO("Using user-specified devices: %s", tc::arr2str(mDeviceIds.data(), numDevices).c_str());
    }

    TLLM_CHECK(mTensorParallelism > 0);
    TLLM_CHECK(mPipelineParallelism > 0);
#else
    // Overriding to default - single GPU
    mRank = 0;
    mGpusPerNode = 1;
    mTensorParallelism = 1;
    mPipelineParallelism = 1;
#endif
}

bool WorldConfig::validMpiConfig() const
{
    return COMM_SESSION.getSize() == getSize();
}

WorldConfig WorldConfig::mpi(SizeType32 gpusPerNode, std::optional<SizeType32> tensorParallelism,
    std::optional<SizeType32> pipelineParallelism, std::optional<std::vector<SizeType32>> const& deviceIds)
{
#if ENABLE_MULTI_DEVICE
    auto& comm = COMM_SESSION;
    auto const mpiSize = comm.getSize();
    auto const mpiRank = comm.getRank();
    auto const mpiLocalSize = LOCAL_COMM_SESSION.getSize();
    TLLM_LOG_INFO("MPI size: %d, MPI local size: %d, rank: %d", mpiSize, mpiLocalSize, mpiRank);
    auto const pp = pipelineParallelism.value_or(1);
    auto const tp = tensorParallelism.value_or(mpiSize / pp);
    TLLM_LOG_DEBUG("TP: %d, PP: %d, gpusPerNode: %d", tp, pp, gpusPerNode);
    TLLM_CHECK_WITH_INFO(mpiSize == tp * pp, "MPI size %d != TP size %d * PP size %d", mpiSize, tp, pp);
    SizeType32 deviceCount{0};
    TLLM_CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if ((mpiSize < gpusPerNode && deviceCount < mpiSize) || (mpiSize >= gpusPerNode && deviceCount < gpusPerNode))
    {
        TLLM_CHECK_WITH_INFO(deviceCount == 1,
            "Detect %d GPUs, the GPU number is incompatible with %d gpusPerNode when MPI size is %d", deviceCount,
            gpusPerNode, mpiSize);
        TLLM_LOG_WARNING("gpusPerNode is %d but only detect single GPU, will set gpusPerNode to 1", gpusPerNode);
        if (std::getenv("CUDA_VISIBLE_DEVICES") != nullptr || std::getenv("NVIDIA_VISIBLE_DEVICES") != nullptr)
        {
            std::ostringstream oss;
            if (std::getenv("CUDA_VISIBLE_DEVICES") != nullptr)
            {
                oss << " CUDA_VISIBLE_DEVICES=" << std::getenv("CUDA_VISIBLE_DEVICES");
            }
            if (std::getenv("NVIDIA_VISIBLE_DEVICES") != nullptr)
            {
                oss << " NVIDIA_VISIBLE_DEVICES=" << std::getenv("NVIDIA_VISIBLE_DEVICES");
            }
            std::string envStr = oss.str();
            TLLM_LOG_WARNING(
                "Detect%s, please provide the full device list instead of limiting to single device, "
                "otherwise allreduce performance may be sub-optimal "
                "since custom allreduce kernel relies on P2P access to peer devices.",
                envStr.c_str());
        }
        gpusPerNode = 1;
    }

    return WorldConfig{tp, pp, mpiRank, gpusPerNode, deviceIds};
#else
    return WorldConfig();
#endif
}

std::vector<SizeType32> WorldConfig::getPipelineParallelGroup() const
{
    auto const pp = getPipelineParallelism();
    auto const tp = getTensorParallelism();
    auto const worldSize = getSize();
    std::vector<SizeType32> group;
    group.reserve(pp);
    for (SizeType32 idx = getTensorParallelRank(); idx < worldSize; idx += tp)
    {
        group.push_back(idx);
    }
    return group;
}

std::vector<SizeType32> WorldConfig::getTensorParallelGroup() const
{
    auto const tp = getTensorParallelism();
    auto const rank = getRank();
    auto const tpRank = getTensorParallelRank();
    std::vector<SizeType32> group;
    group.reserve(tp);
    for (SizeType32 idx = 0; idx < tp; idx++)
    {
        group.push_back(rank - tpRank + idx);
    }
    return group;
}
