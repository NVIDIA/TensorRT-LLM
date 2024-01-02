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
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/mpiUtils.h"
#include "tensorrt_llm/common/stringUtils.h"

#include <algorithm>
#include <numeric>
#include <set>

using namespace tensorrt_llm::runtime;
namespace tc = tensorrt_llm::common;

WorldConfig::WorldConfig(SizeType tensorParallelism, SizeType pipelineParallelism, SizeType rank, SizeType gpusPerNode,
    std::optional<std::vector<SizeType>> const& deviceIds)
    : mTensorParallelism{tensorParallelism}
    , mPipelineParallelism{pipelineParallelism}
    , mRank{rank}
    , mGpusPerNode{gpusPerNode}
    , mDeviceIds{deviceIds.value_or(std::vector<SizeType>(mGpusPerNode))}
{
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
        TLLM_CHECK_WITH_INFO(static_cast<SizeType>(numDevices) <= mGpusPerNode,
            "Number of device IDs %zu is greater than GPUs per node %d", numDevices, mGpusPerNode);

        // all deviceIds is within the range
        TLLM_CHECK(*std::max_element(mDeviceIds.begin(), mDeviceIds.end()) < mGpusPerNode);
        TLLM_CHECK(*std::min_element(mDeviceIds.begin(), mDeviceIds.end()) >= 0);

        // all ids are unique
        std::set<SizeType> const deviceIdSet(mDeviceIds.begin(), mDeviceIds.end());
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

    TLLM_CHECK_WITH_INFO(static_cast<SizeType>(numDevices) >= tensorParallelism * pipelineParallelism,
        "Number of GPUs per node %d must be at least as large as TP (%d) * PP (%d)", mGpusPerNode, mTensorParallelism,
        mPipelineParallelism);
}

bool WorldConfig::validConfig(SizeType tensorParallelism, SizeType pipelineParallelism)
{
    auto const mpiSize = COMM_SESSION.getSize();
    return mpiSize == tensorParallelism * pipelineParallelism;
}

WorldConfig WorldConfig::mpi(SizeType gpusPerNode, std::optional<SizeType> tensorParallelism,
    std::optional<SizeType> pipelineParallelism, std::optional<std::vector<SizeType>> const& deviceIds)
{
    auto& comm = COMM_SESSION;
    auto const mpiSize = comm.getSize();
    auto const mpiRank = comm.getRank();
    TLLM_LOG_INFO("MPI size: %d, rank: %d", mpiSize, mpiRank);
    auto pp = pipelineParallelism.value_or(1);
    auto tp = tensorParallelism.value_or(mpiSize / pp);
    TLLM_CHECK(mpiSize == tp * pp);

    return WorldConfig{tp, pp, mpiRank, gpusPerNode, deviceIds};
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
