/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/common/nativeMemory.h"
#include "tensorrt_llm/common/ncclUtils.h"
#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/common/tllmDataType.h"
#include "tensorrt_llm/runtime/ipcNvlsMemory.h"
#include "tensorrt_llm/runtime/utils/mpiTags.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include "tensorrt_llm/runtime/utils/pgUtils.h"

#include <iterator>
#include <mutex>
#include <sstream>

TRTLLM_NAMESPACE_BEGIN

#if ENABLE_MULTI_DEVICE

std::unordered_map<tensorrt_llm::DataType, ncclDataType_t>* getDtypeMap()
{
    static std::unordered_map<tensorrt_llm::DataType, ncclDataType_t> dtypeMap = {
        {tensorrt_llm::DataType::kFLOAT, ncclFloat32},
        {tensorrt_llm::DataType::kHALF, ncclFloat16},
        {tensorrt_llm::DataType::kBF16, ncclBfloat16},
        {tensorrt_llm::DataType::kFP8, ncclInt8},
        {tensorrt_llm::DataType::kBOOL, ncclInt8},
        {tensorrt_llm::DataType::kINT32, ncclInt32},
        {tensorrt_llm::DataType::kINT64, ncclInt64},
        {tensorrt_llm::DataType::kUINT8, ncclUint8},
        {tensorrt_llm::DataType::kINT8, ncclInt8},
    };
    return &dtypeMap;
}

namespace
{

// Get NCCL unique ID for a group of ranks.
ncclUniqueId getUniqueId(std::set<int> const& group)
{
    if (!useMPI())
    {
        auto const pg = pg_utils::get_world_pg();
        TLLM_CHECK_WITH_INFO(pg != nullptr, "Host process group is not initialized");
        pg_utils::PgHelper host(pg);
        ncclUniqueId id{};
        auto bytes = at::from_blob(&id, {sizeof(id)}, at::TensorOptions().dtype(at::kByte));
        int const root = *group.begin();
        int const tag = static_cast<int>(mpi::MpiTag::kNcclUniqueId);
        if (pg->getRank() == root)
        {
            NCCLCHECK_THROW(ncclGetUniqueId(&id));
            for (auto const rank : group)
            {
                if (rank != root)
                {
                    PGCHECK_THROW(host.send(bytes, rank, tag));
                }
            }
        }
        else
        {
            PGCHECK_THROW(host.recv(bytes, root, tag));
        }
        return id;
    }
    auto const rank = COMM_SESSION.getRank();
    TLLM_LOG_TRACE("%s start for rank %d", __PRETTY_FUNCTION__, rank);
    ncclUniqueId id;
    if (rank == *group.begin())
    {
        NCCLCHECK_THROW(ncclGetUniqueId(&id));
        for (auto it = std::next(std::begin(group), 1); it != group.end(); ++it)
        {
            COMM_SESSION.sendValue(id, *it, tensorrt_llm::mpi::MpiTag::kNcclUniqueId);
        }
    }
    else
    {
        COMM_SESSION.recvValue(id, *group.begin(), tensorrt_llm::mpi::MpiTag::kNcclUniqueId);
    }
    TLLM_LOG_TRACE("%s stop for rank %d", __PRETTY_FUNCTION__, rank);
    return id;
}
} // namespace

std::shared_ptr<ncclComm_t> getComm(std::set<int> const& group)
{
    auto const hostPg = useMPI() ? nullptr : pg_utils::get_world_pg();
    TLLM_CHECK_WITH_INFO(useMPI() || hostPg != nullptr, "Host process group is not initialized");
    auto const rank = useMPI() ? COMM_SESSION.getRank() : hostPg->getRank();
    TLLM_LOG_TRACE("%s start for rank %d", __PRETTY_FUNCTION__, rank);
    static std::map<std::set<int>, std::shared_ptr<ncclComm_t>> commMap;
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    std::ostringstream oss;
    int index = 0;
    for (auto const& rank : group)
    {
        if (index != 0)
        {
            oss << ",";
        }
        oss << rank;
        index++;
    }
    auto groupStr = oss.str();
    auto it = commMap.find(group);
    if (it != commMap.end())
    {
        auto ncclComm = it->second;
        TLLM_LOG_TRACE("NCCL comm for group(%s) is cached for rank %d", groupStr.c_str(), rank);
        return ncclComm;
    }

    TLLM_LOG_TRACE("Init NCCL comm for group(%s) for rank %d", groupStr.c_str(), rank);
    // Finish every environment update before getUniqueId(): on the group root, ncclGetUniqueId() starts NCCL's
    // bootstrap thread, which reads the environment with getenv(), and a concurrent setenv() may reallocate
    // environ underneath it.
#if defined(_WIN32)
    // Need static connection initialization for accurate KV cache size estimation
    if (getenv("NCCL_RUNTIME_CONNECT") == nullptr)
        _putenv_s("NCCL_RUNTIME_CONNECT", "0");
    // Disable graph register to avoid startup hangs
    if (getenv("NCCL_GRAPH_REGISTER") == nullptr)
        _putenv_s("NCCL_GRAPH_REGISTER", "0");
#else
    setenv("NCCL_RUNTIME_CONNECT", "0", 0);
    setenv("NCCL_GRAPH_REGISTER", "0", 0);
    // NCCL aborts during init if it tries NVLS multicast but the fabric/IMEX
    // plane can't bind it. Disable NVLS when the fabric is not usable so NCCL
    // falls back to NVLink P2P. No-overwrite preserves an explicit user setting.
    if (!tensorrt_llm::runtime::ipcNvlsFabricUsable())
    {
        setenv("NCCL_NVLS_ENABLE", "0", 0);
    }
#endif // _WIN32
    ncclUniqueId id = getUniqueId(group);
    int groupRank = 0;
    for (auto const& currentRank : group)
    {
        if (rank == currentRank)
            break;
        ++groupRank;
    }
    TLLM_CHECK(static_cast<size_t>(groupRank) < group.size());
    auto nativeMemory = std::make_shared<tensorrt_llm::common::NativeMemoryRecord>();
    std::shared_ptr<ncclComm_t> ncclComm(new ncclComm_t,
        [nativeMemory](ncclComm_t* comm)
        {
            if (!comm)
            {
                return;
            }

            // STEP 1: Clean up resources and destroy NCCL communicator if it's valid
            if (*comm)
            {
                // Clean up all registered resources FIRST
                // The cleanupResources function uses a destruction guard to safely handle
                // static destruction order issues - it will return early if the singleton
                // is being destroyed (in which case the destructor handles cleanup proactively)
                tensorrt_llm::common::nccl_util::NcclCommResourceManager::getInstance().cleanupResources(*comm);

                // Now destroy the NCCL communicator
                ncclResult_t result = ncclCommDestroy(*comm);
                if (result != ncclSuccess)
                {
                    // Logging may fail during static destruction, so wrap in try-catch
                    try
                    {
                        TLLM_LOG_WARNING("ncclCommDestroy failed with error: %d", result);
                    }
                    catch (...)
                    {
                        // Ignore logging failures during static destruction
                    }
                }

                // Clear the communicator value before freeing the pointer
                *comm = nullptr;
            }

            // STEP 2: Always free the pointer memory (regardless of whether *comm was valid)
            delete comm;
            tensorrt_llm::common::recordNativeRelease(*nativeMemory);
        });
    auto const freeBefore = tensorrt_llm::common::currentDeviceFreeBytes();
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 0)
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    config.graphUsageMode = 1;
    NCCLCHECK_THROW(ncclCommInitRankConfig(ncclComm.get(), group.size(), id, groupRank, &config));
#else
    NCCLCHECK_THROW(ncclCommInitRank(ncclComm.get(), group.size(), id, groupRank));
#endif // NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 0)
    // With NCCL_RUNTIME_CONNECT=0 (set above) NCCL allocates its connection buffers here and reports no size.
    *nativeMemory = tensorrt_llm::common::recordNativeAllocationMeasured(
        tensorrt_llm::common::nativeMemoryLabels::kNcclCommunicators, freeBefore);
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)
    if (common::nccl_util::isNcclWindowSupported())
    {
        std::vector<int> const ranks(group.begin(), group.end());
        if (useMPI())
        {
            common::nccl_util::registerHostCommunicator(*ncclComm, COMM_SESSION, ranks);
        }
        else
        {
            common::nccl_util::registerHostCommunicator(*ncclComm, hostPg, ranks);
        }
    }
#endif
    commMap[group] = ncclComm;
    TLLM_LOG_TRACE("%s stop for rank %d", __PRETTY_FUNCTION__, rank);
    return ncclComm;
}
#endif // ENABLE_MULTI_DEVICE

void const* tensorrt_llm::common::op::getCommSessionHandle()
{
#if ENABLE_MULTI_DEVICE
    return &COMM_SESSION;
#else
    return nullptr;
#endif // ENABLE_MULTI_DEVICE
}

TRTLLM_NAMESPACE_END
