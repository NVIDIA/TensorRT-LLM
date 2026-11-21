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

#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/common/ncclUtils.h"
#include "tensorrt_llm/common/tllmDataType.h"
#include "tensorrt_llm/runtime/ipcNvlsMemory.h"
#include "tensorrt_llm/runtime/utils/mpiTags.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include "tensorrt_llm/runtime/utils/pgUtils.h"

#include "cuda.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <functional>
#include <mutex>
#include <thread>
#include <tuple>
#include <vector>

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
    std::shared_ptr<ncclComm_t> ncclComm(new ncclComm_t,
        [](ncclComm_t* comm)
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
        });
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 0)
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    config.graphUsageMode = 1;
    NCCLCHECK_THROW(ncclCommInitRankConfig(ncclComm.get(), group.size(), id, groupRank, &config));
#else
    NCCLCHECK_THROW(ncclCommInitRank(ncclComm.get(), group.size(), id, groupRank));
#endif // NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 0)
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

namespace
{

// Get current cuda context, a default context will be created if there is no context.
inline CUcontext getCurrentCudaCtx()
{
    CUcontext ctx{};
    CUresult err = cuCtxGetCurrent(&ctx);
    if (err == CUDA_ERROR_NOT_INITIALIZED || ctx == nullptr)
    {
        TLLM_CUDA_CHECK(cudaFree(nullptr));
        err = cuCtxGetCurrent(&ctx);
    }
    TLLM_CHECK(err == CUDA_SUCCESS);
    return ctx;
}

//! Pool of cuBLAS / cuBLASLt handles grouped by CUDA context, with the same rule as PyTorch's DeviceThreadHandlePool:
//! - A thread gets the handle it already holds in the current CUDA context; otherwise an idle handle of that context;
//!   otherwise a new one.
//! - When a thread lets go of its handle (the last std::shared_ptr is released, usually when the thread exits), the
//!   handle is not destroyed but goes back to the idle list of its CUDA context.
//! So a thread that starts late, when the PyTorch caching allocator may hold nearly all device memory, takes over a
//! handle instead of calling cublasCreate, which allocates its default workspace directly from the driver.
//!
//! Lifetime:
//! - Idle handles are never destroyed, not even at process exit: CUDA may already be unloaded by then, and
//!   cublasDestroy, which synchronizes the device, can crash or hang.
//! - Per CUDA context, their number never exceeds the peak number of threads that held a handle of that context at
//!   the same time.
//! - A context that ever had a handle must stay alive until process exit: otherwise its idle handles would dangle,
//!   and a context created later at the same address would be given them.
//! - The pool is a leaked singleton (hence the deleted destructor): a handle may still be given back after static
//!   destruction has started (by a static object or by a thread still running at exit), and giveBack() needs a live
//!   pool.
template <typename T>
class PerCudaCtxHandlePool
{
public:
    using CreateFunc = std::function<std::unique_ptr<T>()>;
    using ResetFunc = std::function<void(T)>;

    //! \param create Creates a handle in the current CUDA context.
    //! \param resetForNewThread Clears the state that the previous thread may have left on an idle handle.
    PerCudaCtxHandlePool(CreateFunc create, ResetFunc resetForNewThread)
        : mCreate{std::move(create)}
        , mResetForNewThread{std::move(resetForNewThread)}
    {
    }

    ~PerCudaCtxHandlePool() = delete;

    std::shared_ptr<T> acquire()
    {
        CUcontext const ctx = getCurrentCudaCtx();
        HolderKey const key{ctx, std::this_thread::get_id()};
        std::unique_ptr<T> handle;
        {
            std::lock_guard<std::mutex> lk{mMutex};
            if (auto held = mHeldHandles[key].lock())
            {
                return held;
            }
            handle = takeIdleHandle(ctx);
        }
        // Outside mMutex: resetting or creating a handle calls into CUDA.
        if (handle)
        {
            mResetForNewThread(*handle);
        }
        else
        {
            handle = mCreate();
        }
        std::shared_ptr<T> result{handle.release(), [this, key](T* released) { giveBack(key, released); }};
        std::lock_guard<std::mutex> lk{mMutex};
        mHeldHandles[key] = result;
        return result;
    }

private:
    using HolderKey = std::tuple<CUcontext, std::thread::id>;

    // Requires mMutex.
    std::unique_ptr<T> takeIdleHandle(CUcontext ctx)
    {
        auto& idleHandles = mIdleHandles[ctx];
        if (idleHandles.empty())
        {
            return nullptr;
        }
        auto handle = std::move(idleHandles.back());
        idleHandles.pop_back();
        return handle;
    }

    // Deleter of the std::shared_ptr returned by acquire().
    void giveBack(HolderKey const& key, T* handle)
    {
        std::lock_guard<std::mutex> lk{mMutex};
        mIdleHandles[std::get<CUcontext>(key)].emplace_back(handle);
        // Forget the holder so that mHeldHandles does not grow with exited threads. Keep the entry if it already
        // observes a newer handle: when the last owner was another thread, the holder may have acquired again before
        // this call took mMutex. expired() rather than lock(): a std::shared_ptr taken here could become the last
        // owner and re-enter giveBack() while mMutex is held.
        auto const it = mHeldHandles.find(key);
        if (it != mHeldHandles.end() && it->second.expired())
        {
            mHeldHandles.erase(it);
        }
    }

    CreateFunc mCreate;
    ResetFunc mResetForNewThread;
    // Guards only the two tables below; no CUDA call is made while holding it.
    std::mutex mMutex;
    std::unordered_map<HolderKey, std::weak_ptr<T>, common::op::OpCustomHash<HolderKey>> mHeldHandles;
    // Each std::unique_ptr owns only the box that holds a handle value: freeing a box leaks the handle, it never
    // destroys it.
    std::unordered_map<CUcontext, std::vector<std::unique_ptr<T>>> mIdleHandles;
};

// Structure to hold memory information
struct MemoryInfo
{
    size_t free_mb;
    size_t total_mb;
    float free_percent;
};

// Helper function to get current memory information
MemoryInfo getMemoryInfo()
{
    size_t free_mem = 0, total_mem = 0;
    TLLM_CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

    size_t const free_mb = free_mem / (1024 * 1024);
    size_t const total_mb = total_mem / (1024 * 1024);
    float const free_percent = (total_mem > 0) ? (static_cast<float>(free_mem) / total_mem * 100.0f) : 0.0f;

    return {free_mb, total_mb, free_percent};
}

// Helper function to log current memory usage
void logMemoryUsage(char const* operation, CUcontext ctx)
{
    auto const mem = getMemoryInfo();
    TLLM_LOG_DEBUG("%s: Context=%p, Free Memory=%zu MB (%.1f%%), Total=%zu MB", operation, ctx, mem.free_mb,
        mem.free_percent, mem.total_mb);
}

// Helper function to throw
void throwCublasErrorWithMemInfo(char const* operation, CUcontext ctx, cublasStatus_t status)
{
    auto const mem = getMemoryInfo();
    TLLM_THROW(
        "Failed to create %s. "
        "Status: %d, Context: %p, Free Memory: %zu MB (%.1f%%), Total: %zu MB. "
        "Consider reducing kv_cache_config.free_gpu_memory_fraction.",
        operation, status, ctx, mem.free_mb, mem.free_percent, mem.total_mb);
}

} // namespace

std::shared_ptr<cublasHandle_t> getCublasHandle()
{
    static auto* const pool = new PerCudaCtxHandlePool<cublasHandle_t>(
        []() -> auto
        {
            CUcontext ctx = getCurrentCudaCtx();
            logMemoryUsage("Creating cublas handle", ctx);

            auto handle = std::make_unique<cublasHandle_t>();
            auto status = cublasCreate(handle.get());

            if (status != CUBLAS_STATUS_SUCCESS)
            {
                throwCublasErrorWithMemInfo("cublas handle", ctx, status);
            }

            return handle;
        },
        [](cublasHandle_t handle)
        {
            // The previous thread may have left its stream and workspace on the handle (CublasMMWrapper sets both),
            // and both may be gone with it. cublasSetStream also resets the workspace to the default one.
            TLLM_CUDA_CHECK(cublasSetStream(handle, nullptr));
        });
    return pool->acquire();
}

std::shared_ptr<cublasLtHandle_t> getCublasLtHandle()
{
    static auto* const pool = new PerCudaCtxHandlePool<cublasLtHandle_t>(
        []() -> auto
        {
            CUcontext ctx = getCurrentCudaCtx();
            logMemoryUsage("Creating cublasLt handle", ctx);

            auto handle = std::make_unique<cublasLtHandle_t>();
            auto status = cublasLtCreate(handle.get());

            if (status != CUBLAS_STATUS_SUCCESS)
            {
                throwCublasErrorWithMemInfo("cublasLt handle", ctx, status);
            }

            return handle;
        },
        // cuBLASLt takes the stream and workspace with each call, so a handle keeps no state of the previous thread.
        [](cublasLtHandle_t) {});
    return pool->acquire();
}

TRTLLM_NAMESPACE_END
