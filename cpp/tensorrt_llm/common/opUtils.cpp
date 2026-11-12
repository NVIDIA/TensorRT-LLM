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

#include "cuda.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

TRTLLM_NAMESPACE_BEGIN

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
