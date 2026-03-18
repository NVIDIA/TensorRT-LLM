/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/ncclUtils.h"

#if ENABLE_MULTI_DEVICE

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"
#include <limits>
#include <stdexcept>

namespace
{

// RAII guard for cudaMalloc — frees the pointer on destruction, logging a warning on failure.
struct CudaMallocGuard
{
    void* ptr{nullptr};

    explicit CudaMallocGuard(void* p) noexcept
        : ptr(p)
    {
    }

    ~CudaMallocGuard()
    {
        if (ptr)
        {
            TLLM_CUDA_CHECK_WARN(cudaFree(ptr));
        }
    }

    void* release() noexcept
    {
        void* p = ptr;
        ptr = nullptr;
        return p;
    }

    CudaMallocGuard(CudaMallocGuard const&) = delete;
    CudaMallocGuard& operator=(CudaMallocGuard const&) = delete;
};

// RAII guard for ncclMemAlloc — frees the pointer on destruction, logging a warning on failure.
struct NcclMemGuard
{
    void* ptr{nullptr};

    explicit NcclMemGuard(void* p) noexcept
        : ptr(p)
    {
    }

    ~NcclMemGuard()
    {
        if (ptr)
        {
            TLLM_NCCL_CHECK_WARN(ncclMemFree(ptr));
        }
    }

    void* release() noexcept
    {
        void* p = ptr;
        ptr = nullptr;
        return p;
    }

    NcclMemGuard(NcclMemGuard const&) = delete;
    NcclMemGuard& operator=(NcclMemGuard const&) = delete;
};

} // namespace

namespace tensorrt_llm::common::nccl_util
{

//==============================================================================
// NcclCommResourceManager Implementation
//==============================================================================

NcclCommResourceManager& NcclCommResourceManager::getInstance() noexcept
{
    static NcclCommResourceManager instance;
    return instance;
}

NcclCommResourceManager::~NcclCommResourceManager()
{
    // Mark that we're in destruction to prevent cleanup attempts from deleters
    // that may run during static destruction
    mIsDestroying.store(true, std::memory_order_release);

    // Proactively clean up all resources before destruction
    // This ensures cleanup happens in a controlled manner before static destruction
    std::vector<std::pair<ncclComm_t, std::vector<ResourceEntry>>> allResources;

    {
        std::lock_guard<std::mutex> lock(mMutex);
        // Move all resources out of the map
        allResources.reserve(mCommResources.size());
        for (auto& [comm, resources] : mCommResources)
        {
            allResources.emplace_back(comm, std::move(resources));
        }
        mCommResources.clear();
    }

    // Clean up all resources outside the lock
    // Note: We don't call ncclCommDestroy here - that's the responsibility
    // of the shared_ptr deleter. We just clean up registered resources.
    for (auto& [comm, resources] : allResources)
    {
        for (auto& [cleanup, name] : resources)
        {
            try
            {
                cleanup();
            }
            catch (...)
            {
                // Ignore exceptions during destruction
            }
        }
    }
}

void NcclCommResourceManager::registerResource(ncclComm_t comm, ResourceCleanupFunc cleanup, char const* debugName)
{
    if (!comm)
    {
        TLLM_LOG_WARNING("[NCCLUtil] Attempted to register resource for null NCCL comm");
        return;
    }

    std::lock_guard<std::mutex> lock(mMutex);
    auto& resources = mCommResources[comm];
    resources.emplace_back(std::move(cleanup), debugName ? debugName : "unnamed");

    TLLM_LOG_TRACE("[NCCLUtil] Registered resource '%s' for NCCL comm %p (total: %zu)",
        debugName ? debugName : "unnamed", static_cast<void*>(comm), resources.size());
}

void NcclCommResourceManager::cleanupResources(ncclComm_t comm) noexcept
{
    if (!comm)
    {
        return;
    }

    // Check if we're in the process of being destroyed
    // If so, skip cleanup - the destructor will handle it proactively
    if (mIsDestroying.load(std::memory_order_acquire))
    {
        return;
    }

    std::vector<ResourceEntry> resourcesToClean;

    {
        // During static destruction, mutex and logging may not be safe.
        // Use try-catch to handle any issues gracefully.
        try
        {
            std::lock_guard<std::mutex> lock(mMutex);

            // Double-check after acquiring lock (destruction may have started)
            if (mIsDestroying.load(std::memory_order_acquire))
            {
                return;
            }

            auto it = mCommResources.find(comm);
            if (it == mCommResources.end())
            {
                // Nothing registered for this comm, nothing to clean up
                return;
            }

            // Move resources out (preserves order) and remove from map
            resourcesToClean = std::move(it->second);
            mCommResources.erase(it);

            // Logging may fail during static destruction, so wrap in try-catch
            try
            {
                TLLM_LOG_TRACE("[NCCLUtil] Cleaning up %zu resources for NCCL comm %p", resourcesToClean.size(),
                    static_cast<void*>(comm));
            }
            catch (...)
            {
                // Ignore logging failures during static destruction
            }
        }
        catch (...)
        {
            // If mutex access fails during static destruction, just return.
            // This prevents segfaults when the singleton is being destroyed.
            return;
        }
    }

    // Clean up outside the lock to avoid deadlocks if cleanup functions try to access the manager
    // Order is preserved: resources are cleaned up in registration order
    for (auto& [cleanup, name] : resourcesToClean)
    {
        try
        {
            // Logging may fail during static destruction, so wrap in try-catch
            try
            {
                TLLM_LOG_TRACE(
                    "[NCCLUtil] Cleaning up resource '%s' for NCCL comm %p", name.c_str(), static_cast<void*>(comm));
            }
            catch (...)
            {
                // Ignore logging failures during static destruction
            }
            cleanup();
        }
        catch (std::exception const& e)
        {
            try
            {
                TLLM_LOG_ERROR("[NCCLUtil] Exception during cleanup of resource '%s' for NCCL comm %p: %s",
                    name.c_str(), static_cast<void*>(comm), e.what());
            }
            catch (...)
            {
                // Ignore logging failures during static destruction
            }
        }
        catch (...)
        {
            try
            {
                TLLM_LOG_ERROR("[NCCLUtil] Unknown exception during cleanup of resource '%s' for NCCL comm %p",
                    name.c_str(), static_cast<void*>(comm));
            }
            catch (...)
            {
                // Ignore logging failures during static destruction
            }
        }
    }
}

bool NcclCommResourceManager::hasResources(ncclComm_t comm) const noexcept
{
    std::lock_guard<std::mutex> lock(mMutex);
    return mCommResources.find(comm) != mCommResources.end();
}

size_t NcclCommResourceManager::getResourceCount(ncclComm_t comm) const noexcept
{
    std::lock_guard<std::mutex> lock(mMutex);
    auto it = mCommResources.find(comm);
    return it != mCommResources.end() ? it->second.size() : 0;
}

//==============================================================================
// NCCLWindowAllocator Implementation
//==============================================================================

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)

NCCLWindowAllocator& NCCLWindowAllocator::getInstance()
{
    static NCCLWindowAllocator instance;
    return instance;
}

NCCLWindowBuffer NCCLWindowAllocator::requestBuffer(ncclComm_t comm, size_t size)
{
    // One-time runtime version check: the runtime NCCL library must also support window buffers.
    static std::once_flag versionCheckFlag;
    static bool runtimeVersionOk = false;
    std::call_once(versionCheckFlag,
        []()
        {
            int version = 0;
            if (ncclGetVersion(&version) == ncclSuccess && version >= NCCL_VERSION(2, 28, 0))
            {
                runtimeVersionOk = true;
            }
            else
            {
                TLLM_LOG_WARNING(
                    "[NCCLUtil] NCCL runtime version %d.%d.%d does not support window buffers; "
                    "falling back to regular tensors.",
                    version / 10000, (version % 10000) / 100, version % 100);
            }
        });
    if (!runtimeVersionOk)
    {
        return NCCLWindowBuffer();
    }

    TLLM_CHECK_WITH_INFO(comm != nullptr, "NCCL communicator cannot be null");
    TLLM_CHECK_WITH_INFO(size > 0, "Buffer size must be greater than 0");

    std::lock_guard<std::mutex> lock(mMutex);

    // Register cleanup callback for this communicator if not already registered
    // This is cheap even if no buffers exist yet - cleanup will just return early
    registerBufferCleanup(comm);

    // Check if we have an available buffer of at least the requested size for this communicator
    // Use best-fit: find the smallest buffer that's >= requested size
    auto& commBuffers = mBufferPool[comm];
    auto bestFit = commBuffers.end();
    size_t bestFitSize = std::numeric_limits<size_t>::max();

    for (auto it = commBuffers.begin(); it != commBuffers.end(); ++it)
    {
        if (!it->inUse && it->buffer.size >= size && it->buffer.size < bestFitSize)
        {
            bestFit = it;
            bestFitSize = it->buffer.size;
        }
    }

    if (bestFit != commBuffers.end())
    {
        bestFit->inUse = true;
        TLLM_LOG_TRACE(
            "[NCCLUtil] Reusing NCCL window buffer for comm %p: handle=%d, ptr=%p, size=%zu (requested: %zu)",
            static_cast<void*>(comm), bestFit->buffer.handle, bestFit->buffer.ptr, bestFit->buffer.size, size);
        return bestFit->buffer;
    }

    // No available buffer found, avoid registration during CUDA graph capture
    auto stream = at::cuda::getCurrentCUDAStream();
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    auto capture_err = cudaStreamIsCapturing(stream, &capture_status);
    if (capture_err != cudaSuccess)
    {
        TLLM_LOG_DEBUG("[NCCLUtil] cudaStreamIsCapturing failed: %s", cudaGetErrorString(capture_err));
    }
    if (capture_err == cudaSuccess && capture_status != cudaStreamCaptureStatusNone)
    {
        TLLM_LOG_DEBUG("[NCCLUtil] Skipping NCCL window allocation during capture for comm %p (requested: %zu)",
            static_cast<void*>(comm), size);
        return NCCLWindowBuffer();
    }

    // No available buffer found, allocate a new one
    TLLM_LOG_TRACE(
        "[NCCLUtil] Allocating new NCCL window buffer for comm %p, size=%zu", static_cast<void*>(comm), size);
    int handle = static_cast<int>(commBuffers.size());
    NCCLWindowBuffer buffer = allocateAndRegisterBuffer(comm, size, handle);
    commBuffers.push_back({buffer, true});

    return buffer;
}

NCCLWindowBuffer NCCLWindowAllocator::searchBuffer(ncclComm_t comm, void* ptr) const
{
    if (!comm || !ptr)
    {
        return NCCLWindowBuffer();
    }

    std::lock_guard<std::mutex> lock(mMutex);
    return searchBufferLocked(comm, ptr);
}

void NCCLWindowAllocator::releaseBuffer(ncclComm_t comm, void* ptr)
{
    if (!comm || !ptr)
    {
        return;
    }

    std::lock_guard<std::mutex> lock(mMutex);
    auto commIt = mBufferPool.find(comm);
    if (commIt == mBufferPool.end())
    {
        TLLM_LOG_WARNING(
            "[NCCLUtil] Attempted to release buffer %p for unknown comm %p", ptr, static_cast<void*>(comm));
        return;
    }

    for (auto& entry : commIt->second)
    {
        if (entry.buffer.ptr == ptr)
        {
            entry.inUse = false;
            TLLM_LOG_TRACE("[NCCLUtil] Released NCCL window buffer for comm %p: ptr=%p", static_cast<void*>(comm), ptr);
            return;
        }
    }

    TLLM_LOG_WARNING("[NCCLUtil] Attempted to release unknown buffer %p for comm %p", ptr, static_cast<void*>(comm));
}

ncclWindow_t NCCLWindowAllocator::getWindow(ncclComm_t comm, void* ptr) const
{
    std::lock_guard<std::mutex> lock(mMutex);
    NCCLWindowBuffer buffer = searchBufferLocked(comm, ptr);
    return buffer.isValid() ? buffer.window : nullptr;
}

size_t NCCLWindowAllocator::getSize(ncclComm_t comm, void* ptr) const
{
    std::lock_guard<std::mutex> lock(mMutex);
    NCCLWindowBuffer buffer = searchBufferLocked(comm, ptr);
    return buffer.isValid() ? buffer.size : 0;
}

NCCLWindowBuffer NCCLWindowAllocator::getBufferInfo(ncclComm_t comm, void* ptr) const
{
    std::lock_guard<std::mutex> lock(mMutex);
    return searchBufferLocked(comm, ptr);
}

size_t NCCLWindowAllocator::getBufferCount(ncclComm_t comm) const
{
    std::lock_guard<std::mutex> lock(mMutex);
    auto commIt = mBufferPool.find(comm);
    return commIt != mBufferPool.end() ? commIt->second.size() : 0;
}

size_t NCCLWindowAllocator::getBufferInUseCount(ncclComm_t comm) const
{
    std::lock_guard<std::mutex> lock(mMutex);
    auto commIt = mBufferPool.find(comm);
    if (commIt == mBufferPool.end())
    {
        return 0;
    }

    size_t count = 0;
    for (auto const& entry : commIt->second)
    {
        if (entry.inUse)
        {
            ++count;
        }
    }
    return count;
}

bool NCCLWindowAllocator::isCommValid(ncclComm_t comm) const noexcept
{
    // Simply check for null - all non-null comms are valid
    // We don't track cleaned-up comms because NCCL can reuse memory addresses,
    // making pointer-based tracking unreliable. New comms will be registered when used.
    return comm != nullptr;
}

NCCLWindowBuffer NCCLWindowAllocator::allocateAndRegisterBuffer(ncclComm_t comm, size_t size, int handle)
{
    // Step 1: Allocate symmetric memory (per-rank, non-collective — can fail asymmetrically).
    void* ncclPtr = nullptr;
    ncclResult_t allocResult = ncclMemAlloc(&ncclPtr, size);
    int localAllocOk = (allocResult == ncclSuccess) ? 1 : 0;
    if (!localAllocOk)
    {
        TLLM_LOG_WARNING(
            "[NCCLUtil] ncclMemAlloc failed on this rank (error: %d, size=%zu); "
            "synchronizing with other ranks before aborting window registration.",
            allocResult, size);
    }
    NcclMemGuard ncclGuard{ncclPtr}; // frees ncclPtr on any early return or exception

    // Step 2: ncclCommWindowRegister is collective — if any rank skips it, all other ranks hang.
    // Synchronize the per-rank alloc status using a small cudaMalloc flag (not ncclMemAlloc, so
    // OOM on symmetric memory does not prevent us from allocating the flag).
    int* rankSyncFlag = nullptr;
    if (cudaMalloc(&rankSyncFlag, sizeof(int)) != cudaSuccess)
    {
        TLLM_THROW("[NCCLUtil] cudaMalloc for rank-sync flag failed; cannot coordinate safely across ranks.");
    }
    CudaMallocGuard flagGuard{rankSyncFlag}; // frees rankSyncFlag on any early return or exception

    // Step 3: Populate flag, reduce with min across ranks (0 if any rank failed), then read back.
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    TLLM_CUDA_CHECK(cudaMemcpy(rankSyncFlag, &localAllocOk, sizeof(int), cudaMemcpyHostToDevice));

    ncclResult_t reduceResult = ncclAllReduce(rankSyncFlag, rankSyncFlag, 1, ncclInt32, ncclMin, comm, stream);
    if (reduceResult != ncclSuccess)
    {
        TLLM_LOG_WARNING(
            "[NCCLUtil] ncclAllReduce for rank-sync flag failed (error: %d); "
            "aborting window registration on this rank.",
            reduceResult);
        return NCCLWindowBuffer{}; // guards free rankSyncFlag and ncclPtr
    }
    TLLM_CUDA_CHECK_WARN(cudaStreamSynchronize(stream));

    int allAllocOk = 0;
    cudaError_t d2hErr = cudaMemcpy(&allAllocOk, rankSyncFlag, sizeof(int), cudaMemcpyDeviceToHost);
    if (d2hErr != cudaSuccess)
    {
        TLLM_LOG_WARNING("[NCCLUtil] cudaMemcpy D2H for rank-sync flag failed: %s; assuming allocation failed.",
            cudaGetErrorString(d2hErr));
        return NCCLWindowBuffer{}; // guards free rankSyncFlag and ncclPtr
    }
    // flagGuard frees rankSyncFlag here at end of its scope

    if (!allAllocOk)
    {
        if (localAllocOk)
        {
            TLLM_LOG_WARNING(
                "[NCCLUtil] ncclMemAlloc failed on at least one other rank; "
                "freeing local allocation (size=%zu) and aborting window registration on all ranks.",
                size);
        }
        return NCCLWindowBuffer{}; // ncclGuard frees ncclPtr
    }

    // Step 4: Register with NCCL as a window (collective — all ranks must reach this call).
    ncclWindow_t window = nullptr;
    ncclResult_t regResult = ncclCommWindowRegister(comm, ncclPtr, size, &window, NCCL_WIN_COLL_SYMMETRIC);
    if (regResult != ncclSuccess)
    {
        TLLM_THROW("ncclCommWindowRegister failed with error: %d", regResult);
        // ncclGuard frees ncclPtr during stack unwinding
    }

    // Step 5: Success — transfer ownership to the returned buffer.
    ncclGuard.release();
    NCCLWindowBuffer buffer{ncclPtr, handle, size, window};
    TLLM_LOG_TRACE("[NCCLUtil] Allocated and registered NCCL window buffer: handle=%d, ptr=%p, size=%zu, window=%p",
        handle, buffer.ptr, buffer.size, static_cast<void*>(buffer.window));
    return buffer;
}

NCCLWindowBuffer NCCLWindowAllocator::searchBufferLocked(ncclComm_t comm, void* ptr) const
{
    auto commIt = mBufferPool.find(comm);
    if (commIt == mBufferPool.end())
    {
        return NCCLWindowBuffer();
    }

    for (auto const& entry : commIt->second)
    {
        if (entry.buffer.ptr == ptr)
        {
            return entry.buffer;
        }
    }

    return NCCLWindowBuffer();
}

void NCCLWindowAllocator::registerBufferCleanup(ncclComm_t comm)
{
    // Don't register if already registered
    if (mRegisteredComms.find(comm) != mRegisteredComms.end())
    {
        return;
    }

    mRegisteredComms.insert(comm);

    // Register cleanup with the resource manager
    NcclCommResourceManager::getInstance().registerResource(
        comm, [this, comm]() { this->cleanupBuffersForComm(comm); }, "NCCLWindowAllocator");
}

void NCCLWindowAllocator::cleanupBuffersForComm(ncclComm_t comm) noexcept
{
    if (!comm)
    {
        return;
    }

    // Synchronize CUDA to ensure all operations using these buffers are complete
    // before we deregister windows and free memory
    cudaError_t cudaErr = cudaDeviceSynchronize();
    if (cudaErr != cudaSuccess)
    {
        TLLM_LOG_WARNING("[NCCLUtil] cudaDeviceSynchronize failed with error: %d before cleanup for comm %p", cudaErr,
            static_cast<void*>(comm));
        // Continue anyway - the sync failure might be from a previous error
    }

    std::lock_guard<std::mutex> lock(mMutex);

    // Check if we've already cleaned up this communicator
    if (mRegisteredComms.find(comm) == mRegisteredComms.end())
    {
        // Already cleaned up or never registered
        return;
    }

    auto commIt = mBufferPool.find(comm);
    if (commIt == mBufferPool.end())
    {
        // No buffers to clean up, but mark as cleaned
        mRegisteredComms.erase(comm);
        return;
    }

    TLLM_LOG_TRACE(
        "[NCCLUtil] Cleaning up %zu NCCL window buffers for comm %p", commIt->second.size(), static_cast<void*>(comm));

    // Check for buffers still in use - this shouldn't happen if cleanup is called properly,
    // but we log a warning if it does
    size_t inUseCount = 0;
    size_t totalBytes = 0;
    for (auto const& entry : commIt->second)
    {
        totalBytes += entry.buffer.size;
        if (entry.inUse)
        {
            ++inUseCount;
        }
    }
    if (inUseCount > 0)
    {
        TLLM_LOG_WARNING(
            "[NCCLUtil] Cleaning up %zu buffers still marked as in-use for comm %p. "
            "This may indicate buffers weren't properly released before cleanup.",
            inUseCount, static_cast<void*>(comm));
    }
    TLLM_LOG_DEBUG("[NCCLUtil] NCCL window allocator teardown for comm %p: %zu buffers, %zu bytes total",
        static_cast<void*>(comm), commIt->second.size(), totalBytes);

    for (auto& entry : commIt->second)
    {
        if (entry.buffer.isValid())
        {
            // Deregister the window - the communicator is still valid at this point
            // (cleanup happens before ncclCommDestroy), but we need to be careful
            // if buffers are still in use by active operations
            if (entry.buffer.window && comm)
            {
                // Note: Even if buffer is marked inUse, we must deregister since
                // the communicator is being destroyed. The communicator is valid,
                // but we should handle potential errors gracefully.
                ncclResult_t result = ncclCommWindowDeregister(comm, entry.buffer.window);
                if (result != ncclSuccess)
                {
                    TLLM_LOG_WARNING(
                        "[NCCLUtil] ncclCommWindowDeregister failed with error: %d for comm %p, "
                        "window %p (buffer inUse: %d)",
                        result, static_cast<void*>(comm), static_cast<void*>(entry.buffer.window), entry.inUse);
                }
            }

            // Free device memory using ncclMemFree
            // This should be safe even if deregister failed
            if (entry.buffer.ptr)
            {
                try
                {
                    ncclResult_t ncclResult = ncclMemFree(entry.buffer.ptr);
                    if (ncclResult != ncclSuccess)
                    {
                        TLLM_LOG_WARNING("[NCCLUtil] ncclMemFree failed with error: %d", ncclResult);
                    }
                }
                catch (...)
                {
                    TLLM_LOG_ERROR("[NCCLUtil] Exception during ncclMemFree for ptr %p", entry.buffer.ptr);
                }
            }

            TLLM_LOG_TRACE(
                "[NCCLUtil] Freed NCCL window buffer: ptr=%p, size=%zu", entry.buffer.ptr, entry.buffer.size);
        }
    }

    mBufferPool.erase(commIt);
    mRegisteredComms.erase(comm);
}

#endif // NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)

} // namespace tensorrt_llm::common::nccl_util

#endif // ENABLE_MULTI_DEVICE
