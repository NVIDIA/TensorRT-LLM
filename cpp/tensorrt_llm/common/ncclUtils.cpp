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
// NCCLHelper Implementation
//==============================================================================

NCCLHelper& NCCLHelper::getInstance()
{
    static NCCLHelper instance;
    return instance;
}

NCCLHelper::NCCLHelper()
    : mLibraryHandle(nullptr)
    , mNCCLCommWindowRegister(nullptr)
    , mNCCLMemAlloc(nullptr)
    , mIsLoaded(false)
{
    loadNCCLLibrary();
}

NCCLHelper::~NCCLHelper()
{
    if (mLibraryHandle)
    {
#ifdef _WIN32
        FreeLibrary(mLibraryHandle);
#else
        dlclose(mLibraryHandle);
#endif
        mLibraryHandle = nullptr;
    }
}

void NCCLHelper::loadNCCLLibrary()
{
    try
    {
#ifdef _WIN32
        char const* libraryNames[] = {"nccl.dll"};
#else
        char const* libraryNames[] = {"libnccl.so"};
#endif

        for (auto const* name : libraryNames)
        {
            mLibraryHandle = loadLibraryHandle(name);
            if (mLibraryHandle)
            {
                TLLM_LOG_INFO("Successfully loaded NCCL library: %s", name);
                break;
            }
        }

        if (!mLibraryHandle)
        {
            TLLM_LOG_WARNING("Failed to load NCCL library");
            return;
        }

        // Load the required symbols
        mNCCLCommWindowRegister
            = reinterpret_cast<ncclCommWindowRegisterFunc>(getSymbolAddress(mLibraryHandle, "ncclCommWindowRegister"));

        mNCCLMemAlloc = reinterpret_cast<ncclMemAllocFunc>(getSymbolAddress(mLibraryHandle, "ncclMemAlloc"));

        if (mNCCLCommWindowRegister == nullptr)
        {
            TLLM_LOG_WARNING("Failed to load ncclCommWindowRegister symbol, NCCL symmetric will not be supported.");
        }

        if (mNCCLMemAlloc == nullptr)
        {
            TLLM_LOG_WARNING("Failed to load ncclMemAlloc symbol, NCCL symmetric will not be supported.");
        }

        if (mNCCLCommWindowRegister != nullptr && mNCCLMemAlloc != nullptr)
        {
            mIsLoaded = true;
        }
        else
        {
            TLLM_LOG_WARNING(
                "Failed to load required NCCL symbols (both ncclCommWindowRegister and ncclMemAlloc are required)");
        }
    }
    catch (std::exception const& e)
    {
        TLLM_LOG_WARNING("Exception while loading NCCL library: %s", e.what());
    }
}

void* NCCLHelper::loadLibraryHandle(char const* libName)
{
#ifdef _WIN32
    return LoadLibraryA(libName);
#else
    return dlopen(libName, RTLD_LAZY | RTLD_GLOBAL);
#endif
}

void* NCCLHelper::getSymbolAddress(void* handle, char const* symbolName)
{
    if (!handle)
    {
        return nullptr;
    }

#ifdef _WIN32
    return GetProcAddress(static_cast<HMODULE>(handle), symbolName);
#else
    return dlsym(handle, symbolName);
#endif
}

NCCLHelper::ncclCommWindowRegisterFunc NCCLHelper::getNCCLCommWindowRegister()
{
    return mNCCLCommWindowRegister;
}

NCCLHelper::ncclMemAllocFunc NCCLHelper::getNCCLMemAlloc()
{
    return mNCCLMemAlloc;
}

bool NCCLHelper::isLoaded() const
{
    return mIsLoaded;
}

//==============================================================================
// NCCLWindowAllocator Implementation
//==============================================================================

NCCLWindowAllocator& NCCLWindowAllocator::getInstance()
{
    static NCCLWindowAllocator instance;
    return instance;
}

NCCLWindowBuffer NCCLWindowAllocator::requestBuffer(ncclComm_t comm, size_t size)
{
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
    NCCLWindowBuffer buffer;
    buffer.handle = handle;

    // Get NCCL helper for dynamic symbol loading
    auto& ncclHelper = NCCLHelper::getInstance();
    if (!ncclHelper.isLoaded())
    {
        TLLM_THROW("NCCL library could not be loaded for dynamic symbol access");
    }

    auto ncclMemAllocFunc = ncclHelper.getNCCLMemAlloc();
    auto ncclCommWindowRegisterFunc = ncclHelper.getNCCLCommWindowRegister();

    // Defensive checks: both function pointers must be non-null
    if (ncclMemAllocFunc == nullptr)
    {
        TLLM_THROW("ncclMemAlloc function pointer is null, cannot allocate NCCL window buffer");
    }

    if (ncclCommWindowRegisterFunc == nullptr)
    {
        TLLM_THROW("ncclCommWindowRegister function pointer is null, cannot register NCCL window buffer");
    }

    // Allocate device memory using ncclMemAlloc
    ncclResult_t allocResult = ncclMemAllocFunc(&buffer.ptr, size);
    if (allocResult != ncclSuccess)
    {
        TLLM_THROW("ncclMemAlloc failed with error: %d", allocResult);
    }
    buffer.size = size;

    // Register the buffer with NCCL as a window
    ncclResult_t regResult
        = ncclCommWindowRegisterFunc(comm, buffer.ptr, size, &buffer.window, NCCL_WIN_COLL_SYMMETRIC);
    if (regResult != ncclSuccess)
    {
        ncclMemFree(buffer.ptr);
        TLLM_THROW("ncclCommWindowRegister failed with error: %d", regResult);
    }

    TLLM_LOG_TRACE("[NCCLUtil] Allocated and registered NCCL window buffer: handle=%d, ptr=%p, size=%zu, window=%p",
        handle, buffer.ptr, size, static_cast<void*>(buffer.window));

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
    for (auto const& entry : commIt->second)
    {
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

} // namespace tensorrt_llm::common::nccl_util

#endif // ENABLE_MULTI_DEVICE
