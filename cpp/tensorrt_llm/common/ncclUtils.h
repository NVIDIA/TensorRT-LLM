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
#pragma once

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"

#if ENABLE_MULTI_DEVICE
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <nccl.h>
#include <torch/extension.h>
#endif

#include <algorithm>
#include <atomic>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if ENABLE_MULTI_DEVICE

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

TRTLLM_NAMESPACE_BEGIN

namespace common::nccl_util
{

//==============================================================================
// NCCL Helper - Dynamic Library Loading
//==============================================================================

// Helper class for dynamically loading NCCL symbols (ncclMemAlloc, ncclCommWindowRegister)
// This allows the code to work with NCCL libraries that may or may not have these symbols
class NCCLHelper
{
public:
    static NCCLHelper& getInstance();

    // Dynamic loading function type definition
    using ncclCommWindowRegisterFunc = ncclResult_t (*)(ncclComm_t, void*, size_t, ncclWindow_t*, int);
    using ncclMemAllocFunc = ncclResult_t (*)(void**, size_t);

    // Get function pointer for ncclCommWindowRegister
    ncclCommWindowRegisterFunc getNCCLCommWindowRegister();

    // Get function pointer for ncclMemAlloc
    ncclMemAllocFunc getNCCLMemAlloc();

    // Check if NCCL library is successfully loaded
    bool isLoaded() const;

    NCCLHelper(NCCLHelper const&) = delete;
    NCCLHelper& operator=(NCCLHelper const&) = delete;
    NCCLHelper(NCCLHelper&&) = delete;
    NCCLHelper& operator=(NCCLHelper&&) = delete;

private:
    NCCLHelper();
    ~NCCLHelper();

    void loadNCCLLibrary();
    void* loadLibraryHandle(char const* libName);
    void* getSymbolAddress(void* handle, char const* symbolName);

#ifdef _WIN32
    HMODULE mLibraryHandle;
#else
    void* mLibraryHandle;
#endif

    ncclCommWindowRegisterFunc mNCCLCommWindowRegister;
    ncclMemAllocFunc mNCCLMemAlloc;
    bool mIsLoaded;
};

//==============================================================================
// NCCL Resource Management
//==============================================================================

// Resource cleanup function type. Called before the NCCL communicator is destroyed.
using ResourceCleanupFunc = std::function<void()>;

// Manages resources associated with NCCL communicators. Thread-safe singleton that maintains
// a pool of resources per NCCL comm. Resources are automatically cleaned up when the
// communicator is destroyed.
class NcclCommResourceManager
{
public:
    static NcclCommResourceManager& getInstance() noexcept;

    // Register a resource cleanup function for a specific NCCL communicator.
    // The cleanup function will be called before ncclCommDestroy.
    // Thread-safe: Uses global mutex to serialize all operations.
    void registerResource(ncclComm_t comm, ResourceCleanupFunc cleanup, char const* debugName = nullptr);

    // Cleanup all resources associated with a communicator. Called automatically by
    // the shared_ptr deleter before ncclCommDestroy.
    // Thread-safe: Uses global mutex to serialize cleanup operations.
    // Order-preserving: Resources are cleaned up in registration order.
    void cleanupResources(ncclComm_t comm) noexcept;

    // Check if a communicator has registered resources.
    bool hasResources(ncclComm_t comm) const noexcept;

    // Get the number of resources registered for a communicator.
    size_t getResourceCount(ncclComm_t comm) const noexcept;

    NcclCommResourceManager(NcclCommResourceManager const&) = delete;
    NcclCommResourceManager& operator=(NcclCommResourceManager const&) = delete;
    NcclCommResourceManager(NcclCommResourceManager&&) = delete;
    NcclCommResourceManager& operator=(NcclCommResourceManager&&) = delete;

private:
    NcclCommResourceManager() = default;
    ~NcclCommResourceManager();

    using ResourceEntry = std::pair<ResourceCleanupFunc, std::string>;

    mutable std::mutex mMutex;
    std::unordered_map<ncclComm_t, std::vector<ResourceEntry>> mCommResources;
    std::atomic<bool> mIsDestroying{false};
};

// RAII helper to register a resource with a NCCL communicator.
// Automatically registers cleanup function on construction.
template <typename ResourceType>
class NcclCommResource
{
public:
    NcclCommResource(ncclComm_t comm, ResourceType&& resource, std::function<void(ResourceType&)> cleanup,
        char const* debugName = nullptr)
        : mComm(comm)
        , mResource(std::forward<ResourceType>(resource))
        , mCleanup(std::move(cleanup))
        , mRegistered(true)
    {
        // Register with the manager
        NcclCommResourceManager::getInstance().registerResource(
            comm,
            [this]()
            {
                if (mCleanup)
                {
                    mCleanup(mResource);
                }
            },
            debugName);
    }

    ResourceType& get()
    {
        return mResource;
    }

    ResourceType const& get() const
    {
        return mResource;
    }

    NcclCommResource(NcclCommResource const&) = delete;
    NcclCommResource& operator=(NcclCommResource const&) = delete;
    NcclCommResource(NcclCommResource&&) = delete;
    NcclCommResource& operator=(NcclCommResource&&) = delete;

private:
    ncclComm_t mComm;
    ResourceType mResource;
    std::function<void(ResourceType&)> mCleanup;
    bool mRegistered;
};

//==============================================================================
// NCCL Window Buffer Allocation
//==============================================================================

// Represents a buffer with an associated NCCL window
struct NCCLWindowBuffer
{
    void* ptr;           // Device pointer (same as UBBuffer.addr)
    int handle;          // Buffer handle/index (for compatibility with UB interface)
    size_t size;         // Size in bytes
    ncclWindow_t window; // NCCL window handle

    NCCLWindowBuffer(void* p = nullptr, int h = -1, size_t s = 0, ncclWindow_t w = nullptr)
        : ptr(p)
        , handle(h)
        , size(s)
        , window(w)
    {
    }

    [[nodiscard]] bool isValid() const
    {
        return ptr != nullptr && handle >= 0 && size > 0 && window != nullptr;
    }

    [[nodiscard]] bool invalid() const
    {
        return !isValid();
    }

    // Alias for compatibility with UBBuffer interface
    void* addr() const
    {
        return ptr;
    }
};

// Manages NCCL window-registered buffers with pooling and automatic cleanup.
// Buffers are tied to the lifetime of their associated NCCL communicator.
class NCCLWindowAllocator
{
public:
    static NCCLWindowAllocator& getInstance();

    // Request a buffer for the given communicator and size.
    // If an unused buffer of at least the requested size exists for this communicator, it will be reused.
    // Uses best-fit strategy: selects the smallest available buffer that meets the size requirement.
    // Otherwise, a new buffer is allocated and registered.
    NCCLWindowBuffer requestBuffer(ncclComm_t comm, size_t size);

    // Search for a buffer by pointer. Returns an invalid buffer if not found.
    // This matches the UBManager.search_buffer() interface.
    NCCLWindowBuffer searchBuffer(ncclComm_t comm, void* ptr) const;

    // Release a buffer back to the pool for potential reuse
    void releaseBuffer(ncclComm_t comm, void* ptr);

    // Get the window handle for a specific buffer pointer
    ncclWindow_t getWindow(ncclComm_t comm, void* ptr) const;

    // Get the size of a specific buffer pointer
    size_t getSize(ncclComm_t comm, void* ptr) const;

    // Get buffer info by pointer
    NCCLWindowBuffer getBufferInfo(ncclComm_t comm, void* ptr) const;

    // Get the number of buffers allocated for a communicator
    size_t getBufferCount(ncclComm_t comm) const;

    // Get the number of buffers in use for a communicator
    size_t getBufferInUseCount(ncclComm_t comm) const;

    // Check if a communicator is valid (non-null)
    // Note: We don't track cleaned-up comms because NCCL can reuse memory addresses.
    // All non-null comms are considered valid and will be registered when first used.
    bool isCommValid(ncclComm_t comm) const noexcept;

    NCCLWindowAllocator(NCCLWindowAllocator const&) = delete;
    NCCLWindowAllocator& operator=(NCCLWindowAllocator const&) = delete;
    NCCLWindowAllocator(NCCLWindowAllocator&&) = delete;
    NCCLWindowAllocator& operator=(NCCLWindowAllocator&&) = delete;

private:
    NCCLWindowAllocator() = default;
    ~NCCLWindowAllocator() = default;

    // Allocate a new buffer and register it with NCCL as a window
    NCCLWindowBuffer allocateAndRegisterBuffer(ncclComm_t comm, size_t size, int handle);

    // Search for a buffer by pointer (assumes mMutex is already locked)
    NCCLWindowBuffer searchBufferLocked(ncclComm_t comm, void* ptr) const;

    // Register cleanup function for all buffers associated with a communicator
    void registerBufferCleanup(ncclComm_t comm);

    // Cleanup all buffers for a specific communicator
    void cleanupBuffersForComm(ncclComm_t comm) noexcept;

    struct BufferEntry
    {
        NCCLWindowBuffer buffer;
        bool inUse;
    };

    mutable std::mutex mMutex;
    std::unordered_map<ncclComm_t, std::vector<BufferEntry>> mBufferPool;
    std::unordered_set<ncclComm_t> mRegisteredComms;
};

// RAII wrapper for NCCL window buffers
class ScopedNCCLWindowBuffer
{
public:
    ScopedNCCLWindowBuffer(ncclComm_t comm, size_t size)
        : mComm(comm)
        , mBuffer(NCCLWindowAllocator::getInstance().requestBuffer(comm, size))
    {
    }

    ~ScopedNCCLWindowBuffer()
    {
        if (mBuffer.isValid())
        {
            NCCLWindowAllocator::getInstance().releaseBuffer(mComm, mBuffer.ptr);
        }
    }

    void* getPtr() const
    {
        return mBuffer.ptr;
    }

    size_t getSize() const
    {
        return mBuffer.size;
    }

    ncclWindow_t getWindow() const
    {
        return mBuffer.window;
    }

    NCCLWindowBuffer const& getBuffer() const
    {
        return mBuffer;
    }

    ScopedNCCLWindowBuffer(ScopedNCCLWindowBuffer const&) = delete;
    ScopedNCCLWindowBuffer& operator=(ScopedNCCLWindowBuffer const&) = delete;
    ScopedNCCLWindowBuffer(ScopedNCCLWindowBuffer&&) = delete;
    ScopedNCCLWindowBuffer& operator=(ScopedNCCLWindowBuffer&&) = delete;

private:
    ncclComm_t mComm;
    NCCLWindowBuffer mBuffer;
};

// Creates a PyTorch tensor backed by an NCCL window buffer.
// The tensor will automatically release the buffer back to the pool when destroyed.
// This is analogous to torch_ext::create_userbuffers_tensor() but for NCCLWindowAllocator.
inline std::pair<torch::Tensor, NCCLWindowBuffer> createNCCLWindowTensor(
    ncclComm_t comm, at::IntArrayRef shape, torch::ScalarType dtype)
{
    // Calculate buffer size
    int64_t buffer_size
        = std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>()) * torch::elementSize(dtype);

    // Calculate strides
    std::vector<int64_t> strides_vec(shape.size());
    if (!shape.empty())
    {
        strides_vec[shape.size() - 1] = 1;
        for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 1; --i)
        {
            strides_vec[i - 1] = strides_vec[i] * shape[i];
        }
    }

    // Request buffer from allocator
    auto& allocator = NCCLWindowAllocator::getInstance();
    NCCLWindowBuffer buffer;

    try
    {
        buffer = allocator.requestBuffer(comm, buffer_size);
    }
    catch (std::exception const& e)
    {
        TLLM_LOG_DEBUG("[createNCCLWindowTensor] requestBuffer failed; returning invalid buffer: %s", e.what());
        return std::make_pair(torch::Tensor(), NCCLWindowBuffer());
    }

    // Defensive validation: ensure buffer is valid before proceeding
    if (!buffer.isValid())
    {
        TLLM_LOG_DEBUG("[createNCCLWindowTensor] invalid buffer returned from requestBuffer; returning invalid buffer");
        return std::make_pair(torch::Tensor(), NCCLWindowBuffer());
    }

    // Create custom deleter that releases the buffer
    auto deleter = [comm, ptr = buffer.ptr](void*) { NCCLWindowAllocator::getInstance().releaseBuffer(comm, ptr); };

    // Create tensor from the buffer
    auto tensor = torch::from_blob(buffer.ptr, shape, strides_vec, deleter, torch::dtype(dtype).device(torch::kCUDA));

    return std::make_pair(tensor, buffer);
}

} // namespace common::nccl_util

TRTLLM_NAMESPACE_END

#endif // ENABLE_MULTI_DEVICE
