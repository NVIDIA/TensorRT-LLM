/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/runtime/utils/multiDeviceUtils.h"

#if ENABLE_MULTI_DEVICE
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <nccl.h>
#include <torch/extension.h>
#endif

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if ENABLE_MULTI_DEVICE

#ifndef TLLM_NCCL_WINDOW_LIFECYCLE_ASSERT
#define TLLM_NCCL_WINDOW_LIFECYCLE_ASSERT 1
#endif

// TLLM_NCCL_CHECK (throw on failure) is provided by multiDeviceUtils.h.

// Warn-only variant: log a warning on NCCL failure but do not throw or abort.
// Use for cleanup/secondary operations where an NCCL error is non-fatal (e.g. ncclMemFree on an error path).
#define TLLM_NCCL_CHECK_WARN(cmd)                                                                                      \
    do                                                                                                                 \
    {                                                                                                                  \
        ncclResult_t const _tllm_nccl_warn_r = (cmd);                                                                  \
        if (TLLM_UNLIKELY(_tllm_nccl_warn_r != ncclSuccess))                                                           \
        {                                                                                                              \
            TLLM_LOG_WARNING(                                                                                          \
                "NCCL error in %s (%s:%d): %s", #cmd, __FILE__, __LINE__, ncclGetErrorString(_tllm_nccl_warn_r));      \
        }                                                                                                              \
    } while (0)

TRTLLM_NAMESPACE_BEGIN

namespace common::nccl_util
{

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
// NCCL Version Check
//==============================================================================

// Returns true if NCCL window buffers (ncclMemAlloc / ncclCommWindowRegister)
// are supported for the given real SM version, integrated-device flag, and runtime NCCL version.
// Exposed for focused unit testing of platform/version gates.
bool isNcclWindowSupportedForPlatform(int realSmVersion, bool isIntegrated, int ncclRuntimeVersion);

// Returns true if the compile-time and runtime NCCL versions support window buffers
// and the current CUDA device is not in a known-unsupported platform/version set.
bool isNcclWindowSupported();

//==============================================================================
// NCCL Window Buffer Allocation
//==============================================================================

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)

// Represents a buffer with an associated NCCL window
struct NCCLWindowBuffer
{
    void* ptr;           // Device pointer (same as UBBuffer.addr)
    int handle;          // Buffer handle/index (for compatibility with UB interface)
    size_t size;         // Size in bytes
    ncclWindow_t window; // NCCL window handle
    uint64_t leaseId;    // Changes every time this allocation is checked out

    NCCLWindowBuffer(void* p = nullptr, int h = -1, size_t s = 0, ncclWindow_t w = nullptr, uint64_t l = 0)
        : ptr(p)
        , handle(h)
        , size(s)
        , window(w)
        , leaseId(l)
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
    // Eager requests only reuse unowned buffers. A request made during CUDA graph capture first
    // reuses buffers owned by that PyTorch graph memory pool, then may claim an unowned buffer for it.
    // Uses best-fit strategy: selects the smallest eligible buffer that meets the size requirement.
    // New allocation/registration is prohibited during capture.
    NCCLWindowBuffer requestBuffer(ncclComm_t comm, size_t size);

    // Select the allocation domain on this host thread. owner == -1 explicitly selects eager;
    // owner >= 0 selects a stable PyTorch graph-memory-pool owner. Other values are invalid.
    void setGraphPoolOwner(int64_t owner);
    // Return this owner's idle buffers to eager use. This changes best-fit eligibility and
    // therefore must be called at the same explicit lifecycle boundary on every rank. Buffers
    // still in use are returned when their deterministic tensor scope ends.
    void releaseGraphPoolOwner(int64_t owner);

    // Search for a buffer by pointer. Returns an invalid buffer if not found.
    // This matches the UBManager.search_buffer() interface.
    NCCLWindowBuffer searchBuffer(ncclComm_t comm, void* ptr) const;

    // Release a buffer that has not been submitted to a CUDA stream, making it reusable on any stream.
    // This overload is intended for preallocation and focused allocator tests.
    void releaseBuffer(ncclComm_t comm, void* ptr);

    // Release a buffer after its final use has been enqueued on stream. The buffer may be reused on
    // that stream; requests on other streams leave it untouched.
    // Returns false when ptr is not a currently active registered buffer.
    bool releaseBuffer(ncclComm_t comm, void* ptr, cudaStream_t stream);

    // Snapshot explicitly released buffers before the caller synchronizes the device, then mark
    // only those releases reusable on any stream. PyTorch CUDA-graph setup uses its existing
    // capture-begin synchronization between these calls. Destructor-released/quarantined buffers
    // remain ineligible.
    uint64_t getBufferReleaseEpoch();
    void promoteBufferReleases(int device, uint64_t releaseEpoch);

    // Bind a PyTorch storage to the exact checkout that backs it. Views share StorageImpl, so
    // explicit release through any view still resolves the original lease generation.
    void registerTensorLease(ncclComm_t comm, c10::StorageImpl const* storage, void* ptr, uint64_t leaseId, int device);
    bool releaseTensorBuffer(c10::StorageImpl const* storage, cudaStream_t stream);
    bool releaseTensorBuffer(ncclComm_t comm, c10::StorageImpl const* storage, cudaStream_t stream);

    // A scope adopts window-backed input storages and all window tensors created while it is
    // active. On successful exit, returned storages escape to the parent scope and every other
    // lease is released on stream. Failed scopes preserve escaped inputs but quarantine every other
    // lease instead of making partially submitted work reusable. Scopes are keyed by CUDA device and
    // host execution thread, so allocations made while the caller temporarily selects an auxiliary
    // stream still join the active compiled scope. Before ending a scope, callers must order
    // auxiliary-stream consumers before the supplied completion stream.
    void beginTensorLeaseScope(std::vector<c10::StorageImpl const*> const& inputStorages, int device);
    void endTensorLeaseScope(
        std::vector<c10::StorageImpl const*> const& outputStorages, int device, cudaStream_t stream, bool failed);

    // Destructor fallback. The lease ID prevents a late tensor destructor from releasing a newer
    // checkout of the same allocation. Fallback-released buffers are quarantined until communicator
    // teardown because destructor timing and stream identity are not rank-deterministic.
    void releaseBufferFromDestructor(ncclComm_t comm, void* ptr, uint64_t leaseId);

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
    friend class NCCLWindowAllocatorTestAccess;

    NCCLWindowAllocator() = default;
    ~NCCLWindowAllocator() = default;

    // Allocate a new buffer and register it with NCCL as a window
    NCCLWindowBuffer allocateAndRegisterBuffer(ncclComm_t comm, size_t size, int handle);

    // Record a failed new symmetric allocation (assumes mMutex is already locked).
    void recordSymmetricFailureLocked(ncclComm_t comm, size_t size);

    enum class FallbackWarning : uint8_t
    {
        kCaptureStateUnknown = 1U << 0U,
        kCaptureWithoutOwner = 1U << 1U,
        kNoEligibleCaptureBuffer = 1U << 2U,
        kDestructorFallback = 1U << 3U,
    };

    // Record a fallback warning for this communicator and return true only for its first occurrence.
    bool markFallbackWarningLogged(ncclComm_t comm, FallbackWarning warning);
    bool markFallbackWarningLoggedLocked(ncclComm_t comm, FallbackWarning warning);

    using CudaGetLastErrorFunc = cudaError_t (*)();

    // Drain a sticky CUDA error when capture state cannot be queried. Registration is unsafe when
    // the capture state is unknown, so the caller must use an unregistered buffer.
    static bool clearCudaErrorIfCaptureQueryFailed(
        cudaError_t captureError, CudaGetLastErrorFunc getLastError = cudaGetLastError) noexcept;

    // Drain the sticky CUDA error left by a failed symmetric allocation.
    static cudaError_t clearCudaErrorIfSymmetricAllocationFailed(
        int localAllocOk, CudaGetLastErrorFunc getLastError = cudaGetLastError) noexcept;

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
        int device;
        // Empty for eager/ungraphed buffers. A value reserves the buffer for captures sharing
        // that exact PyTorch graph memory pool.
        std::optional<uint64_t> graphPoolOwner;
        bool releaseOwnerWhenUnused{false};
        // An explicitly released buffer is reusable everywhere after a synchronization boundary,
        // or only on the stream containing its final consumer. Neither value means it was released
        // by its destructor fallback and is quarantined.
        bool reusableOnAnyStream{false};
        std::optional<cudaStream_t> reusableStream;
        uint64_t releaseEpoch{0};
    };

    struct TensorLeaseScope;

    struct TensorLease
    {
        ncclComm_t comm;
        void* ptr;
        uint64_t leaseId;
        TensorLeaseScope* scope{nullptr};
    };

    struct TensorLeaseScope
    {
        std::unordered_set<c10::StorageImpl const*> storages;
        // Leases whose PyTorch storage died before the deterministic scope boundary.
        std::vector<TensorLease> detachedLeases;
    };

    using TensorLeaseScopeKey = std::pair<int, std::thread::id>;
    static TensorLeaseScopeKey getTensorLeaseScopeKey(int device);

    bool releaseTensorLeaseLocked(TensorLease const& tensorLease, cudaStream_t stream);
    bool releaseTensorBufferLocked(c10::StorageImpl const* storage, cudaStream_t stream, ncclComm_t expectedComm);
    void quarantineTensorLeaseLocked(TensorLease const& tensorLease);
    void assertLifecycleWarning() const noexcept;

    mutable std::mutex mMutex;
    std::atomic_bool mSuppressLifecycleWarningAssertionsForTest{false};
    std::unordered_map<ncclComm_t, std::vector<BufferEntry>> mBufferPool;
    std::unordered_set<ncclComm_t> mRegisteredComms;
    // Smallest request size that is known to fail collectively for each communicator.
    // Requests below the recorded size may still succeed and already-pooled buffers are always
    // reused before consulting this cache.
    std::unordered_map<ncclComm_t, size_t> mMinSymmetricFailureSize;
    // Bitset of fallback warnings already emitted for each live communicator.
    std::unordered_map<ncclComm_t, uint8_t> mLoggedFallbackWarnings;
    std::unordered_map<c10::StorageImpl const*, TensorLease> mTensorLeases;
    uint64_t mNextLeaseId{1};
    uint64_t mNextReleaseEpoch{1};
    // Deque keeps scope addresses stable while scopes are pushed onto the per-thread stack.
    std::map<TensorLeaseScopeKey, std::deque<TensorLeaseScope>> mTensorLeaseScopeStacks;
};

// RAII wrapper for NCCL window buffers
class ScopedNCCLWindowBuffer
{
public:
    ScopedNCCLWindowBuffer(std::shared_ptr<ncclComm_t> comm, size_t size)
        : mComm(std::move(comm))
        , mBuffer{}
    {
        if (mComm && *mComm)
        {
            mBuffer = NCCLWindowAllocator::getInstance().requestBuffer(*mComm, size);
        }
    }

    ~ScopedNCCLWindowBuffer()
    {
        if (mBuffer.isValid())
        {
            NCCLWindowAllocator::getInstance().releaseBuffer(*mComm, mBuffer.ptr);
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
    std::shared_ptr<ncclComm_t> mComm;
    NCCLWindowBuffer mBuffer;
};

// Creates a PyTorch tensor backed by an NCCL window buffer.
// The tensor will automatically release the buffer back to the pool when destroyed.
// This is analogous to torch_ext::create_userbuffers_tensor() but for NCCLWindowAllocator.
inline std::pair<torch::Tensor, NCCLWindowBuffer> createNCCLWindowTensor(
    std::shared_ptr<ncclComm_t> comm, at::IntArrayRef shape, torch::ScalarType dtype)
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

    if (!comm || !*comm)
    {
        TLLM_LOG_DEBUG("[createNCCLWindowTensor] null comm; returning invalid buffer");
        return std::make_pair(torch::Tensor(), NCCLWindowBuffer());
    }

    // Expected resource failures, unknown capture state, and captures without a graph-pool owner
    // return an invalid buffer so callers can use their existing unregistered fallback. Other
    // programming errors must propagate instead of silently changing the collective implementation.
    buffer = allocator.requestBuffer(*comm, buffer_size);

    // Defensive validation: ensure buffer is valid before proceeding
    if (!buffer.isValid())
    {
        TLLM_LOG_DEBUG("[createNCCLWindowTensor] invalid buffer returned from requestBuffer; returning invalid buffer");
        return std::make_pair(torch::Tensor(), NCCLWindowBuffer());
    }

    // Create custom deleter that releases the buffer
    auto deleter = [comm, ptr = buffer.ptr, leaseId = buffer.leaseId](void*)
    { NCCLWindowAllocator::getInstance().releaseBufferFromDestructor(*comm, ptr, leaseId); };

    // Create tensor from the buffer
    auto tensor = torch::from_blob(buffer.ptr, shape, strides_vec, deleter, torch::dtype(dtype).device(torch::kCUDA));
    auto const device = tensor.get_device();
    allocator.registerTensorLease(*comm, tensor.storage().unsafeGetStorageImpl(), buffer.ptr, buffer.leaseId, device);

    return std::make_pair(tensor, buffer);
}

#endif // NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)

} // namespace common::nccl_util

TRTLLM_NAMESPACE_END

#endif // ENABLE_MULTI_DEVICE
