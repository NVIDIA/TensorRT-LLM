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

#include "tensorrt_llm/common/ncclUtils.h"

#if ENABLE_MULTI_DEVICE

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"
#include <limits>
#include <stdexcept>

namespace
{

// RAII guard for cudaMalloc. Frees the pointer on destruction, logging a warning on failure.
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

// RAII guard for ncclMemAlloc. Frees the pointer on destruction, logging a warning on failure.
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

namespace
{

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)
constexpr int kNcclWindowMinRuntimeVersion = NCCL_VERSION(2, 28, 0);
constexpr int kNcclGb10WindowFixedVersion = NCCL_VERSION(2, 30, 4);
constexpr int kGb10RealSmVersion = 121;

bool isGb10Platform(int realSmVersion, bool isIntegrated)
{
    return realSmVersion == kGb10RealSmVersion && isIntegrated;
}
#endif

bool queryNcclWindowSupported()
{
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)
    int version = 0;
    if (ncclGetVersion(&version) != ncclSuccess)
    {
        TLLM_LOG_WARNING("[NCCLUtil] Failed to query NCCL runtime version; falling back to regular tensors.");
        return false;
    }

    if (version < kNcclWindowMinRuntimeVersion)
    {
        TLLM_LOG_WARNING(
            "[NCCLUtil] NCCL runtime version %d.%d.%d does not support window buffers; falling back to regular "
            "tensors.",
            version / 10000, (version % 10000) / 100, version % 100);
        return false;
    }

    if (version >= kNcclGb10WindowFixedVersion)
    {
        return true;
    }

    int device = -1;
    cudaError_t const deviceErr = cudaGetDevice(&device);
    if (deviceErr != cudaSuccess)
    {
        TLLM_LOG_WARNING(
            "[NCCLUtil] Failed to query the current CUDA device while checking NCCL window support: %s; "
            "falling back to regular tensors.",
            cudaGetErrorString(deviceErr));
        return false;
    }

    int isIntegrated = 0;
    cudaError_t const integratedErr = cudaDeviceGetAttribute(&isIntegrated, cudaDevAttrIntegrated, device);
    if (integratedErr != cudaSuccess)
    {
        TLLM_LOG_WARNING(
            "[NCCLUtil] Failed to query CUDA integrated-device attribute for device %d while checking NCCL window "
            "support: %s; falling back to regular tensors.",
            device, cudaGetErrorString(integratedErr));
        return false;
    }

    int realSmVersion = -1;
    try
    {
        realSmVersion = tensorrt_llm::common::getSMVersion(/*queryRealSmArch=*/true);
    }
    catch (std::exception const& e)
    {
        TLLM_LOG_WARNING(
            "[NCCLUtil] Failed to query real CUDA SM version while checking NCCL window support: %s; falling back "
            "to regular tensors.",
            e.what());
        return false;
    }

    bool const supported = !isGb10Platform(realSmVersion, isIntegrated != 0);
    if (!supported)
    {
        TLLM_LOG_WARNING(
            "[NCCLUtil] Disabling NCCL window buffers on integrated SM %d with NCCL runtime version %d.%d.%d; "
            "GB10 requires NCCL 2.30.4 or newer for symmetric window registration.",
            realSmVersion, version / 10000, (version % 10000) / 100, version % 100);
    }
    return supported;
#else
    return false;
#endif
}

} // namespace

bool isNcclWindowSupportedForPlatform(int realSmVersion, bool isIntegrated, int ncclRuntimeVersion)
{
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)
    if (ncclRuntimeVersion < kNcclWindowMinRuntimeVersion)
    {
        return false;
    }

    return !(ncclRuntimeVersion < kNcclGb10WindowFixedVersion && isGb10Platform(realSmVersion, isIntegrated));
#else
    (void) realSmVersion;
    (void) isIntegrated;
    (void) ncclRuntimeVersion;
    return false;
#endif
}

bool isNcclWindowSupported()
{
    static std::once_flag supportCheckFlag;
    static bool windowBuffersSupported = false;
    std::call_once(supportCheckFlag, []() { windowBuffersSupported = queryNcclWindowSupported(); });
    return windowBuffersSupported;
}

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

namespace
{
thread_local std::optional<uint64_t> gGraphPoolOwner;
} // namespace

NCCLWindowAllocator& NCCLWindowAllocator::getInstance()
{
    static NCCLWindowAllocator instance;
    return instance;
}

NCCLWindowAllocator::TensorLeaseScopeKey NCCLWindowAllocator::getTensorLeaseScopeKey(int device, cudaStream_t stream)
{
    return {device, reinterpret_cast<uintptr_t>(stream)};
}

void NCCLWindowAllocator::setGraphPoolOwner(int64_t owner)
{
    TLLM_CHECK_WITH_INFO(owner >= -1, "NCCL window pool owner must be -1 (eager) or non-negative");
    gGraphPoolOwner = owner < 0 ? std::nullopt : std::optional<uint64_t>{static_cast<uint64_t>(owner)};
}

void NCCLWindowAllocator::releaseGraphPoolOwner(int64_t owner)
{
    TLLM_CHECK_WITH_INFO(owner >= 0, "NCCL window graph-pool owner must be non-negative");
    auto const graphPoolOwner = static_cast<uint64_t>(owner);
    std::lock_guard<std::mutex> lock(mMutex);
    for (auto& [comm, buffers] : mBufferPool)
    {
        for (auto& entry : buffers)
        {
            if (entry.graphPoolOwner != graphPoolOwner)
            {
                continue;
            }
            if (entry.inUse)
            {
                TLLM_LOG_WARNING(
                    "[NCCLUtil] Deferring release of graph-pool owner %llu for in-use buffer %p on comm %p",
                    graphPoolOwner, entry.buffer.ptr, static_cast<void*>(comm));
                entry.releaseOwnerWhenUnused = true;
            }
            else
            {
                entry.graphPoolOwner.reset();
                entry.releaseOwnerWhenUnused = false;
            }
        }
    }
}

bool NCCLWindowAllocator::markFallbackWarningLogged(ncclComm_t comm, FallbackWarning warning)
{
    std::lock_guard<std::mutex> lock(mMutex);
    registerBufferCleanup(comm);
    return markFallbackWarningLoggedLocked(comm, warning);
}

bool NCCLWindowAllocator::markFallbackWarningLoggedLocked(ncclComm_t comm, FallbackWarning warning)
{
    auto& loggedWarnings = mLoggedFallbackWarnings[comm];
    auto const warningBit = static_cast<uint8_t>(warning);
    bool const firstOccurrence = (loggedWarnings & warningBit) == 0U;
    loggedWarnings |= warningBit;
    return firstOccurrence;
}

NCCLWindowBuffer NCCLWindowAllocator::requestBuffer(ncclComm_t comm, size_t size)
{
    if (!isNcclWindowSupported())
    {
        return NCCLWindowBuffer();
    }

    TLLM_CHECK_WITH_INFO(comm != nullptr, "NCCL communicator cannot be null");
    TLLM_CHECK_WITH_INFO(size > 0, "Buffer size must be greater than 0");

    cudaStreamCaptureStatus captureStatus = cudaStreamCaptureStatusNone;
    cudaStream_t const currentStream = at::cuda::getCurrentCUDAStream();
    auto const captureError = cudaStreamIsCapturing(currentStream, &captureStatus);
    if (clearCudaErrorIfCaptureStateUnknown(captureError))
    {
        // Production capture paths that can request NCCL windows are serialized within each rank
        // and entered in the same SPMD order on every rank. Consequently this error is not a normal
        // registered/plain selection path: it indicates an unsupported overlapping global capture
        // from another host thread. Retain the unregistered fallback as a defensive diagnostic for
        // external or otherwise unwrapped capture code.
        if (markFallbackWarningLogged(comm, FallbackWarning::kCaptureStateUnknown))
        {
            TLLM_LOG_WARNING(
                "[NCCLUtil] CUDA graph capture state is unknown on comm %p because another thread owns a global "
                "capture. Falling back to an unregistered buffer; use thread-local capture or serialize global "
                "captures across threads.",
                static_cast<void*>(comm));
        }
        else
        {
            TLLM_LOG_DEBUG(
                "[NCCLUtil] Repeated unknown CUDA graph capture state on comm %p; using an unregistered buffer.",
                static_cast<void*>(comm));
        }
        return NCCLWindowBuffer();
    }
    TLLM_CUDA_CHECK(captureError);
    bool const isCapturing = captureStatus != cudaStreamCaptureStatusNone;
    if (isCapturing && !gGraphPoolOwner.has_value())
    {
        if (markFallbackWarningLogged(comm, FallbackWarning::kCaptureWithoutOwner))
        {
            TLLM_LOG_WARNING(
                "[NCCLUtil] NCCL symmetric window requested during CUDA graph capture on comm %p without a PyTorch "
                "graph-pool owner. Refusing unsafe window reuse and falling back to an unregistered buffer; wrap "
                "this capture with nccl_window_graph_capture or nccl_window_graph_owner.",
                static_cast<void*>(comm));
        }
        else
        {
            TLLM_LOG_DEBUG(
                "[NCCLUtil] Repeated CUDA graph capture without a graph-pool owner on comm %p; using an "
                "unregistered buffer.",
                static_cast<void*>(comm));
        }
        return NCCLWindowBuffer();
    }
    TLLM_CHECK_WITH_INFO(isCapturing || !gGraphPoolOwner.has_value(),
        "NCCL window graph-pool owner leaked into an eager buffer request");
    uint64_t const graphOwner = gGraphPoolOwner.value_or(0);

    std::lock_guard<std::mutex> lock(mMutex);

    // Register cleanup callback for this communicator if not already registered
    // This is cheap even if no buffers exist yet - cleanup will just return early
    registerBufferCleanup(comm);

    // Check if we have an eligible buffer of at least the requested size for this communicator
    // Use best-fit: find the smallest eligible buffer that's >= requested size
    auto& commBuffers = mBufferPool[comm];
    auto findBestFit = [&](std::optional<uint64_t> const& graphPoolOwner)
    {
        auto bestFit = commBuffers.end();
        size_t bestFitSize = std::numeric_limits<size_t>::max();
        for (auto it = commBuffers.begin(); it != commBuffers.end(); ++it)
        {
            bool const streamEligible
                = it->reusableOnAnyStream || (it->reusableStream.has_value() && *it->reusableStream == currentStream);
            if (!it->inUse && streamEligible && it->graphPoolOwner == graphPoolOwner && it->buffer.size >= size
                && it->buffer.size < bestFitSize)
            {
                bestFit = it;
                bestFitSize = it->buffer.size;
            }
        }
        return bestFit;
    };

    // Captures first reuse their pool's buffers, then may claim a free eager buffer.
    auto bestFit = isCapturing ? findBestFit(graphOwner) : findBestFit(std::nullopt);
    if (bestFit == commBuffers.end() && isCapturing)
    {
        bestFit = findBestFit(std::nullopt);
        if (bestFit != commBuffers.end())
        {
            bestFit->graphPoolOwner = graphOwner;
        }
    }

    if (bestFit != commBuffers.end())
    {
        bestFit->inUse = true;
        bestFit->reusableOnAnyStream = false;
        bestFit->reusableStream.reset();
        bestFit->buffer.leaseId = mNextLeaseId++;
        TLLM_LOG_TRACE(
            "[NCCLUtil] Reusing NCCL window buffer for comm %p: handle=%d, ptr=%p, size=%zu (requested: %zu)",
            static_cast<void*>(comm), bestFit->buffer.handle, bestFit->buffer.ptr, bestFit->buffer.size, size);
        return bestFit->buffer;
    }

    // Registration is not capture-safe. A capture without a suitable dedicated/eager buffer must
    // use the existing unregistered fallback path. Successful production capture lifecycles, window
    // requests, tensor-scope exits, and explicit graph-owner releases execute in identical SPMD order.
    // Failure and destructor paths abandon or quarantine their buffers instead of changing eligibility.
    // Cross-stream reuse is enabled only at an explicit device-synchronization boundary. Together,
    // these rules keep best-fit selection rank-symmetric: GC timing cannot change pool
    // history, and an NCCL symmetric collective never mixes window-backed and plain pointers.
    if (isCapturing)
    {
        if (markFallbackWarningLoggedLocked(comm, FallbackWarning::kNoEligibleCaptureBuffer))
        {
            TLLM_LOG_WARNING(
                "[NCCLUtil] No eligible NCCL window buffer for capture owner %llu on comm %p (requested: %zu); "
                "falling back to an unregistered buffer. Preallocate this window size eagerly before capture.",
                graphOwner, static_cast<void*>(comm), size);
        }
        else
        {
            TLLM_LOG_DEBUG(
                "[NCCLUtil] Repeated capture request without an eligible NCCL window buffer for owner %llu on comm "
                "%p (requested: %zu); using an unregistered buffer.",
                graphOwner, static_cast<void*>(comm), size);
        }
        return NCCLWindowBuffer();
    }

    // If a previous allocateAndRegisterBuffer call collectively failed for this comm at a size
    // no larger than this request, do not retry the known-failing new allocation path. Smaller
    // requests and already-pooled buffers can still use NCCL windows.
    auto const failureIt = mMinSymmetricFailureSize.find(comm);
    if (failureIt != mMinSymmetricFailureSize.end() && size >= failureIt->second)
    {
        TLLM_LOG_DEBUG("[NCCLUtil] Skipping NCCL window allocation for comm %p, size=%zu; known failure threshold=%zu",
            static_cast<void*>(comm), size, failureIt->second);
        return NCCLWindowBuffer();
    }

    // No available buffer found, allocate a new one
    TLLM_LOG_TRACE(
        "[NCCLUtil] Allocating new NCCL window buffer for comm %p, size=%zu", static_cast<void*>(comm), size);
    int handle = static_cast<int>(commBuffers.size());
    NCCLWindowBuffer buffer = allocateAndRegisterBuffer(comm, size, handle);
    // Only cache valid buffers. allocateAndRegisterBuffer returns an empty buffer when any rank
    // failed ncclMemAlloc (collective fallback to plain allreduce); caching it would leak a
    // permanently "in use" empty entry per request because releaseBuffer is a no-op for nullptr.
    if (buffer.isValid())
    {
        buffer.leaseId = mNextLeaseId++;
        commBuffers.push_back({buffer, true, at::cuda::current_device(), std::nullopt});
    }
    else
    {
        // The collective allreduce inside allocateAndRegisterBuffer agreed that this request
        // cannot use symmetric memory on at least one rank. Remember the smallest failing
        // request size so repeated too-large autotuner probes do not keep stressing this path.
        recordSymmetricFailureLocked(comm, size);
    }

    return buffer;
}

void NCCLWindowAllocator::recordSymmetricFailureLocked(ncclComm_t comm, size_t size)
{
    auto failureIt = mMinSymmetricFailureSize.find(comm);
    if (failureIt == mMinSymmetricFailureSize.end())
    {
        mMinSymmetricFailureSize.emplace(comm, size);
    }
    else if (size < failureIt->second)
    {
        failureIt->second = size;
    }
}

bool NCCLWindowAllocator::clearCudaErrorIfCaptureStateUnknown(
    cudaError_t captureError, CudaGetLastErrorFunc getLastError) noexcept
{
    if (captureError != cudaErrorStreamCaptureImplicit)
    {
        return false;
    }
    static_cast<void>(getLastError());
    return true;
}

cudaError_t NCCLWindowAllocator::clearCudaErrorIfSymmetricAllocationFailed(
    int localAllocOk, CudaGetLastErrorFunc getLastError) noexcept
{
    if (localAllocOk == 0)
    {
        return getLastError();
    }
    return cudaSuccess;
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
        return;
    }

    for (auto& entry : commIt->second)
    {
        if (entry.buffer.ptr == ptr)
        {
            if (!entry.inUse)
            {
                return;
            }
            entry.inUse = false;
            entry.reusableOnAnyStream = true;
            entry.reusableStream.reset();
            if (entry.releaseOwnerWhenUnused)
            {
                entry.graphPoolOwner.reset();
                entry.releaseOwnerWhenUnused = false;
            }
            TLLM_LOG_TRACE("[NCCLUtil] Released NCCL window buffer for comm %p: ptr=%p", static_cast<void*>(comm), ptr);
            return;
        }
    }
}

bool NCCLWindowAllocator::releaseBuffer(ncclComm_t comm, void* ptr, cudaStream_t stream)
{
    if (!comm || !ptr)
    {
        return false;
    }

    std::lock_guard<std::mutex> lock(mMutex);
    auto commIt = mBufferPool.find(comm);
    if (commIt == mBufferPool.end())
    {
        return false;
    }

    for (auto& entry : commIt->second)
    {
        if (entry.buffer.ptr != ptr || !entry.inUse)
        {
            continue;
        }
        entry.inUse = false;
        entry.reusableOnAnyStream = false;
        entry.reusableStream = stream;
        if (entry.releaseOwnerWhenUnused)
        {
            entry.graphPoolOwner.reset();
            entry.releaseOwnerWhenUnused = false;
        }
        TLLM_LOG_TRACE("[NCCLUtil] Explicitly released NCCL window buffer for comm %p: ptr=%p, lease=%llu",
            static_cast<void*>(comm), ptr, static_cast<unsigned long long>(entry.buffer.leaseId));
        return true;
    }
    return false;
}

void NCCLWindowAllocator::synchronizeBufferReleases()
{
    int device = -1;
    TLLM_CUDA_CHECK(cudaGetDevice(&device));

    // Capture setup is serialized, and NCCL-window releases do not run from CUDA host callbacks.
    // Hold the allocator lock so eligibility stays fixed through synchronization and promotion.
    std::lock_guard<std::mutex> lock(mMutex);
    TLLM_CUDA_CHECK(cudaDeviceSynchronize());
    for (auto& commBuffers : mBufferPool)
    {
        for (auto& entry : commBuffers.second)
        {
            if (entry.device == device && !entry.inUse && entry.reusableStream.has_value())
            {
                entry.reusableOnAnyStream = true;
                entry.reusableStream.reset();
            }
        }
    }
}

void NCCLWindowAllocator::registerTensorLease(
    ncclComm_t comm, c10::StorageImpl const* storage, void* ptr, uint64_t leaseId, int device, cudaStream_t stream)
{
    if (!comm || !storage || !ptr || leaseId == 0)
    {
        return;
    }

    std::lock_guard<std::mutex> lock(mMutex);
    auto const stackIt = mTensorLeaseScopeStacks.find(getTensorLeaseScopeKey(device, stream));
    TensorLeaseScope* scope = nullptr;
    if (stackIt != mTensorLeaseScopeStacks.end() && !stackIt->second.empty())
    {
        scope = &stackIt->second.back();
    }
    auto const inserted = mTensorLeases.emplace(storage, TensorLease{comm, ptr, leaseId, scope}).second;
    TLLM_CHECK_WITH_INFO(inserted, "NCCL window storage already has an active lease");
    if (scope)
    {
        scope->storages.insert(storage);
    }
}

void NCCLWindowAllocator::beginTensorLeaseScope(
    std::vector<c10::StorageImpl const*> const& inputStorages, int device, cudaStream_t stream)
{
    std::lock_guard<std::mutex> lock(mMutex);
    auto& scopeStack = mTensorLeaseScopeStacks[getTensorLeaseScopeKey(device, stream)];
    scopeStack.emplace_back();
    auto* scope = &scopeStack.back();

    for (auto const* storage : inputStorages)
    {
        auto leaseIt = mTensorLeases.find(storage);
        if (leaseIt == mTensorLeases.end())
        {
            continue;
        }
        if (leaseIt->second.scope)
        {
            leaseIt->second.scope->storages.erase(storage);
        }
        leaseIt->second.scope = scope;
        scope->storages.insert(storage);
    }
}

void NCCLWindowAllocator::endTensorLeaseScope(
    std::vector<c10::StorageImpl const*> const& outputStorages, int device, cudaStream_t stream, bool failed)
{
    std::unordered_set<c10::StorageImpl const*> const escapedStorages(outputStorages.begin(), outputStorages.end());

    std::lock_guard<std::mutex> lock(mMutex);
    auto stackIt = mTensorLeaseScopeStacks.find(getTensorLeaseScopeKey(device, stream));
    TLLM_CHECK_WITH_INFO(stackIt != mTensorLeaseScopeStacks.end() && !stackIt->second.empty(),
        "NCCL window tensor lease scope stack is empty for device %d and stream %p", device,
        static_cast<void*>(stream));

    auto& scopeStack = stackIt->second;
    auto& scope = scopeStack.back();
    auto* parentScope = scopeStack.size() > 1 ? &scopeStack[scopeStack.size() - 2] : nullptr;
    auto finishLease = [&](TensorLease const& lease)
    {
        if (failed)
        {
            quarantineTensorLeaseLocked(lease);
        }
        else
        {
            releaseTensorLeaseLocked(lease, stream);
        }
    };

    for (auto const* storage : scope.storages)
    {
        auto leaseIt = mTensorLeases.find(storage);
        TLLM_CHECK_WITH_INFO(leaseIt != mTensorLeases.end(), "NCCL window tensor scope contains an unknown storage");
        if (!failed && escapedStorages.find(storage) != escapedStorages.end())
        {
            leaseIt->second.scope = parentScope;
            if (parentScope)
            {
                parentScope->storages.insert(storage);
            }
            continue;
        }

        auto const releasedLease = leaseIt->second;
        mTensorLeases.erase(leaseIt);
        finishLease(releasedLease);
    }

    for (auto const& detachedLease : scope.detachedLeases)
    {
        finishLease(detachedLease);
    }

    scopeStack.pop_back();
    if (scopeStack.empty())
    {
        mTensorLeaseScopeStacks.erase(stackIt);
    }
}

bool NCCLWindowAllocator::releaseTensorBuffer(ncclComm_t comm, c10::StorageImpl const* storage, cudaStream_t stream)
{
    if (!comm || !storage)
    {
        return false;
    }

    std::lock_guard<std::mutex> lock(mMutex);
    return releaseTensorBufferLocked(storage, stream, comm);
}

bool NCCLWindowAllocator::releaseTensorBuffer(c10::StorageImpl const* storage, cudaStream_t stream)
{
    if (!storage)
    {
        return false;
    }

    std::lock_guard<std::mutex> lock(mMutex);
    return releaseTensorBufferLocked(storage, stream, nullptr);
}

bool NCCLWindowAllocator::releaseTensorBufferLocked(
    c10::StorageImpl const* storage, cudaStream_t stream, ncclComm_t expectedComm)
{
    auto const leaseIt = mTensorLeases.find(storage);
    if (leaseIt == mTensorLeases.end() || (expectedComm && leaseIt->second.comm != expectedComm))
    {
        return false;
    }

    auto const lease = leaseIt->second;
    if (lease.scope)
    {
        lease.scope->storages.erase(storage);
    }
    mTensorLeases.erase(leaseIt);
    return releaseTensorLeaseLocked(lease, stream);
}

bool NCCLWindowAllocator::releaseTensorLeaseLocked(TensorLease const& tensorLease, cudaStream_t stream)
{
    auto commIt = mBufferPool.find(tensorLease.comm);
    if (commIt == mBufferPool.end())
    {
        return false;
    }
    for (auto& entry : commIt->second)
    {
        if (entry.buffer.ptr != tensorLease.ptr || entry.buffer.leaseId != tensorLease.leaseId || !entry.inUse)
        {
            continue;
        }
        entry.inUse = false;
        entry.reusableOnAnyStream = false;
        entry.reusableStream = stream;
        if (entry.releaseOwnerWhenUnused)
        {
            entry.graphPoolOwner.reset();
            entry.releaseOwnerWhenUnused = false;
        }
        TLLM_LOG_TRACE("[NCCLUtil] Explicitly released NCCL window tensor for comm %p: ptr=%p, lease=%llu",
            static_cast<void*>(tensorLease.comm), tensorLease.ptr,
            static_cast<unsigned long long>(tensorLease.leaseId));
        return true;
    }
    return false;
}

void NCCLWindowAllocator::quarantineTensorLeaseLocked(TensorLease const& tensorLease)
{
    auto commIt = mBufferPool.find(tensorLease.comm);
    if (commIt == mBufferPool.end())
    {
        return;
    }
    for (auto& entry : commIt->second)
    {
        if (entry.buffer.ptr != tensorLease.ptr || entry.buffer.leaseId != tensorLease.leaseId || !entry.inUse)
        {
            continue;
        }
        entry.inUse = false;
        entry.reusableOnAnyStream = false;
        entry.reusableStream.reset();
        if (entry.releaseOwnerWhenUnused)
        {
            entry.graphPoolOwner.reset();
            entry.releaseOwnerWhenUnused = false;
        }
        TLLM_LOG_DEBUG("[NCCLUtil] Quarantining NCCL window lease %llu on comm %p after a failed tensor lease scope.",
            static_cast<unsigned long long>(tensorLease.leaseId), static_cast<void*>(tensorLease.comm));
        return;
    }
}

void NCCLWindowAllocator::releaseBufferFromDestructor(ncclComm_t comm, void* ptr, uint64_t leaseId)
{
    if (!comm || !ptr || leaseId == 0)
    {
        return;
    }

    std::lock_guard<std::mutex> lock(mMutex);
    for (auto it = mTensorLeases.begin(); it != mTensorLeases.end(); ++it)
    {
        auto const lease = it->second;
        if (lease.comm != comm || lease.ptr != ptr || lease.leaseId != leaseId)
        {
            continue;
        }

        if (lease.scope)
        {
            // Storage lifetime inside a scope is intentionally irrelevant. The scope retains the
            // lease generation and releases it at the same logical boundary on every rank.
            lease.scope->storages.erase(it->first);
            lease.scope->detachedLeases.push_back(lease);
            mTensorLeases.erase(it);
            return;
        }

        mTensorLeases.erase(it);
        break;
    }

    auto commIt = mBufferPool.find(comm);
    if (commIt == mBufferPool.end())
    {
        return;
    }

    for (auto& entry : commIt->second)
    {
        if (entry.buffer.ptr != ptr || entry.buffer.leaseId != leaseId || !entry.inUse)
        {
            continue;
        }
        entry.inUse = false;
        entry.reusableOnAnyStream = false;
        entry.reusableStream.reset();
        if (entry.releaseOwnerWhenUnused)
        {
            entry.graphPoolOwner.reset();
            entry.releaseOwnerWhenUnused = false;
        }
        if (markFallbackWarningLoggedLocked(comm, FallbackWarning::kDestructorFallback))
        {
            TLLM_LOG_WARNING(
                "[NCCLUtil] NCCL window lease %llu on comm %p was released by its tensor destructor instead of "
                "an explicit final consumer. Quarantining this buffer until communicator teardown.",
                static_cast<unsigned long long>(leaseId), static_cast<void*>(comm));
        }
        else
        {
            TLLM_LOG_DEBUG(
                "[NCCLUtil] Another NCCL window lease was released by its tensor destructor on comm %p; "
                "quarantining the buffer.",
                static_cast<void*>(comm));
        }
        return;
    }
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
    // Step 1: Pre-allocate the rank-sync flag before ncclMemAlloc. ncclMemAlloc can fail
    // asymmetrically with ncclUnhandledCudaError on configurations where the symmetric/VMM path
    // is unavailable; that failure may leave a sticky CUDA last-error on the device. If we
    // deferred this cudaMalloc until after the failure, the sticky error would propagate into
    // cudaMalloc, TLLM_CUDA_CHECK would throw, and the failing rank would never reach the
    // collective ncclAllReduce(min) below, hanging every other rank that did succeed.
    int* rankSyncFlag = nullptr;
    TLLM_CUDA_CHECK(cudaMalloc(&rankSyncFlag, sizeof(int)));
    CudaMallocGuard flagGuard{rankSyncFlag}; // frees rankSyncFlag on any early return or exception
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    TLLM_CUDA_CHECK(cudaMemsetAsync(rankSyncFlag, 0, sizeof(int), stream));

    // Step 2: Allocate symmetric memory. This per-rank, non-collective call can fail
    // asymmetrically. When it fails, NCCL may leave a sticky CUDA error behind; clear it before
    // the stream-ordered flag copy and collective fallback so the failing rank still reaches
    // ncclAllReduce with the other ranks.
    void* ncclPtr = nullptr;
    TLLM_NCCL_CHECK_WARN(ncclMemAlloc(&ncclPtr, size));
    int const localAllocOk = (ncclPtr != nullptr) ? 1 : 0;
    NcclMemGuard ncclGuard{ncclPtr}; // frees ncclPtr on any early return or exception
    clearCudaErrorIfSymmetricAllocationFailed(localAllocOk);

    // Step 3: ncclCommWindowRegister is collective. If any rank skips it, all other ranks hang.
    // Populate flag, reduce with min across ranks (0 if any rank failed), then read back.
    // The flag is initialized to 0, so H2D failure is non-fatal and conservatively falls back
    // to regular NCCL while still reaching the collective. allreduce and D2H failures throw.
    if (localAllocOk != 0)
    {
        TLLM_CUDA_CHECK_WARN(
            cudaMemcpyAsync(rankSyncFlag, &localAllocOk, sizeof(localAllocOk), cudaMemcpyHostToDevice, stream));
    }
    TLLM_NCCL_CHECK(ncclAllReduce(rankSyncFlag, rankSyncFlag, 1, ncclInt32, ncclMin, comm, stream));
    TLLM_CUDA_CHECK_WARN(cudaStreamSynchronize(stream));

    int allAllocOk = 0;
    TLLM_CUDA_CHECK(cudaMemcpy(&allAllocOk, rankSyncFlag, sizeof(int), cudaMemcpyDeviceToHost));
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

    // Step 4: Register with NCCL as a window. This is collective, so all ranks must reach it.
    // Failure here is non-fatal: warn and fall back to regular allreduce.
    // ncclGuard frees ncclPtr on return.
    ncclWindow_t window = nullptr;
    ncclResult_t const regResult = ncclCommWindowRegister(comm, ncclPtr, size, &window, NCCL_WIN_COLL_SYMMETRIC);
    TLLM_NCCL_CHECK_WARN(regResult);
    if (regResult != ncclSuccess)
    {
        return NCCLWindowBuffer{};
    }

    // Step 5: Success. Transfer ownership to the returned buffer.
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

    for (auto it = mTensorLeases.begin(); it != mTensorLeases.end();)
    {
        if (it->second.comm != comm)
        {
            ++it;
            continue;
        }
        if (it->second.scope)
        {
            it->second.scope->storages.erase(it->first);
        }
        it = mTensorLeases.erase(it);
    }

    for (auto& stackEntry : mTensorLeaseScopeStacks)
    {
        for (auto& scope : stackEntry.second)
        {
            auto& detachedLeases = scope.detachedLeases;
            detachedLeases.erase(std::remove_if(detachedLeases.begin(), detachedLeases.end(),
                                     [comm](TensorLease const& lease) { return lease.comm == comm; }),
                detachedLeases.end());
        }
    }

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
        mMinSymmetricFailureSize.erase(comm);
        mLoggedFallbackWarnings.erase(comm);
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
    mMinSymmetricFailureSize.erase(comm);
    mLoggedFallbackWarnings.erase(comm);
}

#endif // NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)

} // namespace tensorrt_llm::common::nccl_util

#endif // ENABLE_MULTI_DEVICE
