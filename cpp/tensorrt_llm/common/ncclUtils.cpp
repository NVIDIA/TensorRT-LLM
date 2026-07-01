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
#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/runtime/utils/ncclHostApi.h"
#include <chrono>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <thread>

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
            auto const hostApiLock = tensorrt_llm::runtime::acquireNcclHostApiLock();
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
constexpr auto kRawNcclPollInterval = std::chrono::milliseconds{1};

struct RawNcclCallCompletion
{
    ncclResult_t result;
    bool completed;
};

std::chrono::milliseconds getRawNcclCallTimeout()
{
    constexpr int64_t defaultTimeoutMs = 5000;
    auto const* value = std::getenv("TRTLLM_NCCL_NONBLOCKING_TIMEOUT_MS");
    if (value != nullptr)
    {
        try
        {
            auto const parsed = std::stoll(value);
            if (parsed > 0)
            {
                return std::chrono::milliseconds{parsed};
            }
        }
        catch (...)
        {
        }
    }
    return std::chrono::milliseconds{defaultTimeoutMs};
}

// The caller must hold the process-wide NCCL host-API gate from the initial
// call through this poll. NCCL permits only ncclCommGetAsyncError after a
// nonblocking API returns ncclInProgress.
RawNcclCallCompletion completeRawNcclCall(
    ncclComm_t comm, ncclResult_t initialResult, std::chrono::steady_clock::time_point deadline) noexcept
{
    if (initialResult != ncclInProgress)
    {
        return {initialResult, true};
    }

    while (std::chrono::steady_clock::now() < deadline)
    {
        ncclResult_t asyncResult = ncclInProgress;
        auto const queryResult = ncclCommGetAsyncError(comm, &asyncResult);
        if (queryResult == ncclSuccess && asyncResult != ncclInProgress)
        {
            return {asyncResult, true};
        }
        if (queryResult != ncclSuccess && queryResult != ncclInProgress)
        {
            return {queryResult, true};
        }
        std::this_thread::sleep_for(kRawNcclPollInterval);
    }
    return {ncclInProgress, false};
}

bool isGb10Platform(int realSmVersion, bool isIntegrated)
{
    return realSmVersion == kGb10RealSmVersion && isIntegrated;
}
#endif

bool queryNcclWindowSupported()
{
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)
    auto const hostApiLock = tensorrt_llm::runtime::acquireNcclHostApiLock();
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
    // Communicator registries and watchdogs are process-lifetime. Keep their
    // resource owner alive until process exit instead of relying on undefined
    // cross-translation-unit singleton destruction order.
    static auto* instance = new NcclCommResourceManager;
    return *instance;
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

void NcclCommResourceManager::beginAbortCleanup(ncclComm_t comm) noexcept
{
    if (comm == nullptr)
    {
        return;
    }
    std::lock_guard<std::mutex> lock(mMutex);
    mAbortCleanups.insert(comm);
}

void NcclCommResourceManager::waitForAbortCleanup(ncclComm_t comm) noexcept
{
    if (comm == nullptr)
    {
        return;
    }
    std::unique_lock<std::mutex> lock(mMutex);
    mAbortCleanupComplete.wait(lock, [this, comm]() { return mAbortCleanups.find(comm) == mAbortCleanups.end(); });
}

bool NcclCommResourceManager::cleanupResources(ncclComm_t comm, bool communicatorAborted) noexcept
{
    if (!comm)
    {
        return true;
    }

    // Check if we're in the process of being destroyed
    // If so, skip cleanup - the destructor will handle it proactively
    if (mIsDestroying.load(std::memory_order_acquire))
    {
        if (communicatorAborted)
        {
            std::lock_guard<std::mutex> lock(mMutex);
            mAbortCleanups.erase(comm);
            mAbortCleanupComplete.notify_all();
        }
        return false;
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
                if (communicatorAborted)
                {
                    mAbortCleanups.erase(comm);
                    mAbortCleanupComplete.notify_all();
                }
                return false;
            }

            if (communicatorAborted)
            {
                mAbortCleanups.insert(comm);
            }
            if (mCleanupsInProgress.find(comm) != mCleanupsInProgress.end())
            {
                // Another thread already owns the callbacks. Its completion
                // path rechecks the retirement marker, including one that was
                // installed after that owner started, before waking handle
                // reusers.
                return true;
            }

            auto it = mCommResources.find(comm);
            if (it == mCommResources.end())
            {
                // Nothing registered for this comm, nothing to clean up
                if (mAbortCleanups.erase(comm) != 0)
                {
                    mAbortCleanupComplete.notify_all();
                }
                return true;
            }

            // Move resources out (preserves order) and remove from map
            resourcesToClean = std::move(it->second);
            mCommResources.erase(it);
            mCleanupsInProgress.insert(comm);
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
            return false;
        }
    }

    // Clean up outside the lock to avoid deadlocks if cleanup functions try to access the manager
    // Order is preserved: resources are cleaned up in registration order
    bool cleanupSucceeded = true;
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
            cleanupSucceeded = cleanup() && cleanupSucceeded;
        }
        catch (std::exception const& e)
        {
            cleanupSucceeded = false;
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
            cleanupSucceeded = false;
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

    {
        std::lock_guard<std::mutex> lock(mMutex);
        mCleanupsInProgress.erase(comm);
        if (mAbortCleanups.erase(comm) != 0)
        {
            mAbortCleanupComplete.notify_all();
        }
    }
    return cleanupSucceeded;
}

bool NcclCommResourceManager::isAbortCleanup(ncclComm_t comm) const noexcept
{
    std::lock_guard<std::mutex> lock(mMutex);
    return mAbortCleanups.find(comm) != mAbortCleanups.end();
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
    static auto* instance = new NCCLWindowAllocator;
    return *instance;
}

NCCLWindowBuffer NCCLWindowAllocator::requestBuffer(ncclComm_t comm, size_t size)
{
    return requestBufferImpl(comm, size, nullptr, nullptr);
}

NCCLWindowBuffer NCCLWindowAllocator::requestBuffer(
    std::shared_ptr<ncclComm_t> const& comm, size_t size, ncclComm_t* associatedComm)
{
    TLLM_CHECK_WITH_INFO(comm != nullptr, "NCCL communicator cannot be null");
    if (associatedComm != nullptr)
    {
        *associatedComm = nullptr;
    }
    return requestBufferImpl(nullptr, size, &comm, associatedComm);
}

NCCLWindowBuffer NCCLWindowAllocator::requestBufferImpl(
    ncclComm_t comm, size_t size, std::shared_ptr<ncclComm_t> const* managedComm, ncclComm_t* associatedComm)
{
    if (!isNcclWindowSupported())
    {
        return NCCLWindowBuffer();
    }

    TLLM_CHECK_WITH_INFO(size > 0, "Buffer size must be greater than 0");

    bool const managedFaultTolerance = managedComm != nullptr && isNcclFaultToleranceEnabled();
    if (managedComm == nullptr)
    {
        TLLM_CHECK_WITH_INFO(comm != nullptr, "NCCL communicator cannot be null");
        NcclCommResourceManager::getInstance().waitForAbortCleanup(comm);
    }
    // Pool hits need only mMutex. On a miss, release all communicator/pool
    // locks, serialize allocation attempts, then recheck: another thread may
    // have populated the pool while this thread waited. This preserves
    // collective ordering without adding a second lock to the steady-state
    // pooled-buffer path or inverting the host-gate -> pool-lock cleanup order.
    std::unique_lock<std::mutex> allocationLock(mAllocationMutex, std::defer_lock);
    int handle = 0;
    while (true)
    {
        std::unique_ptr<NcclCommLease> validationLease;
        if (managedComm != nullptr)
        {
            if (managedFaultTolerance)
            {
                // Keep lock ordering consistent with abort cleanup:
                // communicator state / NCCL host gate precede the pool mutex.
                validationLease = std::make_unique<NcclCommLease>(acquireComm(*managedComm));
                comm = validationLease->get();
            }
            else
            {
                TLLM_CHECK_WITH_INFO(*managedComm != nullptr, "NCCL communicator cannot be null");
                comm = **managedComm;
            }
        }
        TLLM_CHECK_WITH_INFO(comm != nullptr, "NCCL communicator cannot be null");
        if (associatedComm != nullptr)
        {
            *associatedComm = comm;
        }
        std::unique_lock<std::mutex> poolLock(mMutex);

        // Register cleanup callback for this communicator if not already registered.
        registerBufferCleanup(comm);
        // Keep validationLease until poolLock releases. Lease destruction can
        // drain abort cleanup, which re-enters this non-recursive pool mutex.

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

        auto const failureIt = mMinSymmetricFailureSize.find(comm);
        if (failureIt != mMinSymmetricFailureSize.end() && size >= failureIt->second)
        {
            TLLM_LOG_DEBUG(
                "[NCCLUtil] Skipping NCCL window allocation for comm %p, size=%zu; known failure threshold=%zu",
                static_cast<void*>(comm), size, failureIt->second);
            return NCCLWindowBuffer();
        }

        auto stream = at::cuda::getCurrentCUDAStream();
        cudaStreamCaptureStatus captureStatus = cudaStreamCaptureStatusNone;
        auto const captureResult = cudaStreamIsCapturing(stream, &captureStatus);
        if (captureResult != cudaSuccess)
        {
            TLLM_LOG_DEBUG("[NCCLUtil] cudaStreamIsCapturing failed: %s", cudaGetErrorString(captureResult));
        }
        if (captureResult == cudaSuccess && captureStatus != cudaStreamCaptureStatusNone)
        {
            TLLM_LOG_DEBUG("[NCCLUtil] Skipping NCCL window allocation during capture for comm %p (requested: %zu)",
                static_cast<void*>(comm), size);
            return NCCLWindowBuffer();
        }

        if (!allocationLock.owns_lock())
        {
            poolLock.unlock();
            validationLease.reset();
            allocationLock.lock();
            continue;
        }

        TLLM_LOG_TRACE(
            "[NCCLUtil] Allocating new NCCL window buffer for comm %p, size=%zu", static_cast<void*>(comm), size);
        handle = static_cast<int>(commBuffers.size());
        poolLock.unlock();
        validationLease.reset();
        break;
    }

    // CUDA allocation must not hold the communicator mutex or process-wide
    // NCCL gate: a watchdog needs both in order to abort stalled work.
    std::unique_ptr<NcclCommLease> completionLease;
    NCCLWindowBuffer buffer = allocateAndRegisterBuffer(comm, size, handle, managedComm, &completionLease);
    // Only cache valid buffers. allocateAndRegisterBuffer returns an empty buffer when any rank
    // failed ncclMemAlloc (collective fallback to plain allreduce); caching it would leak a
    // permanently "in use" empty entry per request because releaseBuffer is a no-op for nullptr.
    if (buffer.isValid())
    {
        // The managed registration lease remains live until the buffer is in
        // the pool. That closes the gap where the watchdog could abort and run
        // cleanup after registration but before the resource became visible.
        {
            std::lock_guard<std::mutex> poolLock(mMutex);
            mBufferPool[comm].push_back({buffer, true});
        }
        completionLease.reset();
    }
    else
    {
        std::unique_ptr<NcclCommLease> healthyLease;
        if (managedFaultTolerance)
        {
            healthyLease = std::make_unique<NcclCommLease>(acquireComm(*managedComm));
            TLLM_CHECK_WITH_INFO(
                healthyLease->get() == comm, "NCCL communicator changed while recording a window-allocation fallback");
        }
        // The collective allreduce inside allocateAndRegisterBuffer agreed that this request
        // cannot use symmetric memory on at least one rank. Remember the smallest failing
        // request size so repeated too-large autotuner probes do not keep stressing this path.
        std::lock_guard<std::mutex> poolLock(mMutex);
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

NCCLWindowBuffer NCCLWindowAllocator::allocateAndRegisterBuffer(ncclComm_t comm, size_t size, int handle,
    std::shared_ptr<ncclComm_t> const* managedComm, std::unique_ptr<NcclCommLease>* completionLease)
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
    {
        auto const hostApiLock = tensorrt_llm::runtime::acquireNcclHostApiLock();
        TLLM_NCCL_CHECK_WARN(ncclMemAlloc(&ncclPtr, size));
    }
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
    if (managedComm != nullptr)
    {
        try
        {
            std::uint64_t watchdogToken = 0;
            {
                auto commLease = acquireComm(*managedComm);
                auto const currentComm = commLease.get();
                TLLM_CHECK_WITH_INFO(currentComm == comm,
                    "NCCL communicator changed while allocating a window buffer; retry with the active communicator");
                watchdogToken = commLease.begin(stream, "NCCL window allocation agreement");
                auto const allocAgreementResult
                    = ncclAllReduce(rankSyncFlag, rankSyncFlag, 1, ncclInt32, ncclMin, currentComm, stream);
                commLease.check(allocAgreementResult, "ncclAllReduce(window allocation agreement)");
                commLease.track(watchdogToken, stream);
            }
            if (isNcclFaultToleranceEnabled())
            {
                waitCommOperation(*managedComm, watchdogToken, "NCCL window allocation agreement");
            }
            else
            {
                TLLM_CUDA_CHECK_WARN(cudaStreamSynchronize(stream));
            }
        }
        catch (...)
        {
            // A failed NCCL kernel may still hold the temporary flag even
            // after communicator abort. Quarantine four bytes instead of
            // letting cudaFree introduce a device-wide wait on recovery.
            static_cast<void>(flagGuard.release());
            throw;
        }
    }
    else
    {
        RawNcclCallCompletion completion{ncclSuccess, true};
        {
            auto const hostApiLock = tensorrt_llm::runtime::acquireNcclHostApiLock();
            auto const allocAgreementResult
                = ncclAllReduce(rankSyncFlag, rankSyncFlag, 1, ncclInt32, ncclMin, comm, stream);
            completion = completeRawNcclCall(
                comm, allocAgreementResult, std::chrono::steady_clock::now() + getRawNcclCallTimeout());
        }
        if (!completion.completed)
        {
            // The background enqueue may still acquire the flag after the
            // local deadline. Keep its storage alive rather than racing it.
            static_cast<void>(flagGuard.release());
            TLLM_THROW("NCCL error: window allocation agreement timed out while the communicator was in progress");
        }
        if (completion.result != ncclSuccess)
        {
            // NCCL documents device work as indeterminate after an async
            // communicator error. Quarantine the flag before surfacing it.
            static_cast<void>(flagGuard.release());
            TLLM_NCCL_CHECK(completion.result);
        }
        TLLM_CUDA_CHECK_WARN(cudaStreamSynchronize(stream));
    }

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
    ncclResult_t regResult = ncclSuccess;
    if (managedComm != nullptr)
    {
        auto commLease = acquireComm(*managedComm);
        auto const currentComm = commLease.get();
        TLLM_CHECK_WITH_INFO(currentComm == comm,
            "NCCL communicator changed while registering a window buffer; retry with the active communicator");
        regResult = ncclCommWindowRegister(currentComm, ncclPtr, size, &window, NCCL_WIN_COLL_SYMMETRIC);
        try
        {
            regResult = commLease.checkOptional(regResult, "ncclCommWindowRegister");
        }
        catch (...)
        {
            // Registration may still reference the allocation after an
            // asynchronous communicator failure. Quarantine it instead
            // of calling ncclMemFree on the recovery path.
            static_cast<void>(ncclGuard.release());
            throw;
        }
        if (regResult == ncclSuccess && completionLease != nullptr && isNcclFaultToleranceEnabled())
        {
            *completionLease = std::make_unique<NcclCommLease>(std::move(commLease));
        }
    }
    else
    {
        ncclResult_t initialResult = ncclSuccess;
        RawNcclCallCompletion completion{ncclSuccess, true};
        auto const hostApiLock = tensorrt_llm::runtime::acquireNcclHostApiLock();
        initialResult = ncclCommWindowRegister(comm, ncclPtr, size, &window, NCCL_WIN_COLL_SYMMETRIC);
        completion
            = completeRawNcclCall(comm, initialResult, std::chrono::steady_clock::now() + getRawNcclCallTimeout());
        if (!completion.completed)
        {
            // Registration can still complete in the background and retain
            // ncclPtr. The unmanaged caller owns communicator recovery, so
            // quarantine the allocation locally and surface the timeout.
            static_cast<void>(ncclGuard.release());
            TLLM_THROW("NCCL error: window registration timed out while the communicator was in progress");
        }
        regResult = completion.result;
        if (initialResult == ncclInProgress && regResult != ncclSuccess)
        {
            // Once an asynchronous registration fails, NCCL does not promise
            // that the allocation is no longer referenced.
            static_cast<void>(ncclGuard.release());
            TLLM_THROW("NCCL error: window registration failed asynchronously: %s", ncclGetErrorString(regResult));
        }
    }
    if (regResult != ncclSuccess)
    {
        TLLM_LOG_WARNING("NCCL window registration failed for comm %p; falling back to regular allreduce: %s",
            static_cast<void*>(comm), ncclGetErrorString(regResult));
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
        comm, [this, comm]() { return this->cleanupBuffersForComm(comm); }, "NCCLWindowAllocator");
}

bool NCCLWindowAllocator::cleanupBuffersForComm(ncclComm_t comm) noexcept
{
    if (!comm)
    {
        return true;
    }

    bool const communicatorAborted = NcclCommResourceManager::getInstance().isAbortCleanup(comm);
    bool cudaSynchronized = true;

    // Never hold the process-wide NCCL host gate across a CUDA synchronization:
    // work from another failed communicator may be what the synchronization is
    // waiting for, and that communicator's watchdog needs the gate to abort it.
    tensorrt_llm::runtime::NcclHostApiLock hostApiLock;
    if (!communicatorAborted)
    {
        cudaError_t cudaErr = cudaDeviceSynchronize();
        if (cudaErr != cudaSuccess)
        {
            cudaSynchronized = false;
            TLLM_LOG_WARNING("[NCCLUtil] cudaDeviceSynchronize failed with error: %d before cleanup for comm %p",
                cudaErr, static_cast<void*>(comm));
        }
        if (cudaSynchronized)
        {
            // Healthy cleanup uses NCCL deregistration/free below. Acquire the gate
            // before mMutex to preserve the communicator -> allocator lock order.
            hostApiLock = tensorrt_llm::runtime::acquireNcclHostApiLock();
        }
    }

    std::lock_guard<std::mutex> lock(mMutex);

    // Check if we've already cleaned up this communicator
    if (mRegisteredComms.find(comm) == mRegisteredComms.end())
    {
        // Already cleaned up or never registered
        return true;
    }

    auto commIt = mBufferPool.find(comm);
    if (commIt == mBufferPool.end())
    {
        // No buffers to clean up, but mark as cleaned
        mRegisteredComms.erase(comm);
        mMinSymmetricFailureSize.erase(comm);
        return true;
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

    if (communicatorAborted || !cudaSynchronized)
    {
        // Tensor storages or non-NCCL CUDA work may still reference these
        // allocations. There is no safe stream fence once the communicator
        // has failed, so quarantine them for the remaining process lifetime
        // instead of risking a UAF or another blocking CUDA call. Rank failure
        // is rare and process restart ultimately reclaims the memory.
        TLLM_LOG_WARNING("[NCCLUtil] Quarantining %zu NCCL window buffers (%zu bytes) after %s", commIt->second.size(),
            totalBytes, communicatorAborted ? "communicator abort" : "failed CUDA synchronization");
        mBufferPool.erase(commIt);
        mRegisteredComms.erase(comm);
        mMinSymmetricFailureSize.erase(comm);
        return communicatorAborted;
    }

    bool quarantineRemainingBuffers = false;
    for (auto& entry : commIt->second)
    {
        if (entry.buffer.isValid())
        {
            if (quarantineRemainingBuffers)
            {
                continue;
            }
            // Deregister the window - the communicator is still valid at this point
            // (cleanup happens before ncclCommDestroy), but we need to be careful
            // if buffers are still in use by active operations
            if (entry.buffer.window && comm && !communicatorAborted)
            {
                // Note: Even if buffer is marked inUse, we must deregister since
                // the communicator is being destroyed. The communicator is valid,
                // but we should handle potential errors gracefully.
                auto const initialResult = ncclCommWindowDeregister(comm, entry.buffer.window);
                auto const completion = completeRawNcclCall(
                    comm, initialResult, std::chrono::steady_clock::now() + getRawNcclCallTimeout());
                if (!completion.completed || completion.result != ncclSuccess)
                {
                    TLLM_LOG_WARNING(
                        "[NCCLUtil] ncclCommWindowDeregister %s for comm %p, window %p "
                        "(result: %s, buffer inUse: %d); quarantining this and remaining buffers",
                        completion.completed ? "failed" : "timed out", static_cast<void*>(comm),
                        static_cast<void*>(entry.buffer.window), ncclGetErrorString(completion.result), entry.inUse);
                    quarantineRemainingBuffers = true;
                    continue;
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
    return !quarantineRemainingBuffers;
}

#endif // NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)

} // namespace tensorrt_llm::common::nccl_util

#endif // ENABLE_MULTI_DEVICE
