/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

#include "tensorrt_llm/executor/transferAgent.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/cudaEvent.h"
#include "tensorrt_llm/runtime/cudaStream.h"
#include <atomic>
#include <condition_variable>
#include <cuda.h>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <shared_mutex>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tensorrt_llm::executor::kv_cache
{

// ============================================================================
// VMM Handle Type
// ============================================================================

/// @brief Type of shareable VMM handle used for P2P transfer.
enum class VmmHandleType : uint8_t
{
    kNone = 0,    ///< No shareable handle
    kFabric = 1,  ///< CU_MEM_HANDLE_TYPE_FABRIC (cross-node via NVSwitch)
    kPosixFd = 2, ///< CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR (same-machine via UDS)
    kCudaIpc = 3, ///< cudaMalloc memory via cudaIpcGetMemHandle (same-machine, no UDS needed)
};

inline char const* handleTypeToString(VmmHandleType type)
{
    switch (type)
    {
    case VmmHandleType::kFabric: return "Fabric";
    case VmmHandleType::kPosixFd: return "PosixFd";
    case VmmHandleType::kCudaIpc: return "CudaIpc";
    default: return "None";
    }
}

// ============================================================================
// Serializable P2P memory info (exchanged via AgentDesc)
// ============================================================================

/// @brief Single physical memory chunk metadata (optional fabric handle bytes).
struct P2pMemChunk
{
    uint64_t virtAddrOffset;  ///< Offset relative to pool base
    uint64_t size;            ///< Chunk size in bytes
    uint8_t fabricHandle[64]; ///< CUmemFabricHandle or cudaIpcMemHandle; zeroed for POSIX FD

    void serialize(std::ostream& os) const;
    [[nodiscard]] static P2pMemChunk deserialize(std::istream& is);

    static constexpr size_t serializedSize()
    {
        return sizeof(uint64_t) * 2 + 64;
    }
};

/// @brief One memory pool possibly containing multiple physical chunks.
struct P2pMemPool
{
    int32_t deviceId;
    uint64_t poolBaseAddr;
    uint64_t poolTotalSize;
    uint64_t registeredAddr;
    uint64_t registeredSize;
    uint64_t mappedOffset;
    uint64_t mappedSize;
    std::vector<P2pMemChunk> chunks;

    /// Host-local POSIX file descriptors, one per chunk in chunk order. Owned by the
    /// exporting pool — closed when the pool is removed. NEVER serialized: receivers get
    /// fresh FDs out-of-band over the UDS server. Empty in non-POSIX-FD modes and on the
    /// receive side. Storing fds here (rather than in a global vector) keeps the
    /// chunk-order ↔ fd-order invariant local to the pool, so pool removals can never
    /// close another pool's FDs.
    std::vector<int> fds;

    void serialize(std::ostream& os) const;
    [[nodiscard]] static P2pMemPool deserialize(std::istream& is);
};

/// @brief Full P2P memory info published by a local agent.
struct P2pMemInfo
{
    static constexpr uint32_t kMagic = 0x50325054; ///< "P2PT"
    static constexpr uint32_t kVersion = 1;

    bool supported{false};
    VmmHandleType handleType{VmmHandleType::kNone};
    std::string udsPath; ///< POSIX FD mode only
    std::vector<P2pMemPool> pools;

    [[nodiscard]] std::string serialize() const;
    [[nodiscard]] static std::optional<P2pMemInfo> deserialize(std::string_view data);
};

/// @brief Per-pool result of importing a remote agent's P2P handles locally.
struct RemoteP2pPoolMapping
{
    uint64_t remoteBaseAddr;
    uint64_t totalSize;
    uint64_t remoteRegisteredAddr;
    uint64_t registeredSize;
    uint64_t remoteMappedOffset;
    uint64_t mappedSize;
    CUdeviceptr localVirtAddr;
    std::vector<CUmemGenericAllocationHandle> importedHandles;
};

/// @brief Complete mapping of one remote agent into the local address space.
struct RemoteP2pMapping
{
    std::string remoteName;
    VmmHandleType handleType{VmmHandleType::kNone};
    std::vector<RemoteP2pPoolMapping> pools;
};

// ============================================================================
// P2pAgentCounters — test-only observability for submit sub-paths
// ============================================================================

/// @brief Atomic counters that track which internal path P2pTransferContext takes when
/// submitting a cudaMemcpyBatchAsync request. Single-thread vs multi-thread-worker-pool
/// is a runtime decision inside submitWithMemcpyBatch; no other way to observe it from
/// outside the class exists today. Used by agent-level E2E tests to assert that env
/// flags and batch sizes actually route through the intended branch, rather than silently
/// falling through to the "easy" path. Incrementing two atomics per submit is negligible
/// relative to the CUDA work that follows.
struct P2pAgentCounters
{
    std::atomic<uint64_t> memcpyBatchSingleThread{0};
    std::atomic<uint64_t> memcpyBatchMultiThread{0};

    struct Snapshot
    {
        uint64_t memcpyBatchSingleThread;
        uint64_t memcpyBatchMultiThread;
    };

    [[nodiscard]] Snapshot snapshot() const noexcept
    {
        return {memcpyBatchSingleThread.load(std::memory_order_relaxed),
            memcpyBatchMultiThread.load(std::memory_order_relaxed)};
    }
};

// ============================================================================
// BatchCopyWorkerPool — shared worker pool for parallel cudaMemcpyBatchAsync
// ============================================================================

struct BatchCopyTask
{
    std::vector<void*> dst;
    std::vector<void const*> src;
    std::vector<size_t> sizes;
    cudaStream_t stream;
    /// Worker records this event after cudaMemcpyBatchAsync. Held as shared_ptr so the event
    /// can never be recycled into the CudaEventPool before the worker has used it, even if the
    /// caller drops its P2pTransferStatus.
    std::shared_ptr<runtime::CudaEvent> completionEvent;
    /// Per-batch pending counter. Shared with the P2pTransferStatus so whichever drops last
    /// releases it — survives a prematurely-dropped status.
    std::shared_ptr<std::atomic<int>> batchPending;
};

/// @brief Persistent thread pool. Each worker binds `cudaSetDevice` once then drains the queue.
class BatchCopyWorkerPool
{
public:
    BatchCopyWorkerPool(int numWorkers, int cudaDevice);
    ~BatchCopyWorkerPool();

    BatchCopyWorkerPool(BatchCopyWorkerPool const&) = delete;
    BatchCopyWorkerPool& operator=(BatchCopyWorkerPool const&) = delete;

    void submit(BatchCopyTask&& task);
    void waitAll();
    [[nodiscard]] bool isDone() const;

private:
    void workerLoop();

    std::vector<std::thread> mWorkers;
    std::mutex mMutex;
    std::condition_variable mCv;
    std::condition_variable mDoneCv;
    std::queue<BatchCopyTask> mQueue;
    bool mShutdown{false};
    std::atomic<int> mPending{0};
};

// ============================================================================
// P2pTransferStatus — returned by submit
// ============================================================================

class P2pTransferStatus final : public TransferStatus
{
public:
    /// @brief Single-event mode (cub path and single-thread cudaMemcpyBatch path).
    P2pTransferStatus(std::shared_ptr<runtime::CudaStream> stream, std::shared_ptr<runtime::CudaEvent> completionEvent);

    /// @brief Multi-event mode (multi-thread cudaMemcpyBatch path). Per-batch isolation.
    P2pTransferStatus(
        std::shared_ptr<std::atomic<int>> batchPending, std::vector<std::shared_ptr<runtime::CudaEvent>> workerEvents);

    ~P2pTransferStatus() override = default;

    [[nodiscard]] bool isCompleted() const override;
    [[nodiscard]] TransferState wait(int64_t timeout_ms = -1) const override;

private:
    // Single-event mode
    std::shared_ptr<runtime::CudaStream> mStream;
    std::shared_ptr<runtime::CudaEvent> mCompletionEvent;

    // Multi-event mode
    std::shared_ptr<std::atomic<int>> mBatchPending;
    std::vector<std::shared_ptr<runtime::CudaEvent>> mWorkerEvents;

    mutable std::atomic<bool> mCompleted{false};
};

// ============================================================================
// MixedTransferStatus — composite for the mixed P2P + NIXL routing path
// ============================================================================

/// @brief Waits on both a P2P-fast-path status and a NIXL-fallback status.
/// Used when a single TransferRequest is split across the two backends because
/// some segments' remote addresses had a valid P2P mapping and others did not.
/// Either child may be null (all-mapped or none-mapped cases go straight to the
/// corresponding single-path status and don't need this composite).
/// FAILURE of either child is reported as FAILURE; TIMEOUT of the first child
/// short-circuits without polling the second.
///
/// Lives in p2pTransferAgent.h (and not in nixl_utils/transferAgent.h) so that
/// unit tests can compose it without pulling in nixl.h.
class MixedTransferStatus final : public TransferStatus
{
public:
    MixedTransferStatus(std::unique_ptr<TransferStatus> p2p, std::unique_ptr<TransferStatus> nixl)
        : mP2p(std::move(p2p))
        , mNixl(std::move(nixl))
    {
    }

    [[nodiscard]] bool isCompleted() const override;
    [[nodiscard]] TransferState wait(int64_t timeout_ms = -1) const override;

private:
    std::unique_ptr<TransferStatus> mP2p;
    std::unique_ptr<TransferStatus> mNixl;
};

// ============================================================================
// CudaEventPool — reusable CUDA events to avoid create/destroy overhead
// ============================================================================

class CudaEventPool : public std::enable_shared_from_this<CudaEventPool>
{
public:
    /// @brief Acquire an event. Returned shared_ptr returns the event to the pool on destruction.
    ///
    /// Acquire-then-record contract: a recycled event arrives carrying the result of its
    /// PREVIOUS use. Calling cudaEventQuery() on it before recording will return whatever
    /// state that prior use left behind (typically cudaSuccess), which is meaningless to
    /// the new caller. ALL call sites in this file follow the pattern:
    ///   auto e = pool->acquire();
    ///   stream->record(*e);            // overwrites prior recording
    ///   /* later */ cudaEventQuery(e->get()) / event->synchronize()
    /// Do not query/synchronize before recording. (Not enforced at the type level today;
    /// if a future caller diverges, prefer wrapping in a small RecordedEvent helper.)
    [[nodiscard]] std::shared_ptr<runtime::CudaEvent> acquire();

private:
    void release(runtime::CudaEvent* event);

    std::mutex mMutex;
    std::vector<std::unique_ptr<runtime::CudaEvent>> mFreeEvents;
};

// ============================================================================
// P2pHandleExporter — local handle export (scan, export, UDS server)
// ============================================================================

/// @brief Owns local P2pMemInfo and all POSIX-FD bookkeeping. All public accessors are
/// thread-safe; export/remove and read paths can run concurrently. mInfoMutex protects
/// every mLocalInfo / mDetectedHandleType / mUdsPath read or write — making "is the agent
/// supported" and "give me the wire blob" atomic at the exporter boundary instead of
/// being a documented contract on the caller.
class P2pHandleExporter
{
public:
    explicit P2pHandleExporter(CUdevice localDevice);
    ~P2pHandleExporter();

    P2pHandleExporter(P2pHandleExporter const&) = delete;
    P2pHandleExporter& operator=(P2pHandleExporter const&) = delete;

    /// @brief Detect shareable handle type, scan registered VA with cuMemGetAddressRange, export handles.
    void exportHandles(RegisterDescs const& descs);

    /// @brief Remove handles corresponding to a deregistered range.
    void removeHandles(RegisterDescs const& descs);

    /// @brief Atomically check supported and serialize. Returns empty string if not supported.
    /// Replaces the old isSupported()+getLocalInfo() pair, eliminating the read race where
    /// register/deregister could mutate mLocalInfo between those two calls. Serialization
    /// runs under the lock — this is fine because (a) it's not on the hot path (only invoked
    /// during peer registration), and (b) it's still cheaper than copying the whole struct
    /// out of the lock first.
    [[nodiscard]] std::string serializeIfSupported() const;

    /// @brief Snapshot read of the supported flag. Provided for callers that genuinely only
    /// need a yes/no without the blob (none in production today; kept for future use and
    /// for symmetry with serializeIfSupported). Note that the answer is point-in-time; a
    /// concurrent registerMemory could change it immediately after.
    [[nodiscard]] bool isSupported() const;

private:
    void startUdsServer();
    void stopUdsServer();

    [[nodiscard]] size_t getVmmGranularity() const;

    /// @brief Scan [scanStart, scanStart+scanSize) via cuMemGetAddressRange and export each physical chunk.
    /// Chunks are appended to pool.chunks; for POSIX FD mode the matching FDs are appended to pool.fds
    /// in the same order, so removeHandles only needs to walk one pool.
    void detectAndExportChunks(
        CUdeviceptr scanStart, size_t scanSize, CUdeviceptr poolBase, size_t poolTotalSize, P2pMemPool& pool);

    void exportSingleChunkFabric(CUdeviceptr chunkBase, size_t chunkSize, CUdeviceptr poolBase, P2pMemPool& pool);
    void exportSingleChunkPosixFd(CUdeviceptr chunkBase, size_t chunkSize, CUdeviceptr poolBase, P2pMemPool& pool);
    void exportSingleChunkCudaIpc(CUdeviceptr poolBase, size_t poolTotalSize, P2pMemPool& pool);

    CUdevice mLocalDevice;
    P2pMemInfo mLocalInfo;
    VmmHandleType mDetectedHandleType{VmmHandleType::kNone};

    /// Single mutex protecting mLocalInfo (all fields), mDetectedHandleType, and mUdsPath.
    /// Held by:
    ///   - exportHandles / removeHandles (writers): for the entire mutation pass, so the
    ///     UDS server thread never sees a half-extended pool's FDs or torn struct fields.
    ///   - UDS server thread (reader): brief snapshot of pools[*].fds before sending FDs.
    ///   - serializeIfSupported / isSupported (readers): atomic supported+blob handoff to
    ///     getLocalAgentDesc, eliminating TOCTOU between supported and pool layout.
    /// `mutable` so const accessors can lock.
    mutable std::mutex mInfoMutex;

    // UDS server for POSIX FD sharing
    std::string mUdsPath;
    int mUdsServerSocket{-1};
    std::thread mUdsServerThread;
    std::atomic<bool> mUdsServerRunning{false};
};

// ============================================================================
// P2pRemoteMappingRegistry — thread-safe map of remote mappings
// ============================================================================

class P2pRemoteMappingRegistry
{
public:
    explicit P2pRemoteMappingRegistry(CUdevice localDevice);
    ~P2pRemoteMappingRegistry();

    P2pRemoteMappingRegistry(P2pRemoteMappingRegistry const&) = delete;
    P2pRemoteMappingRegistry& operator=(P2pRemoteMappingRegistry const&) = delete;

    void importAndMap(std::string const& name, P2pMemInfo const& info);
    void cleanup(std::string const& name);

    [[nodiscard]] bool hasMapping(std::string const& name) const;
    [[nodiscard]] bool hasImportFailed(std::string const& name) const;

    /// @brief Return a snapshot pointer. Kept alive by shared_ptr even if cleanup runs concurrently.
    [[nodiscard]] std::shared_ptr<RemoteP2pMapping const> get(std::string const& name) const;

    /// @brief Translate remote address to local mapped address. Throws if [addr, addr+size) exits
    /// the registered range; returns nullptr if the address is not in any pool.
    [[nodiscard]] static void* translate(RemoteP2pMapping const& mapping, uintptr_t remoteAddr, size_t transferSize);

private:
    CUdevice mLocalDevice;
    mutable std::shared_mutex mMutex;
    std::unordered_map<std::string, std::shared_ptr<RemoteP2pMapping>> mMappings;
    std::unordered_set<std::string> mFailed;
};

// ============================================================================
// P2pTransferContext — per-thread resources (no locking)
// ============================================================================

/// @brief Owns per-thread CUDA stream + prealloc buffers + per-worker streams.
/// Created lazily on first use from a given caller thread. Methods are NOT thread-safe —
/// callers must ensure one context per thread (see P2pTransferContextPool).
class P2pTransferContext
{
public:
    P2pTransferContext(CUdevice localDevice, std::shared_ptr<CudaEventPool> eventPool, int batchCopyThreads,
        size_t multiThreadMinOps, bool cubZeroCopy, std::shared_ptr<P2pAgentCounters> counters = nullptr);

    ~P2pTransferContext() = default;

    P2pTransferContext(P2pTransferContext const&) = delete;
    P2pTransferContext& operator=(P2pTransferContext const&) = delete;

    /// @brief cub::DeviceMemcpy::Batched path, tuned for many small segments.
    [[nodiscard]] std::unique_ptr<TransferStatus> submitWithCubBatched(
        std::vector<void*> const& srcPtrs, std::vector<void*> const& dstPtrs, std::vector<size_t> const& sizes);

    /// @brief cudaMemcpyBatchAsync path (single- or multi-thread), tuned for fewer larger segments.
    [[nodiscard]] std::unique_ptr<TransferStatus> submitWithMemcpyBatch(
        std::vector<void*> const& srcPtrs, std::vector<void*> const& dstPtrs, std::vector<size_t> const& sizes);

private:
    void ensureBuffers(size_t batchSize, size_t cubTempBytes);
    void ensureWorkerPoolAndStreams();

    CUdevice mLocalDevice;
    bool mCubZeroCopy;
    int mBatchCopyThreads;
    size_t mMultiThreadMinOps;

    std::shared_ptr<runtime::CudaStream> mSubmitStream;
    std::shared_ptr<runtime::BufferManager> mBufferManager;
    std::shared_ptr<CudaEventPool> mEventPool;
    std::shared_ptr<P2pAgentCounters> mCounters; ///< nullable; only populated when owned by a P2pTransferAgent

    // Per-Context worker pool + per-worker streams. Lazily constructed on the first call
    // that actually needs the multi-thread path (numOps >= mMultiThreadMinOps).
    // Owning the pool per-Context (i.e. per caller thread) eliminates cross-caller queue
    // contention — each caller drives its own N workers independently.
    std::once_flag mWorkerPoolInit;
    std::unique_ptr<BatchCopyWorkerPool> mWorkerPool;
    std::vector<std::shared_ptr<runtime::CudaStream>> mBatchCopyStreams;

    // Prealloc buffers (this thread exclusive — no lock needed)
    runtime::IBuffer::UniquePtr mCombinedGpu;
    runtime::IBuffer::UniquePtr mCombinedPinned;
    runtime::IBuffer::UniquePtr mCubTempStorage;
    size_t mMaxBatchSize{0};
    size_t mCubTempStorageSize{0};
};

// ============================================================================
// P2pTransferContextPool — one context per caller thread
// ============================================================================
//
// Lifetime model: a per-caller-thread Context is auto-erased when the caller thread
// exits. contextForCurrentThread() registers a thread_local guard that, on thread
// destruction, calls eraseForCurrentThread() on this pool. Without this, a long-lived
// process whose callers are transient threads would leak Contexts (each holds a CUDA
// stream + ~64 MB cubTempStorage + an N-thread worker pool); even worse, the OS may
// reuse a thread::id from an exited thread, handing the new thread a stale Context
// created in a different CUDA context.
//
// MUST be held by std::shared_ptr — the thread_local guard captures a weak_ptr so it
// can decline to call back into a pool that has already been destroyed. The private
// ctor + static `create()` enforces this at the type level.

class P2pTransferContextPool : public std::enable_shared_from_this<P2pTransferContextPool>
{
public:
    [[nodiscard]] static std::shared_ptr<P2pTransferContextPool> create(CUdevice localDevice,
        std::shared_ptr<CudaEventPool> eventPool, int batchCopyThreads, size_t multiThreadMinOps, bool cubZeroCopy,
        std::shared_ptr<P2pAgentCounters> counters = nullptr);

    P2pTransferContextPool(P2pTransferContextPool const&) = delete;
    P2pTransferContextPool& operator=(P2pTransferContextPool const&) = delete;

    /// @brief Get (or lazily create) the context for the calling thread.
    /// First call from a given thread also installs a thread-exit guard that will
    /// call eraseForCurrentThread() on this pool when the thread terminates.
    P2pTransferContext& contextForCurrentThread();

    /// @brief Drop this thread's context. Called automatically on thread exit by the
    /// installed guard, but also safe to call manually. No-op if there is no entry.
    /// Marked noexcept because it runs from a thread_local destructor where exceptions
    /// would terminate.
    void eraseForCurrentThread() noexcept;

    /// @brief Test-only: number of per-thread Contexts currently held. Used by the
    /// thread-exit-guard regression test to assert that exited callers don't leak.
    [[nodiscard]] size_t numContexts() const;

private:
    P2pTransferContextPool(CUdevice localDevice, std::shared_ptr<CudaEventPool> eventPool, int batchCopyThreads,
        size_t multiThreadMinOps, bool cubZeroCopy, std::shared_ptr<P2pAgentCounters> counters);

    CUdevice mLocalDevice;
    std::shared_ptr<CudaEventPool> mEventPool;
    int mBatchCopyThreads;
    size_t mMultiThreadMinOps;
    bool mCubZeroCopy;
    std::shared_ptr<P2pAgentCounters> mCounters;

    mutable std::mutex mMutex;
    std::unordered_map<std::thread::id, std::unique_ptr<P2pTransferContext>> mContexts;
};

// ============================================================================
// P2pTransferAgent — facade owned by NixlTransferAgent
// ============================================================================

class P2pTransferAgent
{
public:
    P2pTransferAgent();
    ~P2pTransferAgent() = default;

    P2pTransferAgent(P2pTransferAgent const&) = delete;
    P2pTransferAgent& operator=(P2pTransferAgent const&) = delete;

    [[nodiscard]] P2pHandleExporter& exporter() noexcept
    {
        return mExporter;
    }

    [[nodiscard]] P2pHandleExporter const& exporter() const noexcept
    {
        return mExporter;
    }

    [[nodiscard]] P2pRemoteMappingRegistry& registry() noexcept
    {
        return mRegistry;
    }

    [[nodiscard]] P2pRemoteMappingRegistry const& registry() const noexcept
    {
        return mRegistry;
    }

    /// @brief Return this thread's context (lazily created).
    [[nodiscard]] P2pTransferContext& contextForCurrentThread()
    {
        return mContextPool->contextForCurrentThread();
    }

    /// @brief Atomic supported+serialize handoff for AgentDesc construction. Returns
    /// empty string if no shareable handles were exported.
    [[nodiscard]] std::string serializeLocalInfoIfSupported() const
    {
        return mExporter.serializeIfSupported();
    }

    [[nodiscard]] bool isSupported() const
    {
        return mExporter.isSupported();
    }

    /// @brief Test-only read of single/multi-thread memcpyBatch counters.
    [[nodiscard]] P2pAgentCounters::Snapshot getCountersSnapshot() const noexcept
    {
        return mCounters ? mCounters->snapshot() : P2pAgentCounters::Snapshot{0, 0};
    }

    /// @brief Test-only count of per-thread Contexts. Used to verify thread-exit cleanup.
    [[nodiscard]] size_t numContexts() const
    {
        return mContextPool->numContexts();
    }

private:
    CUdevice mLocalDevice{0};
    std::shared_ptr<CudaEventPool> mEventPool;
    int mBatchCopyThreads{1};
    size_t mMultiThreadMinOps{0};
    bool mCubZeroCopy{false};
    std::shared_ptr<P2pAgentCounters> mCounters;

    P2pHandleExporter mExporter;
    P2pRemoteMappingRegistry mRegistry;
    /// shared_ptr because the per-thread guards installed by contextForCurrentThread
    /// hold weak_ptrs to it; must outlive the agent only briefly while a guard runs.
    std::shared_ptr<P2pTransferContextPool> mContextPool;
};

} // namespace tensorrt_llm::executor::kv_cache
