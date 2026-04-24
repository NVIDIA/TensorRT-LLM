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
    [[nodiscard]] std::shared_ptr<runtime::CudaEvent> acquire();

private:
    void release(runtime::CudaEvent* event);

    std::mutex mMutex;
    std::vector<std::unique_ptr<runtime::CudaEvent>> mFreeEvents;
};

// ============================================================================
// P2pHandleExporter — local handle export (scan, export, UDS server)
// ============================================================================

/// @brief Owns local P2pMemInfo and all POSIX-FD bookkeeping. Thread-safety: register/deregister
/// are expected to be called from the main setup thread; submit-side reads getLocalInfo() after that.
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

    [[nodiscard]] P2pMemInfo const& getLocalInfo() const noexcept
    {
        return mLocalInfo;
    }

    [[nodiscard]] bool isSupported() const noexcept
    {
        return mLocalInfo.supported;
    }

private:
    void startUdsServer();
    void stopUdsServer();

    [[nodiscard]] size_t getVmmGranularity() const;

    /// @brief Scan [scanStart, scanStart+scanSize) via cuMemGetAddressRange and export each physical chunk.
    void detectAndExportChunks(CUdeviceptr scanStart, size_t scanSize, CUdeviceptr poolBase, size_t poolTotalSize,
        std::vector<P2pMemChunk>& chunks);

    void exportSingleChunkFabric(
        CUdeviceptr chunkBase, size_t chunkSize, CUdeviceptr poolBase, std::vector<P2pMemChunk>& chunks);
    void exportSingleChunkPosixFd(
        CUdeviceptr chunkBase, size_t chunkSize, CUdeviceptr poolBase, std::vector<P2pMemChunk>& chunks);
    void exportSingleChunkCudaIpc(CUdeviceptr poolBase, size_t poolTotalSize, std::vector<P2pMemChunk>& chunks);

    CUdevice mLocalDevice;
    P2pMemInfo mLocalInfo;
    VmmHandleType mDetectedHandleType{VmmHandleType::kNone};

    // Exported POSIX file descriptors (one per chunk, in pool/chunk order)
    std::vector<int> mExportedFds;
    std::mutex mExportedFdsMutex;

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
        size_t multiThreadMinOps, bool cubZeroCopy);

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

class P2pTransferContextPool
{
public:
    P2pTransferContextPool(CUdevice localDevice, std::shared_ptr<CudaEventPool> eventPool, int batchCopyThreads,
        size_t multiThreadMinOps, bool cubZeroCopy);

    P2pTransferContextPool(P2pTransferContextPool const&) = delete;
    P2pTransferContextPool& operator=(P2pTransferContextPool const&) = delete;

    /// @brief Get (or lazily create) the context for the calling thread.
    P2pTransferContext& contextForCurrentThread();

private:
    CUdevice mLocalDevice;
    std::shared_ptr<CudaEventPool> mEventPool;
    int mBatchCopyThreads;
    size_t mMultiThreadMinOps;
    bool mCubZeroCopy;

    std::mutex mMutex;
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
        return mContextPool.contextForCurrentThread();
    }

    [[nodiscard]] bool isSupported() const noexcept
    {
        return mExporter.isSupported();
    }

private:
    CUdevice mLocalDevice{0};
    std::shared_ptr<CudaEventPool> mEventPool;
    int mBatchCopyThreads{1};
    size_t mMultiThreadMinOps{0};
    bool mCubZeroCopy{false};

    P2pHandleExporter mExporter;
    P2pRemoteMappingRegistry mRegistry;
    P2pTransferContextPool mContextPool;
};

} // namespace tensorrt_llm::executor::kv_cache
