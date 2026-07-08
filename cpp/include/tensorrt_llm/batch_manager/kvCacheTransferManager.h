/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/cudaEvent.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tr = tensorrt_llm::runtime;
namespace kvc = tensorrt_llm::executor::kv_cache;

#pragma once

namespace tensorrt_llm::testing
{
class KVCacheTransferManagerTestAccess;
} // namespace tensorrt_llm::testing

namespace tensorrt_llm::batch_manager::kv_cache_manager
{

/// @brief Statistics for block transfers. Returned by KVCacheTransferManager::getAndResetTransferStats().
/// All counters are reset on read.
/// - onboard/offload: transfers between secondary (host) and primary (GPU) memory.
/// - intraDeviceCopy: GPU-to-GPU block copies (e.g. partial reuse when source block has refs).
struct KvCacheTransferStats
{
    SizeType32 onboardBlocks{0};
    std::size_t onboardBytes{0};
    SizeType32 offloadBlocks{0};
    std::size_t offloadBytes{0};
    SizeType32 intraDeviceCopyBlocks{0};
    std::size_t intraDeviceCopyBytes{0};
};

// The TransferManager accelerates transfers to/from the GPU by overlapping HtoD and DtoH transfers, and tracks ongoing
// transfers in order to avoid race conditions. It is functionally equivalent to the prior approach of putting all
// transfers into the forward pass stream. This is only ever used as a component of a KVCacheManager.
class KVCacheTransferManager
{
public:
    explicit KVCacheTransferManager(
        tr::BufferManager const& bufferManager, std::shared_ptr<kvc::BaseLoopbackAgent> loopbackAgent = nullptr);

    //! \brief Onboard a block to gpu memory.
    void onboard(BlockPtr const& offloadBlock, BlockPtr const& block, std::vector<KVCacheBlockPool> const& pools,
        int numTokensToCopy = 0, executor::KvCacheTransferMode mode = executor::KvCacheTransferMode::DRAM,
        std::string const& directory = "");

    //! \brief Offload a block to cpu memory.
    void offload(BlockPtr const& block, BlockPtr const& offloadBlock, std::vector<KVCacheBlockPool> const& pools,
        int numTokensToCopy = 0, executor::KvCacheTransferMode mode = executor::KvCacheTransferMode::DRAM,
        std::string const& directory = "");

    //! \brief Synchronize internal streams with bufferManager stream.
    //! \details The buffer manager uses the same stream as the prefill and decode kernels. This method ensures that the
    //! internal kernels used for offloading and onboarding will wait for prefill and decode kernels before performing
    //! any block copies. This method must be called before the first call to
    //! KVCacheManager::addSequenceBatch in every step.
    void syncWithBufferManager();

    //! \brief Synchronize bufferManager stream with internal streams. This method ensures that prefill and decode
    //! kernels for next step will wait for offloading and onboarding work that has already been scheduled. This method
    //! must be called after the last call to KVCacheManager::addSequenceBatch in every step.
    //! \brief Disk tier: write a host-resident block's bytes to its slot files (synchronous POSIX).
    void spillToFile(BlockPtr const& srcHostBlock, SizeType32 diskSlot, std::vector<KVCacheBlockPool> const& pools,
        std::string const& directory);

    //! \brief Disk tier: read a block's slot files into GPU memory. With a reader pool the block's reads are
    //! handed off and tracked under \p trackBlockId (POSIX or GDS, decided by the reader); the owning request
    //! must be gated on isBlockReadPending() / areBlocksReady() before it is forwarded. With no pool
    //! (\p trackBlockId < 0) the read is synchronous on the calling thread.
    void loadFromFile(BlockPtr const& dstPrimaryBlock, SizeType32 diskSlot, std::vector<KVCacheBlockPool> const& pools,
        std::string const& directory, std::int32_t trackBlockId = -1);

    //! \brief True while block blockId still has an in-flight disk read (not yet safe to read on the GPU).
    [[nodiscard]] bool isBlockReadPending(std::int32_t blockId);

    //! \brief Lock-free: true if ANY disk read is in flight, letting areBlocksReady() skip its scan.
    [[nodiscard]] bool anyReadPending() const noexcept
    {
        return mReadInflightCount.load(std::memory_order_acquire) > 0;
    }

    //! \brief True when a reader pool is present, so onboards can be detached from the scheduler thread.
    [[nodiscard]] bool asyncDiskReadEnabled() const
    {
        return mNumDiskReaders >= 1;
    }

    //! \brief Disk tier (unstaged async): spill a host block's bytes by handing the writer the source
    //! pointers directly (no staging memcpy). The caller MUST keep the source host slot pinned until
    //! \p spillId appears in drainCompletedSpills().
    void spillToFileUnstaged(BlockPtr const& srcHostBlock, SizeType32 diskSlot,
        std::vector<KVCacheBlockPool> const& pools, std::string const& directory, std::uint64_t spillId);

    //! \brief Disk tier: return + clear the set of spill ids whose async writes fully completed since
    //! the last call (unstaged path). The block manager reaps the corresponding pinned blocks.
    [[nodiscard]] std::vector<std::uint64_t> drainCompletedSpills();

    //! \brief Whether the background async-store writer thread is active (TLLM_KV_DISK_ASYNC_STORE).
    [[nodiscard]] bool asyncDiskStoreEnabled() const
    {
        return mAsyncDiskStore;
    }

    //! \brief True if the async write queue is at capacity. Used to shed best-effort (non-retained) spills under
    //! writer saturation; retained spills bypass the cap and are never shed.
    [[nodiscard]] bool diskWriteQueueFull();

    void syncTransfers();

    ~KVCacheTransferManager();

    //! \brief Get transfer stats accumulated since last call, and reset the counters.
    [[nodiscard]] KvCacheTransferStats getAndResetTransferStats();

private:
    friend class ::tensorrt_llm::testing::KVCacheTransferManagerTestAccess;

    //! \brief Get pointer to pool specified by cache block.
    static tr::ITensor::SharedPtr computeBlockPointer(
        BlockPtr const& block, std::vector<KVCacheBlockPool> const& pools, size_t poolIdx);

    //! \brief Get pool-qualified index for pending transfer tracking.
    [[nodiscard]] static kernels::KVCacheIndex::UnderlyingType getPendingTransferIndex(BlockPtr const& block);

    /*!
     * \brief The key method that copies the src block to the dst block.
     *
     * \param src             Source block
     * \param dst             Destination block
     * \param pools           Pools describing memory layout for KV blocks
     * \param isOffload       true => GPU->CPU/file, false => CPU/file->GPU
     * \param numTokensToCopy if > 0, partial copy is done
     * \param mode            See \ref executor::KvCacheTransferMode
     * \param directory       Directory to save the file if mode is GDS or POSIX_DEBUG_FALLBACK
     *
     * The default param is set to executor::KvCacheTransferMode::DRAM.
     */
    void copyBlock(BlockPtr const& src, BlockPtr const& dst, std::vector<KVCacheBlockPool> const& pools, bool isOffload,
        int numTokensToCopy = 0, executor::KvCacheTransferMode mode = executor::KvCacheTransferMode::DRAM,
        std::string const& directory = "");

    //! \brief Compute total bytes actually transferred for a block copy across all pools.
    //! \param pools The pool descriptors.
    //! \param numTokensToCopy Number of tokens for partial copy (0 means full block).
    [[nodiscard]] std::size_t computeBlockTransferBytes(
        std::vector<KVCacheBlockPool> const& pools, int numTokensToCopy) const;

    runtime::BufferManager mBufferManager;
    runtime::BufferManager mOnboardManager;
    runtime::BufferManager mOffloadManager;

    // Track reads and writes for blocks. Note that it is the pool-qualified memory pool index
    // that identifies the raw memory blocks involved in I/O, not the block Id.
    std::unordered_map<kernels::KVCacheIndex::UnderlyingType, tr::CudaEvent> mPendingReads;
    std::unordered_map<kernels::KVCacheIndex::UnderlyingType, tr::CudaEvent> mPendingWrites;
    // Reference to parent loopback agent
    std::shared_ptr<kvc::BaseLoopbackAgent> mLoopbackAgent;
    int mDeviceId;
    // Disk-tier onboard (disk->GPU) uses GPUDirect Storage when TLLM_KV_DISK_GDS is set;
    // otherwise POSIX staging. Read once at construction.
    bool mDiskUseGds{std::getenv("TLLM_KV_DISK_GDS") != nullptr};

    // ---- Disk-tier async store: spill host->disk OFF the scheduler thread ----
    // Gated by TLLM_KV_DISK_ASYNC_STORE. Unset => synchronous spill (current behavior).
    bool const mAsyncDiskStore{std::getenv("TLLM_KV_DISK_ASYNC_STORE") != nullptr};

    struct DiskWriteJob
    {
        std::string filename;
        std::size_t bytes{0};
        std::vector<std::uint8_t> staged; // staged-memcpy path (used when src == nullptr)
        void const* src{nullptr};         // unstaged path: writer reads this pinned host pointer directly
        std::uint64_t spillId{0};         // >0 => track per-spill completion for the reserved-pool reap
        bool retained{false};             // retained spill: bypasses the queue cap, never dropped
    };

    std::vector<std::thread> mDiskWriters;
    std::mutex mDiskMutex;
    std::condition_variable mDiskQueueCv;
    std::condition_variable mDiskInflightCv;
    std::queue<DiskWriteJob> mDiskWriteQueue;
    std::unordered_map<std::string, int> mDiskInflight;
    // Unstaged reserved-pool support: outstanding pool-writes per spill; when a spill's last write
    // completes, publish its id so the block manager can reap the pinned source block.
    std::unordered_map<std::uint64_t, int> mSpillRemaining;
    std::vector<std::uint64_t> mCompletedSpills;
    bool mDiskWriterStop{false};
    std::size_t const mDiskWriteQueueMax{[]
        {
            auto* e = std::getenv("TLLM_KV_DISK_WRITE_QUEUE");
            return e ? std::stoul(e) : 1024UL;
        }()};
    // Number of background writer threads draining the shared queue (TLLM_KV_DISK_WRITERS, default 1).
    std::size_t const mNumDiskWriters{[]
        {
            auto* e = std::getenv("TLLM_KV_DISK_WRITERS");
            auto n = e ? std::stoul(e) : 1UL;
            return n < 1UL ? 1UL : n;
        }()};

    void diskWriterLoop();
    void enqueueDiskWrite(std::string filename, void const* src, std::size_t bytes, bool retained);
    void enqueueDiskWriteUnstaged(
        std::string filename, void const* src, std::size_t bytes, std::uint64_t spillId, bool retained);
    void waitForDiskSlotWrites(std::string const& filename);

    // ---- Disk-tier async ONBOARD: read slot files disk->GPU OFF the scheduler thread ----
    // Gated by TLLM_KV_DISK_READERS (0/unset => synchronous read = current behavior; N>=1 => N reader
    // threads drain a shared read queue). A job carries all of one block's pools; a request is forward-safe
    // once areBlocksReady() reports every block it holds has landed. See loadFromFile / isBlockReadPending.
    struct DiskReadJob
    {
        std::int32_t blockId{-1};       // block whose readiness this job satisfies (tracking key)
        bool useGds{false};             // GDS DMA (true) vs POSIX read + H2D copy (false)
        std::vector<void*> dsts;        // GPU destination pointer, per pool
        std::vector<std::string> files; // disk slot file, per pool
        std::vector<std::size_t> bytes; // byte count, per pool
    };

    std::size_t const mNumDiskReaders{[]
        {
            auto* e = std::getenv("TLLM_KV_DISK_READERS");
            return e ? std::stoul(e) : 0UL;
        }()};
    std::vector<std::thread> mDiskReaders;
    std::mutex mReadMutex;
    std::condition_variable mReadQueueCv;
    std::queue<DiskReadJob> mReadQueue;
    // Blocks with a disk read in flight, keyed by the matched-identity block id so every request reusing that
    // prefix gates on the same key. A block is forward-safe once it leaves this set.
    std::unordered_set<std::int32_t> mPendingBlockReads;
    // Lock-free mirror of mPendingBlockReads.size() (updated under mReadMutex on every insert/erase) so
    // areBlocksReady() can skip its per-block scan + the mutex when nothing is in flight (common case).
    std::atomic<std::size_t> mReadInflightCount{0};
    bool mDiskReaderStop{false};

    void diskReaderLoop();
    void enqueueDiskRead(DiskReadJob job);

    // Cumulative transfer statistics, reset on each call to getAndResetTransferStats().
    // Protected by mStatsMutex for thread-safe access.
    mutable std::mutex mStatsMutex;
    SizeType32 mOnboardBlockCount{0};
    std::size_t mOnboardByteCount{0};
    SizeType32 mOffloadBlockCount{0};
    std::size_t mOffloadByteCount{0};
    SizeType32 mIntraDeviceCopyBlockCount{0};
    std::size_t mIntraDeviceCopyByteCount{0};
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager
