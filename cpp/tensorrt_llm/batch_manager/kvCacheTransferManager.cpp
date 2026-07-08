/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cstdint>
#include <cstring>

#include "tensorrt_llm/batch_manager/kvCacheTransferManager.h"

#include "tensorrt_llm/batch_manager/kvCacheEventManager.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/kernels/kvCachePartialCopy.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/cudaEvent.h"
#include "tensorrt_llm/runtime/cudaStream.h"

namespace tr = tensorrt_llm::runtime;
namespace tk = tensorrt_llm::kernels;
namespace kvc = tensorrt_llm::executor::kv_cache;

namespace tensorrt_llm::batch_manager::kv_cache_manager
{

static bool gpuToFilePosix(tr::ITensor::SharedPtr const& srcPtr, std::string const& filename)
{
    int fd = ::open(filename.c_str(), O_CREAT | O_WRONLY, 0664);
    TLLM_CHECK_WITH_INFO(fd >= 0, "Failed to open '%s' for writing (POSIX fallback)", filename.c_str());

    ssize_t numBytes = static_cast<ssize_t>(srcPtr->getSizeInBytes());
    std::vector<uint8_t> hostBuffer(numBytes);

    cudaError_t cpyErr = cudaMemcpy(hostBuffer.data(), srcPtr->data(), numBytes, cudaMemcpyDeviceToHost);
    TLLM_CHECK_WITH_INFO(cpyErr == cudaSuccess, "cudaMemcpy to host failed, error=%d", cpyErr);

    ssize_t written = ::write(fd, hostBuffer.data(), numBytes);
    TLLM_CHECK_WITH_INFO(written == numBytes, "POSIX write: short/failed write %zd/%zd", written, numBytes);

    TLLM_LOG_DEBUG("Wrote %zd bytes to %s (POSIX fallback)", written, filename.c_str());

    ::close(fd);
    return true;
}

static bool fileToGpuPosix(tr::ITensor::SharedPtr const& dstPtr, std::string const& filename)
{
    int fd = ::open(filename.c_str(), O_RDONLY);
    TLLM_CHECK_WITH_INFO(fd >= 0, "Failed to open '%s' for reading (POSIX fallback)", filename.c_str());

    ssize_t numBytes = static_cast<ssize_t>(dstPtr->getSizeInBytes());
    std::vector<uint8_t> hostBuffer(numBytes);

    ssize_t bytesRead = ::read(fd, hostBuffer.data(), numBytes);
    TLLM_CHECK_WITH_INFO(bytesRead == numBytes, "POSIX read: short/failed read %zd/%zd", bytesRead, numBytes);

    TLLM_LOG_DEBUG("Read %zd bytes from %s (POSIX fallback)", bytesRead, filename.c_str());

    cudaError_t cpyErr = cudaMemcpy(dstPtr->data(), hostBuffer.data(), numBytes, cudaMemcpyHostToDevice);
    TLLM_CHECK_WITH_INFO(cpyErr == cudaSuccess, "cudaMemcpy to device failed, error=%d", cpyErr);

    ::close(fd);
    return true;
}

KVCacheTransferManager::KVCacheTransferManager(
    tr::BufferManager const& bufferManager, std::shared_ptr<kvc::BaseLoopbackAgent> loopbackAgent)
    : mBufferManager{bufferManager}
    , mOnboardManager(std::make_shared<tr::CudaStream>())
    , mOffloadManager(std::make_shared<tr::CudaStream>())
    , mLoopbackAgent{loopbackAgent}
{
    TLLM_CUDA_CHECK(cudaGetDevice(&mDeviceId));
    TLLM_CHECK(mDeviceId != -1);
    if (mAsyncDiskStore)
    {
        for (std::size_t i = 0; i < mNumDiskWriters; ++i)
        {
            mDiskWriters.emplace_back(&KVCacheTransferManager::diskWriterLoop, this);
        }
        TLLM_LOG_INFO(
            "[disk-tier] async store ENABLED (writers=%zu, write-queue max=%zu)", mNumDiskWriters, mDiskWriteQueueMax);
    }
    if (asyncDiskReadEnabled())
    {
        for (std::size_t i = 0; i < mNumDiskReaders; ++i)
        {
            mDiskReaders.emplace_back(&KVCacheTransferManager::diskReaderLoop, this);
        }
        TLLM_LOG_INFO("[disk-tier] async onboard ENABLED (readers=%zu)", mNumDiskReaders);
    }
}

KVCacheTransferManager::~KVCacheTransferManager()
{
    if (!mDiskWriters.empty())
    {
        {
            std::lock_guard<std::mutex> lock(mDiskMutex);
            mDiskWriterStop = true;
        }
        mDiskQueueCv.notify_all();
        for (auto& t : mDiskWriters)
        {
            if (t.joinable())
            {
                t.join();
            }
        }
    }
    if (!mDiskReaders.empty())
    {
        {
            std::lock_guard<std::mutex> lock(mReadMutex);
            mDiskReaderStop = true;
        }
        mReadQueueCv.notify_all();
        for (auto& t : mDiskReaders)
        {
            if (t.joinable())
            {
                t.join();
            }
        }
    }
}

tr::ITensor::SharedPtr KVCacheTransferManager::computeBlockPointer(
    BlockPtr const& block, std::vector<KVCacheBlockPool> const& pools, size_t poolIdx)
{
    TLLM_CHECK_WITH_INFO(!pools.empty(), "Pool index %lu is out of bounds", poolIdx);
    auto const& pool = pools.at(poolIdx);
    auto ptr = block->isPrimary() ? pool.primaryPtr : pool.secondaryPtr;
    auto const blockOffset = block->getMemoryPoolBlockIndex();
    tr::ITensor::SharedPtr blockTensor{tr::ITensor::slice(ptr, blockOffset, 1)};
    return blockTensor;
}

tk::KVCacheIndex::UnderlyingType KVCacheTransferManager::getPendingTransferIndex(BlockPtr const& block)
{
    auto const blockOffset = block->getMemoryPoolBlockIndex();
    return block->isPrimary() ? blockOffset : blockOffset | tk::KVCacheIndex::kSecondaryPoolFlag;
}

void KVCacheTransferManager::copyBlock(BlockPtr const& src, BlockPtr const& dst,
    std::vector<KVCacheBlockPool> const& pools, bool isOffload, int numTokensToCopy, executor::KvCacheTransferMode mode,
    std::string const& directory)
{
    TLLM_LOG_DEBUG("copyBlock entered: srcId=%d, dstId=%d, isOffload=%s, mode=%d", src->getBlockId(), dst->getBlockId(),
        (isOffload ? "true" : "false"), static_cast<int>(mode));

    if (mode == executor::KvCacheTransferMode::DRAM)
    {
        TLLM_LOG_DEBUG("Using DRAM-based copy (GPU <-> CPU) for this block.");

        // Iterate over all pools, partial-copy logic
        for (size_t poolIdx = 0; poolIdx < pools.size(); ++poolIdx)
        {
            auto const& pool = pools[poolIdx];

            // For layer-first layout pools, block data is non-contiguous across layers.
            // Pool shape: {numLayers, numBlocks, kvFactor, blockSize}. For a fixed block
            // index, per-layer slices are contiguous rows of (kvFactor * blockSize) elements,
            // separated by a stride of numBlocks rows between layers. Issue this as a single
            // pitched cudaMemcpy2DAsync instead of one cudaMemcpyAsync per layer.
            if (pool.layerFirstLayout)
            {
                auto srcPool = src->isPrimary() ? pool.primaryPtr : pool.secondaryPtr;
                auto dstPool = dst->isPrimary() ? pool.primaryPtr : pool.secondaryPtr;
                auto const srcBlockIdx = static_cast<size_t>(src->getMemoryPoolBlockIndex());
                auto const dstBlockIdx = static_cast<size_t>(dst->getMemoryPoolBlockIndex());

                // Compute pitches from each pool independently: primary and secondary pools
                // may have different block counts (mNumPrimaryBlocks vs mNumSecondaryBlocks),
                // so their per-layer strides differ. Using the primary shape for both pitches
                // would corrupt host-offloaded recurrent state on CPU<->GPU transfers.
                auto const& srcShape = srcPool->getShape();
                auto const& dstShape = dstPool->getShape();
                TLLM_CHECK_WITH_INFO(srcShape.nbDims >= 2,
                    "Expected layer-first KVCache pool to have at least 2 dims, got %d", srcShape.nbDims);
                TLLM_CHECK_WITH_INFO(dstShape.nbDims >= 2,
                    "Expected layer-first KVCache pool to have at least 2 dims, got %d", dstShape.nbDims);
                auto const srcLayerStrideBytes = srcPool->getSizeInBytes() / static_cast<size_t>(pool.numLayers);
                auto const dstLayerStrideBytes = dstPool->getSizeInBytes() / static_cast<size_t>(pool.numLayers);
                // rowBytes is the per-block per-layer payload — identical for primary and secondary.
                auto const rowBytes = srcLayerStrideBytes / static_cast<size_t>(srcShape.d[1]);

                auto* srcBase = static_cast<char*>(srcPool->data()) + srcBlockIdx * rowBytes;
                auto* dstBase = static_cast<char*>(dstPool->data()) + dstBlockIdx * rowBytes;

                auto stream = (isOffload ? mOffloadManager : mOnboardManager).getStream().get();
                TLLM_CUDA_CHECK(cudaMemcpy2DAsync(dstBase, dstLayerStrideBytes, srcBase, srcLayerStrideBytes, rowBytes,
                    static_cast<size_t>(pool.numLayers), cudaMemcpyDefault, stream));
                continue;
            }

            auto srcPtr = computeBlockPointer(src, pools, poolIdx);
            auto dstPtr = computeBlockPointer(dst, pools, poolIdx);

            // Does it contain block scales?
            auto containsBlockScales = pool.containsBlockScales;

            // If no partial tokens or if the dataType is not supported for partial copy, copy entire block.
            // Note that nvfp4 kv cache SFs use an interleaved layout, so we need to copy the entire block.
            if (numTokensToCopy <= 0 || srcPtr->getDataType() == nvinfer1::DataType::kINT4
                || srcPtr->getDataType() == nvinfer1::DataType::kFP4 || containsBlockScales)
            {
                // For partial copy not implemented with these data types,
                // just do a full copy.
                (isOffload ? mOffloadManager : mOnboardManager).copy(*srcPtr, *dstPtr);
            }
            else
            {
                int const tokensPerBlock = pool.tokensPerBlock;
                if (numTokensToCopy >= tokensPerBlock)
                {
                    // If requested tokens >= entire block, just do a full copy.
                    (isOffload ? mOffloadManager : mOnboardManager).copy(*srcPtr, *dstPtr);
                }
                else
                {
                    auto stream = (isOffload ? mOffloadManager : mOnboardManager).getStream().get();
                    int const numLayers = pool.numLayers;
                    int const kvFactor = pool.kvFactor;
                    int const numHeads = pool.numKvHeads;
                    int const sizePerHead = pool.sizePerHead;
                    auto shape = srcPtr->getShape();

                    TLLM_CHECK_WITH_INFO(
                        shape.nbDims == 4, "Expected KVCache block to have 4 dims, got %d", shape.nbDims);

                    tk::kvCacheBlockPartialCopy(*dstPtr, *srcPtr, numLayers, numHeads, tokensPerBlock, sizePerHead,
                        numTokensToCopy, kvFactor, stream);
                }
            }
        }

        TLLM_LOG_DEBUG("copyBlock: DRAM mode complete. Returning...");
        return;
    }

    std::vector<kvc::FileDesc> fileBlobs;
    std::vector<kvc::MemoryDesc> memoryBlobs;

    for (size_t poolIdx = 0; poolIdx < pools.size(); ++poolIdx)
    {
        TLLM_CHECK_WITH_INFO(!pools[poolIdx].layerFirstLayout,
            "File-based offload/onboard is not supported for layer-first layout pools");
        auto ptr = isOffload ? computeBlockPointer(src, pools, poolIdx) : computeBlockPointer(dst, pools, poolIdx);
        auto block_id = src->getBlockId();

        TLLM_CHECK_WITH_INFO(
            !directory.empty(), "Expected a directory path for KVCache offload, but none was provided.");

        int size = std::snprintf(nullptr, 0, "%s/block_%d_pool_%zu.bin", directory.c_str(), block_id, poolIdx);
        std::string filename;
        filename.resize(size + 1);
        std::snprintf(
            filename.data(), filename.size(), "%s/block_%d_pool_%zu.bin", directory.c_str(), block_id, poolIdx);

        if (mode == executor::KvCacheTransferMode::POSIX_DEBUG_FALLBACK)
        {
            TLLM_LOG_INFO("Forcing POSIX fallback for file: %s", filename.c_str());
            if (isOffload)
            {
                gpuToFilePosix(ptr, filename);
            }
            else
            {
                fileToGpuPosix(ptr, filename);
            }
            continue;
        }
        else if (mode == executor::KvCacheTransferMode::GDS)
        {

            int openFlags = isOffload ? (O_CREAT | O_WRONLY) : O_RDONLY;
            fileBlobs.emplace_back(filename, openFlags, 0664, ptr->getSizeInBytes());
            memoryBlobs.emplace_back(ptr->data(), ptr->getSizeInBytes(), mDeviceId);
        }
    }

    if (mode == executor::KvCacheTransferMode::GDS)
    {
        if (mLoopbackAgent == nullptr)
        {
            TLLM_LOG_DEBUG("KVCacheTransferManager: creating mLoopbackAgent lazily");
            kvc::BaseAgentConfig config{std::string("GDSAgent"), true, true};
            mLoopbackAgent = kvc::makeLoopbackAgent("nixl", &config);
        }

        kvc::FileDescs fileDescs(std::move(fileBlobs));
        kvc::MemoryDescs memoryDescs(kvc::MemoryType::kVRAM, memoryBlobs);

        mLoopbackAgent->executeLoopbackRequest(memoryDescs, fileDescs, isOffload);
    }
}

//
// Note about recording events to wait for cudaMemcpyAsync calls between blocks:
// The memory copy involves raw memory blocks, which are identified by the pool-qualified
// memory pool block index. Using getBlockId() when recording events is wrong.
// getBlockId() returns the logical block id, which has nothing to do with the raw memory
// block pointers involved in a cudaMemcpy.
//

//
// Notes about need for synchronization:
//
// Relying on decoder syncing GPU with CPU to ensure that blocks are ready
// for offload/onboard/partial copy is dangerous. We have an asynchronous decoder
// that may not synchronize or synchronize at a later point in the execution stream.
// To avoid synchronization issues caused by changes to decoder design we rely on
// KVCacheTransferManager::syncWithBufferManager() that ensures that internal copy streams
// will wait for prefill and decode kernels that have already been scheduled.
//
// Earlier versions of this code did not account for all possible cases where a new block copy
// needed to wait for a previously scheduled copy to finish. For instance, it is possible
// that two primary blocks are offloaded to the same secondary block in a single step,
// scheduling the second offloading without waiting for the first one to finish leads to
// a corrupted block after offloading. It is possible that partial reuse will copy
// from a block that is currently being onboarded, scheduling the partial copy without
// waiting for the onboarding to finish will lead to a corrupted block. To handle all
// possible cases needing synchronization we record separate events for reads and writes
// to a block. When a new block copy is scheduled, we wait for all writes to the source
// block and all reads and writes to a destination block.
//
// As before, syncTransfers() must be called after the last call to KVCacheManager::addSequenceBatch.
// Failing to do so will lead to corrupted blocks eventually.
//

void KVCacheTransferManager::onboard(BlockPtr const& offloadedBlock, BlockPtr const& block,
    std::vector<KVCacheBlockPool> const& pools, int numTokensToCopy, executor::KvCacheTransferMode mode,
    std::string const& directory)
{
    auto const offloadedBlockIndex = getPendingTransferIndex(offloadedBlock);
    auto const blockIndex = getPendingTransferIndex(block);

    // Wait for any pending writes before reading from offloadedBlock
    auto offloadedBlockPendingWriteItr = mPendingWrites.find(offloadedBlockIndex);
    if (offloadedBlockPendingWriteItr != mPendingWrites.end())
    {
        mOnboardManager.getStream().wait(offloadedBlockPendingWriteItr->second);
        // Don't erase, we are not changing state of offloadedBlock
    }
    // Wait for any pending reads before overwriting block
    auto blockPendingReadItr = mPendingReads.find(blockIndex);
    if (blockPendingReadItr != mPendingReads.end())
    {
        mOnboardManager.getStream().wait(blockPendingReadItr->second);
        mPendingReads.erase(blockPendingReadItr);
    }
    // Wait for any pending writes before overwriting block
    auto blockPendingWriteItr = mPendingWrites.find(blockIndex);
    if (blockPendingWriteItr != mPendingWrites.end())
    {
        mOnboardManager.getStream().wait(blockPendingWriteItr->second);
        mPendingWrites.erase(blockPendingWriteItr);
    }

    copyBlock(offloadedBlock, block, pools, false, numTokensToCopy, mode, directory);

    // Update transfer statistics — distinguish host→GPU onboard from GPU→GPU intra-device copy
    {
        std::lock_guard<std::mutex> lock(mStatsMutex);
        auto bytes = computeBlockTransferBytes(pools, numTokensToCopy);
        if (offloadedBlock->isPrimary())
        {
            ++mIntraDeviceCopyBlockCount;
            mIntraDeviceCopyByteCount += bytes;
        }
        else
        {
            ++mOnboardBlockCount;
            mOnboardByteCount += bytes;
        }
    }

    // Record new pending read from offloadedBlock
    mPendingReads[offloadedBlockIndex] = tr::CudaEvent();
    mOnboardManager.getStream().record(mPendingReads[offloadedBlockIndex]);
    // Record new pending write to block
    mPendingWrites[blockIndex] = tr::CudaEvent();
    mOnboardManager.getStream().record(mPendingWrites[blockIndex]);
}

void KVCacheTransferManager::offload(BlockPtr const& block, BlockPtr const& offloadBlock,
    std::vector<KVCacheBlockPool> const& pools, int numTokensToCopy, executor::KvCacheTransferMode mode,
    std::string const& directory)
{
    auto const blockIndex = getPendingTransferIndex(block);
    auto const offloadBlockIndex = getPendingTransferIndex(offloadBlock);

    // Wait for any pending writes before reading from block
    auto blockPendingWriteItr = mPendingWrites.find(blockIndex);
    if (blockPendingWriteItr != mPendingWrites.end())
    {
        mOffloadManager.getStream().wait(blockPendingWriteItr->second);
        // Don't erase, we are not changing state of block
    }
    // Wait for any pending reads before overwriting offloadBlock
    auto offloadBlockPendingReadItr = mPendingReads.find(offloadBlockIndex);
    if (offloadBlockPendingReadItr != mPendingReads.end())
    {
        mOffloadManager.getStream().wait(offloadBlockPendingReadItr->second);
        mPendingReads.erase(offloadBlockPendingReadItr);
    }
    // Wait for any pending writes before overwriting offloadBlock
    auto offloadBlockPendingWriteItr = mPendingWrites.find(offloadBlockIndex);
    if (offloadBlockPendingWriteItr != mPendingWrites.end())
    {
        mOffloadManager.getStream().wait(offloadBlockPendingWriteItr->second);
        mPendingWrites.erase(offloadBlockPendingWriteItr);
    }

    copyBlock(block, offloadBlock, pools, true, numTokensToCopy, mode, directory);

    // Update transfer statistics
    {
        std::lock_guard<std::mutex> lock(mStatsMutex);
        ++mOffloadBlockCount;
        mOffloadByteCount += computeBlockTransferBytes(pools, numTokensToCopy);
    }

    // Record new pending read from block
    mPendingReads[blockIndex] = tr::CudaEvent();
    mOffloadManager.getStream().record(mPendingReads[blockIndex]);
    // Record new pending write to offloadBlock
    mPendingWrites[offloadBlockIndex] = tr::CudaEvent();
    mOffloadManager.getStream().record(mPendingWrites[offloadBlockIndex]);
}

namespace
{
std::string diskSlotFilename(std::string const& directory, SizeType32 diskSlot, size_t poolIdx)
{
    char buf[512];
    std::snprintf(buf, sizeof(buf), "%s/slot_%d_pool_%zu.bin", directory.c_str(), diskSlot, poolIdx);
    return std::string(buf);
}
} // namespace

void KVCacheTransferManager::diskWriterLoop()
{
    while (true)
    {
        DiskWriteJob job;
        {
            std::unique_lock<std::mutex> lock(mDiskMutex);
            mDiskQueueCv.wait(lock, [this] { return !mDiskWriteQueue.empty() || mDiskWriterStop; });
            if (mDiskWriteQueue.empty())
            {
                if (mDiskWriterStop)
                {
                    return;
                }
                continue;
            }
            job = std::move(mDiskWriteQueue.front());
            mDiskWriteQueue.pop();
        }
        mDiskQueueCv.notify_all(); // a producer may be blocked waiting for queue space

        // The slow, writeback-throttle-prone part runs HERE, off the scheduler thread.
        // Unstaged jobs (src != nullptr) read the pinned host slot directly; staged jobs read the copy.
        void const* data = job.src ? job.src : static_cast<void const*>(job.staged.data());
        int fd = ::open(job.filename.c_str(), O_CREAT | O_WRONLY, 0644);
        TLLM_CHECK_WITH_INFO(fd >= 0,
            "[disk-tier] cannot open %s for async write; failing loudly rather than leaving a corrupt slot",
            job.filename.c_str());
        auto const written = ::pwrite(fd, data, job.bytes, 0);
        ::close(fd);
        TLLM_CHECK_WITH_INFO(written == static_cast<ssize_t>(job.bytes),
            "[disk-tier] short async write to %s (%zd/%zu); failing loudly rather than leaving a corrupt slot",
            job.filename.c_str(), static_cast<ssize_t>(written), job.bytes);

        {
            std::lock_guard<std::mutex> lock(mDiskMutex);
            if (auto it = mDiskInflight.find(job.filename); it != mDiskInflight.end() && --it->second <= 0)
            {
                mDiskInflight.erase(it);
            }
            // Unstaged path: when a spill's last pool-write finishes, its source host slot is safe to
            // reuse -> publish the spill id for the block manager to reap.
            if (job.spillId != 0)
            {
                if (auto sit = mSpillRemaining.find(job.spillId); sit != mSpillRemaining.end() && --sit->second <= 0)
                {
                    mSpillRemaining.erase(sit);
                    mCompletedSpills.push_back(job.spillId);
                }
            }
        }
        mDiskQueueCv.notify_all();    // wake a producer waiting to re-write this slot
        mDiskInflightCv.notify_all(); // wake a loader waiting on this slot
    }
}

void KVCacheTransferManager::enqueueDiskWrite(std::string filename, void const* src, std::size_t bytes, bool retained)
{
    DiskWriteJob job;
    job.filename = std::move(filename);
    job.bytes = bytes;
    job.retained = retained;
    job.staged.resize(bytes);
    std::memcpy(job.staged.data(), src, bytes); // fast host->host copy; frees the slot

    {
        std::unique_lock<std::mutex> lock(mDiskMutex);
        // Per-slot serialization (always) + backpressure. A retained spill must land, so it bypasses the
        // queue cap (bounded by retained volume); a best-effort spill also waits for room. Best-effort shedding
        // under saturation happens upstream at the eviction gate, so a best-effort spill reaching here + then
        // finding the queue full is the rare TOCTOU case.
        mDiskQueueCv.wait(lock,
            [this, &job]
            {
                bool const slotClear = mDiskInflight.find(job.filename) == mDiskInflight.end();
                bool const roomOrRetained = job.retained || mDiskWriteQueue.size() < mDiskWriteQueueMax;
                return (slotClear && roomOrRetained) || mDiskWriterStop;
            });
        if (mDiskWriterStop)
        {
            return;
        }
        ++mDiskInflight[job.filename]; // mark BEFORE the block becomes loadable
        mDiskWriteQueue.push(std::move(job));
    }
    mDiskQueueCv.notify_one();
}

void KVCacheTransferManager::enqueueDiskWriteUnstaged(
    std::string filename, void const* src, std::size_t bytes, std::uint64_t spillId, bool retained)
{
    DiskWriteJob job;
    job.filename = std::move(filename);
    job.bytes = bytes;
    job.src = src; // no staging: the writer reads the pinned host slot directly
    job.spillId = spillId;
    job.retained = retained;

    {
        std::unique_lock<std::mutex> lock(mDiskMutex);
        // Same per-slot serialization + backpressure as enqueueDiskWrite: retained spills bypass the queue cap
        // (never stall the scheduler), best-effort spills also wait for room.
        mDiskQueueCv.wait(lock,
            [this, &job]
            {
                bool const slotClear = mDiskInflight.find(job.filename) == mDiskInflight.end();
                bool const roomOrRetained = job.retained || mDiskWriteQueue.size() < mDiskWriteQueueMax;
                return (slotClear && roomOrRetained) || mDiskWriterStop;
            });
        if (mDiskWriterStop)
        {
            return;
        }
        ++mDiskInflight[job.filename]; // mark BEFORE the block becomes loadable
        mDiskWriteQueue.push(std::move(job));
    }
    mDiskQueueCv.notify_one();
}

std::vector<std::uint64_t> KVCacheTransferManager::drainCompletedSpills()
{
    std::lock_guard<std::mutex> lock(mDiskMutex);
    std::vector<std::uint64_t> out;
    out.swap(mCompletedSpills);
    return out;
}

void KVCacheTransferManager::waitForDiskSlotWrites(std::string const& filename)
{
    std::unique_lock<std::mutex> lock(mDiskMutex);
    mDiskInflightCv.wait(lock,
        [this, &filename]
        {
            auto it = mDiskInflight.find(filename);
            return it == mDiskInflight.end() || it->second <= 0;
        });
}

bool KVCacheTransferManager::diskWriteQueueFull()
{
    std::lock_guard<std::mutex> lock(mDiskMutex);
    return mDiskWriteQueue.size() >= mDiskWriteQueueMax;
}

void KVCacheTransferManager::diskReaderLoop()
{
    // Each reader owns its own CUDA stream (H2D copies stay off the null stream and the onboard/offload
    // managers) and its own GDS agent (readers never share one). cudaStreamSynchronize (POSIX) or the agent's
    // internal wait (GDS) makes each transfer device-complete before the block is published, so per-block
    // readiness alone is a sufficient pre-forward gate.
    // Bind this reader thread to the manager's device: a fresh std::thread starts on device 0, so
    // without this the stream and POSIX H2D copies target the wrong GPU on TP ranks != 0 (crash).
    TLLM_CUDA_CHECK(cudaSetDevice(mDeviceId));
    cudaStream_t stream;
    TLLM_CUDA_CHECK(cudaStreamCreate(&stream));
    // Per-reader PINNED staging buffer, grown on demand to one job's total pool bytes (fixed after the first
    // job). Pinned so cudaMemcpyAsync is a real async DMA (~20+ GB/s) instead of the driver's hidden pageable
    // staging copy (~5-8 GB/s, secretly synchronous). pageableFallback is used only if a pinned alloc is refused.
    std::uint8_t* pinnedBuf = nullptr;
    std::size_t pinnedCap = 0;
    std::vector<std::uint8_t> pageableFallback;
    std::shared_ptr<kvc::BaseLoopbackAgent> gdsAgent; // per-reader GDS agent, created lazily on the first GDS job
    while (true)
    {
        DiskReadJob job;
        {
            std::unique_lock<std::mutex> lock(mReadMutex);
            mReadQueueCv.wait(lock, [this] { return !mReadQueue.empty() || mDiskReaderStop; });
            if (mReadQueue.empty())
            {
                if (mDiskReaderStop)
                {
                    break;
                }
                continue;
            }
            job = std::move(mReadQueue.front());
            mReadQueue.pop();
        }

        // A just-spilled slot may still have its write queued; wait it out before reading -- but HERE on the
        // reader, not on the scheduler in loadFromFile. The owning request stays parked via isBlockReadPending
        // until this read lands, so the engine loop is never blocked behind the writer queue.
        if (mAsyncDiskStore)
        {
            for (auto const& f : job.files)
            {
                waitForDiskSlotWrites(f);
            }
        }

        // Perform the block's transfer. GDS issues one batched SSD->GPU DMA for all pools; POSIX reads each
        // pool into a host buffer and copies it up. Both are device-complete before we publish the block.
        if (job.useGds)
        {
            try
            {
                if (gdsAgent == nullptr)
                {
                    kvc::BaseAgentConfig config{std::string("GDSReaderAgent"), true, true};
                    gdsAgent = kvc::makeLoopbackAgent("nixl", &config);
                }
                std::vector<kvc::FileDesc> fileBlobs;
                std::vector<kvc::MemoryDesc> memoryBlobs;
                for (size_t i = 0; i < job.dsts.size(); ++i)
                {
                    fileBlobs.emplace_back(job.files[i], O_RDONLY, 0664, job.bytes[i]);
                    memoryBlobs.emplace_back(job.dsts[i], job.bytes[i], mDeviceId);
                }
                kvc::FileDescs fileDescs(std::move(fileBlobs));
                kvc::MemoryDescs memoryDescs(kvc::MemoryType::kVRAM, memoryBlobs);
                gdsAgent->executeLoopbackRequest(memoryDescs, fileDescs, /*isOffload=*/false);
            }
            catch (std::exception const& e)
            {
                TLLM_LOG_WARNING("disk tier: GDS onboard failed (%s); falling back to POSIX", e.what());
                job.useGds = false; // fall through to POSIX for this job
            }
        }
        if (!job.useGds)
        {
            // Stage the whole block through the pinned buffer, each pool at its OWN offset, so no region is
            // overwritten while its DMA is still reading it (a pinned cudaMemcpyAsync returns before the DMA
            // finishes). One stream sync after the loop makes the buffer safe to reuse for the next job.
            std::size_t total = 0;
            for (auto const b : job.bytes)
            {
                total += b;
            }
            if (total > pinnedCap)
            {
                if (pinnedBuf != nullptr)
                {
                    cudaFreeHost(pinnedBuf);
                    pinnedBuf = nullptr;
                }
                if (cudaHostAlloc(reinterpret_cast<void**>(&pinnedBuf), total, cudaHostAllocDefault) == cudaSuccess)
                {
                    pinnedCap = total;
                }
                else
                {
                    pinnedBuf = nullptr;
                    pinnedCap = 0;
                    TLLM_LOG_WARNING(
                        "[disk-tier] pinned host alloc of %zu B refused; falling back to pageable staging", total);
                }
            }
            std::uint8_t* buf = pinnedBuf;
            if (buf == nullptr)
            {
                pageableFallback.resize(total);
                buf = pageableFallback.data();
            }
            std::size_t off = 0;
            for (size_t i = 0; i < job.dsts.size(); ++i)
            {
                int fd = ::open(job.files[i].c_str(), O_RDONLY);
                TLLM_CHECK_WITH_INFO(fd >= 0,
                    "[disk-tier] cannot open %s for async read; failing loudly rather than serving corrupt KV",
                    job.files[i].c_str());
                auto const got = ::read(fd, buf + off, job.bytes[i]);
                ::close(fd);
                TLLM_CHECK_WITH_INFO(got == static_cast<ssize_t>(job.bytes[i]),
                    "[disk-tier] short async read from %s (%zd/%zu); failing loudly rather than serving corrupt KV",
                    job.files[i].c_str(), static_cast<ssize_t>(got), job.bytes[i]);
                TLLM_CUDA_CHECK(cudaMemcpyAsync(job.dsts[i], buf + off, job.bytes[i], cudaMemcpyHostToDevice, stream));
                off += job.bytes[i];
            }
            TLLM_CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        {
            std::lock_guard<std::mutex> lock(mReadMutex);
            // The transfer is device-complete, so publish the block: any request holding it can be forwarded.
            mPendingBlockReads.erase(job.blockId);
            mReadInflightCount.store(mPendingBlockReads.size(), std::memory_order_release);
        }
    }
    if (pinnedBuf != nullptr)
    {
        cudaFreeHost(pinnedBuf);
    }
    cudaStreamDestroy(stream);
}

void KVCacheTransferManager::enqueueDiskRead(DiskReadJob job)
{
    {
        std::lock_guard<std::mutex> lock(mReadMutex);
        mPendingBlockReads.insert(job.blockId); // mark pending before the job becomes poppable
        mReadInflightCount.store(mPendingBlockReads.size(), std::memory_order_release);
        mReadQueue.push(std::move(job));
    }
    mReadQueueCv.notify_one();
}

bool KVCacheTransferManager::isBlockReadPending(std::int32_t blockId)
{
    std::lock_guard<std::mutex> lock(mReadMutex);
    return mPendingBlockReads.find(blockId) != mPendingBlockReads.end();
}

void KVCacheTransferManager::spillToFile(BlockPtr const& srcHostBlock, SizeType32 diskSlot,
    std::vector<KVCacheBlockPool> const& pools, std::string const& directory)
{
    TLLM_CHECK_WITH_INFO(!directory.empty(), "disk tier requires a directory");
    bool const retained = srcHostBlock->isRetainedNow(); // retained spill must land -> bypasses the queue cap
    // The victim's bytes may still be the target of an in-flight async GPU->host copy;
    // wait it out before reading host memory (same event discipline as copyBlock).
    auto const idx = getPendingTransferIndex(srcHostBlock);
    if (auto it = mPendingWrites.find(idx); it != mPendingWrites.end())
    {
        it->second.synchronize();
    }
    for (size_t poolIdx = 0; poolIdx < pools.size(); ++poolIdx)
    {
        TLLM_CHECK_WITH_INFO(!pools[poolIdx].layerFirstLayout, "disk tier does not support layer-first layout pools");
        auto ptr = computeBlockPointer(srcHostBlock, pools, poolIdx);
        auto const filename = diskSlotFilename(directory, diskSlot, poolIdx);
        auto const bytes = static_cast<std::size_t>(ptr->getSizeInBytes());
        if (!mAsyncDiskStore)
        {
            // Synchronous path -- byte-for-byte the original behavior.
            int fd = ::open(filename.c_str(), O_CREAT | O_WRONLY, 0644);
            TLLM_CHECK_WITH_INFO(fd >= 0, "disk tier: cannot open %s", filename.c_str());
            auto const written = ::pwrite(fd, ptr->data(), bytes, 0);
            ::close(fd);
            TLLM_CHECK_WITH_INFO(
                written == static_cast<ssize_t>(bytes), "disk tier: short write to %s", filename.c_str());
            continue;
        }
        // Async path: copy bytes out (frees the slot immediately) and hand to the writer.
        enqueueDiskWrite(filename, ptr->data(), bytes, retained);
    }
}

void KVCacheTransferManager::spillToFileUnstaged(BlockPtr const& srcHostBlock, SizeType32 diskSlot,
    std::vector<KVCacheBlockPool> const& pools, std::string const& directory, std::uint64_t spillId)
{
    TLLM_CHECK_WITH_INFO(!directory.empty(), "disk tier requires a directory");
    TLLM_CHECK_WITH_INFO(mAsyncDiskStore, "spillToFileUnstaged requires the async writer");
    bool const retained = srcHostBlock->isRetainedNow(); // retained spill must land -> bypasses the queue cap
    // Same event discipline as spillToFile: the victim's bytes may still be the target of an in-flight
    // GPU->host copy; block until it lands before the writer reads host memory.
    auto const idx = getPendingTransferIndex(srcHostBlock);
    if (auto it = mPendingWrites.find(idx); it != mPendingWrites.end())
    {
        it->second.synchronize();
    }
    // Register the whole slot's write count up-front, so an early single-pool completion cannot publish
    // the spill before the remaining pools are enqueued.
    {
        std::lock_guard<std::mutex> lock(mDiskMutex);
        mSpillRemaining[spillId] = static_cast<int>(pools.size());
    }
    for (size_t poolIdx = 0; poolIdx < pools.size(); ++poolIdx)
    {
        TLLM_CHECK_WITH_INFO(!pools[poolIdx].layerFirstLayout, "disk tier does not support layer-first layout pools");
        auto ptr = computeBlockPointer(srcHostBlock, pools, poolIdx);
        auto const filename = diskSlotFilename(directory, diskSlot, poolIdx);
        auto const bytes = static_cast<std::size_t>(ptr->getSizeInBytes());
        // The pool buffer outlives the manager, so this raw pointer stays valid after `ptr` (a view)
        // is destroyed -- the writer reads it later off-thread.
        enqueueDiskWriteUnstaged(filename, ptr->data(), bytes, spillId, retained);
    }
}

void KVCacheTransferManager::loadFromFile(BlockPtr const& dstPrimaryBlock, SizeType32 diskSlot,
    std::vector<KVCacheBlockPool> const& pools, std::string const& directory, std::int32_t trackBlockId)
{
    TLLM_CHECK_WITH_INFO(!directory.empty(), "disk tier requires a directory");
    auto const idx = getPendingTransferIndex(dstPrimaryBlock);
    if (auto it = mPendingWrites.find(idx); it != mPendingWrites.end())
    {
        it->second.synchronize();
    }
    if (auto it = mPendingReads.find(idx); it != mPendingReads.end())
    {
        it->second.synchronize();
    }
    // Async store may still be persisting this slot; a read must not race the write. On the detached path
    // the reader performs this wait itself (see diskReaderLoop), so it never blocks the scheduler; only the
    // synchronous fallback below waits here.
    bool const detach = asyncDiskReadEnabled() && trackBlockId >= 0;
    if (mAsyncDiskStore && !detach)
    {
        for (size_t poolIdx = 0; poolIdx < pools.size(); ++poolIdx)
        {
            waitForDiskSlotWrites(diskSlotFilename(directory, diskSlot, poolIdx));
        }
    }
    // Detached onboard: when a reader pool is present, package all of the block's pools into one tracked job
    // and hand it off. The reader performs the transfer (GDS DMA if enabled, else POSIX read + copy) and makes
    // it device-complete before publishing the block; the request is forward-safe once areBlocksReady() sees
    // its blocks land. trackBlockId < 0 (no pool) takes the synchronous fallback below.
    if (detach)
    {
        DiskReadJob job;
        job.blockId = trackBlockId;
        job.useGds = mDiskUseGds;
        for (size_t poolIdx = 0; poolIdx < pools.size(); ++poolIdx)
        {
            TLLM_CHECK_WITH_INFO(
                !pools[poolIdx].layerFirstLayout, "disk tier does not support layer-first layout pools");
            auto ptr = computeBlockPointer(dstPrimaryBlock, pools, poolIdx);
            job.dsts.push_back(ptr->data());
            job.files.push_back(diskSlotFilename(directory, diskSlot, poolIdx));
            job.bytes.push_back(static_cast<std::size_t>(ptr->getSizeInBytes()));
        }
        enqueueDiskRead(std::move(job));
        return;
    }

    // Synchronous fallback (no reader pool): read inline on the calling thread.
    if (mDiskUseGds)
    {
        // GDS onboard: direct SSD -> GPU VRAM via the NIXL loopback agent (cuFile).
        // Falls back to POSIX staging if the GDS backend can't be created/run.
        try
        {
            if (mLoopbackAgent == nullptr)
            {
                kvc::BaseAgentConfig config{std::string("GDSAgent"), true, true};
                mLoopbackAgent = kvc::makeLoopbackAgent("nixl", &config);
            }
            std::vector<kvc::FileDesc> fileBlobs;
            std::vector<kvc::MemoryDesc> memoryBlobs;
            for (size_t poolIdx = 0; poolIdx < pools.size(); ++poolIdx)
            {
                TLLM_CHECK_WITH_INFO(
                    !pools[poolIdx].layerFirstLayout, "disk tier does not support layer-first layout pools");
                auto ptr = computeBlockPointer(dstPrimaryBlock, pools, poolIdx);
                fileBlobs.emplace_back(
                    diskSlotFilename(directory, diskSlot, poolIdx), O_RDONLY, 0664, ptr->getSizeInBytes());
                memoryBlobs.emplace_back(ptr->data(), ptr->getSizeInBytes(), mDeviceId);
            }
            kvc::FileDescs fileDescs(std::move(fileBlobs));
            kvc::MemoryDescs memoryDescs(kvc::MemoryType::kVRAM, memoryBlobs);
            mLoopbackAgent->executeLoopbackRequest(memoryDescs, fileDescs, /*isOffload=*/false);
            return;
        }
        catch (std::exception const& e)
        {
            TLLM_LOG_WARNING("disk tier: GDS onboard failed (%s); falling back to POSIX", e.what());
        }
    }
    for (size_t poolIdx = 0; poolIdx < pools.size(); ++poolIdx)
    {
        TLLM_CHECK_WITH_INFO(!pools[poolIdx].layerFirstLayout, "disk tier does not support layer-first layout pools");
        auto ptr = computeBlockPointer(dstPrimaryBlock, pools, poolIdx);
        fileToGpuPosix(ptr, diskSlotFilename(directory, diskSlot, poolIdx));
    }
}

void KVCacheTransferManager::syncWithBufferManager()
{
    tr::CudaEvent readyForOffloadEvent;
    mBufferManager.getStream().record(readyForOffloadEvent);
    mOffloadManager.getStream().wait(readyForOffloadEvent);

    tr::CudaEvent readyForOnboardEvent;
    mBufferManager.getStream().record(readyForOnboardEvent);
    mOnboardManager.getStream().wait(readyForOnboardEvent);

    // Once we synchronize, clear our list of pending transfers.
    mPendingReads.clear();
    mPendingWrites.clear();
}

void KVCacheTransferManager::syncTransfers()
{
    tr::CudaEvent offloadEvent;
    mOffloadManager.getStream().record(offloadEvent);
    mBufferManager.getStream().wait(offloadEvent);

    tr::CudaEvent onboardEvent;
    mOnboardManager.getStream().record(onboardEvent);
    mBufferManager.getStream().wait(onboardEvent);

    // Once we synchronize, clear our list of pending transfers.
    mPendingReads.clear();
    mPendingWrites.clear();
}

KvCacheTransferStats KVCacheTransferManager::getAndResetTransferStats()
{
    std::lock_guard<std::mutex> lock(mStatsMutex);
    KvCacheTransferStats stats;
    stats.onboardBlocks = mOnboardBlockCount;
    stats.onboardBytes = mOnboardByteCount;
    stats.offloadBlocks = mOffloadBlockCount;
    stats.offloadBytes = mOffloadByteCount;
    stats.intraDeviceCopyBlocks = mIntraDeviceCopyBlockCount;
    stats.intraDeviceCopyBytes = mIntraDeviceCopyByteCount;
    mOnboardBlockCount = 0;
    mOnboardByteCount = 0;
    mOffloadBlockCount = 0;
    mOffloadByteCount = 0;
    mIntraDeviceCopyBlockCount = 0;
    mIntraDeviceCopyByteCount = 0;
    return stats;
}

std::size_t KVCacheTransferManager::computeBlockTransferBytes(
    std::vector<KVCacheBlockPool> const& pools, int numTokensToCopy) const
{
    std::size_t totalBytes = 0;
    for (auto const& pool : pools)
    {
        if (!pool.primaryPtr || pool.primaryPtr->getSize() == 0)
        {
            continue;
        }

        auto const dataType = pool.primaryPtr->getDataType();
        auto const numElements = static_cast<std::size_t>(pool.primaryPtr->getSize());
        if (numElements == 0)
        {
            continue; // empty pool contributes 0 bytes; avoids divide-by-zero
        }
        auto const bytesPerElement = pool.primaryPtr->getSizeInBytes() / numElements;

        // Mirror the logic in copyBlock: a partial copy only happens when numTokensToCopy > 0,
        // the data type supports it (not kINT4/kFP4), not block scales, and numTokensToCopy < tokensPerBlock.
        bool const isPartialCopy = numTokensToCopy > 0 && dataType != nvinfer1::DataType::kINT4
            && dataType != nvinfer1::DataType::kFP4 && !pool.containsBlockScales
            && numTokensToCopy < pool.tokensPerBlock;

        if (isPartialCopy)
        {
            // Partial copy transfers: numLayers * kvFactor * numKvHeads * sizePerHead * numTokensToCopy elements
            totalBytes += static_cast<std::size_t>(pool.numLayers) * pool.kvFactor * pool.numKvHeads * pool.sizePerHead
                * numTokensToCopy * bytesPerElement;
        }
        else
        {
            // Full block copy: numLayers * kvFactor * blockSize elements
            // where blockSize = numKvHeads * sizePerHead * tokensPerBlock
            totalBytes += static_cast<std::size_t>(pool.numLayers) * pool.kvFactor * pool.blockSize * bytesPerElement;
        }
    }
    return totalBytes;
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager
