/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    TLLM_CHECK_WITH_INFO(written >= 0, "POSIX write error=%zd", written);

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
    TLLM_CHECK_WITH_INFO(bytesRead >= 0, "POSIX read error=%zd", bytesRead);

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
            auto srcPtr = computeBlockPointer(src, pools, poolIdx);
            auto dstPtr = computeBlockPointer(dst, pools, poolIdx);

            // If no partial tokens or if the dataType is not supported for partial copy, copy entire block.
            if (numTokensToCopy <= 0 || srcPtr->getDataType() == nvinfer1::DataType::kINT4
                || srcPtr->getDataType() == nvinfer1::DataType::kFP4)
            {
                // For partial copy not implemented with these data types,
                // just do a full copy.
                (isOffload ? mOffloadManager : mOnboardManager).copy(*srcPtr, *dstPtr);
            }
            else
            {
                int const tokensPerBlock = pools[poolIdx].tokensPerBlock;
                if (numTokensToCopy >= tokensPerBlock)
                {
                    // If requested tokens >= entire block, just do a full copy.
                    (isOffload ? mOffloadManager : mOnboardManager).copy(*srcPtr, *dstPtr);
                }
                else
                {
                    auto stream = (isOffload ? mOffloadManager : mOnboardManager).getStream().get();
                    int const numLayers = pools[poolIdx].numLayers;
                    int const kvFactor = pools[poolIdx].kvFactor;
                    int const numHeads = pools[poolIdx].numKvHeads;
                    int const sizePerHead = pools[poolIdx].sizePerHead;
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

void KVCacheTransferManager::onboard(BlockPtr const& offloadBlock, BlockPtr const& block,
    std::vector<KVCacheBlockPool> const& pools, int numTokensToCopy, executor::KvCacheTransferMode mode,
    std::string const& directory)
{
    if (mode != executor::KvCacheTransferMode::DRAM
        && mPendingOffloads.find(offloadBlock->getBlockId()) == mPendingOffloads.end())
    {
        TLLM_LOG_DEBUG("Skipping onboard for block %d because it was never previously offloaded to disk",
            offloadBlock->getBlockId());
        return;
    }

    if (mPendingOffloads.find(offloadBlock->getBlockId()) != mPendingOffloads.end())
    {
        mOnboardManager.getStream().wait(mPendingOffloads[offloadBlock->getBlockId()]);
    }
    copyBlock(offloadBlock, block, pools, false, numTokensToCopy, mode, directory);
}

void KVCacheTransferManager::offload(BlockPtr const& block, BlockPtr const& offloadBlock,
    std::vector<KVCacheBlockPool> const& pools, int numTokensToCopy, executor::KvCacheTransferMode mode,
    std::string const& directory)
{
    mPendingOffloads[block->getBlockId()] = tr::CudaEvent();
    copyBlock(block, offloadBlock, pools, true, numTokensToCopy, mode, directory);
    mOffloadManager.getStream().record(mPendingOffloads[block->getBlockId()]);
}

void KVCacheTransferManager::syncTransfers()
{
    tr::CudaEvent offloadEvent;
    mOffloadManager.getStream().record(offloadEvent);

    tr::CudaEvent onboardEvent;
    mOnboardManager.getStream().record(onboardEvent);

    mBufferManager.getStream().wait(offloadEvent);
    mBufferManager.getStream().wait(onboardEvent);

    // Once we synchronize, clear our list of pending thransfers.
    mPendingOffloads.clear();
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager
