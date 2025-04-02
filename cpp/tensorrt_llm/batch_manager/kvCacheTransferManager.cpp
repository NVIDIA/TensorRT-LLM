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
#include "tensorrt_llm/kernels/kvCachePartialCopy.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/cudaEvent.h"
#include "tensorrt_llm/runtime/cudaStream.h"

// For GPUDirect Storage (cuFile)
#include <cufile.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <cstring>
#include <sys/types.h>
#include <sys/stat.h>

namespace tr = tensorrt_llm::runtime;
namespace tk = tensorrt_llm::kernels;

namespace tensorrt_llm::batch_manager::kv_cache_manager
{

KVCacheTransferManager::KVCacheTransferManager(tr::BufferManager const& bufferManager)
    : mBufferManager{bufferManager}
    , mOnboardManager(std::make_shared<tr::CudaStream>())
    , mOffloadManager(std::make_shared<tr::CudaStream>())
{
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

void KVCacheTransferManager::copyBlock(
    BlockPtr const& src,
    BlockPtr const& dst,
    std::vector<KVCacheBlockPool> const& pools,
    bool isOffload,
    bool DRAMDestination = true)  // default to true to keep old calls valid
{
    // Indicate which mode was requested
    printf("ENTERED COPY BLOCK: isOffload=%s, DRAMDestination=%s\n",
        isOffload ? "true" : "false",
        DRAMDestination ? "true" : "false");

    // If DRAMDestination = true, use the original GPU->GPU copy logic
    if (DRAMDestination)
    {
        printf("[INFO] DRAMDestination is true; using original GPU-to-GPU copy\n");
        auto const numPools = pools.size();
        for (size_t poolIdx = 0; poolIdx < numPools; poolIdx++)
        {
            auto const srcPtr = computeBlockPointer(src, pools, poolIdx);
            auto const dstPtr = computeBlockPointer(dst, pools, poolIdx);
            (isOffload ? mOffloadManager : mOnboardManager).copy(*srcPtr, *dstPtr);
        }
        // Done, no file I/O when DRAMDestination is used
        printf("[DEBUG] Exiting copyBlock (DRAMDestination path)\n\n");
        return;
    }

    // Otherwise, proceed with GDS or POSIX (offload or onboard)
    auto const numPools = pools.size();
    for (size_t poolIdx = 0; poolIdx < numPools; poolIdx++)
    {
        auto srcPtr = computeBlockPointer(src, pools, poolIdx);
        auto dstPtr = computeBlockPointer(dst, pools, poolIdx);

        // DEBUG: Show pointers and buffer sizes
        printf("[DEBUG]   poolIdx=%zu: srcPtr=%p, dstPtr=%p, getSizeInBytes=%zu\n",
            poolIdx, srcPtr->data(), dstPtr->data(), srcPtr->getSizeInBytes());

        // Build a unique filename for this block
        // Example: /mnt/weka/block_<srcID>_pool_<poolIdx>.bin
        char filename[256];
        std::snprintf(filename, sizeof(filename),
            "/mnt/weka/block_%d_pool_%zu.bin", src->getBlockId(), poolIdx);

        // Open the file for R/W. We create it if offloading, read it if onboarding.
        int openFlags = isOffload ? (O_CREAT | O_WRONLY) : O_RDONLY;
        int fd = ::open(filename, openFlags, 0664);
        if (fd < 0)
        {
            printf("[ERROR] Failed to open '%s' for %s\n", filename, isOffload ? "writing" : "reading");
            continue;
        }

        // If debugNeverGDS is set, skip GDS registration entirely
        if (mDebugNeverGDS)
        {
            printf("[INFO] mDebugNeverGDS=true; forcing POSIX fallback for %s\n", filename);
            // Go directly to the POSIX path
            fallbackPosixIO(srcPtr, dstPtr, fd, filename, isOffload);
            ::close(fd);
            continue;
        }

        // Attempt cuFile registration
        CUfileDescr_t cufileDesc;
        memset(&cufileDesc, 0, sizeof(CUfileDescr_t));
        cufileDesc.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
        cufileDesc.handle.fd = fd;

        CUfileHandle_t cufileHandle;
        CUfileError_t status = cuFileHandleRegister(&cufileHandle, &cufileDesc);

        // Check success. Otherwise, fallback to POSIX.
        if (status.err == CU_FILE_SUCCESS)
        {
            printf("[DEBUG] Using GDS mode for file: %s\n", filename);

            // GDS read/write logic
            ssize_t numBytes = static_cast<ssize_t>(srcPtr->getSizeInBytes());
            if (isOffload)
            {
                // Write GPU data to file
                printf("[DEBUG]   cuFileWrite: writing %zd bytes from GPU to %s\n", numBytes, filename);

                ssize_t bytesWritten = cuFileWrite(cufileHandle, srcPtr->data(), numBytes, 0, 0);
                if (bytesWritten < 0)
                {
                    printf("[ERROR]   cuFileWrite error=%zd\n", bytesWritten);
                }
                else
                {
                    printf("[DEBUG]   Wrote %zd bytes to %s\n", bytesWritten, filename);
                }
            }
            else
            {
                // Read GPU data from file (into dstPtr->data())
                printf("[DEBUG]   cuFileRead: reading %zd bytes from %s into GPU\n", numBytes, filename);

                ssize_t bytesRead = cuFileRead(cufileHandle, dstPtr->data(), numBytes, 0, 0);
                if (bytesRead < 0)
                {
                    printf("[ERROR]   cuFileRead error=%zd\n", bytesRead);
                }
                else
                {
                    printf("[DEBUG]   Read %zd bytes from %s\n", bytesRead, filename);
                }
            }

            // Cleanup GDS handle
            cuFileHandleDeregister(cufileHandle);
        }
        else
        {
            printf("[WARN] cuFileHandleRegister failed (err=%d). Falling back to POSIX for file: %s\n",
                status.err, filename);
            fallbackPosixIO(srcPtr, dstPtr, fd, filename, isOffload);
        }

        ::close(fd);
    }

    // DEBUG: Done with this block
    printf("[DEBUG] Exiting copyBlock: srcId=%d, dstId=%d, isOffload=%s\n\n",
        src->getBlockId(), dst->getBlockId(), isOffload ? "true" : "false");
}

void KVCacheTransferManager::onboard(BlockPtr const& offloadBlock, BlockPtr const& block,
    std::vector<KVCacheBlockPool> const& pools, int numTokensToCopy)
{
    if (mPendingOffloads.find(offloadBlock->getBlockId()) != mPendingOffloads.end())
    {
        mOnboardManager.getStream().wait(mPendingOffloads[offloadBlock->getBlockId()]);
    }
    copyBlock(offloadBlock, block, pools, false, numTokensToCopy);
}

void KVCacheTransferManager::offload(BlockPtr const& block, BlockPtr const& offloadBlock,
    std::vector<KVCacheBlockPool> const& pools, int numTokensToCopy)
{
    mPendingOffloads[block->getBlockId()] = tr::CudaEvent();
    copyBlock(block, offloadBlock, pools, true, numTokensToCopy);
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
