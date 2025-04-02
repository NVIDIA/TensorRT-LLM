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
    int numTokensToCopy,
    bool DRAMDestination = true, // default to true to keep old calls valid
    bool debugNeverGDS = false)  // Debugging for when cufile API is unavailable (for DMABUF to be unlocked: requires CUDA on OpenRM
                                 // also need nvidia_fs, and filesystem supporting GDS)
{
    // Indicate which mode was requested
    printf("ENTERED COPY BLOCK: isOffload=%s, DRAMDestination=%s\n",
        isOffload ? "true" : "false",
        DRAMDestination ? "true" : "false");

    // If DRAMDestination = true, use the original CPU->GPU (or GPU->CPU) copy logic,
    // but now with partial-copy functionality from the newer commit.
    if (DRAMDestination)
    {
        printf("[INFO] DRAMDestination is true; using original GPU-to-CPU copy\n");

        // TODO: Replace computeBlockPointer with getKOrVBlockPointer calls
        // block spans multiple pool - copy in each pool
        auto const numPools = pools.size();
        for (size_t poolIdx = 0; poolIdx < numPools; poolIdx++)
        {
            auto const srcPtr = computeBlockPointer(src, pools, poolIdx);
            auto const dstPtr = computeBlockPointer(dst, pools, poolIdx);

            // The partial-copy logic from the newer commit:
            if (numTokensToCopy <= 0 
                || srcPtr->getDataType() == nvinfer1::DataType::kINT4
                || srcPtr->getDataType() == nvinfer1::DataType::kFP4)
            {
                // numTokensToCopy <= 0 indicates entire block should be copied.
                // Partial copy has not been implemented yet for data types INT4 and FP4
                (isOffload ? mOffloadManager : mOnboardManager).copy(*srcPtr, *dstPtr);
            }
            else
            {
                int const tokensPerBlock = pools[poolIdx].tokensPerBlock;
                if (numTokensToCopy >= tokensPerBlock)
                {
                    (isOffload ? mOffloadManager : mOnboardManager).copy(*srcPtr, *dstPtr);
                }
                else
                {
                    auto stream = (isOffload ? mOffloadManager : mOnboardManager).getStream().get();
                    int const numLayers   = pools[poolIdx].numLayers;
                    int const numHeads    = pools[poolIdx].numKvHeads;
                    int const sizePerHead = pools[poolIdx].sizePerHead;
                    auto shape = srcPtr->getShape();

                    TLLM_LOG_DEBUG("block.Shape = %s", srcPtr->toString(shape).c_str());
                    TLLM_CHECK_WITH_INFO(
                        shape.nbDims == 4, 
                        "Expected KVCache block to have 4 dimensions, but it has %d", 
                        shape.nbDims);
                    TLLM_CHECK_WITH_INFO(
                        (shape.d[0] == 1) && (shape.d[1] == numLayers) && (shape.d[2] == 2)
                          && (shape.d[3] == numHeads * tokensPerBlock * sizePerHead),
                        "Block shape is incorrect");
                    TLLM_CHECK_WITH_INFO(
                        numTokensToCopy <= tokensPerBlock,
                        "numTokensToCopy (%d) must be <= tokensPerBlock (%d)", 
                        numTokensToCopy, tokensPerBlock);

                    // Invoke partial copy
                    tk::kvCacheBlockPartialCopy(
                        *dstPtr, *srcPtr, 
                        numLayers, numHeads, tokensPerBlock, sizePerHead, 
                        numTokensToCopy, 
                        stream);
                }
            }
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

        // Open the file for R/W. We create it if offloading, read if onboarding.
        int openFlags = isOffload ? (O_CREAT | O_WRONLY) : O_RDONLY;
        int fd = ::open(filename, openFlags, 0664);
        if (fd < 0)
        {
            printf("[ERROR] Failed to open '%s' for %s\n",
                   filename, isOffload ? "writing" : "reading");
            continue;
        }

        // If debugNeverGDS is set, skip GDS registration and go straight to POSIX
        if (debugNeverGDS)
        {
            printf("[INFO] mDebugNeverGDS=true; forcing POSIX fallback for %s\n", filename);

            // Inline POSIX fallback logic
            ssize_t numBytes = static_cast<ssize_t>(srcPtr->getSizeInBytes());
            void* hostBuffer = std::malloc(numBytes);
            if (!hostBuffer)
            {
                printf("[ERROR] Host memory allocation failed for POSIX fallback\n");
                ::close(fd);
                continue;
            }

            if (isOffload)
            {
                // GPU -> host -> file
                printf("[DEBUG] Using POSIX write: writing %zd bytes from GPU to %s\n", numBytes, filename);
                cudaMemcpy(hostBuffer, srcPtr->data(), numBytes, cudaMemcpyDeviceToHost);

                ssize_t bytesWritten = ::write(fd, hostBuffer, numBytes);
                if (bytesWritten < 0)
                {
                    printf("[ERROR]   POSIX write error=%zd\n", bytesWritten);
                }
                else
                {
                    printf("[DEBUG]   Wrote %zd bytes to %s (POSIX fallback)\n", bytesWritten, filename);
                }
            }
            else
            {
                // file -> host -> GPU
                printf("[DEBUG] Using POSIX read: reading %zd bytes from %s into GPU\n", numBytes, filename);
                ssize_t bytesRead = ::read(fd, hostBuffer, numBytes);
                if (bytesRead < 0)
                {
                    printf("[ERROR]   POSIX read error=%zd\n", bytesRead);
                }
                else
                {
                    printf("[DEBUG]   Read %zd bytes from %s (POSIX fallback)\n", bytesRead, filename);
                    cudaMemcpy(dstPtr->data(), hostBuffer, numBytes, cudaMemcpyHostToDevice);
                }
            }
            std::free(hostBuffer);

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

        // If registration fails, fallback to inline POSIX I/O
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

            // Inline POSIX fallback logic
            ssize_t numBytes = static_cast<ssize_t>(srcPtr->getSizeInBytes());
            void* hostBuffer = std::malloc(numBytes);
            if (!hostBuffer)
            {
                printf("[ERROR] Host memory allocation failed for POSIX fallback\n");
                ::close(fd);
                continue;
            }

            if (isOffload)
            {
                // GPU -> host -> file
                printf("[DEBUG] Using POSIX write: writing %zd bytes from GPU to %s\n", numBytes, filename);
                cudaMemcpy(hostBuffer, srcPtr->data(), numBytes, cudaMemcpyDeviceToHost);

                ssize_t bytesWritten = ::write(fd, hostBuffer, numBytes);
                if (bytesWritten < 0)
                {
                    printf("[ERROR]   POSIX write error=%zd\n", bytesWritten);
                }
                else
                {
                    printf("[DEBUG]   Wrote %zd bytes to %s (POSIX fallback)\n", bytesWritten, filename);
                }
            }
            else
            {
                // file -> host -> GPU
                printf("[DEBUG] Using POSIX read: reading %zd bytes from %s into GPU\n", numBytes, filename);
                ssize_t bytesRead = ::read(fd, hostBuffer, numBytes);
                if (bytesRead < 0)
                {
                    printf("[ERROR]   POSIX read error=%zd\n", bytesRead);
                }
                else
                {
                    printf("[DEBUG]   Read %zd bytes from %s (POSIX fallback)\n", bytesRead, filename);
                    cudaMemcpy(dstPtr->data(), hostBuffer, numBytes, cudaMemcpyHostToDevice);
                }
            }
            std::free(hostBuffer);
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
