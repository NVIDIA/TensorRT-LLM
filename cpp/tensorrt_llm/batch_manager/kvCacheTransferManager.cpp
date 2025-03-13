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

void KVCacheTransferManager::copyBlock(BlockPtr const& src, BlockPtr const& dst,
    std::vector<KVCacheBlockPool> const& pools, bool isOffload, int numTokensToCopy)
{
    // TODO: Replace computeBlockPointer with getKOrVBlockPointer calls
    // block spans multiple pool - copy in each pool
    auto const numPools = pools.size();
    for (size_t poolIdx = 0; poolIdx < numPools; poolIdx++)
    {
        auto const srcPtr = computeBlockPointer(src, pools, poolIdx);
        auto dstPtr = computeBlockPointer(dst, pools, poolIdx);
        if (numTokensToCopy <= 0 || srcPtr->getDataType() == nvinfer1::DataType::kINT4
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
                int const numLayers = pools[poolIdx].numLayers;
                int const numHeads = pools[poolIdx].numKvHeads;
                int const sizePerHead = pools[poolIdx].sizePerHead;
                auto shape = srcPtr->getShape();
                TLLM_LOG_DEBUG("block.Shape = %s", srcPtr->toString(shape).c_str());
                TLLM_CHECK_WITH_INFO(
                    shape.nbDims == 4, "Expected KVCache block to have 4 dimensions, but it has %d", shape.nbDims);
                TLLM_CHECK_WITH_INFO((shape.d[0] == 1) && (shape.d[1] == numLayers) && (shape.d[2] == 2)
                        && (shape.d[3] == numHeads * tokensPerBlock * sizePerHead),
                    "Block shape is incorrect");
                TLLM_CHECK_WITH_INFO(numTokensToCopy <= tokensPerBlock,
                    "numTokensToCopy (%d) must be <= tokensPerBlock (%d)", numTokensToCopy, tokensPerBlock);
                tk::kvCacheBlockPartialCopy(
                    *dstPtr, *srcPtr, numLayers, numHeads, tokensPerBlock, sizePerHead, numTokensToCopy, stream);
            }
        }
    }
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
