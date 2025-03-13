/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/cudaEvent.h"

namespace tr = tensorrt_llm::runtime;

#pragma once

namespace tensorrt_llm::batch_manager::kv_cache_manager
{

// The TransferManager accelerates transfers to/from the GPU by overlapping HtoD and DtoH transfers, and tracks ongoing
// transfers in order to avoid race conditions. It is functionally equivalent to the prior approach of putting all
// transfers into the forward pass stream. This is only ever used as a component of a KVCacheManager.
class KVCacheTransferManager
{
public:
    explicit KVCacheTransferManager(tr::BufferManager const& bufferManager);

    //! \brief Onboard a block to gpu memory.
    void onboard(BlockPtr const& offloadBlock, BlockPtr const& block, std::vector<KVCacheBlockPool> const& pools,
        int numTokensToCopy = 0);

    //! \brief Offload a block to cpu memory.
    void offload(BlockPtr const& block, BlockPtr const& offloadBlock, std::vector<KVCacheBlockPool> const& pools,
        int numTokensToCopy = 0);

    //! \brief Synchronize the offload/onboard streams with the bufferManager stream.
    void syncTransfers();

private:
    //! \brief Get pointer to pool specified by cache block.
    static tr::ITensor::SharedPtr computeBlockPointer(
        BlockPtr const& block, std::vector<KVCacheBlockPool> const& pools, size_t poolIdx);

    //! \brief Copy content of src block to dst.
    void copyBlock(BlockPtr const& src, BlockPtr const& dst, std::vector<KVCacheBlockPool> const& pools, bool isOffload,
        int numTokensToCopy = 0);

    runtime::BufferManager mBufferManager;
    runtime::BufferManager mOnboardManager;
    runtime::BufferManager mOffloadManager;

    // Track the block ids offloaded in this iteration.
    std::unordered_map<int32_t, tr::CudaEvent> mPendingOffloads;
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager
