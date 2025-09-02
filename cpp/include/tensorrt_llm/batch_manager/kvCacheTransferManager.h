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
#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/cudaEvent.h"

namespace tr = tensorrt_llm::runtime;
namespace kvc = tensorrt_llm::executor::kv_cache;

#pragma once

namespace tensorrt_llm::batch_manager::kv_cache_manager
{

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

    //! \brief Synchronize the offload/onboard streams with the bufferManager stream.
    void syncTransfers();

private:
    //! \brief Get pointer to pool specified by cache block.
    static tr::ITensor::SharedPtr computeBlockPointer(
        BlockPtr const& block, std::vector<KVCacheBlockPool> const& pools, size_t poolIdx);

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

    runtime::BufferManager mBufferManager;
    runtime::BufferManager mOnboardManager;
    runtime::BufferManager mOffloadManager;

    // Track the block ids offloaded in this iteration.
    std::unordered_map<int32_t, tr::CudaEvent> mPendingOffloads;
    // Reference to parent loopback agent
    std::shared_ptr<kvc::BaseLoopbackAgent> mLoopbackAgent;
    int mDeviceId;
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager
