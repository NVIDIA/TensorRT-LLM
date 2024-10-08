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

#pragma once

#include "tensorrt_llm/batch_manager/kvCacheManager.h"

#include <vector>

using namespace tensorrt_llm::batch_manager::kv_cache_manager;

namespace tensorrt_llm::batch_manager::eviction_policy
{

class BaseEvictionPolicy
{
public:
    virtual ~BaseEvictionPolicy() = default;

    virtual void initialize(
        std::vector<BlockPtr>& mAllBlocksById, SizeType32 numPrimaryBlocks, SizeType32 numSecondaryBlocks)
        = 0;

    // Get a free block from the primary memory pool
    virtual BlockPtr getFreePrimaryBlock() = 0;
    // Get a free block from the secondary memory pool
    virtual BlockPtr getFreeSecondaryBlock() = 0;
    // Release a block. Prioritize the block for eviction if toFront=true
    virtual void releaseBlock(BlockPtr block, bool toFront = false) = 0;
    // Get the amount of free blocks in the primary memory pool
    virtual SizeType32 getNumFreePrimaryBlocks() = 0;
    // Get the amount of free blocks in the secondary memory pool
    virtual SizeType32 getNumFreeSecondaryBlocks() = 0;
    // Claim a free block. Called when the cache manager allocates or reuses a new block
    virtual void claimBlock(KVCacheBlock block) = 0;
};

class LRUEvictionPolicy : public BaseEvictionPolicy
{
public:
    void initialize(
        std::vector<BlockPtr>& mAllBlocksById, SizeType32 numPrimaryBlocks, SizeType32 numSecondaryBlocks) override;
    BlockPtr getFreePrimaryBlock() override;
    BlockPtr getFreeSecondaryBlock() override;
    void releaseBlock(BlockPtr block, bool toFront = false) override;
    SizeType32 getNumFreePrimaryBlocks() override;
    SizeType32 getNumFreeSecondaryBlocks() override;

    void claimBlock(KVCacheBlock block);

private:
    FreeBlocksQueue mFreePrimaryBlocks;
    FreeBlocksQueue mFreeSecondaryBlocks;

    std::vector<std::optional<FreeBlocksQueue::iterator>> mFreeBlockIterators;

    SizeType32 mFreePrimaryBlocksSize;
    SizeType32 mFreeSecondaryBlocksSize;
};

} // namespace tensorrt_llm::batch_manager::eviction_policy
