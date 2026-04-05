/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/batch_manager/blockPool.h"

#include <numeric>
#include <stdexcept>
#include <string>

namespace tensorrt_llm::batch_manager::state_manager {

BlockPool::BlockPool(vmm::CudaVmmArena*       arena,
                     std::size_t              elementSize,
                     std::vector<std::size_t> dimensions)
    : arena_(arena)
    , elementSize_(elementSize)
    , dimensions_(std::move(dimensions))
    , blockSizeBytes_(0)
{
    if (!arena_)
        throw std::invalid_argument("BlockPool: arena must not be null.");
    if (elementSize_ == 0)
        throw std::invalid_argument("BlockPool: elementSize must be > 0.");
    if (dimensions_.empty())
        throw std::invalid_argument("BlockPool: dimensions must not be empty.");
    for (std::size_t d : dimensions_)
        if (d == 0)
            throw std::invalid_argument("BlockPool: every dimension value must be > 0.");

    blockSizeBytes_ = elementSize_
        * std::accumulate(dimensions_.begin(), dimensions_.end(),
                          std::size_t{1}, std::multiplies<std::size_t>{});

    // Reserve the full maximum capacity upfront so that pointers into blocks_
    // remain stable through all future grow() calls.
    blocks_.reserve(arena_->max_size() / blockSizeBytes_);
}

void BlockPool::grow(std::size_t newNumBlocks)
{
    if (newNumBlocks <= blocks_.size())
        throw std::invalid_argument(
            "BlockPool::grow(): newNumBlocks must be greater than the current block_count.");

    // Grow the arena first; this may throw if the arena's max_size is exceeded.
    arena_->grow(newNumBlocks * blockSizeBytes_);

    // Append metadata for each newly committed block and enqueue on free list.
    // No reallocation occurs because the vector was reserved to max capacity.
    for (std::size_t i = blocks_.size(); i < newNumBlocks; ++i)
    {
        blocks_.emplace_back(i);
        freeBlocks_.push_back(&blocks_.back());
    }
}

void BlockPool::shrink(std::size_t newNumBlocks)
{
    if (newNumBlocks >= blocks_.size())
        throw std::invalid_argument(
            "BlockPool::shrink(): newNumBlocks must be less than the current block_count.");

    // Refuse to release any block that is still referenced.
    for (std::size_t i = newNumBlocks; i < blocks_.size(); ++i)
    {
        if (!blocks_[i].isFree())
            throw std::runtime_error(
                "BlockPool::shrink(): block " + std::to_string(blocks_[i].offset)
                + " has non-zero reference count (" + std::to_string(blocks_[i].refCount()) + ").");
    }

    // Remove tail block pointers from the free list before destroying metadata.
    Block* tailBegin = &blocks_[newNumBlocks];
    Block* tailEnd   = &blocks_[blocks_.size()];
    freeBlocks_.remove_if([tailBegin, tailEnd](Block* p)
    {
        return p >= tailBegin && p < tailEnd;
    });

    // Destroy tail Block metadata.
    blocks_.erase(blocks_.begin() + static_cast<std::ptrdiff_t>(newNumBlocks), blocks_.end());

    // Shrink the arena; releases physical pages backing the removed blocks.
    arena_->shrink(newNumBlocks * blockSizeBytes_);
}

Block& BlockPool::acquireBlock()
{
    if (freeBlocks_.empty())
        throw std::runtime_error("BlockPool::acquireBlock(): no free blocks available.");

    Block* block = freeBlocks_.front();
    freeBlocks_.pop_front();
    block->incRef();
    return *block;
}

void BlockPool::releaseBlock(Block& block)
{
    block.decRef();
    if (block.isFree())
        freeBlocks_.push_back(&block);
}

} // namespace tensorrt_llm::batch_manager::state_manager
