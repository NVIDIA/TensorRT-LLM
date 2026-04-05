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

#ifndef TRTLLM_BLOCKPOOL_H
#define TRTLLM_BLOCKPOOL_H

#include "tensorrt_llm/batch_manager/cudaVmmArena.h"

#include <cstddef>
#include <list>
#include <stdexcept>
#include <vector>

namespace tensorrt_llm::batch_manager::state_manager {

/// Metadata for a single block stored in a CudaVmmArena.
///
/// A block represents a contiguous region of `block_size_bytes` bytes starting
/// at byte offset `offset * block_size_bytes` from the arena base pointer.
/// The offset is 0-based and monotonically increasing: block 0 starts at the
/// arena base, block 1 immediately follows it, and so on.
///
/// Each Block carries a reference count that starts at 0. Callers must
/// increment the count before using a block and decrement it when done.
/// BlockPool::shrink() refuses to release any block whose reference count
/// is non-zero.
class Block {
public:
    /// Index of this block within the pool (0-based).
    /// The raw byte offset into the arena is `offset * pool.block_size_bytes()`.
    std::size_t offset;

    explicit Block(std::size_t offset) noexcept
        : offset(offset)
        , refCount_(0) {}

    /// Increment the reference count by 1.
    void incRef() noexcept { ++refCount_; }

    /// Decrement the reference count by 1.
    /// @throws std::underflow_error if the count is already 0.
    void decRef()
    {
        if (refCount_ == 0)
            throw std::underflow_error("Block::decRef(): reference count is already 0.");
        --refCount_;
    }

    /// Return true if the reference count is 0 (block is unreferenced).
    bool isFree() const noexcept { return refCount_ == 0; }

    /// Current reference count (for diagnostics / assertions).
    std::size_t refCount() const noexcept { return refCount_; }

private:
    std::size_t refCount_;
};

/// Manages a pool of fixed-size memory blocks backed by a CudaVmmArena.
///
/// Each block is a multi-dimensional array of elements. The pool grows and
/// shrinks by delegating physical-page management to the underlying arena;
/// Block metadata objects are created and destroyed in lock-step with the
/// committed physical memory.
///
/// Typical usage:
///   CudaVmmArena arena(1ULL << 30, 0);
///   BlockPool pool(&arena, sizeof(float), {32, 64}); // 32×64-float blocks
///   pool.grow(16);                                    // commit 16 blocks
///   float* p = pool.block_ptr<float>(pool.blocks()[0]);
///   pool.shrink(8);                                   // release upper 8
///
/// Thread safety: not thread-safe; external synchronization is required.
class BlockPool {
public:
    /// Construct a BlockPool backed by `arena`.
    ///
    /// @param arena        Non-owning pointer to the arena supplying VA space
    ///                     and physical pages. Must outlive this BlockPool.
    /// @param elementSize  Size in bytes of one element within a block.
    /// @param dimensions   Sizes of each dimension of a block in elements.
    ///                     The total elements per block is the product of all
    ///                     dimension values.
    /// @throws std::invalid_argument if arena is null, elementSize is 0,
    ///         dimensions is empty, or any dimension value is 0.
    explicit BlockPool(vmm::CudaVmmArena*       arena,
                       std::size_t              elementSize,
                       std::vector<std::size_t> dimensions);

    ~BlockPool() = default;

    // Non-copyable, non-movable: tied to a specific arena instance.
    BlockPool(const BlockPool&)            = delete;
    BlockPool& operator=(const BlockPool&) = delete;
    BlockPool(BlockPool&&)                 = delete;
    BlockPool& operator=(BlockPool&&)      = delete;

    // -----------------------------------------------------------------------
    // Resize operations
    // -----------------------------------------------------------------------

    /// Grow the pool to `newNumBlocks` total blocks.
    /// Grows the underlying arena to accommodate the new blocks, appends Block
    /// metadata objects for each newly committed slot, and enqueues them on
    /// the free list.
    /// Throws if newNumBlocks <= block_count(), or if the arena cannot grow.
    void grow(std::size_t newNumBlocks);

    /// Shrink the pool to `newNumBlocks` total blocks.
    /// Removes tail Block metadata objects from the free list, releases the
    /// underlying arena pages, then destroys the metadata.
    /// Throws if newNumBlocks >= block_count(), or if any block that would be
    /// removed has a non-zero reference count.
    void shrink(std::size_t newNumBlocks);

    /// Remove the front block from the free list, increment its reference
    /// count, and return a reference to it.
    /// Throws std::runtime_error if the free list is empty.
    Block& acquireBlock();

    /// Decrement the reference count of `block`. If the count reaches zero,
    /// return the block to the back of the free list.
    /// Throws std::underflow_error (via Block::decRef) if the count is
    /// already 0.
    void releaseBlock(Block& block);

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Total bytes occupied by one block: elementSize * product(dimensions).
    std::size_t blockSizeBytes() const noexcept { return blockSizeBytes_; }

    /// Number of blocks currently committed in the pool.
    std::size_t blockCount() const noexcept { return blocks_.size(); }

    /// Read-only ordered view of all committed Block metadata objects.
    const std::vector<Block>& blocks() const noexcept { return blocks_; }

    /// Number of blocks currently on the free list.
    std::size_t freeCount() const noexcept { return freeBlocks_.size(); }

    /// Raw CUdeviceptr to the first byte of `block` in the arena.
    CUdeviceptr blockPtr(const Block& block) const noexcept {
        return arena_->ptr() + block.offset * blockSizeBytes_;
    }

    /// Typed device pointer to the first element of `block` in the arena.
    template <typename T>
    T* blockPtr(const Block& block) const noexcept {
        return reinterpret_cast<T*>(blockPtr(block));
    }

    /// The arena this pool is backed by.
    vmm::CudaVmmArena* arena() const noexcept { return arena_; }

    /// Size in bytes of a single element.
    std::size_t elementSize() const noexcept { return elementSize_; }

    /// Block dimensions (elements per dimension).
    const std::vector<std::size_t>& dimensions() const noexcept { return dimensions_; }

private:
    vmm::CudaVmmArena*       arena_;
    std::size_t              elementSize_;
    std::vector<std::size_t> dimensions_;
    std::size_t              blockSizeBytes_;
    /// All committed Block objects, in creation order.
    /// Reserved to max capacity at construction so pointers into this vector
    /// remain stable for the lifetime of the pool.
    std::vector<Block>       blocks_;
    /// FIFO free list; each entry is a non-owning pointer into blocks_.
    std::list<Block*>        freeBlocks_;
};

} // namespace tensorrt_llm::batch_manager::state_manager

#endif // TRTLLM_BLOCKPOOL_H
