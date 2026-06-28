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

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tensorrt_llm::executor::kv_cache::bounce
{

// ============================================================================
// BuddyAllocator — power-of-two buddy allocator over a logical byte space (DESIGN)
// ----------------------------------------------------------------------------
// Pure logic, no GPU / threads / IO — fully unit-testable. It is the data-region allocator for the
// bounce v2 arena (one shared registered buffer): instead of fixed full slots, each chunk gets a
// region sized to its actual bytes, so MANY small transfers fit (high concurrency, no waste) while
// a transfer LARGER than the whole buffer still streams through (its chunks are each ≤ a modest
// maxChunkBytes and recycled per ACK via the credit window — see DESIGN.md §5/§9).
//
// alloc(bytes) rounds up to a power-of-two multiple of minBlock and returns the byte offset of a
// free block of that order (splitting a larger one if needed). free(offset) coalesces with the
// buddy block whenever it is also free. Properties:
//   - NO external fragmentation: blocks only ever merge with their power-of-two buddy.
//   - Internal fragmentation ≤ 2x (a request rounds up to the next power-of-two block).
//   - alloc returns nullopt when no block of the needed order is free (caller applies backpressure;
//     a large alloc may transiently fail under fragmentation but a later free+coalesce frees a
//     high-order block — no deadlock, since frees come from independently-completing flows).
//
// The allocator manages OFFSETS in [0, capacity); the actual GPU buffer (base ptr) lives elsewhere
// (BounceArena); CreditScheduler wraps this allocator to hand out region offsets over that buffer.
//
// SIZING
//   capacity is rounded DOWN to the largest minBlock * 2^L that fits (the remainder is unused);
//   minBlock is rounded UP to a power of two. So capacity()/minBlock() may differ from the ctor args
//   — always query them, never assume the raw inputs.
//
// INTERNALS
//   Block sizes are exactly minBlock<<order, order in [0, maxOrder]; a block of size S always starts
//   at an offset that is a multiple of S (the buddy alignment invariant). State:
//     mFree[order] : set of free block offsets at that order (mFree[maxOrder] = {0} initially).
//     mAllocOrder  : offset -> order for every live block (so free() knows its size to coalesce).
//   alloc(bytes): want = the order of the smallest power-of-two block >= max(bytes, minBlock); find
//     the LOWEST order >= want that has a free block; SPLIT down to `want`, pushing the upper half of
//     each split back to mFree[order-1] and descending into the lower half; return the block offset.
//   free(offset): look up its order; while the buddy (offset XOR (minBlock<<order)) is also free at
//     the same order, remove it and MERGE (merged block starts at min(offset,buddy), order+1); insert
//     the result. An offset not in mAllocOrder (double-free / never-allocated) is ignored (robust).
//
//   Example (minBlock=256, capacity=1024 -> orders 0..2; mFree shown as {order:[offsets]}):
//     start        mFree={2:[0]}                          one 1024-block at offset 0
//     alloc(200)   split 1024 -> upper 512@512, 256@256;  return 0   mFree={0:[256], 1:[512]}
//     alloc(200)   take the order-0 block at 256;         return 256 mFree={1:[512]}
//     free(0)      buddy(0)=256 is LIVE -> no merge                  mFree={0:[0], 1:[512]}
//     free(256)    buddy(256)=0 free -> merge to 512@0; buddy=512 free -> merge to 1024@0  mFree={2:[0]}
// ============================================================================
class BuddyAllocator
{
public:
    /// @param capacity  total bytes (rounded DOWN to the largest minBlock * 2^L that fits).
    /// @param minBlock  smallest allocatable block (rounded UP to a power of two); the order-0 size.
    BuddyAllocator(std::size_t capacity, std::size_t minBlock);

    /// Allocate at least `bytes` (>0). Returns the block's OFFSET only (not its size), or nullopt if
    /// no free block of the required order exists right now.
    ///
    /// Why offset-only is sufficient: free() is offset-keyed (the order/size is looked up internally
    /// via mAllocOrder), so the caller never needs to remember the block size to release it. And the
    /// caller must use its OWN requested length for the actual transfer — NOT the rounded-up block
    /// size: the slack between `bytes` and minBlock<<order is internal fragmentation that is unmapped
    /// to no real data, so writing/reading it would move garbage. In bounce, CreditScheduler carries
    /// the requested chunk bytes as Grant.len and the sender RDMA-writes exactly that many bytes.
    /// (Use blockBytes() if you ever genuinely need the rounded-up size, e.g. for metrics.)
    [[nodiscard]] std::optional<std::uint64_t> alloc(std::size_t bytes);

    /// Free a block previously returned by alloc(). Coalesces with its buddy when possible.
    void free(std::uint64_t offset);

    /// The actual (rounded-up, power-of-two) size of the live block at `offset`, or 0 if `offset` is
    /// not a live allocation. For metrics/inspection — normal callers use their own requested length.
    [[nodiscard]] std::size_t blockBytes(std::uint64_t offset) const noexcept;

    [[nodiscard]] std::size_t capacity() const noexcept
    {
        return mUsable;
    }

    [[nodiscard]] std::size_t minBlock() const noexcept
    {
        return mMinBlock;
    }

    /// Sum of all free block sizes (bytes).
    [[nodiscard]] std::size_t freeBytes() const noexcept;

    /// Largest single alloc that can succeed right now (0 if none) — i.e. minBlock << (highest
    /// order with a free block). Lets the caller cap maxChunkBytes / detect "won't fit".
    [[nodiscard]] std::size_t maxAllocBytes() const noexcept;

    /// Number of currently-allocated blocks (for tests / metrics / leak checks).
    [[nodiscard]] std::size_t liveBlocks() const noexcept
    {
        return mAllocOrder.size();
    }

private:
    static std::size_t roundUpPow2(std::size_t v);
    [[nodiscard]] std::size_t orderForBytes(std::size_t bytes) const; // smallest order fitting bytes

    std::size_t mMinBlock{};                                          // order-0 block size (power of two)
    std::size_t mUsable{};                                            // minBlock << mMaxOrder
    std::uint32_t mMaxOrder{};                                    // top order (one block of this size = whole arena)
    std::vector<std::unordered_set<std::uint64_t>> mFree;         // mFree[order] = free block offsets at that order
    std::unordered_map<std::uint64_t, std::uint32_t> mAllocOrder; // allocated offset -> order
};

} // namespace tensorrt_llm::executor::kv_cache::bounce
