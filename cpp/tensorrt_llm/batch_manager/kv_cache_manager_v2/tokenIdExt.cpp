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

#include "kv_cache_manager_v2/tokenIdExt.h"

#include "kv_cache_manager_v2/utils/math.h" // DynamicBitset
#include "tensorrt_llm/common/assert.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <mutex>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

// ---------------------------------------------------------------------------
// DigestPool — process-global, address-stable store of 32-byte multi-modal
// Digests, referenced by a 31-bit slot index packed into a TokenIdExt. It is a
// pure implementation detail of TokenIdExt, so it lives here (anonymous
// namespace) rather than in the header.
//
// Digests are rare (multi-modal only), so all access is guarded by a single
// mutex; the all-normal-token hashing fast path never touches the pool. Storage
// is a std::deque (element references stay valid across push_back — unlike a
// reallocating std::vector). Occupancy is a DynamicBitset; alloc() takes the
// LOWEST free slot (front-packing via a rolling minFreeHint) so live digests
// cluster at the front and free() can pop trailing free slots off the deque
// (never remapping a live index → outstanding borrows stay valid). The index is
// unobservable (equality/hashing use the 32 bytes, never the index), so a single
// static singleton is safe.
// ---------------------------------------------------------------------------
namespace
{

class DigestPool
{
public:
    static DigestPool& instance()
    {
        static DigestPool pool;
        return pool;
    }

    ~DigestPool()
    {
        // Runs at process exit (single-threaded → no lock). Every digest
        // TokenIdExt should have freed its slot by now; a nonzero count means one
        // outlived the pool — a lifetime bug. Use fprintf, not the logger, whose
        // lifetime at static destruction is not guaranteed.
        if (size_t const inUse = mInUse.numSetBits(); inUse != 0)
        {
            (void) std::fprintf(stderr, "[ERROR] DigestPool destroyed with %zu digest slot(s) still in use\n", inUse);
        }
    }

    DigestPool(DigestPool const&) = delete;
    DigestPool& operator=(DigestPool const&) = delete;
    DigestPool(DigestPool&&) = delete;
    DigestPool& operator=(DigestPool&&) = delete;

    // Store a copy of `digest` in the lowest free slot; return its index.
    uint32_t alloc(Digest const& digest)
    {
        std::lock_guard<std::mutex> const lock(mMutex);
        return allocLocked(digest);
    }

    // Duplicate the digest at slot `idx` into a fresh slot; return the new index.
    uint32_t duplicate(uint32_t idx)
    {
        std::lock_guard<std::mutex> const lock(mMutex);
        return allocLocked(mStore[idx]); // Safe for deque
    }

    // The digest at slot `idx`. The reference stays valid after the lock is
    // released and across later shrinks (which only pop free tail slots). Uses
    // at() so a bad index (e.g. the sentinel of a moved-from handle) throws
    // rather than reading out of bounds; digests are rare so the check is cheap.
    [[nodiscard]] Digest const& get(uint32_t idx) const
    {
        std::lock_guard<std::mutex> const lock(mMutex);
        return mStore.at(idx);
    }

    // Clear slot `idx` and reclaim trailing free slots.
    void free(uint32_t idx)
    {
        if (idx == TokenIdExt::kValueMask)
        {
            return; // the default / moved-from sentinel index — nothing to free
        }
        std::lock_guard<std::mutex> const lock(mMutex);
        mInUse.clear(idx);
        if (idx < mMinFreeHint)
        {
            mMinFreeHint = idx; // a lower slot is now free
        }
        shrinkTailLocked();
    }

    [[nodiscard]] size_t liveCount() const
    {
        std::lock_guard<std::mutex> const lock(mMutex);
        return mInUse.numSetBits();
    }

private:
    DigestPool() = default;

    // Slot count. Also checks the occupancy bitset stays sized to the store.
    // Precondition: caller holds mMutex and mStore/mInUse are in sync (i.e. not
    // called between growing/shrinking one and resizing the other).
    [[nodiscard]] size_t capacity() const
    {
        TLLM_CHECK_DEBUG(mInUse.size() == mStore.size());
        return mStore.size();
    }

    // Precondition: caller holds mMutex. Store `digest` in the lowest free slot
    // (front-packing), growing the deque only when no free slot exists.
    uint32_t allocLocked(Digest const& digest)
    {
        size_t const cap = capacity();
        size_t idx = mMinFreeHint;
        while (idx < cap && mInUse.get(idx))
        {
            ++idx;
        }
        if (idx == cap)
        {
            // No free slot below the high-water mark — grow by one. Indices stay
            // strictly below kValueMask, which is reserved as the bad-handle sentinel.
            TLLM_CHECK_WITH_INFO(cap < TokenIdExt::kValueMask, "DigestPool exhausted the 31-bit index space");
            mStore.push_back(digest);
            mInUse.resize(mStore.size()); // re-sync the bitset; new bit is clear
        }
        else
        {
            mStore[idx] = digest;
        }
        mInUse.set(idx);
        mMinFreeHint = idx + 1; // everything below is now occupied
        return static_cast<uint32_t>(idx);
    }

    // Precondition: caller holds mMutex. Pop free tail slots off the deque when
    // the trailing-free run reaches kSlackHigh, leaving ~kSlackLow of buffer.
    void shrinkTailLocked()
    {
        size_t const cap = capacity();
        // Highest in-use index + 1 (scan down over the trailing free run only).
        size_t liveEnd = cap;
        while (liveEnd > 0 && !mInUse.get(liveEnd - 1))
        {
            --liveEnd;
        }
        size_t const trailingFree = cap - liveEnd;
        if (trailingFree < kSlackHigh)
        {
            return;
        }
        size_t const newCapacity = liveEnd + kSlackLow; // leave a small buffer
        // Only free tail slots are dropped, so no live index is remapped.
        mStore.resize(newCapacity);
        mInUse.resize(newCapacity);
        if (mMinFreeHint > liveEnd)
        {
            mMinFreeHint = liveEnd; // lowest free slot is now at the reclaimed tail
        }
    }

    // Absolute slack (in slots), NOT a ratio: a deque grows/shrinks one fixed
    // chunk at a time, so we only damp boundary-block churn.
    static constexpr size_t kSlackHigh = 256;
    static constexpr size_t kSlackLow = 64;

    mutable std::mutex mMutex;
    std::deque<Digest> mStore; // slot storage; mStore.size() == the slot count (== bitset capacity)
    DynamicBitset mInUse{0};   // bit i set == slot i occupied
    size_t mMinFreeHint{0};    // lower bound on the lowest free slot index
};

} // namespace

// ---------------------------------------------------------------------------
// TokenIdExt — RAII members that touch the pool (construct/copy=alloc, dtor=free).
// ---------------------------------------------------------------------------

TokenIdExt::TokenIdExt(Digest const& digestValue)
    : mBits(DigestPool::instance().alloc(digestValue) | kTagMask)
{
}

void TokenIdExt::freeSlot(uint32_t index) noexcept
{
    DigestPool::instance().free(index);
}

uint32_t TokenIdExt::duplicateSlot(uint32_t index)
{
    return DigestPool::instance().duplicate(index);
}

Digest const& TokenIdExt::digest() const
{
    return DigestPool::instance().get(digestIndex());
}

namespace detail
{

size_t digestPoolLiveCount()
{
    return DigestPool::instance().liveCount();
}

} // namespace detail

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
