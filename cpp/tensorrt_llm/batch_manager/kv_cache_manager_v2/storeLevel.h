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

#include "kv_cache_manager_v2/common.h"
#include "kv_cache_manager_v2/storeLevelIo.h"
#include "kv_cache_manager_v2/utils/cudaEvent.h"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cuda.h>
#include <deque>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

namespace detail
{
//! State the store level's host functions reach. Every queued host function holds a reference, so it outlives the
//! level when a host function runs late.
struct StoreLevelShared
{
    std::atomic<uint64_t> opsSubmitted{0};
    //! The host function waits for its op instead of the stream (TLLM_KV_STORE_TIER_BLOCKING_HOSTFN=1).
    bool blockingHostFunction = false;
};
} // namespace detail

//! Counters of the store level. All zero, with level -1, when none is configured. Lock-free; safe after destroy().
struct StoreLevelStats
{
    int level = -1;
    uint64_t quotaBytes = 0;          // bytes of every store-level slot
    uint64_t numSlots = 0;            // slots over every pool group
    uint64_t demotedPages = 0;        // pages written into the store (from the host level or the GPU)
    uint64_t demotedBytes = 0;        // bytes written into the store
    uint64_t promotedPages = 0;       // pages read out of the store (into the host level or the GPU)
    uint64_t promotedBytes = 0;       // bytes read out of the store
    uint64_t promotedToGpuPages = 0;  // the promoted pages read straight into the GPU (not staged on the host level)
    uint64_t prefetchedPages = 0;     // store pages a prefetch moved to the host level
    uint64_t opsEnqueued = 0;         // IO ops placed on CUDA streams
    uint64_t opsSubmitted = 0;        // IO ops handed to the IO table by their host function
    uint64_t reserveCalls = 0;        // keepFreeBytes() calls
    uint64_t reserveEvictedSlots = 0; // slots keepFreeBytes() evicted one level down
    uint64_t parkedStreams = 0;       // migration streams held back from the stream pool (see StoreLevel)
    uint64_t stagingFallbacks = 0;    // GPU-bound store batches read through staging: the host level had no room
    uint64_t reserveFailures = 0;     // keepFreeBytes() calls the level below refused (nothing evicted)
    // Degrade fault mode only (see StoreLevel); zero otherwise.
    uint64_t failedPutOps = 0;      // put ops the IO completed with TLLM_KV_STORE_DONE_FAILED
    uint64_t failedGetOps = 0;      // get ops the IO completed with TLLM_KV_STORE_DONE_FAILED (their bytes are invalid)
    uint64_t lostSlots = 0;         // store slots whose last put failed (their pages are never matched)
    uint64_t unreadableMatches = 0; // store pages a reuse match refused: lost, unconfirmed or inside the breaker window
    bool faultDegrade = false;      // the degrade fault mode was on at the IO's attach
};

//! The storage counters plus the IO table's own (zeros before the first attach; the last values once detached).
struct StoreLevelStatsAll
{
    StoreLevelStats storage;
    TllmKvStoreLevelIoStats io{};
    bool ioAttached = false;
};

//! Layout the IO table is configured with: per store-level pool group, the bytes of one slot and the slot count.
struct StoreLevelLayout
{
    std::vector<uint64_t> slotBytes;
    std::vector<uint64_t> numSlots;
};

//! Orders the IO of a store level on CUDA streams.
//!
//! An op is placed on a stream as a host function, which only hands it to the IO table (submit() copies it and
//! returns), followed by a stream wait on a word of a pinned, device-mapped completion ring that the IO table writes
//! once every row of the op succeeded. The stream (a migration's temporary stream) therefore reaches its finish event,
//! and with it the pages' and slots' readyEvents, only after the store finished, and no thread waits on the store.
//!
//! Op seq s uses word s % kDoneWords and completes it with value s / kDoneWords + 1, so an op may take a word only
//! once the op one lap earlier completed it. attach(), detach(), enqueue() and parkStream() run under the manager's
//! exclusive lock; the counters and ioStats() may be read from any thread.
//!
//! A stream that holds store ops stays here (parkStream()) until all its work finished, instead of going back to the
//! process-wide stream pool when its migration returns: a pooled stream handed to an unrelated migration while a
//! store op still waits on it would make that migration wait for the store.
//!
//! Degrade fault mode (TLLM_KV_STORE_TIER_FAULT_MODE=degrade at attach(); the IO table must run its own degrade mode):
//! the IO completes an op with a failed row too, with TLLM_KV_STORE_DONE_FAILED in its done word. The level then
//! remembers the store slots each put op wrote; reapCompletedOps() folds the completed ops in: a failed put marks its
//! slots lost and opens a breaker window of TLLM_KV_STORE_TIER_BREAKER_S seconds (default 30). slotReadable() answers
//! false for a lost slot, for a slot whose last put is not confirmed yet, and for every slot inside the window, and the
//! reuse match then stops before such a page: a miss, recomputed, instead of a read the store may not answer. The ring
//! word of a failed op is reset to its plain value before the op one lap later reuses it. Without the mode (default)
//! nothing is tracked and slotReadable() is always true.
class StoreLevel
{
public:
    static constexpr uint32_t kDoneWords = 65536;
    static constexpr double kLapWaitS = 30.0;
    static constexpr double kDefaultBreakerS = 30.0;

    StoreLevel() = default;
    ~StoreLevel();

    StoreLevel(StoreLevel const&) = delete;
    StoreLevel& operator=(StoreLevel const&) = delete;

    //! Validates the table (abiVersion, structBytes, maxRowsPerOp, entry points), self-tests the completion ring,
    //! registers `ranges`, then configures `layout`. Throws std::invalid_argument for a bad table or when attached;
    //! std::runtime_error("store level: <step> failed with <code>") when the table refuses a step (the ranges
    //! registered so far are unregistered first), or when the ring self-test fails.
    void attach(TllmKvStoreLevelIoV1 const* table, std::vector<std::pair<MemAddress, size_t>> const& ranges,
        StoreLevelLayout const& layout);

    //! Waits until every enqueued op was submitted and the table drained (timeoutS in all), then unregisters every
    //! range and detaches. Throws std::runtime_error when ops remain (staying attached) or when a range fails to
    //! unregister (detached). No-op when not attached.
    void detach(double timeoutS);

    [[nodiscard]] bool attached() const noexcept
    {
        return mTable != nullptr;
    }

    //! Places `numRows` rows on `stream` as ops of at most maxRowsPerOp rows each, in order. An op whose completion
    //! word still belongs to the op one lap earlier waits for that op, up to the lap wait timeout. Throws LogicError
    //! when not attached, and std::runtime_error when that wait times out.
    void enqueue(CUstream stream, uint32_t kind, TllmKvStoreRow const* rows, size_t numRows);

    //! How long enqueue() waits for the op one lap earlier (default kLapWaitS).
    void setLapWaitTimeout(double seconds) noexcept
    {
        mLapWaitS = seconds;
    }

    [[nodiscard]] uint64_t opsEnqueued() const noexcept
    {
        return mOpsEnqueued.load(std::memory_order_relaxed);
    }

    [[nodiscard]] uint64_t opsSubmitted() const noexcept
    {
        return mShared->opsSubmitted.load(std::memory_order_relaxed);
    }

    //! Keeps `stream`, on which store ops were enqueued, until its work finished; then it returns to the pool. Also
    //! returns the parked streams that went idle since the last call (oldest first, up to the first busy one).
    void parkStream(CachedCudaStream stream);

    //! Returns the parked streams that went idle (oldest first, up to the first busy one) to the pool.
    void reapParkedStreams();

    //! Streams parked now. Lock-free.
    [[nodiscard]] uint64_t numParkedStreams() const noexcept
    {
        return mNumParked.load(std::memory_order_relaxed);
    }

    //! (attached, stats): the table's stats while attached, else the snapshot taken at the last detach.
    [[nodiscard]] std::pair<bool, TllmKvStoreLevelIoStats> ioStats() const;

    //! The completion word of op `seq` (host address), for tests that inspect the ring.
    [[nodiscard]] uint32_t const* doneWord(uint64_t seq) const noexcept
    {
        return mDoneHost == nullptr ? nullptr : mDoneHost + seq % kDoneWords;
    }

    //! The seq the next enqueued op takes.
    [[nodiscard]] uint64_t nextSeq() const noexcept
    {
        return mNextSeq;
    }

    // ---- Degrade fault mode (see the class comment) -------------------------------------------------------------

    //! Whether the degrade fault mode was on at attach().
    [[nodiscard]] bool tracksFailures() const noexcept
    {
        return mTrackFailures;
    }

    //! Folds the completed ops into the slot states (a failed put marks its slots lost and opens the breaker window;
    //! a successful one clears them). No-op without the degrade mode. Requires the manager's exclusive lock.
    void reapCompletedOps();

    //! Whether a store page in slot `slot` of store pool group `poolGroup` may be read: false for a lost slot, a slot
    //! whose last put is not confirmed yet, and every slot inside the breaker window. Always true without the degrade
    //! mode. Requires the manager's lock (shared or exclusive).
    [[nodiscard]] bool slotReadable(uint32_t poolGroup, uint64_t slot) const noexcept;

    //! Counts a store page a reuse match refused. Lock-free.
    void noteUnreadableMatch() const noexcept
    {
        mUnreadableMatches.fetch_add(1, std::memory_order_relaxed);
    }

    //! Degrade-mode counters (zeros otherwise). Lock-free.
    struct FailureStats
    {
        uint64_t failedPutOps = 0;
        uint64_t failedGetOps = 0;
        uint64_t lostSlots = 0;
        uint64_t unreadableMatches = 0;
    };

    [[nodiscard]] FailureStats failureStats() const noexcept
    {
        return {mFailedPutOps.load(std::memory_order_relaxed), mFailedGetOps.load(std::memory_order_relaxed),
            mLostSlots.load(std::memory_order_relaxed), mUnreadableMatches.load(std::memory_order_relaxed)};
    }

private:
    void ensureRing();
    void selfTestRing();
    void setSlotLost(uint32_t poolGroup, uint64_t slot, bool lost) noexcept;

    //! An enqueued op whose outcome reapCompletedOps() has not folded in yet (degrade mode only).
    struct TrackedOp
    {
        uint64_t seq = 0;
        uint32_t kind = 0;
        uint32_t* word = nullptr;
        uint32_t value = 0;
        std::vector<std::pair<uint32_t, uint64_t>> putSlots; // (store pool group, slot) a put writes
    };

    static constexpr uint64_t kNoPendingPut = ~uint64_t{0};
    bool mTrackFailures = false;
    double mBreakerS = kDefaultBreakerS;
    std::deque<TrackedOp> mTrackedOps;
    // Per store pool group, per slot: the seq of the slot's unconfirmed put (kNoPendingPut: none), and whether the
    // slot's last put failed.
    std::vector<std::vector<uint64_t>> mPendingPutSeq;
    std::vector<std::vector<uint8_t>> mSlotLost;
    std::chrono::steady_clock::time_point mBreakerUntil{};
    std::atomic<uint64_t> mFailedPutOps{0};
    std::atomic<uint64_t> mFailedGetOps{0};
    std::atomic<uint64_t> mLostSlots{0};
    mutable std::atomic<uint64_t> mUnreadableMatches{0};

    TllmKvStoreLevelIoV1 const* mTable = nullptr;
    std::vector<std::pair<MemAddress, size_t>> mRanges;
    std::shared_ptr<detail::StoreLevelShared> mShared = std::make_shared<detail::StoreLevelShared>();
    // kDoneWords ring words, then one word for the self-test.
    uint32_t* mDoneHost = nullptr;
    CUdeviceptr mDoneDevice = 0;
    uint32_t mSelfTestValue = 0;
    uint64_t mNextSeq = 0;
    double mLapWaitS = kLapWaitS;
    std::atomic<uint64_t> mOpsEnqueued{0};
    // Streams that hold store ops, oldest first (see parkStream()); mNumParked mirrors the size for lock-free stats.
    std::deque<std::unique_ptr<CachedCudaStream>> mParked;
    std::atomic<uint64_t> mNumParked{0};
    // Guards mTable against ioStats() while detach() clears it, and the snapshot taken then.
    mutable std::mutex mStatsMutex;
    TllmKvStoreLevelIoStats mLastIoStats{};
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
