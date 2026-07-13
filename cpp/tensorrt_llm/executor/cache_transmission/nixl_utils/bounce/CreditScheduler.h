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

#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/BuddyAllocator.h"

#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tensorrt_llm::executor::kv_cache::bounce
{

/// A credit handed to a flow: exclusive write permission for one variable-size receiver region.
/// `offset` is the region's arena offset (its opaque handle), `addr = baseAddr + offset` the
/// absolute device address, `len` its byte size (the chunk's packed bytes).
struct Grant
{
    std::string flow;       // flow id ("peer<sep>rid") the grant belongs to (NOT a bare agent name)
    std::uint64_t offset{}; // arena offset (region handle)
    std::uint64_t addr{};   // baseAddr + offset (what the sender RDMA-writes to)
    std::uint32_t len{};    // region size = the chunk's packed bytes
};

/// Receiver-side credit allocator + fair scheduler (R3) over a single shared arena — pure logic,
/// no threads / IO / CUDA (the GPU buffer lives in the transport; this owns only a BuddyAllocator
/// over byte offsets + the base address for computing absolute addrs).
///
/// VARIABLE REGIONS: instead of fixed full slots, each chunk gets a region sized to its actual
/// bytes, so MANY small requests fit (high concurrency, no waste) AND a request larger than the
/// whole arena streams through (its chunks are each ≤ maxChunkBytes and recycled per ACK). The
/// per-flow window cap is in REGION COUNT (pipeline depth W); the arena capacity bounds aggregate
/// concurrency (alloc fails -> backpressure, never deadlock).
///
/// TERMINOLOGY: the string identifying a client is an opaque **flow id** ("peerName<sep>rid"), NOT
/// an agent name (cf. `peer` in ControlChannel/TransferEngine). reclaimByPrefix() is the only
/// peer-level op. The local sender shares this same arena via acquireLocal() (gather staging).
///
/// Each event method mutates state and returns the GRANTs the caller should now send.
///
/// THREADING CONTRACT: every public method takes the internal mutex, so calls are safe from any
/// thread. The IO thread remains the primary owner (all flow-state events happen there); the one
/// cross-thread caller is acquireLocal() from submit() app threads (eager gather staging).
class CreditScheduler
{
public:
    /// @param baseAddr   device address of arena offset 0 (Grant.addr = baseAddr + offset).
    /// @param arenaBytes total arena size; @param minBlock buddy min block; @param maxWindow per-flow
    ///                   in-flight region cap W (pipeline depth; must be > 0).
    CreditScheduler(std::uint64_t baseAddr, std::size_t arenaBytes, std::size_t minBlock, std::uint32_t maxWindow);

    /// Flow announces the per-chunk byte sizes it wants to write (in order). EMPTY = cancel.
    [[nodiscard]] std::vector<Grant> onWant(std::string const& flow, std::vector<std::uint32_t> const& chunkBytes);

    /// A region finished scattering on the receiver -> free it and re-schedule. Idempotent.
    [[nodiscard]] std::vector<Grant> onScatterDone(std::string const& flow, std::uint64_t offset);

    /// Reclaim every flow whose id starts with `prefix` (drop all flows of a gone peer). DEFERS
    /// freeing any held region whose offset is in `busy` (a scatter is still reading it): those are
    /// appended to `deferredOut` instead of freed, and the caller MUST later call freeOrphanRegion()
    /// for each once its scatter completes. (Pass an empty `busy` for "free everything now".)
    [[nodiscard]] std::vector<Grant> reclaimByPrefix(std::string const& prefix,
        std::unordered_set<std::uint64_t> const& busy, std::vector<std::uint64_t>& deferredOut);

    /// Free a region deferred by reclaimByPrefix (its in-flight scatter has finished) + re-schedule.
    [[nodiscard]] std::vector<Grant> freeOrphanRegion(std::uint64_t offset);

    /// Cancel ONE flow (explicit abort / empty WANT): free its held regions and drop it. Any held
    /// region in `busy` (a scatter is still reading it) is deferred to `deferredOut` instead of freed
    /// (caller later calls freeOrphanRegion). Frees the granted-but-unwritten regions a failed sender
    /// would otherwise leak until peer loss.
    [[nodiscard]] std::vector<Grant> reclaimFlow(std::string const& flow, std::unordered_set<std::uint64_t> const& busy,
        std::vector<std::uint64_t>& deferredOut);

    /// True if `flow` currently holds region `offset`. Lets the transport drop a late DATA for a
    /// region this flow no longer owns (cancelled/reclaimed) — scattering a freed/re-granted region
    /// would corrupt another flow's data.
    [[nodiscard]] bool heldByFlow(std::string const& flow, std::uint64_t offset) const;

    // ---- local (sender) role: gather staging from the SAME arena ----
    /// Allocate a region of `bytes` for local gather staging (non-blocking). Returns its offset, or
    /// nullopt if the arena can't fit it right now (caller parks and retries). With `eager` the
    /// allocation is additionally capped so that all eager (credit-less) local regions together stay
    /// under HALF the arena capacity — on a bidirectional deployment this guarantees each side can
    /// always still grant incoming regions, so two eager senders can never starve each other into a
    /// circular wait (chunk waits for a GRANT the peer can't give because ITS arena is full of eager
    /// staging, and vice versa). Credit-backed (non-eager) acquisitions are not capped.
    [[nodiscard]] std::optional<std::uint64_t> acquireLocal(std::size_t bytes, bool eager = false);
    /// An eager region's credit arrived: it is now credit-backed, so stop counting it against the
    /// eager budget (otherwise steady-state pipelining would be throttled to the eager cap even
    /// though every in-flight chunk has its credit). No-op for non-eager offsets.
    void promoteLocal(std::uint64_t offset);
    /// Return a locally-held region (its chunk was ACKed / failed) to the arena + re-schedule.
    [[nodiscard]] std::vector<Grant> releaseLocal(std::uint64_t offset);

    [[nodiscard]] std::size_t localHeldCount() const
    {
        std::lock_guard<std::mutex> lk(mMu);
        return mLocalHeld.size();
    }

    // ---- inspectors (for tests / metrics) ----
    [[nodiscard]] std::size_t freeBytes()
    {
        std::lock_guard<std::mutex> lk(mMu);
        return mArena.freeBytes();
    }

    /// Largest region a fully-drained arena can ever hand out (the buddy allocator's usable capacity,
    /// rounded DOWN to minBlock<<maxOrder). A chunk larger than this can never be granted, so callers
    /// clamp maxChunkBytes to it.
    [[nodiscard]] std::size_t arenaCapacity() const noexcept
    {
        return mArena.capacity();
    }

    /// Byte size of the buddy block backing a granted region offset (0 if not allocated). The whole
    /// block belongs to one flow, so it bounds how far a scatter may read without touching another
    /// flow's region.
    [[nodiscard]] std::size_t regionBytes(std::uint64_t offset) const
    {
        std::lock_guard<std::mutex> lk(mMu);
        return mArena.blockBytes(offset);
    }

    [[nodiscard]] std::size_t heldCount(std::string const& flow) const;
    [[nodiscard]] std::size_t activeFlows() const;

    [[nodiscard]] std::size_t trackedFlows() const
    {
        std::lock_guard<std::mutex> lk(mMu);
        return mFlows.size();
    }

private:
    struct FlowState
    {
        std::deque<std::uint32_t> pending;      // per-chunk byte sizes still wanting a grant (FIFO)
        std::unordered_set<std::uint64_t> held; // region offsets currently leased to this flow
    };

    std::vector<Grant> schedule();              // grant while the arena has room and eligible flows exist
    void ensureInRing(std::string const& flow); // add flow to the round-robin ring if absent
    void dropFromRing(std::string const& flow); // remove flow from the round-robin ring
    void eraseIfDone(std::string const& flow);  // pending empty && held empty -> drop the flow
    // Free one flow's held regions (busy ones deferred to deferredOut + tracked as orphans), then
    // erase the flow + drop it from the ring. Shared by reclaimByPrefix and reclaimFlow.
    void dropFlow(std::string const& flow, std::unordered_set<std::uint64_t> const& busy,
        std::vector<std::uint64_t>& deferredOut);

    // Guards ALL mutable state below. Uncontended in practice: everything runs on the IO thread
    // except acquireLocal() from submit() app threads (eager gather staging).
    mutable std::mutex mMu;

    BuddyAllocator mArena;                             // the single shared region allocator (byte offsets)
    std::uint64_t mBaseAddr{};                         // device addr of offset 0
    std::uint32_t mMaxWindow{};                        // per-flow in-flight region cap (W)

    std::unordered_map<std::string, FlowState> mFlows; // per-flow state (opaque flow id, NOT agent name)
    std::unordered_set<std::uint64_t> mLocalHeld;      // regions taken for local gather staging
    // Rounded (buddy-block) bytes of local regions acquired EAGERLY (before their credit arrived),
    // and the cap they must stay under (half the arena) — see acquireLocal(). Offsets still tracked
    // in mEagerHeld so releaseLocal() knows how much to subtract.
    std::unordered_map<std::uint64_t, std::size_t> mEagerHeld;
    std::size_t mEagerHeldBytes{0};
    std::size_t mEagerBudgetBytes{0};
    std::unordered_set<std::uint64_t>
        mOrphans; // regions deferred by reclaimByPrefix (busy scatter), awaiting freeOrphanRegion
    // Round-robin ring of active flow keys (insertion order). NOTE: "ring" not "order" — distinct
    // from BuddyAllocator's size `order` (mArena), which is the power-of-two block exponent.
    std::vector<std::string> mRing;
    std::size_t mCursor{0}; // round-robin cursor into mRing
};

} // namespace tensorrt_llm::executor::kv_cache::bounce
