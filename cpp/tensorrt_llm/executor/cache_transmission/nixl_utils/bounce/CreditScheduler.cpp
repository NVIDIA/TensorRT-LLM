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

#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/CreditScheduler.h"

#include "tensorrt_llm/common/assert.h"

#include <algorithm>

namespace tensorrt_llm::executor::kv_cache::bounce
{

CreditScheduler::CreditScheduler(
    std::uint64_t baseAddr, std::size_t arenaBytes, std::size_t minBlock, std::uint32_t maxWindow)
    : mArena(arenaBytes, minBlock)
    , mBaseAddr(baseAddr)
    , mMaxWindow(maxWindow == 0 ? 1 : maxWindow)
    , mEagerBudgetBytes(mArena.capacity() / 2)
{
}

std::size_t CreditScheduler::activeFlows() const
{
    std::lock_guard<std::mutex> lk(mMu);
    std::size_t n = 0;
    for (auto const& [_, st] : mFlows)
    {
        if (!st.pending.empty())
        {
            ++n;
        }
    }
    return n;
}

std::size_t CreditScheduler::heldCount(std::string const& flow) const
{
    std::lock_guard<std::mutex> lk(mMu);
    auto it = mFlows.find(flow);
    return it == mFlows.end() ? 0 : it->second.held.size();
}

void CreditScheduler::ensureInRing(std::string const& flow)
{
    if (std::find(mRing.begin(), mRing.end(), flow) == mRing.end())
    {
        mRing.push_back(flow);
    }
}

void CreditScheduler::dropFromRing(std::string const& flow)
{
    auto it = std::find(mRing.begin(), mRing.end(), flow);
    if (it != mRing.end())
    {
        auto const idx = static_cast<std::size_t>(it - mRing.begin());
        mRing.erase(it);
        if (mRing.empty())
        {
            mCursor = 0;
        }
        else if (idx < mCursor)
        {
            --mCursor;
        }
        mCursor %= mRing.size();
    }
}

// Hand out as many region grants as possible RIGHT NOW, fairly and bounded. Called whenever space
// frees up or new demand arrives (onWant / onScatterDone / releaseLocal / reclaim*). Three rules:
//   1. Fair: rotate over flows round-robin (mRing + mCursor) so no flow starves.
//   2. Per-flow window W: a flow may hold at most mMaxWindow regions in flight (held.size() < W);
//      more must wait for an ACK to free one. This bounds a single flow's pipeline depth.
//   3. Arena capacity: all flows' regions share one buddy arena; if the next chunk doesn't fit, skip
//      this flow (a smaller chunk elsewhere may still fit) -> backpressure, never an error.
// Shape: each inner sweep grants AT MOST ONE region then breaks, advancing the cursor past the flow
// just served; the outer loop re-sweeps from there. That one-at-a-time + advance gives strict
// rotation (A,B,A,B,...) instead of draining one flow's whole window first. It stops when a full
// sweep grants nothing (every flow is done / window-full / can't fit). Returns the GRANTs to send.
std::vector<Grant> CreditScheduler::schedule()
{
    std::vector<Grant> grants;
    bool progress = true;
    // Re-sweep as long as the previous sweep granted something (a grant may free a slot/leave room for
    // the next flow); stop when a whole sweep makes no progress, or the ring is empty.
    while (progress && !mRing.empty())
    {
        progress = false;
        // One round-robin sweep from the cursor; grant the first eligible flow whose next chunk fits.
        for (std::size_t k = 0; k < mRing.size(); ++k)
        {
            std::size_t const idx = (mCursor + k) % mRing.size();
            // find() (not operator[]) — the ring/flows invariant says the key exists, but [] would
            // fabricate an un-erasable empty FlowState tombstone if it ever didn't (reintroducing the
            // very leak the design fights). Skip loudly instead.
            auto fit = mFlows.find(mRing[idx]);
            TLLM_CHECK_DEBUG(fit != mFlows.end());
            if (fit == mFlows.end())
            {
                continue;
            }
            auto& st = fit->second;
            if (st.pending.empty() || st.held.size() >= mMaxWindow)
            {
                continue; // nothing more wanted, or at the per-flow window cap
            }
            std::uint32_t const want = st.pending.front();
            // NOTE: no aging — a flow whose FRONT chunk can't fit right now is skipped while smaller
            // chunks elsewhere keep cycling, so a large chunk can be passed over under sustained small
            // traffic (HOL; never a deadlock — buddy coalescing eventually frees a high-order block,
            // and maxChunkBytes <= arena guarantees it can fit a drained arena).
            auto off = mArena.alloc(want);
            if (!off)
            {
                continue; // arena can't fit this chunk now -> try another flow (smaller may fit)
            }
            st.pending.pop_front();
            st.held.insert(*off);
            grants.push_back(Grant{mRing[idx], *off, mBaseAddr + *off, want});
            mCursor = (idx + 1) % mRing.size(); // next sweep starts AFTER this flow -> round-robin
            progress = true;
            // One grant per sweep: break out and let the outer loop re-sweep from the advanced cursor,
            // so grants alternate across flows (strict rotation) rather than filling one flow first.
            break;
        }
    }
    return grants;
}

void CreditScheduler::eraseIfDone(std::string const& flow)
{
    auto it = mFlows.find(flow);
    if (it != mFlows.end() && it->second.pending.empty() && it->second.held.empty())
    {
        mFlows.erase(it);
        dropFromRing(flow);
    }
}

std::vector<Grant> CreditScheduler::onWant(std::string const& flow, std::vector<std::uint32_t> const& chunkBytes)
{
    std::lock_guard<std::mutex> lk(mMu);
    auto& st = mFlows[flow];
    st.pending.assign(chunkBytes.begin(), chunkBytes.end());
    if (!chunkBytes.empty())
    {
        ensureInRing(flow);
    }
    else
    {
        // cancel: drop now if nothing is still in flight; otherwise reclaimed when held drains.
        eraseIfDone(flow);
    }
    return schedule();
}

std::vector<Grant> CreditScheduler::onScatterDone(std::string const& flow, std::uint64_t offset)
{
    std::lock_guard<std::mutex> lk(mMu);
    auto it = mFlows.find(flow);
    if (it != mFlows.end() && it->second.held.erase(offset) > 0)
    {
        mArena.free(offset);
    }
    // else: not held by this flow (dup ACK / already reclaimed) -> ignore, stay idempotent.
    eraseIfDone(flow);
    return schedule();
}

void CreditScheduler::dropFlow(
    std::string const& flow, std::unordered_set<std::uint64_t> const& busy, std::vector<std::uint64_t>& deferredOut)
{
    auto it = mFlows.find(flow);
    if (it == mFlows.end())
    {
        return;
    }
    for (auto off : it->second.held)
    {
        if (busy.count(off) > 0)
        {
            // A scatter is still reading this region -> caller frees it later via freeOrphanRegion;
            // track it as an orphan so that call only frees a genuine deferred region.
            deferredOut.push_back(off);
            mOrphans.insert(off);
        }
        else
        {
            mArena.free(off);
        }
    }
    mFlows.erase(it);
    dropFromRing(flow);
}

std::vector<Grant> CreditScheduler::reclaimByPrefix(
    std::string const& prefix, std::unordered_set<std::uint64_t> const& busy, std::vector<std::uint64_t>& deferredOut)
{
    // Guard the degenerate empty prefix: compare(0,0,"")==0 for EVERY key, so it would reclaim all
    // flows of all peers. Callers always pass a real "peer<sep>" prefix; refuse empty defensively.
    if (prefix.empty())
    {
        return {};
    }
    std::lock_guard<std::mutex> lk(mMu);
    std::vector<std::string> victims;
    for (auto const& [key, st] : mFlows)
    {
        if (key.size() >= prefix.size() && key.compare(0, prefix.size(), prefix) == 0)
        {
            victims.push_back(key);
        }
    }
    for (auto const& key : victims)
    {
        dropFlow(key, busy, deferredOut);
    }
    return schedule();
}

std::vector<Grant> CreditScheduler::reclaimFlow(
    std::string const& flow, std::unordered_set<std::uint64_t> const& busy, std::vector<std::uint64_t>& deferredOut)
{
    std::lock_guard<std::mutex> lk(mMu);
    dropFlow(flow, busy, deferredOut);
    return schedule();
}

bool CreditScheduler::heldByFlow(std::string const& flow, std::uint64_t offset) const
{
    std::lock_guard<std::mutex> lk(mMu);
    auto it = mFlows.find(flow);
    return it != mFlows.end() && it->second.held.count(offset) > 0;
}

std::vector<Grant> CreditScheduler::freeOrphanRegion(std::uint64_t offset)
{
    std::lock_guard<std::mutex> lk(mMu);
    // Only free a region we actually deferred as an orphan. Guards against a stray/duplicate call
    // freeing an offset that may since have been re-allocated to a live flow (defense in depth; the
    // transport already gates on its own mScattering orphaned-flag map).
    if (mOrphans.erase(offset) > 0)
    {
        mArena.free(offset);
    }
    return schedule();
}

std::optional<std::uint64_t> CreditScheduler::acquireLocal(std::size_t bytes, bool eager)
{
    std::lock_guard<std::mutex> lk(mMu);
    auto off = mArena.alloc(bytes);
    if (!off)
    {
        return std::nullopt; // arena full/fragmented -> caller parks + retries (never blocks)
    }
    if (eager)
    {
        // Cap all eager (credit-less) staging at half the arena so incoming grants can always
        // progress (see header). Budget accounting uses the ROUNDED buddy-block size — that is what
        // the arena actually loses.
        std::size_t const rounded = mArena.blockBytes(*off);
        if (mEagerHeldBytes + rounded > mEagerBudgetBytes)
        {
            mArena.free(*off);
            return std::nullopt; // over the eager budget -> caller parks; credit path unaffected
        }
        mEagerHeld.emplace(*off, rounded);
        mEagerHeldBytes += rounded;
    }
    mLocalHeld.insert(*off);
    return *off;
}

void CreditScheduler::promoteLocal(std::uint64_t offset)
{
    std::lock_guard<std::mutex> lk(mMu);
    auto it = mEagerHeld.find(offset);
    if (it != mEagerHeld.end())
    {
        mEagerHeldBytes -= it->second;
        mEagerHeld.erase(it);
    }
}

std::vector<Grant> CreditScheduler::releaseLocal(std::uint64_t offset)
{
    std::lock_guard<std::mutex> lk(mMu);
    if (mLocalHeld.erase(offset) > 0)
    {
        auto it = mEagerHeld.find(offset);
        if (it != mEagerHeld.end())
        {
            mEagerHeldBytes -= it->second;
            mEagerHeld.erase(it);
        }
        mArena.free(offset);
    }
    return schedule(); // freed bytes may now let a waiting remote flow alloc its next chunk
}

} // namespace tensorrt_llm::executor::kv_cache::bounce
