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

CreditScheduler::CreditScheduler(std::uint64_t baseAddr, std::size_t arenaSizeBytes,
    std::size_t arenaAllocationGranularityBytes, std::uint32_t maxInflightChunksPerRequest,
    std::function<std::chrono::steady_clock::time_point()> clock)
    : mArena(arenaSizeBytes, arenaAllocationGranularityBytes)
    , mBaseAddr(baseAddr)
    , mMaxInflightChunksPerRequest(maxInflightChunksPerRequest == 0 ? 1 : maxInflightChunksPerRequest)
    , mEagerBudgetBytes(mArena.capacity() / 2)
    , mClock(clock ? std::move(clock) : [] { return std::chrono::steady_clock::now(); })
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
    if (mDrainFlow && *mDrainFlow == flow)
    {
        mDrainFlow.reset();
    }
    auto it = std::find(mRing.begin(), mRing.end(), flow);
    if (it != mRing.end())
    {
        auto const idx = static_cast<std::size_t>(it - mRing.begin());
        mRing.erase(it);
        if (mRing.empty())
        {
            // Erased the LAST flow: the modulo below would divide by zero (SIGFPE), so reset and bail.
            mCursor = 0;
            return;
        }
        if (idx < mCursor)
        {
            --mCursor;
        }
        mCursor %= mRing.size();
    }
}

void CreditScheduler::maybeActivateDrain()
{
    if (mDrainFlow || mRing.empty())
    {
        return;
    }

    std::uint64_t const ringSize = static_cast<std::uint64_t>(mRing.size());
    std::uint64_t const bypassThreshold = std::max(kMinimumBypassGrants, kBypassRounds * ringSize);
    std::optional<std::uint64_t> oldestBlockedSequence;

    // Scan from the round-robin cursor so equal-age candidates retain the scheduler's normal order.
    for (std::size_t k = 0; k < mRing.size(); ++k)
    {
        std::size_t const idx = (mCursor + k) % mRing.size();
        auto const fit = mFlows.find(mRing[idx]);
        TLLM_CHECK_DEBUG(fit != mFlows.end());
        if (fit == mFlows.end())
        {
            continue;
        }
        auto const& st = fit->second;
        if (st.pending.empty() || st.held.size() >= mMaxInflightChunksPerRequest || !st.blockedAtGrantSequence)
        {
            continue;
        }
        std::uint64_t const bypassed = mGrantSequence - *st.blockedAtGrantSequence;
        if (bypassed < bypassThreshold)
        {
            continue;
        }
        if (!oldestBlockedSequence || *st.blockedAtGrantSequence < *oldestBlockedSequence)
        {
            oldestBlockedSequence = st.blockedAtGrantSequence;
            mDrainFlow = mRing[idx];
        }
    }
}

// Hand out as many region grants as possible RIGHT NOW, fairly and bounded. Called whenever space
// frees up or new demand arrives (onWant / onScatterDone / releaseLocal / reclaim*). Three rules:
//   1. Fair: rotate over flows round-robin (mRing + mCursor). A head chunk that repeatedly fails to
//      fit while other grants pass it eventually enters receiver drain mode: no NEW remote grants
//      are issued until that head fits, so sustained smaller traffic cannot starve it.
//   2. Per-request cap: a flow may hold at most mMaxInflightChunksPerRequest allocations; more must
//      wait for scatter completion to free one.
//   3. Arena capacity: all flows' regions share one buddy arena; if the next chunk doesn't fit, skip
//      this flow (a smaller chunk elsewhere may still fit) -> backpressure, never an error.
// Shape: each inner sweep grants AT MOST ONE region then breaks, advancing the cursor past the flow
// just served; the outer loop re-sweeps from there. That one-at-a-time + advance gives strict
// rotation (A,B,A,B,...) instead of filling one request to its in-flight cap first. It stops when a full
// sweep grants nothing (every flow is done / at its in-flight cap / can't fit). Returns the GRANTs.
std::vector<Grant> CreditScheduler::schedule()
{
    std::vector<Grant> grants;
    // Re-sweep as long as the previous sweep granted something; stop when a whole sweep makes no
    // progress or the ring is empty.
    while (!mRing.empty())
    {
        maybeActivateDrain();
        if (mDrainFlow)
        {
            auto fit = mFlows.find(*mDrainFlow);
            TLLM_CHECK_DEBUG(fit != mFlows.end());
            if (fit == mFlows.end() || fit->second.pending.empty()
                || fit->second.held.size() >= mMaxInflightChunksPerRequest)
            {
                mDrainFlow.reset();
                continue;
            }

            auto& st = fit->second;
            std::uint32_t const want = st.pending.front();
            auto off = mArena.alloc(want);
            if (!off)
            {
                // Existing remote regions keep progressing and freeing space, but do not refill them
                // with smaller chunks. acquireLocal() is deliberately unaffected: this is a
                // receiver-only admission barrier and cannot introduce a bidirectional circular wait.
                return grants;
            }

            auto const ringIt = std::find(mRing.begin(), mRing.end(), *mDrainFlow);
            TLLM_CHECK_DEBUG(ringIt != mRing.end());
            std::size_t const idx = static_cast<std::size_t>(ringIt - mRing.begin());
            st.pending.pop_front();
            st.held.insert(*off);
            st.blockedAtGrantSequence.reset();
            st.lastProgress = mClock(); // issuing a grant renews the flow's lease
            grants.push_back(Grant{*mDrainFlow, *off, mBaseAddr + *off, want});
            mCursor = (idx + 1) % mRing.size();
            ++mGrantSequence;
            mDrainFlow.reset();
            continue;
        }

        bool progress = false;
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
            if (st.pending.empty() || st.held.size() >= mMaxInflightChunksPerRequest)
            {
                continue; // nothing more wanted, or at the per-request in-flight cap
            }
            std::uint32_t const want = st.pending.front();
            auto off = mArena.alloc(want);
            if (!off)
            {
                if (!st.blockedAtGrantSequence)
                {
                    st.blockedAtGrantSequence = mGrantSequence;
                }
                continue; // arena can't fit this chunk now -> try another flow (smaller may fit)
            }
            st.pending.pop_front();
            st.held.insert(*off);
            st.blockedAtGrantSequence.reset();
            st.lastProgress = mClock();         // issuing a grant renews the flow's lease
            grants.push_back(Grant{mRing[idx], *off, mBaseAddr + *off, want});
            mCursor = (idx + 1) % mRing.size(); // next sweep starts AFTER this flow -> round-robin
            ++mGrantSequence;
            progress = true;
            // One grant per sweep: break out and let the outer loop re-sweep from the advanced cursor,
            // so grants alternate across flows (strict rotation) rather than filling one flow first.
            break;
        }
        if (!progress)
        {
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
    st.blockedAtGrantSequence.reset();
    st.lastProgress = mClock(); // a fresh WANT renews the flow's lease
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
        it->second.lastProgress = mClock(); // a completed scatter is flow progress: renew the lease
    }
    // else: not held by this flow (dup ACK / already reclaimed) -> ignore, stay idempotent.
    eraseIfDone(flow);
    return schedule();
}

void CreditScheduler::dropFlow(std::string const& flow, std::unordered_set<std::uint64_t> const& busy,
    std::vector<std::uint64_t>& deferredOut, std::chrono::milliseconds quarantineFor)
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
        else if (quarantineFor.count() > 0)
        {
            // Receiver-initiated reclaim: the peer may still be RDMA-writing this granted region
            // (a one-sided write cannot be aborted, and no sender-side drain preceded this call).
            // Keep it allocated (never re-grantable) until reapQuarantine() passes the deadline.
            mQuarantined.emplace(off, mClock() + quarantineFor);
        }
        else
        {
            mArena.free(off);
        }
    }
    mFlows.erase(it);
    dropFromRing(flow);
}

std::vector<Grant> CreditScheduler::reclaimByPrefix(std::string const& prefix,
    std::unordered_set<std::uint64_t> const& busy, std::vector<std::uint64_t>& deferredOut,
    std::chrono::milliseconds quarantineFor)
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
        dropFlow(key, busy, deferredOut, quarantineFor);
    }
    return schedule();
}

std::vector<Grant> CreditScheduler::reclaimFlow(std::string const& flow, std::unordered_set<std::uint64_t> const& busy,
    std::vector<std::uint64_t>& deferredOut, std::chrono::milliseconds quarantineFor)
{
    std::lock_guard<std::mutex> lk(mMu);
    dropFlow(flow, busy, deferredOut, quarantineFor);
    return schedule();
}

std::vector<std::string> CreditScheduler::staleFlows(std::chrono::milliseconds idleLimit) const
{
    std::lock_guard<std::mutex> lk(mMu);
    auto const now = mClock();
    std::vector<std::string> stale;
    for (auto const& [key, st] : mFlows)
    {
        // The lease protects arena REGIONS, so only flows holding one can go stale. A pending-only
        // flow ties up no memory and may legitimately be idle for long: it is simply queued behind a
        // full arena (its own sender's requestTimeoutMs bounds that wait). If its sender is dead it
        // lingers as a few bytes of bookkeeping until a grant finally starts its lease.
        if (!st.held.empty() && now - st.lastProgress > idleLimit)
        {
            stale.push_back(key);
        }
    }
    return stale;
}

std::vector<Grant> CreditScheduler::reapQuarantine()
{
    std::lock_guard<std::mutex> lk(mMu);
    auto const now = mClock();
    bool freed = false;
    for (auto it = mQuarantined.begin(); it != mQuarantined.end();)
    {
        if (now >= it->second)
        {
            mArena.free(it->first);
            it = mQuarantined.erase(it);
            freed = true;
        }
        else
        {
            ++it;
        }
    }
    // Nothing freed -> nothing changed; skip the schedule() ring sweep this polling tick runs into.
    return freed ? schedule() : std::vector<Grant>{};
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
