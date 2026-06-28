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
{
}

std::size_t CreditScheduler::activeFlows() const
{
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

std::vector<Grant> CreditScheduler::schedule()
{
    std::vector<Grant> grants;
    bool progress = true;
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
            // and maxChunkBytes <= arena guarantees it can fit a drained arena). See DESIGN.md §5.
            auto off = mArena.alloc(want);
            if (!off)
            {
                continue; // arena can't fit this chunk now -> try another flow (smaller may fit)
            }
            st.pending.pop_front();
            st.held.insert(*off);
            grants.push_back(Grant{mRing[idx], *off, mBaseAddr + *off, want});
            mCursor = (idx + 1) % mRing.size();
            progress = true;
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
    dropFlow(flow, busy, deferredOut);
    return schedule();
}

bool CreditScheduler::heldByFlow(std::string const& flow, std::uint64_t offset) const
{
    auto it = mFlows.find(flow);
    return it != mFlows.end() && it->second.held.count(offset) > 0;
}

std::vector<Grant> CreditScheduler::freeOrphanRegion(std::uint64_t offset)
{
    // Only free a region we actually deferred as an orphan. Guards against a stray/duplicate call
    // freeing an offset that may since have been re-allocated to a live flow (defense in depth; the
    // transport already gates on its own mScattering orphaned-flag map).
    if (mOrphans.erase(offset) > 0)
    {
        mArena.free(offset);
    }
    return schedule();
}

std::optional<std::uint64_t> CreditScheduler::acquireLocal(std::size_t bytes)
{
    auto off = mArena.alloc(bytes);
    if (!off)
    {
        return std::nullopt; // arena full/fragmented -> caller parks + retries (never blocks)
    }
    mLocalHeld.insert(*off);
    return *off;
}

std::vector<Grant> CreditScheduler::releaseLocal(std::uint64_t offset)
{
    if (mLocalHeld.erase(offset) > 0)
    {
        mArena.free(offset);
    }
    return schedule(); // freed bytes may now let a waiting remote flow alloc its next chunk
}

} // namespace tensorrt_llm::executor::kv_cache::bounce
