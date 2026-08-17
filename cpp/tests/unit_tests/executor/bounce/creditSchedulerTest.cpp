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

#include <gtest/gtest.h>

#include <chrono>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace b = tensorrt_llm::executor::kv_cache::bounce;

namespace
{

// The variable-region scheduler carves byte offsets out of one arena. Most tests model an arena of
// N EQUAL regions by making the buddy min block == one region size, so each "want a region" is one
// minimum block and the arena holds exactly N of them — i.e. the old fixed-slot semantics, now
// expressed in bytes. (VariableSizeRegions exercises the genuinely variable case.)
constexpr std::uint64_t kBase = 0x100000ULL; // arena base device address (Grant.addr = kBase + offset)
constexpr std::uint32_t kRegion = 0x1000ULL; // 4096B: one "slot"-sized region (== buddy min block)

// Scheduler over an arena holding exactly `nRegions` regions of kRegion bytes.
b::CreditScheduler makeSched(std::uint32_t nRegions, std::uint32_t maxInflightChunksPerRequest)
{
    return b::CreditScheduler(
        kBase, static_cast<std::size_t>(nRegions) * kRegion, kRegion, maxInflightChunksPerRequest);
}

// "want n one-region chunks" -> n equal chunks of kRegion bytes (FIFO order).
std::vector<std::uint32_t> want(std::uint32_t n)
{
    return std::vector<std::uint32_t>(n, kRegion);
}

// Free regions currently available (all equal-sized here, so bytes / region size).
std::size_t freeRegions(b::CreditScheduler& s)
{
    return s.freeBytes() / kRegion;
}

// Model the Python transceiver's bounded worker queues: each sender has at most `threadsPerSender`
// synchronous submit+wait requests active, and a worker submits its next one-chunk request as soon as
// the previous request completes. The resulting small-request load is sustained without ever
// exceeding TRTLLM_KV_TRANSFER_NUM_THREADS active requests per sender.
void expectLargeChunkProgressWithBoundedSenderConcurrency(std::uint32_t numSenders, std::uint32_t threadsPerSender)
{
    struct WorkerSlot
    {
        std::uint32_t sender{};
        std::uint64_t nextRid{};
        std::string flow;
        std::optional<std::uint64_t> heldOffset;
    };

    std::uint32_t const numSmallFlows = numSenders * threadsPerSender;
    ASSERT_GT(numSmallFlows, 1U);
    // Both test instantiations use a power-of-two flow count, matching the buddy arena's full usable
    // capacity. One one-region request per worker slot fills it exactly.
    b::CreditScheduler s(kBase, static_cast<std::size_t>(numSmallFlows) * kRegion, kRegion,
        /*maxInflightChunksPerRequest=*/1);

    auto makeFlow = [](std::uint32_t sender, std::uint64_t rid)
    { return std::string("smallPeer") + std::to_string(sender) + '\x1f' + std::to_string(rid); };

    std::vector<WorkerSlot> slots;
    slots.reserve(numSmallFlows);
    std::unordered_map<std::string, std::size_t> flowToSlot;
    for (std::uint32_t sender = 0; sender < numSenders; ++sender)
    {
        for (std::uint32_t thread = 0; thread < threadsPerSender; ++thread)
        {
            WorkerSlot slot;
            slot.sender = sender;
            slot.nextRid = thread;
            slot.flow = makeFlow(sender, slot.nextRid);
            slot.nextRid += threadsPerSender;
            auto grants = s.onWant(slot.flow, want(1));
            ASSERT_EQ(grants.size(), 1U);
            slot.heldOffset = grants.front().offset;
            flowToSlot.emplace(slot.flow, slots.size());
            slots.push_back(std::move(slot));
        }
    }
    EXPECT_EQ(s.freeBytes(), 0U);

    std::string const largeFlow = std::string("largePeer\x1f") + "0";
    std::uint32_t const largeBytes = (numSmallFlows / 2) * kRegion;
    EXPECT_TRUE(s.onWant(largeFlow, {largeBytes}).empty());

    bool largeGranted = false;
    std::size_t nextSlot = 0;
    std::size_t smallCompletions = 0;
    std::size_t const kMaxCompletions = static_cast<std::size_t>(numSmallFlows) * 32;
    auto recordGrants = [&](std::vector<b::Grant> const& grants)
    {
        for (auto const& grant : grants)
        {
            if (grant.flow == largeFlow)
            {
                EXPECT_EQ(grant.len, largeBytes);
                largeGranted = true;
                continue;
            }
            auto const it = flowToSlot.find(grant.flow);
            ASSERT_NE(it, flowToSlot.end());
            slots[it->second].heldOffset = grant.offset;
        }
    };

    while (!largeGranted && smallCompletions < kMaxCompletions)
    {
        std::size_t checked = 0;
        while (checked < slots.size() && !slots[nextSlot].heldOffset)
        {
            nextSlot = (nextSlot + 1) % slots.size();
            ++checked;
        }
        ASSERT_LT(checked, slots.size()) << "drain stopped every in-flight small request before the large grant";

        auto& slot = slots[nextSlot];
        std::string const completedFlow = slot.flow;
        std::uint64_t const completedOffset = *slot.heldOffset;
        slot.heldOffset.reset();
        flowToSlot.erase(completedFlow);
        recordGrants(s.onScatterDone(completedFlow, completedOffset));
        ++smallCompletions;
        if (largeGranted)
        {
            break;
        }

        // The synchronous Python worker can submit exactly one replacement after its old request
        // completes. If receiver drain is active this new flow remains pending instead of refilling
        // the just-freed region, allowing a large buddy block to coalesce.
        slot.flow = makeFlow(slot.sender, slot.nextRid);
        slot.nextRid += threadsPerSender;
        flowToSlot.emplace(slot.flow, nextSlot);
        recordGrants(s.onWant(slot.flow, want(1)));
        nextSlot = (nextSlot + 1) % slots.size();
    }

    EXPECT_TRUE(largeGranted) << "sustained bounded-concurrency small requests starved the large chunk after "
                              << smallCompletions << " completions";
}

// Conservation: every region is either free, held by some flow, or locally held. With N equal
// regions this is the byte-budget invariant that, in v1, broke as a double-free on peer loss.
void checkConservation(b::CreditScheduler& s, std::vector<std::string> const& flows, std::uint32_t nRegions)
{
    std::size_t held = 0;
    for (auto const& f : flows)
    {
        held += s.heldCount(f);
    }
    EXPECT_EQ(freeRegions(s) + held + s.localHeldCount(), nRegions) << "region conservation violated";
}

// Track who-holds-what from the grant/return stream so we can assert no region is ever double-
// granted (held by two flows at once), and that grants carry a consistent addr/len.
struct Mirror
{
    std::map<std::uint64_t, std::string> owner; // offset -> flow

    void grant(std::vector<b::Grant> const& gs)
    {
        for (auto const& g : gs)
        {
            EXPECT_EQ(owner.count(g.offset), 0u) << "region " << g.offset << " granted while still held";
            owner[g.offset] = g.flow;
            EXPECT_EQ(g.addr, kBase + g.offset) << "grant carried wrong addr";
            EXPECT_EQ(g.len, kRegion) << "grant carried wrong region length";
        }
    }

    void free(std::uint64_t offset)
    {
        owner.erase(offset);
    }
};

} // namespace

// One WANT against an (arena, per-request cap) pair grants exactly min(cap, nRegions): the
// per-request limit binds even with arena room to spare, a large limit fills the arena exactly
// once, and a want far beyond the arena neither over-grants nor loops.
TEST(CreditScheduler, SingleWantGrantsMinOfCapAndArena)
{
    struct Case
    {
        std::uint32_t nRegions;
        std::uint32_t cap;
        std::uint32_t wantChunks;
        std::size_t expectGrants;
    };

    for (auto const& tc : {
             Case{8, 4, 100, 4},   // capped by the per-request limit
             Case{8, 16, 100, 8},  // large limit -> fills the arena
             Case{8, 16, 2000, 8}, // huge want -> exactly N, no more, no hang
             Case{8, 2, 100, 2},   // limit binds even with arena room
         })
    {
        auto s = makeSched(tc.nRegions, tc.cap);
        auto g = s.onWant("A", want(tc.wantChunks));
        std::string const ctx = " nRegions=" + std::to_string(tc.nRegions) + " cap=" + std::to_string(tc.cap)
            + " want=" + std::to_string(tc.wantChunks);
        EXPECT_EQ(g.size(), tc.expectGrants) << ctx;
        EXPECT_EQ(s.heldCount("A"), tc.expectGrants) << ctx;
        EXPECT_EQ(freeRegions(s), tc.nRegions - tc.expectGrants) << ctx;
        checkConservation(s, {"A"}, tc.nRegions);
    }
}

TEST(CreditScheduler, RecyclingOnScatterDone)
{
    auto s = makeSched(/*nRegions=*/4, /*maxInflightChunksPerRequest=*/16);
    Mirror m;
    m.grant(s.onWant("A", want(10))); // K=10 > N=4 -> gets 4
    EXPECT_EQ(s.heldCount("A"), 4u);
    // Complete one chunk: its region frees and is immediately re-granted (remaining still > 0).
    auto firstOff = m.owner.begin()->first;
    m.free(firstOff);
    auto re = s.onScatterDone("A", firstOff);
    m.grant(re);
    EXPECT_EQ(re.size(), 1u);
    EXPECT_EQ(s.heldCount("A"), 4u); // in-flight allocation count stays at its cap
    checkConservation(s, {"A"}, 4);
}

TEST(CreditScheduler, BoundedInflightLimitGivesFairSplit)
{
    // With a per-request limit of four on an eight-region arena, two senders split it evenly.
    auto s = makeSched(/*nRegions=*/8, /*maxInflightChunksPerRequest=*/4);
    Mirror m;
    m.grant(s.onWant("A", want(100)));
    m.grant(s.onWant("B", want(100)));
    EXPECT_EQ(s.heldCount("A"), 4u);
    EXPECT_EQ(s.heldCount("B"), 4u);
    EXPECT_EQ(freeRegions(s), 0u);
    checkConservation(s, {"A", "B"}, 8);
}

TEST(CreditScheduler, MoreSendersThanRegionsNoStarvation)
{
    auto s = makeSched(/*nRegions=*/2, /*maxInflightChunksPerRequest=*/16);
    Mirror m;
    m.grant(s.onWant("A", want(5)));
    m.grant(s.onWant("B", want(5)));
    m.grant(s.onWant("C", want(5)));
    EXPECT_EQ(s.heldCount("A") + s.heldCount("B") + s.heldCount("C"), 2u);
    EXPECT_EQ(freeRegions(s), 0u);

    // Recycling regions must serve the starved flows in turn — no flow is permanently starved.
    std::set<std::string> served;
    for (auto const& p : {"A", "B", "C"})
    {
        if (s.heldCount(p) > 0)
        {
            served.insert(p);
        }
    }
    int guard = 0;
    while (served.size() < 3 && guard++ < 50)
    {
        ASSERT_FALSE(m.owner.empty()) << "no region is held; cannot drive recycling";
        auto off = m.owner.begin()->first;
        auto flow = m.owner.begin()->second;
        m.free(off);
        m.grant(s.onScatterDone(flow, off));
        for (auto const& p : {"A", "B", "C"})
        {
            if (s.heldCount(p) > 0)
            {
                served.insert(p);
            }
        }
        checkConservation(s, {"A", "B", "C"}, 2);
    }
    EXPECT_EQ(served.size(), 3u) << "some sender was permanently starved";
}

TEST(CreditScheduler, PeerGoneReclaimsNoDoubleFree)
{
    auto s = makeSched(/*nRegions=*/4, /*maxInflightChunksPerRequest=*/16);
    (void) s.onWant("A", want(2));
    (void) s.onWant("B", want(2));
    checkConservation(s, {"A", "B"}, 4);
    std::vector<std::uint64_t> deferred;
    (void) s.reclaimFlow("A", {}, deferred); // A's held regions return to the arena (none busy)
    checkConservation(s, {"A", "B"}, 4);
    EXPECT_EQ(s.heldCount("A"), 0u);
    EXPECT_EQ(freeRegions(s) + s.heldCount("B"), 4u);
}

TEST(CreditScheduler, PeerGoneMidRecycleHandsRegionsToWaiter)
{
    // A holds the whole arena; B is waiting. A disappears mid-flight -> its regions must be reclaimed
    // AND immediately re-granted to the starved waiter B, with no double-free and no region lost.
    auto s = makeSched(/*nRegions=*/4, /*maxInflightChunksPerRequest=*/16);
    Mirror m;
    m.grant(s.onWant("A", want(100))); // A grabs all 4
    EXPECT_EQ(s.heldCount("A"), 4u);
    m.grant(s.onWant("B", want(100))); // B wants 4 but nothing free yet
    EXPECT_EQ(s.heldCount("B"), 0u);

    std::vector<std::uint64_t> aOffsets;
    for (auto const& [off, owner] : m.owner)
    {
        if (owner == "A")
        {
            aOffsets.push_back(off);
        }
    }
    for (auto off : aOffsets)
    {
        m.free(off); // mirror: A no longer owns these
    }
    std::vector<std::uint64_t> deferred;
    auto re = s.reclaimFlow("A", {}, deferred);
    m.grant(re);
    EXPECT_EQ(s.heldCount("A"), 0u);
    EXPECT_EQ(s.heldCount("B"), 4u) << "reclaimed regions did not flow to the waiting sender";
    EXPECT_EQ(freeRegions(s), 0u);
    checkConservation(s, {"A", "B"}, 4);
}

TEST(CreditScheduler, ReclaimByPrefixDropsAllFlowsOfPeer)
{
    // Transport keys flows as "peer\x1f rid". reclaimByPrefix("p1\x1f") must drop EVERY p1 flow
    // (multiple concurrent requests from one peer) and hand the freed regions to an unrelated peer.
    constexpr char sep = '\x1f';
    std::string const p1a = std::string("p1") + sep + "1";
    std::string const p1b = std::string("p1") + sep + "2";
    std::string const p2 = std::string("p2") + sep + "1";
    auto s = makeSched(/*nRegions=*/4, /*maxInflightChunksPerRequest=*/16);
    Mirror m;
    m.grant(s.onWant(p1a, want(2)));
    m.grant(s.onWant(p1b, want(2))); // p1 now holds all 4 across two flows
    m.grant(s.onWant(p2, want(4)));  // p2 waits (nothing free)
    EXPECT_EQ(s.heldCount(p1a) + s.heldCount(p1b), 4u);
    EXPECT_EQ(s.heldCount(p2), 0u);

    std::vector<std::uint64_t> p1Offsets;
    for (auto const& [off, owner] : m.owner)
    {
        if (owner == p1a || owner == p1b)
        {
            p1Offsets.push_back(off);
        }
    }
    for (auto off : p1Offsets)
    {
        m.free(off);
    }
    std::vector<std::uint64_t> deferred;
    auto re = s.reclaimByPrefix(std::string("p1") + sep, {}, deferred);
    m.grant(re);
    EXPECT_EQ(s.heldCount(p1a), 0u);
    EXPECT_EQ(s.heldCount(p1b), 0u);
    EXPECT_EQ(s.heldCount(p2), 4u) << "freed regions did not go to the surviving peer";
    checkConservation(s, {p1a, p1b, p2}, 4);

    // A non-matching prefix is a no-op.
    auto none = s.reclaimByPrefix(std::string("nomatch") + sep, {}, deferred);
    EXPECT_TRUE(none.empty());
    EXPECT_EQ(s.heldCount(p2), 4u);

    // An EMPTY prefix must NOT reclaim everything (compare(0,0,"")==0 matches all keys) — guarded.
    auto empty = s.reclaimByPrefix(std::string(), {}, deferred);
    EXPECT_TRUE(empty.empty());
    EXPECT_EQ(s.heldCount(p2), 4u) << "empty prefix wrongly reclaimed a live flow";
}

TEST(CreditScheduler, CompletedFlowsReclaimedNoTombstoneLeak)
{
    // The transport keys each request as "peer\x1f rid" with a monotonic, never-reused rid.
    // A long-running server runs many flows; completed flows MUST be reclaimed, else mFlows/mOrder
    // grow without bound and schedule() degrades to O(historical requests).
    auto s = makeSched(/*nRegions=*/4, /*maxInflightChunksPerRequest=*/16);
    for (int rid = 0; rid < 1000; ++rid)
    {
        std::string const key = std::string("peerA\x1f") + std::to_string(rid);
        auto g = s.onWant(key, want(2));
        ASSERT_EQ(g.size(), 2u);
        (void) s.onScatterDone(key, g[0].offset); // one still held -> flow kept
        EXPECT_EQ(s.trackedFlows(), 1u);
        (void) s.onScatterDone(key, g[1].offset); // last held region drains -> flow reclaimed
        EXPECT_EQ(s.trackedFlows(), 0u) << "completed flow left a tombstone at rid=" << rid;
    }
    EXPECT_EQ(s.trackedFlows(), 0u); // bounded by in-flight flows, NOT lifetime request count
    EXPECT_EQ(freeRegions(s), 4u);
}

TEST(CreditScheduler, LastFlowLeavingRingResetsCursorAndRingRefills)
{
    // Regression: dropFromRing used to run `mCursor %= mRing.size()` AFTER erasing the LAST ring
    // element -> modulo by zero (UB; a deterministic SIGFPE in -O0 builds). Any normal completion
    // of the only active flow hits it. Drain a single flow to empty the ring, then refill and
    // verify scheduling still rotates fairly from a sane cursor.
    auto s = makeSched(/*nRegions=*/4, /*maxInflightChunksPerRequest=*/16);
    std::string const k1 = std::string("p\x1f") + "1";
    auto g = s.onWant(k1, want(1));
    ASSERT_EQ(g.size(), 1u);
    (void) s.onScatterDone(k1, g[0].offset); // last held drains -> flow erased -> ring now EMPTY
    EXPECT_EQ(s.trackedFlows(), 0u);
    EXPECT_EQ(freeRegions(s), 4u);

    // Ring refills after going empty; grants keep alternating (cursor was reset, not left dangling).
    std::string const k2 = std::string("p\x1f") + "2";
    std::string const k3 = std::string("p\x1f") + "3";
    auto g2 = s.onWant(k2, want(2));
    auto g3 = s.onWant(k3, want(2));
    ASSERT_EQ(g2.size() + g3.size(), 4u);
    Mirror m;
    m.grant(g2);
    m.grant(g3); // no double-grant across the empty->refill transition
    for (auto const& x : g2)
    {
        (void) s.onScatterDone(k2, x.offset);
    }
    for (auto const& x : g3)
    {
        (void) s.onScatterDone(k3, x.offset);
    }
    EXPECT_EQ(s.trackedFlows(), 0u); // BOTH flows drain -> ring empties again (erase-behind-cursor path)
    EXPECT_EQ(freeRegions(s), 4u);
}

TEST(CreditScheduler, CancelledFlowReclaimed)
{
    auto s = makeSched(/*nRegions=*/4, /*maxInflightChunksPerRequest=*/16);
    std::string const k1 = std::string("p\x1f") + "1";
    std::string const k2 = std::string("p\x1f") + "2";
    // Empty WANT with nothing in flight -> no tombstone created.
    (void) s.onWant(k1, want(0));
    EXPECT_EQ(s.trackedFlows(), 0u);
    // Grant, then cancel while regions are still held -> kept until they drain, then reclaimed.
    auto g = s.onWant(k2, want(2));
    ASSERT_EQ(g.size(), 2u);
    (void) s.onWant(k2, want(0));    // cancel; 2 regions still in flight
    EXPECT_EQ(s.trackedFlows(), 1u); // must stay until held drains (regions still leased)
    (void) s.onScatterDone(k2, g[0].offset);
    (void) s.onScatterDone(k2, g[1].offset);
    EXPECT_EQ(s.trackedFlows(), 0u); // reclaimed after the last held region returns
    EXPECT_EQ(freeRegions(s), 4u);
}

TEST(CreditScheduler, ReclaimDefersBusyRegionThenFreeOrphan)
{
    // A region whose scatter is still running on the receiver must NOT be freed/re-granted when the
    // peer is reclaimed (forgetPeer) — else another sender's RDMA write races the worker's read.
    // reclaimByPrefix defers such "busy" regions; freeOrphanRegion releases them once scatter is done.
    auto s = makeSched(/*nRegions=*/2, /*maxInflightChunksPerRequest=*/16);
    std::string const keyA = std::string("A\x1f") + "1";
    std::string const prefixA = std::string("A\x1f");
    std::string const keyB = std::string("B\x1f") + "1";

    auto g = s.onWant(keyA, want(2));
    ASSERT_EQ(g.size(), 2u); // A holds both regions
    std::uint64_t const busyOff = g[0].offset;
    std::uint64_t const idleOff = g[1].offset;

    std::unordered_set<std::uint64_t> busy{busyOff}; // a scatter is still reading busyOff
    std::vector<std::uint64_t> deferred;
    auto re = s.reclaimByPrefix(prefixA, busy, deferred);
    EXPECT_TRUE(re.empty()); // no waiter yet
    ASSERT_EQ(deferred.size(), 1u);
    EXPECT_EQ(deferred[0], busyOff);
    EXPECT_EQ(s.trackedFlows(), 0u); // flow A erased
    EXPECT_EQ(freeRegions(s), 1u);   // only the idle region is free; busy one is in limbo

    // A new sender B wanting 2 may take only the idle region — NEVER the busy (in-limbo) one.
    auto gb = s.onWant(keyB, want(2));
    ASSERT_EQ(gb.size(), 1u);
    EXPECT_EQ(gb[0].offset, idleOff);
    EXPECT_EQ(freeRegions(s), 0u);

    // Scatter finishes -> free the orphan -> now B can finally get it.
    auto re2 = s.freeOrphanRegion(busyOff);
    ASSERT_EQ(re2.size(), 1u);
    EXPECT_EQ(re2[0].offset, busyOff);
    EXPECT_EQ(s.heldCount(keyB), 2u);
    checkConservation(s, {keyA, keyB}, 2);

    // A second freeOrphanRegion for the same (now non-orphan) offset is a no-op — must NOT free
    // busyOff again, which is now live under keyB.
    auto re3 = s.freeOrphanRegion(busyOff);
    EXPECT_TRUE(re3.empty());
    EXPECT_EQ(s.heldCount(keyB), 2u);
    checkConservation(s, {keyA, keyB}, 2);
}

TEST(CreditScheduler, FreeOrphanRegionIgnoresNonOrphan)
{
    // freeOrphanRegion must only free regions actually deferred as orphans; a call for a LIVE,
    // never-deferred region is a no-op (defense in depth against a stray caller).
    auto s = makeSched(/*nRegions=*/2, /*maxInflightChunksPerRequest=*/16);
    std::string const flow = std::string("A\x1f") + "1";
    auto g = s.onWant(flow, want(1));
    ASSERT_EQ(g.size(), 1u);
    auto const live = g[0].offset; // a live region, never reclaim-deferred
    EXPECT_EQ(s.heldCount(flow), 1u);

    auto re = s.freeOrphanRegion(live);
    EXPECT_TRUE(re.empty());
    EXPECT_EQ(s.heldCount(flow), 1u);   // still held — NOT freed
    EXPECT_EQ(freeRegions(s), 1u);      // unchanged

    (void) s.onScatterDone(flow, live); // normal path still frees it exactly once
    EXPECT_EQ(freeRegions(s), 2u);
    EXPECT_EQ(s.heldCount(flow), 0u);
}

TEST(CreditScheduler, ReclaimFlowFreesHeldDefersBusyAndHeldByFlow)
{
    // Explicit cancel of one flow (empty WANT path): free its granted-but-unwritten regions now,
    // defer any whose scatter is still running; heldByFlow lets the transport drop late DATA.
    auto s = makeSched(/*nRegions=*/4, /*maxInflightChunksPerRequest=*/16);
    std::string const flow = std::string("p\x1f") + "1";
    auto g = s.onWant(flow, want(3));
    ASSERT_EQ(g.size(), 3u); // 3 held, 1 free

    EXPECT_TRUE(s.heldByFlow(flow, g[0].offset));
    EXPECT_FALSE(s.heldByFlow(flow, 0x999999)); // not a held offset
    EXPECT_FALSE(
        s.heldByFlow("p\x1f"
                     "2",
            g[0].offset));                               // held by a different flow key

    std::unordered_set<std::uint64_t> busy{g[1].offset}; // a scatter is still reading g[1]
    std::vector<std::uint64_t> deferred;
    auto re = s.reclaimFlow(flow, busy, deferred);
    EXPECT_TRUE(re.empty()); // no other flow waiting
    EXPECT_EQ(s.trackedFlows(), 0u);
    ASSERT_EQ(deferred.size(), 1u);
    EXPECT_EQ(deferred[0], g[1].offset);
    EXPECT_FALSE(s.heldByFlow(flow, g[0].offset)); // flow erased
    EXPECT_EQ(freeRegions(s), 3u);                 // g0,g2 freed + the originally-free one; g1 in limbo

    auto re2 = s.freeOrphanRegion(g[1].offset);    // scatter finished -> free the deferred one
    EXPECT_EQ(freeRegions(s), 4u);

    // Cancelling an unknown flow is a harmless no-op.
    std::vector<std::uint64_t> none;
    EXPECT_TRUE(s.reclaimFlow("nope", {}, none).empty());
    EXPECT_TRUE(none.empty());
}

// Scheduler over `nRegions` regions with a controllable fake clock: tests advance *now instead of
// sleeping, keeping the lease/quarantine tests deterministic and instant.
b::CreditScheduler makeTimedSched(
    std::uint32_t nRegions, std::uint32_t maxInflight, std::shared_ptr<std::chrono::steady_clock::time_point> now)
{
    return b::CreditScheduler(
        kBase, static_cast<std::size_t>(nRegions) * kRegion, kRegion, maxInflight, [now] { return *now; });
}

TEST(CreditScheduler, ReceiverReclaimQuarantinesUnwrittenRegionsUntilReaped)
{
    // Receiver-initiated reclaim (peer loss / lease expiry): granted-but-unwritten regions may STILL
    // be RDMA-written by the peer, so quarantineFor > 0 must keep them out of the arena — not free
    // them — until reapQuarantine() passes the time barrier.
    auto now = std::make_shared<std::chrono::steady_clock::time_point>(std::chrono::steady_clock::now());
    auto s = makeTimedSched(/*nRegions=*/2, /*maxInflight=*/16, now);
    std::string const flowA = std::string("A\x1f") + "1";
    std::string const flowB = std::string("B\x1f") + "1";
    auto g = s.onWant(flowA, want(2));
    ASSERT_EQ(g.size(), 2u);

    std::vector<std::uint64_t> deferred;
    auto re = s.reclaimFlow(flowA, /*busy=*/{}, deferred, std::chrono::seconds(30));
    EXPECT_TRUE(re.empty());                        // no waiter yet
    EXPECT_TRUE(deferred.empty());
    EXPECT_EQ(s.trackedFlows(), 0u);                // flow gone...
    EXPECT_FALSE(s.heldByFlow(flowA, g[0].offset)); // ...so late DATA gets dropped
    EXPECT_EQ(freeRegions(s), 0u);                  // ...but NOTHING went back to the arena

    // A new flow must not be granted a quarantined region, and reaping early frees nothing.
    EXPECT_TRUE(s.onWant(flowB, want(1)).empty());
    EXPECT_TRUE(s.reapQuarantine().empty());
    EXPECT_EQ(freeRegions(s), 0u);

    // Quarantine over -> the regions re-enter circulation; one serves the waiter.
    *now += std::chrono::seconds(31);
    auto re2 = s.reapQuarantine();
    ASSERT_EQ(re2.size(), 1u);
    EXPECT_EQ(re2[0].flow, flowB);
    EXPECT_EQ(s.heldCount(flowB), 1u);
    EXPECT_EQ(freeRegions(s), 1u);

    // Reaping again is a harmless no-op (the reaped offsets are live/free now, not re-freed).
    EXPECT_TRUE(s.reapQuarantine().empty());
    EXPECT_EQ(freeRegions(s), 1u);
}

TEST(CreditScheduler, ReceiverReclaimBusyRegionsDeferAsOrphansNotQuarantine)
{
    // Busy takes precedence: a region a scatter worker still READS follows the orphan path
    // (freeOrphanRegion on scatter completion), never the quarantine path — reapQuarantine must not
    // free it even after the quarantine window.
    auto now = std::make_shared<std::chrono::steady_clock::time_point>(std::chrono::steady_clock::now());
    auto s = makeTimedSched(/*nRegions=*/2, /*maxInflight=*/16, now);
    std::string const flow = std::string("A\x1f") + "1";
    auto g = s.onWant(flow, want(2));
    ASSERT_EQ(g.size(), 2u);

    std::unordered_set<std::uint64_t> busy{g[0].offset};
    std::vector<std::uint64_t> deferred;
    (void) s.reclaimFlow(flow, busy, deferred, std::chrono::seconds(30));
    ASSERT_EQ(deferred.size(), 1u);
    EXPECT_EQ(deferred[0], g[0].offset);

    *now += std::chrono::seconds(31);
    EXPECT_TRUE(s.reapQuarantine().empty());
    EXPECT_EQ(freeRegions(s), 1u); // only the quarantined region came back; the orphan is still busy

    (void) s.freeOrphanRegion(g[0].offset);
    EXPECT_EQ(freeRegions(s), 2u);
}

TEST(CreditScheduler, StaleFlowsReportsIdleFlowsOnly)
{
    // The lease that makes a dead sender observable: a flow with no progress (WANT / grant issued /
    // scatter completed) beyond the idle limit is reported stale; any progress renews the lease.
    auto now = std::make_shared<std::chrono::steady_clock::time_point>(std::chrono::steady_clock::now());
    auto s = makeTimedSched(/*nRegions=*/2, /*maxInflight=*/1, now);
    std::string const dead = std::string("A\x1f") + "1";
    std::string const live = std::string("B\x1f") + "1";
    auto ga = s.onWant(dead, want(1));
    ASSERT_EQ(ga.size(), 1u);
    auto gb = s.onWant(live, want(2)); // one granted, one pending (in-flight cap 1)
    ASSERT_EQ(gb.size(), 1u);

    // t+40s: `live` completes its first chunk -> frees the region + gets its next grant = progress.
    *now += std::chrono::seconds(40);
    ASSERT_EQ(s.onScatterDone(live, gb[0].offset).size(), 1u);

    // t+70s: `dead` has been silent for 70s -> stale; `live` progressed 30s ago -> not stale.
    *now += std::chrono::seconds(30);
    auto stale = s.staleFlows(std::chrono::seconds(60));
    ASSERT_EQ(stale.size(), 1u);
    EXPECT_EQ(stale[0], dead);
    EXPECT_TRUE(s.staleFlows(std::chrono::seconds(90)).empty()); // longer lease: nobody is stale yet

    // Reclaim + quarantine + reap: the arena fully recovers the dead flow's region.
    std::vector<std::uint64_t> deferred;
    (void) s.reclaimFlow(dead, /*busy=*/{}, deferred, std::chrono::seconds(30));
    EXPECT_EQ(freeRegions(s), 0u); // quarantined, not yet back
    *now += std::chrono::seconds(31);
    (void) s.reapQuarantine();
    EXPECT_EQ(freeRegions(s), 1u); // recovered (live still holds its in-flight second chunk)
}

TEST(CreditScheduler, SharedArenaLocalAndRemoteShareOneAllocator)
{
    // Shared single arena: the local sender (gather staging, acquireLocal) and remote flows (grants)
    // draw from the SAME allocator. Conservation holds across {free, remote-held, local-held}; a
    // released local region flows to a waiting remote flow.
    auto s = makeSched(/*nRegions=*/4, /*maxInflightChunksPerRequest=*/16);
    std::string const flow = std::string("p\x1f") + "1";

    auto a0 = s.acquireLocal(kRegion);
    auto a1 = s.acquireLocal(kRegion);
    ASSERT_TRUE(a0.has_value() && a1.has_value());
    EXPECT_EQ(s.localHeldCount(), 2u);
    EXPECT_EQ(freeRegions(s), 2u);

    // Remote flow wants 4 but only 2 are free now (local holds 2) -> gets 2.
    auto g = s.onWant(flow, want(4));
    EXPECT_EQ(g.size(), 2u);
    EXPECT_EQ(s.heldCount(flow), 2u);
    EXPECT_EQ(freeRegions(s), 0u);
    checkConservation(s, {flow}, 4);

    // No region free -> acquireLocal returns nullopt (caller would park and retry).
    EXPECT_FALSE(s.acquireLocal(kRegion).has_value());

    // Release one local region -> the remote flow (still wants 2 more) immediately grabs it.
    auto re = s.releaseLocal(*a0);
    EXPECT_EQ(re.size(), 1u);
    EXPECT_EQ(s.heldCount(flow), 3u);
    EXPECT_EQ(s.localHeldCount(), 1u);
    checkConservation(s, {flow}, 4);

    // Release the last local region -> remote flow reaches its wanted 4.
    (void) s.releaseLocal(*a1);
    EXPECT_EQ(s.heldCount(flow), 4u);
    EXPECT_EQ(s.localHeldCount(), 0u);
    EXPECT_EQ(freeRegions(s), 0u);
    checkConservation(s, {flow}, 4);

    // releaseLocal on a region that isn't locally held is an idempotent no-op.
    auto none = s.releaseLocal(*a0);
    EXPECT_TRUE(none.empty());
    checkConservation(s, {flow}, 4);
}

TEST(CreditScheduler, ScatterDoneIdempotentForUnknownRegion)
{
    auto s = makeSched(/*nRegions=*/4, /*maxInflightChunksPerRequest=*/16);
    (void) s.onWant("A", want(1));            // A holds 1, 3 free
    auto before = freeRegions(s);
    auto re = s.onScatterDone("A", 0x999000); // an offset A never held
    EXPECT_TRUE(re.empty());
    EXPECT_EQ(freeRegions(s), before);        // no spurious free
    checkConservation(s, {"A"}, 4);
}

TEST(CreditScheduler, WantEmptyStopsGranting)
{
    auto s = makeSched(/*nRegions=*/8, /*maxInflightChunksPerRequest=*/16);
    (void) s.onWant("A", want(4));
    EXPECT_EQ(s.heldCount("A"), 4u);
    auto g = s.onWant("A", want(0)); // cancel further grants
    EXPECT_TRUE(g.empty());
    EXPECT_EQ(s.activeFlows(), 0u);
}

TEST(CreditScheduler, VariableSizeRegionsPackAndBackpressure)
{
    // The genuinely variable case: a flow's chunks have different byte sizes, each granted a region
    // of exactly that size, packed densely. A chunk that cannot fit right now is parked (no grant)
    // while smaller following chunks are NOT reordered ahead of it (FIFO per flow).
    constexpr std::size_t kArena = 8 * kRegion; // 32 KiB, min block kRegion
    b::CreditScheduler s(kBase, kArena, kRegion, /*maxInflightChunksPerRequest=*/16);

    // Chunk sizes: 1, 2, 4, 1 regions (total 8 -> exactly fills the arena).
    std::vector<std::uint32_t> sizes{kRegion, 2 * kRegion, 4 * kRegion, kRegion};
    auto g = s.onWant("A", sizes);
    ASSERT_EQ(g.size(), 4u);
    EXPECT_EQ(g[0].len, kRegion);
    EXPECT_EQ(g[1].len, 2 * kRegion);
    EXPECT_EQ(g[2].len, 4 * kRegion);
    EXPECT_EQ(g[3].len, kRegion);
    // Distinct, non-overlapping regions.
    for (std::size_t i = 0; i < g.size(); ++i)
    {
        for (std::size_t j = i + 1; j < g.size(); ++j)
        {
            bool const disjoint = g[i].offset + g[i].len <= g[j].offset || g[j].offset + g[j].len <= g[i].offset;
            EXPECT_TRUE(disjoint) << "regions " << i << " and " << j << " overlap";
        }
    }
    EXPECT_EQ(s.freeBytes(), 0u); // arena exactly full

    // Free the 4-region chunk; a fresh flow wanting a 4-region chunk can now be granted.
    (void) s.onScatterDone("A", g[2].offset);
    auto g2 = s.onWant("B", std::vector<std::uint32_t>{4 * kRegion});
    ASSERT_EQ(g2.size(), 1u);
    EXPECT_EQ(g2[0].len, 4 * kRegion);
}

TEST(CreditScheduler, ManySingleThreadSendersCannotStarveLargeChunk)
{
    expectLargeChunkProgressWithBoundedSenderConcurrency(/*numSenders=*/8, /*threadsPerSender=*/1);
}

TEST(CreditScheduler, FourThreadsPerSenderCannotStarveLargeChunk)
{
    expectLargeChunkProgressWithBoundedSenderConcurrency(/*numSenders=*/4, /*threadsPerSender=*/4);
}
