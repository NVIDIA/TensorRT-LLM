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

#include <map>
#include <set>
#include <string>
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

// Scheduler over an arena holding exactly `nRegions` regions of kRegion bytes, per-flow cap `window`.
b::CreditScheduler makeSched(std::uint32_t nRegions, std::uint32_t window)
{
    return b::CreditScheduler(kBase, static_cast<std::size_t>(nRegions) * kRegion, kRegion, window);
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

TEST(CreditScheduler, SingleSenderGetsWindow)
{
    auto s = makeSched(/*nRegions=*/8, /*window=*/4);
    auto g = s.onWant("A", want(100)); // wants a lot
    EXPECT_EQ(g.size(), 4u);           // capped by the per-flow window
    EXPECT_EQ(s.heldCount("A"), 4u);
    EXPECT_EQ(freeRegions(s), 4u);
    checkConservation(s, {"A"}, 8);
}

TEST(CreditScheduler, SingleSenderFillsArenaWhenWindowLarge)
{
    auto s = makeSched(/*nRegions=*/8, /*window=*/16); // window > N -> arena is the only bound
    auto g = s.onWant("A", want(100));
    EXPECT_EQ(g.size(), 8u);
    EXPECT_EQ(freeRegions(s), 0u);
    checkConservation(s, {"A"}, 8);
}

TEST(CreditScheduler, RecyclingOnScatterDone)
{
    auto s = makeSched(/*nRegions=*/4, /*window=*/16);
    Mirror m;
    m.grant(s.onWant("A", want(10))); // K=10 > N=4 -> gets 4
    EXPECT_EQ(s.heldCount("A"), 4u);
    // Complete one chunk: its region frees and is immediately re-granted (remaining still > 0).
    auto firstOff = m.owner.begin()->first;
    m.free(firstOff);
    auto re = s.onScatterDone("A", firstOff);
    m.grant(re);
    EXPECT_EQ(re.size(), 1u);
    EXPECT_EQ(s.heldCount("A"), 4u); // window stays full
    checkConservation(s, {"A"}, 4);
}

TEST(CreditScheduler, HugeWantNeverOverGrantsOrLoops)
{
    auto s = makeSched(/*nRegions=*/8, /*window=*/16);
    auto g = s.onWant("A", want(2000)); // far more than the arena can hold
    EXPECT_EQ(g.size(), 8u);            // exactly N, no more, no hang
    checkConservation(s, {"A"}, 8);
}

TEST(CreditScheduler, BoundedWindowGivesFairSplit)
{
    // With a bounded window (W=4) on an 8-region arena, two senders each cap at 4 -> instant fair.
    auto s = makeSched(/*nRegions=*/8, /*window=*/4);
    Mirror m;
    m.grant(s.onWant("A", want(100)));
    m.grant(s.onWant("B", want(100)));
    EXPECT_EQ(s.heldCount("A"), 4u);
    EXPECT_EQ(s.heldCount("B"), 4u);
    EXPECT_EQ(freeRegions(s), 0u);
    checkConservation(s, {"A", "B"}, 8);
}

TEST(CreditScheduler, WindowCapsPerFlowEvenWithArenaRoom)
{
    // The per-flow window bounds pipeline depth independently of arena capacity: a lone sender with
    // W=2 holds at most 2 regions even though 6 sit free.
    auto s = makeSched(/*nRegions=*/8, /*window=*/2);
    auto g = s.onWant("A", want(100));
    EXPECT_EQ(g.size(), 2u);
    EXPECT_EQ(s.heldCount("A"), 2u);
    EXPECT_EQ(freeRegions(s), 6u);
    checkConservation(s, {"A"}, 8);
}

TEST(CreditScheduler, MoreSendersThanRegionsNoStarvation)
{
    auto s = makeSched(/*nRegions=*/2, /*window=*/16); // N=2, 3 senders, arena is the bound
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
            served.insert(p);
    }
    int guard = 0;
    while (served.size() < 3 && guard++ < 50)
    {
        auto off = m.owner.begin()->first;
        auto flow = m.owner.begin()->second;
        m.free(off);
        m.grant(s.onScatterDone(flow, off));
        for (auto const& p : {"A", "B", "C"})
        {
            if (s.heldCount(p) > 0)
                served.insert(p);
        }
        checkConservation(s, {"A", "B", "C"}, 2);
    }
    EXPECT_EQ(served.size(), 3u) << "some sender was permanently starved";
}

TEST(CreditScheduler, PeerGoneReclaimsNoDoubleFree)
{
    auto s = makeSched(/*nRegions=*/4, /*window=*/16);
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
    auto s = makeSched(/*nRegions=*/4, /*window=*/16);
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
    auto s = makeSched(/*nRegions=*/4, /*window=*/16);
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
    auto s = makeSched(/*nRegions=*/4, /*window=*/16);
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

TEST(CreditScheduler, CancelledFlowReclaimed)
{
    auto s = makeSched(/*nRegions=*/4, /*window=*/16);
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
    auto s = makeSched(/*nRegions=*/2, /*window=*/16);
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
    auto s = makeSched(/*nRegions=*/2, /*window=*/16);
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
    auto s = makeSched(/*nRegions=*/4, /*window=*/16);
    std::string const flow = std::string("p\x1f") + "1";
    auto g = s.onWant(flow, want(3));
    ASSERT_EQ(g.size(), 3u); // 3 held, 1 free

    EXPECT_TRUE(s.heldByFlow(flow, g[0].offset));
    EXPECT_FALSE(s.heldByFlow(flow, 0x999999));          // not a held offset
    EXPECT_FALSE(s.heldByFlow("p\x1f2", g[0].offset));   // held by a different flow key

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

TEST(CreditScheduler, SharedArenaLocalAndRemoteShareOneAllocator)
{
    // Shared single arena: the local sender (gather staging, acquireLocal) and remote flows (grants)
    // draw from the SAME allocator. Conservation holds across {free, remote-held, local-held}; a
    // released local region flows to a waiting remote flow.
    auto s = makeSched(/*nRegions=*/4, /*window=*/16);
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
    auto s = makeSched(/*nRegions=*/4, /*window=*/16);
    (void) s.onWant("A", want(1));            // A holds 1, 3 free
    auto before = freeRegions(s);
    auto re = s.onScatterDone("A", 0x999000); // an offset A never held
    EXPECT_TRUE(re.empty());
    EXPECT_EQ(freeRegions(s), before);        // no spurious free
    checkConservation(s, {"A"}, 4);
}

TEST(CreditScheduler, WantEmptyStopsGranting)
{
    auto s = makeSched(/*nRegions=*/8, /*window=*/16);
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
    b::CreditScheduler s(kBase, kArena, kRegion, /*window=*/16);

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
