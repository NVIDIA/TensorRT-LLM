# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Credit scheduler tests: port of creditSchedulerTest.cpp.

Conservation invariant, double-grant Mirror tracker, round-robin fairness,
drain anti-starvation, lease/quarantine with an injected clock, region
recycling, shared local/remote arena, plus the Python-only eager budget.
All time-dependent tests use an injected fake clock — no real sleeps.
"""

from __future__ import annotations

import threading
from typing import Optional, Sequence

import pytest
from conftest import load_bounce_v2

_b = load_bounce_v2()
CreditScheduler = _b.CreditScheduler
Grant = _b.Grant

# Most tests model an arena of N EQUAL regions by making the buddy min block
# == one region size, so each "want a region" is one minimum block and the
# arena holds exactly N of them.
K_BASE = 0x100000  # arena base device address (Grant.addr = K_BASE + offset)
K_REGION = 0x1000  # 4096B: one "slot"-sized region (== buddy min block)

SEP = "\x1f"


class FakeClock:
    """Injected monotonic clock; tests advance it instead of sleeping."""

    def __init__(self, start: float = 1_000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def make_sched(
    n_regions: int, max_inflight: int, now_fn: Optional[FakeClock] = None
) -> "CreditScheduler":
    kwargs = {} if now_fn is None else {"now_fn": now_fn}
    return CreditScheduler(K_BASE, n_regions * K_REGION, K_REGION, max_inflight, **kwargs)


def want(n: int) -> list[int]:
    """N equal chunks of K_REGION bytes (FIFO order)."""
    return [K_REGION] * n


def free_regions(s: "CreditScheduler") -> int:
    return s.free_bytes() // K_REGION


def check_conservation(s: "CreditScheduler", flows: Sequence[str], n_regions: int) -> None:
    """Every region is either free, held by some flow, or locally held."""
    held = sum(s.held_count(f) for f in flows)
    assert free_regions(s) + held + s.local_held_count() == n_regions, (
        "region conservation violated"
    )


class Mirror:
    """Track who-holds-what from the grant/return stream.

    No region may ever be double-granted, and grants must carry a consistent
    addr/len.
    """

    def __init__(self, expect_len: Optional[int] = K_REGION) -> None:
        self.owner: dict[int, str] = {}
        self._expect_len = expect_len

    def grant(self, grants: Sequence["Grant"]) -> None:
        for g in grants:
            assert g.offset not in self.owner, f"region {g.offset} granted while still held"
            self.owner[g.offset] = g.flow
            assert g.addr == K_BASE + g.offset, "grant carried wrong addr"
            if self._expect_len is not None:
                assert g.length == self._expect_len, "grant carried wrong region length"

    def free(self, offset: int) -> None:
        del self.owner[offset]


# C++: CreditScheduler.SingleWantGrantsMinOfCapAndArena (table-driven)
@pytest.mark.parametrize(
    "n_regions,cap,want_chunks,expect_grants",
    [
        (8, 4, 100, 4),  # capped by the per-request limit
        (8, 16, 100, 8),  # large limit -> fills the arena
        (8, 16, 2000, 8),  # huge want -> exactly N, no more, no hang
        (8, 2, 100, 2),  # limit binds even with arena room
    ],
)
def test_single_want_grants_min_of_cap_and_arena(
    n_regions: int, cap: int, want_chunks: int, expect_grants: int
) -> None:
    s = make_sched(n_regions, cap)
    g = s.on_want("A", want(want_chunks))
    assert len(g) == expect_grants
    assert s.held_count("A") == expect_grants
    assert free_regions(s) == n_regions - expect_grants
    check_conservation(s, ["A"], n_regions)


# C++: CreditScheduler.RecyclingOnScatterDone
def test_recycling_on_scatter_done() -> None:
    s = make_sched(4, 16)
    m = Mirror()
    m.grant(s.on_want("A", want(10)))  # K=10 > N=4 -> gets 4
    assert s.held_count("A") == 4
    # Complete one chunk: its region frees and is immediately re-granted.
    first_off = next(iter(m.owner))
    m.free(first_off)
    re = s.on_scatter_done("A", first_off)
    m.grant(re)
    assert len(re) == 1
    assert s.held_count("A") == 4  # in-flight count stays at its cap
    check_conservation(s, ["A"], 4)


# C++: CreditScheduler.BoundedInflightLimitGivesFairSplit
def test_bounded_inflight_limit_gives_fair_split() -> None:
    # Per-request limit 4 on an 8-region arena -> two senders split evenly.
    s = make_sched(8, 4)
    m = Mirror()
    m.grant(s.on_want("A", want(100)))
    m.grant(s.on_want("B", want(100)))
    assert s.held_count("A") == 4
    assert s.held_count("B") == 4
    assert free_regions(s) == 0
    check_conservation(s, ["A", "B"], 8)


# C++: CreditScheduler.MoreSendersThanRegionsNoStarvation
def test_more_senders_than_regions_no_starvation() -> None:
    s = make_sched(2, 16)
    m = Mirror()
    m.grant(s.on_want("A", want(5)))
    m.grant(s.on_want("B", want(5)))
    m.grant(s.on_want("C", want(5)))
    assert s.held_count("A") + s.held_count("B") + s.held_count("C") == 2
    assert free_regions(s) == 0

    # Recycling must serve the starved flows in turn — none starves forever.
    served = {p for p in ("A", "B", "C") if s.held_count(p) > 0}
    guard = 0
    while len(served) < 3 and guard < 50:
        guard += 1
        assert m.owner, "no region is held; cannot drive recycling"
        off, flow = next(iter(m.owner.items()))
        m.free(off)
        m.grant(s.on_scatter_done(flow, off))
        served.update(p for p in ("A", "B", "C") if s.held_count(p) > 0)
        check_conservation(s, ["A", "B", "C"], 2)
    assert len(served) == 3, "some sender was permanently starved"


# C++: CreditScheduler.PeerGoneReclaimsNoDoubleFree
def test_peer_gone_reclaims_no_double_free() -> None:
    s = make_sched(4, 16)
    s.on_want("A", want(2))
    s.on_want("B", want(2))
    check_conservation(s, ["A", "B"], 4)
    _grants, deferred = s.forget("A")  # A's regions return (none busy)
    assert deferred == []
    check_conservation(s, ["A", "B"], 4)
    assert s.held_count("A") == 0
    assert free_regions(s) + s.held_count("B") == 4


# C++: CreditScheduler.PeerGoneMidRecycleHandsRegionsToWaiter
def test_peer_gone_mid_recycle_hands_regions_to_waiter() -> None:
    s = make_sched(4, 16)
    m = Mirror()
    m.grant(s.on_want("A", want(100)))  # A grabs all 4
    assert s.held_count("A") == 4
    m.grant(s.on_want("B", want(100)))  # B wants 4 but nothing free yet
    assert s.held_count("B") == 0

    for off in [o for o, owner in m.owner.items() if owner == "A"]:
        m.free(off)  # mirror: A no longer owns these
    re, deferred = s.forget("A")
    assert deferred == []
    m.grant(re)
    assert s.held_count("A") == 0
    assert s.held_count("B") == 4, "reclaimed regions did not flow to the waiter"
    assert free_regions(s) == 0
    check_conservation(s, ["A", "B"], 4)


# C++: CreditScheduler.ReclaimByPrefixDropsAllFlowsOfPeer
def test_forget_prefix_drops_all_flows_of_peer() -> None:
    p1a = "p1" + SEP + "1"
    p1b = "p1" + SEP + "2"
    p2 = "p2" + SEP + "1"
    s = make_sched(4, 16)
    m = Mirror()
    m.grant(s.on_want(p1a, want(2)))
    m.grant(s.on_want(p1b, want(2)))  # p1 now holds all 4 across two flows
    m.grant(s.on_want(p2, want(4)))  # p2 waits (nothing free)
    assert s.held_count(p1a) + s.held_count(p1b) == 4
    assert s.held_count(p2) == 0

    for off in [o for o, owner in m.owner.items() if owner in (p1a, p1b)]:
        m.free(off)
    re, deferred = s.forget_prefix("p1" + SEP)
    assert deferred == []
    m.grant(re)
    assert s.held_count(p1a) == 0
    assert s.held_count(p1b) == 0
    assert s.held_count(p2) == 4, "freed regions did not go to the survivor"
    check_conservation(s, [p1a, p1b, p2], 4)

    # A non-matching prefix is a no-op.
    none, _ = s.forget_prefix("nomatch" + SEP)
    assert none == []
    assert s.held_count(p2) == 4

    # An EMPTY prefix must NOT reclaim everything — guarded.
    empty, _ = s.forget_prefix("")
    assert empty == []
    assert s.held_count(p2) == 4, "empty prefix wrongly reclaimed a live flow"


# C++: CreditScheduler.CompletedFlowsReclaimedNoTombstoneLeak
def test_completed_flows_reclaimed_no_tombstone_leak() -> None:
    s = make_sched(4, 16)
    for rid in range(1000):
        key = f"peerA{SEP}{rid}"
        g = s.on_want(key, want(2))
        assert len(g) == 2
        s.on_scatter_done(key, g[0].offset)  # one still held -> flow kept
        assert s.tracked_flows() == 1
        s.on_scatter_done(key, g[1].offset)  # last region drains -> reclaimed
        assert s.tracked_flows() == 0, f"tombstone left at rid={rid}"
    assert s.tracked_flows() == 0  # bounded by in-flight, NOT lifetime count
    assert free_regions(s) == 4


# C++: CreditScheduler.LastFlowLeavingRingResetsCursorAndRingRefills
def test_last_flow_leaving_ring_resets_cursor_and_ring_refills() -> None:
    s = make_sched(4, 16)
    k1 = "p" + SEP + "1"
    g = s.on_want(k1, want(1))
    assert len(g) == 1
    s.on_scatter_done(k1, g[0].offset)  # ring now EMPTY
    assert s.tracked_flows() == 0
    assert free_regions(s) == 4

    # Ring refills after going empty; grants keep alternating.
    k2 = "p" + SEP + "2"
    k3 = "p" + SEP + "3"
    g2 = s.on_want(k2, want(2))
    g3 = s.on_want(k3, want(2))
    assert len(g2) + len(g3) == 4
    m = Mirror()
    m.grant(g2)
    m.grant(g3)  # no double-grant across the empty->refill transition
    for x in g2:
        s.on_scatter_done(k2, x.offset)
    for x in g3:
        s.on_scatter_done(k3, x.offset)
    assert s.tracked_flows() == 0  # both drain -> ring empties again
    assert free_regions(s) == 4


# C++: CreditScheduler.CancelledFlowReclaimed
def test_cancelled_flow_reclaimed() -> None:
    s = make_sched(4, 16)
    k1 = "p" + SEP + "1"
    k2 = "p" + SEP + "2"
    # Empty WANT with nothing in flight -> no tombstone created.
    s.on_want(k1, want(0))
    assert s.tracked_flows() == 0
    # Grant, then cancel while regions are held -> kept until they drain.
    g = s.on_want(k2, want(2))
    assert len(g) == 2
    s.on_want(k2, want(0))  # cancel; 2 regions still in flight
    assert s.tracked_flows() == 1  # stays until held drains
    s.on_scatter_done(k2, g[0].offset)
    s.on_scatter_done(k2, g[1].offset)
    assert s.tracked_flows() == 0  # reclaimed after the last region returns
    assert free_regions(s) == 4


# C++: CreditScheduler.ReclaimDefersBusyRegionThenFreeOrphan
def test_forget_defers_busy_region_then_free_orphan() -> None:
    s = make_sched(2, 16)
    key_a = "A" + SEP + "1"
    prefix_a = "A" + SEP
    key_b = "B" + SEP + "1"

    g = s.on_want(key_a, want(2))
    assert len(g) == 2  # A holds both regions
    busy_off = g[0].offset
    idle_off = g[1].offset

    re, deferred = s.forget_prefix(prefix_a, busy={busy_off})
    assert re == []  # no waiter yet
    assert deferred == [busy_off]
    assert s.tracked_flows() == 0  # flow A erased
    assert free_regions(s) == 1  # only the idle region is free

    # A new sender B wanting 2 takes only the idle region — NEVER the busy one.
    gb = s.on_want(key_b, want(2))
    assert len(gb) == 1
    assert gb[0].offset == idle_off
    assert free_regions(s) == 0

    # Scatter finishes -> free the orphan -> now B can finally get it.
    re2 = s.free_orphan_region(busy_off)
    assert len(re2) == 1
    assert re2[0].offset == busy_off
    assert s.held_count(key_b) == 2
    check_conservation(s, [key_a, key_b], 2)

    # A second free_orphan_region for the same offset is a no-op — must NOT
    # free busy_off again, which is now live under key_b.
    re3 = s.free_orphan_region(busy_off)
    assert re3 == []
    assert s.held_count(key_b) == 2
    check_conservation(s, [key_a, key_b], 2)


# C++: CreditScheduler.FreeOrphanRegionIgnoresNonOrphan
def test_free_orphan_region_ignores_non_orphan() -> None:
    s = make_sched(2, 16)
    flow = "A" + SEP + "1"
    g = s.on_want(flow, want(1))
    assert len(g) == 1
    live = g[0].offset  # a live region, never reclaim-deferred
    assert s.held_count(flow) == 1

    re = s.free_orphan_region(live)
    assert re == []
    assert s.held_count(flow) == 1  # still held — NOT freed
    assert free_regions(s) == 1  # unchanged

    s.on_scatter_done(flow, live)  # normal path still frees it exactly once
    assert free_regions(s) == 2
    assert s.held_count(flow) == 0


# C++: CreditScheduler.ReclaimFlowFreesHeldDefersBusyAndHeldByFlow
def test_forget_frees_held_defers_busy_and_held_by_flow() -> None:
    s = make_sched(4, 16)
    flow = "p" + SEP + "1"
    g = s.on_want(flow, want(3))
    assert len(g) == 3  # 3 held, 1 free

    assert s.held_by_flow(flow, g[0].offset)
    assert not s.held_by_flow(flow, 0x999999)  # not a held offset
    assert not s.held_by_flow("p" + SEP + "2", g[0].offset)  # different flow

    re, deferred = s.forget(flow, busy={g[1].offset})
    assert re == []  # no other flow waiting
    assert s.tracked_flows() == 0
    assert deferred == [g[1].offset]
    assert not s.held_by_flow(flow, g[0].offset)  # flow erased
    assert free_regions(s) == 3  # g0,g2 freed + originally-free; g1 in limbo

    s.free_orphan_region(g[1].offset)  # scatter finished -> free the deferred
    assert free_regions(s) == 4

    # Cancelling an unknown flow is a harmless no-op.
    none, none_deferred = s.forget("nope")
    assert none == []
    assert none_deferred == []


# C++: CreditScheduler.ReceiverReclaimQuarantinesUnwrittenRegionsUntilReaped
def test_receiver_reclaim_quarantines_unwritten_regions_until_reaped() -> None:
    clock = FakeClock()
    s = make_sched(2, 16, now_fn=clock)
    flow_a = "A" + SEP + "1"
    flow_b = "B" + SEP + "1"
    g = s.on_want(flow_a, want(2))
    assert len(g) == 2

    re, deferred = s.forget(flow_a, quarantine_s=30.0)
    assert re == []  # no waiter yet
    assert deferred == []
    assert s.tracked_flows() == 0  # flow gone...
    assert not s.held_by_flow(flow_a, g[0].offset)  # ...late DATA is dropped
    assert free_regions(s) == 0  # ...but NOTHING went back to the arena

    # A new flow must not get a quarantined region; early reap frees nothing.
    assert s.on_want(flow_b, want(1)) == []
    assert s.reap_quarantine() == []
    assert free_regions(s) == 0

    # Quarantine over -> the regions re-enter circulation; one serves B.
    clock.advance(31.0)
    re2 = s.reap_quarantine()
    assert len(re2) == 1
    assert re2[0].flow == flow_b
    assert s.held_count(flow_b) == 1
    assert free_regions(s) == 1

    # Reaping again is a harmless no-op (the reaped offsets are not re-freed).
    assert s.reap_quarantine() == []
    assert free_regions(s) == 1


# C++: CreditScheduler.ReceiverReclaimBusyRegionsDeferAsOrphansNotQuarantine
def test_receiver_reclaim_busy_regions_defer_as_orphans_not_quarantine() -> None:
    clock = FakeClock()
    s = make_sched(2, 16, now_fn=clock)
    flow = "A" + SEP + "1"
    g = s.on_want(flow, want(2))
    assert len(g) == 2

    _re, deferred = s.forget(flow, busy={g[0].offset}, quarantine_s=30.0)
    assert deferred == [g[0].offset]

    clock.advance(31.0)
    assert s.reap_quarantine() == []
    assert free_regions(s) == 1  # only the quarantined region came back

    s.free_orphan_region(g[0].offset)
    assert free_regions(s) == 2


# C++: CreditScheduler.StaleFlowsReportsIdleFlowsOnly
def test_stale_flows_reports_idle_flows_only() -> None:
    clock = FakeClock()
    s = make_sched(2, 1, now_fn=clock)
    dead = "A" + SEP + "1"
    live = "B" + SEP + "1"
    ga = s.on_want(dead, want(1))
    assert len(ga) == 1
    gb = s.on_want(live, want(2))  # one granted, one pending (cap 1)
    assert len(gb) == 1

    # t+40s: `live` completes its first chunk -> progress.
    clock.advance(40.0)
    assert len(s.on_scatter_done(live, gb[0].offset)) == 1

    # t+70s: `dead` silent for 70s -> stale; `live` progressed 30s ago -> not.
    clock.advance(30.0)
    stale = s.stale_flows(60.0)
    assert stale == [dead]
    assert s.stale_flows(90.0) == []  # longer lease: nobody is stale yet

    # Reclaim + quarantine + reap: the arena fully recovers the dead region.
    s.forget(dead, quarantine_s=30.0)
    assert free_regions(s) == 0  # quarantined, not yet back
    clock.advance(31.0)
    s.reap_quarantine()
    assert free_regions(s) == 1  # recovered (live still holds its 2nd chunk)


# Python composition: check_timeouts = stale_flows + forget + reap_quarantine.
def test_check_timeouts_reclaims_stale_and_reaps() -> None:
    clock = FakeClock()
    s = make_sched(2, 16, now_fn=clock)
    dead = "A" + SEP + "1"
    g = s.on_want(dead, want(1))
    assert len(g) == 1

    clock.advance(70.0)
    reclaimed, grants, deferred = s.check_timeouts(60.0, quarantine_s=30.0)
    assert reclaimed == [dead]
    assert grants == []
    assert deferred == []
    assert free_regions(s) == 1  # the dead region is quarantined, not free

    clock.advance(31.0)
    reclaimed2, grants2, _ = s.check_timeouts(60.0, quarantine_s=30.0)
    assert reclaimed2 == []
    assert grants2 == []  # nothing pending to grant
    assert free_regions(s) == 2  # quarantine reaped

    # idle_limit <= 0 disables the lease sweep (only the quarantine is reaped).
    g2 = s.on_want(dead, want(1))
    assert len(g2) == 1
    clock.advance(10_000.0)
    reclaimed3, _, _ = s.check_timeouts(0.0, quarantine_s=30.0)
    assert reclaimed3 == []
    assert s.held_count(dead) == 1  # not reclaimed with timeouts disabled


# C++: CreditScheduler.SharedArenaLocalAndRemoteShareOneAllocator
def test_shared_arena_local_and_remote_share_one_allocator() -> None:
    s = make_sched(4, 16)
    flow = "p" + SEP + "1"

    a0 = s.acquire_local(K_REGION)
    a1 = s.acquire_local(K_REGION)
    assert a0 is not None and a1 is not None
    assert s.local_held_count() == 2
    assert free_regions(s) == 2

    # Remote flow wants 4 but only 2 are free (local holds 2) -> gets 2.
    g = s.on_want(flow, want(4))
    assert len(g) == 2
    assert s.held_count(flow) == 2
    assert free_regions(s) == 0
    check_conservation(s, [flow], 4)

    # No region free -> acquire_local returns None (caller parks + retries).
    assert s.acquire_local(K_REGION) is None

    # Release one local region -> the remote flow immediately grabs it.
    re = s.release_local(a0)
    assert len(re) == 1
    assert s.held_count(flow) == 3
    assert s.local_held_count() == 1
    check_conservation(s, [flow], 4)

    # Release the last local region -> remote flow reaches its wanted 4.
    s.release_local(a1)
    assert s.held_count(flow) == 4
    assert s.local_held_count() == 0
    assert free_regions(s) == 0
    check_conservation(s, [flow], 4)

    # release_local on a region that isn't locally held is an idempotent no-op.
    none = s.release_local(a0)
    assert none == []
    check_conservation(s, [flow], 4)


# C++: CreditScheduler.ScatterDoneIdempotentForUnknownRegion
def test_scatter_done_idempotent_for_unknown_region() -> None:
    s = make_sched(4, 16)
    s.on_want("A", want(1))  # A holds 1, 3 free
    before = free_regions(s)
    re = s.on_scatter_done("A", 0x999000)  # an offset A never held
    assert re == []
    assert free_regions(s) == before  # no spurious free
    check_conservation(s, ["A"], 4)


# C++: CreditScheduler.WantEmptyStopsGranting
def test_want_empty_stops_granting() -> None:
    s = make_sched(8, 16)
    s.on_want("A", want(4))
    assert s.held_count("A") == 4
    g = s.on_want("A", want(0))  # cancel further grants
    assert g == []
    assert s.active_flows() == 0


# C++: CreditScheduler.VariableSizeRegionsPackAndBackpressure
def test_variable_size_regions_pack_and_backpressure() -> None:
    # Chunks of different byte sizes each get a region of exactly that size,
    # packed densely; FIFO per flow (no reordering ahead of a parked chunk).
    k_arena = 8 * K_REGION  # 32 KiB, min block K_REGION
    s = CreditScheduler(K_BASE, k_arena, K_REGION, 16)

    sizes = [K_REGION, 2 * K_REGION, 4 * K_REGION, K_REGION]  # exactly fills
    g = s.on_want("A", sizes)
    assert len(g) == 4
    assert [x.length for x in g] == sizes
    # Distinct, non-overlapping regions.
    for i in range(len(g)):
        for j in range(i + 1, len(g)):
            disjoint = (
                g[i].offset + g[i].length <= g[j].offset or g[j].offset + g[j].length <= g[i].offset
            )
            assert disjoint, f"regions {i} and {j} overlap"
    assert s.free_bytes() == 0  # arena exactly full

    # Free the 4-region chunk; a fresh flow's 4-region chunk can be granted.
    s.on_scatter_done("A", g[2].offset)
    g2 = s.on_want("B", [4 * K_REGION])
    assert len(g2) == 1
    assert g2[0].length == 4 * K_REGION


# C++: CreditScheduler.ManySingleThreadSendersCannotStarveLargeChunk +
#      CreditScheduler.FourThreadsPerSenderCannotStarveLargeChunk
@pytest.mark.parametrize("num_senders,threads_per_sender", [(8, 1), (4, 4)])
def test_bounded_sender_concurrency_cannot_starve_large_chunk(
    num_senders: int, threads_per_sender: int
) -> None:
    """Model the transceiver's bounded worker queues.

    Each sender keeps at most threads_per_sender one-chunk requests in
    flight, submitting a replacement as soon as one completes. The sustained
    small-request load must not starve a half-arena-sized chunk (receiver
    drain barrier).
    """
    num_small_flows = num_senders * threads_per_sender
    assert num_small_flows > 1
    s = CreditScheduler(K_BASE, num_small_flows * K_REGION, K_REGION, 1)

    def make_flow(sender: int, rid: int) -> str:
        return f"smallPeer{sender}{SEP}{rid}"

    slots: list[dict] = []
    flow_to_slot: dict[str, int] = {}
    for sender in range(num_senders):
        for thread in range(threads_per_sender):
            flow = make_flow(sender, thread)
            grants = s.on_want(flow, want(1))
            assert len(grants) == 1
            flow_to_slot[flow] = len(slots)
            slots.append(
                {
                    "sender": sender,
                    "next_rid": thread + threads_per_sender,
                    "flow": flow,
                    "held": grants[0].offset,
                }
            )
    assert s.free_bytes() == 0

    large_flow = "largePeer" + SEP + "0"
    large_bytes = (num_small_flows // 2) * K_REGION
    assert s.on_want(large_flow, [large_bytes]) == []

    large_granted = False

    def record_grants(grants: Sequence["Grant"]) -> None:
        nonlocal large_granted
        for grant in grants:
            if grant.flow == large_flow:
                assert grant.length == large_bytes
                large_granted = True
                continue
            slots[flow_to_slot[grant.flow]]["held"] = grant.offset

    next_slot = 0
    small_completions = 0
    max_completions = num_small_flows * 32
    while not large_granted and small_completions < max_completions:
        checked = 0
        while checked < len(slots) and slots[next_slot]["held"] is None:
            next_slot = (next_slot + 1) % len(slots)
            checked += 1
        assert checked < len(slots), (
            "drain stopped every in-flight small request before the large grant"
        )

        slot = slots[next_slot]
        completed_flow = slot["flow"]
        completed_offset = slot["held"]
        slot["held"] = None
        del flow_to_slot[completed_flow]
        record_grants(s.on_scatter_done(completed_flow, completed_offset))
        small_completions += 1
        if large_granted:
            break

        # The synchronous worker submits exactly one replacement request.
        slot["flow"] = make_flow(slot["sender"], slot["next_rid"])
        slot["next_rid"] += threads_per_sender
        flow_to_slot[slot["flow"]] = next_slot
        record_grants(s.on_want(slot["flow"], want(1)))
        next_slot = (next_slot + 1) % len(slots)

    assert large_granted, (
        f"bounded-concurrency small requests starved the large chunk after "
        f"{small_completions} completions"
    )


# ---- Python-only: eager gather budget over the shared arena ----


def test_eager_budget_caps_eager_local_regions_at_half_arena() -> None:
    # 4 regions -> eager budget = capacity // 2 = 2 regions.
    s = make_sched(4, 16)
    e0 = s.acquire_local(K_REGION, eager=True)
    e1 = s.acquire_local(K_REGION, eager=True)
    assert e0 is not None and e1 is not None
    # Third eager acquisition exceeds the half-arena budget -> refused even
    # though the arena has free space...
    assert s.acquire_local(K_REGION, eager=True) is None
    assert free_regions(s) == 2
    # ...while a credit-backed (non-eager) acquisition is NOT capped.
    c0 = s.acquire_local(K_REGION)
    assert c0 is not None
    assert s.local_held_count() == 3
    check_conservation(s, [], 4)

    # Releasing an eager region refunds its budget.
    s.release_local(e0)
    e2 = s.acquire_local(K_REGION, eager=True)
    assert e2 is not None


def test_promote_local_frees_eager_budget_without_freeing_region() -> None:
    s = make_sched(4, 16)
    e0 = s.acquire_local(K_REGION, eager=True)
    e1 = s.acquire_local(K_REGION, eager=True)
    assert e0 is not None and e1 is not None
    assert s.acquire_local(K_REGION, eager=True) is None  # budget full

    # e0's credit arrived: it stops counting against the eager budget but
    # remains held (not freed).
    s.promote_local(e0)
    assert s.local_held_count() == 2
    e2 = s.acquire_local(K_REGION, eager=True)  # budget freed -> succeeds
    assert e2 is not None
    assert s.local_held_count() == 3

    # promote_local on a non-eager / unknown offset is a no-op.
    s.promote_local(0xDEAD000)
    check_conservation(s, [], 4)

    # release_local of a promoted region still frees the arena bytes.
    s.release_local(e0)
    s.release_local(e1)
    s.release_local(e2)
    assert s.local_held_count() == 0
    assert free_regions(s) == 4


def test_eager_budget_uses_rounded_block_bytes() -> None:
    # Budget accounting uses the ROUNDED buddy-block size (what the arena
    # actually loses): an odd-sized eager request consumes its rounded block.
    s = make_sched(4, 16)  # budget = 2 * K_REGION
    e0 = s.acquire_local(K_REGION + 1, eager=True)  # rounds to 2 * K_REGION
    assert e0 is not None
    # The whole eager budget is consumed by the rounded block.
    assert s.acquire_local(K_REGION, eager=True) is None
    assert s.acquire_local(K_REGION) is not None  # non-eager still fine


def test_grant_length_is_raw_bytes_not_rounded_buddy_block() -> None:
    """Grant.length is the RAW requested chunk bytes, not the buddy block.

    The sender RDMA-writes exactly Grant.length bytes, so a regression that
    returns the rounded buddy-block size would make it write the block slack.
    """
    s = make_sched(4, 16)
    flow = "p" + SEP + "1"
    raw = K_REGION + 1  # non-power-of-two -> buddy block rounds to 2*K_REGION
    g = s.on_want(flow, [raw])
    assert len(g) == 1
    assert g[0].length == raw  # the raw requested bytes travel in the grant
    assert g[0].addr == K_BASE + g[0].offset
    # The backing buddy block is the rounded power-of-two size (block slack
    # exists but must never be written).
    assert s.region_bytes(g[0].offset) == 2 * K_REGION
    assert s.region_bytes(g[0].offset) > g[0].length
    # Byte-level conservation: the arena lost exactly the ROUNDED block.
    assert s.free_bytes() + s.region_bytes(g[0].offset) == 4 * K_REGION


# ---- on_want internal validation guard (round-3 fix, NB-3) -----------------


def test_on_want_rejects_oversized_chunk_outright() -> None:
    """An announcement with any chunk > arena capacity is rejected WHOLE.

    Internal last-line guard, independent of the reactor's caller-side WANT
    validation: no grants, no flow state (an unallocatable head chunk would
    otherwise age into drain mode and stall ALL remote granting), and the
    same key still works with a valid announcement afterwards.
    """
    s = make_sched(4, 16)
    cap = s.arena_capacity
    assert s.on_want("A", [cap + 1]) == []
    assert s.tracked_flows() == 0, "rejected announcement still registered flow state"
    assert free_regions(s) == 4
    # One bad chunk poisons the WHOLE announcement, even with valid siblings.
    assert s.on_want("A", [K_REGION, cap + K_REGION]) == []
    assert s.tracked_flows() == 0
    assert free_regions(s) == 4
    # The same key recovers: a subsequent valid WANT grants normally.
    g = s.on_want("A", want(2))
    assert len(g) == 2
    assert s.held_count("A") == 2
    check_conservation(s, ["A"], 4)


def test_on_want_rejects_zero_and_negative_chunks() -> None:
    """A chunk size <= 0 rejects the whole announcement (no grants, no flow)."""
    s = make_sched(4, 16)
    assert s.on_want("A", [0]) == []
    assert s.on_want("A", [-K_REGION]) == []
    assert s.on_want("A", [K_REGION, 0]) == []
    assert s.tracked_flows() == 0
    assert free_regions(s) == 4
    g = s.on_want("A", want(1))
    assert len(g) == 1
    check_conservation(s, ["A"], 4)


# ---- concurrency smoke: cross-thread acquire_local vs reactor-side events --


def test_concurrent_local_and_remote_events_keep_conservation() -> None:
    """Concurrency smoke over the scheduler's internal lock.

    N threads hammer acquire_local/release_local (the submit()-thread path)
    while the main thread runs on_want/on_scatter_done cycles (the reactor
    path). Deterministically bounded; asserts no exception escaped and the
    region-conservation invariant at the end.
    """
    n_regions = 8
    n_threads = 4
    iters = 300
    s = make_sched(n_regions, 4)
    flow = "p" + SEP + "1"
    errors: list[BaseException] = []
    barrier = threading.Barrier(n_threads + 1)

    def hammer() -> None:
        try:
            barrier.wait()
            for _ in range(iters):
                off = s.acquire_local(K_REGION, eager=True)
                if off is not None:
                    s.release_local(off)
        except BaseException as exc:  # noqa: BLE001 - surfaced via `errors`
            errors.append(exc)

    threads = [threading.Thread(target=hammer) for _ in range(n_threads)]
    for t in threads:
        t.start()
    barrier.wait()
    for _ in range(iters):
        for g in s.on_want(flow, want(2)):
            s.on_scatter_done(flow, g.offset)
    for t in threads:
        t.join()

    assert errors == []
    # Note: release_local() inside the hammer threads may have re-scheduled
    # grants to `flow` that nobody completed; conservation must hold anyway.
    check_conservation(s, [flow], n_regions)
    # Drain the flow entirely -> every region returns to the arena.
    s.forget(flow)
    assert s.local_held_count() == 0
    assert free_regions(s) == n_regions
    assert s.tracked_flows() == 0
