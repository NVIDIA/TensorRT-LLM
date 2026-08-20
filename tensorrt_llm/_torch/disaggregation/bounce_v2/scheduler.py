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
"""Receiver-side credit allocator + fair scheduler over a single shared arena.

Python port of the C++ ``CreditScheduler``
(cpp/tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/CreditScheduler.{h,cpp}).
Pure logic, no threads / IO / CUDA — the GPU buffer lives in the transport;
this owns only a :class:`~.buddy.BuddyAllocator` over byte offsets plus the
base address for computing absolute grant addresses.

VARIABLE REGIONS: each chunk requests only its packed byte extent (the buddy
rounds up to a power of two), so many small transfers pack tightly (R8) while
a request larger than the whole arena streams through chunk by chunk (R1).

TERMINOLOGY: a client is identified by an opaque **flow id** string
("peer<sep>rid"), NOT an agent name; :meth:`CreditScheduler.forget_prefix` is
the only peer-level operation. The local sender shares this same arena via
:meth:`CreditScheduler.acquire_local` (gather staging).

THREADING: every public method takes the internal lock, so calls are safe
from any thread. The reactor thread is the primary owner (all flow-state
events happen there); the one cross-thread caller is ``acquire_local`` from
``submit()`` app threads (eager gather staging).

TIME: all durations/timestamps are SECONDS (floats) from the injected
``now_fn`` (default ``time.monotonic``); config values in milliseconds must
be converted by the caller. Tests inject a fake clock and advance it instead
of sleeping.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import AbstractSet, Callable, Optional, Sequence

from .buddy import BuddyAllocator

__all__ = ["Grant", "CreditScheduler"]

_EMPTY_SET: frozenset[int] = frozenset()


@dataclass(frozen=True)
class Grant:
    """A credit handed to a flow: exclusive write permission for one receiver
    arena allocation.

    ``offset`` is the region's arena offset (its opaque handle),
    ``addr = base_addr + offset`` the absolute device address, and ``length``
    the chunk's packed transfer length — the buddy block backing the region
    may be larger, and the slack must never be written.
    """

    flow: str
    offset: int
    addr: int
    length: int


@dataclass
class _FlowState:
    pending: deque[int] = field(default_factory=deque)  # chunk bytes awaiting a grant (FIFO)
    held: set[int] = field(default_factory=set)  # region offsets granted to this flow
    blocked_at_grant_sequence: Optional[int] = None  # first grant seq where head did not fit
    # Lease stamp: last WANT / grant issued / scatter completed. stale_flows()
    # reports flows idle beyond the receiver's lease so a dead sender cannot
    # leak regions forever.
    last_progress: float = 0.0


class CreditScheduler:
    """Fair, bounded credit scheduler over one shared buddy arena."""

    # Receiver-only anti-starvation barrier thresholds: a flow whose head
    # allocation keeps failing becomes the drain flow once at least
    # max(_MINIMUM_BYPASS_GRANTS, _BYPASS_ROUNDS * ring size) other grants
    # have passed it.
    _MINIMUM_BYPASS_GRANTS = 8
    _BYPASS_ROUNDS = 2

    def __init__(
        self,
        base_addr: int,
        arena_size_bytes: int,
        arena_allocation_granularity_bytes: int,
        max_inflight_chunks_per_request: int,
        now_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create the scheduler.

        Args:
            base_addr: Device address of arena offset 0
                (``Grant.addr = base_addr + offset``).
            arena_size_bytes: Total arena size (buddy rounds it down).
            arena_allocation_granularity_bytes: Smallest buddy allocation.
            max_inflight_chunks_per_request: Per-flow in-flight allocation cap
                (values < 1 are clamped to 1, like the C++).
            now_fn: Injectable monotonic clock in seconds, for the flow leases
                and quarantine deadlines.
        """
        self._mu = threading.Lock()
        self._arena = BuddyAllocator(arena_size_bytes, arena_allocation_granularity_bytes)
        self._base_addr = base_addr
        self._max_inflight = max(1, max_inflight_chunks_per_request)
        self._now = now_fn

        self._flows: dict[str, _FlowState] = {}
        self._local_held: set[int] = set()
        # Rounded (buddy-block) bytes of local regions acquired EAGERLY
        # (before their credit arrived), and the cap they must stay under.
        # Eager staging is capped at HALF the arena so that on a bidirectional
        # deployment each side can always still grant incoming regions — two
        # eager senders can never starve each other into a circular wait.
        self._eager_held: dict[int, int] = {}
        self._eager_held_bytes = 0
        self._eager_budget_bytes = self._arena.capacity // 2
        # Regions deferred by flow/peer reclamation while a scatter was still
        # reading them, awaiting free_orphan_region().
        self._orphans: set[int] = set()
        # Regions reclaimed by a RECEIVER-initiated teardown while possibly
        # still being RDMA-written by the peer (granted, DATA never arrived),
        # mapped to their reuse deadline. A one-sided write cannot be aborted,
        # so time is the only barrier: they stay allocated (never
        # re-grantable) until reap_quarantine() passes the deadline.
        self._quarantined: dict[int, float] = {}
        # Anti-starvation state (see schedule()).
        self._grant_sequence = 0
        self._drain_flow: Optional[str] = None
        # Round-robin ring of active flow ids (insertion order) + cursor.
        self._ring: list[str] = []
        self._cursor = 0

    # ------------------------------------------------------------------ #
    # internal helpers (call with self._mu held)
    # ------------------------------------------------------------------ #

    def _ensure_in_ring(self, flow: str) -> None:
        if flow not in self._ring:
            self._ring.append(flow)

    def _drop_from_ring(self, flow: str) -> None:
        if self._drain_flow == flow:
            self._drain_flow = None
        try:
            idx = self._ring.index(flow)
        except ValueError:
            return
        del self._ring[idx]
        if not self._ring:
            self._cursor = 0
            return
        if idx < self._cursor:
            self._cursor -= 1
        self._cursor %= len(self._ring)

    def _erase_if_done(self, flow: str) -> None:
        st = self._flows.get(flow)
        if st is not None and not st.pending and not st.held:
            del self._flows[flow]
            self._drop_from_ring(flow)

    def _maybe_activate_drain(self) -> None:
        """Age a repeatedly bypassed flow into receiver drain mode."""
        if self._drain_flow is not None or not self._ring:
            return
        ring_size = len(self._ring)
        bypass_threshold = max(self._MINIMUM_BYPASS_GRANTS, self._BYPASS_ROUNDS * ring_size)
        oldest_blocked: Optional[int] = None
        # Scan from the round-robin cursor so equal-age candidates retain the
        # scheduler's normal order.
        for k in range(ring_size):
            idx = (self._cursor + k) % ring_size
            st = self._flows.get(self._ring[idx])
            if st is None:
                continue
            if (
                not st.pending
                or len(st.held) >= self._max_inflight
                or st.blocked_at_grant_sequence is None
            ):
                continue
            bypassed = self._grant_sequence - st.blocked_at_grant_sequence
            if bypassed < bypass_threshold:
                continue
            if oldest_blocked is None or st.blocked_at_grant_sequence < oldest_blocked:
                oldest_blocked = st.blocked_at_grant_sequence
                self._drain_flow = self._ring[idx]

    def _schedule(self) -> list[Grant]:
        """Hand out as many region grants as possible RIGHT NOW, fairly and
        bounded. Called whenever space frees up or new demand arrives. Rules:
          1. Fair round-robin over flows; a head chunk that repeatedly fails
             to fit while other grants pass it enters drain mode: no NEW
             remote grants until that head fits, so sustained smaller traffic
             cannot starve it (``acquire_local`` is deliberately unaffected —
             a receiver-only barrier cannot create a bidirectional circular
             wait).
          2. Per-flow in-flight cap.
          3. Arena capacity: if the next chunk does not fit, skip the flow (a
             smaller chunk elsewhere may still fit) — backpressure, never an
             error.
        Each inner sweep grants AT MOST ONE region then re-sweeps from past
        the flow just served, giving strict rotation (A,B,A,B,...) instead of
        filling one flow to its cap first.
        """
        grants: list[Grant] = []
        while self._ring:
            self._maybe_activate_drain()
            if self._drain_flow is not None:
                st = self._flows.get(self._drain_flow)
                if st is None or not st.pending or len(st.held) >= self._max_inflight:
                    self._drain_flow = None
                    continue
                want = st.pending[0]
                off = self._arena.allocate(want)
                if off is None:
                    # Existing regions keep progressing and freeing space, but
                    # do not refill them with smaller chunks while draining.
                    return grants
                idx = self._ring.index(self._drain_flow)
                st.pending.popleft()
                st.held.add(off)
                st.blocked_at_grant_sequence = None
                st.last_progress = self._now()  # issuing a grant renews the lease
                grants.append(Grant(self._drain_flow, off, self._base_addr + off, want))
                self._cursor = (idx + 1) % len(self._ring)
                self._grant_sequence += 1
                self._drain_flow = None
                continue

            progress = False
            for k in range(len(self._ring)):
                idx = (self._cursor + k) % len(self._ring)
                st = self._flows.get(self._ring[idx])
                if st is None:
                    continue  # ring/flows invariant says this cannot happen; skip
                if not st.pending or len(st.held) >= self._max_inflight:
                    continue
                want = st.pending[0]
                off = self._arena.allocate(want)
                if off is None:
                    if st.blocked_at_grant_sequence is None:
                        st.blocked_at_grant_sequence = self._grant_sequence
                    continue  # try another flow (a smaller chunk may fit)
                st.pending.popleft()
                st.held.add(off)
                st.blocked_at_grant_sequence = None
                st.last_progress = self._now()
                grants.append(Grant(self._ring[idx], off, self._base_addr + off, want))
                self._cursor = (idx + 1) % len(self._ring)
                self._grant_sequence += 1
                progress = True
                break  # one grant per sweep -> strict rotation
            if not progress:
                break
        return grants

    def _drop_flow(
        self,
        flow: str,
        busy: AbstractSet[int],
        deferred_out: list[int],
        quarantine_s: float,
    ) -> None:
        """Free one flow's held regions (busy ones deferred as orphans; with
        ``quarantine_s > 0`` the non-busy ones are quarantined instead of
        freed), then erase the flow and drop it from the ring."""
        st = self._flows.get(flow)
        if st is None:
            return
        for off in st.held:
            if off in busy:
                # A scatter is still reading this region -> the caller frees
                # it later via free_orphan_region().
                deferred_out.append(off)
                self._orphans.add(off)
            elif quarantine_s > 0:
                # Receiver-initiated reclaim: the peer may still be
                # RDMA-writing this granted region. Keep it allocated (never
                # re-grantable) until reap_quarantine() passes the deadline.
                self._quarantined[off] = self._now() + quarantine_s
            else:
                self._arena.free(off)
        del self._flows[flow]
        self._drop_from_ring(flow)

    # ------------------------------------------------------------------ #
    # receiver-role events
    # ------------------------------------------------------------------ #

    def on_want(self, flow: str, chunk_bytes: Sequence[int]) -> list[Grant]:
        """A flow announces the per-chunk byte sizes it wants to write (in
        order). EMPTY = cancel. Returns the grants to send now.

        The caller (reactor) MUST validate decoded WANT chunk sizes —
        ``0 < size <= min(max_chunk_size_bytes, arena_capacity)`` — before
        calling this: an unallocatable head chunk eventually enters drain
        mode and stalls ALL remote granting until the flow is forgotten.
        Inherited C++ behavior — there the capability handshake clamps chunk
        sizes for well-behaved peers, so only a malformed/hostile WANT can
        carry an oversized chunk.
        """
        with self._mu:
            # Internal last-line guard on top of the caller-side validation
            # above: an unallocatable chunk (<= 0 or beyond the arena) would
            # age into drain mode and stall ALL remote granting, so reject the
            # whole announcement regardless of the caller contract.
            if any(int(b) <= 0 or int(b) > self._arena.capacity for b in chunk_bytes):
                return []
            st = self._flows.setdefault(flow, _FlowState())
            st.pending = deque(int(b) for b in chunk_bytes)
            st.blocked_at_grant_sequence = None
            st.last_progress = self._now()  # a fresh WANT renews the lease
            if st.pending:
                self._ensure_in_ring(flow)
            else:
                # cancel: drop now if nothing is in flight; otherwise it is
                # reclaimed when held drains.
                self._erase_if_done(flow)
            return self._schedule()

    def on_scatter_done(self, flow: str, offset: int) -> list[Grant]:
        """A region finished scattering on the receiver -> free it and
        re-schedule. Idempotent: a duplicate/late notification for a region
        the flow no longer holds is ignored."""
        with self._mu:
            st = self._flows.get(flow)
            if st is not None and offset in st.held:
                st.held.discard(offset)
                self._arena.free(offset)
                st.last_progress = self._now()  # scatter completion is progress
            self._erase_if_done(flow)
            return self._schedule()

    def forget(
        self,
        flow: str,
        busy: AbstractSet[int] = _EMPTY_SET,
        quarantine_s: float = 0.0,
    ) -> tuple[list[Grant], list[int]]:
        """Cancel ONE flow (explicit abort / empty WANT / lease expiry): free
        its held regions and drop it.

        Any held region in ``busy`` (a scatter is still reading it) is
        DEFERRED instead of freed; the caller MUST later call
        :meth:`free_orphan_region` for each once its scatter completes. The
        immediate-free default (``quarantine_s == 0``) is safe ONLY when the
        SENDER initiated the reclaim (its cancel is sent after draining
        in-flight writes); receiver-initiated reclaims (peer loss / lease
        expiry) pass ``quarantine_s > 0`` so possibly-still-being-written
        regions stay out of the arena until the deadline.

        Returns:
            ``(grants, deferred_offsets)``.
        """
        with self._mu:
            deferred: list[int] = []
            self._drop_flow(flow, busy, deferred, quarantine_s)
            return self._schedule(), deferred

    def forget_prefix(
        self,
        prefix: str,
        busy: AbstractSet[int] = _EMPTY_SET,
        quarantine_s: float = 0.0,
    ) -> tuple[list[Grant], list[int]]:
        """Reclaim every flow whose id starts with ``prefix`` (all flows of a
        gone peer). Same busy/quarantine semantics as :meth:`forget`. An empty
        prefix is refused (it would match every flow of every peer).

        Returns:
            ``(grants, deferred_offsets)``.
        """
        if not prefix:
            return [], []
        with self._mu:
            deferred: list[int] = []
            victims = [key for key in self._flows if key.startswith(prefix)]
            for key in victims:
                self._drop_flow(key, busy, deferred, quarantine_s)
            return self._schedule(), deferred

    def free_orphan_region(self, offset: int) -> list[Grant]:
        """Free a region deferred by :meth:`forget`/:meth:`forget_prefix`
        (its in-flight scatter has finished) + re-schedule. Only a genuinely
        deferred orphan is freed — a stray/duplicate call cannot free an
        offset that has since been re-allocated to a live flow."""
        with self._mu:
            if offset in self._orphans:
                self._orphans.discard(offset)
                self._arena.free(offset)
            return self._schedule()

    def stale_flows(self, idle_limit_s: float) -> list[str]:
        """Flows HOLDING at least one region with no progress (no WANT, no
        grant issued, no scatter completed) for longer than ``idle_limit_s``.

        A dead sender emits neither DATA nor a cancel — unobservable through
        the protocol alone — so the receiver reclaims these via
        ``forget(quarantine_s > 0)``. Pending-only flows tie up no memory and
        are never reported: they may legitimately queue behind a full arena
        for a long time.
        """
        with self._mu:
            now = self._now()
            return [
                key
                for key, st in self._flows.items()
                if st.held and now - st.last_progress > idle_limit_s
            ]

    def reap_quarantine(self) -> list[Grant]:
        """Free every quarantined region whose deadline has passed (no write
        posted before its flow's lease expired can plausibly still be in
        flight) + re-schedule."""
        with self._mu:
            now = self._now()
            expired = [off for off, deadline in self._quarantined.items() if now >= deadline]
            for off in expired:
                self._arena.free(off)
                del self._quarantined[off]
            # Nothing freed -> nothing changed; skip the ring sweep.
            return self._schedule() if expired else []

    def check_timeouts(
        self,
        idle_limit_s: float,
        quarantine_s: float,
        busy: AbstractSet[int] = _EMPTY_SET,
    ) -> tuple[list[str], list[Grant], list[int]]:
        """Periodic lease sweep: reclaim (with quarantine) every stale flow,
        then reap expired quarantined regions.

        Convenience composition of :meth:`stale_flows`, :meth:`forget` and
        :meth:`reap_quarantine`. NOTE: the reactor open-codes this composition
        in its lease sweep instead of calling it, so the stale set can also
        drive its scatter-backlog purge before the busy set is computed. When
        ``idle_limit_s <= 0`` the lease sweep is disabled (mirrors the C++
        "request_timeout_ms <= 0 disables timeouts") and only the quarantine
        is reaped.

        REACTOR-THREAD-ONLY: this method drops the lock between the stale
        sweep, the per-flow ``forget`` calls and the quarantine reap, so it is
        only correct under the module's threading contract that ALL flow
        events (``on_want`` / ``on_scatter_done`` / ``forget*``) run on one
        thread — only ``acquire_local``/``release_local`` are cross-thread
        safe. If a flow could progress on another thread between the sweep
        and its ``forget``, it would still be reclaimed here despite that
        progress.

        Returns:
            ``(reclaimed_flow_ids, grants, deferred_offsets)``.
        """
        stale = self.stale_flows(idle_limit_s) if idle_limit_s > 0 else []
        grants: list[Grant] = []
        deferred: list[int] = []
        for flow in stale:
            flow_grants, flow_deferred = self.forget(flow, busy, quarantine_s)
            grants.extend(flow_grants)
            deferred.extend(flow_deferred)
        grants.extend(self.reap_quarantine())
        return stale, grants, deferred

    def held_by_flow(self, flow: str, offset: int) -> bool:
        """True if ``flow`` currently holds region ``offset``. Lets the
        transport drop a late DATA for a region the flow no longer owns —
        scattering a freed/re-granted region would corrupt another flow's
        data."""
        with self._mu:
            st = self._flows.get(flow)
            return st is not None and offset in st.held

    # ------------------------------------------------------------------ #
    # local (sender) role: gather staging from the SAME arena
    # ------------------------------------------------------------------ #

    def acquire_local(self, nbytes: int, eager: bool = False) -> Optional[int]:
        """Allocate a region of ``nbytes`` for local gather staging
        (non-blocking). Returns its offset, or ``None`` if the arena cannot
        fit it right now (the caller parks and retries).

        With ``eager`` the allocation is additionally capped so that all
        eager (credit-less) local regions together stay under HALF the arena
        capacity: on a bidirectional deployment this guarantees each side can
        always still grant incoming regions, so two eager senders can never
        starve each other into a circular wait. Credit-backed (non-eager)
        acquisitions are not capped.
        """
        with self._mu:
            off = self._arena.allocate(nbytes)
            if off is None:
                return None
            if eager:
                # Budget accounting uses the ROUNDED buddy-block size — that
                # is what the arena actually loses.
                rounded = self._arena.block_bytes(off)
                if self._eager_held_bytes + rounded > self._eager_budget_bytes:
                    self._arena.free(off)
                    return None
                self._eager_held[off] = rounded
                self._eager_held_bytes += rounded
            self._local_held.add(off)
            return off

    def promote_local(self, offset: int) -> None:
        """An eager region's credit arrived: it is now credit-backed, so stop
        counting it against the eager budget (otherwise steady-state
        pipelining would be throttled to the eager cap even though every
        in-flight chunk has its credit). No-op for non-eager offsets."""
        with self._mu:
            rounded = self._eager_held.pop(offset, None)
            if rounded is not None:
                self._eager_held_bytes -= rounded

    def release_local(self, offset: int) -> list[Grant]:
        """Return a locally-held region (its chunk was ACKed / failed) to the
        arena + re-schedule (freed bytes may let a waiting remote flow alloc
        its next chunk)."""
        with self._mu:
            if offset in self._local_held:
                self._local_held.discard(offset)
                rounded = self._eager_held.pop(offset, None)
                if rounded is not None:
                    self._eager_held_bytes -= rounded
                self._arena.free(offset)
            return self._schedule()

    # ------------------------------------------------------------------ #
    # inspectors (tests / metrics)
    # ------------------------------------------------------------------ #

    def held_count(self, flow: str) -> int:
        """Number of regions currently granted to ``flow``."""
        with self._mu:
            st = self._flows.get(flow)
            return 0 if st is None else len(st.held)

    def local_held_count(self) -> int:
        """Number of regions held for local gather staging."""
        with self._mu:
            return len(self._local_held)

    def free_bytes(self) -> int:
        """Free arena bytes (sum of free buddy blocks)."""
        with self._mu:
            return self._arena.free_bytes

    @property
    def arena_capacity(self) -> int:
        """Largest region a fully-drained arena can ever hand out (the buddy
        usable capacity). Callers clamp/validate max_chunk_size against it."""
        return self._arena.capacity

    def region_bytes(self, offset: int) -> int:
        """Byte size of the buddy block backing a live region offset (0 if
        not allocated). The whole block belongs to one flow, so it bounds how
        far a scatter may read without touching another flow's region."""
        with self._mu:
            return self._arena.block_bytes(offset)

    def active_flows(self) -> int:
        """Flows that still have pending (ungranted) chunks."""
        with self._mu:
            return sum(1 for st in self._flows.values() if st.pending)

    def tracked_flows(self) -> int:
        """All flows with any state (pending or held)."""
        with self._mu:
            return len(self._flows)
