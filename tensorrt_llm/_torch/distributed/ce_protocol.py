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
"""Boundary bookkeeping and the signal-balance discipline that copy-engine push protocols share
over a :class:`~tensorrt_llm._torch.distributed.symm_mem_pool.SymmMemPool`.

Why
---
A copy-engine push protocol (the VisualGen token-sharded TP all-gather and reduce-scatter, the
head-chunked Ulysses exchange) copies its blocks straight into the peers' regions of a
:class:`SymmMemPool` on side streams and orders the consumer's reads behind the copies with the
pool's flags: one ``put`` per ``wait`` on every ``(channel, src -> dst)`` pair. Everything around
the copies is the same in all three protocols: side streams forked from the compute stream by an
event and joined back at the end of the boundary, a per-slot record of the waits a boundary owes
and the puts it issued, a release that waits every flag a peer put that no wait consumed, a
recovery path for the op's ``except`` branch, the rules a CUDA graph imposes on reallocation, and
a per-group state that probes once, warns once and falls back. This module holds that
intersection, once; the copies, the region layouts, the channel maps and the schedules stay
with their protocols.

Boundary
--------
A boundary is one use of one transport slot by every rank of the group: opened by
:meth:`CeTransport.open` with the ``(src, channel)`` flags it will consume, closed by
:meth:`CeTransport.done`, or by :meth:`CeTransport.reset` from the op's ``except`` path. In
between, the protocol forks each destination's copy stream from the current stream
(:meth:`CopyStreams.fork`), enqueues its copies there, puts the destination's flag behind them
(:meth:`CeTransport.mark_pushed`), and makes the current stream wait a source's flag before it
reads that source's region (:meth:`CeTransport.wait_ready`). ``done`` and ``reset`` both take
the slot from IN FLIGHT to IDLE through the same release: wait every owed flag that no wait
consumed, join the copy streams forked since the previous join into the current stream, clear
the slot. The invariant after every completed boundary, for every ``(channel, src -> dst)``
pair, is that the flag is clear and ``puts == waits``, so each wait of the next boundary finds
exactly one flag.

``reset`` repairs a boundary that every rank abandoned at the same point (a rank-symmetric
raise): it trims the owed waits to the flags the peers did put, derived from this rank's own
pushes under rank symmetry (:meth:`CeTransport._arrived`), and releases. A one-sided raise
followed by a retry cannot be told apart from the next boundary by one-bit flags, so ``reset``
also latches a resync: the next eager ``open`` all-gathers each rank's count of completed
boundaries once and raises when the ranks disagree (the op body then disables the protocol for
the group, :meth:`CeState.disable`).

Rules
-----
* One put per wait on every ``(channel, src -> dst)`` pair, within one boundary: ``mark_pushed``
  records the put, ``wait_ready`` the wait, and the release waits whatever is still owed. A
  protocol pushes each channel to its destinations in one fixed order per boundary and passes
  the destination-side mirror of that order as ``arrival_order``; that is all ``reset`` needs.
* Fork, then join what was forked. The current stream waits only on the copy streams forked
  since the previous join (one graph edge per forked stream under capture); a stream never
  forked into a capture is never joined into it (that would be a cross-capture dependency). The
  forked set is per transport, not per slot: with two slots in flight the first ``done`` joins
  the streams both forked (an extra edge, never a missing one). ``reset`` joins every stream
  when eager and refuses to do so under capture.
* Reserve eagerly, never under capture (the pool raises), never in flight, and never so that an
  allocation a CUDA graph captured would GROW (growth frees the captured buffers). A first
  reservation of a never-reserved pool frees nothing and stays allowed after captures. A
  reallocation drops every cached view. The request (``slot_nbytes``, ``slots``) must be
  rank-symmetric, as the protocols derive it from agreed plan ints: the growth guard raises
  locally before the pool's collective reserve, so a request that grows on one rank only would
  leave the other ranks in the pool's all-gather.
* A boundary that ran under capture marks the transport ``captured`` only when it completes;
  a captured boundary that raised (its capture is discarded) leaves no mark.
* A channel is validated where the protocol declares it (``open``, ``mark_pushed``), never
  first by the pool's ``wait`` inside the release. A release that raises anyway (a wait the
  pool refuses, a join the capture refuses) still takes the slot to IDLE: the pairs waited
  before the raise are not waited again, the pairs after it stay unwaited (a later put on such
  a pair blocks at the deadline, which the pool's triage hint names), and ``reset`` and
  ``close`` remain possible instead of the slot wedging the transport and its pool.
* Every rank issues the same sequence of boundaries per transport. A rank-symmetric raise is
  repaired by ``reset``; a retry after a one-sided raise is safe only through the resync above,
  and a one-sided raise ends the group loudly rather than silently: the rank that reset
  all-gathers at its next ``open`` (a warning names the resync first) while the peers, which
  did not reset, trap at the deadline on the flags it never put. A resync that finds the counts
  unequal raises :class:`CeResyncError`, which the op body routes to :meth:`CeState.disable`.
* ``close`` releases the pool and must follow the CUDA-graph runners' ``clear()`` (a graph may
  still reference the pool); the caller orders it so.
* Deadline: the state passes ``timeout_ms`` to the pools it builds; the default is the pool's
  (:data:`~tensorrt_llm._torch.distributed.symm_mem_pool.DEFAULT_TIMEOUT_MS`, 10 minutes),
  which the TP pair keeps. The head-chunked Ulysses exchange passes 60 s on every pool it
  builds, so that a one-sided failure surfaces on the other ranks within the time the
  synchronous Ulysses barrier already allows (``UlyssesAttention._barrier_timeout_ms``).
"""

from collections.abc import Callable, Iterable, Sequence

import torch
import torch.distributed as dist

from tensorrt_llm.logger import logger

from .symm_mem_pool import DEFAULT_TIMEOUT_MS, PoolLayout, SymmMemPool, probe_symmetric_memory

__all__ = [
    "CeResyncError",
    "CeState",
    "CeTransport",
    "CopyStreams",
]


def _is_capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def _gather_boundary_counts(group: dist.ProcessGroup, count: int) -> list[int]:
    """Collective: every rank's count of completed boundaries, in rank order."""
    counts: list[int | None] = [None] * dist.get_world_size(group)
    dist.all_gather_object(counts, int(count), group=group)
    return [-1 if c is None else int(c) for c in counts]


class CeResyncError(RuntimeError):
    """The resync after a failed boundary found the ranks at different boundary counts: the
    failure was one-sided and a retry on the copy-engine path is not safe. The op body turns it
    into :meth:`CeState.disable` (the group continues on the NCCL path)."""


# =============================================================================
# Copy streams
# =============================================================================


class CopyStreams:
    """The side streams of one protocol object: one per destination (default) or one for all.

    One stream per destination lets the pushes to different peers be enqueued independently
    (``per_peer_streams`` of the reduce-scatter, measured 750 vs 731 GB/s against one stream;
    the branch's per-destination copy streams); one for all is the all-gather's single side
    stream. ``priority`` is the protocol's choice: the head-chunked Ulysses exchange runs its
    copy streams at ``-1`` so that its own launches on them win the first retiring CTA slot
    beside the attention kernel; the TP pair keeps the default ``0``.

    Args:
        world_size: Ranks of the group.
        rank: This process's rank in the group.
        device: The CUDA device the streams belong to.
        per_peer: One stream per destination (``True``) or one stream for every destination.
        priority: CUDA stream priority of the copy streams.
    """

    def __init__(
        self,
        world_size: int,
        rank: int,
        device: torch.device,
        *,
        per_peer: bool = True,
        priority: int = 0,
    ) -> None:
        if world_size < 2 or not 0 <= rank < world_size:
            raise ValueError(
                f"CopyStreams needs world_size >= 2 and 0 <= rank < world_size (got "
                f"{rank}/{world_size})."
            )
        self.rank = rank
        self.world_size = world_size
        self.peers: tuple[int, ...] = tuple(p for p in range(world_size) if p != rank)
        """tuple[int, ...]: Every other rank, ascending (the destinations a stream can carry)."""
        n = len(self.peers) if per_peer else 1
        self._streams = [torch.cuda.Stream(device=device, priority=priority) for _ in range(n)]
        self._of = {dst: self._streams[i if per_peer else 0] for i, dst in enumerate(self.peers)}
        self._forked: set[torch.cuda.Stream] = set()  # streams forked since the previous join

    def __len__(self) -> int:
        return len(self._streams)

    def stream(self, dst: int) -> torch.cuda.Stream:
        """The stream that carries the pushes to destination ``dst`` (tests inject delays on it
        to perturb the arrival order)."""
        stream = self._of.get(dst)
        if stream is None:
            raise ValueError(
                f"CopyStreams: destination {dst} is not a peer of rank {self.rank}/"
                f"{self.world_size}."
            )
        return stream

    def fork(self, dst: int, event: torch.cuda.Event) -> torch.cuda.Stream:
        """Make ``dst``'s stream wait on ``event`` (recorded by the protocol on the current stream
        behind whatever produced the block) and note it as forked until the next join; returns
        the stream so the caller enqueues its copies and the put on it."""
        stream = self.stream(dst)
        stream.wait_event(event)
        self._forked.add(stream)
        return stream

    def join_forked(self) -> None:
        """The end of every boundary: the current stream waits on the streams forked since the
        previous join, once each (one graph edge per forked stream under capture, never an edge
        to a stream outside the capture), and the forked set is cleared -- also when a wait
        raises, so a failed join is attempted once (module docstring, Rules)."""
        current = torch.cuda.current_stream()
        try:
            for stream in self._streams:  # in stream order: deterministic edges
                if stream in self._forked:
                    current.wait_stream(stream)
        finally:
            self._forked.clear()

    def join_all_eager(self) -> None:
        """The last step of :meth:`CeTransport.reset`: the current stream waits on every copy
        stream, forked or not, so nothing of the failed boundary is pending anywhere. Refused
        under CUDA-graph capture, where a stream outside the capture may not be joined."""
        if _is_capturing():
            raise RuntimeError(
                "CopyStreams.join_all_eager under CUDA-graph capture: only the forked streams can "
                "be joined into a capture (join_forked)."
            )
        current = torch.cuda.current_stream()
        for stream in self._streams:
            current.wait_stream(stream)
        self._forked.clear()


# =============================================================================
# Transport
# =============================================================================


class CeTransport:
    """Boundary bookkeeping and the signal-balance discipline over a :class:`SymmMemPool`.

    Flags are one bit per ``(channel, src -> dst)`` on the pad of ``pool`` -- the SIGNAL pool,
    which is the protocol's data pool (the TP pair) or a tiny per-group flag pool whose data
    pools live beside the transport (the Ulysses exchange); ``put`` blocks on a set flag, ``wait``
    on a clear one. The transport guarantees that after every completed boundary every
    ``(channel, src -> me)`` flag whose put was issued in it has been waited exactly once and
    every forked copy stream is joined into the current stream (module docstring, Boundary).
    Protocols subclass it: their copies, layouts and channel maps call :meth:`open`,
    :meth:`mark_pushed`, :meth:`wait_ready` and :meth:`done`; the op body calls :meth:`reset`
    on its ``except`` path.

    Args:
        pool: The signal pool (its ``signal`` / ``wait`` carry the deadline).
        world_size: Ranks of the group.
        rank: This process's rank in the group.
        device: The CUDA device the copy streams run on.
        slots: Transport slots (independent boundary bookkeeping each).
        arrival_order: The remote sources in the order their pushes of one channel reach this
            rank: the destination-side mirror of the protocol's fixed push order (see
            :meth:`_arrived`). Every peer exactly once.
        per_peer_streams: One copy stream per destination (default) or one for all.
        stream_priority: CUDA priority of the copy streams.
    """

    def __init__(
        self,
        pool: SymmMemPool,
        world_size: int,
        rank: int,
        device: torch.device,
        *,
        slots: int,
        arrival_order: Sequence[int],
        per_peer_streams: bool = True,
        stream_priority: int = 0,
    ) -> None:
        if pool is None:
            raise ValueError(f"{type(self).__name__} needs a SymmMemPool; got None.")
        if world_size < 2 or not 0 <= rank < world_size:
            raise ValueError(
                f"{type(self).__name__} needs world_size >= 2 and 0 <= rank < world_size (got "
                f"{rank}/{world_size})."
            )
        if slots < 1:
            raise ValueError(f"{type(self).__name__} needs slots >= 1 (got {slots}).")
        self.arrival_order: tuple[int, ...] = tuple(int(s) for s in arrival_order)
        """tuple[int, ...]: Remote sources in the order one channel's pushes reach this rank."""
        if sorted(self.arrival_order) != [p for p in range(world_size) if p != rank]:
            raise ValueError(
                f"{type(self).__name__}: arrival_order must list every peer of rank {rank}/"
                f"{world_size} exactly once (got {self.arrival_order})."
            )
        self.pool = pool
        self.world_size = world_size
        self.rank = rank
        self.device = device
        self.slots = slots
        self.streams = CopyStreams(
            world_size, rank, device, per_peer=per_peer_streams, priority=stream_priority
        )
        self._in_flight = [False] * slots
        self._pending: list[set[tuple[int, int]]] = [set() for _ in range(slots)]  # (src, ch) owed
        self._pushed_to: list[set[tuple[int, int]]] = [set() for _ in range(slots)]  # (dst, ch) put
        self._views: dict[tuple, torch.Tensor] = {}  # key[0] is the slot; valid for one allocation
        self._capturing = [False] * slots  # per slot: its boundary is open under capture
        self._captured = False  # a captured boundary COMPLETED: a graph holds the pool's addresses
        self._needs_resync = False  # a reset() happened; the next eager open() agrees the counts
        self._boundaries = 0  # completed boundaries (done or reset): the resync counter

    # --- allocation -------------------------------------------------------------------------

    def _covers(self, slot_nbytes: int, slots: int) -> bool:
        """Whether the pool's current allocation holds ``slots`` slots of ``slot_nbytes`` bytes
        (``False`` before the first reservation)."""
        if not self.pool.reserved:
            return False
        want = PoolLayout.for_request(slot_nbytes, slots, self.world_size, self.pool.align)
        return self.pool.slots >= slots and self.pool.slot_nbytes >= want.slot_nbytes

    def _reserve(self, slot_nbytes: int, slots: int, min_channels: int) -> bool:
        """``pool.reserve`` under the shared rules; returns whether the pool (re)allocated.

        Raises with a boundary in flight and when the reservation would GROW an allocation
        after a captured boundary completed (growth frees the buffers the graph holds). A first
        reservation of a never-reserved pool frees nothing and is allowed after captures. The
        pool itself refuses any reservation under capture; that stays the only capture-time
        refusal. Cached views are dropped before the pool frees and after it reallocated.
        Protocol-specific checks (an exact region size, a first-use reset) wrap this call in
        the subclass. ``slot_nbytes`` and ``slots`` must be rank-symmetric (plan ints the
        ranks agreed on): the growth guard raises locally, before the pool's collective
        reserve, so a request that grows on one rank only would leave its peers in the pool's
        all-gather.
        """
        if any(self._in_flight):
            raise RuntimeError(
                f"{type(self).__name__}: reserve with a boundary in flight; call done() first."
            )
        if self.pool.reserved and not self._covers(slot_nbytes, slots):
            if self._captured:
                need = PoolLayout.for_request(slot_nbytes, slots, self.world_size, self.pool.align)
                raise RuntimeError(
                    f"{type(self).__name__}: the symmetric pool would grow after a CUDA graph "
                    f"captured its buffers (the pool holds {self.pool.slots} x "
                    f"{self.pool.slot_nbytes} bytes, the request needs {slots} x "
                    f"{need.slot_nbytes}); warm up the largest shape first, or run the warm-up "
                    "with capture disabled before capturing."
                )
            # Drop the views into the current allocation first so the pool can free it before
            # allocating the larger one (otherwise both are mapped at once).
            self._drop_views()
        realloc = self.pool.reserve(slot_nbytes, slots, min_channels)  # collective, eager only
        if realloc:
            self._drop_views()
        return realloc

    def _drop_views(self, slot: int | None = None) -> None:
        """Forget the cached views of every slot, or of ``slot`` only (a slot whose layout
        changes while another slot is in flight keeps that slot's views)."""
        if slot is None:
            self._views.clear()
            return
        for key in [key for key in self._views if key[0] == slot]:
            del self._views[key]

    def view(self, key: tuple, make: Callable[[], torch.Tensor]) -> torch.Tensor:
        """The memoized view ``make()`` for ``key``, whose first element is the slot it belongs
        to; valid for one allocation (dropped by :meth:`_reserve` on reallocation) and one slot
        layout (the protocol drops a slot's views when it rebinds the layout)."""
        view = self._views.get(key)
        if view is None:
            view = self._views[key] = make()
        return view

    # --- the boundary -----------------------------------------------------------------------

    def _check_slot(self, slot: int) -> None:
        if not 0 <= slot < self.slots:
            raise ValueError(f"slot {slot} out of range (transport has {self.slots} slots).")

    def _require_in_flight(self, slot: int, op: str) -> None:
        self._check_slot(slot)
        if not self._in_flight[slot]:
            raise RuntimeError(
                f"{type(self).__name__}.{op}(slot={slot}) needs an open({slot}) in flight."
            )

    def _check_peer(self, peer: int, who: str) -> None:
        if peer == self.rank or not 0 <= peer < self.world_size:
            raise ValueError(
                f"{type(self).__name__}.{who}: rank {peer} is not a peer of rank {self.rank}/"
                f"{self.world_size}."
            )

    def _check_channel(self, channel: int, who: str) -> None:
        # Where the protocol declares a channel, not in the pool's wait inside the release: a
        # raise there would leave the slot in flight with its waits half enqueued.
        if not 0 <= channel < self.pool.channels:
            raise ValueError(
                f"{type(self).__name__}.{who}: channel {channel} out of range (the pool offers "
                f"{self.pool.channels} data channels)."
            )

    def open(self, slot: int, expected: Iterable[tuple[int, int]]) -> None:
        """Start a boundary on ``slot``, which owes the waits ``expected`` (``(src, channel)``
        pairs, one per flag a peer will put to this rank in it).

        Refuses a slot still in flight, a source that is not a peer and a channel the pool
        lacks, before anything is in flight. After a :meth:`reset`, agrees the count of
        completed boundaries with the group once (eager only: under capture the latched
        transport raises and the caller runs one eager call first). Notes a capture on the
        slot, which :meth:`done` promotes to "captured". Called by the protocol AFTER its own
        host checks, so a host raise leaves no slot in flight. Enqueues nothing.
        """
        self._check_slot(slot)
        if self._in_flight[slot]:
            raise RuntimeError(
                f"{type(self).__name__}.open(slot={slot}) before done({slot}) of its previous "
                "use: the peers would wait for its flags forever."
            )
        owed = {(int(src), int(ch)) for src, ch in expected}
        for src, ch in owed:
            self._check_peer(src, "open")
            self._check_channel(ch, "open")
        capturing = _is_capturing()
        if self._needs_resync:
            if capturing:
                raise RuntimeError(
                    f"{type(self).__name__}.open under CUDA-graph capture after a failed boundary: "
                    "the ranks must agree on their boundary count first; run one eager call "
                    "before capturing."
                )
            self._resync()
        self._capturing[slot] = capturing
        self._pending[slot] = owed
        self._pushed_to[slot] = set()
        self._in_flight[slot] = True

    def mark_pushed(self, slot: int, dst: int, channel: int) -> None:
        """Put flag ``(channel, me -> dst)`` on the stream the caller is on (``dst``'s copy
        stream, behind the copies it enqueued there) and record the put for ``slot``.

        The copies are the protocol's; this orders the put after them. The put is the pool's
        ``signal`` unless the subclass overrides :meth:`_put`. The destination and the channel
        are checked before the put; the protocol must reject a duplicate destination BEFORE it
        enqueues anything: a check here would run after the copies, where it protects nothing.
        """
        self._require_in_flight(slot, "mark_pushed")
        self._check_peer(dst, "mark_pushed")
        self._check_channel(channel, "mark_pushed")
        self._put(dst, channel)
        self._pushed_to[slot].add((dst, int(channel)))

    def _put(self, dst: int, channel: int) -> None:
        """The flag write behind the copies on the current stream (overridable: a protocol with
        an SM-free signal writes the pad word itself)."""
        self.pool.signal(dst, channel)

    def wait_ready(self, slot: int, src: int, channel: int) -> None:
        """Make the current stream wait flag ``(channel, src -> me)`` once per boundary; a
        second call for the same pair in the boundary is a no-op (the flag was consumed)."""
        self._require_in_flight(slot, "wait_ready")
        pair = (src, int(channel))
        if pair in self._pending[slot]:
            self.pool.wait(src, channel)
            self._pending[slot].discard(pair)

    def _consume(self, slot: int, pairs: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
        """Book ``pairs`` as waited without enqueuing the pool's per-source wait (a protocol that
        waits a whole group of flags in one launch calls this with the pairs that launch
        covers); returns the pairs that were still owed, in the given order."""
        self._require_in_flight(slot, "_consume")
        owed = [(src, int(ch)) for src, ch in pairs if (src, int(ch)) in self._pending[slot]]
        self._pending[slot].difference_update(owed)
        return owed

    def _release(self, slot: int) -> None:
        # IN FLIGHT -> IDLE, from done() and reset() alike: wait every owed flag that no wait
        # consumed (one put per wait), join the forked copy streams, clear the slot. A raise
        # (a wait the pool refuses, a join the capture refuses) propagates, but the slot goes
        # IDLE regardless, with the pairs already waited dropped (never waited twice) and the
        # boundary not counted: reset() and close() stay possible, and the resync latch, which
        # reset() sets first, catches a rank that completed fewer boundaries than its peers.
        pending = self._pending[slot]
        try:
            for pair in sorted(pending):
                self.pool.wait(*pair)
                pending.discard(pair)
            self.streams.join_forked()
            self._boundaries += 1
        finally:
            pending.clear()
            self._pushed_to[slot] = set()
            self._in_flight[slot] = False

    def done(self, slot: int) -> None:
        """Complete the boundary on ``slot``: release it (module docstring, Boundary) and, when it
        ran under capture, mark the transport captured. Call after the last consumer of the
        boundary was enqueued on the current stream."""
        self._require_in_flight(slot, "done")
        self._release(slot)
        if self._capturing[slot]:
            self._captured = True
        self._capturing[slot] = False

    def _arrived(self, slot: int) -> set[tuple[int, int]]:
        """The ``(src, channel)`` flags the PEERS put to this rank before a rank-symmetric raise,
        derived from this rank's own puts.

        Every protocol pushes each channel to its destinations in one fixed order per boundary,
        and under rank symmetry every rank raised after the same number of pushes. When this
        rank pushed channel ``ch`` to its first ``m`` destinations, the sources whose push of
        ``ch`` reached it are ``arrival_order[:m]`` -- the destination-side mirror of the push
        order (a sender permutation ``me + 1, me + 2, ...`` reaches destination ``d`` from
        ``d - 1, d - 2, ...``; the push order ``me - 1, me - 2, ...`` of the TP pair arrives from
        ``me + 1, me + 2, ...``). This covers a raise inside any channel's push loop after any
        number of destinations; it is the one rule, not overridden per protocol.
        """
        by_channel: dict[int, int] = {}
        for _dst, channel in self._pushed_to[slot]:
            by_channel[channel] = by_channel.get(channel, 0) + 1
        return {(src, ch) for ch, m in by_channel.items() for src in self.arrival_order[:m]}

    def reset(self) -> None:
        """The op's ``except`` path: release every slot in flight with its owed waits trimmed to
        the flags the peers did put (:meth:`_arrived`), drop an open capture, join every copy
        stream (eager only), and latch the resync the next eager :meth:`open` runs.

        Repairs the protocol when every rank raised at the same point; a peer that already
        trapped cannot be helped. A slot not in flight is left alone: its flags are balanced.
        The resync is latched first, so it holds even when a release raises.
        """
        self._needs_resync = True
        try:
            for slot in range(self.slots):
                if self._in_flight[slot]:
                    self._pending[slot] &= self._arrived(slot)
                    self._release(slot)
        finally:
            self._capturing = [False] * self.slots
        if not _is_capturing():
            self.streams.join_all_eager()

    def _resync(self) -> None:
        """Collective (eager): all-gather the count of completed boundaries; equal on every rank
        clears the latch, unequal raises :class:`CeResyncError` naming the ranks (the op body
        turns that into :meth:`CeState.disable`: the group continues on the NCCL path instead of
        trapping). A warning names the gather first: after a ONE-SIDED failure the peers never
        enter it, this rank blocks in the collective until the process group's timeout and the
        peers trap at the deadline on the flags it did not put; the log line is what tells the
        operator which rank failed (the peers' trap names a push / consume imbalance)."""
        logger.warning(
            f"{type(self).__name__}[{self.pool.group_name}]: resync after a failed boundary "
            f"(this rank completed {self._boundaries}): all-gathering the boundary counts; a "
            "peer that did not fail will not answer, and its flag waits trap at the deadline."
        )
        counts = _gather_boundary_counts(self.pool.group, self._boundaries)
        if len(set(counts)) != 1:
            detail = ", ".join(f"rank {r}: {c}" for r, c in enumerate(counts))
            raise CeResyncError(
                f"{type(self).__name__}: the ranks disagree on the number of completed boundaries "
                f"after a failed one ({detail}); a retry after a one-sided failure is not safe "
                "on the copy-engine path."
            )
        self._needs_resync = False

    # --- lifecycle ----------------------------------------------------------------------------

    def drain(self) -> None:
        """Check that no boundary is in flight. Nothing is pending on a copy stream after
        :meth:`done`, so this touches no stream (safe under capture)."""
        for slot in range(self.slots):
            if self._in_flight[slot]:
                raise RuntimeError(
                    f"{type(self).__name__}.drain with a boundary in flight on slot {slot}."
                )

    def close(self) -> None:
        """Drain, drop the cached views and release the pool (collective by symmetry: every
        rank calls it). Must FOLLOW the CUDA-graph runners' ``clear()``: a graph may reference
        the pool; the caller orders it so. Idempotent."""
        self.drain()
        self._drop_views()
        self.pool.release()


# =============================================================================
# Per-group state and registry
# =============================================================================


class CeState:
    """Per-group lifecycle shared by every protocol: probe once (rank-agreed), ``effective`` /
    ``reason``, one warning naming the fallback, closed-state refusal, and a registry keyed by
    group name -- ONE registry per subclass, so two protocols over one group never collide.

    A subclass builds its pool(s) and transport in :meth:`_make_transport`; the op body looks the
    state up with :meth:`get`, checks ``effective`` rank-symmetrically and runs the protocol
    inside ``try: ... except Exception: state.transport.reset(); raise``.

    Args:
        group: The process group.
        group_name: ``group.group_name`` (the registry key and the op's handle).
        timeout_ms: Device-side deadline of every signal wait of the pools the state builds;
            the pool's default (module docstring, Rules: the Ulysses exchange passes 60 s).
    """

    _registry: dict[str, "CeState"] = {}

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls._registry = {}  # one dict per subclass: protocols over one group never collide

    def __init__(
        self, group: dist.ProcessGroup, group_name: str, *, timeout_ms: int = DEFAULT_TIMEOUT_MS
    ) -> None:
        if group is None:
            raise ValueError(
                f"{type(self).__name__} needs a torch.distributed process group; got None."
            )
        if timeout_ms < 0:
            raise ValueError(f"{type(self).__name__} needs timeout_ms >= 0 (got {timeout_ms}).")
        self.group = group
        self.group_name = group_name
        self.timeout_ms = timeout_ms
        self.world_size: int = dist.get_world_size(group)
        self.rank: int = dist.get_rank(group)
        self.effective: bool = False
        """bool: Whether the protocol is in use for this group (False before the probe, after a
        failed one, after :meth:`disable` and after :meth:`close`: the caller keeps its NCCL
        path)."""
        self.reason: str = "not probed yet"
        """str: Why the protocol is not effective, or "" while it is."""
        self.pool: SymmMemPool | None = None
        self.transport: CeTransport | None = None
        self._probed = False
        self._closed = False
        self._warned = False  # the one fallback warning of this state
        self._what = type(self).__name__
        self._device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

    # --- registry -----------------------------------------------------------------------------

    @classmethod
    def get(cls, group_name: str) -> "CeState":
        """The registered state of ``group_name``; raises ``KeyError`` when there is none."""
        state = cls._registry.get(group_name)
        if state is None:
            raise KeyError(group_name)
        return state

    @classmethod
    def register(cls, state: "CeState") -> "CeState":
        """Register ``state`` for its group name, or return the one already registered for it."""
        return cls._registry.setdefault(state.group_name, state)

    @classmethod
    def release(cls, group_name: str) -> None:
        """Close and unregister the state of ``group_name`` (no-op when there is none)."""
        state = cls._registry.get(group_name)
        if state is not None:
            state.close()

    # --- probe --------------------------------------------------------------------------------

    def probe_once(self, *, min_channels: int, what: str) -> bool:
        """Decide ``effective`` for the state's lifetime; later calls return the stored verdict.

        The first call runs ``probe_symmetric_memory`` (collective, rank-agreed: every rank calls
        it, eagerly, in the same order; under capture it raises instead). On success the
        subclass builds pool and transport (:meth:`_make_transport`); on failure ONE warning
        names the fallback: ``"<what> unavailable (<reason>); keeping the NCCL path"``. A
        protocol with its own host pre-check overrides this method and calls it on the pass.
        A raise out of :meth:`_make_transport` stores a failed verdict (no second collective
        probe, no second allocation), releases the pool the subclass reserved before raising
        and propagates.
        """
        self.require_open()
        if self._probed:
            return self.effective
        self._what = what
        if _is_capturing():
            raise RuntimeError(
                f"{what}: the first call of group {self.group_name!r} ran under CUDA-graph "
                "capture; run an eager warm-up call first."
            )
        ok, reason = probe_symmetric_memory(self.group, self._device, min_channels=min_channels)
        if ok:
            try:
                self._make_transport()
                if self.transport is None:
                    raise RuntimeError(
                        f"{type(self).__name__}._make_transport left no transport for group "
                        f"{self.group_name!r}."
                    )
            except Exception as e:
                self._probed = True
                self.effective = False
                self.reason = f"{what} setup failed: {type(e).__name__}: {e}"
                self.transport = None
                self._release_pool()  # collective by symmetry: the raise was rank-symmetric
                raise
        else:
            self._warn(f"{what} unavailable ({reason}); keeping the NCCL path.")
        self.effective = ok
        self.reason = reason
        self._probed = True
        return ok

    def _make_transport(self) -> None:
        """Build :attr:`pool` (``SymmMemPool(self.group, self._device, self.timeout_ms, ...)``)
        and :attr:`transport` after a successful probe; the subclass's job. A raise here must be
        rank-symmetric, as the pool's reserve already assumes: :meth:`probe_once` then releases
        the pool collectively on every rank."""
        raise NotImplementedError(f"{type(self).__name__} must implement _make_transport().")

    def _release_pool(self) -> None:
        pool, self.pool = self.pool, None
        if pool is not None:
            pool.release()

    def _warn(self, message: str) -> None:
        if not self._warned:
            logger.warning(message)
            self._warned = True

    # --- state ------------------------------------------------------------------------------

    def require_open(self) -> None:
        """Raise when the state is closed (a sibling helper still holding it gets a clear error
        rather than a silent fallback)."""
        if self._closed:
            raise RuntimeError(
                f"{self._what}: the state of group {self.group_name!r} is closed; build a new one "
                "for further calls."
            )

    def disable(self, reason: str) -> None:
        """Take the group off the protocol for the rest of the state's lifetime (``effective``
        False, ``reason`` set, the one warning if none was given yet); the pools stay allocated
        until :meth:`close`. The op body calls it when the resync after a failed boundary finds
        the ranks disagreeing, so the group continues on NCCL instead of trapping."""
        self.effective = False
        self.reason = reason
        self._warn(
            f"{self._what} disabled for group {self.group_name!r} ({reason}); keeping the NCCL "
            "path."
        )

    def close(self) -> None:
        """Close the transport (releasing the pool; collective by symmetry when one exists),
        forget it, mark the state closed and unregister it. Idempotent. Call on every rank before
        the process group is destroyed, after the CUDA-graph runners' ``clear()``. A pool held
        without a transport (a subclass's setup raised after reserving it) is released too."""
        if self.transport is not None:
            self.transport.close()
        self.transport = None
        self._release_pool()
        self.effective = False
        self.reason = "closed"
        self._closed = True
        registry = type(self)._registry
        if registry.get(self.group_name) is self:
            del registry[self.group_name]
