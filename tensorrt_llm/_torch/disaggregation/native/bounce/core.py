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
"""Pure core of the KV bounce subsystem: the transport contract and the per-region lifetime state
machine. No CUDA or NIXL imports, so it is unit-testable on CPU."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple


class TransferState(Enum):
    """Lifecycle of one bounced receive region."""

    INIT = "init"  # leased, not yet advertised to any sender
    ACTIVE = "active"  # writers in flight
    SCATTERING = "scattering"  # all writers succeeded; scattering back into the cache
    COMPLETED = "completed"  # scattered cleanly, slot released
    FAILED = "failed"  # all writers done, at least one failed, slot released
    QUARANTINED = "quarantined"  # a writer is in doubt, slot held out of reuse


class ScatterState(Enum):
    IDLE = "idle"
    QUEUED = "queued"
    DONE = "done"
    FAILED = "failed"


class Disposition(Enum):
    """What to do with the slot when the transfer finishes."""

    RELEASE = "release"  # drained: return it to the free pool
    QUARANTINE = "quarantine"  # in doubt: hold it out of the pool for a grace period


@dataclass
class Settlement:
    """What the transport carries out on finish: the slot, whether to release or quarantine it,
    whether the transfer succeeded, and the callback to fire. A just-in-time context releases its
    per-writer slots one by one, so its settlement carries no slot (None)."""

    slot_id: Optional[int]
    disposition: Disposition
    success: bool
    on_done: Optional[Callable[[bool], None]]


@dataclass
class TransferContext:
    """Lifetime state machine for one bounced receive region. The region is released only after every
    writer reports (success or failure both mean its write has drained); a writer given up on (a
    cancelled or timed-out transfer, whose one-sided write cannot be aborted) makes it quarantined
    instead of reused."""

    rid_slice: Tuple[int, int]
    slot_id: int
    base_addr: int
    per_writer_bytes: int
    num_writers: int
    on_done: Optional[Callable[[bool], None]] = None
    # Exact per-writer layout for a fan-in whose writers send different byte counts (an uneven PP
    # layer split, or layer groups with different slot sizes). None keeps the equal split above.
    writer_offsets: Optional[Tuple[int, ...]] = None
    writer_sizes: Optional[Tuple[int, ...]] = None

    # Whether each writer that reported succeeded, keyed by rank; presence means it reported.
    _writer_ok: Dict[int, bool] = field(default_factory=dict)
    # per successful writer: where it wrote, plus the fragments to scatter back
    _scatter_descs: List[tuple] = field(default_factory=list)
    _writer_cohort: Optional[frozenset[int]] = None
    _publication_failed: bool = False
    _orphaned: bool = False
    scatter_state: ScatterState = ScatterState.IDLE
    state: TransferState = TransferState.INIT
    settled: bool = False

    def writer_base(self, writer_index: int) -> int:
        """Where this fan-in writer writes in the region."""
        if self.writer_offsets is not None:
            return self.base_addr + self.writer_offsets[writer_index]
        return self.base_addr + writer_index * self.per_writer_bytes

    def writer_capacity(self, writer_index: int) -> Optional[int]:
        """The bytes this writer's sub-region holds when the layout is exact, else None."""
        if self.writer_sizes is None:
            return None
        return self.writer_sizes[writer_index]

    def _writers_final(self) -> bool:
        return self.settled or self.state in (
            TransferState.SCATTERING,
            TransferState.COMPLETED,
            TransferState.FAILED,
            TransferState.QUARANTINED,
        )

    def _all_writers_reported(self) -> bool:
        if self._writer_cohort is not None:
            return self._writer_cohort.issubset(self._writer_ok)
        return len(self._writer_ok) >= self.num_writers

    def _all_writers_succeeded(self) -> bool:
        if not self._all_writers_reported():
            return False
        if self._writer_cohort is not None:
            return all(self._writer_ok[rank] for rank in self._writer_cohort)
        return all(self._writer_ok.values())

    # Mutations: call only while holding the transport's reservation lock.
    def record_writer_result(
        self,
        peer_rank: int,
        succeeded: bool,
        *,
        src_base: Optional[int] = None,
        dst_ptrs=None,
        sizes=None,
    ) -> None:
        """Record one writer's terminal report. Repeat or late reports are ignored, so duplicate or
        out-of-order messages are harmless."""
        # A duplicate or late report can still arrive after the writer set is final, because
        # scatter runs on another thread while the region stays live. For example a retransmitted
        # notification, or a stray failure that would flip a good transfer to failed; drop it.
        if self._writers_final() or peer_rank in self._writer_ok:
            return
        if self._writer_cohort is not None and peer_rank not in self._writer_cohort:
            raise RuntimeError(f"writer {peer_rank} is outside the published bounce cohort")
        self._writer_ok[peer_rank] = succeeded
        if succeeded and dst_ptrs is not None and int(dst_ptrs.size) > 0:
            self._scatter_descs.append(
                (src_base if src_base is not None else self.base_addr, dst_ptrs, sizes)
            )
        if self.state is TransferState.INIT:
            self.state = TransferState.ACTIVE

    def mark_orphaned(self) -> None:
        """Give up on writers that never reported (cancel, timeout, shutdown): a one-sided write
        cannot be aborted, so the region is quarantined on settle. No-op once scattering or done."""
        if self._writers_final():
            return
        self._orphaned = True

    def abort_publication(self, published_writers: set[int]) -> None:
        """Close a failed fan-out around only the writers that were published.

        Published writers still have to report terminal evidence. Successful
        partial data is intentionally not scattered because the request is
        already incomplete.
        """
        if self._writers_final():
            return
        published = frozenset(published_writers)
        if len(published) > self.num_writers or not self._writer_ok.keys() <= published:
            raise RuntimeError(
                "published bounce cohort is inconsistent with terminal evidence: "
                f"reported={sorted(self._writer_ok)} published={sorted(published)} "
                f"reserved_count={self.num_writers}"
            )
        self._writer_cohort = published
        self._publication_failed = True

    def begin_scatter(self) -> None:
        self.state = TransferState.SCATTERING
        self.scatter_state = ScatterState.QUEUED

    def sorted_scatter_descs(self) -> List[tuple]:
        """Success fragments ordered by source, for a deterministic scatter."""
        return sorted(self._scatter_descs, key=lambda t: t[0])

    def finish_scatter(self, ok: bool) -> None:
        self.scatter_state = ScatterState.DONE if ok else ScatterState.FAILED

    def ready_to_scatter(self) -> bool:
        """All writers succeeded, there is data to scatter, and scatter has not started."""
        return (
            self.state is TransferState.ACTIVE
            and not self._orphaned
            and not self._publication_failed
            and self._all_writers_succeeded()
            and bool(self._scatter_descs)
        )

    def ready_to_settle(self) -> bool:
        if self.settled:
            return False
        if self._orphaned:
            return True  # in doubt: settle now and quarantine
        if not self._all_writers_reported():
            return False  # a writer has not reported yet
        if self._publication_failed:
            return True  # every published writer drained; incomplete data is discarded
        if self._all_writers_succeeded() and self._scatter_descs:
            return self.scatter_state in (ScatterState.DONE, ScatterState.FAILED)
        return True  # nothing to scatter, or a failure among them: either way drained

    def settle(self) -> Optional[Settlement]:
        """Finalize the transfer once and return the decision, or None if already done. Every
        termination path calls this; only the first call has any effect."""
        if self.settled:
            return None
        self.settled = True
        if self._orphaned:
            self.state = TransferState.QUARANTINED
            return Settlement(self.slot_id, Disposition.QUARANTINE, False, self.on_done)
        success = (
            not self._publication_failed
            and self._all_writers_succeeded()
            and self.scatter_state is not ScatterState.FAILED
        )
        self.state = TransferState.COMPLETED if success else TransferState.FAILED
        return Settlement(self.slot_id, Disposition.RELEASE, success, self.on_done)


class JitVerdict(IntEnum):
    """The receiver's answer to a just-in-time grant request (the wire code of BOUNCE_GRANT)."""

    NO = 0  # never for this flow: the writer takes the per-fragment path
    GRANT = 1  # a slot of the requested size is reserved for this writer
    RETRY = 2  # the region is full right now; ask again


class JitGrantCheck(Enum):
    """What JitTransferContext.grant_check asks the transport to do."""

    NO = "no"  # answer NO
    ALLOCATE = "allocate"  # try to reserve a slot without waiting
    REGRANT = "regrant"  # the writer already holds a slot: answer with the same grant


class WriterLeaseState(Enum):
    """Lifecycle of one writer's slot under the just-in-time grant."""

    NONE = "none"  # no slot
    GRANTED = "granted"  # holds a slot; its write may be in flight
    SCATTERING = "scattering"  # reported success; its fragments are being scattered
    DONE = "done"  # terminal; holds no slot


@dataclass
class WriterLease:
    state: WriterLeaseState = WriterLeaseState.NONE
    slot_id: Optional[int] = None
    base: Optional[int] = None
    nbytes: int = 0
    reported: bool = False
    ok: bool = False


# Side effects a JitTransferContext mutation asks the transport to carry out with its lock released:
# ("scatter", rank, (src_base, dst_ptrs, sizes)), ("release", slot_id, nbytes) and
# ("quarantine", slot_id, nbytes).
JitAction = tuple


@dataclass
class JitTransferContext:
    """Lifetime state machine for one request slice under the just-in-time per-writer grant.

    No slot is held at dispatch. Each writer asks for a slot of its own exact size right before it
    gathers, writes, reports, and its fragments are scattered and its slot released on their own,
    so a slot is held only for one writer's gather, write, result and scatter. The group completes
    (on_done) exactly once, after every writer of the cohort is terminal and no scatter is running.
    """

    rid_slice: Tuple[int, int]
    writer_ranks: Tuple[int, ...]
    # the most bytes each writer may ask for, keyed by its rank
    writer_caps: Dict[int, int]
    on_done: Optional[Callable[[bool], None]] = None

    _leases: Dict[int, WriterLease] = field(default_factory=dict)
    _cohort: Optional[FrozenSet[int]] = None
    _publication_failed: bool = False
    _orphaned: bool = False
    _scatter_failed: bool = False
    settled: bool = False

    def __post_init__(self) -> None:
        if not self._leases:
            self._leases = {rank: WriterLease() for rank in self.writer_ranks}
        if self._cohort is None:
            self._cohort = frozenset(self.writer_ranks)

    def lease(self, rank: int) -> Optional[WriterLease]:
        return self._leases.get(rank)

    def held_slots(self) -> List[Tuple[int, int]]:
        """The (slot_id, nbytes) of every slot a writer holds."""
        return [
            (lease.slot_id, lease.nbytes)
            for lease in self._leases.values()
            if lease.slot_id is not None
        ]

    def is_scattering(self) -> bool:
        return any(lease.state is WriterLeaseState.SCATTERING for lease in self._leases.values())

    # Mutations: call only while holding the transport's reservation lock.
    def grant_check(self, rank: int, nbytes: int) -> JitGrantCheck:
        """Decide a grant request without side effects."""
        if self.settled or self._orphaned or self._publication_failed:
            return JitGrantCheck.NO
        lease = self._leases.get(rank)
        if lease is None or rank not in self._cohort or lease.reported:
            return JitGrantCheck.NO
        if nbytes <= 0 or nbytes > self.writer_caps.get(rank, 0):
            return JitGrantCheck.NO
        if lease.state is WriterLeaseState.GRANTED:
            # A resend after a lost or late reply: hand out the same slot again, if it is enough.
            return JitGrantCheck.REGRANT if nbytes <= lease.nbytes else JitGrantCheck.NO
        if lease.state is not WriterLeaseState.NONE:
            return JitGrantCheck.NO
        return JitGrantCheck.ALLOCATE

    def on_granted(self, rank: int, slot_id: int, base: int, nbytes: int) -> None:
        lease = self._leases[rank]
        lease.state = WriterLeaseState.GRANTED
        lease.slot_id = slot_id
        lease.base = base
        lease.nbytes = nbytes

    def _drop_slot(self, lease: WriterLease, kind: str) -> List[JitAction]:
        actions: List[JitAction] = []
        if lease.slot_id is not None:
            actions.append((kind, lease.slot_id, lease.nbytes))
        lease.slot_id = None
        lease.state = WriterLeaseState.DONE
        return actions

    def record_report(
        self,
        rank: int,
        ok: bool,
        *,
        src_base: Optional[int] = None,
        dst_ptrs=None,
        sizes=None,
    ) -> List[JitAction]:
        """Record one writer's terminal report (success or failure both mean its write drained).
        Duplicate and late reports are ignored."""
        lease = self._leases.get(rank)
        if self.settled or lease is None or lease.reported:
            return []
        lease.reported = True
        has_tail = dst_ptrs is not None and int(dst_ptrs.size) > 0
        if ok and has_tail and lease.state is not WriterLeaseState.GRANTED:
            # A coalesced write this context did not grant (or already gave up on): its data is not
            # where the scatter could read it, so the transfer cannot be complete.
            ok = False
        if ok and has_tail and src_base is not None and src_base != lease.base:
            ok = False  # the writer wrote somewhere other than its grant
        lease.ok = ok
        if (
            ok
            and has_tail
            and not self._orphaned
            and not self._publication_failed
            and rank in self._cohort
        ):
            lease.state = WriterLeaseState.SCATTERING
            return [("scatter", rank, (lease.base, dst_ptrs, sizes))]
        return self._drop_slot(lease, "release")

    def finish_writer_scatter(self, rank: int, ok: bool) -> List[JitAction]:
        """A writer's scatter finished: release its slot. A failed scatter fails the group."""
        lease = self._leases.get(rank)
        if lease is None or lease.state is not WriterLeaseState.SCATTERING:
            return []
        if not ok:
            self._scatter_failed = True
            lease.ok = False
        return self._drop_slot(lease, "release")

    def mark_orphaned(self) -> List[JitAction]:
        """Give up on writers that never reported (cancel, timeout, shutdown). A granted writer's
        one-sided write cannot be aborted, so its slot is quarantined; a scattering writer releases
        at finish; a writer with no slot gets NO for any later grant."""
        if self.settled:
            return []
        self._orphaned = True
        actions: List[JitAction] = []
        for lease in self._leases.values():
            if lease.state is WriterLeaseState.GRANTED:
                actions += self._drop_slot(lease, "quarantine")
        return actions

    def abort_publication(self, published_writers: set[int]) -> List[JitAction]:
        """Close a failed fan-out around only the writers that were published (the same rule as
        TransferContext.abort_publication). Partial data is not scattered."""
        if self.settled:
            return []
        published = frozenset(published_writers)
        reported = {rank for rank, lease in self._leases.items() if lease.reported}
        if len(published) > len(self.writer_ranks) or not reported <= published:
            raise RuntimeError(
                "published bounce cohort is inconsistent with terminal evidence: "
                f"reported={sorted(reported)} published={sorted(published)} "
                f"reserved_count={len(self.writer_ranks)}"
            )
        self._cohort = published
        self._publication_failed = True
        return []

    def maybe_settle(self) -> Optional[Settlement]:
        """Settle exactly once: when every cohort writer is terminal (or the context is orphaned)
        and no scatter is running."""
        if self.settled or self.is_scattering():
            return None
        if not self._orphaned and not all(
            self._leases[rank].reported for rank in self._cohort if rank in self._leases
        ):
            return None
        self.settled = True
        success = (
            not self._orphaned
            and not self._publication_failed
            and not self._scatter_failed
            and all(self._leases[rank].ok for rank in self._cohort if rank in self._leases)
        )
        return Settlement(None, Disposition.RELEASE, success, self.on_done)


def stage_layer_ranges(layer_num_per_pp: Sequence[int]) -> List[Tuple[int, int]]:
    """The [start, end) layer range of every PP stage, from the per-stage layer counts (the same
    prefix sum PeerRegistrar.get_peer_overlap uses)."""
    ranges: List[Tuple[int, int]] = []
    start = 0
    for count in layer_num_per_pp:
        ranges.append((start, start + int(count)))
        start += int(count)
    return ranges


def pp_fanin_writer_block_bytes(
    layer_bytes_per_group: Sequence[Optional[Dict[int, int]]],
    writer_stages: Sequence[int],
    layer_num_per_pp: Sequence[int],
) -> Optional[List[List[int]]]:
    """Per fan-in writer and receiver layer group, the bytes that writer sends for one block.

    ``layer_bytes_per_group[g]`` maps each global layer id of receiver group g to the bytes one
    block of that layer holds on the receiver (None for a non-paged group, which contributes 0 here
    and is guarded by the transport). ``writer_stages[i]`` is the PP stage of fan-in writer i, whose
    layers are its range in ``layer_num_per_pp``. A layer outside every writer's range is written by
    nobody and is not counted. Returns None when a writer names an unknown stage."""
    ranges = stage_layer_ranges(layer_num_per_pp)
    out: List[List[int]] = []
    for stage in writer_stages:
        if stage < 0 or stage >= len(ranges):
            return None
        start, end = ranges[stage]
        out.append(
            [
                0
                if layer_bytes is None
                else sum(b for layer, b in layer_bytes.items() if start <= layer < end)
                for layer_bytes in layer_bytes_per_group
            ]
        )
    return out


class BounceTransport(ABC):
    """Contract both transports implement so they cannot drift; the factory returns one, and the rest
    of the code depends only on this interface."""

    #: True for a real transport, False for the disabled null object.
    enabled: bool = False

    @property
    def can_send(self) -> bool:
        """Whether the transport has a send region (TRTLLM_KV_BOUNCE_DIRECTION)."""
        return False

    @property
    def can_recv(self) -> bool:
        """Whether the transport has a receive region (TRTLLM_KV_BOUNCE_DIRECTION)."""
        return False

    @abstractmethod
    def build_request(self, write_meta):
        """Gather the request's cache into a send region and build the coalesced write, or None to
        fall back to the per-fragment path."""

    @abstractmethod
    def release_send(self, slot_id) -> None:
        """Release a send region after its write completes."""

    @abstractmethod
    def reserve(
        self,
        recv_req,
        num_writers: int = 1,
        *,
        timeout: Optional[float] = None,
        extra_bytes: int = 0,
        writers_split_layers: bool = True,
        writer_block_bytes: Optional[Sequence[Sequence[int]]] = None,
    ) -> bool:
        """Reserve a region and record its address for the senders. False falls back to per-fragment.
        ``extra_bytes`` sizes the sender's non-paged payload (mamba/KDA recurrent state) that rides
        the same coalesced write as the KV blocks. ``writers_split_layers`` says whether the fan-in
        writers hold different layers (PP fan-in) rather than a head slice of every layer.
        ``writer_block_bytes`` (one list per writer, one entry per layer group) gives each writer's
        exact bytes per block, for a fan-in whose writers send different sizes."""

    def register_jit(
        self,
        recv_req,
        *,
        writer_ranks: Sequence[int],
        extra_bytes: int = 0,
        writers_split_layers: bool = True,
        writer_block_bytes: Optional[Sequence[Sequence[int]]] = None,
    ) -> bool:
        """Register a request slice for the just-in-time per-writer grant without reserving any
        memory (TRTLLM_KV_BOUNCE_JIT). False falls back to the per-fragment path."""
        return False

    def grant(self, rid_slice, rank: int, nbytes: int) -> Tuple[int, int, int]:
        """Answer one writer's grant request without blocking: (verdict, base, capacity)."""
        return int(JitVerdict.NO), 0, 0

    @abstractmethod
    def writer_base(self, rid_slice, writer_index: int) -> Optional[int]:
        """Where the given fan-in writer writes in the region."""

    def writer_capacity(self, rid_slice, writer_index: int) -> Optional[int]:
        """The size of the given writer's sub-region when the layout is exact, else None."""
        return None

    @abstractmethod
    def is_bounced(self, rid_slice) -> bool:
        """Whether this request and slice name a live bounced region."""

    @abstractmethod
    def release_idle_reservation(self, rid_slice) -> None:
        """Immediately free a reservation cancelled before any address went out."""

    @abstractmethod
    def orphan_reservation(self, rid_slice) -> None:
        """Give up on an in-flight reservation (cancel/timeout/lost result); quarantine, don't leak."""

    @abstractmethod
    def abort_publication(self, rid_slice, published_writers: set[int]) -> None:
        """Limit a failed fan-out to writers whose REQUEST_DATA was queued."""

    @abstractmethod
    def record_result(
        self, rid_slice, peer_rank, dst_ptrs=None, sizes=None, src_base=None, on_done=None
    ) -> None:
        """Handle a writer's success; scatter and finalize once all writers reported."""

    @abstractmethod
    def record_failure(self, rid_slice, peer_rank) -> None:
        """Handle a writer's failure; free the region once all writers reported."""

    @abstractmethod
    def close(self) -> None:
        """Stop the scatter worker, deregister memory, free the arenas."""
