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
from bisect import bisect_right
from dataclasses import InitVar, dataclass, field
from enum import Enum
from heapq import merge
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

_MAX_ADDRESS = (1 << 64) - 1


class GatherSourceInDoubtError(RuntimeError):
    """A failed gather may still access its source KV pages.

    Callers must retain the source request/KV owner until local CUDA
    quiescence is established; this is not an ordinary terminal transfer
    failure.
    """


class TransferState(Enum):
    """Lifecycle of one bounced receive region."""

    INIT = "init"  # leased, not yet advertised to any sender
    ACTIVE = "active"  # writers in flight
    SCATTERING = "scattering"  # all writers succeeded; scattering back into the cache
    COMPLETED = "completed"  # scattered cleanly, slot released
    FAILED = "failed"  # all writers done, at least one failed, slot released
    QUARANTINED = "quarantined"  # logical failure; an exposed writer is still in doubt


class ScatterState(Enum):
    IDLE = "idle"
    QUEUED = "queued"
    DONE = "done"
    FAILED = "failed"


class WriterState(Enum):
    """Physical-access state for one planned writer."""

    UNEXPOSED = "unexposed"
    EXPOSED = "exposed"
    NO_ACCESS = "no_access"
    TERMINAL = "terminal"


class Disposition(Enum):
    """What to do with the slot when the transfer finishes."""

    RELEASE = "release"  # drained: return it to the free pool
    QUARANTINE = "quarantine"  # in doubt: hold it until explicit quiescence evidence


@dataclass
class Settlement:
    """What the transport carries out on finish: the slot, whether to release or quarantine it,
    whether the transfer succeeded, and the callback to fire."""

    slot_id: int
    disposition: Disposition
    success: bool
    on_done: Optional[Callable[[bool], None]]


@dataclass
class RecvBounceContext:
    """Lifetime state machine for one bounced receive region.

    ``writer_ranks`` is the exact ordered publication plan. Logical failure suppresses scatter and
    prevents publication to writers that were not exposed yet, but it does not retire an exposed
    writer. The slot remains owned until every possibly exposed writer reports a terminal result or
    backend-wide quiescence provides equivalent evidence.
    """

    rid_slice: Tuple[int, int]
    slot_id: int
    base_addr: int
    per_writer_bytes: int
    writer_ranks: Tuple[int, ...]
    destination_intervals: InitVar[Optional[Iterable[tuple[int, int]]]] = None
    on_done: Optional[Callable[[bool], None]] = None
    on_logical_failure: Optional[Callable[[], None]] = None

    # Whether each physically terminal writer succeeded, keyed by its exact planned rank.
    _writer_ok: Dict[int, bool] = field(default_factory=dict)
    _writer_state: Dict[int, WriterState] = field(init=False)
    # Per successful bounced writer: where it wrote, plus the fragments to scatter back.
    _scatter_descs: Dict[int, tuple] = field(default_factory=dict)
    _scatter_destination_intervals: Tuple[tuple[int, int], ...] = field(default_factory=tuple)
    _logical_failed: bool = False
    _backend_quiesced: bool = False
    _protocol_conflict: bool = False
    _destination_intervals: Optional[Tuple[tuple[int, int], ...]] = field(init=False, default=None)
    scatter_state: ScatterState = ScatterState.IDLE
    state: TransferState = TransferState.INIT
    settled: bool = False
    _logical_failure_notification_in_progress: bool = False
    _logical_failure_notification_delivered: bool = False

    def __post_init__(self, destination_intervals: Optional[Iterable[tuple[int, int]]]) -> None:
        self.writer_ranks = tuple(self.writer_ranks)
        if not self.writer_ranks:
            raise ValueError("RecvBounceContext requires at least one writer rank")
        if any(
            not isinstance(rank, int) or isinstance(rank, bool) or rank < 0
            for rank in self.writer_ranks
        ):
            raise ValueError(
                f"RecvBounceContext writer ranks must be non-negative integers: {self.writer_ranks}"
            )
        if len(set(self.writer_ranks)) != len(self.writer_ranks):
            raise ValueError(f"RecvBounceContext writer ranks must be unique: {self.writer_ranks}")
        self._writer_state = {rank: WriterState.UNEXPOSED for rank in self.writer_ranks}
        if destination_intervals is not None:
            normalized_intervals = []
            for start, size in destination_intervals:
                start, size = int(start), int(size)
                end = start + size
                if (
                    start <= 0
                    or start > _MAX_ADDRESS
                    or size <= 0
                    or size > _MAX_ADDRESS
                    or end > _MAX_ADDRESS + 1
                ):
                    raise ValueError(f"invalid trusted destination interval ({start}, {size})")
                normalized_intervals.append((start, end))
            if not normalized_intervals:
                raise ValueError("trusted destination intervals must not be empty")
            merged_intervals = []
            for start, end in sorted(normalized_intervals):
                if merged_intervals and start <= merged_intervals[-1][1]:
                    previous_start, previous_end = merged_intervals[-1]
                    merged_intervals[-1] = (previous_start, max(previous_end, end))
                else:
                    merged_intervals.append((start, end))
            authorized_bytes = sum(end - start for start, end in merged_intervals)
            reserved_bytes = self.per_writer_bytes * len(self.writer_ranks)
            if authorized_bytes != reserved_bytes:
                raise ValueError(
                    "trusted destination interval union covers "
                    f"{authorized_bytes} bytes, but the bounce reservation covers "
                    f"{reserved_bytes} bytes"
                )
            self._destination_intervals = tuple(merged_intervals)

    @staticmethod
    def _non_overlapping_destination_intervals(
        dst_values: Tuple[int, ...], size_values: Tuple[int, ...]
    ) -> Optional[Tuple[tuple[int, int], ...]]:
        intervals = []
        for dst, size in zip(dst_values, size_values, strict=True):
            end = dst + size
            if (
                dst <= 0
                or dst > _MAX_ADDRESS
                or size <= 0
                or size > _MAX_ADDRESS
                or end > _MAX_ADDRESS + 1
            ):
                return None
            intervals.append((dst, end))
        if not intervals:
            return None
        intervals.sort()
        if any(
            start < previous_end for (_, previous_end), (start, _) in zip(intervals, intervals[1:])
        ):
            return None
        return tuple(intervals)

    def _scatter_destinations_disjoint(
        self, candidate_intervals: Tuple[tuple[int, int], ...]
    ) -> bool:
        existing_index = 0
        for start, end in candidate_intervals:
            while (
                existing_index < len(self._scatter_destination_intervals)
                and self._scatter_destination_intervals[existing_index][1] <= start
            ):
                existing_index += 1
            if (
                existing_index < len(self._scatter_destination_intervals)
                and self._scatter_destination_intervals[existing_index][0] < end
            ):
                return False
        return True

    def _scatter_destinations_authorized(
        self, dst_values: Tuple[int, ...], size_values: Tuple[int, ...]
    ) -> bool:
        if self._destination_intervals is None:
            return True
        for dst, size in zip(dst_values, size_values, strict=True):
            end = dst + size
            if (
                dst <= 0
                or dst > _MAX_ADDRESS
                or size <= 0
                or size > _MAX_ADDRESS
                or end > _MAX_ADDRESS + 1
            ):
                return False
            interval_index = bisect_right(self._destination_intervals, (dst, _MAX_ADDRESS + 1)) - 1
            if interval_index < 0:
                return False
            start, interval_end = self._destination_intervals[interval_index]
            if not (start <= dst and end <= interval_end):
                return False
        return True

    @property
    def logical_failed(self) -> bool:
        return self._logical_failed

    @property
    def pending_exposed_writers(self) -> Tuple[int, ...]:
        return tuple(
            rank for rank in self.writer_ranks if self._writer_state[rank] is WriterState.EXPOSED
        )

    def writer_base(self, peer_rank: int) -> int:
        """Return the planned sub-region base for ``peer_rank``."""
        try:
            writer_index = self.writer_ranks.index(peer_rank)
        except ValueError as e:
            raise KeyError(f"unexpected bounce writer rank {peer_rank}") from e
        return self.base_addr + writer_index * self.per_writer_bytes

    def _writers_final(self) -> bool:
        return self.settled or self.state in (
            TransferState.SCATTERING,
            TransferState.COMPLETED,
            TransferState.FAILED,
        )

    def _all_writers_terminal(self) -> bool:
        return all(
            state in (WriterState.NO_ACCESS, WriterState.TERMINAL)
            for state in self._writer_state.values()
        )

    def _all_writers_succeeded(self) -> bool:
        return self._all_writers_terminal() and all(
            self._writer_ok.get(rank, False) for rank in self.writer_ranks
        )

    def set_on_done(self, on_done: Optional[Callable[[bool], None]]) -> None:
        if on_done is not None:
            self.on_done = on_done

    def set_on_logical_failure(self, on_logical_failure: Optional[Callable[[], None]]) -> None:
        """Install the notification used when launched local CUDA work fails.

        This notification is deliberately separate from ``on_done``.  The
        latter acknowledges physical slot settlement, which is not safe after
        a scatter error without a positive CUDA quiescence fence.
        """
        if on_logical_failure is not None:
            self.on_logical_failure = on_logical_failure

    def begin_logical_failure_notification(self) -> Optional[Callable[[], None]]:
        """Claim the one retryable logical-failure notification, if ready."""
        if (
            self.scatter_state is not ScatterState.FAILED
            or self.on_logical_failure is None
            or self._logical_failure_notification_delivered
            or self._logical_failure_notification_in_progress
        ):
            return None
        self._logical_failure_notification_in_progress = True
        return self.on_logical_failure

    def finish_logical_failure_notification(self, delivered: bool) -> None:
        """Commit or reopen a claimed logical-failure notification."""
        if not self._logical_failure_notification_in_progress:
            return
        self._logical_failure_notification_in_progress = False
        if delivered:
            self._logical_failure_notification_delivered = True

    def mark_writer_exposed(self, peer_rank: int) -> bool:
        """Enter the publication boundary for a planned writer.

        False means publication must be suppressed because the writer is unknown, the context is
        logically failed, or its operation is already terminal. Repeated calls for an already
        exposed writer are idempotent.
        """
        writer_state = self._writer_state.get(peer_rank)
        if writer_state is None or self._writers_final() or self._logical_failed:
            return False
        if writer_state is WriterState.UNEXPOSED:
            self._writer_state[peer_rank] = WriterState.EXPOSED
            if self.state is TransferState.INIT:
                self.state = TransferState.ACTIVE
            return True
        return writer_state is WriterState.EXPOSED

    def mark_writer_no_access(self, peer_rank: int, *, succeeded: bool = True) -> bool:
        """Record positive evidence that ``peer_rank`` cannot access the receive slot."""
        writer_state = self._writer_state.get(peer_rank)
        if writer_state is None or self._writers_final():
            return False
        if writer_state in (WriterState.NO_ACCESS, WriterState.TERMINAL):
            return False
        self._writer_state[peer_rank] = WriterState.NO_ACCESS
        self._writer_ok[peer_rank] = succeeded
        if not succeeded:
            self.mark_logical_failure()
        return True

    def mark_logical_failure(self) -> None:
        """Fail logically and close every not-yet-exposed writer.

        Writers already exposed remain in doubt and keep the slot owned. Because this transition
        gates future calls to :meth:`mark_writer_exposed`, changing an unexposed writer to
        ``NO_ACCESS`` is safe and atomic when called under the transport lock.
        """
        if self.settled:
            return
        self._logical_failed = True
        self._scatter_descs.clear()
        self._scatter_destination_intervals = ()
        for rank, writer_state in self._writer_state.items():
            if writer_state is WriterState.UNEXPOSED:
                self._writer_state[rank] = WriterState.NO_ACCESS
                self._writer_ok[rank] = False
        if self.pending_exposed_writers and self.state is not TransferState.SCATTERING:
            self.state = TransferState.QUARANTINED

    def mark_protocol_conflict(self) -> None:
        """Retain the slot until backend-wide quiescence resolves ambiguity."""
        if self.settled:
            return
        self._protocol_conflict = True
        self.mark_logical_failure()

    def mark_backend_quiesced(self) -> None:
        """Retire every remote accessor after backend-wide quiescence is proven."""
        if self.settled:
            return
        self._backend_quiesced = True
        self.mark_logical_failure()
        for rank, writer_state in self._writer_state.items():
            if writer_state not in (WriterState.NO_ACCESS, WriterState.TERMINAL):
                self._writer_state[rank] = WriterState.NO_ACCESS
                self._writer_ok[rank] = False
        if self.state is TransferState.QUARANTINED:
            self.state = TransferState.ACTIVE

    # Mutations: call only while holding the transport's reservation lock.
    def record_writer_result(
        self,
        peer_rank: int,
        succeeded: bool,
        *,
        src_base: Optional[int] = None,
        dst_ptrs=None,
        sizes=None,
    ) -> bool:
        """Record one writer's terminal report. Repeat or late reports are ignored, so duplicate or
        out-of-order messages are harmless."""
        # A duplicate or late report can still arrive after the writer set is final, because
        # scatter runs on another thread while the region stays live. For example a retransmitted
        # notification, or a stray failure that would flip a good transfer to failed; drop it.
        writer_state = self._writer_state.get(peer_rank)
        if (
            writer_state is None
            or self._writers_final()
            or writer_state in (WriterState.NO_ACCESS, WriterState.TERMINAL)
        ):
            return False

        # A result itself proves that the writer crossed its publication boundary. This implicit
        # transition preserves compatibility while callers are migrated to mark exposure before
        # sending an address.
        if writer_state is WriterState.UNEXPOSED:
            self._writer_state[peer_rank] = WriterState.EXPOSED

        scatter_intervals = None
        tail_absent = dst_ptrs is None and sizes is None and src_base is None
        if succeeded and not tail_absent:
            valid_tail = (
                dst_ptrs is not None
                and sizes is not None
                and int(dst_ptrs.size) == int(sizes.size)
                and int(dst_ptrs.size) > 0
            )
            expected_src = self.writer_base(peer_rank)
            valid_tail = valid_tail and src_base is not None and int(src_base) == expected_src
            if valid_tail:
                dst_values = tuple(int(ptr) for ptr in dst_ptrs)
                size_values = tuple(int(size) for size in sizes)
                scatter_intervals = self._non_overlapping_destination_intervals(
                    dst_values, size_values
                )
                # ``src_base`` names this writer's fixed bounce sub-region and
                # scatter consumes the fragments contiguously from that base.
                # Requiring the exact contribution prevents a success tail
                # from silently truncating the authorized source mapping while
                # still allowing any equivalent fragment segmentation.
                valid_tail = (
                    scatter_intervals is not None and sum(size_values) == self.per_writer_bytes
                )
                valid_tail = valid_tail and self._scatter_destinations_authorized(
                    dst_values, size_values
                )
                valid_tail = valid_tail and self._scatter_destinations_disjoint(scatter_intervals)
            if not valid_tail:
                succeeded = False

        self._writer_state[peer_rank] = WriterState.TERMINAL
        self._writer_ok[peer_rank] = succeeded
        if (
            succeeded
            and not self._logical_failed
            and dst_ptrs is not None
            and int(dst_ptrs.size) > 0
        ):
            assert scatter_intervals is not None
            self._scatter_descs[peer_rank] = (int(src_base), dst_ptrs, sizes)
            self._scatter_destination_intervals = tuple(
                merge(self._scatter_destination_intervals, scatter_intervals)
            )
        elif not succeeded:
            self.mark_logical_failure()
        if self.state is TransferState.INIT:
            self.state = TransferState.ACTIVE
        return True

    def begin_scatter(self) -> None:
        self.state = TransferState.SCATTERING
        self.scatter_state = ScatterState.QUEUED

    def sorted_scatter_descs(self) -> List[tuple]:
        """Success fragments ordered by the exact writer publication plan."""
        return [
            self._scatter_descs[rank] for rank in self.writer_ranks if rank in self._scatter_descs
        ]

    def finish_scatter(self, ok: bool) -> None:
        """Record the local scatter outcome.

        A CUDA error is not positive evidence that queued memory accesses have
        stopped. ``FAILED`` therefore remains owned indefinitely rather than
        making the slot or destination KV observable as drained.
        """
        self.scatter_state = ScatterState.DONE if ok else ScatterState.FAILED
        if not ok:
            # Logical failure is known even though the slot and destination
            # remain physically owned until local CUDA quiescence is proven.
            self.mark_logical_failure()

    def suppress_scatter(self) -> None:
        """Fail work that was queued but provably never launched on CUDA."""
        self.mark_logical_failure()
        self.scatter_state = ScatterState.DONE

    def ready_to_scatter(self) -> bool:
        """All writers succeeded, there is data to scatter, and scatter has not started."""
        return (
            self.state is TransferState.ACTIVE
            and not self._logical_failed
            and self._all_writers_succeeded()
            and bool(self._scatter_descs)
        )

    def ready_to_settle(self) -> bool:
        if self.settled:
            return False
        if self._protocol_conflict and not self._backend_quiesced:
            return False
        if not self._all_writers_terminal():
            return False  # an exposed writer may still access the slot
        if self.scatter_state in (ScatterState.QUEUED, ScatterState.FAILED):
            return False  # queued or failed CUDA work has no positive local fence
        if not self._logical_failed and self._all_writers_succeeded() and self._scatter_descs:
            return self.scatter_state is ScatterState.DONE
        return True  # nothing to scatter, or a failure among them: either way drained

    def settle(self) -> Optional[Settlement]:
        """Finalize the transfer once and return the decision, or None if already done. Every
        termination path calls this; only the first call has any effect."""
        if self.settled or not self.ready_to_settle():
            return None
        self.settled = True
        success = (
            not self._logical_failed
            and self._all_writers_succeeded()
            and self.scatter_state is not ScatterState.FAILED
        )
        self.state = TransferState.COMPLETED if success else TransferState.FAILED
        return Settlement(self.slot_id, Disposition.RELEASE, success, self.on_done)


class BounceTransport(ABC):
    """Contract both transports implement so they cannot drift; the factory returns one, and the rest
    of the code depends only on this interface."""

    #: True for a real transport, False for the disabled null object.
    enabled: bool = False

    @abstractmethod
    def build_request(self, write_meta):
        """Gather the request's cache into a send region and build the coalesced write, or None to
        fall back to the per-fragment path."""

    @abstractmethod
    def release_send(self, slot_id) -> None:
        """Release a send region after its write completes."""

    @abstractmethod
    def quarantine_send(self, slot_id) -> None:
        """Permanently retain a send region whose NIXL access remains ambiguous."""

    @abstractmethod
    def reserve(
        self,
        recv_req,
        writer_ranks: Sequence[int] = (),
        *,
        timeout: Optional[float] = None,
        destination_intervals: Optional[Iterable[tuple[int, int]]] = None,
        destination_intervals_factory: Optional[Callable[[], Iterable[tuple[int, int]]]] = None,
    ) -> bool:
        """Reserve for one exact writer-rank sequence, or return False for direct fallback.

        ``destination_intervals_factory`` is evaluated only after allocator admission succeeds, so
        an expensive trusted destination manifest is not built for a transfer that falls back.
        """

    @abstractmethod
    def writer_base(self, rid_slice, peer_rank: int) -> Optional[int]:
        """Where the given exact fan-in writer rank writes in the region."""

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
    def mark_writer_exposed(self, rid_slice, peer_rank: int) -> bool:
        """Mark the writer possibly able to access the slot before publishing its address."""

    @abstractmethod
    def record_no_access(
        self, rid_slice, peer_rank: int, *, succeeded: bool = True, on_done=None
    ) -> None:
        """Record positive evidence that a planned writer cannot access the slot."""

    @abstractmethod
    def mark_logical_failure(self, rid_slice, on_done=None) -> None:
        """Suppress future publication and drain already-exposed writers."""

    @abstractmethod
    def mark_protocol_conflict(self, rid_slice, on_done=None) -> None:
        """Retain an ambiguous receive slot until backend-wide quiescence."""

    @abstractmethod
    def mark_backend_quiesced(self, rid_slice=None, on_done=None) -> None:
        """Record backend-wide evidence and report when the bounce resource settles."""

    @abstractmethod
    def retry_settlements(self, scope=None) -> bool:
        """Retry durable settlement acknowledgements in scope.

        ``scope`` is ``None`` for the whole transport, a request ID for every
        slice of that request, or an exact ``(request_id, slice_id)`` key.
        Return whether the selected scope has no pending acknowledgement.
        """

    @abstractmethod
    def record_result(
        self,
        rid_slice,
        peer_rank,
        dst_ptrs=None,
        sizes=None,
        src_base=None,
        on_done=None,
        on_logical_failure=None,
    ) -> None:
        """Handle a writer's success; scatter and finalize once all writers reported.

        ``on_logical_failure`` reports a launched local scatter failure without
        claiming that the bounce slot has physically settled.
        """

    @abstractmethod
    def record_failure(self, rid_slice, peer_rank, on_done=None) -> None:
        """Handle a writer's failure; free the region once all writers reported."""

    @abstractmethod
    def close(self) -> None:
        """Stop the scatter worker, deregister its memory, and free the arenas."""
