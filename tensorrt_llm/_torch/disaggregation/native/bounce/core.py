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
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple


class TransferState(Enum):
    """Lifecycle of one bounced receive region."""

    INIT = "init"  # leased, not yet advertised to any sender
    ACTIVE = "active"  # writers in flight
    SCATTERING = "scattering"  # all writers succeeded; scattering back into the cache
    COMPLETED = "completed"  # scattered cleanly, slot released
    FAILED = "failed"  # all writers done, at least one failed, slot released
    CANCELLED_DRAINED = "cancelled_drained"  # cancellation settled only after exact drain proof


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
    whether the transfer succeeded, and the callback to fire."""

    slot_id: int
    disposition: Disposition
    success: bool
    on_done: Optional[Callable[[bool], None]]


@dataclass
class TransferContext:
    """Lifetime state machine for one bounced receive region. The region is released only after every
    writer reports (success or failure both mean its write has drained), or an exact sender ACK proves
    that every advertised write for the request incarnation has drained. Cancellation alone never
    makes the region reusable."""

    rid_slice: Tuple[int, int]
    slot_id: int
    base_addr: int
    per_writer_bytes: int
    num_writers: int
    allowed_destination_ranges: Tuple[Tuple[int, int], ...] = ()
    expected_destination_plans: Dict[int, Tuple[Tuple[int, int], ...]] = field(default_factory=dict)
    on_done: Optional[Callable[[bool], None]] = None
    on_settled: Optional[Callable[[bool], None]] = None

    # Whether each writer that reported succeeded, keyed by rank; presence means it reported.
    _writer_ok: Dict[int, bool] = field(default_factory=dict)
    _expected_writer_bases: Dict[int, int] = field(default_factory=dict)
    # per successful writer: where it wrote, plus the fragments to scatter back
    _scatter_descs: List[tuple] = field(default_factory=list)
    _orphaned: bool = False
    _drain_proven: bool = False
    scatter_state: ScatterState = ScatterState.IDLE
    state: TransferState = TransferState.INIT
    settled: bool = False

    def writer_base(self, writer_index: int) -> int:
        """Where this fan-in writer writes in the region."""
        return self.base_addr + writer_index * self.per_writer_bytes

    def _writers_final(self) -> bool:
        return self.settled or self.state in (
            TransferState.SCATTERING,
            TransferState.COMPLETED,
            TransferState.FAILED,
            TransferState.CANCELLED_DRAINED,
        )

    def _all_writers_reported(self) -> bool:
        return len(self._writer_ok) >= self.num_writers

    def _all_writers_succeeded(self) -> bool:
        return self._all_writers_reported() and all(self._writer_ok.values())

    # Mutations: call only while holding the transport's reservation lock.
    def bind_writer(self, peer_rank: int, src_base: int) -> None:
        """Bind one immutable writer identity to its advertised bounce sub-region."""
        if self._writers_final() or peer_rank in self._expected_writer_bases:
            raise RuntimeError(f"bounce writer {peer_rank} was bound more than once")
        if len(self._expected_writer_bases) >= self.num_writers:
            raise RuntimeError("bounce writer binding exceeds the reserved fan-in")
        if self.expected_destination_plans and peer_rank not in self.expected_destination_plans:
            raise RuntimeError(f"bounce writer {peer_rank} has no receiver-derived scatter plan")
        self._expected_writer_bases[peer_rank] = src_base

    def set_completion_callback(self, on_settled: Callable[[bool], None]) -> None:
        """Install the immutable reservation-lifetime callback before advertisement."""
        if self._writers_final() or self.on_settled is not None:
            raise RuntimeError("bounce completion callback was installed more than once")
        self.on_settled = on_settled

    def _combined_callback(self) -> Optional[Callable[[bool], None]]:
        if self.on_done is None:
            return self.on_settled
        if self.on_settled is None:
            return self.on_done
        on_done = self.on_done
        on_settled = self.on_settled

        def complete(success: bool) -> None:
            try:
                on_done(success)
            finally:
                # Session lifetime credit must retire even if optional task/perf
                # completion handling raises.
                on_settled(success)

        return complete

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
        expected_src_base = self._expected_writer_bases.get(peer_rank)
        if expected_src_base is None:
            raise RuntimeError(f"result from unexpected bounce writer rank {peer_rank}")
        if succeeded:
            if src_base is None or dst_ptrs is None or sizes is None:
                raise RuntimeError(
                    "successful bounce writer result requires a complete scatter tail"
                )
            if src_base != expected_src_base:
                raise RuntimeError(
                    f"bounce writer {peer_rank} returned source base {src_base}, "
                    f"expected {expected_src_base}"
                )
        elif src_base is not None and src_base != expected_src_base:
            raise RuntimeError(
                f"bounce writer {peer_rank} returned source base {src_base}, "
                f"expected {expected_src_base}"
            )
        # A duplicate or late report can still arrive after the writer set is final, because
        # scatter runs on another thread while the region stays live. For example a retransmitted
        # notification, or a stray failure that would flip a good transfer to failed; drop it.
        if self._writers_final() or peer_rank in self._writer_ok:
            return
        self._writer_ok[peer_rank] = succeeded
        if succeeded and dst_ptrs is not None and int(dst_ptrs.size) > 0:
            self._scatter_descs.append(
                (src_base if src_base is not None else self.base_addr, dst_ptrs, sizes)
            )
        if self.state is TransferState.INIT:
            self.state = TransferState.ACTIVE

    def mark_orphaned(self) -> None:
        """Give up on writers that never reported (cancel, timeout, shutdown): a one-sided write
        cannot be aborted, so retain the region pending explicit drain proof. No-op once scattering
        or done."""
        if self._writers_final():
            return
        self._orphaned = True

    def confirm_drained(self) -> None:
        """Record external proof that no writer can touch this region again."""
        self._drain_proven = True

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
            and self._all_writers_succeeded()
            and bool(self._scatter_descs)
        )

    def ready_to_settle(self) -> bool:
        if self.settled:
            return False
        if self._orphaned:
            # Cancellation alone says nothing about a one-sided remote write.
            # Settle only after every writer reports terminal or the sender's
            # drain ACK provides equivalent proof.
            return self._drain_proven or self._all_writers_reported()
        if not self._all_writers_reported():
            return False  # a writer has not reported yet
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
            self.state = TransferState.CANCELLED_DRAINED
            # Despite the historical state name, reaching here now requires
            # exact drain proof, so the region is safe to release immediately.
            return Settlement(
                self.slot_id,
                Disposition.RELEASE,
                False,
                self._combined_callback(),
            )
        success = self._all_writers_succeeded() and self.scatter_state is not ScatterState.FAILED
        self.state = TransferState.COMPLETED if success else TransferState.FAILED
        return Settlement(
            self.slot_id,
            Disposition.RELEASE,
            success,
            self._combined_callback(),
        )


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
    def reserve(
        self,
        recv_req,
        num_writers: int = 1,
        *,
        timeout: Optional[float] = None,
        expected_destination_plans=None,
    ) -> bool:
        """Reserve a region and record its address for the senders. False falls back to per-fragment."""

    @abstractmethod
    def writer_base(self, rid_slice, writer_index: int) -> Optional[int]:
        """Where the given fan-in writer writes in the region."""

    @abstractmethod
    def bind_writer(self, rid_slice, peer_rank: int, writer_index: int) -> Optional[int]:
        """Bind a rank to its immutable advertised sub-region and return that base."""

    @abstractmethod
    def set_completion_callback(self, rid_slice, on_settled: Callable[[bool], None]) -> None:
        """Install the callback that retires the reservation's lifetime credit."""

    @abstractmethod
    def is_bounced(self, rid_slice) -> bool:
        """Whether this request and slice name a live bounced region."""

    @abstractmethod
    def release_idle_reservation(self, rid_slice) -> None:
        """Immediately free a reservation cancelled before any address went out."""

    @abstractmethod
    def orphan_reservation(self, rid_slice) -> None:
        """Retain an in-flight reservation until explicit drain proof."""

    @abstractmethod
    def confirm_drained(self, rid_slice) -> None:
        """Release an orphan only after all possible remote writes drained."""

    @abstractmethod
    def record_result(
        self, rid_slice, peer_rank, dst_ptrs=None, sizes=None, src_base=None, on_done=None
    ) -> None:
        """Handle a writer's success; scatter and finalize once all writers reported."""

    @abstractmethod
    def record_failure(self, rid_slice, peer_rank, on_done=None) -> None:
        """Handle a writer's failure; free the region once all writers reported."""

    @abstractmethod
    def close(self) -> None:
        """Stop the scatter worker, deregister memory, free the arenas."""
