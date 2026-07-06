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
    whether the transfer succeeded, and the callback to fire."""

    slot_id: int
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

    # Whether each writer that reported succeeded, keyed by rank; presence means it reported.
    _writer_ok: Dict[int, bool] = field(default_factory=dict)
    # per successful writer: where it wrote, plus the fragments to scatter back
    _scatter_descs: List[tuple] = field(default_factory=list)
    _orphaned: bool = False
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
            TransferState.QUARANTINED,
        )

    def _all_writers_reported(self) -> bool:
        return len(self._writer_ok) >= self.num_writers

    def _all_writers_succeeded(self) -> bool:
        return self._all_writers_reported() and all(self._writer_ok.values())

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
            return True  # in doubt: settle now and quarantine
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
            self.state = TransferState.QUARANTINED
            return Settlement(self.slot_id, Disposition.QUARANTINE, False, self.on_done)
        success = self._all_writers_succeeded() and self.scatter_state is not ScatterState.FAILED
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
    def reserve(self, recv_req, num_writers: int = 1, *, timeout: Optional[float] = None) -> bool:
        """Reserve a region and record its address for the senders. False falls back to per-fragment."""

    @abstractmethod
    def writer_base(self, rid_slice, writer_index: int) -> Optional[int]:
        """Where the given fan-in writer writes in the region."""

    @abstractmethod
    def is_bounced(self, rid_slice) -> bool:
        """Whether this request and slice name a live bounced region."""

    @abstractmethod
    def release_idle_reservation(self, rid_slice) -> None:
        """Immediately free a reservation cancelled before any address went out."""

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
