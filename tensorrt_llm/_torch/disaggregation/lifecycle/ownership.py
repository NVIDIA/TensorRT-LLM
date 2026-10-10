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

"""Internal physical-access bookkeeping shared by transfer integrations.

This is not an allocator or a public cache-backend API. Callers supply already
mapped participants and backend-defined access-end evidence, and retain manager
loans until the deadline permits retirement. Logical Attempt outcomes remain
separate. Native publication ordering, transport submission and local-copy
synchronization are not implemented here.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Optional

from .retirement import RetirementDeadline

if TYPE_CHECKING:
    from ..base.agent import TransferStatus


class TransferNotSubmittedError(RuntimeError):
    """Admission rejected an operation before the backend could access memory."""


class PhysicalOperationState(Enum):
    """Sender-side evidence for one peer's access to a task's source memory.

    The legal forward paths are::

        ADMITTED -> NOT_SUBMITTED
        ADMITTED -> SUBMITTING -> SUBMITTED -> BACKEND_DONE
        SUBMITTING -> IN_DOUBT
        SUBMITTED -> IN_DOUBT -> BACKEND_DONE (same retained status reports DONE)

    Only NOT_SUBMITTED and BACKEND_DONE prove that the operation can no longer
    access the source. An IN_DOUBT operation without a retained backend status
    cannot retire. Repeating a terminal transition is idempotent, and repeating
    IN_DOUBT preserves the retained backend evidence.
    """

    ADMITTED = "ADMITTED"
    SUBMITTING = "SUBMITTING"
    SUBMITTED = "SUBMITTED"
    NOT_SUBMITTED = "NOT_SUBMITTED"
    BACKEND_DONE = "BACKEND_DONE"
    IN_DOUBT = "IN_DOUBT"


_DRAINED_PHYSICAL_OPERATION_STATES = frozenset(
    (PhysicalOperationState.NOT_SUBMITTED, PhysicalOperationState.BACKEND_DONE)
)


@dataclass
class _PhysicalOperation:
    """State plus strong backend roots retained until physical quiescence."""

    state: PhysicalOperationState
    request: Optional[object] = None
    status: Optional[TransferStatus] = None


class SendOperationOwner:
    """Retain one source operation per participant independently of logical results.

    The caller maps participant IDs and retains allocation/registration loans.
    This helper owns only the operation request and completion-status roots.
    Backend submission, evidence translation and resource release stay with
    the caller; an ambiguous error is never safe completion.
    """

    def __init__(self, retirement: Optional[RetirementDeadline] = None) -> None:
        """Create bookkeeping before admitting any physical operations.

        Args:
            retirement: Optional session deadline sharing the admission arbiter.
        """
        self._retirement = retirement
        self._physical_lock = threading.Lock() if retirement is None else retirement.lock
        self._physical_operations: dict[int, _PhysicalOperation] = {}

    def bind_retirement(self, retirement: Optional[RetirementDeadline]) -> None:
        """Bind the session arbiter before the owner is exposed to workers.

        Args:
            retirement: Deadline used by every operation in the session.
        """
        self._retirement = retirement
        if retirement is not None:
            self._physical_lock = retirement.lock

    def begin_physical_operation(self, peer_rank: int) -> bool:
        with self._physical_lock:
            if peer_rank in self._physical_operations:
                return False
            self._physical_operations[peer_rank] = _PhysicalOperation(
                PhysicalOperationState.ADMITTED
            )
            return True

    def _require_physical_operation_locked(
        self,
        peer_rank: int,
        expected_states: tuple[PhysicalOperationState, ...],
    ) -> _PhysicalOperation:
        operation = self._physical_operations.get(peer_rank)
        if operation is None:
            raise RuntimeError(f"physical operation {peer_rank} was not admitted")
        if operation.state not in expected_states:
            expected = ", ".join(state.value for state in expected_states)
            raise RuntimeError(
                f"physical operation {peer_rank} is {operation.state.value}, expected {expected}"
            )
        return operation

    def begin_backend_submission(
        self,
        peer_rank: int,
        request: object,
    ) -> None:
        with self._physical_lock:
            operation = self._require_physical_operation_locked(
                peer_rank, (PhysicalOperationState.ADMITTED,)
            )
            if self._retirement is not None and not self._retirement.expose(operation):
                operation.state = PhysicalOperationState.NOT_SUBMITTED
                raise TransferNotSubmittedError("source retirement admission is closed")
            operation.request = request
            operation.state = PhysicalOperationState.SUBMITTING

    def record_backend_submission(self, peer_rank: int, status: TransferStatus) -> None:
        with self._physical_lock:
            operation = self._require_physical_operation_locked(
                peer_rank, (PhysicalOperationState.SUBMITTING,)
            )
            operation.status = status
            operation.state = PhysicalOperationState.SUBMITTED

    def mark_physical_operation_in_doubt(self, peer_rank: int) -> None:
        with self._physical_lock:
            operation = self._require_physical_operation_locked(
                peer_rank,
                (
                    PhysicalOperationState.SUBMITTING,
                    PhysicalOperationState.SUBMITTED,
                    PhysicalOperationState.IN_DOUBT,
                ),
            )
            operation.state = PhysicalOperationState.IN_DOUBT
            if self._retirement is not None:
                self._retirement.request_drain("backend quiescence unproven")

    def retire_unsubmitted_physical_operation(self, peer_rank: int) -> None:
        with self._physical_lock:
            operation = self._require_physical_operation_locked(
                peer_rank,
                (
                    PhysicalOperationState.ADMITTED,
                    PhysicalOperationState.NOT_SUBMITTED,
                ),
            )
            operation.state = PhysicalOperationState.NOT_SUBMITTED

    def retire_backend_done_physical_operation(self, peer_rank: int) -> bool:
        with self._physical_lock:
            operation = self._require_physical_operation_locked(
                peer_rank,
                (
                    PhysicalOperationState.SUBMITTED,
                    PhysicalOperationState.BACKEND_DONE,
                ),
            )
            if operation.state is PhysicalOperationState.BACKEND_DONE:
                return True
            if self._retirement is not None and not self._retirement.settle(operation):
                operation.state = PhysicalOperationState.IN_DOUBT
                return False
            operation.request = None
            operation.status = None
            operation.state = PhysicalOperationState.BACKEND_DONE
            return True

    def poll_in_doubt_physical_operation(self, peer_rank: int) -> bool:
        """Retire once, only after a fresh DONE query on the retained status.

        Polling does not change the task's logical outcome. Keep strong local
        roots across the query, and reject a result if its operation changed.
        """
        with self._physical_lock:
            operation = self._physical_operations.get(peer_rank)
            if operation is None or operation.state is not PhysicalOperationState.IN_DOUBT:
                return False
            request, status = operation.request, operation.status
        if status is None:
            return False
        try:
            completed = status.is_completed()
        except Exception:
            # A backend query failure is not evidence that its accessors stopped.
            return False
        if completed is not True:
            return False
        with self._physical_lock:
            if (
                self._physical_operations.get(peer_rank) is not operation
                or operation.state is not PhysicalOperationState.IN_DOUBT
                or operation.request is not request
                or operation.status is not status
            ):
                return False
            if self._retirement is not None and not self._retirement.settle(operation):
                return False
            operation.request = None
            operation.status = None
            operation.state = PhysicalOperationState.BACKEND_DONE
            return True

    def has_started_physical_operation(self, peer_rank: int) -> bool:
        with self._physical_lock:
            return peer_rank in self._physical_operations

    @property
    def resources_drained(self) -> bool:
        with self._physical_lock:
            return (self._retirement is None or self._retirement.can_retire()) and all(
                operation.state in _DRAINED_PHYSICAL_OPERATION_STATES
                for operation in self._physical_operations.values()
            )


class ReceiveOperationOwner:
    """Track destination access independently from a task's logical result."""

    def __init__(self, retirement: Optional[RetirementDeadline] = None) -> None:
        self._retirement = retirement
        self._lock = threading.Lock() if retirement is None else retirement.lock
        self._publication_pending = False
        self._cancelled_unpublished = False
        self._expected_writers: Optional[int] = None
        self._writer_cohort: Optional[frozenset[int]] = None
        self._writer_candidates: Optional[frozenset[int]] = None
        self._published_writers: Optional[frozenset[int]] = None
        self._quiesced_sessions: set[int] = set()
        self._writer_results: dict[int, bool] = {}
        self._in_doubt_writers: set[int] = set()
        self._settled_writers: set[int] = set()
        self._publication_failed = False
        self._local_completion_pending = False
        self._invalid_evidence = False

    def _settle_if_drained_locked(self) -> None:
        """Remove this claim only after the complete cohort and local work settle."""
        if self._retirement is not None and self._resources_drained_locked():
            self._retirement.settle(self)

    def _invalidate_evidence_locked(self) -> None:
        """Quarantine conflicting proof, including proof disputed after settlement."""
        self._invalid_evidence = True
        if self._retirement is not None:
            self._retirement.retain_unproven(self, "invalid receive ownership evidence")

    def begin_publication(self) -> None:
        with self._lock:
            if self._publication_pending or self._expected_writers is not None:
                raise RuntimeError("destination publication was already started")
            self._publication_pending = True

    def seal_writer_cohort(
        self,
        expected_writers: int,
        writer_cohort: Optional[set[int]] = None,
        *,
        published_writers: Optional[set[int]] = None,
    ) -> None:
        if expected_writers < 0:
            raise ValueError(f"expected_writers must be non-negative, got {expected_writers}")
        cohort = None if writer_cohort is None else frozenset(writer_cohort)
        candidates = cohort if published_writers is None else frozenset(published_writers)
        if cohort is not None and len(cohort) != expected_writers:
            raise ValueError(
                f"writer cohort has {len(cohort)} member(s), expected {expected_writers}"
            )
        if candidates is not None and (
            len(candidates) < expected_writers or (cohort is not None and not cohort <= candidates)
        ):
            raise ValueError("published candidates must cover every eligible writer")
        with self._lock:
            if self._expected_writers is not None:
                if (
                    self._expected_writers != expected_writers
                    or self._writer_cohort != cohort
                    or self._writer_candidates != candidates
                ):
                    raise RuntimeError("writer cohort was already sealed differently")
                return
            self._expected_writers = expected_writers
            self._writer_cohort = cohort
            self._writer_candidates = candidates
            self._published_writers = candidates

    def finish_publication(self) -> None:
        """Record that every authorized REQUEST_DATA message was sent."""
        with self._lock:
            self._publication_pending = False
            self._settle_if_drained_locked()

    def abort_publication(self, published_writers: set[int]) -> None:
        """Close a failed fan-out around the writers whose sends succeeded."""
        with self._lock:
            published = frozenset(published_writers)
            if self._writer_cohort is not None and not published.issubset(self._writer_cohort):
                self._invalidate_evidence_locked()
                raise RuntimeError("publication recorded a writer outside the sealed cohort")
            if not (self._writer_results.keys() | self._in_doubt_writers) <= published:
                self._invalidate_evidence_locked()
                raise RuntimeError("terminal evidence came from an unpublished writer")
            self._expected_writers = len(published)
            self._writer_cohort = frozenset(published)
            self._published_writers = published
            self._publication_failed = True
            self._cancelled_unpublished = not published
            self._publication_pending = False
            self._settle_if_drained_locked()

    def cancel_unpublished(self) -> bool:
        """Close a publication that did not authorize a remote writer."""
        with self._lock:
            if self._invalid_evidence:
                return False
            if self._expected_writers is not None:
                return self._cancelled_unpublished
            self._cancelled_unpublished = True
            self._expected_writers = 0
            self._writer_cohort = frozenset()
            self._published_writers = frozenset()
            self._publication_pending = False
            self._settle_if_drained_locked()
            return True

    def record_session_quiesced(self, peer_rank: int) -> None:
        """Record no-future-access proof without inventing a per-piece result."""
        with self._lock:
            if self._writer_candidates is None or peer_rank not in self._writer_candidates:
                self._invalidate_evidence_locked()
                raise RuntimeError(f"session acknowledgment from unknown writer {peer_rank}")
            if self._published_writers is not None and peer_rank in self._published_writers:
                self._quiesced_sessions.add(peer_rank)
                self._settle_if_drained_locked()

    def record_writer_in_doubt(self, peer_rank: int) -> bool:
        """Retain ownership after a writer reports no safe terminal evidence."""
        with self._lock:
            if self._expected_writers is None:
                self._invalidate_evidence_locked()
                raise RuntimeError(
                    f"writer {peer_rank} reported ambiguous evidence before publication"
                )
            if self._writer_cohort is not None and peer_rank not in self._writer_cohort:
                self._invalidate_evidence_locked()
                raise RuntimeError(f"writer {peer_rank} is outside the sealed cohort")
            if peer_rank in self._in_doubt_writers or peer_rank in self._settled_writers:
                return False
            self._in_doubt_writers.add(peer_rank)
            if self._retirement is not None:
                self._retirement.retain_unproven(self, "receive ownership is in doubt")
            return True

    def record_writer_settlement(self, peer_rank: int) -> bool:
        """Accept physical DONE for a writer that previously reported IN_DOUBT.

        Ordinary FAILED is not this proof: it may describe a later, unsubmitted
        chunk while the earlier ambiguous write is still touching the destination.
        """
        with self._lock:
            if self._expected_writers is None or (
                self._writer_cohort is not None and peer_rank not in self._writer_cohort
            ):
                self._invalidate_evidence_locked()
                raise RuntimeError(f"writer {peer_rank} settled outside the published cohort")
            if peer_rank in self._settled_writers:
                return False
            if peer_rank not in self._in_doubt_writers:
                self._invalidate_evidence_locked()
                raise RuntimeError(f"writer {peer_rank} settled without prior ambiguous evidence")
            if self._writer_results.get(peer_rank) is True:
                self._invalidate_evidence_locked()
                raise RuntimeError(f"writer {peer_rank} settled after contradictory success")
            self._writer_results[peer_rank] = False
            self._in_doubt_writers.remove(peer_rank)
            self._settled_writers.add(peer_rank)
            self._settle_if_drained_locked()
            return True

    def record_writer_result(
        self,
        peer_rank: int,
        succeeded: bool,
        *,
        wait_for_local_completion: bool,
    ) -> tuple[bool, bool]:
        """Record one writer and return ``(accepted, all_succeeded)``."""
        with self._lock:
            if self._expected_writers is None:
                self._invalidate_evidence_locked()
                raise RuntimeError(
                    f"writer {peer_rank} reported terminal evidence before publication"
                )
            if self._writer_cohort is not None and peer_rank not in self._writer_cohort:
                self._invalidate_evidence_locked()
                raise RuntimeError(f"writer {peer_rank} is outside the sealed cohort")
            if peer_rank in self._in_doubt_writers:
                if succeeded:
                    self._invalidate_evidence_locked()
                    raise RuntimeError(f"writer {peer_rank} reported success while in doubt")
                return False, False
            previous = self._writer_results.get(peer_rank)
            if previous is not None:
                if previous != succeeded:
                    self._invalidate_evidence_locked()
                    raise RuntimeError(
                        f"writer {peer_rank} reported contradictory terminal evidence"
                    )
                return False, False
            if len(self._writer_results) >= self._expected_writers:
                return False, False
            self._writer_results[peer_rank] = succeeded
            all_reported = len(self._writer_results) == self._expected_writers
            all_succeeded = (
                all_reported and not self._publication_failed and all(self._writer_results.values())
            )
            if all_succeeded and wait_for_local_completion:
                self._local_completion_pending = True
            self._settle_if_drained_locked()
            return True, all_succeeded

    def finish_local_completion(self) -> None:
        with self._lock:
            self._local_completion_pending = False
            self._settle_if_drained_locked()

    @property
    def all_writers_reported(self) -> bool:
        """Whether every writer of the sealed cohort has reported a terminal result."""
        with self._lock:
            return (
                self._expected_writers is not None
                and len(self._writer_results) == self._expected_writers
                and not self._in_doubt_writers
            )

    @property
    def resources_drained(self) -> bool:
        with self._lock:
            return self._resources_drained_locked() and (
                self._retirement is None or self._retirement.can_retire()
            )

    def _resources_drained_locked(self) -> bool:
        """Check physical evidence without consulting the session's other owners."""
        return (
            self._expected_writers is not None
            and (
                len(self._writer_results) == self._expected_writers
                or (
                    self._published_writers is not None
                    and self._published_writers <= self._quiesced_sessions
                )
            )
            and not self._publication_pending
            and not self._local_completion_pending
            and not self._in_doubt_writers
            and not self._invalid_evidence
        )
