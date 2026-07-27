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

"""Cross-side obligations and local submission fences for disaggregated KV.

These state machines deliberately do not own allocator memory. Expiry and
logical cancellation only close responsibility or future submission; an
allocator lease can be settled separately only after a reusable physical
disposition is established.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import Enum
from uuid import UUID

from tensorrt_llm._torch.disaggregation.lifecycle import PhysicalDisposition
from tensorrt_llm._torch.disaggregation.protocol import (
    AttemptIdentity,
    EndpointIdentity,
    OperationIdentity,
    TransferProtocolIdentity,
)


class ObligationConflictError(RuntimeError):
    """Raised when a duplicate identity asserts a contradictory fact."""


class GenerationGrantState(str, Enum):
    ACTIVE = "ACTIVE"
    REVOKED = "REVOKED"
    RELEASED = "RELEASED"


class ArtifactObligationState(str, Enum):
    ACTIVE = "ACTIVE"
    ABANDONED = "ABANDONED"
    RELEASED = "RELEASED"


class SubmissionFenceState(str, Enum):
    OPEN = "OPEN"
    FENCING = "FENCING"
    FENCED = "FENCED"
    QUIESCED = "QUIESCED"
    IN_DOUBT = "IN_DOUBT"


class ReceiveCommitState(str, Enum):
    OPEN = "OPEN"
    COMMITTED = "COMMITTED"
    ABORTED = "ABORTED"


@dataclass(frozen=True, slots=True)
class GenerationGrantIdentity:
    """Identity of one GEN-owned admission and capacity obligation."""

    consumer_grant_id: UUID
    attempt: AttemptIdentity
    generation_endpoint: EndpointIdentity

    def __post_init__(self) -> None:
        if self.consumer_grant_id.int == 0:
            raise ValueError("consumer_grant_id must be a non-nil UUID")


class GenerationIntentGrant:
    """GEN-owned queue/capacity responsibility with a monotonic TTL."""

    def __init__(
        self,
        identity: GenerationGrantIdentity,
        *,
        issued_at_s: float,
        expires_at_s: float,
    ) -> None:
        if expires_at_s <= issued_at_s:
            raise ValueError("GEN grant expiry must be after issuance")
        self.identity = identity
        self.issued_at_s = issued_at_s
        self.expires_at_s = expires_at_s
        self._state = GenerationGrantState.ACTIVE
        self._reason = ""
        self._lock = threading.Lock()

    @property
    def state(self) -> GenerationGrantState:
        with self._lock:
            return self._state

    @property
    def reason(self) -> str:
        with self._lock:
            return self._reason

    def check_expiry(self, now_s: float) -> GenerationGrantState:
        """Revoke an unconsumed/live obligation after its GEN-owned deadline."""
        with self._lock:
            if self._state is GenerationGrantState.ACTIVE and now_s >= self.expires_at_s:
                self._state = GenerationGrantState.REVOKED
                self._reason = "generation intent grant expired"
            return self._state

    def renew(self, *, now_s: float, expires_at_s: float) -> GenerationGrantState:
        """Extend a live grant; stale or terminal renewals are inert."""
        with self._lock:
            if self._state is not GenerationGrantState.ACTIVE:
                return self._state
            if now_s >= self.expires_at_s:
                self._state = GenerationGrantState.REVOKED
                self._reason = "generation intent grant expired before renewal"
                return self._state
            if expires_at_s <= self.expires_at_s:
                return self._state
            self.expires_at_s = expires_at_s
            return self._state

    def revoke(self, reason: str) -> GenerationGrantState:
        with self._lock:
            if self._state is GenerationGrantState.RELEASED:
                return self._state
            if self._state is GenerationGrantState.REVOKED:
                if self._reason != reason:
                    raise ObligationConflictError(
                        "generation grant was revoked with a conflicting reason"
                    )
                return self._state
            self._state = GenerationGrantState.REVOKED
            self._reason = reason
            return self._state

    def release(self) -> GenerationGrantState:
        with self._lock:
            if self._state is GenerationGrantState.REVOKED:
                return self._state
            self._state = GenerationGrantState.RELEASED
            return self._state


@dataclass(frozen=True, slots=True)
class ArtifactObligationIdentity:
    """One GEN consumer's renewable obligation for an immutable artifact."""

    grant: GenerationGrantIdentity

    @property
    def attempt(self) -> AttemptIdentity:
        return self.grant.attempt


class ArtifactObligationLease:
    """Renewable control-plane obligation; never an allocator reuse proof."""

    def __init__(
        self,
        identity: ArtifactObligationIdentity,
        *,
        expires_at_s: float,
    ) -> None:
        self.identity = identity
        self.expires_at_s = expires_at_s
        self._state = ArtifactObligationState.ACTIVE
        self._last_renewal_sequence = -1
        self._last_renewal_expiry_s: float | None = None
        self._lock = threading.Lock()

    @property
    def state(self) -> ArtifactObligationState:
        with self._lock:
            return self._state

    def renew(
        self,
        *,
        sequence: int,
        now_s: float,
        expires_at_s: float,
    ) -> ArtifactObligationState:
        """Apply a generation-safe, monotonically sequenced renewal."""
        if sequence < 0:
            raise ValueError("renewal sequence must be non-negative")
        if expires_at_s <= now_s:
            raise ValueError("renewal expiry must be in the future")
        with self._lock:
            if self._state is not ArtifactObligationState.ACTIVE:
                return self._state
            if now_s >= self.expires_at_s:
                self._state = ArtifactObligationState.ABANDONED
                return self._state
            if sequence < self._last_renewal_sequence:
                return self._state
            if sequence == self._last_renewal_sequence:
                if expires_at_s != self._last_renewal_expiry_s:
                    raise ObligationConflictError(
                        "artifact renewal sequence asserted a conflicting expiry"
                    )
                return self._state
            self._last_renewal_sequence = sequence
            self._last_renewal_expiry_s = expires_at_s
            self.expires_at_s = max(self.expires_at_s, expires_at_s)
            return self._state

    def check_expiry(self, now_s: float) -> ArtifactObligationState:
        with self._lock:
            if self._state is ArtifactObligationState.ACTIVE and now_s >= self.expires_at_s:
                self._state = ArtifactObligationState.ABANDONED
            return self._state

    def release(self) -> ArtifactObligationState:
        with self._lock:
            if self._state is ArtifactObligationState.ABANDONED:
                return self._state
            self._state = ArtifactObligationState.RELEASED
            return self._state

    def abandon(self) -> ArtifactObligationState:
        """Record explicit abandonment without pretending a timer proved it."""
        with self._lock:
            if self._state is ArtifactObligationState.RELEASED:
                return self._state
            self._state = ArtifactObligationState.ABANDONED
            return self._state


class SubmissionFence:
    """Local publication/submission frontier for one transfer session."""

    def __init__(self, identity: TransferProtocolIdentity) -> None:
        self.identity = identity
        self._state = SubmissionFenceState.OPEN
        self._authorized: set[OperationIdentity] = set()
        self._active: set[OperationIdentity] = set()
        self._completed: dict[OperationIdentity, PhysicalDisposition] = {}
        self._lock = threading.Lock()

    @property
    def state(self) -> SubmissionFenceState:
        with self._lock:
            return self._state

    def authorize(self, operation: OperationIdentity) -> None:
        with self._lock:
            self._require_operation_session(operation)
            if self._state is not SubmissionFenceState.OPEN:
                raise RuntimeError("submission fence is closed to new authorization")
            self._authorized.add(operation)

    def begin(self, operation: OperationIdentity) -> bool:
        """Enter the active accessor set, or replay an already-started begin."""
        with self._lock:
            self._require_operation_session(operation)
            if operation in self._completed:
                return False
            if operation in self._active:
                return True
            if self._state is not SubmissionFenceState.OPEN:
                raise RuntimeError("submission fence is closed to new operations")
            if operation not in self._authorized:
                raise RuntimeError("operation was not authorized for publication")
            self._active.add(operation)
            return True

    def complete(
        self,
        operation: OperationIdentity,
        disposition: PhysicalDisposition,
    ) -> SubmissionFenceState:
        """Record exact operation quiescence and advance a pending fence."""
        if not disposition.is_reusable:
            raise ValueError("operation completion requires a reusable disposition")
        with self._lock:
            self._require_operation_session(operation)
            existing = self._completed.get(operation)
            if existing is not None:
                if existing is not disposition:
                    raise ObligationConflictError(
                        "operation completed with a conflicting physical disposition"
                    )
                return self._state
            if operation not in self._active:
                raise RuntimeError("cannot complete an operation that is not active")
            self._active.remove(operation)
            self._completed[operation] = disposition
            if self._state is SubmissionFenceState.FENCING and not self._active:
                self._state = SubmissionFenceState.FENCED
            return self._state

    def fence(self) -> SubmissionFenceState:
        """Close future submission; existing operations must still drain."""
        with self._lock:
            if self._state is SubmissionFenceState.OPEN:
                self._state = (
                    SubmissionFenceState.FENCED
                    if not self._active
                    else SubmissionFenceState.FENCING
                )
            return self._state

    def mark_quiesced(self) -> SubmissionFenceState:
        with self._lock:
            if self._state is SubmissionFenceState.QUIESCED:
                return self._state
            if self._state is not SubmissionFenceState.FENCED or self._active:
                raise RuntimeError("submission must be fenced and drained before quiescence")
            self._state = SubmissionFenceState.QUIESCED
            return self._state

    def mark_in_doubt(self) -> SubmissionFenceState:
        with self._lock:
            if self._state is not SubmissionFenceState.QUIESCED:
                self._state = SubmissionFenceState.IN_DOUBT
            return self._state

    def _require_operation_session(self, operation: OperationIdentity) -> None:
        if operation.publication.session != self.identity:
            raise ValueError("operation belongs to a different transfer session")


class ReceiveCommitGate:
    """Serialize HANDOFF_COMMITTED against ABORT_REQUESTED."""

    def __init__(self, identity: TransferProtocolIdentity) -> None:
        self.identity = identity
        self._state = ReceiveCommitState.OPEN
        self._reason = ""
        self._lock = threading.Lock()

    @property
    def state(self) -> ReceiveCommitState:
        with self._lock:
            return self._state

    def commit(self) -> ReceiveCommitState:
        with self._lock:
            if self._state is ReceiveCommitState.ABORTED:
                return self._state
            self._state = ReceiveCommitState.COMMITTED
            return self._state

    def abort(self, reason: str) -> ReceiveCommitState:
        with self._lock:
            if self._state is ReceiveCommitState.COMMITTED:
                return self._state
            if self._state is ReceiveCommitState.ABORTED and self._reason != reason:
                raise ObligationConflictError(
                    "receive commit gate was aborted with a conflicting reason"
                )
            self._state = ReceiveCommitState.ABORTED
            self._reason = reason
            return self._state


__all__ = [
    "ArtifactObligationIdentity",
    "ArtifactObligationLease",
    "ArtifactObligationState",
    "GenerationGrantIdentity",
    "GenerationGrantState",
    "GenerationIntentGrant",
    "ObligationConflictError",
    "ReceiveCommitGate",
    "ReceiveCommitState",
    "SubmissionFence",
    "SubmissionFenceState",
]
