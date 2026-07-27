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
"""Endpoint-owned control-plane registries for disaggregated handoff.

The registries below own cross-side responsibility only. They deliberately do
not hold or settle allocator leases. A grant or artifact obligation expiring
can initiate abort/fencing, but cannot make an allocation reusable.
"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass
from typing import Callable
from uuid import UUID

from tensorrt_llm._torch.disaggregation.obligations import (
    ArtifactObligationIdentity,
    ArtifactObligationLease,
    ArtifactObligationState,
    GenerationGrantIdentity,
    GenerationGrantState,
    GenerationIntentGrant,
    ObligationConflictError,
)


class TerminalIdentityFilter:
    """Bounded rotating memory of terminal UUIDs over a replay horizon.

    Exact tombstones retain conflict diagnostics. This filter preserves the
    fail-closed replay decision when those bounded tombstones are evicted,
    without accumulating false positives for the full endpoint lifetime.
    Two horizon-sized generations guarantee at least one full replay horizon
    of retention and bound the maximum retention to two horizons.
    """

    _BITS_PER_IDENTITY = 48
    _HASH_COUNT = 8

    def __init__(
        self,
        capacity: int,
        *,
        replay_horizon_s: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if capacity <= 0:
            raise ValueError("terminal identity filter capacity must be positive")
        if replay_horizon_s <= 0:
            raise ValueError("terminal identity replay horizon must be positive")
        self._bit_count = max(1024, capacity * self._BITS_PER_IDENTITY)
        self._current_bits = self._empty_bits()
        self._previous_bits = self._empty_bits()
        self._replay_horizon_s = replay_horizon_s
        self._clock = clock
        self._current_started_at_s: float | None = None

    def add(self, identity: UUID, *, now_s: float | None = None) -> None:
        self.advance(self._clock() if now_s is None else now_s)
        for position in self._positions(identity):
            self._current_bits[position // 8] |= 1 << (position % 8)

    def contains(self, identity: UUID, *, now_s: float | None = None) -> bool:
        if now_s is not None:
            self.advance(now_s)
        return all(
            (self._current_bits[position // 8] | self._previous_bits[position // 8])
            & (1 << (position % 8))
            for position in self._positions(identity)
        )

    def advance(self, now_s: float) -> None:
        """Rotate elapsed replay windows without depending on request volume."""
        if self._current_started_at_s is None:
            self._current_started_at_s = now_s
            return
        if now_s < self._current_started_at_s + self._replay_horizon_s:
            return
        rotations = int((now_s - self._current_started_at_s) // self._replay_horizon_s)
        if rotations == 1:
            self._previous_bits = self._current_bits
            self._current_bits = self._empty_bits()
        else:
            self._previous_bits = self._empty_bits()
            self._current_bits = self._empty_bits()
        self._current_started_at_s += rotations * self._replay_horizon_s

    def _empty_bits(self) -> bytearray:
        return bytearray((self._bit_count + 7) // 8)

    def _positions(self, identity: UUID) -> tuple[int, ...]:
        digest = hashlib.blake2b(
            identity.bytes,
            digest_size=64,
            person=b"trtllm-terminal",
        ).digest()
        return tuple(
            int.from_bytes(digest[offset : offset + 8], "little") % self._bit_count
            for offset in range(0, self._HASH_COUNT * 8, 8)
        )


@dataclass(frozen=True, slots=True)
class GenerationGrantDecision:
    """Result of a GEN-owned admission decision."""

    accepted: bool
    identity: GenerationGrantIdentity
    expires_at_s: float | None
    reason: str = ""


@dataclass(frozen=True, slots=True)
class GenerationGrantRenewalDecision:
    """Endpoint-owned result for one sequenced grant renewal."""

    state: GenerationGrantState
    expires_at_s: float | None


@dataclass(slots=True)
class _GenerationGrantRecord:
    grant: GenerationIntentGrant
    issued_expires_at_s: float
    scheduler_inserted_at_s: float | None = None
    last_renewal_sequence: int = -1
    last_renewal_expires_at_s: float | None = None


@dataclass(frozen=True, slots=True)
class _GenerationGrantTombstone:
    identity: GenerationGrantIdentity
    state: GenerationGrantState
    reason: str
    retain_until_s: float


class GenerationAdmissionRegistry:
    """GEN authority for admission credit and queue ownership.

    A successful ``issue`` means GEN, rather than the router, owns the queue
    obligation. ``mark_scheduler_inserted`` is the handoff point at which
    artifact-renewal responsibility begins.
    """

    def __init__(
        self,
        *,
        max_live_grants: int,
        max_tombstones: int = 4096,
        replay_filter_capacity: int = 262144,
        replay_horizon_s: float = 1200.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_live_grants <= 0:
            raise ValueError("max_live_grants must be positive")
        if max_tombstones <= 0:
            raise ValueError("max_tombstones must be positive")
        if replay_filter_capacity <= 0:
            raise ValueError("replay_filter_capacity must be positive")
        if replay_horizon_s <= 0:
            raise ValueError("replay_horizon_s must be positive")
        self._max_live_grants = max_live_grants
        self._max_tombstones = max_tombstones
        self._replay_horizon_s = replay_horizon_s
        self._clock = clock
        self._records: dict[UUID, _GenerationGrantRecord] = {}
        self._tombstones: dict[UUID, _GenerationGrantTombstone] = {}
        self._terminal_ids = TerminalIdentityFilter(
            replay_filter_capacity,
            replay_horizon_s=replay_horizon_s,
            clock=clock,
        )
        self._lock = threading.Lock()

    @property
    def live_grant_count(self) -> int:
        with self._lock:
            return len(self._records)

    def issue(
        self,
        identity: GenerationGrantIdentity,
        *,
        issued_at_s: float,
        expires_at_s: float,
    ) -> GenerationGrantDecision:
        with self._lock:
            self._purge_tombstones_locked(issued_at_s)
            record = self._records.get(identity.consumer_grant_id)
            if record is not None:
                if record.grant.identity != identity:
                    raise ObligationConflictError(
                        "consumer_grant_id was replayed with conflicting admission facts"
                    )
                return GenerationGrantDecision(
                    accepted=True,
                    identity=identity,
                    expires_at_s=record.grant.expires_at_s,
                )

            tombstone = self._tombstones.get(identity.consumer_grant_id)
            if tombstone is not None:
                if tombstone.identity != identity:
                    raise ObligationConflictError(
                        "consumer_grant_id identifies a different terminal grant"
                    )
                return GenerationGrantDecision(
                    accepted=False,
                    identity=identity,
                    expires_at_s=None,
                    reason=tombstone.reason or "generation grant is already terminal",
                )
            if self._terminal_ids.contains(identity.consumer_grant_id):
                return GenerationGrantDecision(
                    accepted=False,
                    identity=identity,
                    expires_at_s=None,
                    reason=(
                        "generation grant identity is terminal or replay protection is saturated"
                    ),
                )

            if len(self._records) >= self._max_live_grants:
                return GenerationGrantDecision(
                    accepted=False,
                    identity=identity,
                    expires_at_s=None,
                    reason="generation admission credit is exhausted",
                )
            grant = GenerationIntentGrant(
                identity,
                issued_at_s=issued_at_s,
                expires_at_s=expires_at_s,
            )
            self._records[identity.consumer_grant_id] = _GenerationGrantRecord(
                grant=grant,
                issued_expires_at_s=expires_at_s,
            )
            return GenerationGrantDecision(
                accepted=True,
                identity=identity,
                expires_at_s=expires_at_s,
            )

    def mark_scheduler_inserted(
        self,
        identity: GenerationGrantIdentity,
        *,
        now_s: float,
    ) -> bool:
        """Consume the exact grant into the GEN-owned scheduler queue."""
        with self._lock:
            record = self._require_live_record(identity)
            if record.grant.check_expiry(now_s) is not GenerationGrantState.ACTIVE:
                self._retire_locked(
                    record,
                    record.grant.state,
                    record.grant.reason,
                    now_s=now_s,
                )
                return False
            if record.scheduler_inserted_at_s is None:
                record.scheduler_inserted_at_s = now_s
            return True

    def validate_active(
        self,
        identity: GenerationGrantIdentity,
        *,
        now_s: float,
    ) -> bool:
        """Validate an exact grant immediately before scheduler submission.

        This check does not transfer queue ownership. The caller must follow it
        with ``mark_scheduler_inserted`` only after the scheduler accepts the
        request.
        """
        with self._lock:
            record = self._require_live_record(identity)
            if record.grant.check_expiry(now_s) is GenerationGrantState.ACTIVE:
                return True
            self._retire_locked(
                record,
                record.grant.state,
                record.grant.reason,
                now_s=now_s,
            )
            return False

    def renew(
        self,
        identity: GenerationGrantIdentity,
        *,
        sequence: int,
        now_s: float,
        ttl_s: float,
    ) -> GenerationGrantRenewalDecision:
        if sequence < 0:
            raise ValueError("generation grant renewal sequence must be non-negative")
        if ttl_s <= 0:
            raise ValueError("generation grant renewal TTL must be positive")
        with self._lock:
            record = self._require_live_record(identity)
            if record.grant.check_expiry(now_s) is not GenerationGrantState.ACTIVE:
                state = record.grant.state
                self._retire_locked(
                    record,
                    state,
                    record.grant.reason,
                    now_s=now_s,
                )
                return GenerationGrantRenewalDecision(state, None)
            if sequence <= record.last_renewal_sequence:
                return GenerationGrantRenewalDecision(
                    GenerationGrantState.ACTIVE,
                    record.grant.expires_at_s,
                )
            expires_at_s = now_s + ttl_s
            state = record.grant.renew(now_s=now_s, expires_at_s=expires_at_s)
            if state is GenerationGrantState.ACTIVE:
                record.last_renewal_sequence = sequence
                record.last_renewal_expires_at_s = record.grant.expires_at_s
            else:
                self._retire_locked(
                    record,
                    state,
                    record.grant.reason,
                    now_s=now_s,
                )
            return GenerationGrantRenewalDecision(
                state,
                record.grant.expires_at_s if state is GenerationGrantState.ACTIVE else None,
            )

    def release(
        self,
        identity: GenerationGrantIdentity,
        *,
        now_s: float | None = None,
    ) -> GenerationGrantState:
        retired_at_s = self._clock() if now_s is None else now_s
        with self._lock:
            self._purge_tombstones_locked(retired_at_s)
            record = self._records.get(identity.consumer_grant_id)
            if record is None:
                tombstone = self._require_tombstone(identity)
                return tombstone.state
            self._require_identity(record.grant.identity, identity)
            state = record.grant.release()
            self._retire_locked(
                record,
                state,
                record.grant.reason,
                now_s=retired_at_s,
            )
            return state

    def revoke(
        self,
        identity: GenerationGrantIdentity,
        reason: str,
        *,
        now_s: float | None = None,
    ) -> GenerationGrantState:
        retired_at_s = self._clock() if now_s is None else now_s
        with self._lock:
            self._purge_tombstones_locked(retired_at_s)
            record = self._records.get(identity.consumer_grant_id)
            if record is None:
                tombstone = self._require_tombstone(identity)
                if tombstone.state is not GenerationGrantState.REVOKED:
                    raise ObligationConflictError("cannot revoke a grant that was already released")
                return tombstone.state
            self._require_identity(record.grant.identity, identity)
            state = record.grant.revoke(reason)
            self._retire_locked(
                record,
                state,
                reason,
                now_s=retired_at_s,
            )
            return state

    def sweep_expired(self, now_s: float) -> tuple[GenerationGrantIdentity, ...]:
        """Revoke expired responsibility; callers perform abort fan-out."""
        expired: list[GenerationGrantIdentity] = []
        with self._lock:
            self._purge_tombstones_locked(now_s)
            for record in tuple(self._records.values()):
                if record.grant.check_expiry(now_s) is GenerationGrantState.REVOKED:
                    expired.append(record.grant.identity)
                    self._retire_locked(
                        record,
                        GenerationGrantState.REVOKED,
                        record.grant.reason,
                        now_s=now_s,
                    )
        return tuple(expired)

    def scheduler_inserted(
        self,
        identity: GenerationGrantIdentity,
    ) -> bool:
        with self._lock:
            record = self._records.get(identity.consumer_grant_id)
            if record is None:
                return False
            self._require_identity(record.grant.identity, identity)
            return record.scheduler_inserted_at_s is not None

    def validate_identity(
        self,
        identity: GenerationGrantIdentity,
    ) -> GenerationGrantState:
        """Validate exact grant identity without changing lifecycle state."""
        with self._lock:
            record = self._records.get(identity.consumer_grant_id)
            if record is not None:
                self._require_identity(record.grant.identity, identity)
                return record.grant.state
            tombstone = self._require_tombstone(identity)
            return tombstone.state

    def _require_live_record(
        self,
        identity: GenerationGrantIdentity,
    ) -> _GenerationGrantRecord:
        record = self._records.get(identity.consumer_grant_id)
        if record is None:
            tombstone = self._tombstones.get(identity.consumer_grant_id)
            if tombstone is not None:
                self._require_identity(tombstone.identity, identity)
                raise RuntimeError("generation grant is already terminal")
            if self._terminal_ids.contains(identity.consumer_grant_id):
                raise RuntimeError(
                    "generation grant identity is terminal or replay protection is saturated"
                )
            raise KeyError(f"generation grant {identity.consumer_grant_id} does not exist")
        self._require_identity(record.grant.identity, identity)
        return record

    def _require_tombstone(
        self,
        identity: GenerationGrantIdentity,
    ) -> _GenerationGrantTombstone:
        tombstone = self._tombstones.get(identity.consumer_grant_id)
        if tombstone is None:
            if self._terminal_ids.contains(identity.consumer_grant_id):
                raise RuntimeError(
                    "generation grant identity is terminal or replay protection is saturated"
                )
            raise KeyError(f"generation grant {identity.consumer_grant_id} does not exist")
        self._require_identity(tombstone.identity, identity)
        return tombstone

    @staticmethod
    def _require_identity(
        actual: GenerationGrantIdentity,
        expected: GenerationGrantIdentity,
    ) -> None:
        if actual != expected:
            raise ObligationConflictError(
                "consumer_grant_id was replayed with a conflicting generation identity"
            )

    def _retire_locked(
        self,
        record: _GenerationGrantRecord,
        state: GenerationGrantState,
        reason: str,
        *,
        now_s: float,
    ) -> None:
        grant_id = record.grant.identity.consumer_grant_id
        self._records.pop(grant_id, None)
        self._terminal_ids.add(grant_id, now_s=now_s)
        existing = self._tombstones.get(grant_id)
        retain_until_s = now_s + self._replay_horizon_s
        if existing is not None:
            if existing.identity != record.grant.identity or existing.state is not state:
                raise ObligationConflictError(
                    "generation grant retirement conflicts with its terminal tombstone"
                )
            # A revoke reason is diagnostic context, not a safety identity fact.
            self._tombstones[grant_id] = _GenerationGrantTombstone(
                existing.identity,
                existing.state,
                existing.reason or reason,
                max(existing.retain_until_s, retain_until_s),
            )
            return
        if len(self._tombstones) >= self._max_tombstones:
            self._tombstones.pop(next(iter(self._tombstones)))
        self._tombstones[grant_id] = _GenerationGrantTombstone(
            record.grant.identity,
            state,
            reason,
            retain_until_s,
        )

    def _purge_tombstones_locked(self, now_s: float) -> None:
        self._terminal_ids.advance(now_s)
        for grant_id, tombstone in tuple(self._tombstones.items()):
            if now_s >= tombstone.retain_until_s:
                self._tombstones.pop(grant_id, None)


@dataclass(frozen=True, slots=True)
class _PendingArtifactRenewal:
    identity: ArtifactObligationIdentity
    sequence: int
    expires_at_s: float


@dataclass(frozen=True, slots=True)
class ArtifactRenewalDecision:
    """Endpoint-owned result for one sequenced artifact renewal."""

    state: ArtifactObligationState
    expires_at_s: float | None
    artifact_ready: bool


@dataclass(frozen=True, slots=True)
class _ArtifactTombstone:
    identity: ArtifactObligationIdentity
    state: ArtifactObligationState
    retain_until_s: float


class ArtifactObligationRegistry:
    """CTX authority for one renewable artifact obligation per GEN grant."""

    def __init__(
        self,
        *,
        max_tombstones: int = 4096,
        max_pending_renewals: int = 4096,
        max_live_obligations: int = 4096,
        replay_filter_capacity: int = 262144,
        replay_horizon_s: float = 1200.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_tombstones <= 0:
            raise ValueError("max_tombstones must be positive")
        if max_pending_renewals <= 0:
            raise ValueError("max_pending_renewals must be positive")
        if max_live_obligations <= 0:
            raise ValueError("max_live_obligations must be positive")
        if replay_filter_capacity <= 0:
            raise ValueError("replay_filter_capacity must be positive")
        if replay_horizon_s <= 0:
            raise ValueError("replay_horizon_s must be positive")
        self._max_tombstones = max_tombstones
        self._max_pending_renewals = max_pending_renewals
        self._max_live_obligations = max_live_obligations
        self._replay_horizon_s = replay_horizon_s
        self._clock = clock
        self._leases: dict[UUID, ArtifactObligationLease] = {}
        self._pending_renewals: dict[UUID, _PendingArtifactRenewal] = {}
        # One reservation spans every non-terminal artifact-retention state.
        # In particular, a renewal-before-artifact promotion must not consume
        # a second slot or bypass the hard live-obligation capacity.
        self._capacity_reservations: set[UUID] = set()
        self._applied_renewals: dict[UUID, tuple[int, float]] = {}
        self._tombstones: dict[UUID, _ArtifactTombstone] = {}
        self._terminal_ids = TerminalIdentityFilter(
            replay_filter_capacity,
            replay_horizon_s=replay_horizon_s,
            clock=clock,
        )
        self._lock = threading.Lock()

    @property
    def live_obligation_count(self) -> int:
        with self._lock:
            return len(self._leases)

    @property
    def reserved_obligation_count(self) -> int:
        """Return capacity held by pending, unbound, or live artifacts."""
        with self._lock:
            return len(self._capacity_reservations)

    def reserve_unbound(
        self,
        grant_id: UUID,
        *,
        now_s: float,
    ) -> None:
        """Reserve CTX retention capacity before the GEN identity is known."""
        with self._lock:
            self._purge_tombstones_locked(now_s)
            self._expire_pending_locked(now_s)
            if self._terminal_ids.contains(grant_id):
                raise RuntimeError(
                    "artifact identity is terminal or replay protection is saturated"
                )
            self._reserve_capacity_locked(grant_id)

    def release_unbound(self, grant_id: UUID) -> None:
        """Release a reservation that never became a pending/live obligation."""
        with self._lock:
            if grant_id in self._leases or grant_id in self._pending_renewals:
                raise RuntimeError(
                    "cannot release unbound artifact capacity after obligation promotion"
                )
            self._capacity_reservations.discard(grant_id)

    def register(
        self,
        identity: ArtifactObligationIdentity,
        *,
        now_s: float,
        expires_at_s: float,
    ) -> ArtifactObligationLease:
        if expires_at_s <= now_s:
            raise ValueError("artifact obligation expiry must be in the future")
        grant_id = identity.grant.consumer_grant_id
        with self._lock:
            self._purge_tombstones_locked(now_s)
            self._expire_pending_locked(now_s)
            existing = self._leases.get(grant_id)
            if existing is not None:
                self._require_identity(existing.identity, identity)
                return existing
            tombstone = self._tombstones.get(grant_id)
            if tombstone is not None:
                self._require_identity(tombstone.identity, identity)
                raise RuntimeError("artifact obligation is already terminal")
            if self._terminal_ids.contains(grant_id):
                raise RuntimeError(
                    "artifact identity is terminal or replay protection is saturated"
                )

            pending = self._pending_renewals.get(grant_id)
            if pending is not None:
                self._require_identity(pending.identity, identity)
            reservation_created = self._reserve_capacity_locked(grant_id)
            lease = ArtifactObligationLease(identity, expires_at_s=expires_at_s)
            try:
                if pending is not None:
                    lease.renew(
                        sequence=pending.sequence,
                        now_s=now_s,
                        expires_at_s=pending.expires_at_s,
                    )
                    self._applied_renewals[grant_id] = (
                        pending.sequence,
                        pending.expires_at_s,
                    )
                self._leases[grant_id] = lease
                self._pending_renewals.pop(grant_id, None)
            except Exception:
                if reservation_created:
                    self._capacity_reservations.discard(grant_id)
                raise
            return lease

    def renew_or_defer(
        self,
        identity: ArtifactObligationIdentity,
        *,
        sequence: int,
        now_s: float,
        ttl_s: float,
    ) -> ArtifactRenewalDecision:
        """Accept a generation-safe renewal even if CTX has not arrived yet."""
        if sequence < 0:
            raise ValueError("renewal sequence must be non-negative")
        if ttl_s <= 0:
            raise ValueError("artifact renewal TTL must be positive")
        grant_id = identity.grant.consumer_grant_id
        with self._lock:
            self._purge_tombstones_locked(now_s)
            self._expire_pending_locked(now_s)
            lease = self._leases.get(grant_id)
            if lease is not None:
                self._require_identity(lease.identity, identity)
                applied = self._applied_renewals.get(grant_id)
                if applied is not None and sequence < applied[0]:
                    state = lease.check_expiry(now_s)
                    if state is not ArtifactObligationState.ACTIVE:
                        self._retire_locked(identity, state, now_s=now_s)
                    return ArtifactRenewalDecision(
                        state,
                        lease.expires_at_s if state is ArtifactObligationState.ACTIVE else None,
                        True,
                    )
                if applied is not None and sequence == applied[0]:
                    state = lease.check_expiry(now_s)
                    if state is not ArtifactObligationState.ACTIVE:
                        self._retire_locked(identity, state, now_s=now_s)
                    return ArtifactRenewalDecision(
                        state,
                        lease.expires_at_s if state is ArtifactObligationState.ACTIVE else None,
                        True,
                    )
                expires_at_s = now_s + ttl_s
                state = lease.renew(
                    sequence=sequence,
                    now_s=now_s,
                    expires_at_s=expires_at_s,
                )
                if state is ArtifactObligationState.ACTIVE:
                    self._applied_renewals[grant_id] = (
                        sequence,
                        expires_at_s,
                    )
                else:
                    self._retire_locked(identity, state, now_s=now_s)
                return ArtifactRenewalDecision(
                    state,
                    lease.expires_at_s if state is ArtifactObligationState.ACTIVE else None,
                    True,
                )
            tombstone = self._tombstones.get(grant_id)
            if tombstone is not None:
                self._require_identity(tombstone.identity, identity)
                return ArtifactRenewalDecision(tombstone.state, None, False)
            if self._terminal_ids.contains(grant_id):
                return ArtifactRenewalDecision(
                    ArtifactObligationState.ABANDONED,
                    None,
                    False,
                )
            pending = self._pending_renewals.get(grant_id)
            if pending is not None:
                self._require_identity(pending.identity, identity)
                if sequence < pending.sequence:
                    return ArtifactRenewalDecision(
                        ArtifactObligationState.ACTIVE,
                        pending.expires_at_s,
                        False,
                    )
                if sequence == pending.sequence:
                    return ArtifactRenewalDecision(
                        ArtifactObligationState.ACTIVE,
                        pending.expires_at_s,
                        False,
                    )
            if pending is None and len(self._pending_renewals) >= self._max_pending_renewals:
                raise RuntimeError("pending artifact-renewal capacity is exhausted")
            expires_at_s = now_s + ttl_s
            reservation_created = self._reserve_capacity_locked(grant_id)
            try:
                self._pending_renewals[grant_id] = _PendingArtifactRenewal(
                    identity=identity,
                    sequence=sequence,
                    expires_at_s=expires_at_s,
                )
            except Exception:
                if reservation_created:
                    self._capacity_reservations.discard(grant_id)
                raise
            return ArtifactRenewalDecision(
                ArtifactObligationState.ACTIVE,
                expires_at_s,
                False,
            )

    def release(
        self,
        identity: ArtifactObligationIdentity,
        *,
        now_s: float | None = None,
    ) -> ArtifactObligationState:
        return self._retire(
            identity,
            ArtifactObligationState.RELEASED,
            now_s=self._clock() if now_s is None else now_s,
        )

    def abandon(
        self,
        identity: ArtifactObligationIdentity,
        *,
        now_s: float | None = None,
    ) -> ArtifactObligationState:
        return self._retire(
            identity,
            ArtifactObligationState.ABANDONED,
            now_s=self._clock() if now_s is None else now_s,
        )

    def sweep_expired(self, now_s: float) -> tuple[ArtifactObligationIdentity, ...]:
        """Abandon expired obligations; callers begin local fencing/abort."""
        expired: list[ArtifactObligationIdentity] = []
        with self._lock:
            self._purge_tombstones_locked(now_s)
            for lease in tuple(self._leases.values()):
                if lease.check_expiry(now_s) is ArtifactObligationState.ABANDONED:
                    expired.append(lease.identity)
                    self._retire_locked(
                        lease.identity,
                        ArtifactObligationState.ABANDONED,
                        now_s=now_s,
                    )
            expired.extend(self._expire_pending_locked(now_s))
        return tuple(expired)

    def _retire(
        self,
        identity: ArtifactObligationIdentity,
        state: ArtifactObligationState,
        *,
        now_s: float,
    ) -> ArtifactObligationState:
        with self._lock:
            self._purge_tombstones_locked(now_s)
            self._expire_pending_locked(now_s)
            grant_id = identity.grant.consumer_grant_id
            lease = self._leases.get(grant_id)
            if lease is not None:
                self._require_identity(lease.identity, identity)
                actual = (
                    lease.release()
                    if state is ArtifactObligationState.RELEASED
                    else lease.abandon()
                )
                if actual is not state:
                    raise ObligationConflictError(
                        "artifact obligation already has a conflicting terminal state"
                    )
                self._retire_locked(identity, state, now_s=now_s)
                return state

            pending = self._pending_renewals.get(grant_id)
            if pending is not None:
                self._require_identity(pending.identity, identity)
                self._pending_renewals.pop(grant_id, None)
                self._remember_tombstone_locked(identity, state, now_s=now_s)
                return state

            tombstone = self._tombstones.get(grant_id)
            if tombstone is None:
                # Abort-before-create is itself a generation-safe terminal fact.
                self._remember_tombstone_locked(identity, state, now_s=now_s)
                return state
            self._require_identity(tombstone.identity, identity)
            if tombstone.state is not state:
                raise ObligationConflictError(
                    "artifact obligation already has a conflicting terminal state"
                )
            return tombstone.state

    def _retire_locked(
        self,
        identity: ArtifactObligationIdentity,
        state: ArtifactObligationState,
        *,
        now_s: float,
    ) -> None:
        grant_id = identity.grant.consumer_grant_id
        self._leases.pop(grant_id, None)
        self._pending_renewals.pop(grant_id, None)
        self._applied_renewals.pop(grant_id, None)
        self._remember_tombstone_locked(identity, state, now_s=now_s)

    def _remember_tombstone_locked(
        self,
        identity: ArtifactObligationIdentity,
        state: ArtifactObligationState,
        *,
        now_s: float,
    ) -> None:
        grant_id = identity.grant.consumer_grant_id
        self._capacity_reservations.discard(grant_id)
        self._terminal_ids.add(grant_id, now_s=now_s)
        retain_until_s = now_s + self._replay_horizon_s
        existing = self._tombstones.get(grant_id)
        if existing is not None:
            if existing.identity != identity or existing.state is not state:
                raise ObligationConflictError(
                    "artifact obligation retirement conflicts with its terminal tombstone"
                )
            self._tombstones[grant_id] = _ArtifactTombstone(
                identity,
                state,
                max(existing.retain_until_s, retain_until_s),
            )
            return
        if len(self._tombstones) < self._max_tombstones:
            self._tombstones[grant_id] = _ArtifactTombstone(
                identity,
                state,
                retain_until_s,
            )

    def _expire_pending_locked(
        self,
        now_s: float,
    ) -> list[ArtifactObligationIdentity]:
        expired: list[ArtifactObligationIdentity] = []
        for pending in tuple(self._pending_renewals.values()):
            if now_s >= pending.expires_at_s:
                expired.append(pending.identity)
                self._pending_renewals.pop(
                    pending.identity.grant.consumer_grant_id,
                    None,
                )
                self._applied_renewals.pop(
                    pending.identity.grant.consumer_grant_id,
                    None,
                )
                self._remember_tombstone_locked(
                    pending.identity,
                    ArtifactObligationState.ABANDONED,
                    now_s=now_s,
                )
        return expired

    def _reserve_capacity_locked(self, grant_id: UUID) -> bool:
        if grant_id in self._capacity_reservations:
            return False
        if len(self._capacity_reservations) >= self._max_live_obligations:
            raise RuntimeError("artifact obligation capacity is exhausted")
        self._capacity_reservations.add(grant_id)
        return True

    def _purge_tombstones_locked(self, now_s: float) -> None:
        self._terminal_ids.advance(now_s)
        for grant_id, tombstone in tuple(self._tombstones.items()):
            if now_s >= tombstone.retain_until_s:
                self._tombstones.pop(grant_id, None)

    @staticmethod
    def _require_identity(
        actual: ArtifactObligationIdentity,
        expected: ArtifactObligationIdentity,
    ) -> None:
        if actual != expected:
            raise ObligationConflictError(
                "consumer_grant_id was replayed with a conflicting artifact identity"
            )


class ObligationExpirySupervisor:
    """Runs expiry actions without treating a deadline as physical proof."""

    def __init__(
        self,
        generation_grants: GenerationAdmissionRegistry,
        artifact_obligations: ArtifactObligationRegistry,
        *,
        on_generation_expired: Callable[[GenerationGrantIdentity], None],
        on_artifact_expired: Callable[[ArtifactObligationIdentity], None],
    ) -> None:
        self._generation_grants = generation_grants
        self._artifact_obligations = artifact_obligations
        self._on_generation_expired = on_generation_expired
        self._on_artifact_expired = on_artifact_expired

    def poll(self, now_s: float) -> None:
        for identity in self._generation_grants.sweep_expired(now_s):
            self._on_generation_expired(identity)
        for identity in self._artifact_obligations.sweep_expired(now_s):
            self._on_artifact_expired(identity)


__all__ = [
    "ArtifactObligationRegistry",
    "GenerationAdmissionRegistry",
    "GenerationGrantDecision",
    "ObligationExpirySupervisor",
    "TerminalIdentityFilter",
]
