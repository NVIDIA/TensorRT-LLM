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

from uuid import UUID

import pytest

from tensorrt_llm._torch.disaggregation.control_plane import (
    ArtifactObligationRegistry,
    GenerationAdmissionRegistry,
)
from tensorrt_llm._torch.disaggregation.obligations import (
    ArtifactObligationIdentity,
    ArtifactObligationState,
    GenerationGrantIdentity,
    GenerationGrantState,
    ObligationConflictError,
)
from tensorrt_llm._torch.disaggregation.protocol import AttemptIdentity, EndpointIdentity


def _uuid(value: int) -> UUID:
    return UUID(int=value)


def _grant(value: int = 7) -> GenerationGrantIdentity:
    return GenerationGrantIdentity(
        consumer_grant_id=_uuid(value),
        attempt=AttemptIdentity(17, _uuid(2), 0, _uuid(3)),
        generation_endpoint=EndpointIdentity("gen", 0, _uuid(6)),
    )


def test_generation_grant_is_the_single_queue_owner() -> None:
    registry = GenerationAdmissionRegistry(max_live_grants=1)
    identity = _grant()

    decision = registry.issue(
        identity,
        issued_at_s=1.0,
        expires_at_s=11.0,
    )

    assert decision.accepted
    assert registry.live_grant_count == 1
    assert registry.mark_scheduler_inserted(identity, now_s=2.0)
    assert registry.scheduler_inserted(identity)

    rejected = registry.issue(
        _grant(8),
        issued_at_s=2.0,
        expires_at_s=12.0,
    )
    assert not rejected.accepted
    assert "credit" in rejected.reason


def test_generation_grant_expiry_retires_credit_but_is_not_a_reuse_proof() -> None:
    registry = GenerationAdmissionRegistry(max_live_grants=1)
    identity = _grant()
    registry.issue(identity, issued_at_s=1.0, expires_at_s=11.0)

    assert registry.sweep_expired(10.0) == ()
    assert registry.sweep_expired(11.0) == (identity,)
    assert registry.live_grant_count == 0
    assert registry.release(identity) is GenerationGrantState.REVOKED


def test_generation_grant_is_checked_before_scheduler_submission() -> None:
    registry = GenerationAdmissionRegistry(max_live_grants=1)
    identity = _grant()
    registry.issue(identity, issued_at_s=1.0, expires_at_s=3.0)

    assert registry.validate_active(identity, now_s=2.0)
    assert not registry.scheduler_inserted(identity)
    assert not registry.validate_active(identity, now_s=3.0)
    assert registry.live_grant_count == 0


def test_artifact_renewal_can_precede_context_registration() -> None:
    registry = ArtifactObligationRegistry()
    identity = ArtifactObligationIdentity(_grant())

    assert (
        registry.renew_or_defer(
            identity,
            sequence=3,
            now_s=1.0,
            ttl_s=29.0,
        ).state
        is ArtifactObligationState.ACTIVE
    )
    lease = registry.register(identity, now_s=2.0, expires_at_s=10.0)

    assert lease.expires_at_s == 30.0
    assert registry.sweep_expired(29.0) == ()
    assert registry.sweep_expired(30.0) == (identity,)


def test_pending_artifact_promotion_preserves_one_hard_capacity_reservation() -> None:
    registry = ArtifactObligationRegistry(
        max_pending_renewals=2,
        max_live_obligations=1,
    )
    first = ArtifactObligationIdentity(_grant(7))
    second = ArtifactObligationIdentity(_grant(8))

    registry.renew_or_defer(
        first,
        sequence=0,
        now_s=1.0,
        ttl_s=10.0,
    )
    assert registry.reserved_obligation_count == 1

    registry.register(first, now_s=2.0, expires_at_s=10.0)
    assert registry.live_obligation_count == 1
    assert registry.reserved_obligation_count == 1

    with pytest.raises(RuntimeError, match="capacity is exhausted"):
        registry.renew_or_defer(
            second,
            sequence=0,
            now_s=2.0,
            ttl_s=10.0,
        )

    registry.release(first, now_s=3.0)
    registry.renew_or_defer(
        second,
        sequence=0,
        now_s=3.0,
        ttl_s=10.0,
    )
    assert registry.reserved_obligation_count == 1


@pytest.mark.parametrize("register_first", [False, True])
def test_artifact_renewal_replay_does_not_extend_expiry(
    register_first: bool,
) -> None:
    registry = ArtifactObligationRegistry()
    identity = ArtifactObligationIdentity(_grant())
    if register_first:
        registry.register(identity, now_s=1.0, expires_at_s=10.0)
    first = registry.renew_or_defer(
        identity,
        sequence=3,
        now_s=2.0,
        ttl_s=28.0,
    )
    duplicate = registry.renew_or_defer(
        identity,
        sequence=3,
        now_s=3.0,
        ttl_s=28.0,
    )

    assert first.state is ArtifactObligationState.ACTIVE
    assert duplicate.state is ArtifactObligationState.ACTIVE
    assert first.expires_at_s == 30.0
    assert duplicate.expires_at_s == first.expires_at_s


def test_abort_before_artifact_registration_blocks_late_creation() -> None:
    registry = ArtifactObligationRegistry()
    identity = ArtifactObligationIdentity(_grant())

    assert registry.abandon(identity) is ArtifactObligationState.ABANDONED
    with pytest.raises(RuntimeError, match="terminal"):
        registry.register(identity, now_s=1.0, expires_at_s=10.0)


def test_generation_and_artifact_replays_fail_closed_on_identity_conflict() -> None:
    grants = GenerationAdmissionRegistry(max_live_grants=2)
    original = _grant()
    grants.issue(original, issued_at_s=1.0, expires_at_s=10.0)
    conflicting = GenerationGrantIdentity(
        consumer_grant_id=original.consumer_grant_id,
        attempt=AttemptIdentity(18, _uuid(2), 0, _uuid(3)),
        generation_endpoint=original.generation_endpoint,
    )

    with pytest.raises(ObligationConflictError):
        grants.issue(conflicting, issued_at_s=1.0, expires_at_s=10.0)

    artifacts = ArtifactObligationRegistry()
    artifacts.register(
        ArtifactObligationIdentity(original),
        now_s=1.0,
        expires_at_s=10.0,
    )
    with pytest.raises(ObligationConflictError):
        artifacts.renew_or_defer(
            ArtifactObligationIdentity(conflicting),
            sequence=1,
            now_s=2.0,
            ttl_s=9.0,
        )


def test_revoke_replay_ignores_diagnostic_reason_differences() -> None:
    registry = GenerationAdmissionRegistry(max_live_grants=1)
    identity = _grant()
    registry.issue(identity, issued_at_s=1.0, expires_at_s=10.0)

    assert (
        registry.revoke(identity, "coordinator abandoned", now_s=2.0)
        is GenerationGrantState.REVOKED
    )
    assert registry.revoke(identity, "request failed", now_s=3.0) is GenerationGrantState.REVOKED


def test_pending_artifact_renewal_expires_and_cannot_resurrect() -> None:
    registry = ArtifactObligationRegistry(replay_horizon_s=5.0)
    identity = ArtifactObligationIdentity(_grant())
    registry.renew_or_defer(
        identity,
        sequence=1,
        now_s=1.0,
        ttl_s=2.0,
    )

    assert registry.sweep_expired(3.0) == (identity,)
    assert (
        registry.renew_or_defer(
            identity,
            sequence=2,
            now_s=4.0,
            ttl_s=2.0,
        ).state
        is ArtifactObligationState.ABANDONED
    )
    with pytest.raises(RuntimeError, match="terminal"):
        registry.register(identity, now_s=4.0, expires_at_s=6.0)


def test_terminal_filter_preserves_replay_horizon_after_exact_tombstone_expires() -> None:
    registry = GenerationAdmissionRegistry(
        max_live_grants=1,
        replay_horizon_s=2.0,
    )
    identity = _grant()
    registry.issue(identity, issued_at_s=1.0, expires_at_s=10.0)
    registry.revoke(identity, "done", now_s=2.0)

    replay = registry.issue(identity, issued_at_s=5.0, expires_at_s=10.0)

    assert not replay.accepted
    assert "terminal" in replay.reason


def test_generation_replay_history_does_not_throttle_sustained_throughput() -> None:
    registry = GenerationAdmissionRegistry(
        max_live_grants=1,
        max_tombstones=2,
        replay_filter_capacity=20000,
        replay_horizon_s=100.0,
    )

    for value in range(1, 10001):
        identity = _grant(value)
        decision = registry.issue(
            identity,
            issued_at_s=1.0,
            expires_at_s=10.0,
        )
        assert decision.accepted
        registry.release(identity, now_s=1.0)

    assert len(registry._tombstones) == 2
    replay = registry.issue(
        _grant(1),
        issued_at_s=50.0,
        expires_at_s=60.0,
    )
    assert not replay.accepted

    # Once two bounded filter generations elapse, this endpoint no longer
    # accumulates endpoint-lifetime false positives.
    replacement = registry.issue(
        _grant(1),
        issued_at_s=201.0,
        expires_at_s=211.0,
    )
    assert replacement.accepted


def test_generation_grant_renewal_replay_does_not_extend_expiry() -> None:
    registry = GenerationAdmissionRegistry(max_live_grants=1)
    identity = _grant()
    registry.issue(identity, issued_at_s=1.0, expires_at_s=10.0)

    first = registry.renew(
        identity,
        sequence=3,
        now_s=2.0,
        ttl_s=10.0,
    )
    duplicate = registry.renew(
        identity,
        sequence=3,
        now_s=5.0,
        ttl_s=10.0,
    )
    stale = registry.renew(
        identity,
        sequence=2,
        now_s=6.0,
        ttl_s=10.0,
    )
    fresh = registry.renew(
        identity,
        sequence=4,
        now_s=7.0,
        ttl_s=10.0,
    )

    assert first.state is GenerationGrantState.ACTIVE
    assert first.expires_at_s == 12.0
    assert duplicate.expires_at_s == first.expires_at_s
    assert stale.expires_at_s == first.expires_at_s
    assert fresh.expires_at_s == 17.0
