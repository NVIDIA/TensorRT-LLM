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

from tensorrt_llm._torch.disaggregation.lifecycle import PhysicalDisposition
from tensorrt_llm._torch.disaggregation.obligations import (
    ArtifactObligationIdentity,
    ArtifactObligationLease,
    ArtifactObligationState,
    GenerationGrantIdentity,
    GenerationGrantState,
    GenerationIntentGrant,
    ObligationConflictError,
    ReceiveCommitGate,
    ReceiveCommitState,
    SubmissionFence,
    SubmissionFenceState,
)
from tensorrt_llm._torch.disaggregation.protocol import (
    AllocationWireIdentity,
    AttemptIdentity,
    EndpointIdentity,
    OperationIdentity,
    PublicationIdentity,
    TransferProtocolIdentity,
)


def _uuid(value: int) -> UUID:
    return UUID(int=value)


def _attempt() -> AttemptIdentity:
    return AttemptIdentity(17, _uuid(2), 0, _uuid(3))


def _session() -> TransferProtocolIdentity:
    return TransferProtocolIdentity(
        attempt=_attempt(),
        transfer_session_id=_uuid(4),
        source_endpoint=EndpointIdentity("ctx", 0, _uuid(5)),
        destination_endpoint=EndpointIdentity("gen", 0, _uuid(6)),
    )


def _operation(value: int) -> OperationIdentity:
    return OperationIdentity(
        publication=PublicationIdentity(
            session=_session(),
            destination_allocation=AllocationWireIdentity("gen", 17, 2),
            operation_id=_uuid(value),
            slice_id=value,
            writer_rank=0,
        ),
        source_allocation=AllocationWireIdentity("ctx", 17, 1),
    )


def _grant() -> GenerationGrantIdentity:
    return GenerationGrantIdentity(
        consumer_grant_id=_uuid(7),
        attempt=_attempt(),
        generation_endpoint=EndpointIdentity("gen", 0, _uuid(6)),
    )


def test_grant_expiry_revokes_responsibility_without_physical_reuse_claim() -> None:
    grant = GenerationIntentGrant(_grant(), issued_at_s=1.0, expires_at_s=11.0)

    assert grant.check_expiry(10.0) is GenerationGrantState.ACTIVE
    assert grant.check_expiry(11.0) is GenerationGrantState.REVOKED
    assert grant.reason == "generation intent grant expired"


def test_artifact_renewal_is_monotone_and_expiry_only_abandons_obligation() -> None:
    lease = ArtifactObligationLease(
        ArtifactObligationIdentity(_grant()),
        expires_at_s=10.0,
    )

    assert lease.renew(sequence=1, now_s=2.0, expires_at_s=20.0) is ArtifactObligationState.ACTIVE
    assert lease.renew(sequence=0, now_s=3.0, expires_at_s=30.0) is ArtifactObligationState.ACTIVE
    with pytest.raises(ObligationConflictError):
        lease.renew(sequence=1, now_s=3.0, expires_at_s=21.0)
    assert lease.check_expiry(20.0) is ArtifactObligationState.ABANDONED


def test_submission_fence_separates_no_new_work_from_quiescence() -> None:
    fence = SubmissionFence(_session())
    first = _operation(8)
    second = _operation(9)
    fence.authorize(first)
    fence.authorize(second)
    assert fence.begin(first)
    assert fence.begin(second)

    assert fence.fence() is SubmissionFenceState.FENCING
    with pytest.raises(RuntimeError, match="closed"):
        fence.authorize(_operation(10))
    assert (
        fence.complete(first, PhysicalDisposition.QUIESCED_SUCCESS) is SubmissionFenceState.FENCING
    )
    with pytest.raises(RuntimeError, match="fenced and drained"):
        fence.mark_quiesced()
    assert (
        fence.complete(second, PhysicalDisposition.QUIESCED_FAILURE) is SubmissionFenceState.FENCED
    )
    assert fence.mark_quiesced() is SubmissionFenceState.QUIESCED


def test_submission_fence_rejects_conflicting_duplicate_completion() -> None:
    fence = SubmissionFence(_session())
    operation = _operation(8)
    fence.authorize(operation)
    fence.begin(operation)
    fence.complete(operation, PhysicalDisposition.QUIESCED_SUCCESS)

    with pytest.raises(ObligationConflictError):
        fence.complete(operation, PhysicalDisposition.QUIESCED_FAILURE)


def test_abort_commit_gate_is_monotone_in_either_race_order() -> None:
    aborted = ReceiveCommitGate(_session())
    assert aborted.abort("client deadline") is ReceiveCommitState.ABORTED
    assert aborted.commit() is ReceiveCommitState.ABORTED

    committed = ReceiveCommitGate(_session())
    assert committed.commit() is ReceiveCommitState.COMMITTED
    assert committed.abort("late client deadline") is ReceiveCommitState.COMMITTED
