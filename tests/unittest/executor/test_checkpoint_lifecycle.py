# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the real lifecycle; replace only its native/GPU transport boundary."""

import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import create_autospec

import pytest
from pydantic import ValidationError

from tensorrt_llm.executor.checkpoint import (
    AbortCheckpointRequest,
    ActivateRequest,
    CheckpointBackend,
    CheckpointContext,
    CheckpointLifecycle,
    CheckpointProfile,
    CheckpointStage,
    CheckpointStep,
    GenerationValidation,
    HostRestoreAttestation,
    PrepareCaptureRequest,
    RankCheckpointEvidence,
    ValidateRuntimeRequest,
)

pytestmark = pytest.mark.cpu_only

CHECKS = [
    "admission_closed",
    "scheduler_drained",
    "no_external_exposure",
    "fresh_restore_session",
    "memory_present",
    "resident_state_matches",
    "private_validation_only",
    "public_admission_closed",
    "validation_retired",
    "warmed_graph_replayed",
    "activation_prepared",
    "runtime_valid",
    "activation_committed",
    "progression_stopped",
    "host_cleanup_required",
]


def evidence(context: CheckpointContext) -> list[RankCheckpointEvidence]:
    """Supply identity-matched native boundary observations for two ranks."""
    return [
        RankCheckpointEvidence(
            rank=rank,
            session_id=context.session_id,
            profile_digest=context.profile_digest,
            cohort_digest=context.cohort_digest,
            checks=CHECKS,
            state_digest="d" * 64,
        )
        for rank in range(2)
    ]


@pytest.fixture
def backend():
    """Mock native coordination, not the lifecycle state machine under test."""
    result = create_autospec(CheckpointBackend, instance=True, spec_set=True)
    result.describe.return_value = CheckpointProfile(
        profile_digest="a" * 64, cohort_digest="b" * 64, ranks=[0, 1]
    )
    result.step.side_effect = lambda step, context, *args: evidence(context)
    result.generate.return_value = [11, 12]
    return result


@pytest.fixture
def lifecycle(backend) -> CheckpointLifecycle:
    """Use the production state machine with a controlled native boundary."""
    return CheckpointLifecycle(backend)


def context(
    lifecycle: CheckpointLifecycle, operation: str, restore: bool = False
) -> CheckpointContext:
    """Build a request from current, engine-owned identity and version."""
    status = lifecycle.get_lifecycle_status()
    return CheckpointContext(
        protocol_version=1,
        operation_id=operation,
        session_id="destination" if restore else status.source_session_id,
        session_kind="restore" if restore else "source",
        template_id="template",
        profile_digest=status.profile.profile_digest,
        cohort_digest=status.profile.cohort_digest,
        expected_state_version=status.state_version,
        time_budget_seconds=30,
    )


def prepare(lifecycle: CheckpointLifecycle) -> PrepareCaptureRequest:
    """Establish the source safety boundary using the actual prepare operation."""
    request = PrepareCaptureRequest(context=context(lifecycle, "prepare"))
    assert lifecycle.prepare_capture(request).status == "completed"
    return request


def validate(lifecycle: CheckpointLifecycle) -> ValidateRuntimeRequest:
    """Run the production restore and private-validation sequence."""
    request = ValidateRuntimeRequest(
        context=context(lifecycle, "validate", True),
        source_session_id=lifecycle.get_lifecycle_status().source_session_id,
        validation=GenerationValidation(prompt_token_ids=[1, 2], expected_token_ids=[11, 12]),
    )
    assert lifecycle.validate_runtime(request).status == "completed"
    return request


def activation(lifecycle: CheckpointLifecycle) -> ActivateRequest:
    """Reference evidence returned by the engine, not caller readiness flags."""
    ctx = context(lifecycle, "activate", True)
    return ActivateRequest(
        context=ctx,
        readiness_id=lifecycle.get_lifecycle_status().operation.readiness_id,
        host_restore=HostRestoreAttestation(
            result_id="host-result",
            status="succeeded",
            session_id=ctx.session_id,
            template_id=ctx.template_id,
            profile_digest=ctx.profile_digest,
            cohort_digest=ctx.cohort_digest,
        ),
    )


def test_full_lifecycle_and_exact_retries(lifecycle, backend) -> None:
    """Only final activation opens admission; retries perform no native mutation."""
    source = prepare(lifecycle)
    assert lifecycle.get_lifecycle_status().readiness.capture_safe
    assert not lifecycle.get_lifecycle_status().readiness.admission_open
    assert lifecycle.prepare_capture(source).status == "completed"
    assert backend.step.call_count == 1
    destination = validate(lifecycle)
    assert lifecycle.get_lifecycle_status().readiness.memory_ready
    assert lifecycle.get_lifecycle_status().readiness.runtime_ready
    assert not lifecycle.get_lifecycle_status().readiness.serving_ready
    before = backend.step.call_count
    assert lifecycle.validate_runtime(destination).status == "completed"
    assert backend.step.call_count == before
    request = activation(lifecycle)
    assert lifecycle.activate(request).status == "completed"
    assert lifecycle.get_lifecycle_status().readiness.admission_open
    before = backend.step.call_count
    assert lifecycle.activate(request).status == "completed"
    assert backend.step.call_count == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("profile_digest", "c" * 64),
        ("cohort_digest", "c" * 64),
        ("session_id", "stale"),
        ("expected_state_version", 99),
    ],
)
def test_stale_identity_rejected_before_mutation(lifecycle, backend, field, value) -> None:
    """Reject invalid source identity without touching the native runtime."""
    ctx = context(lifecycle, "prepare").model_copy(update={field: value})
    with pytest.raises(ValueError):
        lifecycle.prepare_capture(PrepareCaptureRequest(context=ctx))
    backend.step.assert_not_called()


def test_conflicting_retry_rejected(lifecycle) -> None:
    """An operation ID cannot start a new budget or change its payload."""
    request = prepare(lifecycle)
    changed = request.model_copy(
        update={"context": request.context.model_copy(update={"time_budget_seconds": 60})}
    )
    with pytest.raises(ValueError, match="different arguments"):
        lifecycle.prepare_capture(changed)


@pytest.mark.parametrize(
    "failure", ["missing_rank", "duplicate_rank", "stale_session", "no_checks"]
)
def test_incomplete_evidence_fails_closed(lifecycle, backend, failure) -> None:
    """A partial or stale group acknowledgement cannot establish capture safety."""
    ctx = context(lifecycle, "prepare")
    ranks = evidence(ctx)
    if failure == "missing_rank":
        ranks.pop()
    elif failure == "duplicate_rank":
        ranks[1] = ranks[0]
    elif failure == "stale_session":
        ranks[0].session_id = "stale"
    else:
        ranks[0].checks = []
    backend.step.side_effect = None
    backend.step.return_value = ranks
    result = lifecycle.prepare_capture(PrepareCaptureRequest(context=ctx))
    assert result.status == "failed"
    assert not lifecycle.get_lifecycle_status().readiness.capture_safe
    assert not lifecycle.get_lifecycle_status().readiness.admission_open
    backend.fail_closed.assert_called_once()


def test_validation_mismatch_retires_probe_but_never_activates(lifecycle, backend) -> None:
    """Generation parity failure must still retire private work and fail closed."""
    prepare(lifecycle)
    backend.generate.return_value = [99, 99]
    request = ValidateRuntimeRequest(
        context=context(lifecycle, "validate", True),
        source_session_id=lifecycle.get_lifecycle_status().source_session_id,
        validation=GenerationValidation(prompt_token_ids=[1], expected_token_ids=[11, 12]),
    )
    result = lifecycle.validate_runtime(request)
    assert result.status == "failed"
    assert backend.step.call_args.args[0] == CheckpointStep.VALIDATION_END
    assert result.readiness_id is None
    assert not lifecycle.get_lifecycle_status().readiness.admission_open


def test_host_attestation_must_match_destination(lifecycle, backend) -> None:
    """Final host success for another session cannot activate this destination."""
    prepare(lifecycle)
    validate(lifecycle)
    request = activation(lifecycle)
    request.host_restore.session_id = "other"
    before = backend.step.call_count
    with pytest.raises(ValueError, match="another destination"):
        lifecycle.activate(request)
    assert backend.step.call_count == before


def test_status_is_detached(lifecycle) -> None:
    """Callers cannot mutate engine-owned readiness through a returned object."""
    prepare(lifecycle)
    status = lifecycle.get_lifecycle_status()
    status.readiness.admission_open = True
    status.operation.rank_results.clear()
    assert not lifecycle.get_lifecycle_status().readiness.admission_open
    assert len(lifecycle.get_lifecycle_status().operation.rank_results) == 2


def test_retry_during_drain_joins_existing_operation(lifecycle, backend) -> None:
    """An in-flight retry does not launch a second native drain."""
    entered, release = threading.Event(), threading.Event()

    def blocked_step(step, ctx, *args):
        entered.set()
        assert release.wait(5)
        return evidence(ctx)

    backend.step.side_effect = blocked_step
    request = PrepareCaptureRequest(context=context(lifecycle, "prepare"))
    with ThreadPoolExecutor() as pool:
        future = pool.submit(lifecycle.prepare_capture, request)
        try:
            assert entered.wait(5)
            assert lifecycle.prepare_capture(request).status == "in_progress"
            assert backend.step.call_count == 1
        finally:
            release.set()
        assert future.result(5).status == "completed"


def test_abort_never_resumes_source(lifecycle) -> None:
    """Abort only stops progression; it does not prove host cleanup."""
    prepare(lifecycle)
    request = AbortCheckpointRequest(context=context(lifecycle, "abort"), reason="operator")
    result = lifecycle.abort_operation(request)
    assert result.cleanup_status == "host_confirmation_required"
    assert lifecycle.get_lifecycle_status().stage == CheckpointStage.ABORTED
    assert not lifecycle.get_lifecycle_status().readiness.admission_open
    assert lifecycle.abort_operation(request) == result


def test_failed_activation_retry_returns_same_failure(lifecycle, backend) -> None:
    """Failed operations retain retry identity after their readiness ID is revoked."""
    prepare(lifecycle)
    validate(lifecycle)
    request = activation(lifecycle)
    backend.step.side_effect = TimeoutError("missing rank")
    result = lifecycle.activate(request)
    assert result.status == "failed"
    before = backend.step.call_count
    assert lifecycle.activate(request) == result
    assert backend.step.call_count == before
    assert not lifecycle.get_lifecycle_status().readiness.admission_open


def test_abort_interrupts_drain_without_duplicate_cleanup(lifecycle, backend) -> None:
    """An exact concurrent abort retry joins one recorded abort operation."""
    entered, release, cancelled = threading.Event(), threading.Event(), threading.Event()

    def blocked_step(step, ctx, *args):
        if step == CheckpointStep.PREPARE:
            entered.set()
            assert release.wait(5)
        elif step == CheckpointStep.ABORT:
            assert lifecycle.get_lifecycle_status().stage == CheckpointStage.ABORTING
        return evidence(ctx)

    backend.step.side_effect = blocked_step
    backend.fail_closed.side_effect = lambda *args: cancelled.set()
    request = PrepareCaptureRequest(context=context(lifecycle, "prepare"))
    with ThreadPoolExecutor() as pool:
        preparing = pool.submit(lifecycle.prepare_capture, request)
        try:
            assert entered.wait(5)
            abort = AbortCheckpointRequest(context=context(lifecycle, "abort"), reason="cancel")
            stopping = pool.submit(lifecycle.abort_operation, abort)
            assert cancelled.wait(5)
            assert lifecycle.abort_operation(abort).status == "in_progress"
            assert not lifecycle.get_lifecycle_status().readiness.admission_open
        finally:
            release.set()
        assert preparing.result(5).status == "failed"
        result = stopping.result(5)
        assert result.status == "completed"
    before = backend.step.call_count
    assert lifecycle.abort_operation(abort) == result
    assert backend.step.call_count == before
    assert not lifecycle.get_lifecycle_status().readiness.capture_safe


def test_expired_attempt_cannot_activate_or_extend_budget(lifecycle, backend, monkeypatch) -> None:
    """A later activate command cannot restart the destination validation budget."""
    prepare(lifecycle)
    validate(lifecycle)
    request = activation(lifecycle)
    request.context.time_budget_seconds = 3600
    monkeypatch.setattr("tensorrt_llm.executor.checkpoint.time.monotonic", lambda: 10**15)
    before = backend.step.call_count
    with pytest.raises(TimeoutError):
        lifecycle.activate(request)
    assert backend.step.call_count == before
    assert not lifecycle.get_lifecycle_status().readiness.runtime_ready


@pytest.mark.parametrize("tokens", [[-1], [], [1]])
def test_validation_requires_nonnegative_multi_token_reference(tokens) -> None:
    """Require at least one decode step and reject invalid token IDs."""
    with pytest.raises(ValidationError):
        GenerationValidation(prompt_token_ids=[1], expected_token_ids=tokens)
