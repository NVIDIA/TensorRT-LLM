# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the real lifecycle; replace only its native/GPU transport boundary."""

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import create_autospec, patch

import pytest
from pydantic import ValidationError

from tensorrt_llm.executor.checkpoint import (
    AbortCheckpointRequest,
    ActivateRequest,
    CheckpointBackend,
    CheckpointContext,
    CheckpointLifecycle,
    CheckpointPrepareNotDispatchedError,
    CheckpointProfile,
    CheckpointStage,
    CheckpointStep,
    GenerationValidation,
    HostRestoreAttestation,
    PrepareCaptureRequest,
    RankCheckpointEvidence,
    ValidateRuntimeRequest,
    checkpoint_enabled,
    checkpoint_probe_sampling_params,
    validate_checkpoint_config,
    validate_checkpoint_evidence,
)
from tensorrt_llm.llmapi.llm_args import KvCacheConfig, TorchLlmArgs

pytestmark = pytest.mark.cpu_only


def test_checkpoint_opt_in_is_per_engine(monkeypatch) -> None:
    monkeypatch.setenv("TRTLLM_ENABLE_CHECKPOINT_LIFECYCLE", "1")
    assert not checkpoint_enabled(None)
    assert not checkpoint_enabled({})
    assert not checkpoint_enabled({"TRTLLM_ENABLE_CHECKPOINT_LIFECYCLE": "0"})
    assert checkpoint_enabled({"TRTLLM_ENABLE_CHECKPOINT_LIFECYCLE": "1"})


@pytest.mark.parametrize("length", [2, 8, 64])
def test_probe_sampling_helper(length) -> None:
    params = checkpoint_probe_sampling_params(length)
    assert params.max_tokens == length
    assert params.temperature == 0 and params.ignore_eos
    assert params.end_id == -1 and params.pad_id == 0
    params.max_tokens = 99
    assert checkpoint_probe_sampling_params(length).max_tokens == length


@pytest.mark.parametrize("length", [0, 1, 65])
def test_probe_sampling_helper_rejects_invalid_length(length) -> None:
    with pytest.raises(ValueError, match="2 through 64"):
        checkpoint_probe_sampling_params(length)


@pytest.mark.parametrize(
    "field,value",
    [
        ("tensor_parallel_size", 4),
        ("pipeline_parallel_size", 2),
        ("context_parallel_size", 2),
        ("allreduce_strategy", "AUTO"),
        ("gather_generation_logits", True),
        ("num_serve_frontends", 2),
        ("orchestrator_type", "ray"),
        ("num_postprocess_workers", 1),
    ],
)
def test_requested_profile_preflight_names_unsupported_field(monkeypatch, field, value) -> None:
    monkeypatch.delenv("TLLM_WORKER_USE_SINGLE_PROCESS", raising=False)
    args = TorchLlmArgs(
        model="/tmp/dummy_model",
        allreduce_strategy="NCCL",
        kv_cache_config=KvCacheConfig(
            use_kv_cache_manager_v2=True, enable_block_reuse=False, host_cache_size=0
        ),
    )
    validate_checkpoint_config(args)
    changed = args.model_copy(update={field: value})
    with pytest.raises(ValueError, match=field):
        validate_checkpoint_config(changed)


@pytest.mark.parametrize(
    "field,value",
    [
        ("use_kv_cache_manager_v2", False),
        ("enable_block_reuse", True),
        ("host_cache_size", None),
        ("disk_cache_size", 1024),
    ],
)
def test_requested_profile_preflight_rejects_unqualified_storage(field, value) -> None:
    args = TorchLlmArgs(
        model="/tmp/dummy_model",
        allreduce_strategy="NCCL",
        kv_cache_config=KvCacheConfig(
            use_kv_cache_manager_v2=True, enable_block_reuse=False, host_cache_size=0
        ),
    )
    args.kv_cache_config = args.kv_cache_config.model_copy(update={field: value})
    with pytest.raises(ValueError, match=f"kv_cache_config.{field}"):
        validate_checkpoint_config(args)


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


@pytest.mark.parametrize(
    "field,value",
    [
        ("checks", ["progression_stopped"]),
        ("state_digest", ""),
        ("session_id", "old"),
        ("profile_digest", "old"),
        ("cohort_digest", "old"),
    ],
)
def test_abort_requires_complete_evidence(lifecycle, backend, field, value) -> None:
    lifecycle.prepare_capture(PrepareCaptureRequest(context=context(lifecycle, "prepare")))
    request = AbortCheckpointRequest(context=context(lifecycle, "abort"), reason="test")
    ranks = evidence(request.context)
    ranks[1] = ranks[1].model_copy(update={field: value})
    backend.step.return_value = ranks
    backend.step.side_effect = None
    result = lifecycle.abort_operation(request)
    assert result.status == "failed"
    assert not lifecycle.admission_open()


@pytest.mark.parametrize("step", list(CheckpointStep))
def test_shared_evidence_validation_covers_every_step(lifecycle, step) -> None:
    ctx = context(lifecycle, "step")
    ranks = evidence(ctx)
    validate_checkpoint_evidence(step, ctx, ranks, [0, 1])
    ranks[1].checks = []
    with pytest.raises(RuntimeError, match="incomplete"):
        validate_checkpoint_evidence(step, ctx, ranks, [0, 1])


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


@pytest.mark.parametrize("terminal", ["validation_failure", "abort"])
def test_new_activation_after_terminal_attempt_requires_cleanup(
    lifecycle, backend, terminal
) -> None:
    """A revoked readiness ID must not hide a terminal attempt as a retryable conflict."""
    prepare(lifecycle)
    if terminal == "validation_failure":
        backend.generate.side_effect = TimeoutError()
        result = lifecycle.validate_runtime(
            ValidateRuntimeRequest(
                context=context(lifecycle, "validate", True),
                source_session_id=lifecycle.get_lifecycle_status().source_session_id,
                validation=GenerationValidation(prompt_token_ids=[1], expected_token_ids=[11, 12]),
            )
        )
        assert result.status == "failed"
        assert not result.native_prepare_not_dispatched
        assert result.cleanup_status == "host_confirmation_required"
        backend.fail_closed.assert_called_once()
        assert backend.step.call_args.args[0] == CheckpointStep.VALIDATION_START
    else:
        result = lifecycle.abort_operation(
            AbortCheckpointRequest(context=context(lifecycle, "abort"), reason="operator")
        )
        assert result.status == "completed"
    ctx = context(lifecycle, "activate", terminal == "validation_failure")
    request = ActivateRequest(
        context=ctx,
        readiness_id="c" * 64,
        host_restore=HostRestoreAttestation(
            result_id="host-result",
            status="succeeded",
            session_id=ctx.session_id,
            template_id=ctx.template_id,
            profile_digest=ctx.profile_digest,
            cohort_digest=ctx.cohort_digest,
        ),
    )
    before = backend.mock_calls.copy()
    with pytest.raises(RuntimeError, match="terminal; progression is not permitted"):
        lifecycle.activate(request)
    assert backend.mock_calls == before
    assert not lifecycle.admission_open()


def test_prepare_budget_expires_before_backend_dispatch(lifecycle, backend) -> None:
    """An accepted operation can fail before any backend work without native revocation."""
    request = PrepareCaptureRequest(context=context(lifecycle, "prepare"))
    with patch.object(lifecycle, "_remaining", side_effect=TimeoutError("expired")):
        result = lifecycle.prepare_capture(request)
    assert result.status == "failed"
    assert result.native_prepare_not_dispatched
    assert result.cleanup_status == "not_requested"
    assert not lifecycle.admission_open()
    assert not lifecycle.get_lifecycle_status().readiness.capture_safe
    assert lifecycle.prepare_capture(request) == result
    backend.step.assert_not_called()
    backend.fail_closed.assert_not_called()


def test_predispatch_failure_does_not_override_explicit_abort(lifecycle, backend) -> None:
    """A concurrent abort retains its native revocation and cannot reopen serving."""
    entered, release, revoked = threading.Event(), threading.Event(), threading.Event()

    def step(action, ctx, *args):
        if action == CheckpointStep.PREPARE:
            entered.set()
            assert release.wait(5)
            raise CheckpointPrepareNotDispatchedError("HTTP drain expired")
        assert action == CheckpointStep.ABORT
        return evidence(ctx)

    backend.step.side_effect = step
    backend.fail_closed.side_effect = lambda *args: revoked.set()
    request = PrepareCaptureRequest(context=context(lifecycle, "prepare"))
    with ThreadPoolExecutor(max_workers=2) as pool:
        preparing = pool.submit(lifecycle.prepare_capture, request)
        try:
            assert entered.wait(5)
            abort = AbortCheckpointRequest(context=context(lifecycle, "abort"), reason="operator")
            stopping = pool.submit(lifecycle.abort_operation, abort)
            assert revoked.wait(5)
        finally:
            release.set()
        prepared = preparing.result(5)
        stopped = stopping.result(5)
    assert prepared.status == "failed"
    assert prepared.native_prepare_not_dispatched
    assert prepared.cleanup_status == "host_confirmation_required"
    assert stopped.status == "completed"
    assert lifecycle.get_lifecycle_status().stage == CheckpointStage.ABORTED
    assert not lifecycle.admission_open()
    assert lifecycle.prepare_capture(request) == prepared
    backend.fail_closed.assert_called_once()


@pytest.mark.parametrize("cleanup_error", [ValueError("invalid RPC"), TypeError("bad reply")])
def test_failure_cleanup_cannot_leave_journal_in_progress(
    lifecycle, backend, cleanup_error
) -> None:
    """An unexpected cleanup exception cannot hide the original terminal failure."""
    backend.step.side_effect = RuntimeError("prepare failed")
    backend.fail_closed.side_effect = cleanup_error
    request = PrepareCaptureRequest(context=context(lifecycle, "prepare"))
    result = lifecycle.prepare_capture(request)
    assert result.status == "failed"
    assert result.error == "prepare failed"
    assert lifecycle._active_operation is None
    assert lifecycle._idle.is_set()
    assert not lifecycle.admission_open()
    assert lifecycle.prepare_capture(request) == result
    backend.step.assert_called_once()
    backend.fail_closed.assert_called_once()


@pytest.mark.parametrize("failure_at", ["fail_closed", "step"])
def test_abort_with_empty_exception_message_is_failed(lifecycle, backend, failure_at) -> None:
    """Both revocation and abort RPC failures remain failures when str(error) is empty."""
    prepare(lifecycle)
    getattr(backend, failure_at).side_effect = TimeoutError()
    request = AbortCheckpointRequest(context=context(lifecycle, "abort"), reason="operator")
    result = lifecycle.abort_operation(request)
    assert result.status == "failed"
    assert result.error == "TimeoutError"
    assert lifecycle.get_lifecycle_status().stage == CheckpointStage.FAILED
    assert not lifecycle.admission_open()
    before = backend.mock_calls.copy()
    assert lifecycle.abort_operation(request) == result
    assert backend.mock_calls == before


@pytest.mark.parametrize("cancelled", [False, True])
def test_fast_admission_checks_deadline_and_cancellation(lifecycle, monkeypatch, cancelled) -> None:
    """The fast path preserves the full status path's readiness validity rules."""
    assert lifecycle.admission_open()
    if cancelled:
        lifecycle._cancelled.set()
    else:
        lifecycle._deadline = 1
        monkeypatch.setattr(
            "tensorrt_llm.executor.checkpoint.time", SimpleNamespace(monotonic=lambda: 2)
        )
    assert not lifecycle.admission_open()
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
    monkeypatch.setattr(
        "tensorrt_llm.executor.checkpoint.time", SimpleNamespace(monotonic=lambda: 10**15)
    )
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


def test_invalid_probe_does_not_consume_restore_session(lifecycle, backend) -> None:
    prepare(lifecycle)
    backend.validate_probe.side_effect = ValueError("probe exceeds max_seq_len")
    request = ValidateRuntimeRequest(
        context=context(lifecycle, "invalid", True),
        source_session_id=lifecycle.get_lifecycle_status().source_session_id,
        validation=GenerationValidation(prompt_token_ids=[1], expected_token_ids=[2, 3]),
    )
    calls = backend.step.call_count
    with pytest.raises(ValueError, match="max_seq_len"):
        lifecycle.validate_runtime(request)
    status = lifecycle.get_lifecycle_status()
    assert status.restore_session_id is None and status.stage is CheckpointStage.CAPTURE_SAFE
    assert backend.step.call_count == calls


def test_legacy_request_has_no_private_validation_permit() -> None:
    from tensorrt_llm.executor.request import GenerationRequest
    from tensorrt_llm.sampling_params import SamplingParams

    request = GenerationRequest([1], SamplingParams(max_tokens=2))
    state = request.__getstate__()
    state.pop("_checkpoint_validation_token")
    restored = object.__new__(GenerationRequest)
    restored.__setstate__(state)
    assert restored._checkpoint_validation_token is None
