# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Controller-neutral operations for a fixed-cohort worker checkpoint.

Host attestations come from the authenticated controller, not from restored
memory. Engine evidence is produced by the backend and is never accepted as
client-supplied readiness flags.
"""

import hashlib
import json
import os
import secrets
import threading
import time
from enum import Enum
from typing import TYPE_CHECKING, Literal, Protocol

from pydantic import Field, NonNegativeInt

from tensorrt_llm.llmapi.utils import StrictBaseModel
from tensorrt_llm.logger import logger
from tensorrt_llm.sampling_params import SamplingParams

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

CHECKPOINT_ENABLE_ENV = "TRTLLM_ENABLE_CHECKPOINT_LIFECYCLE"
CHECKPOINT_REQUIRED_ENV = {
    "TLLM_NCCL_SYMMETRIC_ZERO_COPY": "0",
    "TRTLLM_PROXY_BATCH_REQUESTS": "0",
    "TRTLLM_GPU_KEEPALIVE": "0",
}
_MAX_OPERATION_HISTORY = 128


def checkpoint_enabled(env_overrides: dict[str, str] | None) -> bool:
    """Read the private per-engine opt-in, never the inherited process environment.

    Args:
        env_overrides: Explicit engine overrides; raw CLI scalar values are normalized
            like the LLM-args environment validator.

    Returns:
        Whether this engine explicitly enables native checkpoint controls.
    """
    return str((env_overrides or {}).get(CHECKPOINT_ENABLE_ENV, "0")) == "1"


def checkpoint_probe_sampling_params(num_tokens: int) -> SamplingParams:
    """Build identical sampling settings for reference and private probe generation.

    Pass the result to LLM.generate for references. Internal submitters must apply
    LLM's generation-config defaults before constructing a GenerationRequest.

    Args:
        num_tokens: Reference completion length, from 2 through 64.

    Returns:
        Fresh deterministic probe settings.

    Raises:
        ValueError: If the completion length is unsupported.
    """
    if not 2 <= num_tokens <= 64:
        raise ValueError("Checkpoint probes require 2 through 64 output tokens.")
    return SamplingParams(
        temperature=0, max_tokens=num_tokens, ignore_eos=True, end_id=-1, pad_id=0
    )


def validate_checkpoint_config(args: "TorchLlmArgs") -> None:
    """Reject unsupported requested settings before loading or warming a model.

    This preflight does not replace native checks on resolved resource owners.

    Args:
        args: Validated per-engine arguments, before model initialization.

    Raises:
        ValueError: If requested settings are outside the supported profile.
    """
    required = {
        "tensor_parallel_size": args.tensor_parallel_size in (1, 2),
        "pipeline_parallel_size": args.pipeline_parallel_size == 1,
        "context_parallel_size": args.context_parallel_size == 1,
        "enable_attention_dp": not args.enable_attention_dp,
        "num_serve_frontends": args.num_serve_frontends == 1,
        "num_postprocess_workers": args.num_postprocess_workers == 0,
        "enable_resource_governor": not args.enable_resource_governor,
        "orchestrator_type": args.orchestrator_type is None,
        "allreduce_strategy": args.allreduce_strategy == "NCCL",
        "torch_compile_config": args.torch_compile_config is None,
        "guided_decoding_backend": args.guided_decoding_backend is None,
        "gather_generation_logits": not args.gather_generation_logits,
        "kv_cache_config.use_kv_cache_manager_v2": args.kv_cache_config.use_kv_cache_manager_v2,
        "kv_cache_config.enable_block_reuse": not args.kv_cache_config.enable_block_reuse,
        "kv_cache_config.host_cache_size": args.kv_cache_config.host_cache_size == 0,
        "kv_cache_config.disk_cache_size": not args.kv_cache_config.disk_cache_size,
        "TLLM_WORKER_USE_SINGLE_PROCESS": os.environ.get("TLLM_WORKER_USE_SINGLE_PROCESS") != "1",
    }
    invalid = [name for name, supported in required.items() if not supported]
    if invalid:
        raise ValueError(
            "Unsupported checkpoint settings: "
            + ", ".join(invalid)
            + ". See the resident checkpoint profile requirements."
        )


class CheckpointContext(StrictBaseModel):
    """Identity and bounded budget for one mutation and its exact retries."""

    protocol_version: Literal[1] = Field(description="Checkpoint protocol version.")
    operation_id: str = Field(min_length=1, max_length=128, description="Unique action identifier.")
    session_id: str = Field(min_length=1, max_length=128, description="Source or fresh restore ID.")
    session_kind: Literal["source", "restore"] = Field(description="Session identity domain.")
    template_id: str = Field(
        min_length=1, max_length=128, description="Immutable artifact identity."
    )
    profile_digest: str = Field(
        pattern=r"^[0-9a-f]{64}$", description="Resolved engine profile hash."
    )
    cohort_digest: str = Field(pattern=r"^[0-9a-f]{64}$", description="Fixed participant set hash.")
    expected_state_version: int = Field(ge=0, description="Observed lifecycle version.")
    time_budget_seconds: float = Field(
        gt=0,
        le=3600,
        allow_inf_nan=False,
        description="Relative budget; retries do not restart it.",
    )


class PrepareCaptureRequest(StrictBaseModel):
    """Close admission and establish a resident, naturally drained template."""

    context: CheckpointContext = Field(description="Source operation identity and budget.")
    drain_policy: Literal["natural"] = Field(
        default="natural", description="Wait for accepted requests to finish."
    )
    capture_policy: Literal["resident"] = Field(
        default="resident", description="Retain the resident GPU storage."
    )


class GenerationValidation(StrictBaseModel):
    """A deterministic private probe selected with the artifact profile."""

    prompt_token_ids: list[NonNegativeInt] = Field(
        min_length=1, max_length=4096, description="Tokenized private probe prompt."
    )
    expected_token_ids: list[NonNegativeInt] = Field(
        min_length=2, max_length=64, description="Exact deterministic reference completion."
    )


class ValidateRuntimeRequest(StrictBaseModel):
    """Bind a fresh destination session and run engine-owned validation."""

    context: CheckpointContext = Field(
        description="Fresh destination operation identity and budget."
    )
    source_session_id: str = Field(
        min_length=1, max_length=128, description="Captured source identity."
    )
    validation: GenerationValidation = Field(description="Probe selected for the artifact profile.")


class HostRestoreAttestation(StrictBaseModel):
    """Final host success attested by the authorized, external controller."""

    result_id: str = Field(
        min_length=1, max_length=128, description="Controller's final host-result reference."
    )
    session_id: str = Field(min_length=1, max_length=128, description="Fresh destination session.")
    template_id: str = Field(
        min_length=1, max_length=128, description="Restored artifact identity."
    )
    profile_digest: str = Field(
        pattern=r"^[0-9a-f]{64}$", description="Host-validated engine profile."
    )
    cohort_digest: str = Field(
        pattern=r"^[0-9a-f]{64}$", description="Host-validated participant set."
    )
    status: Literal["succeeded"] = Field(description="Final host success, not a restored marker.")


class ActivateRequest(StrictBaseModel):
    """Reference current engine evidence and final host success."""

    context: CheckpointContext = Field(description="Current destination activation operation.")
    readiness_id: str = Field(
        min_length=1, max_length=128, description="Engine-issued runtime readiness reference."
    )
    host_restore: HostRestoreAttestation = Field(
        description="Final success attested by the authenticated controller."
    )


class AbortCheckpointRequest(StrictBaseModel):
    """Stop a pre-activation operation without authorizing source continuation."""

    context: CheckpointContext = Field(description="Current source or destination abort operation.")
    reason: str = Field(
        min_length=1, max_length=512, description="Reason for stopping pre-activation progress."
    )


class CheckpointStage(str, Enum):
    """Local lifecycle state, separate from native memory sleep state."""

    RUNNING = "running"
    PREPARING = "preparing"
    CAPTURE_SAFE = "capture_safe"
    VALIDATING = "validating"
    RUNTIME_READY = "runtime_ready"
    ACTIVATING = "activating"
    SERVING = "serving"
    ABORTED = "aborted"
    ABORTING = "aborting"
    FAILED = "failed"


class CheckpointStep(str, Enum):
    """Internal all-rank steps; none is an independent serving authorization."""

    PREPARE = "prepare"
    RESTORE = "restore"
    VALIDATION_START = "validation_start"
    VALIDATION_END = "validation_end"
    ACTIVATION_PREPARE = "activation_prepare"
    ACTIVATION_COMMIT = "activation_commit"
    ABORT = "abort"


class CheckpointProbeLimits(StrictBaseModel):
    """Resolved native limits used to reject malformed probes before restore."""

    vocab_size: int = Field(gt=0, description="Unpadded model vocabulary size.")
    max_seq_len: int = Field(gt=0, description="Maximum prompt plus completion length.")
    max_input_len: int = Field(gt=0, description="Maximum native prompt length.")
    max_num_tokens: int = Field(gt=0, description="Maximum scheduled tokens per batch.")
    chunked_prefill: bool = Field(description="Whether a prompt can span prefill iterations.")


class CheckpointProfile(StrictBaseModel):
    """Resolved profile identity, produced by the native worker group."""

    profile_digest: str = Field(description="Hash of resolved engine configuration.")
    cohort_digest: str = Field(description="Hash of the fixed native participant identities.")
    ranks: list[int] = Field(description="All required MPI ranks.")
    probe_limits: CheckpointProbeLimits | None = Field(
        default=None, description="Native validation limits; required by the serving backend."
    )
    capture_policies: list[Literal["resident"]] = Field(
        default_factory=lambda: ["resident"], description="Implemented capture policies."
    )
    drain_policies: list[Literal["natural"]] = Field(
        default_factory=lambda: ["natural"], description="Implemented request retirement policies."
    )


class RankCheckpointEvidence(StrictBaseModel):
    """Checks made on one participant at the protected scheduler boundary."""

    rank: int = Field(description="Participant that performed the checks.")
    session_id: str = Field(description="Session protected by this evidence.")
    profile_digest: str = Field(description="Validated engine profile identity.")
    cohort_digest: str = Field(description="Validated participant-set identity.")
    checks: list[str] = Field(description="Native checks completed at the protected boundary.")
    state_digest: str = Field(description="Hash of the retained resident inventory.")


class CheckpointReadiness(StrictBaseModel):
    """Independent readiness facts; only activation opens public admission."""

    capture_safe: bool = Field(
        default=False, description="Source is drained and protected for capture."
    )
    memory_ready: bool = Field(
        default=False, description="Resident storage matches the captured inventory."
    )
    runtime_ready: bool = Field(
        default=False, description="Private generation and native validation completed."
    )
    validation_passed: bool = Field(
        default=False, description="The probe matched its reference and retired."
    )
    serving_ready: bool = Field(
        default=False, description="Explicit all-rank activation completed."
    )
    admission_open: bool = Field(
        default=False, description="Public requests may enter the serving path."
    )


class CheckpointResult(StrictBaseModel):
    """Recorded result for one exactly identified operation."""

    operation_id: str = Field(description="Exact mutation and retry identity.")
    session_id: str = Field(description="Source or destination session.")
    template_id: str = Field(description="Immutable artifact identity.")
    profile_digest: str = Field(description="Engine configuration identity.")
    cohort_digest: str = Field(description="Required participant identity.")
    status: Literal["in_progress", "completed", "failed"] = Field(
        description="Recorded operation outcome."
    )
    state_version: int = Field(description="Lifecycle version observed at this result.")
    readiness_id: str | None = Field(
        default=None, description="Current engine-issued validation reference."
    )
    rank_results: list[RankCheckpointEvidence] = Field(
        default_factory=list, description="Collected participant evidence."
    )
    error: str | None = Field(default=None, description="Failure reason, if the operation failed.")
    native_prepare_not_dispatched: bool = Field(
        default=False,
        description="This failed prepare cannot dispatch native work; not permission to resume.",
    )
    cleanup_status: Literal["not_requested", "host_confirmation_required"] = Field(
        default="not_requested",
        description="Whether external host cleanup evidence is still required.",
    )


class CheckpointStatus(StrictBaseModel):
    """Read-only status; positive flags are not an admission capability."""

    source_session_id: str = Field(description="Identity created before source capture.")
    restore_session_id: str | None = Field(
        description="Fresh external destination identity, if bound."
    )
    profile: CheckpointProfile = Field(description="Resolved engine and participant profile.")
    stage: CheckpointStage = Field(description="Current lifecycle stage.")
    state_version: int = Field(description="Version required for the next mutation.")
    remaining_seconds: float | None = Field(
        default=None, description="Current local attempt budget; not a durable host deadline."
    )
    readiness: CheckpointReadiness = Field(description="Independent, current readiness facts.")
    operation: CheckpointResult | None = Field(
        default=None, description="Most recent operation result."
    )


class CheckpointBackend(Protocol):
    """Native engine bridge shared by standalone and external controllers."""

    def describe(self) -> CheckpointProfile:
        """Read the resolved fixed-cohort profile."""
        ...

    def step(
        self,
        step: CheckpointStep,
        context: CheckpointContext,
        remaining_seconds: float,
        validation_token: str | None = None,
        source_session_id: str | None = None,
    ) -> list[RankCheckpointEvidence]:
        """Run a bounded all-rank step; reject incomplete participant evidence."""
        ...

    def generate(
        self, validation: GenerationValidation, token: str, remaining_seconds: float
    ) -> list[int]:
        """Submit one real private request through the normal generation path."""
        ...

    def validate_probe(self, validation: GenerationValidation) -> None:
        """Reject invalid token IDs and lengths before consuming a restore attempt."""
        ...

    def fail_closed(self, remaining_seconds: float = 0) -> None:
        """Revoke admission without claiming native or host resource cleanup."""
        ...


class CheckpointPrepareNotDispatchedError(TimeoutError):
    """Preparation ended before dispatch, with no native PREPARE able to run later.

    Backends must not use this error for an RPC timeout or uncertain dispatch.
    It does not authorize reopening admission or override an explicit abort.
    """


def checkpoint_digest(value: object) -> str:
    """Hash JSON-compatible contract data without implicit string coercion.

    Args:
        value: A JSON-compatible configuration or observation.

    Returns:
        SHA-256 of canonical JSON.
    """
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def validate_checkpoint_evidence(
    step: CheckpointStep,
    context: CheckpointContext,
    results: list[RankCheckpointEvidence],
    ranks: list[int],
) -> None:
    """Require exactly one current, complete acknowledgement per expected rank.

    The native coordinator calls this before committing the leader. The frontend
    also checks evidence after transport, including terminal abort results.

    Args:
        step: Native step whose checks must have completed.
        context: Current operation identity.
        results: Participant evidence received from the engine.
        ranks: Exact participant set required at this boundary.

    Raises:
        RuntimeError: If participants, identities or required checks disagree.
    """
    required = {
        CheckpointStep.PREPARE: {"admission_closed", "scheduler_drained", "no_external_exposure"},
        CheckpointStep.RESTORE: {
            "fresh_restore_session",
            "memory_present",
            "resident_state_matches",
        },
        CheckpointStep.VALIDATION_START: {"private_validation_only", "public_admission_closed"},
        CheckpointStep.VALIDATION_END: {
            "validation_retired",
            "scheduler_drained",
            "warmed_graph_replayed",
            "resident_state_matches",
        },
        CheckpointStep.ACTIVATION_PREPARE: {
            "activation_prepared",
            "runtime_valid",
            "public_admission_closed",
        },
        CheckpointStep.ACTIVATION_COMMIT: {"activation_committed"},
        CheckpointStep.ABORT: {
            "progression_stopped",
            "public_admission_closed",
            "host_cleanup_required",
        },
    }[step]
    if sorted(result.rank for result in results) != sorted(ranks):
        raise RuntimeError("Missing, duplicate, or unexpected checkpoint rank acknowledgement.")
    if any(
        result.session_id != context.session_id
        or result.profile_digest != context.profile_digest
        or result.cohort_digest != context.cohort_digest
        or not required.issubset(result.checks)
        or not result.state_digest
        for result in results
    ):
        raise RuntimeError("Stale or incomplete checkpoint evidence.")


class CheckpointLifecycle:
    """Serialize lifecycle mutations and retain retry identity across capture.

    The backend owns scheduler and resource protection. A timeout never grants
    permission to resume or reuse resources. A fresh restore session has a fresh
    relative deadline; later commands cannot extend that attempt's budget.
    """

    def __init__(self, backend: CheckpointBackend) -> None:
        """Initialize a source session from the resolved worker profile.

        Args:
            backend: Engine bridge that supplies evidence and private generation.
        """
        self._backend = backend
        self._profile = backend.describe()
        if not self._profile.ranks or sorted(self._profile.ranks) != list(
            range(len(self._profile.ranks))
        ):
            raise ValueError("Checkpoint requires one contiguous fixed participant set.")
        self._source_session_id = secrets.token_hex(16)
        self._restore_session_id: str | None = None
        self._template_id: str | None = None
        self._stage = CheckpointStage.RUNNING
        self._state_version = 0
        self._readiness = CheckpointReadiness(admission_open=True)
        self._readiness_id: str | None = None
        self._deadline: float | None = None
        self._lock = threading.Lock()
        self._operations: dict[str, tuple[str, CheckpointResult]] = {}
        # An abort can be the latest operation while an earlier mutation still runs.
        self._active_operation: str | None = None
        self._last_operation: str | None = None
        self._abort_operation_id: str | None = None
        self._cancelled = threading.Event()
        self._idle = threading.Event()
        self._idle.set()

    def _readiness_expired(self) -> bool:
        """Check cancellation and the attempt deadline while holding the state lock."""
        return self._cancelled.is_set() or (
            self._stage != CheckpointStage.SERVING
            and self._deadline is not None
            and time.monotonic() >= self._deadline
        )

    def admission_open(self) -> bool:
        """Read the admission latch without copying profile or operation evidence."""
        with self._lock:
            return self._readiness.admission_open and not self._readiness_expired()

    def get_lifecycle_status(self) -> CheckpointStatus:
        """Read a detached status snapshot without changing engine state.

        Returns:
            Current readiness and an immutable copy of the operation result.
        """
        with self._lock:
            selected = self._last_operation
            readiness = self._readiness.model_copy(deep=True)
            if self._readiness_expired():
                readiness = CheckpointReadiness()
            operation = self._operations[selected][1] if selected is not None else None
            return CheckpointStatus(
                source_session_id=self._source_session_id,
                restore_session_id=self._restore_session_id,
                profile=self._profile.model_copy(deep=True),
                stage=self._stage,
                state_version=self._state_version,
                remaining_seconds=(
                    max(0, self._deadline - time.monotonic())
                    if self._deadline is not None and self._stage != CheckpointStage.SERVING
                    else None
                ),
                readiness=readiness,
                operation=operation.model_copy(deep=True) if operation is not None else None,
            )

    def _begin(
        self,
        name: str,
        request: StrictBaseModel,
        context: CheckpointContext,
        expected: tuple[CheckpointStage, ...],
        stage: CheckpointStage,
    ) -> tuple[CheckpointResult, bool]:
        """Record one transition before invoking its side effects.

        Args:
            name: Public operation name used in the retry fingerprint.
            request: Complete validated payload.
            context: Operation identity and relative budget.
            expected: Legal predecessor stages.
            stage: In-progress stage to publish before native work.

        Returns:
            Recorded result and whether this caller must execute the operation.

        Raises:
            ValueError: If identity, retry arguments, version or transition order disagree.
            RuntimeError: If the attempt is terminal or the operation history is full.
            TimeoutError: If the attempt budget expired.
        """
        fingerprint = checkpoint_digest([name, request.model_dump(mode="json")])
        with self._lock:
            previous = self._retry_result(context.operation_id, fingerprint)
            if previous is not None:
                return previous, False
            if self._stage in (CheckpointStage.FAILED, CheckpointStage.ABORTED):
                raise RuntimeError("Checkpoint attempt is terminal; progression is not permitted.")
            if len(self._operations) >= _MAX_OPERATION_HISTORY:
                raise RuntimeError("Checkpoint operation history is full; use a fresh worker.")
            if self._active_operation is not None or self._stage not in expected:
                raise ValueError("Conflicting checkpoint lifecycle transition.")
            if context.expected_state_version != self._state_version:
                raise ValueError("Stale checkpoint state version.")
            if (
                context.profile_digest != self._profile.profile_digest
                or context.cohort_digest != self._profile.cohort_digest
            ):
                raise ValueError("Checkpoint profile or cohort does not match the worker.")
            if self._template_id not in (None, context.template_id):
                raise ValueError("Checkpoint template identity changed.")
            if name == "prepare_capture":
                if (
                    context.session_kind != "source"
                    or context.session_id != self._source_session_id
                ):
                    raise ValueError("Capture requires the current source session.")
            elif name == "validate_runtime":
                if (
                    context.session_kind != "restore"
                    or context.session_id == self._source_session_id
                ):
                    raise ValueError("Runtime validation requires a fresh restore session.")
                if self._restore_session_id is not None:
                    raise ValueError("This destination already has a restore attempt.")
            else:
                session = self._restore_session_id or self._source_session_id
                kind = "restore" if self._restore_session_id is not None else "source"
                if context.session_id != session or context.session_kind != kind:
                    raise ValueError("Stale checkpoint session.")
            now = time.monotonic()
            deadline = now + context.time_budget_seconds
            if name not in ("prepare_capture", "validate_runtime") and self._deadline is not None:
                deadline = min(deadline, self._deadline)
            if deadline <= now:
                raise TimeoutError("Checkpoint attempt budget expired.")
            if name == "validate_runtime":
                self._restore_session_id = context.session_id
            self._deadline = deadline
            self._template_id = context.template_id
            self._stage = stage
            self._state_version += 1
            self._readiness.capture_safe = False
            self._readiness.admission_open = False
            self._active_operation = context.operation_id
            self._idle.clear()
            self._last_operation = context.operation_id
            result = CheckpointResult(
                operation_id=context.operation_id,
                session_id=context.session_id,
                template_id=context.template_id,
                profile_digest=context.profile_digest,
                cohort_digest=context.cohort_digest,
                status="in_progress",
                state_version=self._state_version,
            )
            self._operations[context.operation_id] = (fingerprint, result)
            return result.model_copy(deep=True), True

    def _retry_result(self, operation_id: str, fingerprint: str) -> CheckpointResult | None:
        """Read an exact retry while the caller holds the lifecycle lock."""
        previous = self._operations.get(operation_id)
        if previous is None:
            return None
        if previous[0] != fingerprint:
            raise ValueError("Operation ID was reused with different arguments.")
        return previous[1].model_copy(deep=True)

    def _remaining(self) -> float:
        """Read the original attempt budget; no retry extends it.

        Returns:
            Positive remaining seconds.

        Raises:
            RuntimeError: If cancelled or no operation owns a deadline.
            TimeoutError: If the deadline expired.
        """
        if self._cancelled.is_set():
            raise RuntimeError("Checkpoint operation was aborted.")
        if self._deadline is None:
            raise RuntimeError("Checkpoint operation has no deadline.")
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Checkpoint operation budget expired.")
        return remaining

    def _step(
        self,
        step: CheckpointStep,
        context: CheckpointContext,
        token: str | None = None,
        source_session_id: str | None = None,
    ) -> list[RankCheckpointEvidence]:
        """Require exactly one identity-matched result from every fixed rank.

        Args:
            step: Native transition to execute.
            context: Current operation identity.
            token: Private validation permit, if needed by this transition.
            source_session_id: Captured source identity for restore binding.

        Returns:
            Complete, current participant evidence.

        Raises:
            RuntimeError: If native checks or participant identities are incomplete.
            TimeoutError: If the shared attempt budget expires.
        """
        try:
            remaining = self._remaining()
        except TimeoutError as error:
            if step == CheckpointStep.PREPARE:
                raise CheckpointPrepareNotDispatchedError(str(error)) from error
            raise
        results = self._backend.step(step, context, remaining, token, source_session_id)
        self._remaining()
        validate_checkpoint_evidence(step, context, results, self._profile.ranks)
        return results

    def _finish(
        self,
        context: CheckpointContext,
        stage: CheckpointStage,
        ranks: list[RankCheckpointEvidence],
        error: Exception | None = None,
        *,
        native_prepare_not_dispatched: bool = False,
    ) -> CheckpointResult:
        """Publish a terminal operation result without reopening admission on error.

        Args:
            context: Operation being completed.
            stage: Desired terminal stage, subject to cancellation checks.
            ranks: Validated participant observations.
            error: Native or validation failure, if any.
            native_prepare_not_dispatched: Proven pre-dispatch prepare failure,
                not evidence about a concurrent abort or shutdown.

        Returns:
            Detached terminal result recorded in the retry journal.
        """
        if error is not None and not native_prepare_not_dispatched:
            try:
                self._backend.fail_closed()
            except Exception as cleanup_error:
                # Cleanup is best effort; every operation must still get a terminal journal entry.
                logger.warning(f"Checkpoint failure cleanup could not complete: {cleanup_error!r}")
        with self._lock:
            if error is not None or self._cancelled.is_set():
                error = error or RuntimeError("Checkpoint operation was aborted.")
                stage = CheckpointStage.FAILED
                self._readiness = CheckpointReadiness()
                self._readiness_id = None
            elif stage == CheckpointStage.CAPTURE_SAFE:
                self._readiness.capture_safe = True
            elif stage == CheckpointStage.RUNTIME_READY:
                self._readiness.runtime_ready = True
                self._readiness.validation_passed = True
                self._readiness_id = secrets.token_hex(32)
            elif stage == CheckpointStage.SERVING:
                self._readiness.serving_ready = True
                self._readiness.admission_open = True
            fingerprint, result = self._operations[context.operation_id]
            if self._abort_operation_id is None:
                self._stage = stage
            self._state_version += 1
            result = result.model_copy(
                update={
                    "status": "failed" if error is not None else "completed",
                    "state_version": self._state_version,
                    "rank_results": ranks,
                    "error": (str(error) or type(error).__name__) if error is not None else None,
                    "native_prepare_not_dispatched": native_prepare_not_dispatched,
                    "readiness_id": self._readiness_id,
                    "cleanup_status": "host_confirmation_required"
                    if stage in (CheckpointStage.FAILED, CheckpointStage.ABORTED)
                    and (not native_prepare_not_dispatched or self._cancelled.is_set())
                    else "not_requested",
                },
                deep=True,
            )
            self._operations[context.operation_id] = (fingerprint, result)
            self._active_operation = None
            self._idle.set()
            return result.model_copy(deep=True)

    def prepare_capture(self, request: PrepareCaptureRequest) -> CheckpointResult:
        """Naturally drain and latch safety on every rank without releasing KV.

        Args:
            request: Source-session operation with the expected profile and cohort.

        Returns:
            Capture evidence, an existing retry result, or a failed operation.
        """
        context = request.context
        result, execute = self._begin(
            "prepare_capture",
            request,
            context,
            (CheckpointStage.RUNNING,),
            CheckpointStage.PREPARING,
        )
        if not execute:
            return result
        try:
            ranks = self._step(CheckpointStep.PREPARE, context)
            return self._finish(context, CheckpointStage.CAPTURE_SAFE, ranks)
        except CheckpointPrepareNotDispatchedError as error:
            return self._finish(
                context, CheckpointStage.FAILED, [], error, native_prepare_not_dispatched=True
            )
        except Exception as error:
            return self._finish(context, CheckpointStage.FAILED, [], error)

    def validate_runtime(self, request: ValidateRuntimeRequest) -> CheckpointResult:
        """Validate restored resources, run private generation, and retire its work.

        Args:
            request: Fresh destination identity and deterministic validation policy.

        Returns:
            Engine-generated readiness evidence without public admission.
        """
        context = request.context
        if request.source_session_id != self._source_session_id:
            raise ValueError("Restore notification does not identify the captured source.")
        self._backend.validate_probe(request.validation)
        result, execute = self._begin(
            "validate_runtime",
            request,
            context,
            (CheckpointStage.CAPTURE_SAFE,),
            CheckpointStage.VALIDATING,
        )
        if not execute:
            return result
        try:
            self._step(CheckpointStep.RESTORE, context, source_session_id=request.source_session_id)
            with self._lock:
                self._readiness.memory_ready = True
            token = secrets.token_hex(32)
            self._step(CheckpointStep.VALIDATION_START, context, token)
            tokens = self._backend.generate(request.validation, token, self._remaining())
            ranks = self._step(CheckpointStep.VALIDATION_END, context)
            if tokens != request.validation.expected_token_ids:
                raise RuntimeError("Private generation differs from the reference tokens.")
            return self._finish(context, CheckpointStage.RUNTIME_READY, ranks)
        except Exception as error:
            return self._finish(context, CheckpointStage.FAILED, [], error)

    def activate(self, request: ActivateRequest) -> CheckpointResult:
        """Open admission only after current evidence and host success agree.

        Args:
            request: Current readiness reference and authenticated host attestation.

        Returns:
            All-rank activation result, never inferred from memory status.
        """
        context = request.context
        host = request.host_restore
        with self._lock:
            previous = self._retry_result(
                context.operation_id,
                checkpoint_digest(["activate", request.model_dump(mode="json")]),
            )
            if previous is not None:
                return previous
            if self._stage in (CheckpointStage.FAILED, CheckpointStage.ABORTED):
                raise RuntimeError("Checkpoint attempt is terminal; progression is not permitted.")
            if request.readiness_id != self._readiness_id:
                raise ValueError("Unknown or stale engine readiness reference.")
        if (
            host.session_id != context.session_id
            or host.template_id != context.template_id
            or host.profile_digest != context.profile_digest
            or host.cohort_digest != context.cohort_digest
        ):
            raise ValueError("Host restore success belongs to another destination or artifact.")
        result, execute = self._begin(
            "activate",
            request,
            context,
            (CheckpointStage.RUNTIME_READY,),
            CheckpointStage.ACTIVATING,
        )
        if not execute:
            return result
        try:
            self._step(CheckpointStep.ACTIVATION_PREPARE, context)
            ranks = self._step(CheckpointStep.ACTIVATION_COMMIT, context)
            return self._finish(context, CheckpointStage.SERVING, ranks)
        except Exception as error:
            return self._finish(context, CheckpointStage.FAILED, [], error)

    def abort_operation(self, request: AbortCheckpointRequest) -> CheckpointResult:
        """Stop pre-activation progress; external cleanup still needs host evidence.

        Args:
            request: Current session identity and abort reason.

        Returns:
            An abort result that does not claim host cleanup or resource reuse.
        """
        context = request.context
        fingerprint = checkpoint_digest(["abort_operation", request.model_dump(mode="json")])
        deadline = time.monotonic() + context.time_budget_seconds
        with self._lock:
            previous = self._retry_result(context.operation_id, fingerprint)
            if previous is not None:
                return previous
            if self._stage in (
                CheckpointStage.RUNNING,
                CheckpointStage.SERVING,
                CheckpointStage.ABORTING,
                CheckpointStage.ABORTED,
            ):
                raise ValueError("No abortable checkpoint operation.")
            if self._abort_operation_id is not None:
                raise ValueError("This checkpoint attempt already has an abort operation.")
            session = self._restore_session_id or self._source_session_id
            kind = "restore" if self._restore_session_id is not None else "source"
            if (
                context.session_id != session
                or context.session_kind != kind
                or context.template_id != self._template_id
                or context.profile_digest != self._profile.profile_digest
                or context.cohort_digest != self._profile.cohort_digest
                or context.expected_state_version != self._state_version
            ):
                raise ValueError("Abort does not identify the current checkpoint attempt.")
            if len(self._operations) >= _MAX_OPERATION_HISTORY:
                raise RuntimeError("Checkpoint operation history is full.")
            self._cancelled.set()
            self._readiness = CheckpointReadiness()
            self._readiness_id = None
            self._state_version += 1
            self._stage = CheckpointStage.ABORTING
            self._abort_operation_id = context.operation_id
            result = CheckpointResult(
                operation_id=context.operation_id,
                session_id=context.session_id,
                template_id=context.template_id,
                profile_digest=context.profile_digest,
                cohort_digest=context.cohort_digest,
                status="in_progress",
                state_version=self._state_version,
                cleanup_status="host_confirmation_required",
            )
            self._operations[context.operation_id] = (fingerprint, result)
            self._last_operation = context.operation_id
        error = None
        ranks = []
        try:
            self._backend.fail_closed(max(0, deadline - time.monotonic()))
            if not self._idle.wait(timeout=max(0, deadline - time.monotonic())):
                raise TimeoutError("Abort stopped admission; native work has not retired.")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Abort budget expired; host cleanup is required.")
            ranks = self._backend.step(CheckpointStep.ABORT, context, remaining)
            if time.monotonic() >= deadline:
                raise TimeoutError("Abort budget expired; host cleanup is required.")
            validate_checkpoint_evidence(CheckpointStep.ABORT, context, ranks, self._profile.ranks)
        except Exception as exc:
            error = str(exc) or type(exc).__name__
        with self._lock:
            self._stage = CheckpointStage.FAILED if error is not None else CheckpointStage.ABORTED
            self._state_version += 1
            result = result.model_copy(
                update={
                    "status": "failed" if error is not None else "completed",
                    "state_version": self._state_version,
                    "rank_results": ranks,
                    "error": error,
                },
                deep=True,
            )
            self._operations[context.operation_id] = (fingerprint, result)
            return result.model_copy(deep=True)
