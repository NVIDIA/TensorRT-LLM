# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in lifecycle diagnostics for disaggregated inference.

The event stream is deliberately observational and fail-open.  It does not
coordinate request state across context and generation workers, and a timer
event does not imply that transport work is quiescent or that resources are
safe to reuse.

Each event contains both a process-local monotonic timestamp and a Unix wall
timestamp.  Durations are valid only between events with the same ``clock_id``;
wall timestamps are for best-effort cross-host correlation.

``transceiver_handoff`` is captured immediately before the adapter call and
does not claim that transport was submitted or made progress.
``transfer_adapter_complete`` means only that the executor observed an adapter
terminal result.  The adapter may already have performed topology consensus;
the event is not a backend-local physical-completion timestamp, a cross-side
commit, or a resource-reuse fence.

Consumers must order concurrently emitted lines by ``(clock_id, seq)`` rather
than file order.  This permits callers to capture an exact boundary under
their own lock and publish the record after releasing that lock.
"""

from __future__ import annotations

import os
import queue
import secrets
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from tensorrt_llm.logger import logger

DISAGG_DIAGNOSTICS_ENV = "TRTLLM_DISAGG_TRANSFER_DIAGNOSTICS"
DISAGG_DIAGNOSTICS_RANKS_ENV = "TRTLLM_DISAGG_TRANSFER_DIAGNOSTICS_RANKS"
DISAGG_DIAGNOSTICS_QUEUE_SIZE_ENV = "TRTLLM_DISAGG_TRANSFER_DIAGNOSTICS_QUEUE_SIZE"
_LOG_PREFIX = "[DISAGG_DIAG][lifecycle]"
_SCHEMA_VERSION = 1
_DEFAULT_QUEUE_SIZE = 8192
_DEFAULT_CLOSE_TIMEOUT_SECONDS = 1.0
_WRITER_STOP = object()


class DisaggLifecycleEvent(str, Enum):
    CTX_ARRIVED = "ctx_arrived"
    GEN_ARRIVED = "gen_arrived"
    CTX_DEQUEUED = "ctx_dequeued"
    GEN_DEQUEUED = "gen_dequeued"
    CTX_DISPATCHED = "ctx_dispatched"
    GEN_DISPATCHED = "gen_dispatched"
    CTX_WAITING_RELEASED = "ctx_waiting_released"
    GEN_WAITING_RELEASED = "gen_waiting_released"
    CTX_LOCAL_SCHEDULER_ACTIVATED = "ctx_local_scheduler_activated"
    GEN_LOCAL_SCHEDULER_ACTIVATED = "gen_local_scheduler_activated"
    GEN_ADMISSION_CHANGED = "gen_admission_changed"
    CTX_ARTIFACT_READY = "ctx_artifact_ready"
    GEN_RESOURCES_READY = "gen_resources_ready"
    TRANSCEIVER_HANDOFF = "transceiver_handoff"
    TRANSCEIVER_CALL_RETURNED = "transceiver_call_returned"
    TRANSFER_ADAPTER_COMPLETE = "transfer_adapter_complete"
    TRANSFER_ADAPTER_ERROR = "transfer_adapter_error"
    ADAPTER_STATUS_POLL_ERROR = "adapter_status_poll_error"
    HANDOFF_DEADLINE_CROSSED = "handoff_deadline_crossed"
    TIMEOUT_ARMED = "timeout_armed"
    TIMEOUT_OBSERVED = "timeout_observed"
    GEN_READY = "gen_ready"
    RECORDS_DROPPED = "records_dropped"


class DisaggLifecycleEmitterName(str, Enum):
    DIAGNOSTICS_WRITER = "diagnostics_writer"
    EXECUTOR_QUEUE = "executor_queue"
    PYEXECUTOR = "pyexecutor"


class DisaggLifecycleRole(str, Enum):
    CTX = "ctx"
    GEN = "gen"
    PROCESS = "process"


class DisaggLifecycleScope(str, Enum):
    LOCAL_OBSERVER = "local_observer"
    GROUP = "group"


class DisaggLifecycleGate(str, Enum):
    SCHEDULER = "scheduler"
    TRANSFER = "transfer"


class DisaggLifecycleDecision(str, Enum):
    ELIGIBLE = "eligible"
    ADMIT = "admit"
    BYPASS = "bypass"
    DEFER = "defer"


class DisaggLifecycleOutcome(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"


class DisaggLifecycleReason(str, Enum):
    ADAPTER_EXCEPTION = "adapter_exception"
    ADMISSION_DISABLED = "admission_disabled"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    GEN_ONLY_NO_CONTEXT = "gen_only_no_context"
    TRANSFER_BUDGET = "transfer_budget"


class DisaggLifecycleOperation(str, Enum):
    ASYNC_RECEIVE = "async_receive"
    ASYNC_SEND = "async_send"
    SYNC_RECEIVE = "sync_receive"


class DisaggLifecycleTimerBasis(str, Enum):
    EXECUTOR_WATCHDOG = "executor_watchdog"
    TRANSCEIVER_HANDOFF = "transceiver_handoff"


class DisaggLifecycleScheduleStyle(str, Enum):
    CONTEXT_FIRST = "context_first"
    GENERATION_FIRST = "generation_first"


@dataclass(frozen=True, slots=True)
class DisaggCorrelation:
    """The three request-ID domains used by the lifecycle event stream."""

    disagg_request_id: Optional[int]
    ctx_request_id: Optional[int]
    local_request_id: Optional[int]


@dataclass(frozen=True, slots=True)
class DisaggLifecycleStamp:
    """A minimal timestamp captured at an exact caller-owned boundary."""

    sequence: int
    monotonic_ns: int
    wall_ns: int


@dataclass(slots=True)
class DisaggRequestLifecycleState:
    """Per-request diagnostic state with the same lifetime as a request."""

    scheduler_eligible: bool = False
    admission_decision: Optional[DisaggLifecycleDecision] = None
    transfer_handoff_time: Optional[float] = None
    handoff_deadline_crossed: bool = False
    watchdog_timeout_observed: bool = False


def _optional_int(value) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _enum_name(value) -> Optional[str]:
    if value is None:
        return None
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name.lower()
    return str(value).rsplit(".", maxsplit=1)[-1].lower()


def get_disagg_role(request) -> Optional[DisaggLifecycleRole]:
    """Return the disaggregated worker role represented by ``request``."""

    request_type = getattr(request, "py_llm_request_type", None)
    if request_type is None:
        request_type = getattr(request, "request_type", None)
    request_type_name = _enum_name(request_type)
    if request_type_name is None:
        return None
    if "context_only" in request_type_name:
        return DisaggLifecycleRole.CTX
    if "generation_only" in request_type_name:
        return DisaggLifecycleRole.GEN
    return None


def get_disagg_schedule_style(
    request,
) -> Optional[DisaggLifecycleScheduleStyle]:
    params = getattr(request, "py_disaggregated_params", None)
    if params is None:
        params = getattr(request, "disaggregated_params", None)
    schedule_style = getattr(params, "schedule_style", None) if params is not None else None
    if schedule_style is None:
        schedule_style = getattr(request, "schedule_style", None)
    schedule_style_name = _enum_name(schedule_style)
    if schedule_style_name is None:
        return None
    if "generation_first" in schedule_style_name:
        return DisaggLifecycleScheduleStyle.GENERATION_FIRST
    if "context_first" in schedule_style_name:
        return DisaggLifecycleScheduleStyle.CONTEXT_FIRST
    return None


def get_disagg_correlation(request, local_request_id: Optional[int] = None) -> DisaggCorrelation:
    """Extract IDs without synthesizing one ID domain from another."""

    params = getattr(request, "py_disaggregated_params", None)
    if params is None:
        params = getattr(request, "disaggregated_params", None)

    disagg_request_id = getattr(request, "disagg_request_id", None)
    if disagg_request_id is None and params is not None:
        disagg_request_id = getattr(params, "disagg_request_id", None)

    ctx_request_id = getattr(request, "ctx_request_id", None)
    if ctx_request_id is None and params is not None:
        ctx_request_id = getattr(params, "ctx_request_id", None)
    if local_request_id is None:
        local_request_id = getattr(request, "py_request_id", None)
    if local_request_id is None:
        local_request_id = getattr(request, "request_id", None)

    return DisaggCorrelation(
        disagg_request_id=_optional_int(disagg_request_id),
        ctx_request_id=_optional_int(ctx_request_id),
        local_request_id=_optional_int(local_request_id),
    )


def _token(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def _write_lifecycle_record(record: str) -> None:
    """Write independently of the general TRT-LLM log-level threshold."""

    print(record, flush=True)


class DisaggLifecycleEmitter:
    """Emit a deterministic, default-off per-process lifecycle stream."""

    def __init__(
        self,
        *,
        enabled: bool,
        rank: int,
        runtime: str,
        backend: str,
        clock_id: Optional[str] = None,
        rank_scope: str = "world",
        async_output: bool = True,
        queue_capacity: Optional[int] = None,
    ):
        self.enabled = bool(enabled)
        self._rank = int(rank)
        self._runtime = str(runtime).lower()
        self._backend = str(backend).lower()
        self._clock_id = (
            clock_id
            if clock_id is not None
            else (f"py-{os.getpid()}-{secrets.token_hex(8)}" if enabled else "disabled")
        )
        self._rank_scope = rank_scope
        self._sequence = 0
        self._sequence_lock = threading.Lock()
        self._publish_lock = threading.Lock()
        self._accept_lock = threading.Lock()
        self._drop_lock = threading.Lock()
        self._failure_lock = threading.Lock()
        self._dropped_records_total = 0
        self._pending_dropped_records = 0
        self._failure_pending = False
        self._failure_reported = False
        self._closed = False
        self._stop_requested = threading.Event()
        self._stop_signal_enqueued = False
        self._async_output = self.enabled and async_output
        self._output_queue = None
        self._writer_thread = None
        if self._async_output:
            capacity = queue_capacity or self._queue_capacity_from_environment()
            self._output_queue = queue.Queue(maxsize=max(1, int(capacity)))
            self._writer_thread = threading.Thread(
                target=self._writer_loop,
                name=f"disagg-lifecycle-writer-rank-{self._rank}",
                daemon=True,
            )
            self._writer_thread.start()

    @staticmethod
    def _queue_capacity_from_environment() -> int:
        configured_capacity = os.getenv(DISAGG_DIAGNOSTICS_QUEUE_SIZE_ENV)
        if configured_capacity is None:
            return _DEFAULT_QUEUE_SIZE
        try:
            capacity = int(configured_capacity)
            return capacity if capacity > 0 else _DEFAULT_QUEUE_SIZE
        except ValueError:
            return _DEFAULT_QUEUE_SIZE

    @staticmethod
    def _rank_is_enabled(rank: int) -> bool:
        configured_ranks = os.getenv(DISAGG_DIAGNOSTICS_RANKS_ENV)
        if configured_ranks is None or configured_ranks.strip() in ("", "*"):
            return True
        try:
            return int(rank) in {
                int(value.strip()) for value in configured_ranks.split(",") if value.strip()
            }
        except ValueError:
            return False

    @classmethod
    def from_environment(
        cls,
        *,
        rank: int,
        runtime: Optional[str],
        backend: Optional[str],
    ) -> "DisaggLifecycleEmitter":
        enabled = os.getenv(DISAGG_DIAGNOSTICS_ENV) == "1"
        return cls(
            enabled=enabled and cls._rank_is_enabled(rank),
            rank=rank,
            runtime=runtime or "CPP",
            backend=backend or "UNKNOWN",
        )

    def _mark_failure(self) -> None:
        with self._failure_lock:
            self._failure_pending = True

    def _report_failure_once(self) -> None:
        with self._failure_lock:
            if not self._failure_pending or self._failure_reported:
                return
            self._failure_reported = True
        try:
            logger.warning(
                "Disaggregated lifecycle diagnostics dropped an event; "
                "request execution is continuing."
            )
        except Exception:
            pass

    def record_failure(self) -> None:
        """Report an instrumentation failure without propagating it."""

        self._mark_failure()
        self._report_failure_once()

    def capture_stamp(self) -> Optional[DisaggLifecycleStamp]:
        """Capture only clocks and sequence for a latency-sensitive boundary."""

        if not self.enabled:
            return None
        try:
            with self._sequence_lock:
                self._sequence += 1
                return DisaggLifecycleStamp(
                    sequence=self._sequence,
                    monotonic_ns=time.monotonic_ns(),
                    wall_ns=time.time_ns(),
                )
        except Exception:
            self._mark_failure()
            return None

    def capture(
        self,
        event: DisaggLifecycleEvent,
        *,
        emitter: DisaggLifecycleEmitterName,
        role: DisaggLifecycleRole,
        correlation: DisaggCorrelation,
        scope: DisaggLifecycleScope = DisaggLifecycleScope.LOCAL_OBSERVER,
        outcome: Optional[DisaggLifecycleOutcome] = None,
        reason: Optional[DisaggLifecycleReason] = None,
        gate: Optional[DisaggLifecycleGate] = None,
        decision: Optional[DisaggLifecycleDecision] = None,
        operation: Optional[DisaggLifecycleOperation] = None,
        schedule_style: Optional[DisaggLifecycleScheduleStyle] = None,
        blocks: Optional[int] = None,
        budget_blocks: Optional[int] = None,
        active_blocks: Optional[int] = None,
        bytes_: Optional[int] = None,
        timeout_ms: Optional[int] = None,
        elapsed_ms: Optional[float] = None,
        timer_basis: Optional[DisaggLifecycleTimerBasis] = None,
        request_count: Optional[int] = None,
        dropped_records: Optional[int] = None,
        local_state: Optional[str] = None,
        state_domain: Optional[str] = None,
        stamp: Optional[DisaggLifecycleStamp] = None,
    ) -> Optional[str]:
        if not self.enabled:
            return None

        try:
            stamp = stamp or self.capture_stamp()
            if stamp is None:
                return None
            fields = (
                ("schema", _SCHEMA_VERSION),
                ("event", event),
                ("mono_ns", stamp.monotonic_ns),
                ("wall_ns", stamp.wall_ns),
                ("clock_id", self._clock_id),
                ("seq", stamp.sequence),
                ("runtime", self._runtime),
                ("backend", self._backend),
                ("emitter", emitter),
                ("role", role),
                ("rank", self._rank),
                ("rank_scope", self._rank_scope),
                ("disagg_request_id", correlation.disagg_request_id),
                ("ctx_request_id", correlation.ctx_request_id),
                ("local_request_id", correlation.local_request_id),
                ("scope", scope),
                ("outcome", outcome),
                ("reason", reason),
                ("gate", gate),
                ("decision", decision),
                ("operation", operation),
                ("schedule_style", schedule_style),
                ("blocks", blocks),
                ("budget_blocks", budget_blocks),
                ("active_blocks", active_blocks),
                ("bytes", bytes_),
                ("timeout_ms", timeout_ms),
                (
                    "elapsed_ms",
                    f"{elapsed_ms:.6f}" if elapsed_ms is not None else None,
                ),
                ("timer_basis", timer_basis),
                ("request_count", request_count),
                ("dropped_records", dropped_records),
                ("local_state", local_state),
                ("state_domain", state_domain),
            )
            payload = " ".join(f"{key}={_token(value)}" for key, value in fields)
            return f"{_LOG_PREFIX} {payload}"
        except Exception:
            self._mark_failure()
            return None

    def _record_drop(self) -> None:
        with self._drop_lock:
            self._dropped_records_total += 1
            self._pending_dropped_records += 1

    def _take_pending_drops(self) -> int:
        with self._drop_lock:
            dropped_records = self._pending_dropped_records
            self._pending_dropped_records = 0
            return dropped_records

    @property
    def dropped_record_count(self) -> int:
        with self._drop_lock:
            return self._dropped_records_total

    def _writer_loop(self) -> None:
        assert self._output_queue is not None
        while True:
            record = self._output_queue.get()
            try:
                if record is _WRITER_STOP:
                    return
                _write_lifecycle_record(record)
                dropped_records = self._take_pending_drops()
                if dropped_records:
                    notice = self.capture(
                        DisaggLifecycleEvent.RECORDS_DROPPED,
                        emitter=(DisaggLifecycleEmitterName.DIAGNOSTICS_WRITER),
                        role=DisaggLifecycleRole.PROCESS,
                        correlation=DisaggCorrelation(None, None, None),
                        dropped_records=dropped_records,
                    )
                    if notice is not None:
                        _write_lifecycle_record(notice)
            except Exception:
                self.record_failure()
            finally:
                self._output_queue.task_done()
            if self._stop_requested.is_set() and self._output_queue.empty():
                return

    def close(self, timeout: float = _DEFAULT_CLOSE_TIMEOUT_SECONDS) -> bool:
        """Drain accepted async records and stop the writer within ``timeout``."""

        if self._output_queue is None or self._writer_thread is None:
            return True
        if not self._writer_thread.is_alive():
            return True

        deadline = time.monotonic() + max(0.0, timeout)
        remaining = max(0.0, deadline - time.monotonic())
        accept_lock_acquired = (
            self._accept_lock.acquire(blocking=False)
            if remaining == 0
            else self._accept_lock.acquire(timeout=remaining)
        )
        if not accept_lock_acquired:
            return False
        try:
            self._closed = True
            if not self._stop_signal_enqueued:
                try:
                    self._output_queue.put(_WRITER_STOP, block=False)
                    self._stop_signal_enqueued = True
                except queue.Full:
                    # The writer exits after draining accepted records.
                    pass
                except Exception:
                    self._mark_failure()
            self._stop_requested.set()
        finally:
            self._accept_lock.release()

        remaining = max(0.0, deadline - time.monotonic())
        self._writer_thread.join(timeout=remaining)
        return not self._writer_thread.is_alive()

    def publish(self, record: Optional[str]) -> None:
        if record is None:
            self._report_failure_once()
            return
        if self._output_queue is not None:
            publication_failed = False
            with self._accept_lock:
                if self._closed:
                    return
                try:
                    self._output_queue.put_nowait(record)
                except queue.Full:
                    self._record_drop()
                except Exception:
                    publication_failed = True
            if publication_failed:
                self.record_failure()
            return
        try:
            with self._publish_lock:
                _write_lifecycle_record(record)
        except Exception:
            self._mark_failure()
        self._report_failure_once()

    def emit(self, event: DisaggLifecycleEvent, **kwargs) -> None:
        self.publish(self.capture(event, **kwargs))

    def capture_for_request(
        self,
        event: DisaggLifecycleEvent,
        request,
        *,
        emitter: DisaggLifecycleEmitterName,
        role: Optional[DisaggLifecycleRole] = None,
        local_request_id: Optional[int] = None,
        **kwargs,
    ) -> Optional[str]:
        if not self.enabled:
            return None
        try:
            role = role or get_disagg_role(request)
            if role is None:
                return None
            schedule_style = kwargs.pop("schedule_style", get_disagg_schedule_style(request))
            local_state = kwargs.pop("local_state", _enum_name(getattr(request, "state", None)))
            return self.capture(
                event,
                emitter=emitter,
                role=role,
                correlation=get_disagg_correlation(request, local_request_id),
                schedule_style=schedule_style,
                local_state=local_state,
                state_domain="llm_request" if local_state is not None else None,
                **kwargs,
            )
        except Exception:
            self._mark_failure()
            return None

    def emit_for_request(self, event: DisaggLifecycleEvent, request, **kwargs) -> None:
        self.publish(self.capture_for_request(event, request, **kwargs))

    def capture_for_role(
        self,
        *,
        ctx_event: DisaggLifecycleEvent,
        gen_event: DisaggLifecycleEvent,
        request,
        **kwargs,
    ) -> Optional[str]:
        """Capture a role-specific event without exposing request access errors."""

        if not self.enabled:
            return None
        try:
            role = get_disagg_role(request)
            if role is None:
                return None
            event = ctx_event if role == DisaggLifecycleRole.CTX else gen_event
            return self.capture_for_request(
                event,
                request,
                role=role,
                **kwargs,
            )
        except Exception:
            self._mark_failure()
            return None

    def emit_for_role(self, **kwargs) -> None:
        self.publish(self.capture_for_role(**kwargs))
