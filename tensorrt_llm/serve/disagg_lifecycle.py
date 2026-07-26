# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in orchestrator-level lifecycle event stream for disaggregated serving.

Each record is a single newline-terminated JSON line written to stdout.  The
stream is *observational*: it does not coordinate request state across context
and generation workers.  Enable with::

    TRTLLM_DISAGG_ORCHESTRATOR_DIAGNOSTICS = 1

Correlated with executor-layer events (``TRTLLM_DISAGG_TRANSFER_DIAGNOSTICS``)
via ``disagg_request_id`` and, once WS3 attempt fencing lands, via
``attempt_id``.

Timer semantics (Fable lifecycle design §5):

* ``ctx_dispatch`` → ``ctx_complete``: measures full CTX leg including network
  and prefill.  The gap between ``ctx_complete`` and ``gen_dispatch`` is the
  orchestrator's own scheduling overhead.
* ``gen_dispatch`` → ``gen_complete``: measures the GEN leg.  Note that the
  active-transfer clock should start at ``receiver_ready`` (WS2, not yet
  implemented here), *not* at ``gen_dispatch`` — this field reflects the
  orchestrator's wall-clock observation, which includes GEN queueing delay.
* On disconnect or abort the ``abort`` record carries the reason and the
  elapsed wall time so consumers can distinguish queueing delay from transfer
  stalls.

Records may arrive out of order in high-concurrency deployments.  Consumers
should order by ``(wall_ns, seq)`` within a single ``clock_id``.
"""

from __future__ import annotations

import json
import os
import secrets
import threading
import time
from enum import Enum
from typing import Any, Dict, Optional

_ENV_VAR = "TRTLLM_DISAGG_ORCHESTRATOR_DIAGNOSTICS"
_LOG_PREFIX = "[DISAGG_DIAG][orchestrator]"
_SCHEMA_VERSION = 1


class OrchestratorEvent(str, Enum):
    """Lifecycle events emitted by the disagg orchestrator."""

    CTX_DISPATCH = "ctx_dispatch"
    CTX_COMPLETE = "ctx_complete"
    CTX_ERROR = "ctx_error"
    GEN_DISPATCH = "gen_dispatch"
    GEN_COMPLETE = "gen_complete"
    GEN_ERROR = "gen_error"
    # GEN worker returned an HTTP 4xx/5xx — maps to GEN_REJECT in §4
    GEN_REJECTED = "gen_rejected"
    # Client disconnect detected before both legs completed
    CLIENT_DISCONNECT = "client_disconnect"
    # Abort triggered by any leg error, timeout, or disconnect
    ABORT = "abort"


class OrchestratorScheduleStyle(str, Enum):
    CONTEXT_FIRST = "context_first"
    GENERATION_FIRST = "generation_first"


class DisaggOrchestratorLifecycle:
    """Lightweight per-process lifecycle emitter for the disagg orchestrator.

    Thread-safe; individual ``emit`` calls take a sequence number under a lock
    but release it before formatting and printing so the hot path is not
    serialised end-to-end.
    """

    def __init__(self, *, enabled: bool, node_id: Optional[str] = None):
        self.enabled = bool(enabled)
        self._node_id = str(node_id) if node_id is not None else "-"
        self._clock_id = f"orch-{os.getpid()}-{secrets.token_hex(6)}" if enabled else "disabled"
        self._seq = 0
        self._lock = threading.Lock()

    @classmethod
    def from_environment(cls, *, node_id: Optional[str] = None) -> "DisaggOrchestratorLifecycle":
        enabled = os.environ.get(_ENV_VAR, "0") not in ("0", "", "false", "False", "FALSE")
        return cls(enabled=enabled, node_id=node_id)

    def tracer(
        self,
        *,
        disagg_request_id: Optional[int] = None,
        schedule_style: Optional[OrchestratorScheduleStyle] = None,
    ) -> "OrchestratorRequestTracer":
        return OrchestratorRequestTracer(
            self, disagg_request_id=disagg_request_id, schedule_style=schedule_style
        )

    def _next_seq(self) -> int:
        with self._lock:
            self._seq += 1
            return self._seq

    def emit(
        self,
        event: OrchestratorEvent,
        *,
        disagg_request_id: Optional[int] = None,
        ctx_server: Optional[str] = None,
        gen_server: Optional[str] = None,
        schedule_style: Optional[OrchestratorScheduleStyle] = None,
        elapsed_ms: Optional[float] = None,
        http_status: Optional[int] = None,
        error: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit one lifecycle record.  A no-op when disabled."""
        if not self.enabled:
            return
        seq = self._next_seq()
        wall_ns = time.time_ns()
        record: Dict[str, Any] = {
            "v": _SCHEMA_VERSION,
            "clock_id": self._clock_id,
            "seq": seq,
            "wall_ns": wall_ns,
            "node": self._node_id,
            "event": event.value,
        }
        if disagg_request_id is not None:
            record["rid"] = disagg_request_id
        if ctx_server is not None:
            record["ctx"] = ctx_server
        if gen_server is not None:
            record["gen"] = gen_server
        if schedule_style is not None:
            record["sched"] = schedule_style.value
        if elapsed_ms is not None:
            record["elapsed_ms"] = round(elapsed_ms, 3)
        if http_status is not None:
            record["http_status"] = http_status
        if error is not None:
            record["error"] = error[:256]
        if extra:
            record.update(extra)
        try:
            print(f"{_LOG_PREFIX} {json.dumps(record, separators=(',', ':'))}", flush=True)
        except Exception:
            pass


class OrchestratorRequestTracer:
    """Per-request helper that snapshots timestamps and emits paired events.

    Usage::

        tracer = lifecycle.tracer(disagg_request_id=rid, ...)
        tracer.ctx_dispatch(ctx_server)
        ...
        tracer.ctx_complete()
        tracer.gen_dispatch(gen_server)
        ...
        tracer.gen_complete()

    All times are wall-clock milliseconds elapsed since the tracer was created.
    """

    def __init__(
        self,
        emitter: DisaggOrchestratorLifecycle,
        *,
        disagg_request_id: Optional[int] = None,
        schedule_style: Optional[OrchestratorScheduleStyle] = None,
    ):
        self._emitter = emitter
        self._rid = disagg_request_id
        self._style = schedule_style
        self._t0_ns = time.monotonic_ns()
        self._ctx_dispatch_ns: Optional[int] = None
        self._gen_dispatch_ns: Optional[int] = None

    def _elapsed_ms(self, since_ns: Optional[int] = None) -> float:
        now = time.monotonic_ns()
        ref = since_ns if since_ns is not None else self._t0_ns
        return (now - ref) / 1e6

    def _emit(self, event: OrchestratorEvent, **kw) -> None:
        self._emitter.emit(
            event,
            disagg_request_id=self._rid,
            schedule_style=self._style,
            elapsed_ms=self._elapsed_ms(),
            **kw,
        )

    def ctx_dispatch(self, server: str) -> None:
        self._ctx_dispatch_ns = time.monotonic_ns()
        self._emit(OrchestratorEvent.CTX_DISPATCH, ctx_server=server)

    def ctx_complete(self, ctx_server: str = "") -> None:
        since = self._ctx_dispatch_ns
        elapsed = self._elapsed_ms(since) if since is not None else None
        self._emitter.emit(
            OrchestratorEvent.CTX_COMPLETE,
            disagg_request_id=self._rid,
            schedule_style=self._style,
            elapsed_ms=elapsed,
            ctx_server=ctx_server or None,
        )

    def ctx_error(
        self, error: str, ctx_server: str = "", http_status: Optional[int] = None
    ) -> None:
        self._emit(
            OrchestratorEvent.CTX_ERROR,
            ctx_server=ctx_server or None,
            http_status=http_status,
            error=error,
        )

    def gen_dispatch(self, server: str) -> None:
        self._gen_dispatch_ns = time.monotonic_ns()
        self._emit(OrchestratorEvent.GEN_DISPATCH, gen_server=server)

    def gen_complete(self, gen_server: str = "") -> None:
        since = self._gen_dispatch_ns
        elapsed = self._elapsed_ms(since) if since is not None else None
        self._emitter.emit(
            OrchestratorEvent.GEN_COMPLETE,
            disagg_request_id=self._rid,
            schedule_style=self._style,
            elapsed_ms=elapsed,
            gen_server=gen_server or None,
        )

    def gen_error(
        self, error: str, gen_server: str = "", http_status: Optional[int] = None
    ) -> None:
        event = (
            OrchestratorEvent.GEN_REJECTED
            if http_status is not None and http_status >= 400
            else OrchestratorEvent.GEN_ERROR
        )
        self._emit(event, gen_server=gen_server or None, http_status=http_status, error=error)

    def client_disconnect(self) -> None:
        self._emit(OrchestratorEvent.CLIENT_DISCONNECT)

    def abort(self, reason: str) -> None:
        self._emit(OrchestratorEvent.ABORT, error=reason)
