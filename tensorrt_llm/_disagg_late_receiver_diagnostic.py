# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in, single-node late-receiver experiment coordination.

This module deliberately uses only the standard library so its controller and
tests can load it without importing the TensorRT-LLM runtime. A shared directory
and a fresh arm select one context-first request. No timeout is modified, and
only the router coroutine for that request waits for an observed retirement.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import socket
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Union

_DIRECTORY_ENV = "TRTLLM_LATE_RECEIVER_DIAGNOSTIC_DIR"
_NO_RETRY_ENV = "TRTLLM_DISAGG_NO_RETRY"
_TRIAL_ID = re.compile(r"[0-9a-f]{32}")
_MAX_EVENTS = 512
_MAX_JSON_BYTES = 16384
_POLL_SECONDS = 0.05
Scalar = Union[str, int, float, bool, None]
Record = dict[str, Scalar]


class DiagnosticError(RuntimeError):
    """An experiment cannot establish the required causal ordering."""


def _read_json(path: Path) -> Record:
    with path.open("rb") as stream:
        payload = stream.read(_MAX_JSON_BYTES + 1)
    if len(payload) > _MAX_JSON_BYTES:
        raise DiagnosticError(f"Diagnostic record is too large: {path.name}")
    record = json.loads(payload)
    if not isinstance(record, dict) or any(
        not isinstance(key, str) or not isinstance(value, (str, int, float, bool, type(None)))
        for key, value in record.items()
    ):
        raise DiagnosticError(f"Diagnostic record must contain scalar fields: {path.name}")
    return record


def _write_json(path: Path, record: Record) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, sort_keys=True, allow_nan=False)
            stream.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _trial_path(root: Path, trial_id: str) -> Path:
    if not isinstance(trial_id, str) or _TRIAL_ID.fullmatch(trial_id) is None:
        raise DiagnosticError("Invalid diagnostic trial ID")
    return root / "trials" / trial_id


def _validate_arm(record: Record) -> Record:
    _trial_path(Path("."), record.get("trial_id"))
    wait_seconds = record.get("wait_seconds")
    if (
        record.get("schema_version") != 1
        or record.get("mode") not in ("control", "after_retirement")
        or not isinstance(wait_seconds, (int, float))
        or isinstance(wait_seconds, bool)
        or not math.isfinite(wait_seconds)
        or not 0 < wait_seconds <= 180
    ):
        raise DiagnosticError("Invalid diagnostic arm configuration")
    return record


@contextmanager
def _arm_lock(root: Path) -> Iterator[None]:
    lock = root / ".arm-lock"
    try:
        lock.mkdir()
    except FileExistsError as error:
        raise DiagnosticError(
            "Diagnostic arm operation already active; inspect its lock"
        ) from error
    try:
        yield
    finally:
        lock.rmdir()


def arm(root: Path, mode: str, wait_seconds: float = 90) -> Record:
    """Arm one fresh request, refusing to replace an existing/stale arm.

    Args:
        root: Node-local directory shared by the controller and all servers.
        mode: Either ``control`` or ``after_retirement``.
        wait_seconds: Maximum asynchronous gate wait, at most 180 seconds.

    Returns:
        The immutable arm configuration, including its fresh trial ID.
    """
    root = Path(root)
    record = _validate_arm(
        {
            "schema_version": 1,
            "trial_id": uuid.uuid4().hex,
            "mode": mode,
            "wait_seconds": wait_seconds,
        }
    )
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    with _arm_lock(root):
        if (root / "armed.json").exists():
            raise DiagnosticError("Diagnostic already armed; preserve and disarm that trial first")
        (_trial_path(root, record["trial_id"]) / "events").mkdir(parents=True)
        _write_json(root / "armed.json", record)
    return record


def disarm(root: Path, trial_id: str) -> None:
    """Remove only the matching arm; retain its claim and all trace records.

    Args:
        root: The directory used to arm the experiment.
        trial_id: Exact trial ID returned by ``arm``.
    """
    root = Path(root)
    _trial_path(root, trial_id)
    with _arm_lock(root):
        current = _validate_arm(_read_json(root / "armed.json"))
        if current["trial_id"] != trial_id:
            raise DiagnosticError("Refusing to disarm a different trial")
        (root / "armed.json").unlink()


def is_enabled() -> bool:
    """Return whether this process opted into diagnostic instrumentation."""
    return bool(os.environ.get(_DIRECTORY_ENV))


def validate_environment() -> None:
    """Require no-retry configuration before constructing diagnostic servers."""
    if is_enabled() and os.environ.get(_NO_RETRY_ENV) != "1":
        raise DiagnosticError(f"{_NO_RETRY_ENV}=1 must precede diagnostic server startup")


def _active() -> tuple[Path, Record] | None:
    directory = os.environ.get(_DIRECTORY_ENV)
    if not directory:
        return None
    root = Path(directory)
    try:
        current = _read_json(root / "armed.json")
    except FileNotFoundError:
        return None
    return root, _validate_arm(current)


def _event(root: Path, trial: Record, disagg_request_id: int, event: str, **fields: Scalar) -> None:
    directory = _trial_path(root, trial["trial_id"]) / "events"
    if sum(1 for _ in directory.glob("*.json")) >= _MAX_EVENTS:
        raise DiagnosticError("Diagnostic event budget exhausted")
    record = dict(fields)
    record.update(
        {
            "schema_version": 1,
            "trial_id": trial["trial_id"],
            "mode": trial["mode"],
            "event": event,
            "disagg_request_id": disagg_request_id,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "monotonic_ns": time.monotonic_ns(),
            "utc_ns": time.time_ns(),
            "no_retry": os.environ.get(_NO_RETRY_ENV) == "1",
        }
    )
    _write_json(directory / f"{record['monotonic_ns']}-{uuid.uuid4().hex}.json", record)


def read_events(root: Path, trial_id: str) -> list[Record]:
    """Read immutable trace records, ordered by their same-host monotonic clock.

    Args:
        root: The directory used to arm the experiment.
        trial_id: Exact trial ID returned by ``arm``.

    Returns:
        Scalar event records. Callers must reject mixed hosts before comparing
        their monotonic timestamps.
    """
    paths = list((_trial_path(Path(root), trial_id) / "events").glob("*.json"))
    if len(paths) > _MAX_EVENTS:
        raise DiagnosticError("Diagnostic event budget exhausted")
    events = [_read_json(path) for path in paths]
    if any(
        event.get("schema_version") != 1
        or event.get("trial_id") != trial_id
        or not isinstance(event.get("monotonic_ns"), int)
        for event in events
    ):
        raise DiagnosticError("Invalid diagnostic event identity or timestamp")
    return sorted(events, key=lambda event: event["monotonic_ns"])


def claim_request(request_id: int, *, eligible: bool, clients_no_retry: bool) -> Record | None:
    """Claim one request before its CTX dispatch; leave other requests alone.

    Args:
        request_id: The initial disaggregated request ID.
        eligible: Whether this is CTX-first without an existing GEN reservation.
        clients_no_retry: Both clients' cached no-retry setting, not just the env.

    Returns:
        The selected request's claim, or None when unarmed/already claimed.
    """
    active = _active()
    if active is None:
        return None
    root, trial = active
    validate_environment()
    directory = _trial_path(root, trial["trial_id"])
    if (directory / ".claimed").exists():
        return None
    if not eligible or not clients_no_retry:
        raise DiagnosticError("Diagnostic requires CTX-first, no GEN reservation, and no retries")
    try:
        (directory / ".claimed").mkdir()
    except FileExistsError:
        return None
    claim = dict(trial, disagg_request_id=request_id, attempt=0)
    _write_json(directory / "claim.json", claim)
    _event(root, trial, request_id, "router_claimed", attempt=0)
    return claim


def observe(event: str, disagg_request_id: int, **fields: Scalar) -> None:
    """Best-effort observations never change executor/transport state.

    A missing observation fails the experiment's gate or final classification;
    it must not turn a diagnostic I/O error into a product failure.
    """
    try:
        active = _active()
        if active is None:
            return
        root, trial = active
        claim = _read_json(_trial_path(root, trial["trial_id"]) / "claim.json")
        if claim.get("disagg_request_id") != disagg_request_id:
            return
        _event(root, trial, disagg_request_id, event, **fields)
    except (OSError, ValueError, TypeError, DiagnosticError) as error:
        logging.getLogger(__name__).warning(
            "Late-receiver diagnostic observation unavailable: %s", error
        )


def _owner(event: Record) -> tuple[Scalar, Scalar, Scalar]:
    return event.get("host"), event.get("pid"), event.get("rank")


def _sender(events: list[Record], request_id: int, ctx_dp_rank: int) -> Record | None:
    if any(
        event.get("disagg_request_id") != request_id
        or event.get("host") != socket.gethostname()
        or event.get("no_retry") is not True
        for event in events
    ):
        raise DiagnosticError("Trace must use one request, host, and no-retry configuration")
    sends = [event for event in events if event["event"] == "ctx_send_published"]
    if not sends:
        return None
    if len(sends) != 1:
        raise DiagnosticError("Expected exactly one context send owner")
    sender = sends[0]
    if (
        sender.get("ctx_dp_rank") != ctx_dp_rank
        or sender.get("pp_size") != 1
        or sender.get("cp_size") != 1
        or sender.get("attention_dp") is not True
        or sender.get("bridge_enabled") is not False
        or sender.get("pipeline_transfer") is not False
        or sender.get("kv_transfer_timeout_ms") != 60000
        or not isinstance(sender.get("rank"), int)
    ):
        raise DiagnosticError(
            "Diagnostic requires non-pipelined PP1/CP1 attention-DP, bridge off, and 60000ms timeout"
        )
    return sender


def _confirmed_retirement(events: list[Record], sender: Record) -> bool:
    owner = _owner(sender)
    starts = [
        event
        for event in events
        if event["event"] == "ctx_timeout_started" and _owner(event) == owner
    ]
    closes = [
        event
        for event in events
        if event["event"] == "ctx_native_closed" and _owner(event) == owner
    ]
    retired = [
        event for event in events if event["event"] == "ctx_retired" and _owner(event) == owner
    ]
    if not starts or not closes or not retired:
        return False
    if len(starts) != 1 or len(closes) != 1 or len(retired) != 1:
        raise DiagnosticError("Ambiguous context lifecycle observations")
    start, close, retirement = starts[0], closes[0], retired[0]
    if (
        retirement.get("ctx_dp_rank") != sender["ctx_dp_rank"]
        or retirement.get("timed_out") is not True
        or retirement.get("close_confirmed") is not True
        or retirement.get("session_absent") is not True
        or retirement.get("retired_flag") is not True
        or not sender["monotonic_ns"]
        <= start["monotonic_ns"]
        <= close["monotonic_ns"]
        <= retirement["monotonic_ns"]
    ):
        raise DiagnosticError("Observed retirement does not prove timed-out sender removal")
    return True


async def before_generation(
    claim: Record | None, request_id: int, ctx_request_id: int, ctx_dp_rank: int
) -> None:
    """Gate only the claimed router coroutine, before GEN routing/admission.

    Args:
        claim: Selected request's claim, or None for a normal request.
        request_id: Effective disaggregated request ID after CTX responds.
        ctx_request_id: Sender session ID returned by CTX.
        ctx_dp_rank: Actual CTX owner returned by CTX, not every executor rank.
    """
    if claim is None:
        return
    active = _active()
    if active is None or active[1]["trial_id"] != claim["trial_id"]:
        raise DiagnosticError("Diagnostic was disarmed or replaced before GEN routing")
    root, trial = active
    validate_environment()
    if request_id != claim["disagg_request_id"] or ctx_request_id != request_id:
        _event(
            root,
            trial,
            claim["disagg_request_id"],
            "gate_invalid",
            reason="request_identity_changed",
        )
        raise DiagnosticError("CTX retry or request identity change invalidates the diagnostic")
    _event(
        root,
        trial,
        request_id,
        "router_gate_enter",
        ctx_request_id=ctx_request_id,
        ctx_dp_rank=ctx_dp_rank,
    )
    deadline = time.monotonic() + trial["wait_seconds"]
    try:
        while True:
            active = _active()
            if active is None or active[1] != trial:
                raise DiagnosticError("Diagnostic arm changed while GEN was gated")
            events = read_events(root, trial["trial_id"])
            sender = _sender(events, request_id, ctx_dp_rank)
            if sender is not None and (
                trial["mode"] == "control" or _confirmed_retirement(events, sender)
            ):
                _event(
                    root,
                    trial,
                    request_id,
                    "router_gate_released",
                    ctx_request_id=ctx_request_id,
                    ctx_dp_rank=ctx_dp_rank,
                    rank=sender["rank"],
                    reason="control" if trial["mode"] == "control" else "confirmed_retirement",
                )
                return
            if time.monotonic() >= deadline:
                _event(root, trial, request_id, "gate_timeout")
                raise DiagnosticError(
                    "Timed out waiting for confirmed CTX retirement; GEN not sent"
                )
            await asyncio.sleep(_POLL_SECONDS)
    except asyncio.CancelledError:
        _event(root, trial, request_id, "router_cancelled")
        raise
    except DiagnosticError as error:
        _event(root, trial, request_id, "gate_invalid", reason=str(error))
        raise
