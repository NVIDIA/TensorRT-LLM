# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in request-edge diagnostics for disaggregated KV transfer.

Set ``TRTLLM_DISAGG_TRANSFER_DIAGNOSTICS=1`` before process startup for a
targeted diagnostic run. Standard and performance runs leave it unset, making
each instrumented edge a module-attribute check with no clock read, request
inspection, serialization, or log emission.
"""

from __future__ import annotations

import atexit
import functools
import json
import os
import queue
import socket
import threading
import time
from typing import TYPE_CHECKING, Optional, TypeAlias

if TYPE_CHECKING:
    from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo

_DIAGNOSTICS_ENV = "TRTLLM_DISAGG_TRANSFER_DIAGNOSTICS"
DIAGNOSTICS_LOG_PREFIX = "[DISAGG_TRANSFER_DIAG] "
DIAGNOSTICS_SCHEMA_VERSION = 1
_DIAGNOSTIC_QUEUE_CAPACITY = 32_768

# Read once so the disabled path is a single caller-side branch. Tests may
# monkeypatch this module attribute without reloading importers.
DISAGG_TRANSFER_DIAGNOSTICS_ENABLED = os.getenv(_DIAGNOSTICS_ENV) == "1"

DiagnosticValue: TypeAlias = str | int | float | bool | None
DiagnosticRecord: TypeAlias = dict[str, DiagnosticValue]
DiagnosticTimestamp: TypeAlias = tuple[int, int]


def capture_timestamp() -> DiagnosticTimestamp:
    """Capture one local event boundary before publishing shared state."""
    return time.monotonic_ns(), time.time_ns()


@functools.lru_cache(maxsize=1)
def _host_identity() -> str:
    """Return the execution-node identity, computed only when diagnostics run."""
    hostname = os.getenv("SLURMD_NODENAME")
    if not hostname:
        try:
            hostname = socket.gethostname()
        except OSError:
            hostname = os.getenv("HOSTNAME", "unknown")
    return "_".join(hostname.split()) if hostname else "unknown"


class _AsyncDiagnosticSink:
    """Serialize and write records away from request-progress threads."""

    def __init__(self, pid: int) -> None:
        self.pid = pid
        self._queue: queue.Queue[DiagnosticRecord] = queue.Queue(maxsize=_DIAGNOSTIC_QUEUE_CAPACITY)
        self._stop_requested = threading.Event()
        self._drop_lock = threading.Lock()
        self._dropped = 0
        self._thread = threading.Thread(
            target=self._run,
            name="disagg-transfer-diagnostics",
            daemon=True,
        )
        self._thread.start()

    def submit(self, record: DiagnosticRecord) -> None:
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            # Never block request progress on diagnostics. A later writer pass
            # reports the loss explicitly when the sink catches up.
            with self._drop_lock:
                self._dropped += 1

    def _take_dropped(self) -> int:
        with self._drop_lock:
            dropped = self._dropped
            self._dropped = 0
        return dropped

    @staticmethod
    def _write(record: DiagnosticRecord) -> None:
        line = (
            f"{DIAGNOSTICS_LOG_PREFIX}{json.dumps(record, separators=(',', ':'), sort_keys=True)}\n"
        )
        os.write(1, line.encode("utf-8"))

    def _write_drop_record(self) -> None:
        dropped = self._take_dropped()
        if dropped == 0:
            return
        self._write(
            {
                "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
                "event": "diagnostics_events_dropped",
                "side": "runtime",
                "request_id": None,
                "local_request_id": None,
                "host": _host_identity(),
                "pid": self.pid,
                "monotonic_ns": time.monotonic_ns(),
                "wall_ns": time.time_ns(),
                "dropped_events": dropped,
            }
        )

    def _run(self) -> None:
        while True:
            try:
                record = self._queue.get(timeout=0.1)
            except queue.Empty:
                if self._stop_requested.is_set():
                    try:
                        self._write_drop_record()
                    except Exception:
                        pass
                    return
                continue

            try:
                self._write_drop_record()
                record["host"] = _host_identity()
                self._write(record)
            except Exception:
                # Diagnostics are best-effort and must never affect request
                # progress, even if stdout is closed or serialization fails.
                pass
            finally:
                self._queue.task_done()

    def flush(self) -> None:
        """Wait for already accepted records; used only by focused tests."""
        self._queue.join()

    def close(self) -> None:
        self._stop_requested.set()
        self._thread.join(timeout=1.0)


_sink_lock = threading.Lock()
_sink: Optional[_AsyncDiagnosticSink] = None


def _reset_sink_after_fork() -> None:
    """Discard inherited thread and lock state in a forked child."""
    global _sink, _sink_lock
    _sink = None
    _sink_lock = threading.Lock()


def _get_sink(pid: int) -> _AsyncDiagnosticSink:
    """Return a per-process sink, replacing inherited pre-fork state."""
    global _sink
    sink = _sink
    if sink is not None and sink.pid == pid:
        return sink
    with _sink_lock:
        sink = _sink
        if sink is None or sink.pid != pid:
            sink = _AsyncDiagnosticSink(pid)
            _sink = sink
    return sink


def _flush_diagnostic_sink_for_tests() -> None:
    sink = _sink
    if sink is not None:
        sink.flush()


def _reset_diagnostic_sink_for_tests() -> None:
    global _sink
    with _sink_lock:
        sink = _sink
        _sink = None
    if sink is not None:
        sink.close()


def _shutdown_diagnostic_sink() -> None:
    sink = _sink
    if sink is not None and sink.pid == os.getpid():
        sink.close()


atexit.register(_shutdown_diagnostic_sink)
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_sink_after_fork)


def emit_event(
    event: str,
    *,
    side: str,
    request_id: Optional[int],
    local_request_id: Optional[int] = None,
    rank_info: Optional["RankInfo"] = None,
    rank: Optional[int] = None,
    instance: Optional[str] = None,
    slice_id: Optional[int] = None,
    peer_rank: Optional[int] = None,
    timestamp: Optional[DiagnosticTimestamp] = None,
    **details: DiagnosticValue,
) -> None:
    """Emit one compact JSON request-edge event when explicitly enabled.

    Callers must also guard this function with
    ``DISAGG_TRANSFER_DIAGNOSTICS_ENABLED`` so argument construction and
    request-state inspection are absent from the disabled path. This internal
    check keeps accidental unguarded calls inexpensive and harmless.
    """
    if not DISAGG_TRANSFER_DIAGNOSTICS_ENABLED:
        return

    try:
        pid = os.getpid()
        if timestamp is None:
            timestamp = capture_timestamp()
        monotonic_ns, wall_ns = timestamp
        record: DiagnosticRecord = {
            "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
            "event": event,
            "side": side,
            "request_id": request_id,
            "local_request_id": local_request_id,
            "pid": pid,
            "monotonic_ns": monotonic_ns,
            "wall_ns": wall_ns,
        }
        if rank_info is not None:
            record.update(
                {
                    "instance": rank_info.instance_name,
                    "rank": rank_info.instance_rank,
                    "tp_rank": rank_info.tp_rank,
                    "pp_rank": rank_info.pp_rank,
                    "cp_rank": rank_info.cp_rank,
                    "dp_rank": rank_info.dp_rank,
                }
            )
        else:
            record["instance"] = instance
            record["rank"] = rank
        if slice_id is not None:
            record["slice_id"] = slice_id
        if peer_rank is not None:
            record["peer_rank"] = peer_rank
        record.update(details)
        _get_sink(pid).submit(record)
    except Exception:
        # Diagnostics are best-effort and must never affect request progress.
        return
