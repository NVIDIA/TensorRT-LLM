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
"""Stdlib-only per-process JSONL event writer for the perf-time-events feature.

Every process that participates in an end-to-end perf timeline appends
**one JSONL line per lifecycle event** (flushed as the event happens) to its own
file, so a request that never completes -- a KV-transfer / fill-gate livelock, a
HangDetector ``MPI_Abort`` at 300 s -- still leaves its partial timeline on disk.
An offline compiler joins the per-process files into one combined per-request
JSONL.

Three kinds of process share this writer:

* **worker** (``PyExecutor`` ctx/gen ranks) -- emits ``ctx_*`` / ``gen_*`` events
  via :func:`emit_event`, keyed on ``TRTLLM_PERF_TIME_EVENTS_PATH``, one file per
  rank (``time_events_rank{N}_pid{P}.jsonl``). This process already imports
  ``torch``, so it is the only place the steady clock is imported (lazily, in
  :func:`_lazy_steady_clock_now`).
* **router/orchestrator** (``OpenAIDisaggServer``, uvicorn) -- emits ``router``
  events, keyed on ``TRTLLM_PERF_TIME_EVENTS_ROUTER_PATH``.
* **benchmark client** (``benchmark_serving`` load generator) -- emits ``client``
  events, keyed on ``TRTLLM_PERF_TIME_EVENTS_CLIENT_PATH``.

The router and client MUST stay torch-free (see
``tests/unittest/others/test_import_gpu_free.py``); they already hold the steady
clock (``responses_utils.get_steady_clock_now_in_seconds``) and pass ``t`` in, so
this module never imports it at top level -- the sole steady-clock import lives
inside :func:`_lazy_steady_clock_now`, reached only from :func:`emit_event` when a
worker leaves ``t`` unset while capture is enabled.

Design: the caller's thread only builds a dict and does a non-blocking
:meth:`queue.Queue.put_nowait`; a lazily-started daemon thread does the actual
``json.dumps`` + write + flush. No process has a single deterministic
``shutdown()`` choke point, so an :mod:`atexit` hook flushes and joins the writer
on interpreter exit.

All three env vars name an output **directory** (symmetric with the KV logger's
``TRTLLM_KVCACHE_TIME_OUTPUT_PATH``). Files are named ``{prefix}_pid{P}.jsonl`` so
multiple processes on a shared filesystem never collide.

The flat per-event record schema (identical across all roles) is built by
:func:`make_event_record`:
``{"role", "event", "request_id", "ctx_request_id", "rank", "t", "pid", **extra}``
where ``role in {router, ctx, gen, client}`` and ``t`` is steady-clock seconds.
"""

import atexit
import json
import os
import queue
import sys
import threading
from typing import Optional

# Sentinel enqueued by ``close()`` to stop the writer thread.
_WRITER_STOP = object()

# Env var naming the router dispatch-timeline output directory.
ROUTER_EVENTS_PATH_ENV = "TRTLLM_PERF_TIME_EVENTS_ROUTER_PATH"
# Env var naming the benchmark-client send-timeline output directory.
CLIENT_EVENTS_PATH_ENV = "TRTLLM_PERF_TIME_EVENTS_CLIENT_PATH"
# Env var naming the worker (ctx/gen) per-rank output directory. Shared with the
# executor's ``PerfMetricsManager`` capture switch; when set it force-enables the
# whole capture path (independent of ``LlmArgs.return_perf_metrics``).
WORKER_EVENTS_PATH_ENV = "TRTLLM_PERF_TIME_EVENTS_PATH"


class JsonlEventWriter:
    """Append-only, off-thread JSONL writer for one process.

    Args:
        events_dir: Output directory (created on first write). If falsy the
            writer is inert -- every :meth:`write` is a no-op -- so callers can
            construct one unconditionally and gate purely on the env var.
        filename_prefix: Basename prefix; the file is
            ``{events_dir}/{filename_prefix}_pid{os.getpid()}.jsonl``.
        max_queue: Bounded queue size; a full queue drops the record (never
            blocks the caller). Matches the worker writer's back-pressure policy.
    """

    def __init__(
        self,
        events_dir: Optional[str],
        filename_prefix: str,
        max_queue: int = 100000,
    ):
        self._events_dir = events_dir or None
        self._filename_prefix = filename_prefix
        self._max_queue = max_queue
        self._events_file = None  # opened lazily by the writer thread
        self._writer_queue: Optional[queue.Queue] = None
        self._writer_thread: Optional[threading.Thread] = None
        self._writer_lock = threading.Lock()
        self._atexit_registered = False
        self._dropped = 0

    @property
    def enabled(self) -> bool:
        return self._events_dir is not None

    def write(self, record: dict) -> None:
        """Enqueue one record for the writer thread (non-blocking).

        No-op when the writer is inert (no output dir configured).
        """
        if self._events_dir is None:
            return
        self._ensure_writer()
        try:
            self._writer_queue.put_nowait(record)
        except queue.Full:
            # Prefer dropping over blocking the router event loop / client
            # request path. Count so close() can surface it once.
            self._dropped += 1

    def _ensure_writer(self) -> None:
        """Lazily create the bounded queue + daemon writer thread (once)."""
        if self._writer_thread is not None:
            return
        with self._writer_lock:
            if self._writer_thread is not None:
                return
            self._writer_queue = queue.Queue(maxsize=self._max_queue)
            thread = threading.Thread(
                target=self._writer_loop,
                name=f"{self._filename_prefix}-writer",
                daemon=True,
            )
            self._writer_thread = thread
            thread.start()
            if not self._atexit_registered:
                atexit.register(self.close)
                self._atexit_registered = True

    def _writer_loop(self) -> None:
        """Daemon loop: drain the queue and append one JSON line per record."""
        path = os.path.join(
            self._events_dir,
            f"{self._filename_prefix}_pid{os.getpid()}.jsonl",
        )
        try:
            os.makedirs(self._events_dir, exist_ok=True)
            self._events_file = open(path, "a", encoding="utf-8")
        except OSError as e:
            # No logger dependency here (stdlib-only, torch-free); print to
            # stderr so a misconfigured path is still visible.
            print(f"WARNING: failed to open perf time-events file {path}: {e}",
                  file=sys.stderr)
            # Go permanently inert on open failure: clear the output dir so
            # write() short-circuits (an unbounded fill to maxsize would otherwise
            # make close()'s put() block forever on an exited consumer) and drop
            # the thread handle so close() early-returns.
            self._events_dir = None
            self._writer_thread = None
            return
        while True:
            record = self._writer_queue.get()
            if record is _WRITER_STOP:
                break
            try:
                self._events_file.write(json.dumps(record, default=str) + "\n")
                self._events_file.flush()
            except (OSError, TypeError) as e:
                print(f"WARNING: failed to write perf time-event record: {e}",
                      file=sys.stderr)
        try:
            self._events_file.close()
        except OSError:
            pass

    def close(self) -> None:
        """Flush and stop the writer thread; safe to call repeatedly / when inert."""
        thread = self._writer_thread
        if thread is None:
            return
        if self._dropped:
            print(
                f"WARNING: perf time-events writer dropped {self._dropped} record(s) "
                f"({self._filename_prefix}) due to a full queue",
                file=sys.stderr,
            )
        try:
            # Bounded put: if the writer thread already exited (e.g. open failure)
            # the queue may be full and never drain, so a blocking put() would hang
            # teardown forever. The timeout plus the join() below bound close()
            # regardless of the thread's state.
            self._writer_queue.put(_WRITER_STOP, timeout=5)
        except queue.Full:
            pass
        except Exception:  # noqa: BLE001 - best-effort on teardown
            pass
        thread.join(timeout=30)
        self._writer_thread = None


def make_env_writer(env_var: str, filename_prefix: str) -> JsonlEventWriter:
    """Build a :class:`JsonlEventWriter` from an env-named directory.

    Returns an inert writer (``enabled`` False, ``write`` a no-op) when the env
    var is unset/empty, so callers never branch on the env themselves.
    """
    return JsonlEventWriter(os.getenv(env_var, ""), filename_prefix)


# ---------------------------------------------------------------------------
# Flat per-event record + role-keyed emit helpers
# ---------------------------------------------------------------------------


def make_event_record(
    role: str,
    event: str,
    request_id=None,
    ctx_request_id=None,
    rank: Optional[int] = None,
    t: Optional[float] = None,
    **extra,
) -> dict:
    """Build one flat per-event record.

    The envelope is identical across every role so the offline compiler can pivot
    a mixed stream long->wide with no role-specific parsing:

    ``{"role", "event", "request_id", "ctx_request_id", "rank", "t", "pid"}``

    ``extra`` carries role-specific provenance (e.g. the client's ``prompt_len``
    / ``output_tokens`` on ``client_send``, or the router's server names); the
    compiler ignores keys it does not model. ``t`` is steady-clock seconds; pass
    it in from a caller that already holds the clock (router / client) -- this
    builder never imports it.
    """
    record = {
        "role": role,
        "event": event,
        "request_id": request_id,
        "ctx_request_id": ctx_request_id,
        "rank": rank,
        "t": t,
        "pid": os.getpid(),
    }
    if extra:
        record.update(extra)
    return record


# Process-global worker writer, built lazily on the first worker ``emit_event`` so
# importing this module (router / client / tests) never touches the worker env
# var or the steady clock. ``set_worker_rank`` fixes the per-rank filename.
_WORKER_WRITER: Optional[JsonlEventWriter] = None
_WORKER_RANK: int = 0
_WORKER_LOCK = threading.Lock()


def set_worker_rank(rank: int) -> None:
    """Pin the global rank used in this worker's ``time_events_rank{N}`` filename.

    Call once from the executor before the first :func:`emit_event`. Safe to call
    repeatedly with the same value; a change after the writer thread has opened
    its file has no effect on the already-open path (rank is fixed at open time),
    so only the value seen at first emit matters.
    """
    global _WORKER_RANK
    _WORKER_RANK = int(rank)


def _get_worker_writer() -> JsonlEventWriter:
    """Return the process-global worker writer, building it once (thread-safe).

    Keyed on ``TRTLLM_PERF_TIME_EVENTS_PATH``; inert (a no-op writer) when unset,
    so :func:`emit_event` costs a dict build + drop when capture is off.
    """
    global _WORKER_WRITER
    writer = _WORKER_WRITER
    if writer is not None:
        return writer
    with _WORKER_LOCK:
        if _WORKER_WRITER is None:
            _WORKER_WRITER = JsonlEventWriter(
                os.getenv(WORKER_EVENTS_PATH_ENV, ""),
                f"time_events_rank{_WORKER_RANK}",
            )
        return _WORKER_WRITER


def _lazy_steady_clock_now() -> float:
    """Return steady-clock seconds, importing the binding lazily.

    The steady clock lives in ``tensorrt_llm.bindings`` (the torch/CUDA
    extension), so importing it at module scope would break the torch-free
    contract for the router / client. This is only ever reached from a WORKER
    process (which already holds ``torch``) when a caller omits ``t`` while
    capture is enabled -- router / client always pass ``t`` in.
    """
    from tensorrt_llm.serve.responses_utils import \
        get_steady_clock_now_in_seconds
    return get_steady_clock_now_in_seconds()


def emit_event(
    role: str,
    event: str,
    request_id=None,
    ctx_request_id=None,
    rank: Optional[int] = None,
    t: Optional[float] = None,
    **extra,
) -> None:
    """Emit one worker (``ctx`` / ``gen``) time-event line; no-op when inert.

    This is the entry point for worker-side call sites that hold no
    ``PerfMetricsManager`` handle (``seq_slot_manager``, ``native/transfer``,
    ``py_executor``). It:

    * short-circuits before building anything when
      ``TRTLLM_PERF_TIME_EVENTS_PATH`` is unset (the writer is inert), so an
      un-instrumented run pays only the ``enabled`` check;
    * defaults ``rank`` to the pinned worker rank and ``t`` to the steady clock
      (lazy import, worker-only);
    * enqueues off the hot path (``put_nowait``, drop-on-full).

    Router and client use ``make_env_writer`` + ``JsonlEventWriter.write`` with
    :func:`make_event_record` directly (they pass their own ``t`` and stay
    torch-free); this helper is worker-specific.
    """
    writer = _get_worker_writer()
    if not writer.enabled:
        return
    if t is None:
        t = _lazy_steady_clock_now()
    if rank is None:
        rank = _WORKER_RANK
    writer.write(
        make_event_record(
            role,
            event,
            request_id=request_id,
            ctx_request_id=ctx_request_id,
            rank=rank,
            t=t,
            **extra,
        )
    )
