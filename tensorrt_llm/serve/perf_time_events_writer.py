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

The PyExecutor worker writes its own per-rank ``time_events_rank{N}_pid{P}.jsonl``
files from inside ``perf_metrics_manager.PerfMetricsManager`` (which imports
``torch``). The two OTHER processes that participate in an end-to-end perf
timeline -- the disaggregated **router/orchestrator** (``OpenAIDisaggServer``,
uvicorn) and the **benchmark client** (``benchmark_serving`` load generator) --
must stay torch-free (see ``tests/unittest/others/test_import_gpu_free.py``), so
they cannot reuse that class. This module gives them the same off-hot-path writer
shape with a stdlib-only footprint (``json`` / ``os`` / ``queue`` / ``threading``
/ ``atexit``).

Design mirrors the worker writer: the caller's thread only does a dict build plus
a non-blocking :meth:`queue.Queue.put_nowait`; a lazily-started daemon thread does
the actual ``json.dumps`` + write + flush. Neither the router nor the client has a
single deterministic ``shutdown()`` choke point, so an :mod:`atexit` hook flushes
and joins the writer on interpreter exit.

Both env vars name an output **directory** (symmetric with the worker's
``TRTLLM_PERF_TIME_EVENTS_PATH`` and the KV logger's
``TRTLLM_KVCACHE_TIME_OUTPUT_PATH``):

* ``TRTLLM_PERF_TIME_EVENTS_ROUTER_PATH`` -- disagg router dispatch timeline.
* ``TRTLLM_PERF_TIME_EVENTS_CLIENT_PATH`` -- benchmark client send timeline.

Files are named ``{prefix}_pid{P}.jsonl`` so multiple processes on a shared
filesystem never collide.
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
