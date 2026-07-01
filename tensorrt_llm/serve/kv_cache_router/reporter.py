# Copyright (c) 2026, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Worker-side reporter that pushes KV-cache events and load to the router.

Runs inside a TRT-LLM instance, next to the engine, and drives two independent
PUSH streams over one socket:

* **events** -- drained directly from the KV-cache *event manager*
  (``KVCacheManagerV2.event_manager``) via its ``get_latest_events(0)``, the
  queue the radix tree enqueues ``stored`` / ``removed`` events into. This is
  in-process and deliberately *not* the executor RPC path
  (``llm.get_kv_cache_events_async()``), which only surfaces events at
  request/iteration boundaries (motivation #1).
* **load** -- a periodic snapshot from the engine's iteration stats.

Each stream carries an independent monotonic ``seq``. ``worker_id`` (the instance
UUID) and ``namespace`` are fixed for the reporter's lifetime. For attention-DP,
run one reporter on rank 0 over the gathered event stream -- the instance reports
as a single routing target.
"""

import threading
import time
from typing import Callable, List, Optional

import zmq

from tensorrt_llm._utils import KVCacheEventSerializer
from tensorrt_llm.executor.ipc import ZeroMqQueue
from tensorrt_llm.logger import logger

from .messages import KvCacheEventReport, WorkerLoadReport

__all__ = ["WorkerReporter"]


def _now_on_synced_clock() -> float:
    """Wall-clock now (epoch seconds) for measuring worker->router report lag.

    Uses time.time() (NOT steady_clock): wall clock is epoch-based and
    NTP-disciplined, so timestamps are directly comparable across nodes. The
    C++ steady_clock_now() binding is unusable here -- it returns a raw
    per-machine steady_clock whose origin differs by ~1e6 s between hosts.
    Lag = router_recv_time - worker_send_time is meaningful only because both
    ends call time.time() on NTP-synced hosts; if NTP is off, lags will be
    implausible (large or negative) and should be discarded."""
    return time.time()


class WorkerReporter:
    """Pushes KV-cache event and load reports from one worker instance.

    Args:
        worker_id: Stable instance id (the ``llm_id`` advertised via
            ``/server_info``).
        namespace: Routing group this worker serves (e.g. ``"ctx"`` / ``"gen"``).
        router_address: Router PULL endpoint, e.g. ``"tcp://host:port"``.
        hmac_key: HMAC key obtained from the router's ``address`` tuple.
        get_events: Callable returning the latest KV-cache events for a
            ``timeout_ms``. Wire this to the **KV-cache event manager**
            (``KVCacheManagerV2.event_manager.get_latest_events``) -- the queue
            owner the radix tree enqueues into -- not the executor RPC API. The
            return value may be raw ``KVCacheEvent`` objects or already-serialized
            dicts; both are handled.
        get_load: Callable returning ``(num_active_requests, num_queued_requests)``.
        max_batch_size: Engine ``max_batch_size`` (sent in every load report).
        event_interval_s: Poll period for the event stream.
        load_interval_s: Send period for the load stream.
    """

    def __init__(self,
                 worker_id: str,
                 namespace: str,
                 router_address: str,
                 hmac_key: Optional[bytes],
                 get_events: Callable[[float], list],
                 get_load: Callable[[], tuple[int, int]],
                 max_batch_size: int,
                 event_interval_s: float = 0.01,
                 load_interval_s: float = 0.5) -> None:
        self._worker_id = worker_id
        self._namespace = namespace
        self._get_events = get_events
        self._get_load = get_load
        self._max_batch_size = max_batch_size
        self._event_interval_s = event_interval_s
        self._load_interval_s = load_interval_s

        self._queue = ZeroMqQueue(
            address=(router_address, hmac_key),
            socket_type=zmq.PUSH,
            is_server=False,
            name=f"kv_cache_reporter[{worker_id}]",
        )
        self._event_seq = 0
        self._load_seq = 0
        self._stop = threading.Event()
        self._event_thread: Optional[threading.Thread] = None
        self._load_thread: Optional[threading.Thread] = None
        # The ZMQ socket is not thread-safe and several producers may send on
        # it: the startup snapshot (caller thread), the periodic load loop, and
        # -- in inline mode -- the engine's executor loop via report_events().
        # Serialize every send under one lock, and hard-gate on _closed so a
        # send racing with stop()/close() becomes a no-op instead of touching a
        # torn-down (None) socket. This is the lifetime bug that crashed the
        # earlier inline attempt ('NoneType has no attribute send').
        self._send_lock = threading.Lock()
        self._closed = False

    def start(self, poll_events: bool = True) -> None:
        """Send the initial full snapshot and start the report loop(s).

        Args:
            poll_events: When True (default), run a background thread that polls
                the KV-cache event queue every ``event_interval_s`` and reports.
                When False, no event thread is started and the caller drives
                event reporting synchronously via :meth:`report_events` (e.g.
                inline at the engine's per-iteration event flush) -- the event
                queue is filled exactly once per iteration, so a separate poll
                clock only adds detection jitter and empty-poll churn. The
                periodic load thread runs in both modes.
        """
        if self._event_thread is not None or self._load_thread is not None:
            raise RuntimeError("WorkerReporter already started")
        # Initial snapshot lets a (re)started router rebuild this worker's table.
        self._push_events(self._drain_events(), is_full_snapshot=True)
        if poll_events:
            self._event_thread = threading.Thread(
                target=self._run_event_loop,
                name="kv_cache_reporter_events",
                daemon=True)
            self._event_thread.start()
        self._load_thread = threading.Thread(target=self._run_load_loop,
                                             name="kv_cache_reporter_load",
                                             daemon=True)
        self._load_thread.start()

    def report_events(self) -> None:
        """Drain and push pending events once, on the calling thread.

        For inline (non-polling) mode: call right after the engine flushes
        KV-cache events for an iteration. Safe to call from the executor loop
        thread -- the send is serialized with the load loop and no-ops after
        :meth:`stop`. Sends nothing when there are no pending events."""
        events = self._drain_events()
        if events:
            self._push_events(events)

    def stop(self) -> None:
        """Stop the loops and close the socket. Idempotent."""
        self._stop.set()
        for t in (self._event_thread, self._load_thread):
            if t is not None:
                t.join(timeout=5.0)
        self._event_thread = self._load_thread = None
        # Take the send lock so we never close the socket underneath an
        # in-flight send (report_events on the executor thread / load loop).
        with self._send_lock:
            if not self._closed:
                self._closed = True
                self._queue.close()

    # ----------------------------------------------------------------- loops

    def _run_event_loop(self) -> None:
        total_events_sent = 0
        total_reports_sent = 0
        empty_polls = 0
        last_log_time = time.monotonic()
        while not self._stop.is_set():
            try:
                events = self._drain_events()
                if events:
                    self._push_events(events)
                    total_events_sent += len(events)
                    total_reports_sent += 1
                    empty_polls = 0
                else:
                    empty_polls += 1
                # Log every 30s
                now = time.monotonic()
                if now - last_log_time >= 30.0:
                    logger.info(
                        f"WorkerReporter[{self._worker_id}] event_loop: "
                        f"reports_sent={total_reports_sent} "
                        f"events_sent={total_events_sent} "
                        f"seq={self._event_seq} "
                        f"empty_polls_since_last={empty_polls}")
                    last_log_time = now
                    empty_polls = 0
            except Exception as e:  # noqa: BLE001 - keep the daemon alive
                logger.error(f"WorkerReporter event loop error: {e}")
            self._stop.wait(self._event_interval_s)

    def _run_load_loop(self) -> None:
        while not self._stop.is_set():
            try:
                load = self._get_load()
                # get_load may return (active, queued) or
                # (active, queued, active_tokens); accept both for compat.
                num_active, num_queued = load[0], load[1]
                num_active_tokens = load[2] if len(load) > 2 else 0
                report = WorkerLoadReport(
                    worker_id=self._worker_id,
                    namespace=self._namespace,
                    seq=self._load_seq,
                    num_active_requests=num_active,
                    num_queued_requests=num_queued,
                    max_batch_size=self._max_batch_size,
                    num_active_tokens=num_active_tokens,
                )
                with self._send_lock:
                    if self._closed:
                        return
                    self._queue.put_noblock(report)
                self._load_seq += 1
            except Exception as e:  # noqa: BLE001 - keep the daemon alive
                logger.error(f"WorkerReporter load loop error: {e}")
            self._stop.wait(self._load_interval_s)

    # -------------------------------------------------------------- internals

    def _drain_events(self) -> List[dict]:
        """Pull and serialize the latest events from the KV-cache event manager."""
        raw = self._get_events(0)  # timeout_ms=0: non-blocking drain
        if not raw:
            return []
        # event_manager.get_latest_events() returns KVCacheEvent objects;
        # serialize to the same dict schema the /kv_cache_events endpoint uses.
        # If a caller already serialized them (list of dicts), pass through.
        if isinstance(raw[0], dict):
            return list(raw)
        return KVCacheEventSerializer.serialize(raw)

    def _push_events(self,
                     events: List[dict],
                     is_full_snapshot: bool = False) -> None:
        report = KvCacheEventReport(
            worker_id=self._worker_id,
            namespace=self._namespace,
            seq=self._event_seq,
            events=events,
            is_full_snapshot=is_full_snapshot,
            send_ts=_now_on_synced_clock(),
        )
        # Serialized with the load loop; no-op after stop() so a send racing
        # teardown never touches a closed (None) socket.
        with self._send_lock:
            if self._closed:
                return
            self._queue.put_noblock(report)
        self._event_seq += 1
