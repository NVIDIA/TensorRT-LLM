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

    def start(self) -> None:
        """Send the initial full snapshot and start both report loops."""
        if self._event_thread is not None:
            raise RuntimeError("WorkerReporter already started")
        # Initial snapshot lets a (re)started router rebuild this worker's table.
        self._push_events(self._drain_events(), is_full_snapshot=True)
        self._event_thread = threading.Thread(target=self._run_event_loop,
                                              name="kv_cache_reporter_events",
                                              daemon=True)
        self._load_thread = threading.Thread(target=self._run_load_loop,
                                             name="kv_cache_reporter_load",
                                             daemon=True)
        self._event_thread.start()
        self._load_thread.start()

    def stop(self) -> None:
        """Stop both loops and close the socket."""
        self._stop.set()
        for t in (self._event_thread, self._load_thread):
            if t is not None:
                t.join(timeout=5.0)
        self._event_thread = self._load_thread = None
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
                num_active, num_queued = self._get_load()
                self._queue.put_noblock(
                    WorkerLoadReport(
                        worker_id=self._worker_id,
                        namespace=self._namespace,
                        seq=self._load_seq,
                        num_active_requests=num_active,
                        num_queued_requests=num_queued,
                        max_batch_size=self._max_batch_size,
                    ))
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
        self._queue.put_noblock(
            KvCacheEventReport(
                worker_id=self._worker_id,
                namespace=self._namespace,
                seq=self._event_seq,
                events=events,
                is_full_snapshot=is_full_snapshot,
            ))
        self._event_seq += 1
