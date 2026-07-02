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

from .messages import KvCacheEventReport

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
    """Pushes KV-cache event reports from one worker instance.

    Load is no longer reported: the coordinator tracks routed-request load itself
    (shared LoadBalancingMixin counter), so the worker only pushes KV-cache events.

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
        event_interval_s: Deprecated/ignored. Events are reported inline at each
            engine iteration flush (see :meth:`report_events`), not polled.
    """

    def __init__(self,
                 worker_id: str,
                 namespace: str,
                 router_address: str,
                 hmac_key: Optional[bytes],
                 get_events: Callable[[float], list],
                 event_interval_s: float = 0.01) -> None:
        self._worker_id = worker_id
        self._namespace = namespace
        self._get_events = get_events
        # event_interval_s is accepted for backward compatibility but ignored:
        # events are reported inline at each engine iteration flush (see
        # report_events), not polled on a timer.

        self._queue = ZeroMqQueue(
            address=(router_address, hmac_key),
            socket_type=zmq.PUSH,
            is_server=False,
            name=f"kv_cache_reporter[{worker_id}]",
        )
        self._event_seq = 0
        self._stop = threading.Event()
        self._started = False
        # The ZMQ socket is not thread-safe; report_events (executor loop) and
        # stop() may race. Serialize sends and gate on _closed so a send racing
        # stop() no-ops instead of hitting a closed (None) socket.
        self._send_lock = threading.Lock()
        self._closed = False

    def start(self) -> None:
        """Send the initial event snapshot. Events are then reported inline via
        :meth:`report_events` (called at the engine's per-iteration KV-event
        flush), so there is no background thread."""
        if self._started:
            raise RuntimeError("WorkerReporter already started")
        self._started = True
        # Initial snapshot lets a (re)started router rebuild this worker's table.
        self._push_events(self._drain_events(), is_full_snapshot=True)

    def report_events(self) -> None:
        """Drain and push pending events once, on the calling thread.

        Call at the engine's per-iteration KV-event flush. Safe from the executor
        loop thread; no-ops after :meth:`stop` or when nothing is pending."""
        events = self._drain_events()
        if events:
            self._push_events(events)

    def stop(self) -> None:
        """Close the socket. Idempotent."""
        self._stop.set()
        # Take the send lock so we never close the socket underneath an
        # in-flight send (report_events on the executor thread).
        with self._send_lock:
            if not self._closed:
                self._closed = True
                self._queue.close()

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
