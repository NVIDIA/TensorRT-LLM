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
"""ZMQ ingest server for the centralized KV-cache router.

Binds a single PULL socket (fan-in) and dispatches each received report to the
:class:`CentralizedKVCacheRouter` by type. Reuses ``ZeroMqQueue`` verbatim, so it
inherits HMAC-signed pickle transport. The bound endpoint + HMAC key are exposed
via :attr:`address` to hand to workers out-of-band.
"""

import threading
from typing import Optional, Tuple

import zmq

from tensorrt_llm.executor.ipc import ZeroMqQueue
from tensorrt_llm.logger import logger

from .messages import KvCacheEventReport, WorkerLoadReport
from .router_core import CentralizedKVCacheRouter

__all__ = ["KVCacheRouterServer"]


class KVCacheRouterServer:
    """Background PULL-socket loop feeding a :class:`CentralizedKVCacheRouter`.

    Args:
        router: The router state to apply reports to.
        address: Optional ``"tcp://host:port"`` to bind. Defaults to an
            ephemeral local port (``tcp://127.0.0.1:*``).
        hmac_key: Optional pre-shared HMAC key; a fresh one is generated if omitted.
        evict_interval_s: How often to run stale-worker eviction, in seconds.
    """

    def __init__(self,
                 router: CentralizedKVCacheRouter,
                 address: Optional[str] = None,
                 hmac_key: Optional[bytes] = None,
                 evict_interval_s: float = 5.0) -> None:
        self._router = router
        self._evict_interval_s = evict_interval_s
        self._queue = ZeroMqQueue(
            address=(address, hmac_key) if address is not None else None,
            socket_type=zmq.PULL,
            is_server=True,
            name="kv_cache_router_ingest",
        )
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_evict = 0.0

    @property
    def address(self) -> Tuple[str, Optional[bytes]]:
        """``(endpoint, hmac_key)`` to distribute to worker reporters."""
        # Touch the socket so it binds and the ephemeral port is resolved.
        self._queue.setup_lazily()
        return self._queue.address

    def start(self) -> None:
        """Start the ingest loop on a background daemon thread."""
        if self._thread is not None:
            raise RuntimeError("KVCacheRouterServer already started")
        self._thread = threading.Thread(target=self._serve,
                                        name="kv_cache_router_ingest",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the loop to stop and join the thread."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._queue.close()

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                # Poll so the loop stays responsive to stop() and eviction.
                if not self._queue.poll(timeout=1):
                    self._maybe_evict()
                    continue
                for report in self._queue.drain():
                    self._dispatch(report)
                self._maybe_evict()
            except Exception as e:  # noqa: BLE001 - keep the daemon alive
                logger.error(f"KVCacheRouterServer ingest error: {e}")

    def _dispatch(self, report) -> None:
        if isinstance(report, KvCacheEventReport):
            self._router.apply_event_report(report)
        elif isinstance(report, WorkerLoadReport):
            self._router.apply_load_report(report)
        else:
            logger.warning(
                f"KVCacheRouterServer: ignoring unknown report type "
                f"{type(report).__name__}")

    def _maybe_evict(self) -> None:
        now = self._router._clock()
        if now - self._last_evict >= self._evict_interval_s:
            self._router.evict_stale_workers(now)
            self._last_evict = now
