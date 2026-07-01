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
"""Wire messages exchanged between TRT-LLM workers and the centralized KV-cache router.

Two report types flow over the same ZMQ channel, each keyed by the worker's
instance UUID (``worker_id``), a routing ``namespace``, and a per-stream
monotonic ``seq`` used to drop stale / out-of-order messages:

* :class:`KvCacheEventReport` -- KV-cache block deltas (``stored`` / ``removed``),
  high frequency, event-driven. Carries the same dict payload that
  ``KVCacheEventSerializer`` already produces for the ``/kv_cache_events`` HTTP
  endpoint, so no new schema is introduced.
* :class:`WorkerLoadReport` -- a workload snapshot (active + queued requests),
  low frequency, state-replacing.

Routing is at instance granularity; attention-DP ranks report as a single
gathered instance (see the design doc).
"""

from dataclasses import dataclass, field
from typing import List, Optional

__all__ = [
    "KvCacheEventReport",
    "WorkerLoadReport",
    "Selection",
]


@dataclass
class KvCacheEventReport:
    """A batch of KV-cache events emitted by one worker instance.

    Attributes:
        worker_id: Instance UUID, identical to the value advertised via
            ``GET /server_info``.
        namespace: Routing group this worker serves (e.g. ``"ctx"`` / ``"gen"``
            for disaggregated serving, or a model / tenant name).
        seq: Monotonically increasing sequence number for this worker's event
            stream. The router drops any report whose ``seq`` is not newer than
            the last applied one.
        events: List of event dicts in ``KVCacheEventSerializer`` form, e.g.
            ``[{"event_id": .., "data": {"type": "stored"|"removed", ..},
            "window_size": ..}, ..]``.
        is_full_snapshot: When ``True`` the report replaces the worker's entire
            block table before applying ``events`` (used on (re)registration and
            for resync after a detected gap).
    """

    worker_id: str
    namespace: str
    seq: int
    events: List[dict] = field(default_factory=list)
    is_full_snapshot: bool = False
    # Steady-clock timestamp (seconds) when the worker enqueued this report,
    # on the disagg-synced timebase shared with the router. Lets the router
    # measure worker->router propagation lag. 0.0 if the worker did not stamp.
    send_ts: float = 0.0


@dataclass
class WorkerLoadReport:
    """A periodic workload snapshot for one worker instance.

    Attributes:
        worker_id: Instance UUID.
        namespace: Routing group this worker serves.
        seq: Monotonically increasing sequence number for this worker's load
            stream (independent of the event stream's ``seq``).
        num_active_requests: Requests currently running on the instance
            (summed across attention-DP ranks).
        num_queued_requests: Queued context + generation requests.
        max_batch_size: Engine ``max_batch_size``, used to normalize load in the
            routing score.
    """

    worker_id: str
    namespace: str
    seq: int
    num_active_requests: int = 0
    num_queued_requests: int = 0
    max_batch_size: int = 1
    # Active KV tokens on this worker/rank (cache-discounted compute load), the
    # metric the worker KVCacheAwareADPRouter scores on. 0 if the reporter does
    # not supply it (older reporters) -> consumers fall back to request count.
    num_active_tokens: int = 0


@dataclass
class Selection:
    """Result of a routing query.

    Attributes:
        worker_id: The chosen instance id (the worker's ``llm_id``).
        address: The chosen worker's server address, resolved from the
            ``worker_id -> address`` mapping the router learns from
            ``/server_info``. ``None`` if the worker reported KV events but its
            address has not been registered yet (the caller should fall back).
        matched_blocks: Number of KV-cache blocks of the request's prefix already
            cached on the chosen worker (for logging / disagg transfer decisions).
    """

    worker_id: str
    address: Optional[str] = None
    matched_blocks: int = 0
    dp_rank: Optional[int] = None
