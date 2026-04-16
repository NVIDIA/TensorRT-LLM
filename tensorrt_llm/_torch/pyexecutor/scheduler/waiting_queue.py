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


import dataclasses
import heapq
import itertools
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable, Iterator

from tensorrt_llm.executor.request import DEFAULT_REQUEST_PRIORITY
from tensorrt_llm.llmapi.llm_args import WaitingQueuePolicy
from tensorrt_llm.logger import logger

from ..executor_request_queue import RequestQueueItem


class WaitingQueue(ABC):
    """Abstract base class for waiting queues."""

    @abstractmethod
    def add_request(self, request: RequestQueueItem) -> None:
        """Add a request to the queue according to the policy."""
        pass

    @abstractmethod
    def add_requests(self, requests: Iterable[RequestQueueItem]) -> None:
        """Add multiple requests to the queue according to the policy."""
        pass

    @abstractmethod
    def pop_request(self) -> RequestQueueItem:
        """Pop a request from the queue according to the policy."""
        pass

    @abstractmethod
    def peek_request(self) -> RequestQueueItem:
        """Peek at the request at the front of the queue without removing it."""
        pass

    @abstractmethod
    def prepend_request(self, request: RequestQueueItem) -> None:
        """Re-insert a request that could not be scheduled.

        Implementations must ensure the request is served before any request
        (of equal priority in the case of a priority queue) that was added
        *after* it originally arrived (i.e. it does not lose
        its place to later-arriving requests of equal priority).  For FCFS
        queues this means inserting at the front; for priority queues this
        means using a tiebreaker that sorts ahead of normally-added entries.
        """
        pass

    @abstractmethod
    def prepend_requests(self, requests: Iterable[RequestQueueItem]) -> None:
        """Re-insert multiple requests that could not be scheduled.

        See prepend_request for the ordering contract.  Requests are
        re-inserted such that the first request in the iterable comes out
        first after re-insertion (i.e. original queue order is preserved).
        """
        pass

    @abstractmethod
    def remove_by_ids(self, request_ids: set[int]) -> None:
        """Remove requests with the given IDs."""
        pass

    @abstractmethod
    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Get number of requests in queue."""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[RequestQueueItem]:
        """Iterate over the queue according to the policy."""
        pass


class FCFSWaitingQueue(deque, WaitingQueue):
    """A first-come-first-served queue that supports deque operations."""

    @staticmethod
    def _warn_if_priority_set(request: RequestQueueItem) -> None:
        if request.request is not None and request.request.priority != DEFAULT_REQUEST_PRIORITY:
            logger.warning(
                "A request has a non-default priority but the FCFS waiting "
                "queue is in use; the priority value will be ignored. "
                "Use WaitingQueuePolicy.PRIORITY to enable priority scheduling."
            )

    def add_request(self, request: RequestQueueItem) -> None:
        """Add a request to the queue according to FCFS policy."""
        self._warn_if_priority_set(request)
        self.append(request)

    def add_requests(self, requests: Iterable[RequestQueueItem]) -> None:
        """Add multiple requests to the queue according to FCFS policy."""
        requests = list(requests)
        for request in requests:
            self._warn_if_priority_set(request)
        self.extend(requests)

    def pop_request(self) -> RequestQueueItem:
        """Pop a request from the queue according to FCFS policy."""
        return self.popleft()

    def peek_request(self) -> RequestQueueItem:
        """Peek at the next request in the queue without removing it."""
        if not self:
            raise IndexError("peek from an empty queue")
        return self[0]

    def prepend_request(self, request: RequestQueueItem) -> None:
        """Prepend a request to the front of the queue."""
        self.appendleft(request)

    def prepend_requests(self, requests: Iterable[RequestQueueItem]) -> None:
        """Prepend all requests from another iterable to the front of this queue.

        Requests are inserted in input order: the first item in ``requests``
        ends up at the front of the queue.
        """
        self.extendleft(reversed(list(requests)))

    def remove_by_ids(self, request_ids: set[int]) -> None:
        """Remove requests with the given IDs."""
        filtered_requests = [req for req in self if req.id not in request_ids]
        self.clear()
        self.extend(filtered_requests)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return len(self) > 0

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return super().__len__()

    def __iter__(self) -> Iterator[RequestQueueItem]:
        """Iterate over the queue according to FCFS policy."""
        return super().__iter__()


class PriorityWaitingQueue(WaitingQueue):
    """A priority queue that serves higher-priority requests first.

    Requests with equal priority are served in FCFS order (arrival order).
    Priority is read from ``item.request.priority``; the default is DEFAULT_REQUEST_PRIORITY.

    Heap entries are ``(neg_priority, insertion_counter, item)`` tuples.
    Because ``insertion_counter`` is strictly monotonically increasing, no two
    entries share the same first two elements, so ``item`` is never compared
    and does not need to implement ``__lt__``.

    Complexity:
        - add_request / add_requests: O(log n)
        - pop_request / peek_request: O(log n) / O(1)
        - remove_by_ids: O(n) rebuild + O(n) heapify
        - __iter__: O(n log n) over a sorted copy (priority order)
    """

    def __init__(self) -> None:
        # Min-heap of (neg_priority, insertion_counter, RequestQueueItem).
        self._heap: list[tuple] = []
        # _counter assigns strictly increasing non-negative integers to
        # normally-added requests for FCFS tiebreaking within a priority cohort.
        # _prepend_counter assigns strictly decreasing negative integers to
        # prepended requests, guaranteeing they sort before any normally-added
        # request of the same priority.
        self._counter = itertools.count()
        self._prepend_counter = itertools.count(-1, -1)

    def _get_priority(self, item: RequestQueueItem) -> float:
        if item.request is not None:
            return item.request.priority
        return DEFAULT_REQUEST_PRIORITY

    def _push(self, item: RequestQueueItem) -> None:
        entry = (-self._get_priority(item), next(self._counter), item)
        heapq.heappush(self._heap, entry)

    def _push_front(self, item: RequestQueueItem) -> None:
        entry = (-self._get_priority(item), next(self._prepend_counter), item)
        heapq.heappush(self._heap, entry)

    def add_request(self, request: RequestQueueItem) -> None:
        """Add a request to the queue in priority order."""
        self._push(request)

    def add_requests(self, requests: Iterable[RequestQueueItem]) -> None:
        """Add multiple requests to the queue in priority order."""
        for request in requests:
            self._push(request)

    def pop_request(self) -> RequestQueueItem:
        """Pop the highest-priority request (FCFS tiebreak)."""
        if not self._heap:
            raise IndexError("pop from an empty queue")
        _, _, item = heapq.heappop(self._heap)
        return item

    def peek_request(self) -> RequestQueueItem:
        """Return the highest-priority request without removing it."""
        if not self._heap:
            raise IndexError("peek from an empty queue")
        return self._heap[0][2]

    def prepend_request(self, request: RequestQueueItem) -> None:
        """Re-insert a request ahead of all normally-added requests of the same priority.

        Uses a strictly decreasing counter so the request sorts before any
        request added via add_request / add_requests, preserving FCFS order
        within a priority cohort when requests are returned to the queue.
        """
        self._push_front(request)

    def prepend_requests(self, requests: Iterable[RequestQueueItem]) -> None:
        """Re-insert requests ahead of all normally-added requests of the same priority.

        Requests are pushed in *reverse* iteration order so that the first
        item in ``requests`` ends up with the most-negative prepend counter
        and therefore sorts first within its priority cohort.  This means
        the first request in the iterable comes out first after re-insertion,
        preserving original queue order without any reversal by the caller.
        """
        for request in reversed(list(requests)):
            self._push_front(request)

    def remove_by_ids(self, request_ids: set[int]) -> None:
        """Remove requests with the given IDs."""
        self._heap = [e for e in self._heap if e[2].id not in request_ids]
        heapq.heapify(self._heap)

    def __bool__(self) -> bool:
        return len(self._heap) > 0

    def __len__(self) -> int:
        return len(self._heap)

    def __iter__(self) -> Iterator[RequestQueueItem]:
        """Iterate over requests in descending priority order.

        Returns a generator over a snapshot of the heap taken at call time.
        Modifications to the queue after iteration begins are not reflected.
        """
        return (e[2] for e in sorted(self._heap))


@dataclasses.dataclass
class SJFEntry:
    """Internal entry for SJFWaitingQueue heap."""

    compute_tokens: int  # prompt_len - cached_len (or prompt_len)
    enqueue_time: float  # time.monotonic() at enqueue
    counter: int  # insertion order tiebreaker
    item: RequestQueueItem  # the actual request


class SJFWaitingQueue(WaitingQueue):
    """Shortest Job First queue with optional cache-aware sorting and aging.

    Requests with fewer estimated compute tokens are served first.
    Aging progressively reduces a request's priority score over time
    to prevent starvation of long requests.

    Priority formula:
        score = compute_tokens * max(0, 1 - aging_factor * elapsed_time)

    Lower score = higher priority. Ties broken by insertion order (FCFS).

    Implementation: min-heap of (score, counter, SJFEntry) tuples.
    Since aging makes scores time-dependent, the heap is rebuilt via
    heapify() before each pop/peek. When aging_factor=0, scores are
    static and reheapify is skipped.
    """

    def __init__(
        self,
        kv_cache_manager=None,
        cache_aware: bool = True,
        aging_factor: float = 0.005,
    ):
        self._heap: list[tuple[float, int, SJFEntry]] = []
        self._kv_cache_manager = kv_cache_manager
        self._cache_aware = cache_aware and kv_cache_manager is not None
        self._aging_factor = aging_factor
        self._counter = itertools.count()
        self._prepend_counter = itertools.count(-1, -1)
        # Track original enqueue times so prepended requests retain their
        # accumulated aging benefit instead of resetting the clock.
        self._original_enqueue_times: dict[int, float] = {}

    # ---- Priority computation ----

    def _estimate_compute_tokens(self, req_item: RequestQueueItem) -> int:
        """Estimate actual prefill compute tokens for a request."""
        req = req_item.request
        if req is None:
            return 0
        input_tokens = getattr(req, "input_token_ids", None) or []
        prompt_len = len(input_tokens)

        if self._cache_aware:
            probe_tokens = input_tokens[:-1] if len(input_tokens) > 1 else []
            lora_config = getattr(req, "lora_config", None)
            lora_task_id = lora_config.task_id if lora_config is not None else None
            cached_len = self._kv_cache_manager.probe_prefix_match_length(
                probe_tokens, lora_task_id
            )
            return max(prompt_len - cached_len, 0)

        return prompt_len

    def _compute_score(self, entry: SJFEntry, now: float) -> float:
        """Compute dynamic priority score. Lower = more urgent."""
        elapsed = now - entry.enqueue_time
        decay = max(0.0, 1.0 - self._aging_factor * elapsed)
        return entry.compute_tokens * decay

    def _reheapify(self) -> None:
        """Rebuild heap with current dynamic scores.

        Skipped when aging_factor=0 (scores are static).
        """
        if self._aging_factor == 0:
            return
        now = time.monotonic()
        self._heap = [
            (self._compute_score(entry, now), entry.counter, entry) for _, _, entry in self._heap
        ]
        heapq.heapify(self._heap)

    # ---- Heap push helpers ----

    def _push(self, entry: SJFEntry) -> None:
        """Push an entry with its current score."""
        now = time.monotonic()
        score = self._compute_score(entry, now)
        heapq.heappush(self._heap, (score, entry.counter, entry))

    # ---- WaitingQueue interface ----

    def add_request(self, request: RequestQueueItem) -> None:
        compute_tokens = self._estimate_compute_tokens(request)
        now = time.monotonic()
        entry = SJFEntry(
            compute_tokens=compute_tokens,
            enqueue_time=now,
            counter=next(self._counter),
            item=request,
        )
        self._original_enqueue_times[request.id] = now
        self._push(entry)

    def add_requests(self, requests: Iterable[RequestQueueItem]) -> None:
        for request in requests:
            self.add_request(request)

    def pop_request(self) -> RequestQueueItem:
        if not self._heap:
            raise IndexError("pop from an empty queue")
        self._reheapify()
        _, _, entry = heapq.heappop(self._heap)
        return entry.item

    def peek_request(self) -> RequestQueueItem:
        if not self._heap:
            raise IndexError("peek from an empty queue")
        self._reheapify()
        return self._heap[0][2].item

    def prepend_request(self, request: RequestQueueItem) -> None:
        """Re-insert a request that could not be scheduled.

        Uses negative counter to sort ahead of normally-added entries
        with the same score.  Preserves the original enqueue_time so
        accumulated aging benefit is not lost.  compute_tokens is
        re-estimated because cache state may have changed.
        """
        compute_tokens = self._estimate_compute_tokens(request)
        enqueue_time = self._original_enqueue_times.get(request.id, time.monotonic())
        entry = SJFEntry(
            compute_tokens=compute_tokens,
            enqueue_time=enqueue_time,
            counter=next(self._prepend_counter),
            item=request,
        )
        self._push(entry)

    def prepend_requests(self, requests: Iterable[RequestQueueItem]) -> None:
        for request in reversed(list(requests)):
            self.prepend_request(request)

    def remove_by_ids(self, request_ids: set[int]) -> None:
        self._heap = [e for e in self._heap if e[2].item.id not in request_ids]
        heapq.heapify(self._heap)
        for rid in request_ids:
            self._original_enqueue_times.pop(rid, None)

    def __bool__(self) -> bool:
        return len(self._heap) > 0

    def __len__(self) -> int:
        return len(self._heap)

    def __iter__(self) -> Iterator[RequestQueueItem]:
        """Iterate in priority order (snapshot)."""
        self._reheapify()
        return (e[2].item for e in sorted(self._heap))


def create_waiting_queue(
    config=None,
    kv_cache_manager=None,
) -> WaitingQueue:
    """Create a waiting queue from a :class:`WaitingQueueConfig`.

    Args:
        config: Waiting queue configuration.  ``None`` uses FCFS default.
        kv_cache_manager: KV cache manager for cache-aware SJF.

    Returns:
        A WaitingQueue instance.
    """
    from tensorrt_llm.llmapi.llm_args import SJFConfig, WaitingQueueConfig

    if config is None:
        config = WaitingQueueConfig()

    policy = config.policy
    logger.info(f"Creating waiting queue with policy: {policy}")
    if policy == WaitingQueuePolicy.FCFS:
        return FCFSWaitingQueue()
    elif policy == WaitingQueuePolicy.PRIORITY:
        return PriorityWaitingQueue()
    elif policy == WaitingQueuePolicy.SJF:
        sjf = config.sjf or SJFConfig()
        logger.info(f"SJF config: cache_aware={sjf.cache_aware}, aging_factor={sjf.aging_factor}")
        return SJFWaitingQueue(
            kv_cache_manager=kv_cache_manager,
            cache_aware=sjf.cache_aware,
            aging_factor=sjf.aging_factor,
        )
    else:
        raise ValueError(f"Unsupported waiting queue policy: {policy}")
