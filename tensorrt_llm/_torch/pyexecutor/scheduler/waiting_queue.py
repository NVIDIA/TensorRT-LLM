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


import heapq
import itertools
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable, Iterator

from tensorrt_llm.bindings.executor import RequestType
from tensorrt_llm.executor.request import DEFAULT_REQUEST_PRIORITY
from tensorrt_llm.llmapi.llm_args import WaitingQueuePolicy
from tensorrt_llm.logger import logger

from ..executor_request_queue import RequestQueueItem


class QueuedLoadTracker:
    """Incremental (requests, tokens) totals for the requests in a waiting queue.

    Queue-depth statistics are read on every sampled iteration, so the totals
    are maintained on insert/remove instead of rescanning the queue. Entries
    are keyed by request id and an id counts once however many queued items
    share it (its tokens are those of the first one added); it stops counting
    when the last of them leaves. Removing an absent id is ignored, so the
    put-back path (pop, then prepend) stays consistent. Items without a
    readable ``input_token_ids`` are not counted.
    """

    def __init__(self) -> None:
        # request id -> [is_generation_only, tokens, queued items with this id]
        self._entries: dict[int, list] = {}
        self.num_ctx_requests = 0
        self.num_ctx_tokens = 0
        self.num_gen_requests = 0
        self.num_gen_kv_tokens = 0

    def __contains__(self, request_id: int) -> bool:
        return request_id in self._entries

    def add(self, item: RequestQueueItem) -> None:
        if not item.is_normal_request or item.request is None:
            return
        entry = self._entries.get(item.id)
        if entry is not None:
            entry[2] += 1
            return
        try:
            tokens = int(item.request.num_input_tokens)
            is_gen = item.request.request_type == RequestType.REQUEST_TYPE_GENERATION_ONLY
        except (AttributeError, TypeError) as e:
            logger.warning(
                f"Excluding queued item {item.id} from queued counters: "
                f"input_token_ids not readable ({type(e).__name__})"
            )
            return
        self._entries[item.id] = [is_gen, tokens, 1]
        self._apply(is_gen, tokens, 1)

    def discard(self, request_id: int, *, all_items: bool = False) -> None:
        """Account for one queued item with this id leaving, or all of them."""
        entry = self._entries.get(request_id)
        if entry is None:
            return
        entry[2] -= 1
        if all_items or entry[2] <= 0:
            del self._entries[request_id]
            self._apply(entry[0], entry[1], -1)

    def _apply(self, is_gen: bool, tokens: int, sign: int) -> None:
        if is_gen:
            self.num_gen_requests += sign
            self.num_gen_kv_tokens += sign * tokens
        else:
            self.num_ctx_requests += sign
            self.num_ctx_tokens += sign * tokens

    def totals(self, excluded_ids: Iterable[int] = ()) -> tuple[int, int, int, int]:
        """Return (ctx_requests, ctx_tokens, gen_requests, gen_kv_tokens).

        ``excluded_ids`` (e.g. cancelled requests) are subtracted without
        mutating the tracker; the cost is O(len(excluded_ids)), not O(queue).
        """
        ctx_req, ctx_tok = self.num_ctx_requests, self.num_ctx_tokens
        gen_req, gen_tok = self.num_gen_requests, self.num_gen_kv_tokens
        for request_id in set(excluded_ids):
            entry = self._entries.get(request_id)
            if entry is None:
                continue
            if entry[0]:
                gen_req -= 1
                gen_tok -= entry[1]
            else:
                ctx_req -= 1
                ctx_tok -= entry[1]
        return ctx_req, ctx_tok, gen_req, gen_tok


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
    def peek_requests(self, k: int) -> list[RequestQueueItem]:
        """Return up to ``k`` requests in pop order without removing them.

        Implementations must not copy or sort the whole queue for a small ``k``.
        """
        pass

    def admission_groups(self, requests: list[RequestQueueItem]) -> list[int]:
        """Admission group of each request from ``peek_requests``, numbered in pop order.

        Out-of-order admission must finish a group before the next one. Default:
        one group.
        """
        return [0] * len(requests)

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

    def mark_admitted(self, requests: Iterable[RequestQueueItem]) -> None:
        """Called with the popped requests that were admitted, not put back."""

    @abstractmethod
    def remove_by_ids(self, request_ids: set[int]) -> None:
        """Remove requests with the given IDs."""
        pass

    queued_load: QueuedLoadTracker

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
    """A first-come-first-served queue that supports deque operations.

    Mutate it only through the WaitingQueue methods: raw deque calls bypass
    ``queued_load``.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.queued_load = QueuedLoadTracker()
        # deque's own iterator: a subclass may override __iter__.
        for item in deque.__iter__(self):
            self.queued_load.add(item)

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
        self.queued_load.add(request)

    def add_requests(self, requests: Iterable[RequestQueueItem]) -> None:
        """Add multiple requests to the queue according to FCFS policy."""
        requests = list(requests)
        for request in requests:
            self._warn_if_priority_set(request)
        self.extend(requests)
        for request in requests:
            self.queued_load.add(request)

    def pop_request(self) -> RequestQueueItem:
        """Pop a request from the queue according to FCFS policy."""
        request = self.popleft()
        self.queued_load.discard(request.id)
        return request

    def peek_request(self) -> RequestQueueItem:
        """Peek at the next request in the queue without removing it."""
        if not self:
            raise IndexError("peek from an empty queue")
        return self[0]

    def peek_requests(self, k: int) -> list[RequestQueueItem]:
        """Return up to ``k`` requests from the front of the queue, O(k)."""
        return list(itertools.islice(self, max(k, 0)))

    def prepend_request(self, request: RequestQueueItem) -> None:
        """Prepend a request to the front of the queue."""
        self.appendleft(request)
        self.queued_load.add(request)

    def prepend_requests(self, requests: Iterable[RequestQueueItem]) -> None:
        """Prepend all requests from another iterable to the front of this queue.

        Requests are inserted in input order: the first item in ``requests``
        ends up at the front of the queue.
        """
        requests = list(requests)
        self.extendleft(reversed(requests))
        for request in requests:
            self.queued_load.add(request)

    def remove_by_ids(self, request_ids: set[int]) -> None:
        """Remove requests with the given IDs."""
        filtered_requests = [req for req in self if req.id not in request_ids]
        self.clear()
        self.extend(filtered_requests)
        for request_id in request_ids:
            self.queued_load.discard(request_id, all_items=True)

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
        - peek_requests(k): O(n log k), without copying or sorting the heap
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
        self.queued_load = QueuedLoadTracker()

    def _get_priority(self, item: RequestQueueItem) -> float:
        if item.request is not None:
            return item.request.priority
        return DEFAULT_REQUEST_PRIORITY

    def _push(self, item: RequestQueueItem) -> None:
        entry = (-self._get_priority(item), next(self._counter), item)
        heapq.heappush(self._heap, entry)
        self.queued_load.add(item)

    def _push_front(self, item: RequestQueueItem) -> None:
        entry = (-self._get_priority(item), next(self._prepend_counter), item)
        heapq.heappush(self._heap, entry)
        self.queued_load.add(item)

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
        self.queued_load.discard(item.id)
        return item

    def peek_request(self) -> RequestQueueItem:
        """Return the highest-priority request without removing it."""
        if not self._heap:
            raise IndexError("peek from an empty queue")
        return self._heap[0][2]

    def peek_requests(self, k: int) -> list[RequestQueueItem]:
        """Return up to ``k`` requests in priority order without removing them."""
        return [e[2] for e in heapq.nsmallest(max(k, 0), self._heap)]

    def admission_groups(self, requests: list[RequestQueueItem]) -> list[int]:
        """One group per priority."""
        groups = []
        group = 0
        for i, request in enumerate(requests):
            if i and self._get_priority(request) != self._get_priority(requests[i - 1]):
                group += 1
            groups.append(group)
        return groups

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
        for request_id in request_ids:
            self.queued_load.discard(request_id, all_items=True)

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


def create_waiting_queue(
    policy: WaitingQueuePolicy = WaitingQueuePolicy.FCFS,
    gen_admission=None,
) -> WaitingQueue:
    """Create a waiting queue based on the specified policy.

    Args:
        policy: The scheduling policy to use.
        gen_admission: Optional ``GenAdmissionConfig``; overrides ``policy`` when enabled.

    Returns:
        A WaitingQueue instance.
    """
    if gen_admission is not None and gen_admission.enabled:
        from .gen_admission_queue import GenAdmissionWaitingQueue

        if policy != WaitingQueuePolicy.FCFS:
            logger.warning(
                f"[gen_admission] {gen_admission.describe()} overrides "
                f"waiting_queue_policy={policy}; request priorities are ignored."
            )
        return GenAdmissionWaitingQueue(gen_admission)
    if policy == WaitingQueuePolicy.FCFS:
        return FCFSWaitingQueue()
    elif policy == WaitingQueuePolicy.PRIORITY:
        return PriorityWaitingQueue()
    else:
        raise ValueError(f"Unsupported waiting queue policy: {policy}")
