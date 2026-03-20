import heapq
import itertools
import warnings
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable, Iterator
from typing import Callable, Optional

from tensorrt_llm.executor.request import DEFAULT_REQUEST_PRIORITY
from tensorrt_llm.llmapi.llm_args import WaitingQueuePolicy

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
        """Prepend a request to the front of the queue."""
        pass

    @abstractmethod
    def prepend_requests(self, requests: Iterable[RequestQueueItem]) -> None:
        """Prepend all requests from another iterable to the front of this queue."""
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
            warnings.warn(
                "A request has a non-default priority but the FCFS waiting "
                "queue is in use; the priority value will be ignored. "
                "Use WaitingQueuePolicy.PRIORITY to enable priority scheduling.",
                UserWarning,
                stacklevel=3,
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

        Note: The requests will be prepended in reverse order of their
        appearance in the `requests` iterable.
        """
        self.extendleft(requests)

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
    Priority is read from ``item.request.priority``; the default is 0.5.

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
        # itertools.count() is thread-safe (backed by a C-level atomic),
        # ensuring correct FCFS tiebreaking under concurrent access.
        self._counter = itertools.count()

    def _get_priority(self, item: RequestQueueItem) -> float:
        if item.request is not None:
            return item.request.priority
        return DEFAULT_REQUEST_PRIORITY

    def _push(self, item: RequestQueueItem) -> None:
        entry = (-self._get_priority(item), next(self._counter), item)
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
        """Re-insert the request at the position dictated by its priority."""
        self._push(request)

    def prepend_requests(self, requests: Iterable[RequestQueueItem]) -> None:
        """Re-insert requests at the positions dictated by their priorities."""
        for request in requests:
            self._push(request)

    def remove_by_ids(self, request_ids: set[int]) -> None:
        """Remove requests with the given IDs."""
        self._heap = [e for e in self._heap if e[2].id not in request_ids]
        heapq.heapify(self._heap)

    def __bool__(self) -> bool:
        return len(self._heap) > 0

    def __len__(self) -> int:
        return len(self._heap)

    def __iter__(self) -> Iterator[RequestQueueItem]:
        """Iterate over requests in descending priority order."""
        return (e[2] for e in sorted(self._heap))


def create_waiting_queue(
    policy: WaitingQueuePolicy = WaitingQueuePolicy.FCFS,
    priority_fn: Optional[Callable[[RequestQueueItem], float]] = None,
) -> WaitingQueue:
    """Create a waiting queue based on the specified policy.

    Args:
        policy: The scheduling policy to use.
        priority_fn: Reserved for future use.

    Returns:
        A WaitingQueue instance.
    """
    if policy == WaitingQueuePolicy.FCFS:
        return FCFSWaitingQueue()
    elif policy == WaitingQueuePolicy.PRIORITY:
        return PriorityWaitingQueue()
    else:
        raise ValueError(f"Unsupported waiting queue policy: {policy}")
