import bisect
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable, Iterator
from typing import Callable, Optional

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

    def add_request(self, request: RequestQueueItem) -> None:
        """Add a request to the queue according to FCFS policy."""
        self.append(request)

    def add_requests(self, requests: Iterable[RequestQueueItem]) -> None:
        """Add multiple requests to the queue according to FCFS policy."""
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

    Requests with equal priority are served in FCFS order.
    Priority is read from ``item.request.priority``; requests whose priority
    is ``None`` are treated as having the default priority of 0.5.
    """

    _DEFAULT_PRIORITY: float = 0.5

    def __init__(self) -> None:
        self._queue: list[RequestQueueItem] = []
        # Parallel list of sort keys: (neg_priority, insertion_counter).
        # bisect operates on this list so insertions stay O(log n).
        self._keys: list[tuple] = []
        self._insertion_counter: int = 0

    def _get_priority(self, item: RequestQueueItem) -> float:
        if item.request is not None and item.request.priority is not None:
            return float(item.request.priority)
        return self._DEFAULT_PRIORITY

    def _insert(self, item: RequestQueueItem) -> None:
        key = (-self._get_priority(item), self._insertion_counter)
        self._insertion_counter += 1
        idx = bisect.bisect_right(self._keys, key)
        self._keys.insert(idx, key)
        self._queue.insert(idx, item)

    def add_request(self, request: RequestQueueItem) -> None:
        self._insert(request)

    def add_requests(self, requests: Iterable[RequestQueueItem]) -> None:
        for request in requests:
            self._insert(request)

    def pop_request(self) -> RequestQueueItem:
        self._keys.pop(0)
        return self._queue.pop(0)

    def peek_request(self) -> RequestQueueItem:
        if not self._queue:
            raise IndexError("peek from an empty queue")
        return self._queue[0]

    def prepend_request(self, request: RequestQueueItem) -> None:
        """Re-insert the request at the position dictated by its priority."""
        self._insert(request)

    def prepend_requests(self, requests: Iterable[RequestQueueItem]) -> None:
        """Re-insert requests at the positions dictated by their priorities."""
        for request in requests:
            self._insert(request)

    def remove_by_ids(self, request_ids: set[int]) -> None:
        pairs = [(k, r) for k, r in zip(self._keys, self._queue) if r.id not in request_ids]
        if pairs:
            self._keys, self._queue = map(list, zip(*pairs))
        else:
            self._keys, self._queue = [], []

    def __bool__(self) -> bool:
        return len(self._queue) > 0

    def __len__(self) -> int:
        return len(self._queue)

    def __iter__(self) -> Iterator[RequestQueueItem]:
        return iter(self._queue)


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
