import heapq
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable, Iterator
from typing import Callable, Optional

from tensorrt_llm.llmapi.llm_args import WaitingQueuePolicy

from .executor_request_queue import RequestQueueItem


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
        filtered_requests = [
            req for req in self if req.id not in request_ids
        ]
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
    """A priority queue that supports heap operations.

    Requests are ordered by priority (smaller value = higher priority),
    with insertion order as tiebreaker (FIFO for same priority).
    """

    def __init__(
            self,
            priority_fn: Optional[Callable[[RequestQueueItem], float]] = None
    ) -> None:
        """Initialize the priority queue.

        Args:
            priority_fn: Function to compute priority for a request.
                         Smaller values = higher priority.
                         If None, uses request.id as priority (FIFO behavior).
        """
        self._heap: list[tuple[float, int, RequestQueueItem]] = []
        self._priority_fn = priority_fn or (lambda req: req.id)
        self._counter = 0  # For stable sorting (FIFO tiebreaker)

    def add_request(self, request: RequestQueueItem) -> None:
        """Add a request to the queue according to priority policy."""
        priority = self._priority_fn(request)
        heapq.heappush(self._heap, (priority, self._counter, request))
        self._counter += 1

    def add_requests(self, requests: Iterable[RequestQueueItem]) -> None:
        """Add multiple requests to the queue according to priority policy."""
        for request in requests:
            self.add_request(request)

    def pop_request(self) -> RequestQueueItem:
        """Pop a request from the queue according to priority policy."""
        if not self._heap:
            raise IndexError("pop from empty priority queue")
        _, _, request = heapq.heappop(self._heap)
        return request

    def peek_request(self) -> RequestQueueItem:
        """Peek at the next request in the queue without removing it."""
        if not self._heap:
            raise IndexError("peek from empty priority queue")
        return self._heap[0][2]

    def prepend_request(self, request: RequestQueueItem) -> None:
        """Add a request to the queue according to priority policy.

        Note: In a priority queue, there is no concept of prepending to the
        front. Requests are ordered by priority.
        """
        self.add_request(request)

    def prepend_requests(self, requests: Iterable[RequestQueueItem]) -> None:
        """Add all requests from another iterable according to priority policy.

        Note: In a priority queue, there is no concept of prepending to the
        front. Requests are ordered by priority.
        """
        for request in requests:
            self.add_request(request)

    def remove_by_ids(self, request_ids: set[int]) -> None:
        """Remove requests with the given IDs."""
        self._heap = [(p, c, r) for p, c, r in self._heap
                      if r.id not in request_ids]
        heapq.heapify(self._heap)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return bool(self._heap)

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return len(self._heap)

    def __iter__(self) -> Iterator[RequestQueueItem]:
        """Iterate over the queue according to priority policy."""
        heap_copy = self._heap[:]
        while heap_copy:
            _, _, request = heapq.heappop(heap_copy)
            yield request


def _priority_by_request_priority(req: RequestQueueItem) -> float:
    """Priority function: by request's priority parameter (smaller = higher)."""
    priority = getattr(req.request, "priority", None)
    if priority is None:
        return 0.0
    return float(priority)


def _priority_by_input_length(req: RequestQueueItem) -> float:
    """Priority function: shortest job first (smaller length = higher priority)."""
    prompt_len = getattr(req.request, "prompt_len", None)
    if prompt_len is None:
        return 0.0
    return float(prompt_len)


def _priority_by_cache_hit_rate(req: RequestQueueItem) -> float:
    """Priority function: by cache hit rate (higher hit rate = higher priority)."""
    cache_hit_rate = getattr(req.request, "cache_hit_rate", None)
    if cache_hit_rate is None:
        return 0.0
    return -float(cache_hit_rate)


def create_waiting_queue(
    policy: WaitingQueuePolicy,
    priority_fn: Optional[Callable[[RequestQueueItem], float]] = None,
) -> WaitingQueue:
    """Create a waiting queue based on the specified policy.

    Args:
        policy: The scheduling policy to use (from llm_args.WaitingQueuePolicy).
        priority_fn: Custom priority function. If provided, overrides the
                     default function for the policy. Smaller values = higher priority.

    Returns:
        A WaitingQueue instance.

    Raises:
        ValueError: If an unknown policy is provided.
    """
    if policy == WaitingQueuePolicy.FCFS:
        return FCFSWaitingQueue()

    elif policy == WaitingQueuePolicy.PRIORITY:
        fn = priority_fn or _priority_by_request_priority
        return PriorityWaitingQueue(priority_fn=fn)

    elif policy == WaitingQueuePolicy.SHORTEST_FIRST:
        fn = priority_fn or _priority_by_input_length
        return PriorityWaitingQueue(priority_fn=fn)

    elif policy == WaitingQueuePolicy.CACHE_AWARE:
        fn = priority_fn or _priority_by_cache_hit_rate
        return PriorityWaitingQueue(priority_fn=fn)

    else:
        raise ValueError(f"Unknown waiting queue policy: {policy}")
