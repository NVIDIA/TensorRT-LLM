# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable, Iterator
from typing import Callable, Optional

from tensorrt_llm.llmapi.llm_args import SjfConfig, WaitingQueuePolicy

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


class SJFWaitingQueue(WaitingQueue):
    """Shortest-Job-First waiting queue with wait-time aging.

    Prioritizes shorter requests while using aging to prevent starvation.
    Score = length_weight * length_score + time_weight * time_score
    where length_score = 1/(1 + prompt_len/length_median) and
    time_score = wait_time/time_median.

    Internally maintains two lists:
    - _prepended: requests returned via prepend_request(s), served first
    - _requests: new requests, lazily sorted by SJF score on peek/pop
    """

    def __init__(self, sjf_config: Optional[SjfConfig] = None):
        self._requests: list[RequestQueueItem] = []
        self._prepended: list[RequestQueueItem] = []
        self._sorted = False
        self._config = sjf_config or SjfConfig()
        self._arrival_times: dict[int, float] = {}

    def _get_arrival_time(self, item: RequestQueueItem) -> float:
        arrival = getattr(item.request, 'py_arrival_time',
                          None) if item.request else None
        if arrival is not None:
            return arrival
        return self._arrival_times.get(item.id, time.time())

    def _compute_score(self, item: RequestQueueItem, now: float) -> float:
        prompt_len = (len(item.request.input_token_ids)
                      if item.request and item.request.input_token_ids else 0)
        wait_time = max(0.0, now - self._get_arrival_time(item))
        length_score = 1.0 / (1.0 + prompt_len / self._config.length_median)
        time_score = wait_time / self._config.time_median
        return (self._config.length_weight * length_score +
                self._config.time_weight * time_score)

    def _ensure_sorted(self) -> None:
        if not self._sorted and self._requests:
            now = time.time()
            self._requests.sort(
                key=lambda item: self._compute_score(item, now), reverse=True)
            self._sorted = True

    def add_request(self, request: RequestQueueItem) -> None:
        """Add a request to the queue."""
        if (not request.request
                or not hasattr(request.request, 'py_arrival_time')
                or request.request.py_arrival_time is None):
            self._arrival_times[request.id] = time.time()
        self._requests.append(request)
        self._sorted = False

    def add_requests(self, requests: Iterable[RequestQueueItem]) -> None:
        """Add multiple requests to the queue."""
        for request in requests:
            self.add_request(request)

    def pop_request(self) -> RequestQueueItem:
        """Pop the highest-priority request."""
        if self._prepended:
            item = self._prepended.pop(0)
            self._arrival_times.pop(item.id, None)
            return item
        self._ensure_sorted()
        if not self._requests:
            raise IndexError("pop from an empty queue")
        item = self._requests.pop(0)
        self._arrival_times.pop(item.id, None)
        return item

    def peek_request(self) -> RequestQueueItem:
        """Peek at the highest-priority request without removing it."""
        if self._prepended:
            return self._prepended[0]
        self._ensure_sorted()
        if not self._requests:
            raise IndexError("peek from an empty queue")
        return self._requests[0]

    def prepend_request(self, request: RequestQueueItem) -> None:
        """Prepend a request to the front of the queue."""
        self._prepended.insert(0, request)

    def prepend_requests(self, requests: Iterable[RequestQueueItem]) -> None:
        """Prepend requests to the front of the queue.

        Note: Matches FCFSWaitingQueue semantics — the requests iterable
        is consumed via extendleft-style reversal. The caller
        (request_utils.py) passes reversed(pending_requests), so the net
        effect is that pending_requests appear at the front in their
        original order.
        """
        incoming = list(requests)
        incoming.reverse()
        self._prepended = incoming + self._prepended

    def remove_by_ids(self, request_ids: set[int]) -> None:
        """Remove requests with the given IDs."""
        self._prepended = [
            req for req in self._prepended if req.id not in request_ids
        ]
        self._requests = [
            req for req in self._requests if req.id not in request_ids
        ]
        for rid in request_ids:
            self._arrival_times.pop(rid, None)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return bool(self._prepended) or bool(self._requests)

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return len(self._prepended) + len(self._requests)

    def __iter__(self) -> Iterator[RequestQueueItem]:
        """Iterate over the queue (prepended first, then sorted requests)."""
        self._ensure_sorted()
        return itertools.chain(self._prepended, self._requests).__iter__()


def create_waiting_queue(
    policy: WaitingQueuePolicy = WaitingQueuePolicy.FCFS,
    priority_fn: Optional[Callable[[RequestQueueItem], float]] = None,
    sjf_config: Optional[SjfConfig] = None,
) -> WaitingQueue:
    """Create a waiting queue based on the specified policy.

    Args:
        policy: The scheduling policy to use.
        priority_fn: Reserved for future use.
        sjf_config: Configuration for SJF scheduling. Only used when
            policy is SJF.

    Returns:
        A WaitingQueue instance.
    """
    if policy == WaitingQueuePolicy.FCFS:
        return FCFSWaitingQueue()
    elif policy == WaitingQueuePolicy.SJF:
        return SJFWaitingQueue(sjf_config)
    else:
        raise ValueError(f"Unsupported waiting queue policy: {policy}")
