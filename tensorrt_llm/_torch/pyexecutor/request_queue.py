# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Pure Request Storage Queue.

This module provides a minimal, single-responsibility queue for storing
incoming requests. Interface is compatible with the original ExecutorRequestQueue
but excludes scheduling-related logic.

Design Note (Option B):
- This is PURE storage: enqueue/dequeue only
- canceled_req_ids and control_requests are handled by RequestFetcher
- This ensures clean separation of concerns

Excluded responsibilities (handled elsewhere):
- Scheduling decisions (handled by Scheduler)
- Request transformation (handled by RequestTransformer)  
- Distributed communication (handled by RequestFetcher)
- ADP load balancing (handled by ADPSolver)
- Cancel/control request tracking (handled by RequestFetcher)
"""

import dataclasses
import queue
import threading
import time
from itertools import repeat
from typing import Dict, Iterable, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .llm_request import ExecutorRequest

# Sentinel values for special request types (same as original)
SHUTDOWN_REQUEST_ID = -1
CONTROL_REQUEST_ID = -2


@dataclasses.dataclass
class RequestQueueItem:
    """
    Item stored in the request queue.
    
    Attributes:
        id: Unique request identifier
        request: The actual ExecutorRequest object (None for special items)
        child_req_ids: IDs for child requests (e.g., num_return_sequences > 1)
        is_canceled_request: True if this is a cancellation marker
        query: Query tokens for StarAttention (optional)
    """
    id: int
    request: Optional['ExecutorRequest'] = None
    _ = dataclasses.KW_ONLY
    child_req_ids: Optional[list] = None
    is_canceled_request: bool = False
    query: Optional[list] = None  # only used in `StarAttention`

    @property
    def is_shutdown_request(self) -> bool:
        """Check if this is a shutdown signal."""
        return self.id == SHUTDOWN_REQUEST_ID

    @property
    def is_normal_request(self) -> bool:
        """Check if this is a normal request (not shutdown, cancel, or control)."""
        return not (self.is_shutdown_request or self.is_canceled_request
                    or self.is_control_request)

    @property
    def is_control_request(self) -> bool:
        """Check if this is a control request."""
        return self.id == CONTROL_REQUEST_ID


class RequestQueue:
    """
    Pure request storage queue.
    
    Responsibilities (pure storage only):
    - Store incoming requests (thread-safe)
    - Provide FIFO access to requests
    - Generate unique request IDs
    
    NOT responsible for (handled by RequestFetcher):
    - Tracking canceled request IDs
    - Tracking control requests
    - Waiting queue management
    """
    
    def __init__(
        self,
        initial_request_id: int = 0,
        enable_iter_perf_stats: bool = False,
        rank: int = 0
    ):
        """
        Initialize the request queue.
        
        Args:
            initial_request_id: Starting value for request ID generation.
                               Original uses max_batch_size as initial value.
            enable_iter_perf_stats: Whether to track request start times for perf stats.
            rank: The rank of the current process. Only rank 0 can enqueue requests.
        """
        self.request_queue: queue.Queue[RequestQueueItem] = queue.Queue()
        self.enqueue_lock = threading.Lock()
        self.next_request_id = initial_request_id
        self.active = True
        self.enable_iter_perf_stats = enable_iter_perf_stats
        self.start_times: Dict[int, float] = {}
        self.rank = rank
    
    # ========== ID Generation ==========
    
    def _get_request_id(self) -> int:
        """
        Generate next request ID.
        
        Same logic as original: (next_request_id + 1) % UINT64_MAX
        """
        current_id = self.next_request_id
        self.next_request_id = (self.next_request_id + 1) & ((1 << 64) - 1)
        return current_id
    
    @staticmethod
    def _get_num_child_requests(request: 'ExecutorRequest') -> int:
        """
        Get number of child requests needed for a request.
        
        Child requests are created for num_return_sequences > 1 (not beam search).
        """
        sampling_config = request.sampling_config
        return 0 if sampling_config.beam_width > 1 else (
            sampling_config.num_return_sequences or 1) - 1
    
    def _generate_child_request_ids(
        self,
        request: 'ExecutorRequest'
    ) -> Optional[List[int]]:
        """
        Generate child request IDs if needed.
        
        Args:
            request: The parent ExecutorRequest
            
        Returns:
            List of child request IDs, or None if no children needed
        """
        child_req_ids = None
        num_children = self._get_num_child_requests(request)
        if num_children > 0:
            child_req_ids = []
            for _ in range(num_children):
                child_req_id = self._get_request_id()
                if self.enable_iter_perf_stats:
                    self.start_times[child_req_id] = time.time()
                child_req_ids.append(child_req_id)
        return child_req_ids
    
    # ========== Enqueue Operations ==========
    
    def _enqueue_impl(
        self,
        requests_and_queries: Iterable[Tuple['ExecutorRequest', Optional[List]]]
    ) -> List[int]:
        """
        Internal implementation for enqueue operations.
        
        Args:
            requests_and_queries: Iterable of (request, query) tuples
        
        Returns:
            List of assigned request IDs
        """
        req_ids = []
        with self.enqueue_lock:
            assert self.active, "RequestQueue has already been shutdown."
            start_time = time.time()
            for request, query in requests_and_queries:
                req_id = self._get_request_id()
                if self.enable_iter_perf_stats:
                    self.start_times[req_id] = start_time
                child_req_ids = self._generate_child_request_ids(request)
                
                self.request_queue.put(
                    RequestQueueItem(
                        req_id,
                        request,
                        child_req_ids=child_req_ids,
                        query=query
                    )
                )
                req_ids.append(req_id)
        return req_ids
    
    def enqueue_requests(self, requests: List['ExecutorRequest']) -> List[int]:
        """
        Enqueue multiple requests.
        
        Args:
            requests: List of ExecutorRequests to enqueue
        
        Returns:
            List of assigned request IDs
        """
        return self._enqueue_impl(zip(requests, repeat(None)))
    
    def enqueue_request(
        self,
        request: 'ExecutorRequest',
        query: Optional[List] = None,
    ) -> int:
        """
        Enqueue a single request.
        
        Args:
            request: The ExecutorRequest to enqueue
            query: Optional query tokens (for StarAttention)
        
        Returns:
            Assigned request ID
        """
        return self._enqueue_impl([(request, query)])[0]
    
    def enqueue_cancel_request(self, req_id: int) -> None:
        """
        Enqueue a cancellation marker for a request.
        
        Note: The actual tracking of canceled request IDs is handled by
        RequestFetcher when processing items from the queue.
        
        Args:
            req_id: ID of the request to cancel
        """
        with self.enqueue_lock:
            self.request_queue.put(
                RequestQueueItem(req_id, is_canceled_request=True)
            )
    
    def enqueue_control_request(self) -> None:
        """
        Enqueue a control request.
        
        Control requests are used to synchronize executor state.
        The actual tracking is handled by RequestFetcher.
        """
        with self.enqueue_lock:
            self.request_queue.put(RequestQueueItem(id=CONTROL_REQUEST_ID))
    
    def enqueue_shutdown_request(self) -> None:
        """
        Signal queue shutdown.
        
        After this call, no new requests can be enqueued.
        """
        with self.enqueue_lock:
            self.request_queue.put(RequestQueueItem(SHUTDOWN_REQUEST_ID))
            self.active = False
    
    def can_enqueue_request(self) -> bool:
        """
        Check if the queue is accepting new requests.
        
        Only rank 0 can enqueue requests (other ranks receive via broadcast).
        
        Returns:
            True if the queue is active and this is rank 0
        """
        with self.enqueue_lock:
            return self.active and self.rank == 0
    
    # ========== Dequeue Operations ==========
    
    def dequeue(self, timeout: Optional[float] = None) -> Optional[RequestQueueItem]:
        """
        Get the next item from the queue.
        
        Args:
            timeout: Maximum seconds to wait
                    - None: Block forever until item available
                    - 0: Non-blocking, return immediately
                    - >0: Wait up to timeout seconds
        
        Returns:
            RequestQueueItem or None if timeout expired
        """
        try:
            if timeout == 0:
                return self.request_queue.get_nowait()
            elif timeout is None:
                return self.request_queue.get(block=True)
            else:
                return self.request_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def dequeue_available(self) -> List[RequestQueueItem]:
        """
        Get all currently available items without blocking.
        
        Returns:
            List of all available items (may be empty)
        """
        items = []
        while True:
            try:
                items.append(self.request_queue.get_nowait())
            except queue.Empty:
                break
        return items
    
    # ========== State Queries ==========
    
    @property
    def is_active(self) -> bool:
        """Check if the queue is still accepting new requests."""
        with self.enqueue_lock:
            return self.active
    
    def is_empty(self) -> bool:
        """Check if the queue is empty."""
        with self.enqueue_lock:
            return self.request_queue.empty()
    
    def get(self, timeout: Optional[float] = None) -> RequestQueueItem:
        """
        Get an item from the queue.
        
        Args:
            timeout: Maximum wait time in seconds (None for infinite)
        
        Returns:
            RequestQueueItem
        
        Raises:
            queue.Empty: If queue is empty and timeout expires
        """
        return self.request_queue.get(timeout=timeout)
    
    def get_nowait(self) -> RequestQueueItem:
        """
        Get an item from the queue without waiting.
        
        Returns:
            RequestQueueItem
        
        Raises:
            queue.Empty: If queue is empty
        """
        return self.request_queue.get_nowait()
