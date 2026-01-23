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
Base Scheduler Interface and Internal Context Types.

This module defines:
1. Internal context types for the three-phase workflow
2. Base class with common state and methods
3. Abstract interface for scheduler-specific logic

For public I/O types (ScheduledRequests, BatchScheduleResult), see types.py.

Scheduling Flow:
    prepare_and_schedule_batch() -> Optional[BatchScheduleResult]
        ├── _pre_schedule()   -> Optional[ScheduleContext] (None if shutdown)
        ├── _schedule()       -> PostScheduleContext
        └── _post_schedule()  -> BatchScheduleResult
"""

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Set, TYPE_CHECKING

from .types import ScheduledRequests, BatchScheduleResult, SchedulerOutput

if TYPE_CHECKING:
    from ..llm_request import LlmRequest
    from .request_queue import RequestQueueItem


# ========== Internal Phase Context Types ==========

@dataclass
class ScheduleContext:
    """Context for scheduling (passed from pre_schedule to schedule)."""
    active_requests: List['LlmRequest'] = field(default_factory=list)
    inflight_request_ids: Set[int] = field(default_factory=set)
    iter_stats: Optional[dict] = None
    use_spec_decode: bool = False
    max_draft_tokens: int = 0


@dataclass
class PostScheduleContext:
    """Context from schedule phase (passed to post_schedule)."""
    scheduled: ScheduledRequests = field(default_factory=ScheduledRequests)
    fitting_disagg_gen_init: List['LlmRequest'] = field(default_factory=list)
    num_fitting_requests: int = 0
    use_spec_decode: bool = False
    max_draft_tokens: int = 0
    iter_stats: Optional[dict] = None


# ========== Base Scheduler Class ==========

class BaseScheduler(ABC):
    """
    Base class for SPMD schedulers.
    
    Provides:
    - Common state (waiting_queue, control_requests, canceled_req_ids)
    - Common methods for state management
    - Abstract interface for scheduler-specific logic
    """
    
    def __init__(self):
        """Initialize common state."""
        # Waiting queue: requests fetched but not yet activated
        self.waiting_queue: Deque['RequestQueueItem'] = deque()
        
        # Control requests that need special handling
        self.control_requests: List['RequestQueueItem'] = []
        
        # Canceled request IDs
        self.canceled_req_ids: List[int] = []
        
        # Accumulated requests (queued during control request handling)
        self.request_accumulated: List['RequestQueueItem'] = []
        
        # Shutdown flag
        self.is_shutdown: bool = False
    
    # ========== Common State Accessors ==========
    
    def has_control_requests(self) -> bool:
        """Check if there are pending control requests."""
        return len(self.control_requests) > 0
    
    def get_num_control_requests(self) -> int:
        """Get number of pending control requests."""
        return len(self.control_requests)
    
    def pop_control_request(self) -> Optional['RequestQueueItem']:
        """Pop and return the first control request."""
        if self.control_requests:
            return self.control_requests.pop(0)
        return None
    
    def pop_all_control_requests(self) -> List['RequestQueueItem']:
        """Get and clear all control requests."""
        requests = self.control_requests
        self.control_requests = []
        return requests
    
    def get_canceled_req_ids(self) -> List[int]:
        """Get the list of canceled request IDs."""
        return self.canceled_req_ids
    
    def get_canceled_req_ids_size(self) -> int:
        """Get the number of canceled request IDs."""
        return len(self.canceled_req_ids)
    
    def clear_canceled_req_ids(self) -> None:
        """Clear the canceled request IDs list."""
        self.canceled_req_ids.clear()
    
    def add_canceled_req_ids(self, req_ids: List[int]) -> None:
        """Add request IDs to the canceled list."""
        self.canceled_req_ids.extend(req_ids)
    
    def update_waiting_queue(self) -> None:
        """Remove canceled requests from waiting queue."""
        if not self.canceled_req_ids:
            return
        canceled_set = set(self.canceled_req_ids)
        self.waiting_queue = deque(
            req for req in self.waiting_queue
            if req.id not in canceled_set
        )
    
    def get_waiting_queue_size(self) -> int:
        """Get current size of waiting queue."""
        return len(self.waiting_queue)
    
    # ========== Template Method ==========
    
    def prepare_and_schedule_batch(
        self,
        active_requests: List['LlmRequest'],
        inflight_request_ids: Set[int],
        new_requests: List['RequestQueueItem'],
        iter_stats: Optional[dict] = None,
        drafter: Optional[object] = None,
        max_batch_size: int = 8,
        max_num_tokens: int = 8192,
    ) -> Optional[BatchScheduleResult]:
        """
        Main scheduling entry point (Template Method).
        
        Orchestrates the three-phase scheduling workflow:
        1. Pre-schedule: enqueue new_requests, drafter setup -> returns ScheduleContext
        2. Schedule: capacity check, micro-batch selection -> returns PostScheduleContext
        3. Post-schedule: disagg preparation, build final result
        
        Args:
            new_requests: New requests fetched and validated by executor using
                RequestFetcher/RequestBroadcaster from scheduler package.
        
        Returns:
            BatchScheduleResult if scheduling completed, None if shutdown.
            Caller should check scheduler.is_shutdown for shutdown state.
        """
        # Phase 1: Pre-schedule (returns ScheduleContext or None if shutdown)
        schedule_ctx = self._pre_schedule(
            active_requests=active_requests,
            inflight_request_ids=inflight_request_ids,
            new_requests=new_requests,
            iter_stats=iter_stats,
            drafter=drafter,
            max_batch_size=max_batch_size,
            max_num_tokens=max_num_tokens,
        )
        
        # Early return if shutdown
        if schedule_ctx is None:
            return None
        
        # Phase 2: Schedule
        post_ctx = self._schedule(schedule_ctx)
        
        # Phase 3: Post-schedule
        result = self._post_schedule(post_ctx)
        
        return result
    
    # ========== Abstract Hooks (Subclass Implementation) ==========
    
    @abstractmethod
    def _pre_schedule(
        self,
        active_requests: List['LlmRequest'],
        inflight_request_ids: Set[int],
        new_requests: List['RequestQueueItem'],
        iter_stats: Optional[dict],
        drafter: Optional[object],
        max_batch_size: int,
        max_num_tokens: int,
    ) -> Optional[ScheduleContext]:
        """
        Pre-schedule phase: enqueue new_requests, drafter setup.
        
        Args:
            new_requests: Already fetched and validated by executor.
        
        Returns:
            ScheduleContext if ready to schedule, None if shutdown.
        """
        pass
    
    @abstractmethod
    def _schedule(
        self,
        ctx: ScheduleContext,
    ) -> PostScheduleContext:
        """Main scheduling phase: capacity check, micro-batch selection."""
        pass
    
    @abstractmethod
    def _post_schedule(
        self,
        ctx: PostScheduleContext,
    ) -> BatchScheduleResult:
        """Post-schedule phase: disagg preparation, build final result."""
        pass
