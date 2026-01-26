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
Waiting Request Assignment Strategy for Two-Phase Scheduling.

This module defines the interface for assigning waiting_queue requests to ranks.
Different strategies can be implemented:
- LegacyADPAssignmentStrategy: Min-heap based load balancing (matches original behavior)
- CapacityAwareAssignmentStrategy: Uses remaining capacity from local reports

Usage:
    strategy = LegacyADPAssignmentStrategy(tp_size=4, max_num_active_requests=128)
    result = strategy.assign(waiting_requests, all_reports, config)
"""

import heapq
from abc import ABC, abstractmethod
from collections import namedtuple
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from .request_queue import RequestQueueItem

if TYPE_CHECKING:
    from .types import LocalScheduleReport, SchedulerConfig


@dataclass
class WaitingAssignmentResult:
    """
    Result of waiting request assignment.
    
    Attributes:
        per_rank_items: Dict mapping rank_id -> list of RequestQueueItems
        unassigned_items: Requests that were not assigned (should be put back to waiting_queue)
    """
    per_rank_items: Dict[int, List[RequestQueueItem]] = field(default_factory=dict)
    unassigned_items: List[RequestQueueItem] = field(default_factory=list)
    
    def get_items_for_rank(self, rank: int) -> List[RequestQueueItem]:
        """Get request items assigned to a specific rank."""
        return self.per_rank_items.get(rank, [])




class WaitingRequestAssignmentStrategy(ABC):
    """
    Strategy for assigning waiting_queue requests to ranks in Phase 2.
    
    This interface allows different assignment algorithms:
    - LegacyADPAssignmentStrategy: Min-heap load balancing (matches original behavior)
    - CapacityAwareAssignmentStrategy: Uses remaining capacity from local reports
    
    All implementations must be SPMD-safe: all ranks calling assign() with
    the same inputs must get the same result.
    """
    
    @abstractmethod
    def assign(
        self,
        waiting_requests: List[RequestQueueItem],
        all_reports: List['LocalScheduleReport'],
        config: 'SchedulerConfig',
    ) -> WaitingAssignmentResult:
        """
        Assign requests from waiting_queue to ranks.
        
        Args:
            waiting_requests: Requests to assign (dequeued from waiting_queue)
            all_reports: LocalScheduleReport from all ranks
            config: Scheduler configuration
            
        Returns:
            WaitingAssignmentResult with per-rank assignments
        """
        pass


class LegacyADPAssignmentStrategy(WaitingRequestAssignmentStrategy):
    """
    Assignment strategy using min-heap load balancing.
    
    Algorithm:
    1. Handle strict target rank requests first
    2. Calculate expected_num_active_requests
    3. Use min-heap to assign remaining requests to ranks with lowest token count
    4. Sort requests by token count (descending) for better balance
    
    This matches the original executor_request_queue.py logic.
    """
    
    def __init__(self, tp_size: int, max_num_active_requests: int):
        self.tp_size = tp_size
        self.max_num_active_requests = max_num_active_requests
    
    def assign(
        self,
        waiting_requests: List[RequestQueueItem],
        all_reports: List['LocalScheduleReport'],
        config: 'SchedulerConfig',
    ) -> WaitingAssignmentResult:
        """Assign using min-heap load balancing."""
        result = WaitingAssignmentResult()
        result.per_rank_items = {i: [] for i in range(self.tp_size)}
        
        if not waiting_requests or not all_reports:
            return result
        
        # Extract from reports
        all_ranks_num_active = [r.num_active_requests for r in all_reports]
        all_ranks_num_tokens = [r.num_active_tokens for r in all_reports]
        
        # Make a copy to avoid modifying the input
        ranks_num_active = list(all_ranks_num_active)
        
        # Step 1: Handle strict target rank requests first
        remaining_requests, unassigned = self._handle_strict_target_requests(
            waiting_requests, result.per_rank_items, ranks_num_active
        )
        
        # Step 2: Calculate expected_num_active_requests
        num_new_requests_all_ranks = len(remaining_requests)
        total_num_active_requests = sum(ranks_num_active)
        expected_num_active = max(
            (total_num_active_requests + num_new_requests_all_ranks +
             self.tp_size - 1) // self.tp_size,
            max(ranks_num_active) if ranks_num_active else 0,
        )
        
        # Step 3: Balance remaining requests across ranks using min-heap
        self._balance_with_min_heap(
            remaining_requests,
            result.per_rank_items,
            ranks_num_active,
            all_ranks_num_tokens,
            expected_num_active,
        )
        
        result.unassigned_items = unassigned
        
        return result
    
    def _handle_strict_target_requests(
        self,
        new_requests: List[RequestQueueItem],
        per_rank_items: Dict[int, List[RequestQueueItem]],
        ranks_num_active: List[int],
    ) -> Tuple[List[RequestQueueItem], List[RequestQueueItem]]:
        """
        Handle requests with target rank requirements.
        
        Matches the original executor_request_queue.py behavior:
        - Strict mode (attention_dp_relax=False): If target rank has no capacity,
          the request is not assigned (will be put back to waiting_queue).
        - Relax mode (attention_dp_relax=True): Can be assigned to other ranks.
        
        Returns:
            (remaining, unassigned):
            - remaining: Requests that can be balanced across any rank
            - unassigned: Requests that could not be assigned
        """
        # Prioritize non-relaxed requests (strict target rank)
        def get_relax_value(req_item):
            scheduling_params = getattr(req_item.request, 'py_scheduling_params', None)
            if scheduling_params is None:
                return True
            return scheduling_params.attention_dp_relax
        
        sorted_requests = sorted(new_requests, key=get_relax_value, reverse=True)
        
        remaining = []   # Can be balanced across ranks
        unassigned = []  # Could not be assigned
        
        for req_item in sorted_requests:
            scheduling_params = getattr(req_item.request, 'py_scheduling_params', None)
            
            if scheduling_params is None:
                # No scheduling params -> can be balanced
                remaining.append(req_item)
                continue
            
            target_dp_rank = scheduling_params.attention_dp_rank
            is_relax = scheduling_params.attention_dp_relax
            
            if target_dp_rank is None:
                # No target rank -> can be balanced
                remaining.append(req_item)
            elif ranks_num_active[target_dp_rank] < self.max_num_active_requests:
                # Target rank has capacity -> assign to target
                ranks_num_active[target_dp_rank] += 1
                per_rank_items[target_dp_rank].append(req_item)
            elif is_relax:
                # Target rank full but can relax -> balance across other ranks
                remaining.append(req_item)
            else:
                # Strict mode and target rank full -> unassigned
                unassigned.append(req_item)
        
        return remaining, unassigned
    
    def _balance_with_min_heap(
        self,
        new_requests: List[RequestQueueItem],
        per_rank_items: Dict[int, List[RequestQueueItem]],
        ranks_num_active: List[int],
        ranks_num_tokens: List[int],
        expected_num_active: int,
    ) -> None:
        """
        Balance remaining requests across ranks using min-heap.
        """
        if not new_requests:
            return
        
        HeapVal = namedtuple('HeapVal', ['num_tokens', 'num_requests', 'rank', 'request_list'])
        
        # Build heap with ranks that have capacity
        heap = [
            HeapVal(ranks_num_tokens[tp_rank], ranks_num_active[tp_rank], tp_rank, [])
            for tp_rank in range(self.tp_size)
            if ranks_num_active[tp_rank] < expected_num_active
        ]
        
        rank_to_new_list = {val.rank: val.request_list for val in heap}
        heapq.heapify(heap)
        
        # Sort requests by token count (descending) for better balance
        sorted_requests = sorted(
            new_requests,
            key=lambda x: len(getattr(x.request, 'input_token_ids', [])) if x.request else 0,
            reverse=True,
        )
        
        # Distribute requests
        for req_item in sorted_requests:
            if not heap:
                break
            
            val = heapq.heappop(heap)
            token_count = len(getattr(req_item.request, 'input_token_ids', [])) if req_item.request else 0
            
            val = val._replace(
                num_tokens=val.num_tokens + token_count,
                num_requests=val.num_requests + 1,
            )
            val.request_list.append(req_item)
            
            if val.num_requests < expected_num_active:
                heapq.heappush(heap, val)
        
        # Extend per_rank_items with balanced requests
        for rank, reqs in rank_to_new_list.items():
            per_rank_items[rank].extend(reqs)


class CapacityAwareAssignmentStrategy(WaitingRequestAssignmentStrategy):
    """
    Assignment strategy using remaining capacity from local reports.
    
    This is the new strategy for two-phase scheduling:
    - Uses remaining_batch_slots and remaining_token_budget from reports
    - Can consider context/generation balance for ADP
    - Foundation for KV cache affinity scheduling
    """
    
    def __init__(self, tp_size: int, enable_adp_balance: bool = False):
        self.tp_size = tp_size
        self.enable_adp_balance = enable_adp_balance
    
    # Sentinel value for strict-mode request with no capacity on target rank
    _UNPROCESSABLE = -1
    
    def assign(
        self,
        waiting_requests: List[RequestQueueItem],
        all_reports: List['LocalScheduleReport'],
        config: 'SchedulerConfig',
    ) -> WaitingAssignmentResult:
        """Assign based on remaining capacity."""
        result = WaitingAssignmentResult()
        result.per_rank_items = {r.rank: [] for r in all_reports}
        
        if not waiting_requests or not all_reports:
            return result
        
        # Build remaining capacity from reports
        remaining_capacity = {
            r.rank: (r.remaining_batch_slots, r.remaining_token_budget)
            for r in all_reports
        }
        
        # Calculate total remaining
        total_remaining_slots = sum(r.remaining_batch_slots for r in all_reports)
        if total_remaining_slots == 0:
            return result
        
        # Assign requests to ranks
        for req_item in waiting_requests:
            best_rank = self._select_rank(req_item, remaining_capacity, all_reports)
            
            if best_rank == self._UNPROCESSABLE or best_rank is None:
                # Could not be assigned -> put back to waiting_queue
                result.unassigned_items.append(req_item)
                continue
            
            result.per_rank_items[best_rank].append(req_item)
            
            # Update remaining capacity
            slots, tokens = remaining_capacity[best_rank]
            req_tokens = self._estimate_tokens(req_item)
            remaining_capacity[best_rank] = (slots - 1, tokens - req_tokens)
        
        return result
    
    def _select_rank(
        self,
        req_item: RequestQueueItem,
        remaining_capacity: Dict[int, Tuple[int, int]],
        all_reports: List['LocalScheduleReport'],
    ) -> Optional[int]:
        """
        Select best rank for a request.
        
        Returns:
            - rank >= 0: assign to this rank
            - None: no capacity, request not assigned (but can try others)
            - _UNPROCESSABLE: strict-mode request with no capacity on target rank
              (should be put back to waiting_queue)
        """
        # Check for hard constraint (target rank)
        request = req_item.request if hasattr(req_item, 'request') else None
        if request is not None:
            scheduling_params = getattr(request, 'py_scheduling_params', None)
            if scheduling_params is not None:
                target_rank = scheduling_params.attention_dp_rank
                if target_rank is not None and not scheduling_params.attention_dp_relax:
                    # Strict mode: must go to target rank
                    slots, _ = remaining_capacity.get(target_rank, (0, 0))
                    if slots > 0:
                        return target_rank
                    # Target rank has no capacity -> unprocessable
                    return self._UNPROCESSABLE
        
        req_tokens = self._estimate_tokens(req_item)
        is_context = self._is_context_request(req_item)
        
        candidates = []
        for rank, (slots, tokens) in remaining_capacity.items():
            if slots <= 0 or tokens < req_tokens:
                continue
            
            score = 0.0
            
            # Load balancing: prefer more capacity
            score += slots * 10
            score += tokens * 0.01
            
            # ADP balance: try to spread context/generation
            if self.enable_adp_balance:
                report = next((r for r in all_reports if r.rank == rank), None)
                if report:
                    if is_context:
                        score -= report.num_context_scheduled * 5
                    else:
                        score -= report.num_generation_scheduled * 5
            
            candidates.append((rank, score))
        
        if not candidates:
            return None
        
        return max(candidates, key=lambda x: x[1])[0]
    
    def _estimate_tokens(self, req_item: RequestQueueItem) -> int:
        """Estimate tokens for a request."""
        request = req_item.request if hasattr(req_item, 'request') else None
        if request is not None:
            return getattr(request, 'py_orig_prompt_len', 1)
        return 1
    
    def _is_context_request(self, req_item: RequestQueueItem) -> bool:
        """Check if request is context/prefill."""
        request = req_item.request if hasattr(req_item, 'request') else None
        if request is not None:
            # Import here to avoid circular dependency
            try:
                from ..llm_request import LlmRequestState
                state = getattr(request, 'state', None)
                return state == LlmRequestState.CONTEXT_INIT
            except ImportError:
                pass
        return True
