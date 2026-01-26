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
Unified SPMD Scheduler Implementation.

Architecture:
    - UnifiedSPMDScheduler: Manages all state (waiting_queue, control_requests, etc.)
    - RequestFetcher: Stateless utility for fetch/broadcast operations
    - WaitingRequestAssignmentStrategy: Pluggable strategy for waiting request distribution

Two-Phase Scheduling with Late Binding:
    
    Phase 1 (Local Scheduling): Each rank independently schedules its active_requests
        - No cross-rank communication during scheduling
        - Reports: scheduled requests, remaining capacity, context/gen breakdown
        
    Phase 2 (Global Assignment): Based on Phase 1 reports, assign waiting_queue
        - All ranks compute the same assignment (SPMD style)
        - Late binding: requests are assigned to ranks based on current state
        - Pluggable via WaitingRequestAssignmentStrategy interface

Scheduling Flow:

    Pre-schedule (`_pre_schedule`):
        1. Fetch new requests from queue (rank 0 only)
        2. Broadcast to all ranks -> add to waiting_queue
        3. Handle special items (shutdown, cancel, control)
        4. Check disagg transfer status
        5. Setup drafter (speculative decoding)

    Schedule (`_schedule`):
        Phase 1 - Local Scheduling:
            1. Capacity scheduling (which requests fit in memory)
            2. Micro-batch scheduling (which requests run this iteration)
            3. Build LocalScheduleReport with remaining capacity
            
        Sync Point:
            4. AllGather reports from all ranks
            
        Phase 2 - Global Assignment:
            5. Use WaitingRequestAssignmentStrategy to assign waiting_queue to ranks
            6. Each rank gets its assigned requests -> add to active_requests
            7. Try to schedule newly assigned requests if capacity allows
            
        Post-processing:
            8. ADP dummy padding (if needed)
            9. ADP context balancing / batch waiting
        
    Post-schedule (`_post_schedule`):
        1. Disable spec decode for scheduled requests if needed
        2. Prepare disagg gen init resources
        3. Build BatchScheduleResult
"""

import datetime
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Set, Tuple, TYPE_CHECKING, Callable

from tensorrt_llm._utils import mpi_disabled, nvtx_range
from tensorrt_llm.logger import logger

from .base import (
    BaseScheduler,
    ScheduleContext,
    PostScheduleContext,
)
from .types import (
    SchedulerConfig,
    ScheduledRequests,
    BatchScheduleResult,
    SchedulerOutput,
    LocalScheduleReport,
)
from ..request_queue import RequestQueueItem
from .request_assignment import (
    WaitingRequestAssignmentStrategy,
    LegacyADPAssignmentStrategy,
    CapacityAwareAssignmentStrategy,
    WaitingAssignmentResult,
)
from ..request_utils import (
    RequestBroadcaster,
    collect_py_objects,
    attach_py_objects,
    merge_requests,
)
from .resource_context import SchedulingResourceContext

if TYPE_CHECKING:
    from ..llm_request import LlmRequest, LlmRequestState
    from ...distributed import Distributed
    from .local_scheduler import PyCapacityScheduler, PyMicroBatchScheduler


class UnifiedSPMDScheduler(BaseScheduler):
    """
    Unified SPMD Scheduler that implements the complete scheduling workflow.
    
    This scheduler:
    - Manages all state (waiting_queue, control_requests, canceled_req_ids)
    - Uses RequestFetcher as a stateless utility for fetch/broadcast operations
    
    Key design:
    - State lives in Scheduler, operations live in Fetcher
    - Clean separation of concerns
    """
    
    def __init__(
        self,
        dist: 'Distributed',
        config: SchedulerConfig,
        capacity_scheduler: 'PyCapacityScheduler',
        micro_batch_scheduler: 'PyMicroBatchScheduler',
        # Callbacks for executor integration
        check_disagg_transfer_cb: Optional[Callable[[], None]] = None,
        prepare_disagg_gen_init_cb: Optional[Callable[[List['LlmRequest']], None]] = None,
        add_dummy_request_cb: Optional[Callable[[int], 'LlmRequest']] = None,
        validate_request_cb: Optional[Callable[['LlmRequest'], bool]] = None,
        # Pluggable assignment strategy
        assignment_strategy: Optional[WaitingRequestAssignmentStrategy] = None,
        use_legacy_adp_assignment: bool = True,  # True = use legacy, False = use capacity-aware
    ):
        """
        Initialize the unified scheduler.
        
        Note: Fetch and broadcast are handled by executor using RequestFetcher/RequestBroadcaster.
        Validation is done via validate_request_cb after merge.
        
        Args:
            validate_request_cb: Callback to validate LlmRequest after merge.
                Returns True if invalid (should be filtered out).
            assignment_strategy: Strategy for assigning waiting requests to ranks.
                If None, will create based on use_legacy_adp_assignment flag.
            use_legacy_adp_assignment: If True (default), use LegacyADPAssignmentStrategy
                which is compatible with original MinHeapADPBalancer behavior.
                If False, use CapacityAwareAssignmentStrategy.
        """
        super().__init__()  # Initialize base state
        
        self.dist = dist
        self.config = config
        
        self.capacity_scheduler = capacity_scheduler
        self.micro_batch_scheduler = micro_batch_scheduler
        
        # Callbacks for executor integration
        self._check_disagg_transfer = check_disagg_transfer_cb
        self._prepare_disagg_gen_init = prepare_disagg_gen_init_cb
        self._add_dummy_request = add_dummy_request_cb
        self._validate_request = validate_request_cb
        
        # Create assignment strategy (only used for ADP)
        if assignment_strategy is not None:
            self.assignment_strategy = assignment_strategy
        elif config.enable_attention_dp:
            if use_legacy_adp_assignment:
                # ADP with legacy MinHeap-based assignment
                self.assignment_strategy = LegacyADPAssignmentStrategy(
                    tp_size=dist.tp_size,
                    max_num_active_requests=config.max_num_active_requests,
                )
            else:
                # ADP with capacity-aware assignment
                self.assignment_strategy = CapacityAwareAssignmentStrategy(
                    tp_size=dist.tp_size,
                    enable_adp_balance=config.enable_adp_balance,
                )
        else:
            # Non-ADP: assignment_strategy not used (_get_waiting_requests is used instead)
            self.assignment_strategy = None
        
        # Performance metrics
        self.num_fetch_requests: int = 0
        self.num_fetch_requests_cur_rank: int = 0
        
        # Disable MPI flag (Ray mode)
        self._disable_mpi = mpi_disabled()
        
        # ADP context balancing state
        self._adp_ctx_waiting_iters: int = 0
        self._adp_ctx_batching_wait_iters: int = 0
        
        # Batch waiting state (non-ADP)
        self._batch_waiting_iters: int = 0
        # Enable batch waiting flag
        self._enable_batch_waiting: bool = (
            config.batch_wait_timeout_iters > 0 or config.batch_wait_max_tokens_ratio > 0
        )
        
        # Exclude last generation logits flag
        self.should_exclude_last_generation_logits: bool = False
    
    def set_exclude_last_generation_logits(
        self,
        disable_overlap_scheduler: bool,
        pp_size: int,
    ) -> None:
        """Set whether to exclude last generation logits."""
        self.should_exclude_last_generation_logits = not disable_overlap_scheduler and pp_size == 1
    
    def _pre_schedule(
        self,
        active_requests: List['LlmRequest'],
        inflight_request_ids: Set[int],
        new_requests: List[RequestQueueItem],
        iter_stats: Optional[dict],
        drafter: Optional[object],
        max_batch_size: int,
        max_num_tokens: int,
    ) -> Optional[ScheduleContext]:
        """
        Pre-schedule phase: enqueue new requests and setup drafter.
        
        Note: Fetch and validation are handled by executor using RequestFetcher/RequestBroadcaster.
        Executor passes new_requests (already fetched/validated RequestQueueItems) here.
        
        Returns:
            ScheduleContext if ready to schedule, None if shutdown.
        """
        # Step 1: Enqueue new requests to waiting_queue
        self._enqueue_new_requests(new_requests)
        
        # Early return if shutdown
        if self.is_shutdown:
            return None
        
        # Step 2: Check disagg transfer status
        if self._check_disagg_transfer is not None:
            self._check_disagg_transfer()
        
        # Step 3: Setup drafter (speculative decoding)
        use_spec_decode = False
        max_draft_tokens = 0
        if drafter is not None:
            use_spec_decode, max_draft_tokens = self._setup_drafter(
                active_requests=active_requests,
                drafter=drafter,
                max_batch_size=max_batch_size,
                max_num_tokens=max_num_tokens,
            )
        
        # Build and return ScheduleContext
        return ScheduleContext(
            active_requests=active_requests,
            inflight_request_ids=inflight_request_ids,
            iter_stats=iter_stats,
            use_spec_decode=use_spec_decode,
            max_draft_tokens=max_draft_tokens,
        )
    
    def _enqueue_new_requests(self, new_requests: List[RequestQueueItem]) -> None:
        """
        Add new requests to waiting_queue.
        
        Note: Requests are already fetched and validated by executor
        using RequestFetcher/RequestBroadcaster from scheduler package.
        """
        # Handle special items (shutdown, cancel, control)
        normal_requests = self._handle_special_items(new_requests)
        
        self.waiting_queue.extend(normal_requests)
        self.num_fetch_requests += len(normal_requests)
    
    def _handle_special_items(
        self,
        new_requests: List[RequestQueueItem],
    ) -> List[RequestQueueItem]:
        """
        Handle special queue items (shutdown, cancel, control).
        
        Updates Scheduler state and returns normal requests.
        """
        normal_requests = []
        
        for idx, req_item in enumerate(new_requests):
            if req_item.is_shutdown_request:
                self.is_shutdown = True
                break
            elif req_item.is_canceled_request:
                self.canceled_req_ids.append(req_item.id)
            elif req_item.is_control_request:
                self.control_requests.append(req_item)
                # Accumulate remaining items for next fetch
                if self.dist.rank == 0:
                    self.request_accumulated.extend(new_requests[idx + 1:])
                break
            else:
                normal_requests.append(req_item)
        
        return normal_requests
    
    def _pad_attention_dp_dummy(
        self,
        all_reports: List[LocalScheduleReport],
    ) -> Optional['LlmRequest']:
        """
        Create a generation dummy request if required, to ensure every 
        attention_dp rank has at least one active request.
        
        Args:
            all_reports: Reports from all ranks
            
        Returns:
            The dummy request if created, None otherwise.
        """
        if not self.config.enable_attention_dp:
            return None
        
        if self._add_dummy_request is None:
            return None
        
        # Calculate max scheduled count across all ranks
        max_per_rank = max(
            r.num_context_scheduled + r.num_generation_scheduled
            for r in all_reports
        ) if all_reports else 0
        
        if max_per_rank == 0:
            return None
        
        # Get this rank's scheduled count from reports
        cur_rank = self.dist.tp_rank
        cur_report = next((r for r in all_reports if r.rank == cur_rank), None)
        num_scheduled = (cur_report.num_context_scheduled + cur_report.num_generation_scheduled) if cur_report else 0
        
        if num_scheduled == 0:
            dummy = self._add_dummy_request(0)
            if dummy is not None:
                dummy.is_attention_dp_dummy = True
                return dummy
        
        return None
    
    # ========== Drafter Setup ==========
    
    def _setup_drafter(
        self,
        active_requests: List['LlmRequest'],
        drafter: object,
        max_batch_size: int,
        max_num_tokens: int,
    ) -> Tuple[bool, int]:
        """Setup drafter for speculative decoding."""
        max_draft_tokens = 0
        use_spec_decode = False
        
        if drafter.draft_len_schedule is not None:
            batch_size = len(active_requests)
            max_draft_tokens = drafter.get_draft_len_for_batch_size(batch_size)
            drafter.update_max_total_draft_tokens(max_draft_tokens)
        
        if drafter.draft_len_schedule is not None and max_draft_tokens == 0:
            use_spec_decode = False
        elif getattr(self, 'speculation_permanently_disabled', False):
            use_spec_decode = False
        else:
            use_spec_decode = drafter.should_use_spec_decode(
                active_requests, max_batch_size, max_num_tokens, max_draft_tokens
            )
        
        self._set_draft_tokens_for_requests(active_requests, max_draft_tokens)
        
        logger.debug(f"Use spec decode: {use_spec_decode}")
        return use_spec_decode, max_draft_tokens
    
    def _set_draft_tokens_for_requests(
        self,
        requests: List['LlmRequest'],
        max_draft_tokens: int,
    ) -> None:
        """
        Set draft_tokens for generation requests.
        
        Only sets draft_tokens for requests in GENERATION_IN_PROGRESS or
        DISAGG_GENERATION_INIT state.
        """
        if not requests:
            return
        
        from ..llm_request import LlmRequestState
        
        draft_tokens = [0] * max_draft_tokens if max_draft_tokens > 0 else []
        for request in requests:
            state = getattr(request, 'state', None)
            if state in (LlmRequestState.GENERATION_IN_PROGRESS, 
                        LlmRequestState.DISAGG_GENERATION_INIT):
                request.draft_tokens = draft_tokens.copy() if draft_tokens else []
    
    @nvtx_range("_schedule_active_requests")
    def _schedule_active_requests(
        self,
        active_requests: List['LlmRequest'],
        inflight_request_ids: Set[int],
        max_batch_size: int,
        max_num_tokens: int,
        resource_context: SchedulingResourceContext,
    ) -> Tuple[LocalScheduleReport, ScheduledRequests, List['LlmRequest']]:
        """
        Phase 1: Local Scheduling.
        
        Each rank independently schedules its own active_requests.
        Returns a report of what was scheduled and remaining capacity.
        
        Args:
            active_requests: This rank's active requests
            inflight_request_ids: Request IDs currently in pipeline (PP)
            max_batch_size: Maximum batch size
            max_num_tokens: Maximum tokens per iteration
            resource_context: Resource context for two-phase scheduling.
            
        Returns:
            Tuple of:
            - LocalScheduleReport: Report for global assignment
            - ScheduledRequests: What was scheduled this iteration
            - fitting_disagg_gen_init: Disagg gen init requests
        """
        assert resource_context is not None, "resource_context must be provided by caller"
        
        # Step 1: Capacity scheduling - which requests fit in memory
        fitting_requests, fitting_disagg_gen_init, paused_requests = \
            self.capacity_scheduler.schedule_request(active_requests, resource_context)
        
        # Step 2: Micro-batch scheduling - which requests run this iteration
        context_requests, generation_requests = \
            self.micro_batch_scheduler.schedule(fitting_requests, inflight_request_ids, resource_context)
        
        # Step 3: Calculate scheduled tokens
        num_context_tokens = sum(
            len(req.get_tokens(0)) for req in context_requests
        )
        num_generation_tokens = sum(
            1 + getattr(req, 'num_draft_tokens', 0)
            for req in generation_requests
        )
        
        # Step 4: Calculate remaining capacity
        num_scheduled = len(context_requests) + len(generation_requests)
        scheduled_tokens = num_context_tokens + num_generation_tokens
        
        remaining_batch_slots = max_batch_size - num_scheduled
        remaining_token_budget = max_num_tokens - scheduled_tokens
        
        # Step 5: Calculate legacy ADP info (for LegacyADPAssignmentStrategy)
        # num_active_requests: total active requests on this rank (before scheduling new)
        # num_active_tokens: total active tokens on this rank
        num_active_requests = len(active_requests)
        num_active_tokens = sum(
            getattr(req, 'py_orig_prompt_len', 0) for req in active_requests
        )
        
        # Step 6: Build report
        report = LocalScheduleReport(
            rank=self.dist.tp_rank if self.config.enable_attention_dp else self.dist.rank,
            # Legacy ADP fields
            num_active_requests=num_active_requests,
            num_active_tokens=num_active_tokens,
            # New capacity-aware fields
            scheduled_request_ids=[req.py_request_id for req in context_requests + generation_requests],
            remaining_batch_slots=max(0, remaining_batch_slots),
            remaining_token_budget=max(0, remaining_token_budget),
            num_context_scheduled=len(context_requests),
            num_generation_scheduled=len(generation_requests),
            num_context_tokens=num_context_tokens,
            num_generation_tokens=num_generation_tokens,
        )
        
        # Step 7: Build scheduled requests
        scheduled = ScheduledRequests()
        scheduled.context_requests = list(context_requests)
        scheduled.generation_requests = list(generation_requests)
        scheduled.paused_requests = list(paused_requests)
        
        return report, scheduled, list(fitting_disagg_gen_init), resource_context
    
    @nvtx_range("_sync_schedule_reports")
    def _sync_schedule_reports(
        self,
        local_report: LocalScheduleReport,
    ) -> List[LocalScheduleReport]:
        """
        Sync Point: Gather reports from all ranks.
        
        Uses AllGather to collect LocalScheduleReport from every rank.
        All ranks end up with the same list of reports.
        
        Args:
            local_report: This rank's local schedule report
            
        Returns:
            List of reports from all ranks, ordered by rank
        """
        if not self.config.enable_attention_dp:
            # Non-ADP: only one "rank" for scheduling purposes
            return [local_report]
        
        # Convert to sync data (list of ints)
        local_data = local_report.to_sync_data()
        
        # AllGather across TP ranks (ADP uses TP dimension)
        all_data = self.dist.tp_allgather(local_data)
        
        # Convert back to LocalScheduleReport objects
        reports = [LocalScheduleReport.from_sync_data(data) for data in all_data]
        
        # Sort by rank to ensure consistent ordering
        reports.sort(key=lambda r: r.rank)
        
        return reports
    
    def _dequeue_waiting_requests(self, max_slots: int) -> List[RequestQueueItem]:
        """
        Dequeue requests from waiting_queue up to max_slots.
        
        Accounts for child requests (num_return_sequences > 1).
        
        Args:
            max_slots: Maximum number of batch slots to fill
            
        Returns:
            List of RequestQueueItem dequeued from waiting_queue
        """
        if not self.waiting_queue or max_slots <= 0:
            return []
        
        requests = []
        req_count = 0
        while req_count < max_slots and self.waiting_queue:
            req_item = self.waiting_queue[0]
            num_children = len(req_item.child_req_ids) if req_item.child_req_ids else 0
            if (req_count + 1 + num_children) > max_slots:
                break
            requests.append(self.waiting_queue.popleft())
            req_count += (1 + num_children)
        
        return requests
    
    def _get_waiting_requests(self, remaining_slots: int) -> List['LlmRequest']:
        """
        Get requests from waiting_queue for non-ADP mode.
        
        All ranks get the same requests (no distribution needed).
        
        Args:
            remaining_slots: Number of available batch slots
            
        Returns:
            List of LlmRequests from waiting_queue
        """
        requests_to_fetch = self._dequeue_waiting_requests(remaining_slots)
        if not requests_to_fetch:
            return []
        
        self.num_fetch_requests_cur_rank += len(requests_to_fetch)
        return self._merge_and_validate(requests_to_fetch)
    
    def _merge_and_validate(
        self,
        request_items: List[RequestQueueItem],
    ) -> List['LlmRequest']:
        """
        Merge RequestQueueItems to LlmRequests and validate.
        
        Invalid requests are filtered out via validate_request_cb.
        """
        llm_requests = merge_requests(
            request_items, self.dist, self.should_exclude_last_generation_logits
        )
        
        if self._validate_request is None:
            return llm_requests
        
        # Filter out invalid requests
        return [req for req in llm_requests if not self._validate_request(req)]
    
    @nvtx_range("_assign_waiting_requests")
    def _assign_waiting_requests(
        self,
        all_reports: List[LocalScheduleReport],
    ) -> List['LlmRequest']:
        """
        Phase 2: Global Assignment.
        
        Delegates to self.assignment_strategy to decide which requests from 
        waiting_queue should be assigned to which rank.
        
        Supports two strategies:
        - LegacyADPAssignmentStrategy: Compatible with original MinHeapADPBalancer
        - CapacityAwareAssignmentStrategy: Uses remaining capacity from reports
        
        Args:
            all_reports: Reports from all ranks (from _sync_schedule_reports)
            
        Returns:
            List of LlmRequests assigned to this rank
        """
        # Calculate how many requests we can potentially assign
        total_remaining_slots = sum(r.remaining_batch_slots for r in all_reports)
        
        # Dequeue requests from waiting_queue
        requests_to_assign = self._dequeue_waiting_requests(total_remaining_slots)
        if not requests_to_assign:
            return []
        
        # Use strategy to assign requests
        strategy_result = self.assignment_strategy.assign(
            waiting_requests=requests_to_assign,
            all_reports=all_reports,
            config=self.config,
        )
        
        # Put unassigned requests back to waiting_queue
        for req_item in reversed(strategy_result.unassigned_items):
            self.waiting_queue.appendleft(req_item)
        
        # Convert this rank's assigned items to LlmRequests
        cur_rank = self.dist.tp_rank if self.config.enable_attention_dp else self.dist.rank
        cur_items = strategy_result.get_items_for_rank(cur_rank)
        
        if cur_items:
            self.num_fetch_requests_cur_rank += len(cur_items)
            return self._merge_and_validate(cur_items)
        return []
    
    # ========== Schedule Phase (Two-Phase Implementation) ==========
    
    def _schedule(
        self,
        ctx: ScheduleContext,
    ) -> PostScheduleContext:
        """
        Main scheduling phase using two-phase approach.
        
        Phase 1: Local Scheduling
            - Each rank independently schedules its active_requests
            - Reports remaining capacity
            
        Sync Point: 
            - Gather reports from all ranks
            
        Phase 2: Global Assignment
            - Assign requests from waiting_queue based on reports
            - Late binding: optimal distribution based on current state
        """
        active_requests = ctx.active_requests
        max_batch_size = getattr(self, '_current_max_batch_size', self.config.max_batch_size)
        max_num_tokens = getattr(self, '_current_max_num_tokens', self.config.max_num_tokens)
        
        # Create resource context at iteration entry - shared between Phase 1 and Phase 2
        resource_context = SchedulingResourceContext()
        
        # Phase 1: Local Scheduling
        local_report, scheduled, fitting_disagg_gen_init = self._schedule_active_requests(
            active_requests=active_requests,
            inflight_request_ids=ctx.inflight_request_ids,
            max_batch_size=max_batch_size,
            max_num_tokens=max_num_tokens,
            resource_context=resource_context,
        )
        
        # ========== Phase 2: Get new requests and schedule them ==========
        if self.config.enable_attention_dp:
            # ADP: Sync reports and assign requests across ranks
            all_reports = self._sync_schedule_reports(local_report)
            new_requests = self._assign_waiting_requests(all_reports) if self.waiting_queue else []
            
            # ADP dummy padding
            dummy = self._pad_attention_dp_dummy(all_reports)
            if dummy is not None:
                new_requests.append(dummy)
        else:
            # Non-ADP: Directly get requests from waiting_queue (all ranks get the same)
            new_requests = self._get_waiting_requests(local_report.remaining_batch_slots)
        
        # Schedule newly assigned requests
        if new_requests:
            # Setup drafter for new requests (they missed the _pre_schedule drafter setup)
            self._set_draft_tokens_for_requests(new_requests, ctx.max_draft_tokens)
            # Add to active_requests for future iterations
            active_requests.extend(new_requests)
            
            # Try to schedule newly assigned requests using the same resource_context
            # This ensures Phase 2 sees Phase 1's consumed resources
            if local_report.remaining_batch_slots > 0:
                additional_context, additional_generation = self._try_schedule_new_requests(
                    new_requests=new_requests,
                    inflight_request_ids=ctx.inflight_request_ids,
                    resource_context=resource_context,
                )
                scheduled.context_requests.extend(additional_context)
                scheduled.generation_requests.extend(additional_generation)
        
        # ADP context balancing
        context_requests = scheduled.context_requests
        generation_requests = scheduled.generation_requests
        
        if self.config.enable_attention_dp and self.config.enable_adp_balance:
            context_requests = self._balance_adp_requests(context_requests, generation_requests)
            scheduled.context_requests = list(context_requests)
        
        # ========== Non-ADP batch waiting ==========
        should_check_waiting = (
            not self.config.enable_attention_dp and
            self._enable_batch_waiting and
            len(context_requests) > 0 and
            len(generation_requests) > 0
        )
        if should_check_waiting:
            context_requests = self._waiting_requests(context_requests, generation_requests)
            scheduled.context_requests = list(context_requests)
        
        return PostScheduleContext(
            scheduled=scheduled,
            fitting_disagg_gen_init=fitting_disagg_gen_init,
            num_fitting_requests=len(scheduled.context_requests) + len(scheduled.generation_requests),
            use_spec_decode=ctx.use_spec_decode,
            max_draft_tokens=ctx.max_draft_tokens,
            iter_stats=ctx.iter_stats,
        )
    
    def _try_schedule_new_requests(
        self,
        new_requests: List['LlmRequest'],
        inflight_request_ids: Set[int],
        resource_context: SchedulingResourceContext,
    ) -> Tuple[List['LlmRequest'], List['LlmRequest']]:
        """
        Phase 2: Try to schedule newly assigned requests using the same resource_context.
        
        This allows requests assigned in Phase 2 to be scheduled in the
        same iteration, sharing resource state with Phase 1.
        
        Args:
            new_requests: Requests assigned to this rank in Phase 2
            inflight_request_ids: Request IDs currently in pipeline (PP)
            resource_context: The same resource context from Phase 1
            
        Returns:
            (context_requests, generation_requests) that can be scheduled
        """
        assert resource_context is not None, "resource_context must be provided by caller"
        
        if not new_requests:
            return [], []
        
        # Use the same resource_context so Phase 2 sees Phase 1's consumed resources
        # Capacity scheduler checks against resource_context.get_scheduled_count()
        fitting_requests, _, _ = self.capacity_scheduler.schedule_request(
            new_requests, resource_context)
        
        if not fitting_requests:
            return [], []
        
        # Micro-batch scheduler also uses the same context
        # It will update resource_context.context_requests and generation_requests
        context_requests, generation_requests = self.micro_batch_scheduler.schedule(
            fitting_requests, inflight_request_ids, resource_context)
        
        return list(context_requests), list(generation_requests)
    
    def _balance_adp_requests(
        self,
        context_requests: List['LlmRequest'],
        generation_requests: List['LlmRequest'],
    ) -> List['LlmRequest']:
        """Balance context requests across ADP ranks."""
        balanced_context_requests = context_requests
        num_ctx = len(context_requests)
        num_gen = len(generation_requests)
        num_tokens = sum(len(req.get_tokens(0)) for req in context_requests) + num_gen
        
        all_data = self.dist.tp_allgather([num_ctx, num_gen, num_tokens])
        all_num_ctx = [d[0] for d in all_data]
        all_num_gen = [d[1] for d in all_data]
        
        all_have_ctx = all(n > 0 for n in all_num_ctx)
        all_have_gen = all(n > 0 for n in all_num_gen)
        all_have_free_slots = all(n < self.config.max_batch_size for n in all_num_gen)
        
        if all_have_free_slots and all_have_ctx:
            self._adp_ctx_waiting_iters = 0
            if all_have_gen:
                if self._adp_ctx_batching_wait_iters < self.config.adp_batching_wait_iters:
                    self._adp_ctx_batching_wait_iters += 1
                    balanced_context_requests = []
                else:
                    self._adp_ctx_batching_wait_iters = 0
        else:
            self._adp_ctx_waiting_iters += 1
            balanced_context_requests = []
            timeout_reached = self._adp_ctx_waiting_iters >= self.config.adp_timeout_iters
            if timeout_reached or not all_have_gen:
                self._adp_ctx_waiting_iters = 0
                balanced_context_requests = context_requests
        
        return balanced_context_requests
    
    def _waiting_requests(
        self,
        context_requests: List['LlmRequest'],
        generation_requests: List['LlmRequest'],
    ) -> List['LlmRequest']:
        """
        Return an empty list if scheduled requests fulfill the waiting conditions, 
        otherwise return the original context requests.
        
        Waiting conditions:
        - The number of scheduled tokens (both context and generation) is smaller than 
          `batch_wait_max_tokens_ratio * max_num_tokens`
        - The number of waiting iterations is smaller than `batch_wait_timeout_iters`.
        """
        # Calculate token counts
        num_scheduled_ctx_tokens = sum(
            len(ctx_req.get_tokens(0)) for ctx_req in context_requests
        )
        num_scheduled_gen_tokens = sum(
            1 + getattr(gen_req, 'num_draft_tokens', 0)
            for gen_req in generation_requests
        )
        num_scheduled_tokens = num_scheduled_ctx_tokens + num_scheduled_gen_tokens
        
        # Check waiting condition
        should_waiting = (
            self._batch_waiting_iters < self.config.batch_wait_timeout_iters and
            num_scheduled_tokens < self.config.batch_wait_max_tokens_ratio * self.config.max_num_tokens
        )
        if should_waiting:
            self._batch_waiting_iters += 1
            return []
        
        self._batch_waiting_iters = 0
        return context_requests
    
    # ========== Post-Schedule Phase ==========
    
    def _post_schedule(
        self,
        ctx: PostScheduleContext,
    ) -> BatchScheduleResult:
        """Post-schedule phase: finalize and prepare resources."""
        if not ctx.use_spec_decode:
            for request in ctx.scheduled.all_requests():
                request.py_disable_speculative_decoding = True
        
        if self._prepare_disagg_gen_init is not None and ctx.fitting_disagg_gen_init:
            self._prepare_disagg_gen_init(ctx.fitting_disagg_gen_init)
        
        logger.debug(
            f'Scheduled {len(ctx.scheduled.context_requests)} context requests and '
            f'{len(ctx.scheduled.generation_requests)} generation requests'
        )
        
        return BatchScheduleResult(
            scheduled_batch=ctx.scheduled,
            iter_stats=ctx.iter_stats,
            fitting_disagg_gen_init=ctx.fitting_disagg_gen_init,
            num_fitting_requests=ctx.num_fitting_requests,
            is_shutdown=False,
            use_spec_decode=ctx.use_spec_decode,
            max_draft_tokens=ctx.max_draft_tokens,
        )


class DisaggContextScheduler(UnifiedSPMDScheduler):
    """Scheduler for Disaggregated Context (Prefill) Server."""
    pass


class DisaggGenerationScheduler(UnifiedSPMDScheduler):
    """Scheduler for Disaggregated Generation (Decode) Server."""
    pass
