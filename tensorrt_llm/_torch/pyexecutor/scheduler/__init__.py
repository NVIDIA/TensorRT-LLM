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
Unified SPMD Scheduler Module

This module provides a refactored scheduler architecture following the SPMD
(Single Program, Multiple Data) paradigm.

Module Structure:
-----------------
- types.py: Public types (SchedulerConfig, ScheduledRequests, BatchScheduleResult)
- request_queue.py: Pure request storage queue
- request_fetcher.py: Queue fetch operations
- request_utils.py: Pure functions for broadcast/merge
- request_assignment.py: Pluggable ADP assignment strategies
- resource_context.py: Per-iteration resource tracking for two-phase scheduling
- base.py: Abstract interface + internal context types
- unified.py: Main scheduler implementation

Scheduling Flow:
----------------
    UnifiedSPMDScheduler.prepare_and_schedule_batch():
    
    Pre-schedule (_pre_schedule)
        → Fetch from queue (rank 0)
        → Broadcast to all ranks
        → Handle special items (shutdown, cancel, control)
        → ADP distribution (via distribution_strategy)
        → Disagg transfer check
        → ADP dummy padding
        → Drafter setup
    
    Schedule (_schedule)
        → Capacity check
        → Micro-batch selection
        → Context balancing (ADP)
    
    Post-schedule (_post_schedule)
        → Disagg gen init preparation

Usage:
------
    from tensorrt_llm._torch.pyexecutor.scheduler import (
        RequestQueue,
        UnifiedSPMDScheduler,
        SchedulerConfig,
        MinHeapADPBalancer,  # Optional: custom distribution strategy
    )

    scheduler = UnifiedSPMDScheduler(
        dist=dist,
        config=SchedulerConfig(...),
        capacity_scheduler=capacity_scheduler,
        micro_batch_scheduler=micro_batch_scheduler,
        request_queue=RequestQueue(),
    )
    
    result = scheduler.prepare_and_schedule_batch(
        active_requests=active_requests,
        inflight_request_ids=inflight_ids,
    )
"""

# Types (public I/O types)
from .types import (
    SchedulerConfig,
    ScheduledRequests,
    BatchScheduleResult,
    SchedulerOutput,
)

# Request Queue (from pyexecutor directory)
from ..request_queue import (
    RequestQueue,
    RequestQueueItem,
    SHUTDOWN_REQUEST_ID,
    CONTROL_REQUEST_ID,
)

# Request Fetcher (from pyexecutor directory)
from ..request_fetcher import (
    RequestFetcher,
    FetcherConfig,
)

# Request Utilities (from pyexecutor directory)
from ..request_utils import (
    RequestBroadcaster,
    collect_py_objects,
    attach_py_objects,
    merge_requests,
)

# Request Assignment Strategy (pluggable load balancing for Attention DP)
from .request_assignment import (
    WaitingRequestAssignmentStrategy,
    LegacyADPAssignmentStrategy,
    CapacityAwareAssignmentStrategy,
    WaitingAssignmentResult,
)

# Local Scheduler (capacity + micro-batch scheduling)
from .local_scheduler import (
    PyCapacityScheduler,
    PyMicroBatchScheduler,
    SchedulerPolicyBase,
    MaxRequestsPolicy,
    GuaranteedNoEvictPolicy,
    MaxUtilizationPolicy,
)

# Resource Context (per-iteration resource tracking)
from .resource_context import SchedulingResourceContext

# Base class and internal context types
from .base import (
    BaseScheduler,
    ScheduleContext,
    PostScheduleContext,
)

# Unified Scheduler (main implementation)
from .unified import (
    UnifiedSPMDScheduler,
    DisaggContextScheduler,
    DisaggGenerationScheduler,
)

__all__ = [
    # Types (public I/O)
    "SchedulerConfig",
    "ScheduledRequests",
    "BatchScheduleResult",
    "SchedulerOutput",
    # Request Queue
    "RequestQueue",
    "RequestQueueItem",
    "SHUTDOWN_REQUEST_ID",
    "CONTROL_REQUEST_ID",
    # Request Fetcher
    "RequestFetcher",
    "FetcherConfig",
    # Request Utilities
    "RequestBroadcaster",
    "collect_py_objects",
    "attach_py_objects",
    "merge_requests",
    # Request Assignment Strategy
    "WaitingRequestAssignmentStrategy",
    "LegacyADPAssignmentStrategy",
    "CapacityAwareAssignmentStrategy",
    "WaitingAssignmentResult",
    # Local Scheduler
    "PyCapacityScheduler",
    "PyMicroBatchScheduler",
    "SchedulerPolicyBase",
    "MaxRequestsPolicy",
    "GuaranteedNoEvictPolicy",
    "MaxUtilizationPolicy",
    # Resource Context
    "SchedulingResourceContext",
    # Base class and internal context types
    "BaseScheduler",
    "ScheduleContext",
    "PostScheduleContext",
    # Unified Scheduler
    "UnifiedSPMDScheduler",
    "DisaggContextScheduler",
    "DisaggGenerationScheduler",
]
