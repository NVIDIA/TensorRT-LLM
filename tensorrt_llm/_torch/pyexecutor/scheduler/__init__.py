# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Scheduler module for TensorRT-LLM PyExecutor.

This module contains:
- Request schedulers (capacity, micro-batch, unified)
- Waiting queues (FCFS)
"""

# Re-export from scheduler.py (interfaces and data structures)
from .adp_router import ADPRouter, DefaultADPRouter, RankState
from .scheduler import (
    RequestList,
    RequestScheduler,
    ScheduledRequests,
    SchedulerOutput,
    ScheduleStepConfig,
    ScheduleStepResult,
    SerializableSchedulerOutput,
)
from .scheduler_v2 import KVCacheV2Scheduler
from .simple_scheduler import BindCapacityScheduler, BindMicroBatchScheduler, SimpleScheduler
from .unified_scheduler import PyCapacityScheduler, UnifiedScheduler

# Re-export from waiting_queue.py
from .waiting_queue import FCFSWaitingQueue, WaitingQueue, create_waiting_queue

__all__ = [
    # Schedulers
    "BindCapacityScheduler",
    "BindMicroBatchScheduler",
    "KVCacheV2Scheduler",
    "PyCapacityScheduler",
    "RequestList",
    "RequestScheduler",
    "ScheduleStepConfig",
    "ScheduleStepResult",
    "ScheduledRequests",
    "SchedulerOutput",
    "SerializableSchedulerOutput",
    "SimpleScheduler",
    "UnifiedScheduler",
    # ADP
    "ADPRouter",
    "DefaultADPRouter",
    "RankState",
    # Waiting queues
    "FCFSWaitingQueue",
    "WaitingQueue",
    "create_waiting_queue",
]
