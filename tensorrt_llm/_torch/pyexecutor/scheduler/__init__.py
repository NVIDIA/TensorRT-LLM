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
"""
Scheduler module for TensorRT-LLM PyExecutor.

This module contains:
- Request schedulers (capacity, micro-batch, unified)
- Waiting queues (FCFS)
"""

# Re-export from scheduler.py (C++ bindings path)
from .adp_router import ADPRouter, DefaultADPRouter, RankState
from .scheduler import (
    BindCapacityScheduler,
    BindMicroBatchScheduler,
    CapacityScheduler,
    KVCacheV2DummyScheduler,
    MicroBatchScheduler,
    RequestList,
    RequestScheduler,
    ScheduledRequests,
    SchedulerOutput,
    SerializableSchedulerOutput,
    SimpleScheduler,
)

# Re-export from unified_scheduler.py (Python-only path)
from .unified_scheduler import (
    PyCapacityScheduler,
    PyMicroBatchScheduler,
    ScheduleResult,
    ScheduleStepConfig,
    SimpleUnifiedScheduler,
)

# Re-export from waiting_queue.py
from .waiting_queue import FCFSWaitingQueue, WaitingQueue, create_waiting_queue

__all__ = [
    # Schedulers (C++ bindings path)
    "BindCapacityScheduler",
    "BindMicroBatchScheduler",
    "CapacityScheduler",
    "KVCacheV2DummyScheduler",
    "MicroBatchScheduler",
    "RequestList",
    "RequestScheduler",
    "ScheduledRequests",
    "SchedulerOutput",
    "SerializableSchedulerOutput",
    "SimpleScheduler",
    # Schedulers (Python-only path)
    "PyCapacityScheduler",
    "PyMicroBatchScheduler",
    "ScheduleResult",
    "ScheduleStepConfig",
    "SimpleUnifiedScheduler",
    # ADP
    "ADPRouter",
    "DefaultADPRouter",
    "RankState",
    # Waiting queues
    "FCFSWaitingQueue",
    "WaitingQueue",
    "create_waiting_queue",
]
