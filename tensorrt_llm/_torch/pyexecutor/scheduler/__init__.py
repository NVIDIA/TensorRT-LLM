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

File layout:
    scheduler.py         — Interfaces (ABC) and shared data structures
    simple_scheduler.py  — SimpleScheduler (C++ binding wrappers)
    unified_scheduler.py — SimpleUnifiedScheduler (pure-Python)
    adp_router.py        — Attention-DP routing
    waiting_queue.py     — FCFS waiting queue
"""
# ruff: noqa: E402

# When mypyc compiles unified_scheduler.py from pyexecutor/ cwd, the .so
# references "scheduler" as a top-level package. Register this package under
# that name so the mypyc runtime module can be found. This MUST run before
# unified_scheduler is imported below.
import sys as _sys  # isort:skip

_sys.modules.setdefault("scheduler", _sys.modules[__name__])
del _sys

# Re-export from scheduler.py (interfaces)
from .adp_router import ADPRouter, DefaultADPRouter, RankState
from .scheduler import (
    CapacityScheduler,
    MicroBatchScheduler,
    RequestList,
    RequestScheduler,
    ScheduledRequests,
    SchedulerOutput,
    SerializableSchedulerOutput,
)

# Re-export from simple_scheduler.py (C++ binding implementations)
from .simple_scheduler import (
    BindCapacityScheduler,
    BindMicroBatchScheduler,
    KVCacheV2DummyScheduler,
    SimpleScheduler,
)

# Re-export from unified_scheduler.py (Python-only implementation)
from .unified_scheduler import (
    PyCapacityScheduler,
    ScheduleResult,
    ScheduleStepConfig,
    SimpleUnifiedScheduler,
    UnifiedScheduleStepOutput,
)

# Re-export from waiting_queue.py
from .waiting_queue import FCFSWaitingQueue, WaitingQueue, create_waiting_queue

__all__ = [
    # Interfaces
    "CapacityScheduler",
    "MicroBatchScheduler",
    "RequestList",
    "RequestScheduler",
    "ScheduledRequests",
    "SchedulerOutput",
    "SerializableSchedulerOutput",
    # SimpleScheduler (C++ bindings)
    "BindCapacityScheduler",
    "BindMicroBatchScheduler",
    "KVCacheV2DummyScheduler",
    "SimpleScheduler",
    # SimpleUnifiedScheduler (Python)
    "PyCapacityScheduler",
    "ScheduleResult",
    "ScheduleStepConfig",
    "SimpleUnifiedScheduler",
    "UnifiedScheduleStepOutput",
    # ADP
    "ADPRouter",
    "DefaultADPRouter",
    "RankState",
    # Waiting queues
    "FCFSWaitingQueue",
    "WaitingQueue",
    "create_waiting_queue",
]
