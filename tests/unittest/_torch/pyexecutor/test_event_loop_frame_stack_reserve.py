# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""PyExecutor._event_loop_wrapper runs the event loop under the frame-stack reserve (no GPU)."""

import contextlib
import sys
import threading
import types

import pytest

from tensorrt_llm import _utils
from tensorrt_llm._torch.pyexecutor import py_executor as pe

pytestmark = pytest.mark.cpu_only


def test_event_loop_runs_directly_under_the_reserve_frame(monkeypatch):
    # Same bypass-__init__ pattern as test_hang_detector_kill.py: only the
    # attributes _event_loop_wrapper reads are set.
    monkeypatch.setattr(pe, "host_profiler_context", lambda enable: contextlib.nullcontext())
    monkeypatch.setattr(pe, "customized_gc_thresholds", lambda threshold: contextlib.nullcontext())
    ex = pe.PyExecutor.__new__(pe.PyExecutor)
    ex.dist = types.SimpleNamespace(world_size=1)
    ex.garbage_collection_gen0_threshold = None
    ex.is_shutdown = False
    ex._event_loop_completed = False
    ex._event_loop_error_delivered = threading.Event()
    ex._executor_loop_cleanup = lambda: None
    callers = []
    ex.event_loop = lambda: callers.append(sys._getframe(1).f_code)

    ex._event_loop_wrapper()

    # The real reserve function, with its rebound code object, is the loop's caller.
    assert callers == [_utils.run_with_frame_stack_reserve.__code__]
