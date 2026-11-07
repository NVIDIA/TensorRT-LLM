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
"""PyExecutor.shutdown releases the process-wide MoE runner workspaces."""

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.custom_ops.torch_custom_ops import MoERunner
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManager, ResourceManagerType

pytestmark = pytest.mark.cpu_only


def _executor(events: list) -> PyExecutor:
    """An executor double whose teardown records its order."""
    engine = SimpleNamespace()
    engine.release_executor_resources = Mock(
        side_effect=lambda *, release_model_owned_graphs: events.append("release_cuda_graphs")
    )
    cache = SimpleNamespace(shutdown=lambda: events.append("resource_manager_shutdown"))
    executor = object.__new__(PyExecutor)
    executor._closed = False
    executor.enable_attention_dp = False
    executor._kv_connector_failed = False
    executor._resources_released = False
    executor._event_loop_error = None
    executor._fatal_error = None
    executor.model_engine = engine
    executor.draft_model_engine = None
    executor.execution_stream = object()
    executor.dist = SimpleNamespace(pp_size=1)
    executor.worker_started = True
    executor._mpi_progress_thread = None
    executor._mpi_progress_thread_exit_hook = None
    executor.worker_thread = Mock()
    executor.executor_request_queue = Mock()
    executor.shutdown_event = threading.Event()
    executor.shutdown_event.set()
    executor._profile_manager = Mock()
    executor.hang_detector = Mock()
    executor.hang_detector.detected.return_value = False
    executor._shutdown_sleep_wakeup_listeners = Mock()
    executor.encoder_launch_executor = None
    executor.kv_connector_manager = None
    executor.kv_cache_transceiver = None
    executor.resource_manager = ResourceManager({ResourceManagerType.KV_CACHE_MANAGER: cache})
    executor.virtual_memory_pools = None
    executor.sampler = object()
    executor.dwdp_manager = None
    executor.pending_encoder_steps = []
    executor.active_requests = []
    executor.responses = {}
    executor.result_wait_queues = {}
    executor._pending_transfer_responses = []
    executor._pending_response_terminations = []
    executor.request_accumulated = []
    executor.control_requests = []
    executor.waiting_queue = []
    executor._pending_adp_dummy_request = None
    return executor


def _record_workspace_release(monkeypatch, events: list) -> None:
    monkeypatch.setattr(
        MoERunner, "clear_all_workspaces", lambda: events.append("clear_moe_workspaces")
    )


def test_shutdown_clears_moe_runner_workspaces_before_graph_release(monkeypatch):
    events = []
    _record_workspace_release(monkeypatch, events)

    _executor(events).shutdown()

    # The cached C++ runners outlive the executor, so their workspaces must be
    # released here, while the graph memory pools that may back them still exist.
    assert events == [
        "clear_moe_workspaces",
        "release_cuda_graphs",
        "resource_manager_shutdown",
    ]


def test_profiling_release_keeps_moe_runner_workspaces(monkeypatch):
    events = []
    _record_workspace_release(monkeypatch, events)

    # The profiling executor's model, and the graphs it owns, are reused.
    _executor(events).release_resources()

    assert events == ["release_cuda_graphs", "resource_manager_shutdown"]
