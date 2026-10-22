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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.custom_ops.torch_custom_ops import MoERunner
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

pytestmark = pytest.mark.cpu_only


def _executor(events: list) -> PyExecutor:
    """An executor double whose shutdown() records the teardown order."""
    executor = object.__new__(PyExecutor)
    executor.dist = SimpleNamespace(rank=0, world_size=1, pp_size=1)
    executor.executor_request_queue = Mock()
    executor.shutdown_event = Mock()
    executor._profile_manager = Mock()
    executor.hang_detector = Mock()
    executor.hang_detector.detected.return_value = False
    executor.worker_thread = Mock()
    executor.worker_started = True
    executor.kv_connector_manager = None
    executor._shutdown_sleep_wakeup_listeners = Mock()
    executor.model_engine = Mock()
    executor.model_engine._release_cuda_graphs.side_effect = lambda: events.append(
        "release_cuda_graphs"
    )
    executor.draft_model_engine = None
    manager = Mock()
    manager.shutdown.side_effect = lambda: events.append("resource_manager_shutdown")
    executor.resource_manager = SimpleNamespace(resource_managers={"kv": manager})
    executor.virtual_memory_pools = None
    executor.sampler = None
    executor.dwdp_manager = None
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


def test_kv_cache_estimation_shutdown_keeps_moe_runner_workspaces(monkeypatch):
    events = []
    _record_workspace_release(monkeypatch, events)

    # The estimation executor's model, and the graphs it owns, are reused.
    _executor(events).shutdown_for_kv_cache_estimation()

    assert events == ["release_cuda_graphs", "resource_manager_shutdown"]
