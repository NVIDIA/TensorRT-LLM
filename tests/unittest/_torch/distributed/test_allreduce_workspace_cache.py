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
"""Process-wide caching of the custom AllReduce IPC workspaces.

Allocating a workspace runs a TP-wide host collective, so every rank must take
the same allocate-or-reuse decision. A rank cannot know which thread its peers
build their model on (RpcWorker builds rank 0 on an RPC executor thread and the
other ranks on their main thread), so a cache keyed per thread lets a reused
worker skip the collective while a peer enters it, and both ranks deadlock.
"""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from tensorrt_llm._torch.distributed import ops
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def allocations(monkeypatch: pytest.MonkeyPatch) -> list:
    """Replace the IPC allocators with recorders and start from empty caches."""
    calls = []

    def allocate_fusion(mapping, size):
        calls.append(("fusion", mapping))
        return [], object()

    def allocate_lowprecision(mapping, size):
        calls.append(("lowprecision", mapping))
        return [], object()

    helper = ops.CustomAllReduceHelper
    monkeypatch.setattr(helper, "allocate_allreduce_fusion_workspace", allocate_fusion)
    monkeypatch.setattr(helper, "allocate_lowprecision_workspace", allocate_lowprecision)
    monkeypatch.setattr(helper, "max_workspace_size_auto", lambda *args, **kwargs: 1)
    monkeypatch.setattr(helper, "max_workspace_size_lowprecision", lambda *args: 1)
    monkeypatch.setattr(helper, "initialize_lowprecision_buffers", lambda *args: None)
    monkeypatch.setattr(ops, "_allreduce_workspaces", {})
    monkeypatch.setattr(ops, "_lowprecision_allreduce_workspaces", {})
    monkeypatch.setattr(ops, "_allreduce_workspace_locks", {})
    return calls


def _run_on_new_thread(fn, *args):
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(fn, *args).result()


def _tp2_mapping() -> Mapping:
    return Mapping(world_size=2, rank=0, gpus_per_node=2, tp_size=2)


def test_workspace_is_reused_from_another_thread(allocations: list) -> None:
    mapping = _tp2_mapping()

    first = ops.get_allreduce_workspace(mapping)
    second = _run_on_new_thread(ops.get_allreduce_workspace, mapping)

    assert second is first
    assert allocations == [("fusion", mapping)]


def test_lowprecision_workspace_is_reused_from_another_thread(allocations: list) -> None:
    mapping = _tp2_mapping()

    ops.allocate_low_presicion_allreduce_workspace(mapping)
    _run_on_new_thread(ops.allocate_low_presicion_allreduce_workspace, mapping)

    assert allocations == [("lowprecision", mapping)]


def test_concurrent_first_use_allocates_once(allocations: list) -> None:
    mapping = _tp2_mapping()
    num_threads = 8
    barrier = threading.Barrier(num_threads)

    def get_after_barrier():
        barrier.wait()
        return ops.get_allreduce_workspace(mapping)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        workspaces = list(executor.map(lambda _: get_after_barrier(), range(num_threads)))

    assert all(workspace is workspaces[0] for workspace in workspaces)
    assert allocations == [("fusion", mapping)]


def test_distinct_mappings_get_distinct_workspaces(allocations: list) -> None:
    tp2 = _tp2_mapping()
    tp2_pp2 = Mapping(world_size=4, rank=0, gpus_per_node=4, tp_size=2, pp_size=2)

    assert ops.get_allreduce_workspace(tp2) is not ops.get_allreduce_workspace(tp2_pp2)
    assert allocations == [("fusion", tp2), ("fusion", tp2_pp2)]
