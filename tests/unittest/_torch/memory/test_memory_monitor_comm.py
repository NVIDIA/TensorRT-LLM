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
"""Communication buffers outside the PyTorch allocator.

Multicast buffers need several GPUs to create, so the providers' reading of
the owning objects is checked against stand-ins shaped like them. That
symmetric memory bypasses the caching allocator was measured separately on
torch 2.14 (256 MiB requested: memory_reserved +0, device +258 MiB).
"""

import pytest
import torch

from tensorrt_llm._torch.memory.monitor import tagging
from tensorrt_llm._torch.memory.monitor.providers import default_providers
from tensorrt_llm._torch.memory.monitor.providers.comm import (
    McastAllReduceProvider,
    SymmMemProvider,
)
from tensorrt_llm._torch.memory.monitor.schema import Basis, Category
from tensorrt_llm._torch.memory.monitor.summary import GROUP_OF, Group

pytestmark = pytest.mark.cpu_only

MB = 1 << 20


@pytest.fixture(autouse=True)
def _monitor_on(monkeypatch):
    monkeypatch.setenv("TLLM_MEM_MONITOR", "1")
    tagging.clear_claims()
    yield
    tagging.clear_claims()


class _SymmInstance:
    def __init__(self, nbytes):
        self.buffer = torch.empty(nbytes, dtype=torch.uint8)


class _AllReduce(torch.nn.Module):
    def __init__(self, inst):
        super().__init__()
        self.symm_mem_allreduce = inst


def test_symm_mem_walks_the_claimed_model_and_counts_shared_buffers_once():
    assert SymmMemProvider().available() is False, "no claimed model, nothing to walk"

    tagging.claim_module_memory(torch.nn.Sequential(_AllReduce(None)), "WEIGHTS")
    assert SymmMemProvider().collect() == [], "symmetric memory not in use"
    tagging.clear_claims()

    shared = _SymmInstance(8 * MB)
    model = torch.nn.Sequential(
        _AllReduce(shared), _AllReduce(shared), _AllReduce(_SymmInstance(2 * MB))
    )
    tagging.claim_module_memory(model, "WEIGHTS")
    entries = SymmMemProvider().collect()
    assert len(entries) == 1
    assert entries[0].size_bytes == 10 * MB
    assert entries[0].category is Category.COMM_SYMM_MEM
    # The allocator rounds up; the request is a lower bound.
    assert entries[0].basis is Basis.REQUESTED


def _workspace(nbytes):
    return {"uc_buffer": torch.empty(nbytes // 4, dtype=torch.float32), "buffer_size_bytes": 0}


def test_mcast_allreduce_counts_active_and_retired_workspaces(monkeypatch):
    from tensorrt_llm._torch.distributed.ops import MNNVLAllReduce

    kinds = {type(p) for p in default_providers()}
    assert SymmMemProvider in kinds and McastAllReduceProvider in kinds

    def workspaces(active, retired):
        monkeypatch.setattr(MNNVLAllReduce, "allreduce_mnnvl_workspaces", active)
        monkeypatch.setattr(MNNVLAllReduce, "allreduce_mnnvl_retired_workspaces", retired)
        return McastAllReduceProvider().collect()

    assert workspaces({}, {}) == [], "no workspace exists"

    # Retired by a resize but kept alive for captured graphs: still held.
    entries = workspaces({"tp0": _workspace(12 * MB)}, {"tp0": [_workspace(6 * MB)]})
    assert entries[0].size_bytes == 18 * MB
    assert entries[0].category is Category.COMM_NVLS
    assert entries[0].basis is Basis.REQUESTED

    # The view spans every Lamport buffer; reading buffer_size_bytes
    # alone would count one of three.
    ws = _workspace(12 * MB)
    ws["buffer_size_bytes"] = 4 * MB
    assert workspaces({"tp0": ws}, {})[0].size_bytes == 12 * MB


def test_nccl_window_counts_pooled_buffers_whether_or_not_checked_out(monkeypatch):
    from tensorrt_llm._torch.memory.monitor.providers.comm import NcclWindowProvider

    def collect(usage):
        monkeypatch.setattr(NcclWindowProvider, "_op", staticmethod(lambda: lambda _d: usage))
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
        return NcclWindowProvider().collect()

    (entry,) = collect([3, 96 * MB, 32 * MB])
    assert entry.size_bytes == 96 * MB
    assert GROUP_OF[entry.category] is Group.COMMUNICATION
    assert collect([0, 0, 0]) == [], "no buffers, no row"
