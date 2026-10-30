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


class TestSymmMem:
    def test_walks_the_claimed_model_and_counts_shared_buffers_once(self):
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

    def test_unavailable_without_a_model(self):
        assert SymmMemProvider().available() is False

    def test_nothing_when_symm_mem_is_not_in_use(self):
        model = torch.nn.Sequential(_AllReduce(None))
        tagging.claim_module_memory(model, "WEIGHTS")
        assert SymmMemProvider().collect() == []


class TestMcastAllReduce:
    def _workspace(self, nbytes):
        return {"uc_buffer": torch.empty(nbytes // 4, dtype=torch.float32), "buffer_size_bytes": 0}

    def test_counts_active_and_retired_workspaces(self, monkeypatch):
        from tensorrt_llm._torch.distributed.ops import MNNVLAllReduce

        monkeypatch.setattr(
            MNNVLAllReduce,
            "allreduce_mnnvl_workspaces",
            {"tp0": self._workspace(12 * MB)},
        )
        # Retired by a resize but kept alive for captured graphs: still held.
        monkeypatch.setattr(
            MNNVLAllReduce,
            "allreduce_mnnvl_retired_workspaces",
            {"tp0": [self._workspace(6 * MB)]},
        )
        entries = McastAllReduceProvider().collect()
        assert entries[0].size_bytes == 18 * MB
        assert entries[0].category is Category.COMM_NVLS
        assert entries[0].basis is Basis.REQUESTED

    def test_sized_from_the_whole_unicast_view(self, monkeypatch):
        # The view spans every Lamport buffer; reading buffer_size_bytes
        # alone would count one of three.
        from tensorrt_llm._torch.distributed.ops import MNNVLAllReduce

        ws = self._workspace(12 * MB)
        ws["buffer_size_bytes"] = 4 * MB
        monkeypatch.setattr(MNNVLAllReduce, "allreduce_mnnvl_workspaces", {"tp0": ws})
        monkeypatch.setattr(MNNVLAllReduce, "allreduce_mnnvl_retired_workspaces", {})
        assert McastAllReduceProvider().collect()[0].size_bytes == 12 * MB

    def test_nothing_when_no_workspace_exists(self, monkeypatch):
        from tensorrt_llm._torch.distributed.ops import MNNVLAllReduce

        monkeypatch.setattr(MNNVLAllReduce, "allreduce_mnnvl_workspaces", {})
        monkeypatch.setattr(MNNVLAllReduce, "allreduce_mnnvl_retired_workspaces", {})
        assert McastAllReduceProvider().collect() == []


def test_both_are_in_the_default_set():
    kinds = {type(p) for p in default_providers()}
    assert SymmMemProvider in kinds and McastAllReduceProvider in kinds


class TestNcclWindow:
    def _provider(self, monkeypatch, usage):
        from tensorrt_llm._torch.memory.monitor.providers.comm import NcclWindowProvider

        monkeypatch.setattr(NcclWindowProvider, "_op", staticmethod(lambda: lambda _d: usage))
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
        return NcclWindowProvider()

    def test_pooled_buffers_count_whether_or_not_checked_out(self, monkeypatch):
        (entry,) = self._provider(monkeypatch, [3, 96 * MB, 32 * MB]).collect()
        assert entry.size_bytes == 96 * MB
        assert GROUP_OF[entry.category] is Group.COMMUNICATION

    def test_no_buffers_no_row(self, monkeypatch):
        assert self._provider(monkeypatch, [0, 0, 0]).collect() == []
