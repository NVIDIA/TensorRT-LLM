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
the owning objects is checked against stand-ins shaped like them.
"""

import threading

import pytest
import torch
from memory_monitor_test_utils import MB

from tensorrt_llm._torch.memory.monitor import tagging
from tensorrt_llm._torch.memory.monitor.providers import default_providers
from tensorrt_llm._torch.memory.monitor.providers.comm import (
    IpcMemoryProvider,
    McastAllReduceProvider,
    SymmMemProvider,
)
from tensorrt_llm._torch.memory.monitor.schema import Basis, Category
from tensorrt_llm._torch.memory.monitor.summary import GROUP_OF, Group


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


@pytest.mark.cpu_only
def test_symm_mem_walks_the_claimed_model_and_counts_shared_buffers_once():
    assert SymmMemProvider().available() is False, "no claimed model, nothing to walk"

    # Claims are weak: keep the model alive, or the provider would find no
    # claimed model rather than one without symmetric memory.
    unused = torch.nn.Sequential(_AllReduce(None))
    tagging.claim_module_memory(unused, "WEIGHTS")
    assert SymmMemProvider().available(), "the claimed model is walked"
    assert SymmMemProvider().collect() == [], "symmetric memory not in use"
    del unused
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


@pytest.mark.cpu_only
def test_mcast_allreduce_counts_active_and_retired_workspaces(monkeypatch):
    from tensorrt_llm._torch.distributed.ops import MNNVLAllReduce

    kinds = {type(p) for p in default_providers()}
    assert {SymmMemProvider, McastAllReduceProvider, IpcMemoryProvider} <= kinds

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

    # The executor thread retires another mapping's workspace while a report
    # reads the dict; the report must not see the dict change size under it.
    retired = {}

    class _RetiredWhileRead(list):
        def __iter__(self):
            retired.setdefault("tp1", []).append(_workspace(MB))
            return super().__iter__()

    retired["tp0"] = _RetiredWhileRead([_workspace(6 * MB)])
    assert workspaces({}, retired)[0].size_bytes == 6 * MB


@pytest.mark.cpu_only
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


def _ipc_stand_in(label, device, nbytes):
    from tensorrt_llm._ipc_utils import IpcMemory

    ipc = object.__new__(IpcMemory)
    ipc.label, ipc.device, ipc.nbytes, ipc.open_ipc = label, device, nbytes, False
    return ipc


@pytest.mark.cpu_only
def test_ipc_buffers_are_counted_by_purpose_on_this_device(monkeypatch):
    from tensorrt_llm._ipc_utils import IpcMemory

    monkeypatch.setattr(IpcMemory, "_live", {})
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    held = [
        _ipc_stand_in("custom allreduce workspace", 0, 64 * MB),
        _ipc_stand_in("custom allreduce workspace", 0, 2 * MB),
        _ipc_stand_in("TP peer exchange buffers", 0, 4 * MB),
        # Another rank's GPU in the same process is not this device's memory.
        _ipc_stand_in("custom allreduce workspace", 1, 64 * MB),
    ]
    for ipc in held:
        ipc._track()
    entries = {e.owner: e for e in IpcMemoryProvider().collect()}
    assert entries["custom allreduce workspace"].size_bytes == 66 * MB
    assert entries["TP peer exchange buffers"].size_bytes == 4 * MB
    assert GROUP_OF[entries["TP peer exchange buffers"].category] is Group.COMMUNICATION
    assert entries["TP peer exchange buffers"].basis is Basis.PHYSICAL_BACKING
    held.clear()
    assert IpcMemoryProvider().collect() == [], "freed buffers leave the report"


@pytest.mark.cpu_only
def test_ipc_buffers_can_be_read_while_other_threads_create_and_free_them(monkeypatch):
    """Reports read the live buffers on their own threads while the executor creates and frees them."""
    from tensorrt_llm._ipc_utils import IpcMemory

    monkeypatch.setattr(IpcMemory, "_live", {})
    stop = threading.Event()

    def churn():
        while not stop.is_set():
            held = [_ipc_stand_in("churn", 0, MB) for _ in range(64)]
            for ipc in held:
                ipc._track()
            held.clear()

    writer = threading.Thread(target=churn)
    writer.start()
    try:
        for _ in range(20000):
            count, nbytes = IpcMemory.live_allocations(0).get("churn", (0, 0))
            assert nbytes == count * MB
    finally:
        stop.set()
        writer.join()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_ipc_memory_records_the_size_it_allocated():
    from tensorrt_llm._ipc_utils import IpcMemory
    from tensorrt_llm._torch.memory.monitor.capability import nvml_process_usage
    from tensorrt_llm.mapping import Mapping

    device = torch.cuda.current_device()
    before = IpcMemory.live_allocations(device).get("unit", (0, 0))
    # This process's usage, which other processes on the GPU cannot change.
    used_before = nvml_process_usage(device)[0]
    ipc = IpcMemory(Mapping(world_size=1, rank=0, tp_size=1), 3 * MB, True, "unit")
    assert ipc.open_ipc
    count, nbytes = IpcMemory.live_allocations(device)["unit"]
    # Rounded up to the 2 MiB IPC alignment ...
    assert (count, nbytes) == (before[0] + 1, before[1] + 4 * MB)
    used_after = nvml_process_usage(device)[0]
    if used_before is not None and used_after is not None:
        # ... and at least as much as this process now holds more.
        assert used_after - used_before >= 4 * MB
    del ipc
    assert IpcMemory.live_allocations(device).get("unit", (0, 0)) == before
