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

import pytest
import torch

from tensorrt_llm._torch.distributed.ops import MNNVLAllReduce


class _FakeComm:
    def __init__(self) -> None:
        self.barrier_count = 0

    def py2f(self) -> int:
        return 42

    def Barrier(self) -> None:
        self.barrier_count += 1


class _FakeMcastBuffer:
    def __init__(self) -> None:
        self.mapped = True
        self.prepare_count = 0
        self.restore_count = 0

    def is_mapped(self) -> bool:
        return self.mapped

    def checkpoint_prepare(self) -> None:
        if self.mapped:
            self.prepare_count += 1
            self.mapped = False

    def checkpoint_restore(self, mpi_comm_fortran_handle: int) -> None:
        assert mpi_comm_fortran_handle == 42
        if not self.mapped:
            self.restore_count += 1
            self.mapped = True


def test_checkpoint_restore_resets_inference_protocol_state(monkeypatch) -> None:
    mapping = object()
    stale_comm = _FakeComm()
    comm = _FakeComm()
    handle = _FakeMcastBuffer()
    with torch.inference_mode():
        uc_buffer = torch.ones(8, dtype=torch.float32)
        buffer_flags = torch.ones(9, dtype=torch.uint32)
    workspace = {
        "handle": handle,
        "uc_buffer": uc_buffer,
        "buffer_flags": buffer_flags,
        "buffer_size_bytes": 1024,
        "mpi_comm": stale_comm,
    }
    allreduce = object.__new__(MNNVLAllReduce)
    torch.nn.Module.__init__(allreduce)
    allreduce.mapping = mapping
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(MNNVLAllReduce, "allreduce_mnnvl_workspaces", {mapping: workspace})

    assert not torch.is_inference_mode_enabled()
    allreduce.checkpoint_prepare()
    allreduce.checkpoint_prepare()
    with pytest.raises(TypeError):
        allreduce.checkpoint_restore()
    allreduce.checkpoint_restore(comm)
    allreduce.checkpoint_restore(comm)

    assert workspace["mpi_comm"] is comm
    assert stale_comm.barrier_count == 0
    assert handle.prepare_count == 1
    assert handle.restore_count == 1
    assert comm.barrier_count == 1
    assert not torch.is_inference_mode_enabled()
    assert torch.all(uc_buffer == 0)
    assert torch.all(torch.signbit(uc_buffer))
    assert torch.equal(
        buffer_flags,
        torch.tensor([0, 2, 1024, 0, 0, 0, 0, 0, 0], dtype=torch.uint32),
    )
