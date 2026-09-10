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

from tensorrt_llm._torch.distributed import ops as ops_module
from tensorrt_llm._torch.distributed.ops import MNNVLAllReduce


class _FakeComm:
    def __init__(self) -> None:
        self.allreduce_count = 0
        self.dup_count = 0
        self.free_count = 0
        self.duplicates = []

    def py2f(self) -> int:
        return 42

    def allreduce(self, value: int) -> int:
        self.allreduce_count += 1
        return value

    def Get_size(self) -> int:
        return 1

    def Dup(self):
        self.dup_count += 1
        duplicate = _FakeComm()
        self.duplicates.append(duplicate)
        return duplicate

    def Free(self) -> None:
        self.free_count += 1


class _FailingDupComm(_FakeComm):
    def Dup(self):
        raise RuntimeError("injected communicator duplication failure")


class _FakeMcastBuffer:
    def __init__(self) -> None:
        self.mapped = True
        self.restore_pending = False
        self.prepare_count = 0
        self.restore_count = 0
        self.complete_count = 0

    def is_mapped(self) -> bool:
        return self.mapped

    def checkpoint_prepare(self) -> None:
        if self.mapped:
            self.prepare_count += 1
            self.mapped = False

    def checkpoint_restore(self, mpi_comm_fortran_handle: int) -> bool:
        assert mpi_comm_fortran_handle == 42
        if not self.mapped:
            self.restore_count += 1
            self.restore_pending = True
            return True
        return False

    def checkpoint_restore_complete(self, local_protocol_reset_succeeded: bool) -> None:
        assert self.restore_pending
        self.complete_count += 1
        self.restore_pending = False
        if not local_protocol_reset_succeeded:
            raise RuntimeError("protocol reset failed on one or more ranks")
        self.mapped = True


def test_checkpoint_restore_resets_inference_protocol_state(monkeypatch) -> None:
    mapping = object()
    stale_comm = _FakeComm()
    comm = _FakeComm()
    redundant_comm = _FakeComm()
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
    allreduce.checkpoint_restore(redundant_comm)

    assert workspace["mpi_comm"] is comm.duplicates[0]
    assert stale_comm.allreduce_count == 0
    assert stale_comm.free_count == 1
    assert handle.prepare_count == 1
    assert handle.restore_count == 1
    assert handle.complete_count == 1
    assert comm.allreduce_count == 0
    assert comm.dup_count == 1
    assert redundant_comm.allreduce_count == 0
    assert redundant_comm.dup_count == 0
    assert redundant_comm.duplicates == []
    assert not torch.is_inference_mode_enabled()
    assert torch.all(uc_buffer == 0)
    assert torch.all(torch.signbit(uc_buffer))
    assert torch.equal(
        buffer_flags,
        torch.tensor([0, 2, 1024, 0, 0, 0, 0, 0, 0], dtype=torch.uint32),
    )


def test_checkpoint_restore_protocol_failure_is_terminal(monkeypatch) -> None:
    mapping = object()
    comm = _FakeComm()
    handle = _FakeMcastBuffer()
    workspace = {
        "handle": handle,
        "uc_buffer": torch.ones(8, dtype=torch.float32),
        "buffer_flags": torch.ones(9, dtype=torch.uint32),
        "buffer_size_bytes": 1024,
        "mpi_comm": _FakeComm(),
    }
    allreduce = object.__new__(MNNVLAllReduce)
    torch.nn.Module.__init__(allreduce)
    allreduce.mapping = mapping
    monkeypatch.setattr(MNNVLAllReduce, "allreduce_mnnvl_workspaces", {mapping: workspace})
    monkeypatch.setattr(
        ops_module,
        "_initialize_allreduce_mnnvl_protocol",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("injected protocol reset failure")
        ),
    )

    allreduce.checkpoint_prepare()
    with pytest.raises(RuntimeError, match="injected protocol reset failure"):
        allreduce.checkpoint_restore(comm)

    assert not handle.is_mapped()
    assert not handle.restore_pending
    assert handle.complete_count == 1
    assert workspace["mpi_comm"] is None
    assert comm.duplicates[0].free_count == 1


def test_checkpoint_restore_communicator_duplication_failure_is_terminal(monkeypatch) -> None:
    mapping = object()
    handle = _FakeMcastBuffer()
    workspace = {
        "handle": handle,
        "uc_buffer": torch.ones(8, dtype=torch.float32),
        "buffer_flags": torch.ones(9, dtype=torch.uint32),
        "buffer_size_bytes": 1024,
        "mpi_comm": _FakeComm(),
    }
    allreduce = object.__new__(MNNVLAllReduce)
    torch.nn.Module.__init__(allreduce)
    allreduce.mapping = mapping
    monkeypatch.setattr(MNNVLAllReduce, "allreduce_mnnvl_workspaces", {mapping: workspace})

    allreduce.checkpoint_prepare()
    with pytest.raises(RuntimeError, match="injected communicator duplication failure"):
        allreduce.checkpoint_restore(_FailingDupComm())

    assert not handle.is_mapped()
    assert not handle.restore_pending
    assert handle.complete_count == 1
    assert workspace["mpi_comm"] is None
