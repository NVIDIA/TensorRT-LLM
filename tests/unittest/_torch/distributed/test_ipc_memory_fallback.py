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
"""
Tests for the CUDA IPC fallback in IpcMemory.

`cudaDeviceCanAccessPeer` reporting P2P support does not guarantee that CUDA IPC
handles can be exported/imported.  When IPC turns out to be unusable, IpcMemory
must degrade to null pointers (so the runtime falls back to NCCL) instead of
raising, and every rank of the TP group must reach the same decision.

The CUDA runtime is stubbed, so these tests need neither a GPU nor MPI:
    pytest tests/unittest/_torch/distributed/test_ipc_memory_fallback.py -v
"""

from enum import IntEnum
from unittest.mock import patch

import pytest

from tensorrt_llm import _ipc_utils
from tensorrt_llm._ipc_utils import IpcMemory
from tensorrt_llm.mapping import Mapping

TP_SIZE = 2


class FakeCudaError(IntEnum):
    cudaSuccess = 0
    cudaErrorInvalidDevice = 101


class FakeIpcMemHandle:
    def __init__(self):
        self.reserved = b"\x00" * 64


class FakeCudart:
    """Minimal stand-in for `cuda.bindings.runtime` that tracks allocations."""

    cudaError_t = FakeCudaError
    cudaIpcMemHandle_t = FakeIpcMemHandle
    cudaIpcMemLazyEnablePeerAccess = 1

    def __init__(self, open_handle_error=FakeCudaError.cudaSuccess):
        self.open_handle_error = open_handle_error
        self.next_ptr = 0x1000
        self.allocated = set()
        self.opened = set()

    def cudaMalloc(self, size):
        self.next_ptr += size
        self.allocated.add(self.next_ptr)
        return FakeCudaError.cudaSuccess, self.next_ptr

    def cudaMemset(self, ptr, value, size):
        return (FakeCudaError.cudaSuccess,)

    def cudaFree(self, ptr):
        self.allocated.discard(ptr)
        return (FakeCudaError.cudaSuccess,)

    def cudaIpcGetMemHandle(self, ptr):
        return FakeCudaError.cudaSuccess, FakeIpcMemHandle()

    def cudaIpcOpenMemHandle(self, handle, flags):
        if self.open_handle_error != FakeCudaError.cudaSuccess:
            return self.open_handle_error, 0
        self.next_ptr += 0x1000
        self.opened.add(self.next_ptr)
        return FakeCudaError.cudaSuccess, self.next_ptr

    def cudaIpcCloseMemHandle(self, ptr):
        self.opened.discard(ptr)
        return (FakeCudaError.cudaSuccess,)


class FakeDist:
    """`tp_allgather` over a fake TP group where the peers mirror this rank.

    `peer_ipc_ok` overrides what the peers report for the boolean agreement
    collective, which lets a test exercise "this rank succeeded but another one
    did not".
    """

    def __init__(self, tp_size, peer_ipc_ok=True):
        self.tp_size = tp_size
        self.peer_ipc_ok = peer_ipc_ok

    def tp_allgather(self, obj):
        if isinstance(obj, bool):
            return [obj] + [self.peer_ipc_ok] * (self.tp_size - 1)
        return [obj] * self.tp_size


@pytest.fixture
def mapping():
    return Mapping(world_size=TP_SIZE, rank=0, tp_size=TP_SIZE)


def _run(mapping, cudart, dist):
    from tensorrt_llm._torch.distributed.communicator import Distributed

    with (
        patch.object(_ipc_utils, "cudart", cudart),
        patch.object(Distributed, "get", lambda _mapping: dist),
    ):
        return IpcMemory(mapping, 1 << 20)


def test_ipc_memory_is_opened_when_ipc_works(mapping):
    cudart = FakeCudart()

    ipc_memory = _run(mapping, cudart, FakeDist(TP_SIZE))

    assert ipc_memory.open_ipc
    assert ipc_memory.local_ptr != 0
    assert all(ptr != 0 for ptr in ipc_memory.peer_ptrs)
    assert len(cudart.opened) == TP_SIZE - 1


def test_local_ipc_open_failure_falls_back_instead_of_raising(mapping):
    # Reproduces github.com/NVIDIA/TensorRT-LLM/issues/16899: cudaIpcOpenMemHandle
    # fails with cudaErrorInvalidDevice even though cudaDeviceCanAccessPeer passed.
    cudart = FakeCudart(open_handle_error=FakeCudaError.cudaErrorInvalidDevice)

    ipc_memory = _run(mapping, cudart, FakeDist(TP_SIZE, peer_ipc_ok=False))

    assert not ipc_memory.open_ipc
    assert ipc_memory.local_ptr == 0
    assert ipc_memory.peer_ptrs == [0] * TP_SIZE
    assert not cudart.allocated, "the local buffer must be released on fallback"


def test_peer_ipc_failure_disables_ipc_on_this_rank(mapping):
    # IPC works locally but another rank of the TP group failed: this rank must
    # fall back as well, otherwise it would use buffers the peers never mapped.
    cudart = FakeCudart()

    ipc_memory = _run(mapping, cudart, FakeDist(TP_SIZE, peer_ipc_ok=False))

    assert not ipc_memory.open_ipc
    assert ipc_memory.local_ptr == 0
    assert ipc_memory.peer_ptrs == [0] * TP_SIZE
    assert not cudart.allocated
    assert not cudart.opened, "handles opened before the fallback must be closed"
