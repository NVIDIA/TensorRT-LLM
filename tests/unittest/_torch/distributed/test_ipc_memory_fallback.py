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
Tests for the CUDA IPC fallback in IpcMemory and its effect on strategy selection.

`cudaDeviceCanAccessPeer` reporting P2P does not guarantee that IPC handles can be
exported/imported.  When IPC turns out to be unusable, IpcMemory keeps null pointers
so the runtime falls back to NCCL instead of raising, and the whole TP group agrees
on that.  Only capability errors fall back: any other IPC error still raises, on
every rank, so a real bug is not hidden behind NCCL.

A workspace with no IPC buffers because P2P was never there (inter-node TP) is not a
failure though, and must not change the strategy, or MNNVL never gets selected.

The CUDA runtime and the collectives are stubbed, so no multi-GPU and no MPI needed.
This directory runs in the CPU-only l0_cpu stage since #16498, so everything here has
to pass with zero GPUs.  `test_lamport_skipped` is the exception and skips there: the
workspace helper it drives allocates device tensors.

    pytest tests/unittest/_torch/distributed/test_ipc_memory_fallback.py -v
"""

import contextlib
from enum import IntEnum
from unittest.mock import patch

import pytest
import torch

from tensorrt_llm import _ipc_utils
from tensorrt_llm._ipc_utils import IpcMemory
from tensorrt_llm.functional import AllReduceStrategy
from tensorrt_llm.mapping import Mapping


def _has_allreduce_op() -> bool:
    """Whether the trtllm custom ops are registered in this build."""
    try:
        return torch.ops.trtllm.allreduce is not None
    except (AttributeError, RuntimeError):
        return False


# AllReduce.__init__ resolves torch.ops.trtllm.allreduce before any of the logic
# under test, so these skip rather than error on a build without the extension.
requires_allreduce_op = pytest.mark.skipif(
    not _has_allreduce_op(), reason="trtllm allreduce custom op not registered"
)

TP_SIZE = 2


class FakeCudaError(IntEnum):
    cudaSuccess = 0
    cudaErrorInvalidValue = 1
    cudaErrorMemoryAllocation = 2
    cudaErrorInvalidDevice = 101
    cudaErrorPeerAccessUnsupported = 217
    cudaErrorInvalidResourceHandle = 400
    cudaErrorNotSupported = 801


class FakeIpcMemHandle:
    def __init__(self):
        self.reserved = b"\x00" * 64


class FakeCudart:
    """Minimal stand-in for `cuda.bindings.runtime` that tracks allocations."""

    cudaError_t = FakeCudaError
    cudaIpcMemHandle_t = FakeIpcMemHandle
    cudaIpcMemLazyEnablePeerAccess = 1

    def __init__(
        self,
        get_handle_error=FakeCudaError.cudaSuccess,
        open_handle_error=FakeCudaError.cudaSuccess,
        malloc_error=FakeCudaError.cudaSuccess,
        memset_error=FakeCudaError.cudaSuccess,
    ):
        self.get_handle_error = get_handle_error
        self.open_handle_error = open_handle_error
        self.malloc_error = malloc_error
        self.memset_error = memset_error
        self.next_ptr = 0x1000
        self.allocated = set()
        self.opened = set()
        self.open_calls = []

    def cudaMalloc(self, size):
        if self.malloc_error != FakeCudaError.cudaSuccess:
            return self.malloc_error, 0
        self.next_ptr += size
        self.allocated.add(self.next_ptr)
        return FakeCudaError.cudaSuccess, self.next_ptr

    def cudaMemset(self, ptr, value, size):
        return (self.memset_error,)

    def cudaFree(self, ptr):
        self.allocated.discard(ptr)
        return (FakeCudaError.cudaSuccess,)

    def cudaIpcGetMemHandle(self, ptr):
        handle = FakeIpcMemHandle()
        handle.reserved = b"\xab" * 64
        return self.get_handle_error, handle

    def cudaIpcOpenMemHandle(self, handle, flags):
        self.open_calls.append(bytes(handle.reserved))
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

    `peer_export_error` and `peer_open_error` set the CUDA error the peers report
    for the handle export and import steps, which lets a test exercise "this rank
    succeeded but another one did not".  None means the peers succeeded.
    """

    def __init__(self, tp_size, peer_export_error=None, peer_open_error=None):
        self.tp_size = tp_size
        self.peer_export_error = peer_export_error
        self.peer_open_error = peer_open_error
        self.calls = 0

    def tp_allgather(self, obj):
        self.calls += 1
        if self.calls == 1:
            # Handle export: the peers send their handle bytes, or an error code.
            if self.peer_export_error is None:
                peer = b"\xcd" * 64
            else:
                peer = int(self.peer_export_error)
        else:
            # Handle import: the peers send None, or an error code.
            peer = None if self.peer_open_error is None else int(self.peer_open_error)
        return [obj] + [peer] * (self.tp_size - 1)


@pytest.fixture
def mapping():
    return Mapping(world_size=TP_SIZE, rank=0, tp_size=TP_SIZE)


def _patched(cudart, dist):
    from tensorrt_llm._torch.distributed.communicator import Distributed

    return (
        patch.object(_ipc_utils, "cudart", cudart),
        patch.object(Distributed, "get", lambda _mapping: dist),
    )


def _run(mapping, cudart, dist, open_ipc=True):
    """Builds an IpcMemory that is expected to end up without IPC buffers."""
    cudart_patch, dist_patch = _patched(cudart, dist)
    with cudart_patch, dist_patch:
        ipc_memory = IpcMemory(mapping, 1 << 20, open_ipc)
    # Nothing for __del__ to release, so it never reaches the real CUDA runtime
    # with the fake pointers once the patches are gone.
    assert not ipc_memory.open_ipc
    return ipc_memory


def test_ipc_ok(mapping):
    cudart = FakeCudart()
    cudart_patch, dist_patch = _patched(cudart, FakeDist(TP_SIZE))

    with cudart_patch, dist_patch:
        ipc_memory = IpcMemory(mapping, 1 << 20, True)

        assert ipc_memory.open_ipc
        assert not ipc_memory.ipc_failed
        assert ipc_memory.local_ptr != 0
        assert all(ptr != 0 for ptr in ipc_memory.peer_ptrs)
        assert len(cudart.opened) == TP_SIZE - 1

        # Release the fake buffers while the fake runtime is still in place.
        IpcMemory.close_ipc_memory(mapping, ipc_memory.peer_ptrs)
        ipc_memory.open_ipc = False

    assert not cudart.allocated
    assert not cudart.opened


def test_local_open_fails(mapping):
    # Reproduces github.com/NVIDIA/TensorRT-LLM/issues/16899: cudaIpcOpenMemHandle
    # fails with cudaErrorInvalidDevice even though cudaDeviceCanAccessPeer passed.
    cudart = FakeCudart(open_handle_error=FakeCudaError.cudaErrorInvalidDevice)

    ipc_memory = _run(
        mapping,
        cudart,
        FakeDist(TP_SIZE, peer_open_error=FakeCudaError.cudaErrorInvalidDevice),
    )

    assert ipc_memory.ipc_failed
    assert ipc_memory.local_ptr == 0
    assert ipc_memory.peer_ptrs == [0] * TP_SIZE
    assert not cudart.allocated, "the local buffer must be released on fallback"


def test_peer_open_fails(mapping):
    # IPC works locally but another rank of the TP group failed: this rank must
    # fall back as well, otherwise it would use buffers the peers never mapped.
    cudart = FakeCudart()

    ipc_memory = _run(
        mapping,
        cudart,
        FakeDist(TP_SIZE, peer_open_error=FakeCudaError.cudaErrorNotSupported),
    )

    assert ipc_memory.ipc_failed
    assert ipc_memory.local_ptr == 0
    assert ipc_memory.peer_ptrs == [0] * TP_SIZE
    assert not cudart.allocated
    assert not cudart.opened, "handles opened before the fallback must be closed"


def test_local_export_fails(mapping):
    cudart = FakeCudart(get_handle_error=FakeCudaError.cudaErrorInvalidDevice)

    ipc_memory = _run(mapping, cudart, FakeDist(TP_SIZE))

    assert ipc_memory.ipc_failed
    assert cudart.open_calls == []
    assert not cudart.allocated


def test_peer_export_fails(mapping):
    # A rank that cannot export contributes its error code rather than an
    # uninitialized cudaIpcMemHandle_t, so the peers never import a garbage handle.
    cudart = FakeCudart()

    ipc_memory = _run(
        mapping,
        cudart,
        FakeDist(TP_SIZE, peer_export_error=FakeCudaError.cudaErrorPeerAccessUnsupported),
    )

    assert ipc_memory.ipc_failed
    assert cudart.open_calls == []
    assert not cudart.allocated


def _expect_raise(mapping, cudart, dist, match):
    cudart_patch, dist_patch = _patched(cudart, dist)
    with cudart_patch, dist_patch:
        with pytest.raises(RuntimeError, match=match):
            IpcMemory(mapping, 1 << 20, True)
    assert not cudart.allocated, "the local buffer must be released before raising"
    assert not cudart.opened, "opened handles must be closed before raising"


def test_local_open_bug_raises(mapping):
    # Not a capability gap: a broken handle round-trip must not be papered over
    # by a silent NCCL fallback.
    cudart = FakeCudart(open_handle_error=FakeCudaError.cudaErrorInvalidResourceHandle)

    _expect_raise(
        mapping,
        cudart,
        FakeDist(TP_SIZE, peer_open_error=FakeCudaError.cudaErrorInvalidResourceHandle),
        "cudaIpcOpenMemHandle failed.*cudaErrorInvalidResourceHandle",
    )


def test_peer_open_bug_raises(mapping):
    # This rank opened every handle, but a peer hit a non-capability error: every
    # rank must raise, not only the one that saw it.
    cudart = FakeCudart()

    _expect_raise(
        mapping,
        cudart,
        FakeDist(TP_SIZE, peer_open_error=FakeCudaError.cudaErrorInvalidResourceHandle),
        "cudaIpcOpenMemHandle failed.*rank 1: .*cudaErrorInvalidResourceHandle",
    )


def test_peer_export_bug_raises(mapping):
    cudart = FakeCudart()

    _expect_raise(
        mapping,
        cudart,
        FakeDist(TP_SIZE, peer_export_error=FakeCudaError.cudaErrorInvalidValue),
        "cudaIpcGetMemHandle failed.*cudaErrorInvalidValue",
    )
    assert cudart.open_calls == []


def test_mixed_errors_raise(mapping):
    # One rank lacks the capability, another hit a real error: the real error wins.
    cudart = FakeCudart(open_handle_error=FakeCudaError.cudaErrorInvalidDevice)

    _expect_raise(
        mapping,
        cudart,
        FakeDist(TP_SIZE, peer_open_error=FakeCudaError.cudaErrorInvalidResourceHandle),
        "cudaIpcOpenMemHandle failed",
    )


@pytest.mark.parametrize(
    "cudart_kwargs",
    [
        {"malloc_error": FakeCudaError.cudaErrorMemoryAllocation},
        {"memset_error": FakeCudaError.cudaErrorInvalidValue},
    ],
    ids=["malloc", "memset"],
)
def test_alloc_error_raises(mapping, cudart_kwargs):
    # Allocation failures are genuine errors, not a missing IPC capability, so they
    # keep raising instead of falling back to NCCL, before any collective runs.
    cudart = FakeCudart(**cudart_kwargs)
    dist = FakeDist(TP_SIZE)
    cudart_patch, dist_patch = _patched(cudart, dist)

    with cudart_patch, dist_patch:
        with pytest.raises(RuntimeError, match="CUDA Runtime API error"):
            IpcMemory(mapping, 1 << 20, True)

    assert dist.calls == 0
    assert cudart.open_calls == []


def test_ipc_not_requested(mapping):
    # `can_access_peer` returning False (inter-node TP, no P2P) is not a failure:
    # the workspace is null by design and strategy selection must not react to it.
    cudart = FakeCudart()

    ipc_memory = _run(mapping, cudart, FakeDist(TP_SIZE), open_ipc=False)

    assert not ipc_memory.ipc_failed
    assert not cudart.allocated


def test_inter_node_tp():
    inter_node = Mapping(world_size=16, rank=0, tp_size=16, gpus_per_node=8)
    cudart = FakeCudart()

    ipc_memory = _run(inter_node, cudart, FakeDist(16))

    assert not ipc_memory.ipc_failed


class FakeMNNVLAllReduce:
    @staticmethod
    def is_mnnvl(mapping, dtype, explicitly_requested=False):
        return True

    def __init__(self, mapping, dtype):
        self.mapping = mapping


@contextlib.contextmanager
def _patched_workspace(ipc_failed):
    from tensorrt_llm._torch.distributed import ops

    workspace = torch.zeros(1, dtype=torch.int64)
    with (
        patch.object(ops, "get_allreduce_workspace", lambda _mapping: workspace),
        patch.object(ops, "allreduce_workspace_ipc_failed", lambda _mapping: ipc_failed),
    ):
        yield workspace


def _build_allreduce(mapping, strategy, ipc_failed):
    """Builds an AllReduce with the workspace and MNNVL support both stubbed out."""
    from tensorrt_llm._torch.distributed import ops

    with _patched_workspace(ipc_failed), patch.object(ops, "MNNVLAllReduce", FakeMNNVLAllReduce):
        return ops.AllReduce(mapping=mapping, strategy=strategy, dtype=torch.bfloat16)


@requires_allreduce_op
def test_mnnvl_kept_without_p2p(mapping):
    # Regression guard: an inter-node workspace holds no IPC buffers, but that must
    # not rewrite the strategy, otherwise MNNVL is never constructed on NVLink
    # multi-node systems.
    allreduce = _build_allreduce(mapping, AllReduceStrategy.AUTO, ipc_failed=False)

    assert allreduce.strategy == AllReduceStrategy.AUTO
    assert allreduce.mnnvl_allreduce is not None


@requires_allreduce_op
def test_downgrade_to_nccl(mapping):
    allreduce = _build_allreduce(mapping, AllReduceStrategy.ONESHOT, ipc_failed=True)

    assert allreduce.strategy == AllReduceStrategy.NCCL
    assert allreduce.workspace is None
    assert allreduce.mnnvl_allreduce is None


@requires_allreduce_op
def test_moe_allreduce_raises(mapping):
    # MoEAllReduce has no NCCL path: it must fail with an actionable message rather
    # than dereference the null peer pointers inside the kernel.
    from tensorrt_llm._torch.distributed import ops

    with _patched_workspace(ipc_failed=True):
        with pytest.raises(RuntimeError, match="requires CUDA IPC"):
            ops.MoEAllReduce(mapping)


@requires_allreduce_op
def test_minimax_allreduce_rms_raises(mapping):
    from tensorrt_llm._torch.distributed import ops

    with _patched_workspace(ipc_failed=True):
        with pytest.raises(RuntimeError, match="requires CUDA IPC"):
            ops.MiniMaxAllReduceRMS(mapping)


@requires_allreduce_op
@pytest.mark.parametrize("ipc_failed", [False, True])
def test_group_args_fall_back_to_nccl(mapping, ipc_failed):
    # The raw-op entry builds its own workspace argument: it must not hand null
    # peer pointers to a custom kernel either.
    from tensorrt_llm import _utils
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm import allreduce as entry

    with (
        _patched_workspace(ipc_failed) as workspace,
        patch.object(_utils, "mpi_disabled", lambda: False),
    ):
        args = entry.allreduce_group_args(mapping)

    if ipc_failed:
        assert args["strategy"] == int(AllReduceStrategy.NCCL)
        assert args["workspace"] is None
    else:
        assert args["strategy"] == int(AllReduceStrategy.AUTO)
        assert args["workspace"] is workspace


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_lamport_skipped(mapping):
    # lamport_initialize writes through lamport_buffers.local_ptr, which is null
    # once IPC has been disabled.
    from tensorrt_llm._torch.distributed import allreduce_helper
    from tensorrt_llm._torch.distributed.allreduce_helper import CustomAllReduceHelper

    cudart = FakeCudart(open_handle_error=FakeCudaError.cudaErrorInvalidDevice)
    cudart_patch, dist_patch = _patched(
        cudart, FakeDist(TP_SIZE, peer_open_error=FakeCudaError.cudaErrorInvalidDevice)
    )
    calls = []

    with (
        cudart_patch,
        dist_patch,
        patch.object(allreduce_helper, "can_access_peer", lambda _mapping: True),
        patch.object(allreduce_helper, "lamport_initialize", lambda *args: calls.append(args)),
    ):
        buffers, _ = CustomAllReduceHelper.allocate_allreduce_fusion_workspace(mapping, 1 << 20)

    assert calls == [], "lamport_initialize must not run on a null buffer"
    assert all(buffer.ipc_failed for buffer in buffers if isinstance(buffer, IpcMemory))
