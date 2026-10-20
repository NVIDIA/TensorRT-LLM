# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TRTLLM_Q35_FAST_COUNT_BCAST: the request-count broadcast of a non-PP MPI executor is one raw int64 Bcast
on the communicator MPIDist.broadcast uses; other executors and the flag-off path keep dist.broadcast."""

import contextlib
import types

import pytest

from tensorrt_llm._torch.distributed.communicator import MPIDist
from tensorrt_llm._torch.pyexecutor import request_utils
from tensorrt_llm._torch.pyexecutor.request_utils import (
    FAST_COUNT_BROADCAST_ENV,
    RequestBroadcaster,
)

pytestmark = pytest.mark.cpu_only


class _FakeMPIDist(MPIDist):
    """MPIDist identity without a mapping: just what the count broadcast reads."""

    def __init__(self, rank, root_value):
        self._rank, self.root_value, self.pickled = rank, root_value, 0

    world_size = property(lambda self: 8)
    has_pp = property(lambda self: False)
    rank = property(lambda self: self._rank)

    def broadcast(self, obj, root=0, chunk_size=0):
        self.pickled += 1
        return self.root_value


class _FakeComm:
    def __init__(self, root_value):
        self.root_value, self.calls = root_value, []

    def Bcast(self, buf, root=0):
        array = buf[0]
        self.calls.append((array.dtype.name, array.shape, root))
        array[0] = self.root_value  # what every rank holds after the root's Bcast


def _broadcaster(dist):
    hang = types.SimpleNamespace(pause=contextlib.nullcontext)
    return RequestBroadcaster(dist, hang)


@pytest.mark.parametrize("rank", [0, 3])
def test_fast_path_one_raw_bcast(monkeypatch, rank):
    monkeypatch.setenv(FAST_COUNT_BROADCAST_ENV, "1")
    comm = _FakeComm(root_value=2)
    import tensorrt_llm._utils as utils

    monkeypatch.setattr(utils, "mpi_comm", lambda: comm)
    dist = _FakeMPIDist(rank, root_value=2)
    b = _broadcaster(dist)
    assert b._broadcast_request_count(2 if rank == 0 else 0) == 2
    assert comm.calls == [("int64", (1,), 0)]
    assert dist.pickled == 0


def test_flag_off_and_non_mpi_keep_dist_broadcast(monkeypatch):
    monkeypatch.delenv(FAST_COUNT_BROADCAST_ENV, raising=False)
    dist = _FakeMPIDist(1, root_value=5)
    assert _broadcaster(dist)._broadcast_request_count(0) == 5
    assert dist.pickled == 1
    monkeypatch.setenv(FAST_COUNT_BROADCAST_ENV, "1")
    other = types.SimpleNamespace(world_size=8, has_pp=False, rank=1, calls=0)

    def broadcast(obj, root=0):
        other.calls += 1
        return 7

    other.broadcast = broadcast
    assert _broadcaster(other)._broadcast_request_count(0) == 7
    assert other.calls == 1
    assert request_utils._fast_count_broadcast_enabled()
