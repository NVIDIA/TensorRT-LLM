# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TRTLLM_Q35_DISAGG_MERGED_VOTES: the two idle-progress votes of a TP-only disagg executor in one raw
Allreduce(MAX) take exactly the branches the two separate collectives take, for every combination of
local and peer votes; other layouts and the flag-off path keep the separate collectives."""

import itertools

import numpy as np
import pytest

import tensorrt_llm._utils as trt_utils
from tensorrt_llm._torch.distributed.communicator import MPIDist
from tensorrt_llm._torch.pyexecutor import py_executor
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

pytestmark = pytest.mark.cpu_only


class _Dist(MPIDist):
    """A TP-only 8-rank MPIDist whose collectives fold in fixed peer votes."""

    def __init__(self, peer_gen, peer_ctx, tp_size=8, cp_size=1):
        self.peer_gen, self.peer_ctx, self.calls = peer_gen, peer_ctx, []
        self._tp, self._cp = tp_size, cp_size

    world_size = property(lambda self: 8)
    tp_size = property(lambda self: self._tp)
    cp_size = property(lambda self: self._cp)

    def allreduce(self, obj, op=None):
        self.calls.append("allreduce")
        return max(obj, self.peer_gen)

    def tp_allreduce(self, obj, op=None):
        self.calls.append("tp_allreduce")
        return max(obj, self.peer_ctx)


class _Comm:
    def __init__(self, dist):
        self.dist = dist

    def Allreduce(self, send, recv, op=None):
        self.dist.calls.append("Allreduce")
        recv[:] = np.maximum(send, [self.dist.peer_gen, self.dist.peer_ctx])


def _executor(dist):
    ex = object.__new__(PyExecutor)
    ex.dist = dist
    ex.events = []
    ex._uses_async_disagg_gen_transfer = lambda: True
    ex._check_disagg_gen_cache_transfer_status = lambda n: ex.events.append(("gen", n))
    ex._check_disagg_ctx_cache_transfer_status = lambda n: ex.events.append(("ctx", n))
    return ex


def _run(monkeypatch, merged, local_fitting, local_wait, peer_gen, peer_ctx, all_gen_first):
    dist = _Dist(peer_gen, peer_ctx)
    if merged:
        monkeypatch.setenv(py_executor.MERGED_DISAGG_VOTES_ENV_VAR_NAME, "1")
        monkeypatch.setattr(trt_utils, "mpi_comm", lambda: _Comm(dist))
    else:
        monkeypatch.delenv(py_executor.MERGED_DISAGG_VOTES_ENV_VAR_NAME, raising=False)
    ex = _executor(dist)
    ex._check_disagg_transfer_progress_when_idle(local_fitting, [], local_wait, all_gen_first)
    return ex.events, dist.calls


@pytest.mark.parametrize(
    "local_fitting, local_wait, peer_gen, peer_ctx, all_gen_first",
    list(itertools.product([0, 1], [False, True], [0, 1], [0, 1], [False, True])),
)
def test_merged_votes_take_the_same_branches(
    monkeypatch, local_fitting, local_wait, peer_gen, peer_ctx, all_gen_first
):
    sep_events, sep_calls = _run(
        monkeypatch, False, local_fitting, local_wait, peer_gen, peer_ctx, all_gen_first
    )
    mrg_events, mrg_calls = _run(
        monkeypatch, True, local_fitting, local_wait, peer_gen, peer_ctx, all_gen_first
    )
    assert mrg_events == sep_events
    assert mrg_calls == ["Allreduce"]
    assert sep_calls[0] == "allreduce" and set(sep_calls) <= {"allreduce", "tp_allreduce"}


def test_not_applicable_layouts_keep_separate_collectives(monkeypatch):
    monkeypatch.setenv(py_executor.MERGED_DISAGG_VOTES_ENV_VAR_NAME, "1")
    for dist in (_Dist(0, 0, tp_size=4), _Dist(0, 0, cp_size=2)):
        ex = _executor(dist)
        assert ex._q35_merged_disagg_votes(False, False) is None
