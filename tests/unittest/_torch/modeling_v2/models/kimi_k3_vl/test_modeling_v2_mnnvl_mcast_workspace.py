# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Kimi K3 family's ``mnnvl_workspace`` without a collective: the TP-group communicator it makes, each other
function the built-in MNNVL all-reduce's own step (so a workspace built from them is set up the way the built-in one
is), and the view of the shared workspace it hands out. The collective behavior on four ranks is
``test_modeling_v2_mnnvl_mcast_workspace_op_matrix.py``."""

from types import SimpleNamespace

import pytest
import torch

import tensorrt_llm._torch.distributed.ops as ops
import tensorrt_llm._utils as utils
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl import mnnvl_workspace as mcast

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize(
    "function, builtin, args",
    [
        (mcast.group_comm, "_get_mnnvl_workspace_comm", ("mapping",)),
        (mcast.device_index, "_mnnvl_device_index", ("mapping",)),
        (mcast.use_fabric_handle, "use_mnnvl_fabric_handle", ("mapping",)),
        (mcast.all_ranks_succeeded, "_mnnvl_workspace_all_succeeded", ("comm", True)),
        (mcast.make_mcast_buffer, "_make_mnnvl_mcast_buffer", ("comm", 4096, "mapping", True)),
    ],
)
def test_each_step_is_the_builtin_one(monkeypatch, function, builtin, args):
    calls = []

    def fake(*a, **k):
        calls.append((a, k))
        return "result"

    monkeypatch.setattr(ops, builtin, fake)
    assert function(*args) == "result"
    assert calls == [(args, {})]


def test_arm_lamport_buffers_is_the_builtin_reset(monkeypatch):
    calls = []
    monkeypatch.setattr(ops, "_initialize_allreduce_mnnvl_protocol", lambda ws: calls.append(ws))
    uc, flags = torch.zeros(12), torch.zeros(9, dtype=torch.int32)
    mcast.arm_lamport_buffers(uc, flags, 16, "comm")
    assert len(calls) == 1
    ws = calls[0]
    assert ws["uc_buffer"] is uc and ws["buffer_flags"] is flags
    assert ws["buffer_size_bytes"] == 16 and ws["comm"] == "comm"


def test_shared_workspace_is_the_builtin_one(monkeypatch):
    uc = torch.zeros(3 * 64 // 4)
    flags = torch.zeros(9, dtype=torch.int32)
    seen = []

    def fake(mapping, dtype, buffer_size_bytes=None):
        seen.append((mapping, dtype, buffer_size_bytes))
        return {"uc_buffer": uc, "buffer_flags": flags, "buffer_size_bytes": 64}

    monkeypatch.setattr(ops, "get_or_scale_allreduce_mnnvl_workspace", fake)
    mapping = SimpleNamespace(tp_size=4)
    ws = mcast.shared_lamport_workspace(mapping, torch.bfloat16, 48)
    assert seen == [(mapping, torch.bfloat16, 48)]
    assert ws.uc_buffer is uc and ws.buffer_flags is flags and ws.buffer_bytes == 64
    assert ws.world_size == 4
    view = ws.comm_buffer(torch.bfloat16)
    assert view.shape == (3, 32) and view.data_ptr() == uc.data_ptr()


def test_shared_workspace_presence(monkeypatch):
    monkeypatch.setattr(ops.MNNVLAllReduce, "allreduce_mnnvl_workspaces", {"a": {"uc_buffer": 0}})
    assert mcast.has_shared_lamport_workspace("a")
    assert not mcast.has_shared_lamport_workspace("b")


@pytest.mark.parametrize("requested", [False, True])
def test_mnnvl_supported_is_the_builtin_decision(monkeypatch, requested):
    seen = []
    monkeypatch.setattr(
        ops.MNNVLAllReduce,
        "is_mnnvl",
        staticmethod(
            lambda m, d, explicitly_requested=False: seen.append(explicitly_requested) or True
        ),
    )
    assert mcast.mnnvl_supported("mapping", torch.bfloat16, requested) is True
    assert seen == [requested]


def test_one_shot_threshold_is_the_builtin_one():
    assert mcast.one_shot_threshold_bytes() == ops._MNNVL_ONE_SHOT_THRESHOLD_BYTES


@pytest.mark.parametrize(
    "multi_node, forced, expected", [(False, "0", False), (True, "0", True), (False, "1", True)]
)
def test_fabric_handle_rule(monkeypatch, multi_node, forced, expected):
    monkeypatch.setenv("TRTLLM_FORCE_MNNVL_AR", forced)
    mapping = SimpleNamespace(is_multi_node=lambda: multi_node)
    assert mcast.use_fabric_handle(mapping) is expected


class _FakeGroup:
    def __init__(self, log, ranks=None):
        self.log, self.ranks = log, ranks

    def Incl(self, ranks):
        self.log.append(("Incl", list(ranks)))
        return _FakeGroup(self.log, list(ranks))

    def Free(self):
        self.log.append(("Free group", self.ranks))


class _FakeSession:
    def __init__(self, log):
        self.log = log

    def Get_group(self):
        return _FakeGroup(self.log)

    def Create_group(self, group):
        self.log.append(("Create_group", group.ranks))
        return "tp comm"


def test_tp_group_comm_under_mpi_is_made_from_the_group_alone(monkeypatch):
    """Under MPI the communicator is ``MPI_Comm_create_group`` over exactly the TP group's ranks, in TP order: only
    those ranks take part. The temporary groups are freed."""
    log = []
    monkeypatch.setattr(utils, "mpi_disabled", lambda: False)
    monkeypatch.setattr(utils, "mpi_comm", lambda: _FakeSession(log))
    mapping = SimpleNamespace(tp_group=[4, 5, 6, 7])
    assert mcast.mnnvl_tp_group_comm(mapping) == "tp comm"
    assert log == [
        ("Incl", [4, 5, 6, 7]),
        ("Free group", None),
        ("Create_group", [4, 5, 6, 7]),
        ("Free group", [4, 5, 6, 7]),
    ]


def test_tp_group_comm_without_mpi_is_the_tp_process_group(monkeypatch):
    monkeypatch.setattr(utils, "mpi_disabled", lambda: True)
    pg = object()
    assert mcast.mnnvl_tp_group_comm(SimpleNamespace(tp_group_pg=pg)) is pg


@pytest.mark.parametrize("disabled", [False, True])
def test_free_group_comm_frees_only_an_mpi_communicator(monkeypatch, disabled):
    freed = []
    monkeypatch.setattr(utils, "mpi_disabled", lambda: disabled)
    mcast.free_group_comm(SimpleNamespace(Free=lambda: freed.append(True)))
    assert freed == ([] if disabled else [True])
