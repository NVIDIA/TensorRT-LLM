# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual MPI fixture's pre-allocation capability gate without GPUs or MPI."""

import ast
from pathlib import Path
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.cpu_only

SOURCE = Path(__file__).resolve().parent / "spec/test_modeling_v2_m31_vr_tp2_draft_prob1.py"


def _run(available: bool, supported: bool) -> tuple[list, Mock]:
    tree = ast.parse(SOURCE.read_text())
    env = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_env")
    stop = next(
        i
        for i, n in enumerate(env.body)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "mapping" for t in n.targets)
    )
    env.body = [n for n in env.body[:stop] if not isinstance(n, (ast.Import, ast.ImportFrom))]
    events = []
    local = Mock(spec_set=["Get_rank", "Get_size", "Free"])
    local.Get_rank.return_value, local.Get_size.return_value = 1, 2
    comm = Mock(spec_set=["Get_rank", "Split_type", "allreduce"])
    comm.Get_rank.return_value = 1
    comm.Split_type.return_value = local
    comm.allreduce.side_effect = [available, supported]
    cuda = Mock(spec_set=["is_available", "device_count", "set_device", "get_device_capability"])
    cuda.is_available.return_value = True
    cuda.device_count.return_value = 2 if available else 1
    cuda.set_device.side_effect = lambda rank: events.append(("set", rank))
    cuda.get_device_capability.side_effect = lambda: events.append("capability") or (10, 7)
    pytest = Mock(spec_set=["skip"])
    pytest.skip.side_effect = RuntimeError("collective skip")
    scope = dict(
        _S={},
        COMM=comm,
        MPI=Mock(spec_set=["COMM_TYPE_SHARED", "LAND"], COMM_TYPE_SHARED="local", LAND="and"),
        torch=Mock(spec_set=["cuda"], cuda=cuda),
        pytest=pytest,
    )
    exec(compile(ast.Module(body=[env], type_ignores=[]), str(SOURCE), "exec"), scope)
    try:
        scope["_env"]()
    except RuntimeError as error:
        assert str(error) == "collective skip"
        assert not available or not supported
    else:
        assert available and supported
    local.Free.assert_called_once()
    return events, comm


def test_insufficient_local_gpus_skip_every_rank_before_device_selection() -> None:
    events, comm = _run(False, False)
    assert events == []
    comm.allreduce.assert_called_once_with(False, op="and")


def test_capability_is_checked_on_the_selected_rank_device() -> None:
    events, comm = _run(True, True)
    assert events == [("set", 1), "capability"]
    assert comm.allreduce.call_count == 2


def test_an_unsupported_peer_causes_a_collective_skip() -> None:
    events, comm = _run(True, False)
    assert events == [("set", 1), "capability"]
    assert comm.allreduce.call_count == 2
