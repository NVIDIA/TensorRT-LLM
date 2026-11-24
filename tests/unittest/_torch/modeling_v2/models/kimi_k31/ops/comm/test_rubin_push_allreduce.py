# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Lamport push workspace builder of ``ops/comm/rubin_push_allreduce.py``, on CPU.

``rubin_lamport_push_workspace`` sizes the shared MNNVL workspace once, for the largest call of 1
to ``max_rows`` rows on the transport each call takes, builds its view and holds the workspace.
``transport`` gives each call's transport and the C++ kernel's ranks per partial sum. Without an
MNNVL bf16 path over more than one rank the builder gives None and scales nothing.
"""

from __future__ import annotations

import types

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.comm import (
    rubin_push_allreduce as P,
)

pytestmark = pytest.mark.cpu_only

HIDDEN = 7168
PDL_ROWS = 128


def _lamport_bytes(rows: int, hidden: int, world: int, one_shot: bool, ag: bool = False) -> int:
    """One Lamport buffer's bytes for a call (the family kernel's rule)."""
    if one_shot:
        return rows * hidden * (world + (1 if ag else 0)) * 2
    return 2 * -(-rows // world) * world * hidden * 2


class _Mnnvl:
    def __init__(self, world: int, rank: int, dtype: torch.dtype):
        self.mapping = types.SimpleNamespace(tp_size=world, tp_rank=rank)
        self.dtype = dtype

    @staticmethod
    def get_required_workspace_size(rows: int, hidden: int, world: int, dtype) -> int:
        one_shot = rows * hidden * world * 2 <= P._MNNVL_ONE_SHOT_THRESHOLD_BYTES
        return _lamport_bytes(rows, hidden, world, one_shot)


def _all_reduce(world=8, rank=3, dtype=torch.bfloat16, mnnvl=True, symm=None):
    return types.SimpleNamespace(
        mapping=types.SimpleNamespace(tp_size=world),
        mnnvl_allreduce=_Mnnvl(world, rank, dtype) if mnnvl else None,
        symm_mem_allreduce=symm,
    )


@pytest.fixture
def scaled(monkeypatch):
    """The sizes the builder asked the shared workspace for."""
    calls = []

    def get_or_scale(mapping, dtype, buffer_size_bytes=None):
        calls.append(buffer_size_bytes)
        return {"buffer_size_bytes": buffer_size_bytes, "mapping": mapping}

    def from_mnnvl(workspace, world, rank):
        return types.SimpleNamespace(world=world, rank=rank, of=workspace)

    monkeypatch.setattr(P, "get_or_scale_allreduce_mnnvl_workspace", get_or_scale)
    monkeypatch.setattr(
        P, "LamportWorkspace", types.SimpleNamespace(from_mnnvl=from_mnnvl), raising=False
    )
    monkeypatch.setattr(P, "required_bytes", _lamport_bytes, raising=False)
    monkeypatch.setattr(P, "_use_pdl", lambda: True)
    return calls


def test_one_workspace_sized_for_every_call_and_held(scaled):
    ws = P.rubin_lamport_push_workspace(_all_reduce(), PDL_ROWS, HIDDEN)
    sizes = []
    for rows in range(1, PDL_ROWS + 1):
        one_shot, _ = ws.transport(rows)
        sizes.append(_lamport_bytes(rows, HIDDEN, 8, one_shot))
    assert scaled == [max(sizes)]
    assert ws.workspace["buffer_size_bytes"] == max(sizes)
    assert ws.view.of is ws.workspace and (ws.view.world, ws.view.rank) == (8, 3)
    assert (ws.hidden, ws.pdl) == (HIDDEN, True)


@pytest.mark.parametrize("world", [4, 8, 16])
def test_transport_is_the_cpp_kernels(scaled, world):
    """One-shot up to the fused-finish bound, two-shot above; the finish sums the ranks in the
    chunks of the C++ kernel the shared threshold selects."""
    ws = P.rubin_lamport_push_workspace(_all_reduce(world, 0), PDL_ROWS, HIDDEN)
    seen = set()
    for rows in range(1, PDL_ROWS + 1):
        nbytes = rows * HIDDEN * world * 2
        one_shot, chunk = ws.transport(rows)
        cpp_one_shot = nbytes <= P._MNNVL_ONE_SHOT_THRESHOLD_BYTES
        assert one_shot == (nbytes <= P._K3_FUSED_FINISH_ONE_SHOT_MAX_BYTES), rows
        assert chunk == ((world if world <= 8 else 8) if cpp_one_shot else min(world, 16)), rows
        seen.add(one_shot)
    if P._MNNVL_ONE_SHOT_THRESHOLD_BYTES < PDL_ROWS * HIDDEN * world * 2:
        assert seen == {True, False}, "the sizing covers both transports"


@pytest.mark.parametrize(
    "why,all_reduce,max_rows,hidden",
    [
        ("no MNNVL path", _all_reduce(mnnvl=False), PDL_ROWS, HIDDEN),
        ("one rank", _all_reduce(world=1, rank=0), PDL_ROWS, HIDDEN),
        ("symmetric memory", _all_reduce(symm=object()), PDL_ROWS, HIDDEN),
        ("fp16 all-reduce", _all_reduce(dtype=torch.float16), PDL_ROWS, HIDDEN),
        ("no PDL rows", _all_reduce(), 0, HIDDEN),
        ("hidden not a multiple of 8", _all_reduce(), PDL_ROWS, HIDDEN + 4),
        ("past a 32-bit size", _all_reduce(), 1024, 1 << 21),
    ],
)
def test_no_push_path_gives_none(scaled, why, all_reduce, max_rows, hidden):
    assert P.rubin_lamport_push_workspace(all_reduce, max_rows, hidden) is None, why
    assert scaled == []
