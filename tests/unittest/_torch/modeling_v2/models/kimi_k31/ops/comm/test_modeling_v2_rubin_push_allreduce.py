# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Lamport push workspace builder of ``ops/comm/rubin_lamport_push.py``, on CPU.

``rubin_lamport_push_workspace`` sizes the shared MNNVL workspace once, for the largest call of 1
to ``max_rows`` rows on the transport each call takes, builds its view and holds the workspace.
``transport`` gives each call's transport and the C++ kernel's ranks per partial sum.

The cells of the family's Lamport entries (``rubin_attn_res_lamport_add_rmsnorm.CELLS``,
``rubin_latent_ar_norm_row.CELLS``) need four SM107 ranks of one tray, which no CI stage has.
"""

from __future__ import annotations

import types

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.comm import rubin_lamport_push as P

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


def _all_reduce(world=8, rank=3):
    return types.SimpleNamespace(
        mapping=types.SimpleNamespace(tp_size=world),
        mnnvl_allreduce=_Mnnvl(world, rank, torch.bfloat16),
        symm_mem_allreduce=None,
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
    monkeypatch.setattr(P, "get_env_enable_pdl", lambda: True)
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


@pytest.mark.parametrize(
    "world, last_one_shot_row, one_shot_chunk, two_shot_chunk", [(8, 9, 8, 8), (16, 4, 8, 16)]
)
def test_transport_is_the_cpp_kernels(
    scaled, world, last_one_shot_row, one_shot_chunk, two_shot_chunk
):
    """One-shot up to the default 1 MiB bound (9 rows of 7168 at 8 ranks, 4 at 16), two-shot
    above; the finish sums the ranks in the C++ kernel's chunks: 8 one-shot, up to 16 two-shot."""
    if P._MNNVL_ONE_SHOT_THRESHOLD_BYTES != 1 << 20:
        pytest.skip("TLLM_MNNVL_ONESHOT_THRESHOLD_BYTES overrides the default bound")
    ws = P.rubin_lamport_push_workspace(_all_reduce(world, 0), PDL_ROWS, HIDDEN)
    for rows in (1, last_one_shot_row, last_one_shot_row + 1, PDL_ROWS):
        one_shot = rows <= last_one_shot_row
        assert ws.transport(rows) == (one_shot, one_shot_chunk if one_shot else two_shot_chunk), (
            rows
        )
