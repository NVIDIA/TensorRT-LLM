# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The order ``kda_handoff_layer`` issues its work in, with and without ``prep_ahead`` (CPU).

The layer runs over stand-ins of its kernels that record each call (the push returns a zeroed
region, the convolution zeros), on every rank's plan of a step. With ``prep_ahead``, a rank with a
lead of ``PREP_AHEAD_MIN_ROWS`` rows or more issues the ``"prep"`` half of each of the lead's
launches before the first hand-off round's push, each with a workspace of its own; the ``"chain"``
half of each follows the push of the round that brings the lead's state, on the same rows, slots
and workspace. Leaving the ``"prep"`` halves
out, the pushes and the launches come in the order they come without ``prep_ahead``, each
``"chain"`` half where the whole lead launch was; a shorter lead runs whole launches either way.

The bit-for-bit claim (the two halves are the whole call's kernels on its inputs) is GPU work:
``test_rubin_kda_prefill_phases.py`` (one GPU) and
``rubin_multirank/test_rubin_kda_handoff_prep_ahead.py`` (the layer across 4 or 8 ranks).
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm import (
    rubin_kda_handoff as ho,
)

pytestmark = pytest.mark.cpu_only

HEADS, D = 8, 128
_REGION = 32 << 20

# (request lengths, cached state per request, snapshots (request, offset), world)
_STEPS = [
    ((16384 * 4,), (True,), (), 4),
    ((4096 * 8 - 700, 300, 400), (True, False, True), ((0, 8256), (2, 256)), 8),
    ((300, 900, 5, 700, 40), (False, True, True, False, True), ((1, 256),), 4),
    ((3, 1, 520, 2, 260, 900, 70), (True, False, True, True, False, True, False), (), 8),
]


class _Ws:
    """A stand-in push workspace: its region size and push count."""

    def __init__(self) -> None:
        self.region = _REGION
        self.pushes = 0


def _layer(monkeypatch, lengths, has_init, snapshots, world, rank, prep_ahead):
    """The layer's recorded calls on rank ``rank``: ``("push",)`` and ``("prefill", phase, first
    row, rows, slots, workspace id)``; and the rank's plan."""
    log = []
    base = {}

    def push(ws, blocks, table):
        ws.pushes += 1
        log.append(("push",))
        return torch.zeros(_REGION, dtype=torch.uint8)

    def conv(src, weight, conv_state, starts, slots, has_init, max_seqlen):
        q = torch.zeros(1, src.shape[0], HEADS, D, dtype=torch.bfloat16)
        base["q"] = q.data_ptr()
        return q, q.clone(), q.clone()

    def prefill(q, k, v, g, beta, state_pool, state_indices, cu_seqlens, *args, **kw):
        first = (q.data_ptr() - base["q"]) // (HEADS * D * q.element_size())
        ws = kw.get("workspace")
        log.append(
            (
                "prefill",
                kw.get("phase", "both"),
                first,
                q.shape[1],
                tuple(state_indices.tolist()),
                None if ws is None else id(ws),
            )
        )
        return kw["out"]

    spaces = []

    def workspace(q, state_pool, state_indices, lower_bound):
        spaces.append(torch.empty(64, dtype=torch.uint8))
        return spaces[-1]

    monkeypatch.setattr(ho, "rubin_cp_push", push)
    monkeypatch.setattr(ho, "rubin_kda_prefill_conv", conv)
    monkeypatch.setattr(ho, "rubin_kda_prefill", prefill)
    monkeypatch.setattr(ho, "rubin_kda_prefill_workspace", workspace)
    slots = torch.arange(len(lengths), dtype=torch.int32)
    snap_slots = (
        torch.arange(len(lengths), len(lengths) + len(snapshots), dtype=torch.int64)
        if snapshots
        else None
    )
    plan = ho.kda_handoff_plan(
        lengths, has_init, snapshots, world, rank, HEADS, slots, _REGION, snap_slots
    )
    dim, hs = HEADS * D, HEADS // world
    nslots = len(lengths) + len(snapshots) + 1
    ho.kda_handoff_layer(
        plan,
        _Ws(),
        torch.zeros(plan.rows, 3 * dim, dtype=torch.bfloat16),
        torch.zeros(plan.rows, dim, dtype=torch.bfloat16),
        torch.zeros(plan.rows, HEADS, dtype=torch.bfloat16),
        torch.zeros(3 * dim, 4, dtype=torch.bfloat16),
        torch.zeros(HEADS),
        torch.zeros(dim),
        torch.zeros(nslots, hs, D, D),
        torch.zeros(nslots, 3 * hs * D, 3, dtype=torch.bfloat16),
        D**-0.5,
        -5.0,
        "bt16",
        prep_ahead=prep_ahead,
    )
    return log, plan


def _without_preps(log):
    """The pushes and launches without the ``"prep"`` halves, a ``"chain"`` half as a whole
    launch, workspaces dropped."""
    return [
        e if e[0] == "push" else ("prefill", "both") + e[2:5] for e in log if e[1:2] != ("prep",)
    ]


@pytest.mark.parametrize("lengths, has_init, snapshots, world", _STEPS)
def test_prep_ahead_moves_only_the_leads_factor_half(
    monkeypatch, lengths, has_init, snapshots, world
) -> None:
    for rank in range(world):
        base, plan = _layer(monkeypatch, lengths, has_init, snapshots, world, rank, False)
        ahead, _ = _layer(monkeypatch, lengths, has_init, snapshots, world, rank, True)
        assert all(e[1] == "both" for e in base if e[0] == "prefill"), rank
        assert _without_preps(ahead) == _without_preps(base), rank
        preps = [i for i, e in enumerate(ahead) if e[1:2] == ("prep",)]
        chains = [i for i, e in enumerate(ahead) if e[1:2] == ("chain",)]
        lead_rows = sum(la.hi - la.lo for la in plan.launch2)
        if plan.lead is None or lead_rows < ho.PREP_AHEAD_MIN_ROWS:
            assert not preps and not chains and ahead == base, rank
            continue
        assert len(preps) == len(chains) == len(plan.launch2) >= 1, rank
        pushes = [i for i, e in enumerate(ahead) if e[0] == "push"]
        x1 = 0 if plan.x1_skip else len(plan.push_x1)
        first_round, recv = pushes[x1], pushes[x1 + plan.recv_round]
        assert all(i < first_round for i in preps), rank
        assert all(i > recv for i in chains), rank
        for p, c in zip(preps, chains):
            assert ahead[p][2:] == ahead[c][2:] and ahead[p][5] is not None, (rank, p, c)
        assert len({ahead[p][5] for p in preps}) == len(preps), rank
