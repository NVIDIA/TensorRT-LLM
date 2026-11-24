# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The cep target's sequence-split KDA with the serial FP32 hand-off (``rubin_kda_handoff``).

The split error, on one GPU: one request from a cached state cut into k = 1, 2, 4, 8 parts (one
per rank), each part run by ``rubin_kda_prefill`` from the FP32 end state of the part before, as
the hand-off rounds do (a push is a bit-exact copy, and each rank's convolution with its halo is
the one-rank convolution bit for bit; ``rubin_multirank/test_rubin_cp_push.py`` checks the layer
across ranks). Against the unsplit FP32 recurrence: the hand-off carries the chain's own FP32
state, so its error stays at the k = 1 floor (only the 16-token chunk grid restarts at a rank
boundary); the relay's composition grows with k (``test_rubin_kda_relay.py``).

The step plan, on the host: the hand-off rounds and the exchanges' push tables agree across ranks.
"""

import pytest
import torch
from _rubin_kda_prefill_inputs import HEAD_DIM, LOWER_BOUND, SCALE, coalesced_pool
from _rubin_kda_split_inputs import SPLIT_KS, rel_err, split_case

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.rubin_kda_handoff import (
    kda_handoff_plan,
    max_span,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.rubin_kda_prefill import (
    rubin_kda_prefill,
)

_GPU = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)),
    reason="the BT16 KDA engine runs on SM100, SM103 and SM107",
)


def handoff_chain(c: dict, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The hand-off of ``split_case`` ``c`` over ``k`` ranks: ``(outputs [T, H, D], end state)``."""
    heads, tokens = c["heads"], c["tokens"]
    q, k_, v = c["q"], c["k"], c["v"]
    g = c["g"].view(1, tokens, heads, HEAD_DIM)
    beta = c["beta"].unsqueeze(0)
    pool = coalesced_pool(2, (heads, HEAD_DIM, HEAD_DIM), torch.float32, q.device)
    pool[1].copy_(c["state0"])
    out = torch.empty_like(q)
    rows = -(-tokens // k)
    row = torch.ones(1, dtype=torch.int32, device=q.device)
    for lo in range(0, tokens, rows):
        hi = min(lo + rows, tokens)
        sl = slice(lo, hi)
        rubin_kda_prefill(
            q[:, sl],
            k_[:, sl],
            v[:, sl],
            g[:, sl],
            beta[:, sl],
            pool,
            row,
            torch.tensor([0, hi - lo], dtype=torch.int64, device=q.device),
            c["a_log"],
            c["dt_bias"],
            SCALE,
            LOWER_BOUND,
            "bt16",
            out=out[:, sl],
        )
    return out[0], pool[1].clone()


@_GPU
@torch.inference_mode()
def test_handoff_split_error_stays_at_the_floor() -> None:
    """The largest error of the end state and of the outputs relative to the unsplit FP32
    recurrence's largest magnitude, k = 1, 2, 4, 8: k = 1 is the kernel's floor; the hand-off
    stays within 1.5x of it at every k."""
    c = split_case()
    errors = {}
    for k in SPLIT_KS:
        out, end = handoff_chain(c, k)
        errors[k] = (rel_err(end, c["ref_state"]), rel_err(out, c["ref_out"]))
    print(
        "\nhand-off split error vs FP32 (end state / outputs): "
        + ", ".join(f"k={k} {s:.3e} / {o:.3e}" for k, (s, o) in errors.items())
    )
    floor_s, floor_o = errors[1]
    for k, (s, o) in errors.items():
        assert s <= 1.5 * floor_s and o <= 1.5 * floor_o, (k, errors)


@pytest.mark.parametrize(
    "lengths, has_init, snapshots, world",
    [
        ((300, 900, 5, 700, 40), (False, True, True, False, True), ((1, 256),), 4),
        ((3, 1, 520, 2, 260, 900, 70), (True, False, True, True, False, True, False), (), 8),
        ((114, 118), (False, True), (), 4),
        ((21, 4), (True, False), (), 8),
        ((4096 * 8 - 700, 300, 400), (True, False, True), ((0, 8256), (2, 256)), 8),
    ],
)
def test_plan_hands_off_in_rank_order(lengths, has_init, snapshots, world) -> None:
    """Every continuing request's end state goes to the next rank in the round in which that
    rank's lead waits for it; the rounds number the span minus one on every rank; the x1 and x3
    pushes of every pair of ranks agree on the rows moved; every snapshot is taken once."""
    slots = torch.arange(len(lengths), dtype=torch.int32)
    plans = [
        kda_handoff_plan(lengths, has_init, snapshots, world, r, 16, slots, 32 << 20)
        for r in range(world)
    ]
    assert all(len(p.push_rounds) == max(0, max_span(lengths, world) - 1) for p in plans)
    for r, p in enumerate(plans):
        if p.p2p_send_row >= 0:
            assert plans[r + 1].lead is not None and plans[r + 1].recv_round == p.send_round, r
        if p.lead is not None:
            assert plans[r - 1].p2p_send_row >= 0, r
    for name in ("push_x1", "push_x3"):
        pushes = [getattr(p, name) for p in plans]
        assert len({len(x) for x in pushes}) == 1, name
        for k in range(len(pushes[0])):
            for d in range(world):
                sent = sum(pushes[s][k][0][3 * d + 2] for s in range(world))
                assert pushes[d][k][2] == sent, (name, k, d)
    assert sum(len(p.x3_snap_local) for p in plans) == len(snapshots)
    assert sorted(i for p in plans for i in (f.req for f in p.frags if f.last)) == list(
        range(len(lengths))
    )
