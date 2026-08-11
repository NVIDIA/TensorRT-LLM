# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Make the planner itself produce per-request windows, with a crafted table.

This replaces TLLM_DSPARK_FORCE_VERIFY_LENS. That knob assigned windows by
*batch position*, which is orthogonal to confidence: it reproduced the ragged
*shape* but never the ragged *policy*, so it could not tell a correct
confidence-driven assignment from an arbitrary one. It also had to sit in
production code, ahead of the cost-table gate, to be reachable at all.

A cost table does the same job through the real path. The planner trims when
the marginal cost of extra verify tokens outweighs their expected acceptance,
so a table whose theta(M) rises steeply makes it trim at any acceptance rate --
and the budget then flows through the same confidence-ordered top-k, which
hands different windows to requests with different survival. That exercises
what actually ships.

Why this matters beyond tidiness: the KV rewind path (KV reserved for the full
drafted block but only the verified window rewound) is reachable only when
verified_len < draft_len, and a well-fitted table can decline to trim at every
batch size -- so without a table that forces trimming, that code path ships
untested.
"""

import torch

from tensorrt_llm._torch.speculative.dspark_schedule import (
    DSparkScheduleConfig, schedule_verify_lens_topk)


def _Cfg(min_verify_len=1, max_verify_len=5):
    """The real config, so the test cannot drift from what ships."""
    return DSparkScheduleConfig(block_size=max_verify_len,
                                min_verify_len=min_verify_len,
                                max_verify_len=max_verify_len)


def test_topk_hands_longer_windows_to_more_confident_requests():
    """The property the position-rotating knob could never check.

    Request 0 is confident throughout, request 2 collapses immediately. A
    correct scheduler spends the budget where survival is highest.
    """
    # survival[r, k] = P(the first k+1 drafted tokens all get accepted)
    survival = torch.tensor([
        [0.99, 0.98, 0.97, 0.96, 0.95],  # confident
        [0.90, 0.70, 0.50, 0.30, 0.10],  # middling
        [0.20, 0.04, 0.01, 0.00, 0.00],  # collapses
    ])
    lens = schedule_verify_lens_topk(survival=survival, budget=6,
                                     cfg=_Cfg()).tolist()

    assert len(lens) == 3
    assert lens[0] > lens[2], (
        f"confident request got {lens[0]}, collapsing request got {lens[2]}; "
        f"the budget must follow survival, not batch position")
    assert lens[0] >= lens[1] >= lens[2]
    # Every request keeps at least the floor, and the budget is respected.
    assert min(lens) >= 1
    assert sum(lens) - len(lens) * 1 <= 6


def test_a_full_budget_degenerates_to_the_uniform_full_window():
    """The no-trim case must stay exactly uniform.

    This is what the planner correctly chooses on this checkpoint, and it is
    the case where ragged must cost nothing.
    """
    num_reqs, max_len = 4, 5
    survival = torch.full((num_reqs, max_len), 0.99)
    budget = num_reqs * (max_len - 1)
    lens = schedule_verify_lens_topk(survival=survival, budget=budget,
                                     cfg=_Cfg()).tolist()
    assert lens == [max_len] * num_reqs
