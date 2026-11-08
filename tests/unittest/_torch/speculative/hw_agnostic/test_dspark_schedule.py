# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSpark per-request allocation tests."""

import pytest
import torch

from tensorrt_llm._torch.speculative.dspark_schedule import (
    DSparkScheduleConfig,
    compute_survival,
    schedule_verify_lens_topk,
)

BLOCK = 7


def _cfg(**kwargs) -> DSparkScheduleConfig:
    return DSparkScheduleConfig(block_size=BLOCK, **kwargs)


def test_allocation_is_a_prefix_and_respects_bounds():
    torch.manual_seed(3)
    cfg = _cfg(min_verify_len=1)
    surv = compute_survival(torch.rand(12, BLOCK))
    lens = schedule_verify_lens_topk(survival=surv, budget=20, cfg=cfg)
    assert lens.dtype == torch.int32
    assert torch.all(lens >= cfg.min_verify_len)
    assert torch.all(lens <= cfg.resolved_max_verify_len)
    # Budget is spent above the floor and never exceeded.
    assert int((lens - cfg.min_verify_len).sum()) <= 20
    # Zero budget degenerates to the floor for everyone.
    cfg_floor = _cfg(min_verify_len=2)
    lens = schedule_verify_lens_topk(
        survival=compute_survival(torch.rand(5, BLOCK)), budget=0, cfg=cfg_floor
    )
    assert torch.equal(lens, torch.full((5,), 2, dtype=torch.int32))
    # A huge budget saturates at the cap.
    cfg_cap = _cfg(min_verify_len=1, max_verify_len=4)
    lens = schedule_verify_lens_topk(
        survival=compute_survival(torch.full((5, BLOCK), 0.99)), budget=10**6, cfg=cfg_cap
    )
    assert torch.equal(lens, torch.full((5,), 4, dtype=torch.int32))
    # An empty batch allocates nothing.
    assert (
        schedule_verify_lens_topk(survival=torch.zeros(0, BLOCK), budget=5, cfg=_cfg()).numel() == 0
    )


def test_survival_eps_excludes_hopeless_positions():
    """Even with unlimited budget, sub-eps candidates are never admitted."""
    cfg = _cfg(min_verify_len=1, survival_eps=1e-3)
    conf = torch.tensor([[1.0, 1.0, 1e-9, 1.0, 1.0, 1.0, 1.0]])
    lens = schedule_verify_lens_topk(survival=compute_survival(conf), budget=10**6, cfg=cfg)
    # positions 0,1 survive; from position 2 on, survival collapses below eps.
    assert int(lens[0]) == 2


def test_tied_confidences_are_balanced_and_deterministic():
    """Ties form a stable balanced prefix on every rank."""
    cfg = _cfg(min_verify_len=1)
    surv = compute_survival(torch.full((9, BLOCK), 0.9))
    lens = schedule_verify_lens_topk(survival=surv, budget=13, cfg=cfg)
    # ``budget`` is allocated above the one-token floor already granted to
    # every row, so the result contains 9 floor tokens plus 13 winners.
    assert lens.tolist() == [3, 3, 3, 3, 2, 2, 2, 2, 2]
    assert int(lens.sum()) == 9 + 13
    assert torch.equal(schedule_verify_lens_topk(survival=surv, budget=13, cfg=cfg), lens)


def test_schedule_config_rejects_bad_bounds():
    with pytest.raises(ValueError, match="min_verify_len must be >= 1"):
        DSparkScheduleConfig(block_size=BLOCK, min_verify_len=0)
    with pytest.raises(ValueError, match="block_size must be >= 1"):
        DSparkScheduleConfig(block_size=0)
    with pytest.raises(ValueError, match="< min_verify_len"):
        DSparkScheduleConfig(block_size=BLOCK, min_verify_len=5, max_verify_len=3)
    with pytest.raises(ValueError, match=r"\[bs, K\]"):
        compute_survival(torch.rand(BLOCK))


def _block5_cfg() -> DSparkScheduleConfig:
    return DSparkScheduleConfig(block_size=5, min_verify_len=1, max_verify_len=5)


def test_topk_hands_longer_windows_to_more_confident_requests():
    """The budget must follow survival, not batch position."""
    # survival[r, k] = P(the first k+1 drafted tokens all get accepted)
    survival = torch.tensor(
        [
            [0.99, 0.98, 0.97, 0.96, 0.95],  # confident
            [0.90, 0.70, 0.50, 0.30, 0.10],  # middling
            [0.20, 0.04, 0.01, 0.00, 0.00],  # collapses
        ]
    )
    lens = schedule_verify_lens_topk(survival=survival, budget=6, cfg=_block5_cfg()).tolist()

    assert len(lens) == 3
    assert lens[0] > lens[2], (
        f"confident request got {lens[0]}, collapsing request got {lens[2]}; "
        f"the budget must follow survival, not batch position"
    )
    assert lens[0] >= lens[1] >= lens[2]
    # Every request keeps at least the floor, and the budget is respected.
    assert min(lens) >= 1
    assert sum(lens) - len(lens) * 1 <= 6
