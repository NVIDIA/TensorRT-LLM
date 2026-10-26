# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The sm_107 TP8 MoE phase plan's half-unit FC1 ranges (MOE_HALFPLAN): with equal per-CTA shares (SH20's 192 routed
CTAs at 8 tokens) a split unit is halved instead of extended to its end, so the partial last FC1 group puts at most
half a unit on any CTA; unequal shares keep the extended ranges; TRTLLM_M31_DISABLE_MOE_HALFPLAN restores them."""

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp8.kernels.m3_moe import (  # noqa: E501
    op as moe_op,
)

M_MAX = 8


def _ranges(plan, g: int, num_ctas: int):
    """Per group p of G = g: the CTAs' [i0, i1) FC1 item ranges."""
    out = []
    for p in range((g + 3) // 4):
        words = [int(plan[g, c, p]) for c in range(num_ctas)]
        out.append([(w & 0xFFFF, w >> 16) if w else (0, 0) for w in words])
    return out


def _max_items(plan, g: int, num_ctas: int) -> int:
    return max(sum(i1 - i0 for i0, i1 in col) for col in zip(*_ranges(plan, g, num_ctas)))


@pytest.mark.parametrize("killed", [False, True])
def test_halfplan_last_group(monkeypatch: pytest.MonkeyPatch, killed: bool) -> None:
    monkeypatch.setattr(moe_op, "HALFPLAN", not killed)
    plan, _ = moe_op.make_phase_plan(192, M_MAX)
    # base: the partial last group's 1 + 5 splits; half units: 3 + 3
    expected = {29: 26, 30: 25, 31: 26} if killed else {29: 24, 30: 24, 31: 24}
    for g, most in expected.items():
        assert _max_items(plan, g, 192) == most
    for g in (28, 32):  # full groups only: 3 items per CTA either way
        assert _max_items(plan, g, 192) == (21 if g == 28 else 24)


@pytest.mark.parametrize("num_ctas", [192, 188])
def test_halfplan_ranges_valid(num_ctas: int) -> None:
    plan, _ = moe_op.make_phase_plan(num_ctas, M_MAX)
    s1 = moe_op._S1
    for g in range(moe_op.TOP_K, moe_op.TOP_K * M_MAX + 1):
        for p, ranges in enumerate(_ranges(plan, g, num_ctas)):
            n_p = min(4, g - 4 * p)
            total = n_p * moe_op.U1 * s1
            pos = 0
            for i0, i1 in ranges:  # contiguous in CTA order, covering the group
                if i1 > i0:
                    assert i0 == pos
                    pos = i1
            assert pos == total
            # every unit has at most one CTA boundary strictly inside it, and a range that starts inside a unit
            # runs to (at least) its end
            inner = [i0 for i0, i1 in ranges if i1 > i0 and i0 % s1]
            assert len({i0 // s1 for i0 in inner}) == len(inner)
            assert all(
                i1 >= (i0 // s1 + 1) * s1 or i1 == total for i0, i1 in ranges if i1 > i0 and i0 % s1
            )


def test_halfplan_unequal_shares_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    # 188 routed CTAs (SH20 off): 4 CTAs carry two FC2 blocks, the shares differ, the plan is the base plan
    on, _ = moe_op.make_phase_plan(188, M_MAX)
    monkeypatch.setattr(moe_op, "HALFPLAN", False)
    off, _ = moe_op.make_phase_plan(188, M_MAX)
    assert bool((on == off).all())
