# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for DSpark confidence-scheduled verification (hardware-agnostic, CPU).

These cover the parts where a silent error would be expensive and invisible at
runtime: the survival semantics, the budget argmax over a staircase cost curve,
the prefix property of the allocation, and cross-rank determinism.
"""

import itertools

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.speculative.dspark_planner import (
    SpsCostTable,
    budget_argmax_over_uniform_lens,
    compute_verify_token_budget,
    derive_verify_len_tiers,
)
from tensorrt_llm._torch.speculative.dspark_schedule import (
    DSparkScheduleConfig,
    compute_survival,
    schedule_verify_lens_topk,
)

BLOCK = 7


def _cfg(**kw) -> DSparkScheduleConfig:
    return DSparkScheduleConfig(block_size=BLOCK, **kw)


# --------------------------------------------------------------------------
# survival
# --------------------------------------------------------------------------


def test_survival_is_cumulative_product():
    conf = torch.tensor([[0.9, 0.8, 0.5, 1.0, 1.0, 1.0, 1.0]])
    surv = compute_survival(conf)
    assert torch.allclose(surv[0, :3], torch.tensor([0.9, 0.72, 0.36]), atol=1e-6)


def test_survival_is_non_increasing():
    """The allocator's prefix property depends on this holding for any input."""
    torch.manual_seed(0)
    conf = torch.rand(16, BLOCK)
    surv = compute_survival(conf)
    assert torch.all(surv[:, 1:] <= surv[:, :-1] + 1e-7)


def test_survival_rejects_wrong_rank():
    with pytest.raises(ValueError, match=r"\[bs, K\]"):
        compute_survival(torch.rand(BLOCK))


# --------------------------------------------------------------------------
# budget argmax
# --------------------------------------------------------------------------


def _brute_force_budget(survival, num_gen, table, min_len=1, max_len=None):
    """Independent O(N) reimplementation used as the oracle for the argmax.

    Spells out the objective: expected emitted tokens per millisecond, where the
    expectation counts one bonus token per request, the always-verified floor
    positions, and the admitted candidates. The cost is looked up on the tokens
    actually submitted to the target -- ``bs * (min_len + 1)`` at the floor,
    because every request also submits its bonus position.
    """
    bs, blk = survival.shape
    cap = min(max_len or blk, blk)
    cand = np.sort(survival[:, min_len:cap].reshape(-1))[::-1]
    base = num_gen + float(survival[:, :min_len].sum())
    floor_tokens = bs * (min_len + 1)
    best_n, best = 0, -np.inf
    for n in range(cand.size + 1):
        theta = (base + float(cand[:n].sum())) / table.step_time(floor_tokens + n, num_gen)
        if theta > best:
            best, best_n = theta, n
    return best_n


def test_tau_counts_the_always_verified_floor_positions():
    """The floor is verified for free, but its yield still belongs in tau.

    argmax((C + f(n)) / g(n)) depends on C, so dropping the floor's contribution
    silently moves the optimum. Two batches that differ only in how good their
    floor positions are must be able to choose different budgets.
    """
    # The riser sits just past the floor (bs*(min_len+1) = 8 tokens), so both
    # shelves are reachable and the constant term can actually decide between
    # them. With the riser outside the reachable range this test would still
    # pass but prove nothing.
    table = SpsCostTable(token_counts=(0, 10), step_time_ms=(1.0, 3.0), fixed_overhead_ms=1.0)
    bs = 4
    good_floor = np.cumprod(np.full((bs, BLOCK), 0.99), axis=1)
    bad_floor = good_floor.copy()
    bad_floor[:, 0] = 0.01  # a hopeless first position drags the whole row down
    bad_floor = np.cumprod(
        np.concatenate([bad_floor[:, :1], np.full((bs, BLOCK - 1), 0.99)], axis=1), axis=1
    )
    budgets = []
    for surv in (good_floor, bad_floor):
        got = compute_verify_token_budget(survival=surv, num_gen_requests=bs, cost_table=table)
        assert got == _brute_force_budget(surv, bs, table)
        budgets.append(got)
    assert budgets[0] != budgets[1], (
        "the two batches differ only in their floor positions' survival, so a "
        "planner that ignored the floor's contribution to tau would give them "
        "the same budget -- keep the riser inside the reachable token range"
    )


def test_budget_matches_brute_force_on_staircase():
    """Theta is not unimodal on a staircase; a greedy first-descent scan is wrong."""
    rng = np.random.default_rng(7)
    # A cost curve with genuine shelves and risers.
    table = SpsCostTable(
        token_counts=(0, 64, 128, 192, 256, 320),
        step_time_ms=(4.0, 4.05, 6.5, 6.55, 9.0, 9.05),
    )
    for trial in range(25):
        bs = int(rng.integers(1, 40))
        conf = rng.uniform(0.55, 0.99, size=(bs, BLOCK))
        surv = np.cumprod(conf, axis=1)
        got = compute_verify_token_budget(survival=surv, num_gen_requests=bs, cost_table=table)
        want = _brute_force_budget(surv, bs, table)
        assert got == want, f"trial {trial}: bs={bs} got={got} want={want}"


def test_greedy_first_descent_would_be_wrong():
    """Pin the specific failure mode the global argmax exists to avoid.

    Theta rises along a shelf, drops at a riser, then rises again along the next
    shelf. A greedy loop that stops at the first non-improvement (as the paper's
    pseudocode does) parks at the end of the *first* shelf; here the second
    shelf's optimum is almost 3x better.
    """
    bs = 4
    # One riser just past the floor (bs*(min_len+1) = 8 tokens), then a long
    # shelf out to the end of the range. The fixed overhead is spelled out
    # rather than assumed: it is part of what makes the first shelf's Theta
    # look competitive, and it is no longer a hidden constant in the planner.
    table = SpsCostTable(token_counts=(0, 10), step_time_ms=(1.0, 3.0), fixed_overhead_ms=1.0)
    # Uniformly high confidence: every extra candidate adds nearly a full token,
    # so tau grows ~7x across the range while cost only doubles.
    surv = np.cumprod(np.full((bs, BLOCK), 0.995), axis=1)
    n_star = compute_verify_token_budget(survival=surv, num_gen_requests=bs, cost_table=table)

    cand = np.sort(surv[:, 1:].reshape(-1))[::-1]

    def theta(n):
        return (bs + cand[:n].sum()) / table.step_time(2 * bs + n, bs)

    greedy = 0
    while greedy + 1 <= cand.size and theta(greedy + 1) > theta(greedy):
        greedy += 1

    assert theta(n_star) >= theta(greedy)
    assert n_star > greedy, "expected the staircase to defeat a first-descent scan"
    assert theta(n_star) > 2 * theta(greedy), "the gap should be large, not marginal"


def test_flat_cost_table_degenerates_to_verify_all():
    """A flat cost model makes every token free -> spend the whole budget."""
    table = SpsCostTable.flat(2.0)
    assert table.is_flat
    surv = np.cumprod(np.full((6, BLOCK), 0.9), axis=1)
    budget = compute_verify_token_budget(survival=surv, num_gen_requests=6, cost_table=table)
    assert budget == 6 * (BLOCK - 1)


def test_low_confidence_yields_small_budget():
    table = SpsCostTable(token_counts=(0, 8, 16, 24, 32), step_time_ms=(2.0, 3.0, 4.0, 5.0, 6.0))
    hi = np.cumprod(np.full((8, BLOCK), 0.98), axis=1)
    lo = np.cumprod(np.full((8, BLOCK), 0.30), axis=1)
    b_hi = compute_verify_token_budget(survival=hi, num_gen_requests=8, cost_table=table)
    b_lo = compute_verify_token_budget(survival=lo, num_gen_requests=8, cost_table=table)
    assert b_lo < b_hi


def test_budget_is_zero_for_empty_batch():
    table = SpsCostTable.flat()
    assert (
        compute_verify_token_budget(
            survival=np.zeros((0, BLOCK)), num_gen_requests=0, cost_table=table
        )
        == 0
    )


def test_cost_table_is_a_staircase_not_an_interpolation():
    table = SpsCostTable(token_counts=(0, 100), step_time_ms=(1.0, 5.0))
    # Everything below the next breakpoint costs the same as the shelf it is on.
    assert table.step_time(1) == table.step_time(99)
    assert table.step_time(100) > table.step_time(99)
    # Above the last breakpoint we clamp rather than extrapolate.
    assert table.step_time(10_000) == table.step_time(100)


def test_cost_table_rejects_malformed_input():
    with pytest.raises(ValueError, match="strictly increasing"):
        SpsCostTable(token_counts=(0, 0), step_time_ms=(1.0, 2.0))
    with pytest.raises(ValueError, match="same length"):
        SpsCostTable(token_counts=(0, 1), step_time_ms=(1.0,))
    with pytest.raises(ValueError, match="positive"):
        SpsCostTable(token_counts=(0,), step_time_ms=(0.0,))


# --------------------------------------------------------------------------
# per-request allocation
# --------------------------------------------------------------------------


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


def test_allocation_prefers_the_most_confident_requests():
    cfg = _cfg(min_verify_len=1)
    conf = torch.tensor(
        [
            [1.0, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99],  # confident
            [1.0, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10],  # not
        ]
    )
    lens = schedule_verify_lens_topk(survival=compute_survival(conf), budget=6, cfg=cfg)
    assert lens[0] > lens[1]


def test_zero_budget_gives_everyone_the_floor():
    cfg = _cfg(min_verify_len=2)
    surv = compute_survival(torch.rand(5, BLOCK))
    lens = schedule_verify_lens_topk(survival=surv, budget=0, cfg=cfg)
    assert torch.equal(lens, torch.full((5,), 2, dtype=torch.int32))


def test_huge_budget_saturates_at_the_cap():
    cfg = _cfg(min_verify_len=1, max_verify_len=4)
    surv = compute_survival(torch.full((5, BLOCK), 0.99))
    lens = schedule_verify_lens_topk(survival=surv, budget=10**6, cfg=cfg)
    assert torch.equal(lens, torch.full((5,), 4, dtype=torch.int32))


def test_survival_eps_excludes_hopeless_positions():
    """Even with unlimited budget, sub-eps candidates are never admitted."""
    cfg = _cfg(min_verify_len=1, survival_eps=1e-3)
    conf = torch.tensor([[1.0, 1.0, 1e-9, 1.0, 1.0, 1.0, 1.0]])
    lens = schedule_verify_lens_topk(survival=compute_survival(conf), budget=10**6, cfg=cfg)
    # positions 0,1 survive; from position 2 on, survival collapses below eps.
    assert int(lens[0]) == 2


def test_allocation_is_deterministic_under_tied_confidences():
    """Two ranks with identical input must produce identical verify lengths."""
    cfg = _cfg(min_verify_len=1)
    surv = compute_survival(torch.full((9, BLOCK), 0.9))
    first = schedule_verify_lens_topk(survival=surv, budget=13, cfg=cfg)
    for _ in range(8):
        assert torch.equal(schedule_verify_lens_topk(survival=surv, budget=13, cfg=cfg), first)


def test_ties_break_toward_earlier_positions():
    """A tie must not hand a later position to one request over an earlier one.

    Every request has identical confidence, so all candidates at a given
    position tie. The allocation must stay balanced (a prefix front), not give
    one request a deep suffix while another gets nothing.
    """
    cfg = _cfg(min_verify_len=1)
    surv = compute_survival(torch.full((4, BLOCK), 1.0))
    lens = schedule_verify_lens_topk(survival=surv, budget=4, cfg=cfg)
    assert int(lens.max()) - int(lens.min()) <= 1


def test_empty_batch_returns_empty():
    cfg = _cfg()
    lens = schedule_verify_lens_topk(survival=torch.zeros(0, BLOCK), budget=5, cfg=cfg)
    assert lens.numel() == 0


def test_schedule_config_rejects_bad_bounds():
    with pytest.raises(ValueError, match="min_verify_len must be >= 1"):
        DSparkScheduleConfig(block_size=BLOCK, min_verify_len=0)
    with pytest.raises(ValueError, match="block_size must be >= 1"):
        DSparkScheduleConfig(block_size=0)
    with pytest.raises(ValueError, match="< min_verify_len"):
        DSparkScheduleConfig(block_size=BLOCK, min_verify_len=5, max_verify_len=3)


# --------------------------------------------------------------------------
# budget conservation: allocation must never exceed what the planner sized
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bs,min_len", list(itertools.product([1, 3, 17], [1, 2])))
def test_planner_and_allocator_agree_on_the_budget(bs, min_len):
    rng = np.random.default_rng(bs * 100 + min_len)
    table = SpsCostTable(token_counts=(0, 16, 32, 64, 128), step_time_ms=(3.0, 3.1, 5.0, 5.1, 9.0))
    conf = rng.uniform(0.4, 0.99, size=(bs, BLOCK))
    surv_np = np.cumprod(conf, axis=1)
    budget = compute_verify_token_budget(
        survival=surv_np, num_gen_requests=bs, cost_table=table, min_verify_len=min_len
    )
    lens = schedule_verify_lens_topk(
        survival=torch.from_numpy(surv_np).float(),
        budget=budget,
        cfg=_cfg(min_verify_len=min_len),
    )
    assert int((lens - min_len).sum()) <= budget
    assert torch.all(lens >= min_len)


# --------------------------------------------------------------------------
# uniform-K projection (what TRT-LLM's graph key can actually express today)
# --------------------------------------------------------------------------


def test_discrete_argmax_beats_round_then_snap_across_a_riser():
    """Optimizing over runnable lengths is not the same as rounding the budget.

    The continuous optimum sits just past a cost riser; snapping down lands on a
    length whose Theta is worse than a length the discrete search finds.
    """
    table = SpsCostTable(token_counts=(0, 8, 24, 40), step_time_ms=(2.0, 2.05, 2.1, 20.0))
    surv = np.cumprod(np.full((8, BLOCK), 0.97), axis=1)
    chosen = budget_argmax_over_uniform_lens(
        survival=surv,
        num_gen_requests=8,
        cost_table=table,
        allowed_lens=[1, 3, 5, 7],
    )
    # 8*(7+1)=64 tokens lands on the 20ms riser; the search must avoid it.
    assert chosen < 7

    def theta(length):
        # Mirrors the implementation exactly, including the bonus token every
        # request submits alongside its drafts.
        tau = 8 + float(surv[:, :length].sum())
        return tau / table.step_time(8 * (length + 1), 8)

    assert theta(chosen) == max(theta(v) for v in (1, 3, 5, 7))


def test_derived_tiers_lose_nothing_versus_continuous_k():
    """The shelf-right-edge property: derived tiers contain the true optimum.

    Within a cost shelf the step time is constant while tau strictly increases,
    so Theta rises monotonically across the shelf and the optimum can only sit
    at a right edge. A tier set built from the right edges must therefore match
    an exhaustive search over *every* length.
    """
    rng = np.random.default_rng(11)
    table = SpsCostTable(
        token_counts=(0, 12, 30, 48, 90, 160),
        step_time_ms=(2.0, 2.4, 3.9, 4.0, 7.5, 12.0),
    )
    for bs in (3, 8, 16, 31):
        surv = np.cumprod(rng.uniform(0.5, 0.995, size=(bs, BLOCK)), axis=1)
        tiers = derive_verify_len_tiers(
            cost_table=table, num_requests=bs, block_size=BLOCK, max_tiers=99
        )

        def theta(length):
            return (bs + float(surv[:, :length].sum())) / table.step_time(bs * (length + 1), bs)

        best_any = max(theta(v) for v in range(1, BLOCK + 1))
        best_tier = max(theta(v) for v in tiers)
        assert best_tier == pytest.approx(best_any), f"bs={bs} tiers={tiers}"


def test_derived_tiers_respect_the_capture_budget():
    table = SpsCostTable(
        token_counts=(0, 5, 9, 14, 20, 27, 35), step_time_ms=(1.0, 2, 3, 4, 5, 6, 7)
    )
    tiers = derive_verify_len_tiers(cost_table=table, num_requests=2, block_size=BLOCK, max_tiers=3)
    assert len(tiers) <= 3
    # Endpoints are always kept: the floor is always runnable and the full block
    # is always a right edge (the last shelf is unbounded).
    assert tiers[0] == 1 and tiers[-1] == BLOCK


def test_derived_tiers_are_a_function_of_batch_size():
    """Tiers move with bs because total tokens = bs * (length + 1).

    With a riser at 40 tokens: at bs=8, length 3 costs 8*4=32 tokens (still on
    the shelf) while length 4 costs 8*5=40 (past it), so 3 is the right edge; at
    bs=32 even length 1 is already past the riser, so the shelf yields nothing
    and only the endpoints remain. A tier set derived once and reused across
    batch sizes would be wrong for one of them.
    """
    table = SpsCostTable(token_counts=(0, 40), step_time_ms=(1.0, 5.0))
    assert derive_verify_len_tiers(
        cost_table=table, num_requests=8, block_size=BLOCK, max_tiers=99
    ) == [1, 3, BLOCK]
    assert derive_verify_len_tiers(
        cost_table=table, num_requests=32, block_size=BLOCK, max_tiers=99
    ) == [1, BLOCK]


def test_flat_cost_table_derives_only_the_endpoints():
    """No shelves to find -> nothing to derive beyond floor and full block."""
    tiers = derive_verify_len_tiers(
        cost_table=SpsCostTable.flat(), num_requests=8, block_size=BLOCK, max_tiers=99
    )
    assert tiers == [1, BLOCK]


def test_discrete_argmax_falls_back_when_nothing_is_allowed():
    table = SpsCostTable.flat()
    surv = np.cumprod(np.full((4, BLOCK), 0.9), axis=1)
    assert (
        budget_argmax_over_uniform_lens(
            survival=surv, num_gen_requests=4, cost_table=table, allowed_lens=[]
        )
        == 1
    )


# --------------------------------------------------------------------------
# host-side planner: relay, fallbacks, cross-rank agreement
# --------------------------------------------------------------------------


def _planner(**kw):
    from tensorrt_llm._torch.speculative.dspark_verify import DSparkVerifyPlanner

    kw.setdefault("cfg", _cfg(min_verify_len=1))
    kw.setdefault(
        "cost_table",
        SpsCostTable(token_counts=(0, 8, 24, 48), step_time_ms=(2.0, 2.1, 4.0, 9.0)),
    )
    kw.setdefault("tiers", [1, 3, BLOCK])
    return DSparkVerifyPlanner(**kw)


def test_planner_returns_max_tier_without_a_snapshot():
    """No confidence yet must mean verify-all, never a stale guess."""
    p = _planner()
    assert p.decide_draft_len(num_gen_requests=8) == BLOCK
    assert p.stats["fallback_no_snapshot"] == 1


def test_planner_refuses_to_trim_on_an_unprofiled_cost_model():
    """A flat cost table makes every token look free -- trimming would be blind."""
    p = _planner(cost_table=SpsCostTable.flat())
    assert p.decide_draft_len(num_gen_requests=8) == BLOCK
    assert p.stats["fallback_flat_cost"] == 1
    p2 = _planner(cost_table=None)
    assert p2.decide_draft_len(num_gen_requests=8) == BLOCK


def test_planner_only_ever_returns_a_captured_tier():
    """Returning an uncaptured length silently drops the step out of the graph."""
    torch.manual_seed(5)
    tiers = [2, 4]
    for trial in range(12):
        p = _planner(tiers=tiers)
        conf = torch.rand(6, BLOCK)
        p._host_buffer = conf.float()
        p._copy_event = None
        p._snapshot_valid = True
        assert p.decide_draft_len(num_gen_requests=6) in tiers, f"trial {trial}"


def test_planner_trims_when_confidence_is_low():
    hi, lo = torch.full((8, BLOCK), 6.0), torch.full((8, BLOCK), -6.0)
    out = []
    for logits in (hi, lo):
        p = _planner()
        p._host_buffer, p._copy_event, p._snapshot_valid = logits, None, True
        out.append(p.decide_draft_len(num_gen_requests=8))
    assert out[1] <= out[0], f"low confidence should not verify more: {out}"


def test_planner_cross_rank_reduction_forces_agreement():
    """Ranks must converge on one length even with different local confidence."""
    peer_choices = {}

    def make(rank, logits):
        p = _planner(all_rank_max=lambda v, r=rank: max(v, peer_choices.get(1 - r, v)))
        p._host_buffer, p._copy_event, p._snapshot_valid = logits, None, True
        return p

    r0 = _planner()
    r0._host_buffer, r0._copy_event, r0._snapshot_valid = torch.full((4, BLOCK), -6.0), None, True
    peer_choices[0] = r0._decide_local(num_gen_requests=4)
    r1 = _planner()
    r1._host_buffer, r1._copy_event, r1._snapshot_valid = torch.full((4, BLOCK), 6.0), None, True
    peer_choices[1] = r1._decide_local(num_gen_requests=4)

    a = make(0, torch.full((4, BLOCK), -6.0)).decide_draft_len(num_gen_requests=4)
    b = make(1, torch.full((4, BLOCK), 6.0)).decide_draft_len(num_gen_requests=4)
    assert a == b, f"ranks disagreed on the graph key: {a} vs {b}"
    assert a == max(peer_choices.values())


def test_planner_cross_rank_result_is_snapped_back_onto_a_tier():
    """A reduction can land off-ladder; the result must still be captured."""
    p = _planner(tiers=[1, 3, BLOCK], all_rank_max=lambda v: 5)  # 5 is not a tier
    p._host_buffer, p._copy_event, p._snapshot_valid = torch.full((4, BLOCK), 0.0), None, True
    assert p.decide_draft_len(num_gen_requests=4) == BLOCK


def test_planner_empty_batch_is_max_tier():
    assert _planner().decide_draft_len(num_gen_requests=0) == BLOCK


def test_planner_refuses_a_snapshot_that_does_not_cover_the_batch():
    """A short snapshot must fall back, never return a short answer.

    The staged snapshot lags one iteration, so a batch that grew since then has
    more requests than the snapshot has rows. Returning one length per staged
    row leaves the tail of the batch without a verify window -- and the two
    consumers disagree about what that means: the input layout is built per
    request (so it goes ragged) while the spec metadata sees a missing window
    and stays uniform. That combination misattributes one request's drafts to
    another with nothing raising.
    """
    p = _planner()
    p._host_buffer, p._copy_event, p._snapshot_valid = torch.zeros(4, BLOCK), None, True
    assert p.decide_verify_lens(num_gen_requests=9) is None
    assert p.stats["fallback_short_snapshot"] == 1
    assert p.decide_draft_len(num_gen_requests=9) == BLOCK


def test_planner_returns_one_verify_len_per_request():
    p = _planner()
    p._host_buffer, p._copy_event, p._snapshot_valid = torch.rand(6, BLOCK), None, True
    lens = p.decide_verify_lens(num_gen_requests=6)
    assert lens is not None and len(lens) == 6


def test_planner_reads_confidence_by_row_not_by_batch_position():
    """The snapshot is slot-indexed; ``rows`` is what re-associates it.

    The buffer is written by slot and read one iteration later, by which point
    joins and departures have reshuffled the batch. Position 0 of this step's
    batch is routinely a different request than the one scored in row 0.
    """
    p = _planner()
    # Row 3 is a confident request, row 0 a hopeless one.
    buf = torch.full((5, BLOCK), -8.0)
    buf[3] = 8.0
    p._host_buffer, p._copy_event, p._snapshot_valid = buf, None, True

    # This step schedules the confident request first, then the hopeless one.
    lens = p.decide_verify_lens(num_gen_requests=2, rows=[3, 0])
    assert lens is not None and lens[0] > lens[1], f"row mapping ignored: {lens}"


def test_planner_rejects_a_row_list_that_does_not_match_the_batch():
    p = _planner()
    p._host_buffer, p._copy_event, p._snapshot_valid = torch.rand(8, BLOCK), None, True
    assert p.decide_verify_lens(num_gen_requests=4, rows=[0, 1]) is None
    assert p.stats["fallback_short_snapshot"] == 1


def test_planner_uses_the_supplied_calibration():
    """STS must be applied before the cumprod, not skipped."""
    seen = {}

    def calib(x):
        seen["called"] = True
        return torch.sigmoid(x / 4.0)

    p = _planner(apply_calibration=calib)
    p._host_buffer, p._copy_event, p._snapshot_valid = torch.full((4, BLOCK), 2.0), None, True
    p.decide_draft_len(num_gen_requests=4)
    assert seen.get("called")


# --------------------------------------------------------------------------
# Acceptance criterion (2): CUDA-graph compatibility
#
# The failure mode this guards is silent, not loud: a runtime draft length with
# no captured graph does not raise -- maybe_get_cuda_graph returns None and the
# step runs eager, costing far more than the trimmed tokens save. So the capture
# set and the planner's output set must be provably identical.
# --------------------------------------------------------------------------


def _capture_set(batch_sizes, tiers, max_draft_len=BLOCK):
    """Invoke PyTorchModelEngine._get_graphs_to_capture with a stub engine."""
    import types

    from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine

    spec_config = types.SimpleNamespace(
        enable_confidence_scheduling=True,
        verify_len_tiers=list(tiers),
        draft_len_schedule=None,
        max_concurrency=None,
        is_linear_tree=True,
        spec_dec_mode=types.SimpleNamespace(
            support_dynamic_draft_len=lambda: False, use_one_engine=lambda: False
        ),
    )
    engine = types.SimpleNamespace(
        is_draft_model=False,
        spec_config=spec_config,
        max_draft_len=max_draft_len,
        max_total_draft_tokens=max_draft_len,
        original_max_draft_len=max_draft_len,
        _dynamic_draft_len_mapping=None,
        _cuda_graph_batch_sizes=list(batch_sizes),
    )
    return PyTorchModelEngine._get_graphs_to_capture(engine, list(batch_sizes), None)


def test_capture_is_the_batch_size_x_tier_cross_product():
    """K is NOT a function of batch size here -- every bs needs every tier."""
    bss, tiers = [1, 8, 64], [1, 3, BLOCK]
    graphs = _capture_set(bss, tiers)
    assert set(graphs) == {(bs, k) for bs in bss for k in tiers}
    assert len(graphs) == len(bss) * len(tiers)


def test_every_length_the_planner_can_pick_has_a_captured_graph():
    """The anti-silent-eager invariant, checked against the planner itself."""
    from tensorrt_llm._torch.speculative.dspark_verify import DSparkVerifyPlanner

    bss, tiers = [1, 16, 64], [1, 3, BLOCK]
    captured = {k for _bs, k in _capture_set(bss, tiers)}

    torch.manual_seed(11)
    table = SpsCostTable(token_counts=(0, 8, 24, 64), step_time_ms=(2.0, 2.2, 4.0, 9.0))
    for trial in range(40):
        p = DSparkVerifyPlanner(cfg=_cfg(min_verify_len=1), cost_table=table, tiers=tiers)
        p._host_buffer = torch.rand(8, BLOCK) * 12 - 6
        p._copy_event, p._snapshot_valid = None, True
        k = p.decide_draft_len(num_gen_requests=8)
        assert k in captured, f"trial {trial}: planner picked K={k}, captured={captured}"


def test_planner_fallbacks_also_land_on_a_captured_graph():
    """Every degraded path (no snapshot, flat cost, empty batch) must stay captured."""
    from tensorrt_llm._torch.speculative.dspark_verify import DSparkVerifyPlanner

    tiers = [1, 3, BLOCK]
    captured = {k for _bs, k in _capture_set([1, 64], tiers)}
    cases = [
        (
            "no snapshot",
            DSparkVerifyPlanner(
                cfg=_cfg(),
                cost_table=SpsCostTable(token_counts=(0, 8), step_time_ms=(2.0, 5.0)),
                tiers=tiers,
            ),
            8,
        ),
        (
            "flat cost",
            DSparkVerifyPlanner(cfg=_cfg(), cost_table=SpsCostTable.flat(), tiers=tiers),
            8,
        ),
        (
            "empty batch",
            DSparkVerifyPlanner(
                cfg=_cfg(),
                cost_table=SpsCostTable(token_counts=(0, 8), step_time_ms=(2.0, 5.0)),
                tiers=tiers,
            ),
            0,
        ),
    ]
    for name, p, n in cases:
        assert p.decide_draft_len(num_gen_requests=n) in captured, name


# --------------------------------------------------------------------------
# Acceptance criterion (3): attention-DP + TP
#
# draft_len is part of the CUDA-graph key but is NOT covered by the ADP
# consistency allgather, so ranks that choose differently select different
# graphs -- one replays, one falls back to eager -- and their collectives
# diverge. Agreement has to be forced before the forward.
# --------------------------------------------------------------------------


def _simulated_ranks(local_confidences, tiers, table, batch_sizes=None):
    """Run the planner on each rank with a real max-reduction across them."""
    from tensorrt_llm._torch.speculative.dspark_verify import DSparkVerifyPlanner

    batch_sizes = batch_sizes or [c.shape[0] for c in local_confidences]
    planners = []
    for conf in local_confidences:
        p = DSparkVerifyPlanner(cfg=_cfg(min_verify_len=1), cost_table=table, tiers=tiers)
        p._host_buffer, p._copy_event, p._snapshot_valid = conf, None, True
        planners.append(p)
    locals_ = [p._decide_local(num_gen_requests=bs) for p, bs in zip(planners, batch_sizes)]
    peak = max(locals_)
    return [
        p.decide_draft_len(num_gen_requests=bs, all_rank_max=lambda _v, m=peak: m)
        for p, bs in zip(planners, batch_sizes)
    ], locals_


def test_adp_ranks_agree_despite_different_confidence():
    tiers = [1, 3, BLOCK]
    table = SpsCostTable(token_counts=(0, 16, 48, 128), step_time_ms=(2.0, 2.2, 4.0, 9.0))
    confs = [
        torch.full((8, BLOCK), -6.0),
        torch.full((8, BLOCK), 6.0),
        torch.full((8, BLOCK), 0.0),
        torch.full((8, BLOCK), 3.0),
    ]
    chosen, locals_ = _simulated_ranks(confs, tiers, table)
    assert len(set(chosen)) == 1, f"ranks disagreed: {chosen} (locals {locals_})"
    assert chosen[0] in tiers


def test_adp_ranks_agree_despite_different_batch_sizes():
    """Under ADP each rank has its own batch; agreement must survive that too."""
    tiers = [1, 3, BLOCK]
    table = SpsCostTable(token_counts=(0, 16, 48, 128), step_time_ms=(2.0, 2.2, 4.0, 9.0))
    torch.manual_seed(3)
    confs = [torch.rand(b, BLOCK) * 8 - 4 for b in (2, 9, 17, 33)]
    chosen, locals_ = _simulated_ranks(confs, tiers, table, batch_sizes=[2, 9, 17, 33])
    assert len(set(chosen)) == 1, f"ranks disagreed: {chosen} (locals {locals_})"


def test_adp_rank_with_empty_batch_does_not_break_agreement():
    """An idle DP rank still has to enter the same graph as everyone else."""
    tiers = [1, 3, BLOCK]
    table = SpsCostTable(token_counts=(0, 16, 48, 128), step_time_ms=(2.0, 2.2, 4.0, 9.0))
    confs = [torch.zeros(0, BLOCK), torch.full((8, BLOCK), 6.0), torch.full((8, BLOCK), -6.0)]
    chosen, _ = _simulated_ranks(confs, tiers, table, batch_sizes=[0, 8, 8])
    assert len(set(chosen)) == 1, f"idle rank diverged: {chosen}"


def test_cross_rank_reduction_is_max_not_min():
    """Max is the safe direction: a rank that wanted less simply verifies more.

    Taking the min would let one pessimistic rank starve the whole batch, and --
    worse -- could drop below another rank's min_verify_len floor.
    """
    tiers = [1, 3, BLOCK]
    table = SpsCostTable(token_counts=(0, 16, 48, 128), step_time_ms=(2.0, 2.2, 4.0, 9.0))
    confs = [torch.full((8, BLOCK), -6.0), torch.full((8, BLOCK), 6.0)]
    chosen, locals_ = _simulated_ranks(confs, tiers, table)
    assert chosen[0] == max(locals_)
