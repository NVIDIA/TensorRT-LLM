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


def test_flat_cost_table_degenerates_to_verify_all():
    """A flat cost model makes every token free -> spend the whole budget."""
    table = SpsCostTable.flat(2.0)
    assert table.is_flat
    surv = np.cumprod(np.full((6, BLOCK), 0.9), axis=1)
    budget = compute_verify_token_budget(survival=surv, num_gen_requests=6, cost_table=table)
    assert budget == 6 * (BLOCK - 1)
    # An empty batch has no candidates to buy, whatever the table says.
    assert (
        compute_verify_token_budget(
            survival=np.zeros((0, BLOCK)), num_gen_requests=0, cost_table=table
        )
        == 0
    )


def test_cost_table_interpolates_between_breakpoints():
    """The consumer contract is clamped linear interpolation, not a floor.

    The floor variant this replaces billed every total below the next
    breakpoint at the previous breakpoint's price. On the sparse live table
    (points at 768 and 1536) that priced a 1512-token full block at the 768
    price -- the planner's cost ratio collapsed below the survival ratio and
    it bought the full block on ~95% of decisions. Flatness between points is
    a claim about the hardware; if a table wants a shelf, it must MEASURE the
    shelf (two breakpoints with equal values), not have the consumer assume it.
    """
    table = SpsCostTable(token_counts=(0, 100), step_time_ms=(1.0, 5.0))
    assert table.step_time(99) > table.step_time(1)
    assert table.step_time(50) == pytest.approx(3.0)
    # A measured shelf stays flat under interpolation.
    shelf = SpsCostTable(token_counts=(0, 99, 100), step_time_ms=(1.0, 1.0, 5.0))
    assert shelf.step_time(1) == shelf.step_time(99)
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
    # Zero budget degenerates to the floor for everyone.
    cfg_floor = _cfg(min_verify_len=2)
    lens = schedule_verify_lens_topk(
        survival=compute_survival(torch.rand(5, BLOCK)), budget=0, cfg=cfg_floor)
    assert torch.equal(lens, torch.full((5,), 2, dtype=torch.int32))
    # A huge budget saturates at the cap.
    cfg_cap = _cfg(min_verify_len=1, max_verify_len=4)
    lens = schedule_verify_lens_topk(
        survival=compute_survival(torch.full((5, BLOCK), 0.99)), budget=10**6, cfg=cfg_cap)
    assert torch.equal(lens, torch.full((5,), 4, dtype=torch.int32))
    # An empty batch allocates nothing.
    assert schedule_verify_lens_topk(
        survival=torch.zeros(0, BLOCK), budget=5, cfg=_cfg()).numel() == 0


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


def test_schedule_config_rejects_bad_bounds():
    with pytest.raises(ValueError, match="min_verify_len must be >= 1"):
        DSparkScheduleConfig(block_size=BLOCK, min_verify_len=0)
    with pytest.raises(ValueError, match="block_size must be >= 1"):
        DSparkScheduleConfig(block_size=0)
    with pytest.raises(ValueError, match="< min_verify_len"):
        DSparkScheduleConfig(block_size=BLOCK, min_verify_len=5, max_verify_len=3)
    with pytest.raises(ValueError, match=r"\[bs, K\]"):
        compute_survival(torch.rand(BLOCK))


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
    # With nothing runnable, fall back to the floor.
    assert (
        budget_argmax_over_uniform_lens(
            survival=surv, num_gen_requests=8, cost_table=table, allowed_lens=[]
        )
        == 1
    )


def test_derived_tiers_lose_nothing_versus_continuous_k():
    """The shelf-right-edge property: derived tiers contain the true optimum.

    Within a cost shelf the step time is constant while tau strictly increases,
    so Theta rises monotonically across the shelf and the optimum can only sit
    at a right edge. A tier set built from the right edges must therefore match
    an exhaustive search over *every* length.
    """
    rng = np.random.default_rng(11)
    # The shelves are ENCODED, breakpoint pairs with equal values: under the
    # interpolating consumer, flatness between points is only real when the
    # table measured it. (Assuming it -- the old floor lookup -- is what
    # priced a 1512-token step at the 768-token price on the live table and
    # disabled trimming outright; the theorem below is conditional on genuine
    # shelves and this encoding is what "genuine" means now.)
    table = SpsCostTable(
        token_counts=(0, 11, 12, 29, 30, 47, 48, 89, 90, 159, 160),
        step_time_ms=(2.0, 2.0, 2.4, 2.4, 3.9, 3.9, 4.0, 4.0, 7.5, 7.5, 12.0),
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


def test_capture_pins_the_draft_length_to_the_top_tier():
    """One graph per batch size, not a cross product with the tier ladder.

    The cross product was right while the uniform tier path existed and chose
    a different draft_len per step. That path is gone and nothing varies
    draft_len at runtime now: cap-accept pins it to planner.max_tier and
    compact moves the ladder to the token axis. Capturing the lower tiers
    anyway spent KV cache -- captured graphs come out of the same pool -- on
    graphs that could never be selected. Measured on a real run: 57 graphs
    where 19 were reachable.
    """
    bss, tiers = [1, 8, 64], [1, 3, BLOCK]
    graphs = _capture_set(bss, tiers)
    assert set(graphs) == {(bs, BLOCK) for bs in bss}
    assert len(graphs) == len(bss), (
        f"expected one graph per batch size, got {len(graphs)}: {graphs}")


def _stats(max_draft_len=5):
    from tensorrt_llm._torch.speculative.dspark_observability import (
        DSparkRaggedStats, RaggedVerifyMode)
    return DSparkRaggedStats(mode=RaggedVerifyMode.COMPACT,
                             max_draft_len=max_draft_len)


def test_trim_regret_counts_only_drafts_alive_at_the_cut():
    """A trimmed request that accepted its whole window lost something.

    This is the quantity a delivered-only acceptance metric cannot recover: it
    separates "trimming was free, those drafts would have died anyway" from
    "trimming bought throughput by discarding acceptance".
    """
    stats = _stats()
    # Given the full block and died early: not trimmed, no regret.
    stats.record_acceptance(accepted=2, window=5)
    # Trimmed to 3 and died at 1: the cut cost nothing.
    stats.record_acceptance(accepted=1, window=3)
    # Trimmed to 3 and accepted all 3: still alive at the cut -> regret.
    stats.record_acceptance(accepted=3, window=3)

    assert stats.requests_scored == 3
    assert stats.requests_trimmed == 2
    assert stats.trimmed_hit_ceiling == 1
    assert stats.trim_regret_rate == 0.5
    assert stats.accept_len == pytest.approx(2.0)
    summary = stats.summary()
    for key in ("accept_len", "requests_scored", "requests_trimmed",
                "trim_regret_rate"):
        assert key in summary, f"{key} missing from the summary"

    # Nothing trimmed: the regret rate is 0/0-safe and stays 0.
    untrimmed = _stats()
    for accepted in (0, 3, 5):
        untrimmed.record_acceptance(accepted=accepted, window=5)
    assert untrimmed.requests_trimmed == 0
    assert untrimmed.trim_regret_rate == 0.0
    assert untrimmed.summary()["accept_len"] == pytest.approx(8 / 3, abs=1e-4)
