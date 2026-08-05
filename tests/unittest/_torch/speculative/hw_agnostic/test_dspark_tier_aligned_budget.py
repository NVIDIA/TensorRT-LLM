# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The budget must be scored with the tau the step actually collects.

``decide_verify_lens`` picks a rung, converts it to ``B = n*(L - min)`` tokens,
and hands B to ``schedule_verify_lens_topk``, which spends it on the B largest
survivals in the batch. But the rung was chosen by
``budget_argmax_over_uniform_lens``, which scores rung L as if every request took
columns ``[min, L)``. Top-k is by construction the largest sum over all size-B
subsets, so the realised tau is >= the scored one at an identical token cost --
and the gap is not the same at every rung, so the argmax can land on the wrong
one.

The fix is not to let the planner choose an arbitrary budget: the executor can
only run ``bs*(t+1)`` tokens for a captured tier, and any other total rounds UP,
which would push a budget deliberately chosen below a cost riser back over it.
So the ladder stays the grid and only the numerator changes.
"""

import numpy as np
import pytest

from tensorrt_llm._torch.speculative.dspark_planner import (
    SpsCostTable, budget_argmax_over_uniform_lens, compute_verify_token_budget)


def _table() -> SpsCostTable:
    """A staircase with genuine risers, so trimming is worth something.

    Breakpoints are spaced so that neither tier is trivially optimal: a flat
    table degenerates to verify-all and would make every assertion here vacuous.
    """
    token_counts = tuple(range(0, 400, 16))
    step_time_ms = tuple(4.0 + 0.6 * (tok // 96) + 0.004 * tok
                         for tok in token_counts)
    return SpsCostTable(token_counts=token_counts, step_time_ms=step_time_ms,
                        fixed_overhead_ms=1.0)


@pytest.mark.parametrize("tiers", [[1, 2, 5], [1, 3, 5], [1, 5]])
def test_restricted_answer_is_always_realisable(tiers):
    """Every returned budget corresponds to a rung the executor captured."""
    rng = np.random.default_rng(20260803)
    table = _table()
    for _ in range(200):
        bs = int(rng.integers(2, 17))
        survival = np.sort(rng.random((bs, 5)), axis=1)[:, ::-1]
        n = compute_verify_token_budget(
            survival=survival, num_gen_requests=bs, cost_table=table,
            min_verify_len=1, allowed_lens=tiers)
        assert n % bs == 0, (
            f"budget {n} for {bs} requests is not n*(t-min) for any tier")
        assert (n // bs) + 1 in tiers, (
            f"budget {n} implies tier {(n // bs) + 1}, not in {tiers}")


def test_agrees_with_the_uniform_scorer_at_a_degenerate_batch():
    """With one row the two taus coincide, so the rungs must match.

    A single request makes the uniform allocation and the top-k allocation the
    same set, which is the one case where any disagreement would be a plain bug
    rather than the intended improvement.
    """
    table = _table()
    rng = np.random.default_rng(7)
    tiers = [1, 2, 5]
    for _ in range(100):
        survival = np.sort(rng.random((1, 5)), axis=1)[:, ::-1]
        n = compute_verify_token_budget(
            survival=survival, num_gen_requests=1, cost_table=table,
            min_verify_len=1, allowed_lens=tiers)
        uniform = budget_argmax_over_uniform_lens(
            survival=survival, num_gen_requests=1, cost_table=table,
            allowed_lens=tiers, min_verify_len=1)
        assert (n // 1) + 1 == uniform


def test_restricted_never_scores_worse_than_the_uniform_choice():
    """Same grid, better numerator: the chosen rung's realised theta wins.

    Both routes evaluate the same discrete cost points, so the restricted form
    cannot be worse under the tau that the step actually collects. This is the
    property that makes the swap safe -- if it ever failed, the change would be
    trading a modelling error for a real one.
    """
    rng = np.random.default_rng(31337)
    table = _table()
    tiers = [1, 2, 5]
    for _ in range(300):
        bs = int(rng.integers(2, 17))
        survival = np.sort(rng.random((bs, 5)), axis=1)[:, ::-1]

        def realised_theta(tier: int) -> float:
            budget = bs * (tier - 1)
            cand = np.sort(survival[:, 1:].reshape(-1))[::-1]
            tau = float(bs) + float(survival[:, :1].sum()) + float(
                cand[:budget].sum())
            tokens = np.array([bs * (tier + 1)])
            return tau / float(table.step_times(tokens, bs)[0])

        n = compute_verify_token_budget(
            survival=survival, num_gen_requests=bs, cost_table=table,
            min_verify_len=1, allowed_lens=tiers)
        chosen = (n // bs) + 1
        uniform = budget_argmax_over_uniform_lens(
            survival=survival, num_gen_requests=bs, cost_table=table,
            allowed_lens=tiers, min_verify_len=1)
        assert realised_theta(chosen) >= realised_theta(uniform) - 1e-12, (
            f"restricted picked tier {chosen} whose realised theta is below "
            f"the uniform scorer's tier {uniform}")


def test_unrestricted_form_is_unchanged():
    """Omitting allowed_lens must still answer the unconstrained question.

    The existing tests use that form as the reference, and it is still the right
    thing to ask when sizing a ladder rather than executing one.
    """
    table = _table()
    rng = np.random.default_rng(99)
    survival = np.sort(rng.random((6, 5)), axis=1)[:, ::-1]
    free = compute_verify_token_budget(
        survival=survival, num_gen_requests=6, cost_table=table,
        min_verify_len=1)
    assert 0 <= free <= 6 * 4
    tied = compute_verify_token_budget(
        survival=survival, num_gen_requests=6, cost_table=table,
        min_verify_len=1, allowed_lens=[1, 2, 5])
    # The restricted answer is one of the grid points; the free one need not be.
    assert tied in {0, 6, 24}
