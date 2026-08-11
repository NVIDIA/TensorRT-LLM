# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decision-level parity with the SGLang budget planner this was ported from.

A throughput gain alone does not validate the port: the wrong economics can
still move the needle in the right direction. This pins the DECISION function
against a line-for-line reference copy of SGLang's
``compute_verify_token_budget`` + ``_additive_step_time_tensor``
(python/sglang/srt/speculative/dspark_components/dspark_planner.py), so a
divergence in either port shows up as an argmax mismatch here rather than as
an unexplained fleet-level regression.

The two planners parameterize the same optimization differently:

- SGLang admits EVERY position (including position 0) as a budget candidate
  and its floor is one bonus token per request: total M = bs + n.
- TensorRT-LLM grants positions below ``min_verify_len`` for free (their
  survival rides in the tau base) and only positions at or beyond it are
  schedulable: total M = bs*(min_verify_len+1) + m.

For ``min_verify_len=1`` these describe the same curve shifted by ``bs``:
survival is non-increasing along a row, so the top-(bs+m) SGLang candidates
are exactly the bs position-0 entries plus the top-m deeper entries, giving
``tau_sgl(bs+m) == tau_trt(m)`` and ``M_sgl(bs+m) == M_trt(m)``. The argmaxes
must therefore agree up to that shift whenever SGLang admits at least the
full position-0 set. Known, deliberate divergences are pinned separately:
SGLang filters candidates by ``survival_eps`` inside the budget function
(TensorRT-LLM applies eps at allocation time instead), and SGLang can choose
sub-floor budgets (windows of zero draft positions) that TensorRT-LLM's
``min_verify_len=1`` floor excludes.
"""

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.speculative.dspark_planner import (
    SpsCostTable, compute_verify_token_budget)

# --------------------------------------------------------------------------
# Reference implementation, copied from SGLang dspark_components/dspark_planner
# (additive-table path) with only cosmetic renames. Do not "fix" this copy:
# its job is to be what SGLang runs.
# --------------------------------------------------------------------------


def _sgl_interp_clamped(xs, ys, x: float) -> float:
    xs = torch.tensor(xs, dtype=torch.float64)
    ys = torch.tensor(ys, dtype=torch.float64)
    x_t = torch.tensor(float(x), dtype=torch.float64).clamp_(xs[0], xs[-1])
    hi = torch.bucketize(x_t, xs, right=True).clamp_(1, xs.numel() - 1)
    lo = hi - 1
    span = (xs[hi] - xs[lo]).clamp_(min=1e-9)
    frac = (x_t - xs[lo]) / span
    return float(ys[lo] + frac * (ys[hi] - ys[lo]))


def _sgl_additive_step_time(table, num_requests: int, num_budgets: int):
    floor = table["bias_seconds"] + _sgl_interp_clamped(
        table["bs_probes"], table["alpha_seconds"], float(num_requests))
    m_probes = torch.tensor(table["m_probes"], dtype=torch.float64)
    theta_vals = torch.tensor(table["theta_seconds"], dtype=torch.float64)
    m = (num_requests + torch.arange(num_budgets, dtype=torch.float64)).clamp_(
        min=float(table["m_probes"][0]), max=float(table["m_probes"][-1]))
    hi = torch.bucketize(m, m_probes, right=True).clamp_(1, m_probes.numel() - 1)
    lo = hi - 1
    span = (m_probes[hi] - m_probes[lo]).clamp_(min=1e-9)
    frac = (m - m_probes[lo]) / span
    theta_at_m = theta_vals[lo] + frac * (theta_vals[hi] - theta_vals[lo])
    return floor + theta_at_m


def _sgl_compute_verify_token_budget(*, history_survival_probs, table,
                                     max_verify_len, survival_eps):
    num_requests = history_survival_probs.shape[0]
    candidates = history_survival_probs[:, :max_verify_len].flatten()
    candidates = candidates[candidates >= survival_eps].to(torch.float64)
    candidates_sorted = torch.sort(candidates, descending=True).values
    prefix_sum = torch.cumsum(candidates_sorted, dim=0)
    tau_star = num_requests + torch.cat(
        [torch.zeros(1, dtype=torch.float64), prefix_sum])
    step_time = _sgl_additive_step_time(table, int(num_requests),
                                        int(tau_star.numel()))
    theta = tau_star / step_time
    return int(torch.argmax(theta))


# --------------------------------------------------------------------------
# One certified-table shape shared by both sides. TensorRT-LLM stores ms,
# SGLang stores seconds; the argmax is scale-invariant but the tables are
# built with the honest factor anyway.
# --------------------------------------------------------------------------

M_PROBES = (64, 96, 128, 192, 384, 512, 768, 1536)
THETA_MS = (9.68, 11.08, 11.95, 13.95, 22.09, 23.85, 35.61, 105.64)
BIAS_MS = 25.244443
BS_PROBES = (32, 64, 128, 256)
ALPHA_MS = (0.0, 2.177477, 11.462212, 19.342661)
BLOCK = 5

TRT_TABLE = SpsCostTable(token_counts=M_PROBES, step_time_ms=THETA_MS,
                         fixed_overhead_ms=BIAS_MS, batch_sizes=BS_PROBES,
                         batch_overhead_ms=ALPHA_MS)
SGL_TABLE = {
    "m_probes": M_PROBES,
    "theta_seconds": tuple(v / 1e3 for v in THETA_MS),
    "bias_seconds": BIAS_MS / 1e3,
    "bs_probes": BS_PROBES,
    "alpha_seconds": tuple(v / 1e3 for v in ALPHA_MS),
}


def _inversion_free_survival(rng, bs):
    """Survivals where every position-0 beats every deeper candidate.

    The exact-parity theorem needs this: SGLang admits position 0 as a budget
    candidate while TensorRT-LLM grants it in the floor, so the two describe
    the same curve (shifted by ``bs``) only when the top-(bs+m) SGLang set is
    exactly "all of position 0, plus the top-m deeper entries". A weak row's
    position-0 falling below a strong row's position-1 breaks that -- see
    test_starving_weak_rows_is_a_known_gap for what SGLang does with it.
    """
    conf0 = rng.uniform(0.85, 0.99, size=(bs, 1))
    deeper = rng.uniform(0.25, 0.80, size=(bs, BLOCK - 1))
    return np.cumprod(np.concatenate([conf0, deeper], axis=1), axis=1)


@pytest.mark.parametrize("bs", [1, 3, 8, 64, 128, 252, 256])
def test_budget_argmax_matches_sglang_up_to_the_floor_shift(bs):
    """Same survival, same table, eps off, no inversions: shift-exact parity."""
    rng = np.random.default_rng(17 + bs)
    for _ in range(5):
        surv = _inversion_free_survival(rng, bs)
        sgl_n = _sgl_compute_verify_token_budget(
            history_survival_probs=torch.tensor(surv, dtype=torch.float32),
            table=SGL_TABLE, max_verify_len=BLOCK, survival_eps=0.0)
        trt_m = compute_verify_token_budget(
            survival=surv, num_gen_requests=bs, cost_table=TRT_TABLE,
            min_verify_len=1)
        if sgl_n >= bs:
            assert trt_m == sgl_n - bs, (
                f"bs={bs}: SGLang admits {sgl_n} candidates "
                f"(= floor {bs} + {sgl_n - bs}), TRT-LLM budget {trt_m}")
        else:
            # SGLang went below the one-draft floor; the closest budget
            # TensorRT-LLM can express under min_verify_len=1 is zero.
            assert trt_m == 0


def test_live_pooled_survivals_park_on_the_breakpoint_together():
    """The campaign's operating point, both consumers, both argmax modes.

    With interpolated segments the continuous optimum parks where the theta
    slope jumps: buying position-2 survivals (0.502/token) beats the
    512..768 slope (0.046 ms/token) but not the 768..1536 one (0.091), so
    both planners stop at exactly M = 768 -- TensorRT-LLM at budget
    768 - 2*252 = 264, SGLang at 768 - 252 = 516. Tier alignment then rounds
    down to the rung-2 budget the executor can capture.
    """
    bs = 252
    surv = np.tile([0.771, 0.693, 0.502, 0.360, 0.315], (bs, 1))
    sgl_n = _sgl_compute_verify_token_budget(
        history_survival_probs=torch.tensor(surv, dtype=torch.float32),
        table=SGL_TABLE, max_verify_len=BLOCK, survival_eps=0.0)
    trt_m = compute_verify_token_budget(
        survival=surv, num_gen_requests=bs, cost_table=TRT_TABLE,
        min_verify_len=1)
    assert trt_m == 768 - 2 * bs
    assert sgl_n - bs == trt_m
    tiered = compute_verify_token_budget(
        survival=surv, num_gen_requests=bs, cost_table=TRT_TABLE,
        min_verify_len=1, allowed_lens=[1, 2, 5])
    assert tiered == bs * 1  # rung-2: one scheduled position past the floor


def test_starving_weak_rows_is_a_known_gap():
    """SGLang can verify NOTHING for a hopeless request; TRT-LLM cannot.

    SGLang's budget admits position-0 candidates individually, so a row whose
    draft is dead on arrival can be starved to a zero-length window and its
    would-be tokens spent on a strong row's deep positions. TensorRT-LLM's
    ``min_verify_len=1`` floor buys position 0 for every request
    unconditionally. On a batch that mixes both kinds, SGLang's achievable
    Theta is therefore strictly better -- pinned here so a future
    min_verify_len=0 port is a deliberate decision with a test to flip.
    """
    # Half the batch is strong all the way down, half is dead on arrival.
    # The batch must be big enough that its token range sits on a genuinely
    # rising part of the theta curve: below the first breakpoint the clamp
    # makes cost flat, every candidate is free, and both planners trivially
    # buy everything at the same theta -- the two-row version of this test
    # sat in that regime and proved nothing.
    bs = 64
    strong = np.tile([0.99, 0.98, 0.97, 0.96, 0.95], (bs // 2, 1))
    dead = np.tile([0.01, 0.008, 0.006, 0.004, 0.002], (bs // 2, 1))
    surv = np.concatenate([strong, dead], axis=0)
    sgl_n = _sgl_compute_verify_token_budget(
        history_survival_probs=torch.tensor(surv, dtype=torch.float32),
        table=SGL_TABLE, max_verify_len=BLOCK, survival_eps=0.0)
    trt_m = compute_verify_token_budget(
        survival=surv, num_gen_requests=bs, cost_table=TRT_TABLE,
        min_verify_len=1)

    def theta_at(tau, tokens):
        return tau / float(TRT_TABLE.step_times(np.asarray([tokens]), bs)[0])

    cand = np.sort(surv.reshape(-1))[::-1]
    sgl_theta = theta_at(bs + cand[:sgl_n].sum(), bs + sgl_n)
    deeper = np.sort(surv[:, 1:].reshape(-1))[::-1]
    trt_theta = theta_at(bs + surv[:, 0].sum() + deeper[:trt_m].sum(),
                         2 * bs + trt_m)
    assert sgl_theta > trt_theta


def test_eps_filtering_is_a_known_divergence():
    """SGLang drops sub-eps candidates inside the budget; TRT-LLM does not.

    TensorRT-LLM applies survival_eps at allocation time
    (schedule_verify_lens_topk's eligibility mask), so its BUDGET can count a
    candidate the allocator then refuses. This pins the divergence so a
    future unification is a deliberate change, not an accident: with a heavy
    sub-eps tail the SGLang budget must not exceed TRT-LLM's shifted one.
    """
    bs = 16
    surv = np.tile([0.9, 0.4, 0.008, 0.004, 0.002], (bs, 1))
    sgl_n = _sgl_compute_verify_token_budget(
        history_survival_probs=torch.tensor(surv, dtype=torch.float32),
        table=SGL_TABLE, max_verify_len=BLOCK, survival_eps=0.01)
    trt_m = compute_verify_token_budget(
        survival=surv, num_gen_requests=bs, cost_table=TRT_TABLE,
        min_verify_len=1)
    assert sgl_n - bs <= trt_m
