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
"""Block verification (Sun et al., arXiv:2403.10444, Algorithm 2): the float64 reference model.

No GPU needed. These tests establish what the fused kernel (``fast_block_verification``,
``test_fast_block_verification.py``) is held to:

  * Theorem 1 (exactness), by exact enumeration: over small random autoregressive target / draft
    models, the output of block verification followed by target completion has the target's joint
    distribution to rounding, for independent, near, identical, uniform, one-hot and truncated
    drafts. Wrong variants (a clipped weight left out, a token-wise residual, ...) are caught by the
    same enumeration, so the check has teeth.
  * The sampling code path (``block_verify`` with explicit uniforms) against the exact tau law and
    against the target by chi-square.
  * Theorem 2 (never worse): ``E[tau]`` of block verification is at least the token-wise one, in
    expectation over the draft block. It does NOT hold for every draft block (a regression below
    pins an explicit counterexample), so a kernel must not be tested for it per block.
  * Where the two verifiers must agree: ``K = 1`` (the same decisions on the same uniforms),
    one-hot drafts (the same tau law and residual), identical models (accept everything).
  * Edge cases: tau = 0, tau = K, no-draft padding rows, the cuRAND stream model (Philox known
    answers).
"""

import numpy as np
import pytest
from block_verify_ref import (
    TreeModel,
    block_accept_probs,
    block_expected_tau,
    block_residual_mass,
    block_tau_pmf,
    block_verify,
    block_weights,
    curand_uniform,
    exact_block_output_joint,
    expected_accepted_over_drafts,
    inv_cdf,
    philox4x32_10,
    residual_weights,
    target_joint,
    token_accept_probs,
    token_expected_tau,
    token_verify,
)

# (vocab, draft length, target seed, draft seed, spread of the log-probabilities)
CASES = [
    (3, 3, 1, 2, 1.0),
    (3, 2, 3, 4, 2.0),
    (4, 3, 5, 6, 1.0),
    (2, 5, 7, 8, 1.5),
    (3, 4, 9, 10, 0.7),
    (2, 6, 11, 12, 3.0),
]


def _truncate(table, keep=0.2):
    """Zero the probabilities below ``keep`` and renormalise (a draft of smaller support)."""
    t = np.where(table >= keep, table, 0.0)
    t[t.sum(-1) == 0.0] = table[t.sum(-1) == 0.0]
    return t / t.sum(-1, keepdims=True)


def _drafts(target, k, seed):
    """Draft models of every character for a given target."""
    v = target.vocab
    other = TreeModel(v, k, seed, 1.5)
    return {
        "independent": other,
        "near": TreeModel.mixture(target, other, 0.2),
        "identical": target,
        "uniform": target.map_tables(lambda t: np.full_like(t, 1.0 / v)),
        "one_hot": target.map_tables(lambda t: np.eye(v)[t.argmax(-1)]),
        "truncated": target.map_tables(_truncate),
    }


@pytest.mark.parametrize("vocab,k,seed_t,seed_d,spread", CASES)
@pytest.mark.parametrize(
    "kind", ["independent", "near", "identical", "uniform", "one_hot", "truncated"]
)
def test_theorem1_output_has_the_target_distribution(vocab, k, seed_t, seed_d, spread, kind):
    """Exact enumeration over every draft block: block verification + target completion = target."""
    target = TreeModel(vocab, k, seed_t, spread)
    draft = _drafts(target, k, seed_d)[kind]
    err = np.abs(exact_block_output_joint(target, draft, k) - target_joint(target, k)).max()
    assert err < 1e-11, err


def _no_clip(p, q, drafts):
    """The weights left unclipped at 1 (a plausible slip)."""
    b, k = drafts.shape
    idx = np.arange(b)
    w = np.ones((b, k + 1))
    for i in range(k):
        d = drafts[:, i]
        w[:, i + 1] = w[:, i] * p[idx, i, d] / q[idx, i, d]
    r = block_residual_mass(p, q, w)
    den = r + (1.0 - w)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.clip(np.where(den > 0.0, r / den, 1.0), 0.0, 1.0)
    h[:, -1] = np.clip(w[:, -1], 0.0, 1.0)
    h[:, 0] = 1.0
    return h, w


def _accept_every_full_block(p, q, drafts):
    """The full-block acceptance probability set to 1 instead of ``w_K``."""
    h, w = block_accept_probs(p, q, drafts)
    h = h.copy()
    h[:, -1] = 1.0
    return h, w


def _drop_the_weight_from_the_denominator(p, q, drafts):
    """``h_i = R_i / (R_i + 1)``: the ``- w_i`` of the denominator left out."""
    h, w = block_accept_probs(p, q, drafts)
    r = block_residual_mass(p, q, w)
    h = h.copy()
    h[:, 1:-1] = (r / (r + 1.0))[:, 1:-1]
    return h, w


def _token_wise_residual(p, q, w, tau):
    """The token-wise residual ``max(p - q, 0)`` (weight 1) with the block acceptance law."""
    return residual_weights(p, q, np.ones_like(w), tau)


WRONG_VARIANTS = {
    "no_clip": {"accept_probs": _no_clip},
    "accept_every_full_block": {"accept_probs": _accept_every_full_block},
    "no_weight_in_denominator": {"accept_probs": _drop_the_weight_from_the_denominator},
    "token_wise_residual": {"residual": _token_wise_residual},
}


@pytest.mark.parametrize("variant", sorted(WRONG_VARIANTS))
def test_wrong_variants_fail_the_exactness_enumeration(variant):
    """The enumeration of ``test_theorem1_*`` is not vacuous: each plausible slip is caught."""
    worst = 0.0
    for vocab, k, seed_t, seed_d, spread in CASES[:3]:
        target = TreeModel(vocab, k, seed_t, spread)
        draft = _drafts(target, k, seed_d)["independent"]
        got = exact_block_output_joint(target, draft, k, **WRONG_VARIANTS[variant])
        worst = max(worst, float(np.abs(got - target_joint(target, k)).max()))
    assert worst > 1e-3, worst


def _random_blocks(rng, n, k, vocab, spread, noise):
    """``n`` independent (target rows, proposals, drafts) triples; drafts drawn from ``q``."""
    zt = rng.normal(0.0, spread, (n, k + 1, vocab))
    p = np.exp(zt - zt.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    zq = zt[:, :k] + rng.normal(0.0, noise, (n, k, vocab))
    q = np.exp(zq - zq.max(-1, keepdims=True))
    q /= q.sum(-1, keepdims=True)
    drafts = np.minimum((rng.random((n, k, 1)) > np.cumsum(q, -1)).sum(-1), vocab - 1)
    return p, q, drafts


def test_tau_law_matches_the_sampling_path():
    """``block_verify`` on fresh uniforms realises ``block_tau_pmf`` (and its mean)."""
    rng = np.random.default_rng(5)
    n = 100_000
    for k, vocab, spread, noise in [(1, 3, 1.0, 1.0), (3, 4, 1.5, 0.7), (5, 4, 1.0, 0.4)]:
        p1, q1, d1 = _random_blocks(rng, 1, k, vocab, spread, noise)
        h, _ = block_accept_probs(p1, q1, d1)
        pmf = block_tau_pmf(h)[0]
        assert abs(pmf.sum() - 1.0) < 1e-12
        assert abs((pmf * np.arange(k + 1)).sum() - block_expected_tau(h)[0]) < 1e-12
        p, q, d = (np.repeat(a, n, axis=0) for a in (p1, q1, d1))
        _, _, tau = block_verify(p, q, d, rng.random((n, k)), rng.random(n))
        freq = np.bincount(tau, minlength=k + 1) / n
        sigma = np.sqrt(np.maximum(pmf * (1.0 - pmf), 1e-12) / n)
        assert np.all(np.abs(freq - pmf) < 5.0 * sigma + 1e-9), (freq, pmf)


def _emitted_joint_counts(target, draft, k, n, rng, verify):
    """Histogram of the first ``k + 1`` emitted tokens (the verifier's output completed by the
    target) over ``n`` independent draft blocks."""
    v = target.vocab
    x, q_rows = draft.sample_path(n, k, rng)
    q = q_rows[:, :k]
    p = np.stack([target.dist(x[:, :i]) for i in range(k + 1)], axis=1)
    acc, _, tau = verify(p, q, x, rng.random((n, k)), rng.random(n))
    for j in range(1, k + 1):
        todo = np.flatnonzero(tau < j)
        d = target.dist(acc[todo, :j])
        draw = np.minimum((rng.random((todo.size, 1)) > np.cumsum(d, -1)).sum(-1), v - 1)
        acc[todo, j] = draw
    flat = np.ravel_multi_index(acc.T, (v,) * (k + 1))
    return np.bincount(flat, minlength=v ** (k + 1))


def _chi2_pvalue(counts, expected_prob):
    stats = pytest.importorskip("scipy.stats")
    n = counts.sum()
    expected = n * expected_prob.reshape(-1)
    keep = expected >= 5.0
    stat = (((counts[keep] - expected[keep]) ** 2) / expected[keep]).sum()
    tail_obs, tail_exp = counts[~keep].sum(), expected[~keep].sum()
    dof = int(keep.sum()) - 1
    if tail_exp >= 5.0:
        stat += (tail_obs - tail_exp) ** 2 / tail_exp
        dof += 1
    return float(stats.chi2.sf(stat, dof))


@pytest.mark.parametrize("kind", ["independent", "near", "uniform"])
def test_sampled_output_follows_the_target(kind):
    """Chi-square of the sampled output (block verification + completion) against the target."""
    vocab, k, seed_t, seed_d, spread = 3, 3, 1, 2, 1.0
    target = TreeModel(vocab, k, seed_t, spread)
    draft = _drafts(target, k, seed_d)[kind]
    counts = _emitted_joint_counts(
        target, draft, k, 400_000, np.random.default_rng(17), block_verify
    )
    assert _chi2_pvalue(counts, target_joint(target, k)) > 1e-3


def test_sampled_output_test_has_power():
    """The chi-square rejects a variant whose full-block acceptance is wrong."""
    vocab, k, seed_t, seed_d, spread = 3, 3, 1, 2, 1.0
    target = TreeModel(vocab, k, seed_t, spread)
    draft = _drafts(target, k, seed_d)["independent"]

    def wrong(p, q, drafts, eta, u):
        return block_verify(p, q, drafts, eta, u, accept_probs=_accept_every_full_block)

    counts = _emitted_joint_counts(target, draft, k, 400_000, np.random.default_rng(17), wrong)
    assert _chi2_pvalue(counts, target_joint(target, k)) < 1e-6


@pytest.mark.parametrize("vocab,k,seed_t,seed_d,spread", CASES)
def test_theorem2_never_worse_in_expectation(vocab, k, seed_t, seed_d, spread):
    """E[tau] of block verification >= token-wise, averaged over the draft block (exactly)."""
    target = TreeModel(vocab, k, seed_t, spread)
    for kind, draft in _drafts(target, k, seed_d).items():
        block, token = expected_accepted_over_drafts(target, draft, k)
        assert block >= token - 1e-12, (kind, block, token)
        if kind in ("independent", "near", "uniform"):
            assert block > token + 1e-3, (kind, block, token)
        if kind in ("identical", "one_hot"):  # nothing to gain: the two laws coincide
            assert abs(block - token) < 1e-12, (kind, block, token)


def test_theorem2_does_not_hold_per_draft_block():
    """A draft block can be accepted less by block verification than by token-wise: the first token
    is likely accepted alone, but the second has a tiny ratio, which block verification charges to
    the whole block. Theorem 2 is an expectation over drafts; a kernel must not be held to it per
    block."""
    p = np.array([[[0.3, 0.35, 0.35], [0.01, 0.5, 0.49], [1 / 3, 1 / 3, 1 / 3]]])
    q = np.array([[[0.5, 0.25, 0.25], [0.34, 0.33, 0.33]]])
    drafts = np.array([[0, 0]])
    h, w = block_accept_probs(p, q, drafts)
    assert w[0, 1] == pytest.approx(0.6)
    assert block_residual_mass(p, q, w)[0, 1] == 0.0
    assert block_expected_tau(h)[0] < 0.05
    assert token_expected_tau(p, q, drafts)[0] > 0.6


def test_k1_decisions_equal_token_wise_on_the_same_uniforms():
    """With one draft token the two verifiers coincide: the accept decision uses the same uniform,
    the correction (or bonus) token the same inverse-CDF draw."""
    rng = np.random.default_rng(23)
    n = 4000
    for vocab, spread, noise in [(3, 1.0, 1.0), (6, 2.0, 0.3), (11, 0.5, 2.0)]:
        p, q, d = _random_blocks(rng, n, 1, vocab, spread, noise)
        eta, u = rng.random((n, 1)), rng.random(n)
        a_blk, n_blk, _ = block_verify(p, q, d, eta, u)
        a_tok, n_tok, _ = token_verify(p, q, d, eta, u)
        np.testing.assert_array_equal(n_blk, n_tok)
        np.testing.assert_array_equal(a_blk, a_tok)


def test_one_hot_draft_has_the_token_wise_law():
    """A deterministic draft (what a placeholder or greedy draft is): the tau law and the
    correction-token law equal token-wise verification's, for any target."""
    rng = np.random.default_rng(31)
    for k, vocab in [(2, 3), (4, 5), (5, 3)]:
        p, _, d = _random_blocks(rng, 200, k, vocab, 1.5, 1.0)
        q = np.eye(vocab)[d]  # [n, K, V]
        h, w = block_accept_probs(p, q, d)
        survive = np.cumprod(token_accept_probs(p, q, d), axis=1)  # P(tau >= i), i = 1..K
        pmf_token = np.concatenate([1.0 - survive[:, :1], survive[:, :-1] - survive[:, 1:]], 1)
        pmf_token = np.concatenate([pmf_token, survive[:, -1:]], 1)
        np.testing.assert_allclose(block_tau_pmf(h), pmf_token, atol=1e-12)
        for tau in range(k + 1):
            t = np.full(len(d), tau)
            blk = residual_weights(p, q, w, t)
            tok = residual_weights(p, q, np.ones_like(w), t)
            np.testing.assert_allclose(
                blk / blk.sum(-1, keepdims=True), tok / tok.sum(-1, keepdims=True), atol=1e-12
            )


def test_identical_models_accept_every_draft():
    rng = np.random.default_rng(41)
    for k, vocab in [(1, 4), (3, 3), (5, 8)]:
        p, _, d = _random_blocks(rng, 50, k, vocab, 1.0, 0.0)
        q = p[:, :k].copy()
        h, w = block_accept_probs(p, q, d)
        np.testing.assert_allclose(w, 1.0, atol=1e-12)
        pmf = block_tau_pmf(h)
        np.testing.assert_allclose(pmf[:, k], 1.0, atol=1e-12)
        eta, u = rng.random((50, k)), rng.random(50)
        acc, num, tau = block_verify(p, q, d, eta, u)
        assert np.all(tau == k) and np.all(num == k + 1)
        np.testing.assert_array_equal(acc[:, :k], d)
        np.testing.assert_array_equal(acc[:, k], inv_cdf(p[:, k], u))


def test_draft_token_impossible_for_the_target_stops_at_once():
    """p_0(d_0) = 0: every sub-block has weight 0, so tau = 0 with certainty and the correction
    token comes from p_0 (here: disjoint supports)."""
    p = np.zeros((1, 3, 4))
    p[0, 0, 2:] = 0.5
    p[0, 1:, :] = 0.25
    q = np.zeros((1, 2, 4))
    q[0, :, :2] = 0.5
    d = np.array([[0, 1]])
    h, w = block_accept_probs(p, q, d)
    np.testing.assert_array_equal(block_tau_pmf(h)[0], [1.0, 0.0, 0.0])
    for u in (1e-9, 0.3, 0.7, 1.0):
        acc, num, tau = block_verify(p, q, d, np.full((1, 2), 0.5), np.array([u]))
        assert tau[0] == 0 and num[0] == 1 and acc[0, 0] in (2, 3) and np.all(acc[0, 1:] == -1)


def test_target_certain_of_every_draft_token_accepts_the_block():
    """p_i(d_i) = 1 along the draft: tau = K for any uniforms, the bonus token comes from p_K."""
    k, v = 4, 5
    rng = np.random.default_rng(3)
    d = rng.integers(0, v, (1, k))
    p = np.zeros((1, k + 1, v))
    p[0, np.arange(k), d[0]] = 1.0
    p[0, k] = rng.dirichlet(np.ones(v))
    q = rng.dirichlet(np.ones(v), size=(1, k))
    h, w = block_accept_probs(p, q, d)
    np.testing.assert_allclose(block_tau_pmf(h)[0, k], 1.0, atol=1e-12)
    acc, num, tau = block_verify(p, q, d, rng.random((1, k)), np.array([0.5]))
    assert tau[0] == k and num[0] == k + 1
    np.testing.assert_array_equal(acc[0, :k], d[0])


def test_selected_residual_always_has_mass():
    """Whenever a sub-block length is possible, its correction distribution is normalisable."""
    rng = np.random.default_rng(7)
    for k, vocab in [(1, 3), (3, 4), (5, 6)]:
        p, q, d = _random_blocks(rng, 500, k, vocab, 2.0, 1.5)
        h, w = block_accept_probs(p, q, d)
        pmf = block_tau_pmf(h)
        for tau in range(k + 1):
            res = residual_weights(p, q, w, np.full(len(d), tau))
            possible = pmf[:, tau] > 1e-12
            assert np.all(res[possible].sum(-1) > 0.0), tau


def test_a_token_the_proposal_excludes_has_the_limit_weight():
    """``q_i(d_i) = 0`` cannot happen for a drawn token. For a stale or mismatched row the weight
    is the limit of ``min(w p / q, 1)`` as ``q`` goes to 0: 1 where the target allows the token
    and the prefix is possible, 0 otherwise; a weight of 0 stays 0."""
    p = np.array([[[0.5, 0.5, 0.0], [0.2, 0.3, 0.5], [0.1, 0.1, 0.8]]])  # K = 2
    q = np.array([[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])
    # d_0 = 0: q = 0, p = 0.5 -> 1; d_1 = 2: p / q = 0.5.
    assert block_weights(p, q, np.array([[0, 2]]))[0].tolist() == [1.0, 1.0, 0.5]
    # d_0 = 2: q = 0 and p = 0 -> 0; d_1 = 0: q = 0, p = 0.2, but the prefix is impossible -> 0.
    assert block_weights(p, q, np.array([[2, 0]]))[0].tolist() == [1.0, 0.0, 0.0]


def test_philox_known_answers():
    """The cuRAND-compatible Philox4x32-10 model reproduces the Random123 known-answer vectors."""
    assert philox4x32_10((0, 0, 0, 0), (0, 0)) == (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)
    ones = (0xFFFFFFFF,) * 4
    assert philox4x32_10(ones, (0xFFFFFFFF, 0xFFFFFFFF)) == (
        0x408F276D,
        0x41C83B0E,
        0xA20BC7C6,
        0x6D5451FD,
    )
    ctr = (0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344)
    assert philox4x32_10(ctr, (0xA4093822, 0x299F31D0)) == (
        0xD16CFE09,
        0x94FDCCEB,
        0x5001E420,
        0x24126EA1,
    )


def test_curand_uniform_is_one_word_stream():
    """Per (seed, subsequence) the draws form one stream of words and the offset skips ahead in it:
    draw ``i`` at offset ``o`` is word ``o + i`` (what makes the per-step offset windows disjoint).
    Uniforms lie in (0, 1]."""
    for seed, sub in [(1, 0), (2**40 + 7, 5), (-5, 2**33 + 1)]:
        base = [curand_uniform(seed, 0, sub, g) for g in range(40)]
        assert all(0.0 < u <= 1.0 for u in base)
        assert len(set(base)) == len(base)
        for offset in (1, 3, 4, 6, 13):
            for i in range(8):
                assert curand_uniform(seed, offset, sub, i) == base[offset + i]


def test_bonus_row_has_weight_one():
    """``tau = K`` selects the bonus row (no proposal, weight 1); every residual is nonnegative."""
    rng = np.random.default_rng(11)
    k = 3
    p, q, d = _random_blocks(rng, 6, k, 5, 1.0, 1.0)
    _, w = block_accept_probs(p, q, d)
    np.testing.assert_allclose(residual_weights(p, q, w, np.full(6, k)), p[:, k])
    for tau in range(k + 1):
        assert np.all(residual_weights(p, q, w, np.full(6, tau)) >= 0.0)
