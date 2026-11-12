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
"""The block-verification kernels (TLLM_SPEC_BLOCK_VERIFY) against the float64 reference model.

Runs on a CUDA GPU, or on the CPU through Triton's interpreter (``TRITON_INTERPRET=1``, set
here when no GPU is present; sample sizes are then reduced). The model is ``block_verify_ref``
(exactness and optimality of the algorithm itself: ``test_block_verification_reference.py``).

Random stream: request ``b`` draws ``eta_i`` (the acceptance of the sub-block of length ``i``)
from word ``i - 1`` and the correction token from word ``K`` of the Philox stream (seed[b],
subsequence b, offset[b]) -- the ``K + 1`` words per step ``fast_rejection_sampling`` uses.

Checks:
  * decisions and tokens vs the float64 model on the same draws: the accepted length ``tau`` and
    the correction token of every request. Every disagreement must be explained by float32
    rounding: ``eta`` within the conditioning-scaled error bar of ``h``, or the correction draw
    within ``_CDF_TOL`` (relative to the row mass) of an inverse-CDF boundary;
  * per-request 64-bit seeds and offsets of every alignment, per-row temperatures, masked logits,
    a runtime draft length below the buffer's, a vocabulary that is not a chunk multiple, CUDA-
    graph padding requests (zero logits, one-hot proposals), ``tau = 0`` / ``tau = K`` extremes;
  * where it must equal the token-wise kernel: ``K = 1`` (bit for bit), identical models (accept
    everything);
  * losslessness at the kernel: the sampled ``tau`` against its exact law, and the emitted tokens
    (kernel output completed by the target) against the target's joint distribution, over small
    random autoregressive models;
  * the stream: no Philox word is used twice, steps are independent, runs are deterministic, and
    CUDA-graph replay equals eager;
  * the ``TLLM_SPEC_BLOCK_VERIFY`` switch of ``SpecWorkerBase`` (temperature-only sampling only).
"""

import math
import os
import types

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    # No GPU: Triton's interpreter runs the kernels on the CPU (read when the kernels are built).
    os.environ.setdefault("TRITON_INTERPRET", "1")

from block_verify_ref import BV_KEY_XOR as REF_BV_KEY_XOR  # noqa: E402
from block_verify_ref import (  # noqa: E402
    TreeModel,
    block_accept_probs,
    block_expected_tau,
    block_residual_mass,
    block_tau_pmf,
    bv_uniform,
    inv_cdf,
    residual_weights,
    target_joint,
)

from tensorrt_llm._torch.speculative import fast_block_verification as fbv  # noqa: E402
from tensorrt_llm._torch.speculative import fast_rejection_sampling as frs  # noqa: E402

ON_GPU = torch.cuda.is_available()
INTERPRETED = type(frs._rs_row_stats_kernel).__name__ == "InterpretedFunction"
pytestmark = pytest.mark.skipif(
    not (ON_GPU or INTERPRETED),
    reason="needs a CUDA GPU, or TRITON_INTERPRET=1 set before the kernels are imported",
)

DEV = "cuda" if ON_GPU else "cpu"
V_ONE, V_RAGGED = 1000, 9000  # one chunk; three chunks with a ragged tail
V_BIG = 163840 if ON_GPU else V_RAGGED
K = 5
TRIALS = 6 if ON_GPU else 2

# float32 error bars: the acceptance probability h_i = R_i / (R_i + 1 - w_i) moves by about
# (dR + dw) / (R_i + 1 - w_i) when R_i (absolute error ~1e-6) and w_i (relative ~3e-7 per factor)
# do; the correction draw's inverse CDF by _CDF_TOL of the row mass (+ the cancellation of
# w * p - q, 1e-6 of absolute mass).
_H_ABS, _H_REL = 1.5e-6, 3e-7
_CDF_TOL = 2e-5
_CDF_CANCEL = 1e-6


def _peaked_logits(rows, vocab, gen, spread=2.0, ncand=48, top=14.0):
    base = torch.randn(rows, vocab, generator=gen, device=DEV) * spread
    idx = torch.randint(0, vocab, (rows, ncand), generator=gen, device=DEV)
    boost = torch.linspace(top, top - 8.0, ncand, device=DEV).unsqueeze(0).expand(rows, -1)
    base.scatter_add_(1, idx, boost + torch.randn(rows, ncand, generator=gen, device=DEV))
    return base.float().contiguous()


def _as_np(x, dtype=np.float64):
    return x.detach().cpu().numpy().astype(dtype)


def _softmax64(logits, temps):
    """float64 softmax(logits / T) with one temperature per row."""
    return torch.softmax(logits.double() / temps.double().unsqueeze(1), dim=-1).cpu().numpy()


def _make_case(g, k, vocab, noise, temperature, gen, trial=0, buf_steps=K, dummy=False):
    """A verify step: target logits, a stored proposal per (request, position) that differs from
    the target by ``noise``, drafts drawn from it, per-request seeds / offsets (every offset
    alignment). With ``dummy`` the last request is CUDA-graph padding (dummy slot, one-hot
    placeholder proposal, draft 0)."""
    tgt = _peaked_logits(g * (k + 1), vocab, gen)
    drf = tgt.view(g, k + 1, vocab)[:, :k] + noise * torch.randn(
        g, k, vocab, generator=gen, device=DEV
    )
    temps = torch.full((g * (k + 1),), temperature, dtype=torch.float32, device=DEV)
    q = torch.softmax(drf.double() / temperature, dim=-1).float()
    drafts = torch.multinomial(q.view(-1, vocab), 1, generator=gen).view(g, k)
    drafts = drafts.to(torch.int32).contiguous()
    slots = torch.randperm(g + 2, generator=torch.Generator().manual_seed(trial)).to(DEV)[:g]
    buf = torch.zeros((g + 3, buf_steps, vocab), dtype=torch.float32, device=DEV)
    buf[slots, :k] = q
    dummy_row = -1
    if dummy:
        dummy_row = g + 2
        buf[dummy_row, :, 0] = 1.0
        slots[-1] = dummy_row
        drafts[-1] = 0
    seeds_py = [1000 + 17 * trial + b for b in range(g)]
    offs_py = [(k + 1) * trial + (b % 4) for b in range(g)]
    return types.SimpleNamespace(
        g=g,
        k=k,
        vocab=vocab,
        logits=tgt,
        temps=temps,
        drafts=drafts,
        slots=slots.to(torch.int64),
        buf=buf,
        dummy_row=dummy_row,
        seeds_py=seeds_py,
        offs_py=offs_py,
        seeds=torch.tensor(seeds_py, dtype=torch.int64, device=DEV),
        offs=torch.tensor(offs_py, dtype=torch.int64, device=DEV),
    )


def _run(case, verify=None):
    verify = verify or fbv.fast_rs_verify_block
    return verify(
        case.logits,
        case.temps,
        case.drafts,
        case.slots,
        case.dummy_row,
        case.buf,
        case.seeds,
        case.offs,
    )


def _model_inputs(case):
    """float64 target rows [G, K + 1, V], proposals [G, K, V] and drafts [G, K] of a case (padding
    requests read zero logits, i.e. a uniform target)."""
    g, k, vocab = case.g, case.k, case.vocab
    p = _softmax64(case.logits, case.temps).reshape(g, k + 1, vocab)
    q = _as_np(case.buf[case.slots, :k])
    if case.dummy_row >= 0:
        pad = _as_np(case.slots) == case.dummy_row
        p[pad] = 1.0 / vocab
    return p, q, _as_np(case.drafts, np.int64)


def _cdf_gap(weights, u, a, b):
    """Smallest delta for which ``u`` lies within delta of both tokens' inverse-CDF intervals
    (float64 running sum of ``weights``)."""
    c = np.cumsum(np.asarray(weights, dtype=np.float64))

    def dist(t):
        lo = c[t - 1] if t > 0 else 0.0
        if lo <= u < c[t]:
            return 0.0
        return lo - u if u < lo else u - c[t]

    return max(dist(a), dist(b))


class _Gaps:
    """Rounding explanations of kernel-vs-model disagreements, normalised so that <= 1 is explained."""

    def __init__(self):
        self.worst = {"accept": 0.0, "token": 0.0}
        self.count = 0

    def add(self, kind, size):
        self.worst[kind] = max(self.worst[kind], size)
        self.count += size > 0.0

    def check(self, total):
        assert self.worst["accept"] <= 1.0 and self.worst["token"] <= 1.0, self.worst
        assert self.count <= max(2, int(0.02 * total)), (self.count, total)


def _check_rows(acc, num, p, q, drafts, seeds, offsets, k, gaps):
    """Every request's accepted length and correction token vs the float64 model on the same
    draws (draw ``i - 1`` is the uniform of the sub-block of length ``i``, draw K the token's)."""
    h, w = block_accept_probs(p, q, drafts)
    r = block_residual_mass(p, q, w)
    taus = []
    for b in range(acc.shape[0]):
        eta = np.array([bv_uniform(seeds[b], offsets[b], b, i) for i in range(k)])
        u = bv_uniform(seeds[b], offsets[b], b, k)
        hit = np.flatnonzero(eta <= h[b, 1:]) + 1
        tau_ref = int(hit.max()) if hit.size else 0
        tau = int(num[b]) - 1
        assert 0 <= tau <= k, (b, tau)
        row = [int(x) for x in acc[b]]
        assert row[:tau] == [int(x) for x in drafts[b, :tau]]
        assert all(x == -1 for x in row[tau + 1 :]), row
        taus.append(tau)
        if tau != tau_ref:
            m = max(tau, tau_ref)  # the longest sub-block whose decision flipped
            den = r[b, m] + 1.0 - w[b, m] if m < k else 1.0
            tol = (_H_ABS * (m < k) + _H_REL * (1 + m)) / max(den, 1e-9) + 1e-7
            gaps.add("accept", abs(eta[m - 1] - h[b, m]) / tol)
            continue
        res = residual_weights(p[b : b + 1], q[b : b + 1], w[b : b + 1], np.array([tau]))[0]
        total = float(res.sum())
        want = int(inv_cdf(res[None], np.array([u]))[0])
        if row[tau] == want:
            gaps.add("token", 0.0)
        else:
            scaled = u * total
            gap = _cdf_gap(res, scaled, row[tau], want) / max(total, 1e-300)
            gaps.add("token", gap / (_CDF_TOL + _CDF_CANCEL / max(total, 1e-12)))
    return taus


@pytest.mark.parametrize("vocab", [V_ONE, V_RAGGED])
@pytest.mark.parametrize("noise", [0.4, 2.0])
@pytest.mark.parametrize("temperature", [1.0, 0.6])
@pytest.mark.parametrize("k", [1, 3, K])
def test_matches_float64_model(k, temperature, noise, vocab):
    gen = torch.Generator(device=DEV).manual_seed(100 + 7 * k)
    g = 8
    gaps, taus = _Gaps(), []
    for trial in range(TRIALS):
        case = _make_case(g, k, vocab, noise, temperature, gen, trial)
        acc, num = _run(case)
        p, q, d = _model_inputs(case)
        taus += _check_rows(
            _as_np(acc, np.int64),
            _as_np(num, np.int64),
            p,
            q,
            d,
            case.seeds_py,
            case.offs_py,
            k,
            gaps,
        )
    print(
        f"K={k} T={temperature} noise={noise} V={vocab}: tau histogram "
        f"{np.bincount(taus, minlength=k + 1).tolist()}, worst {gaps.worst}, rows needing rounding "
        f"{gaps.count}/{len(taus)}"
    )
    gaps.check(len(taus))


def test_per_request_seeds_offsets_and_row_temperatures():
    """64-bit (negative) seeds, offsets past 2**32 and past 2**34 (the counter's high word), a
    temperature per row."""
    gen = torch.Generator(device=DEV).manual_seed(3)
    seeds_py = [1, 2**40 + 7, -5, 0x7FFFFFFFFFFFFFF0, 99, -(2**62), 12345, 31337]
    offs_py = [0, 6, 2**33 + 3, 18, 7, 2**32 - 1, 2**34 + 5, 2**40 + 3]
    g = len(seeds_py)
    gaps, total = _Gaps(), 0
    for trial in range(TRIALS):
        case = _make_case(g, K, V_RAGGED, 0.8, 1.0, gen, trial)
        case.temps = 0.6 + 0.8 * torch.rand(g * (K + 1), generator=gen, device=DEV)
        case.seeds_py, case.offs_py = seeds_py, offs_py
        case.seeds = torch.tensor(seeds_py, dtype=torch.int64, device=DEV)
        case.offs = torch.tensor(offs_py, dtype=torch.int64, device=DEV)
        acc, num = _run(case)
        p, q, d = _model_inputs(case)
        _check_rows(
            _as_np(acc, np.int64), _as_np(num, np.int64), p, q, d, seeds_py, offs_py, K, gaps
        )
        total += g
    gaps.check(total)


def test_masked_logits():
    """-inf logits (guided-decoding style masks) get probability 0, also in chunks with no finite
    logit at all, and every emitted token stays inside the allowed set."""
    gen = torch.Generator(device=DEV).manual_seed(4)
    g, vocab = 4, V_RAGGED
    case = _make_case(g, K, vocab, 0.8, 1.0, gen)
    chunk = frs._CHUNK
    pool = torch.cat([torch.arange(0, chunk), torch.arange(2 * chunk, vocab)])
    allowed = pool[torch.randperm(pool.numel(), generator=torch.Generator().manual_seed(0))[:100]]
    keep = torch.zeros(vocab, dtype=torch.bool, device=DEV)
    keep[allowed.to(DEV)] = True  # the middle chunk has no allowed token
    case.logits = case.logits.masked_fill(~keep, float("-inf"))
    noisy = case.logits.view(g, K + 1, vocab)[:, :K] + 0.5 * torch.randn(
        g, K, vocab, generator=gen, device=DEV
    )
    q = torch.softmax(noisy, dim=-1)  # proposals live on the allowed set
    case.buf[case.slots, :K] = q
    case.drafts = torch.multinomial(q.view(-1, vocab), 1, generator=gen).view(g, K)
    case.drafts = case.drafts.to(torch.int32).contiguous()
    acc, num = _run(case)
    for b in range(g):
        emitted = acc[b, : int(num[b])].long()
        assert bool(keep[emitted].all()), (b, emitted.tolist())
    gaps = _Gaps()
    p, qq, d = _model_inputs(case)
    acc_np, num_np = _as_np(acc, np.int64), _as_np(num, np.int64)
    _check_rows(acc_np, num_np, p, qq, d, case.seeds_py, case.offs_py, K, gaps)
    gaps.check(g)


def test_short_draft_and_ragged_vocab():
    """Runtime draft length 3 in a 5-step buffer, a vocabulary that is not a chunk multiple,
    per-request temperatures."""
    gen = torch.Generator(device=DEV).manual_seed(6)
    k, g = 3, 6
    gaps, total = _Gaps(), 0
    for trial in range(TRIALS):
        case = _make_case(g, k, V_RAGGED + 123, 0.6, 0.7, gen, trial, buf_steps=K)
        acc, num = _run(case)
        p, q, d = _model_inputs(case)
        _check_rows(
            _as_np(acc, np.int64),
            _as_np(num, np.int64),
            p,
            q,
            d,
            case.seeds_py,
            case.offs_py,
            k,
            gaps,
        )
        total += g
    gaps.check(total)


def test_padding_requests_use_the_dummy_slot():
    """A CUDA-graph padding request (dummy slot) reads zero logits (a uniform target) and its
    one-hot placeholder proposal; real requests are unaffected."""
    gen = torch.Generator(device=DEV).manual_seed(8)
    g = 6
    gaps, total = _Gaps(), 0
    for trial in range(TRIALS):
        case = _make_case(g, K, V_RAGGED, 0.8, 1.0, gen, trial, dummy=True)
        # The padding rows' logits are garbage in a real graph: they must be ignored.
        case.logits.view(g, K + 1, -1)[-1] = 1e4 * torch.randn(K + 1, V_RAGGED, device=DEV)
        acc, num = _run(case)
        p, q, d = _model_inputs(case)
        _check_rows(
            _as_np(acc, np.int64),
            _as_np(num, np.int64),
            p,
            q,
            d,
            case.seeds_py,
            case.offs_py,
            K,
            gaps,
        )
        total += g
        assert 1 <= int(num[-1]) <= K + 1
    gaps.check(total)


def _one_hot_logits(rows, vocab, tokens, high=0.0, low=float("-inf")):
    lg = torch.full((rows, vocab), low, device=DEV)
    lg[torch.arange(rows, device=DEV), tokens.to(DEV)] = high
    return lg


def test_tau_zero_when_the_first_draft_token_is_impossible():
    """p_0(d_0) = 0: every sub-block has weight 0, so tau = 0 for any draw; the correction token
    comes from the target's first row, never from the drafts."""
    gen = torch.Generator(device=DEV).manual_seed(12)
    g, vocab = 6, V_RAGGED
    case = _make_case(g, K, vocab, 1.0, 1.0, gen)
    rows = case.logits.view(g, K + 1, vocab)
    rows[torch.arange(g, device=DEV), 0, case.drafts[:, 0].long()] = float("-inf")
    acc, num = _run(case)
    assert torch.equal(num.cpu(), torch.ones(g, dtype=torch.int32))
    assert bool((acc[:, 1:] == -1).all())
    for b in range(g):
        assert int(acc[b, 0]) != int(case.drafts[b, 0])
    gaps = _Gaps()
    p, q, d = _model_inputs(case)
    _check_rows(
        _as_np(acc, np.int64), _as_np(num, np.int64), p, q, d, case.seeds_py, case.offs_py, K, gaps
    )
    gaps.check(g)


def test_tau_k_when_the_target_is_certain_of_every_draft_token():
    """p_i(d_i) = 1 along the draft: every sub-block is accepted, tau = K for any draw, and the
    bonus token comes from the last row (here a point mass)."""
    gen = torch.Generator(device=DEV).manual_seed(13)
    g, vocab = 6, V_RAGGED
    case = _make_case(g, K, vocab, 1.0, 1.0, gen)
    bonus = torch.randint(0, vocab, (g,), generator=torch.Generator().manual_seed(1))
    toks = torch.cat([case.drafts.cpu().long(), bonus[:, None]], dim=1)
    case.logits = _one_hot_logits(g * (K + 1), vocab, toks.reshape(-1)).contiguous()
    acc, num = _run(case)
    assert torch.equal(num.cpu(), torch.full((g,), K + 1, dtype=torch.int32))
    assert torch.equal(acc.cpu().long(), toks)  # no -1 anywhere


def test_empty_batch_returns_empty_outputs():
    logits = torch.empty((0, V_ONE), device=DEV)
    acc, num = fbv.fast_rs_verify_block(
        logits,
        torch.empty(0, device=DEV),
        torch.empty((0, K), dtype=torch.int32, device=DEV),
        torch.empty(0, dtype=torch.int64, device=DEV),
        -1,
        torch.empty((1, K, V_ONE), device=DEV),
        torch.empty(0, dtype=torch.int64, device=DEV),
        torch.empty(0, dtype=torch.int64, device=DEV),
    )
    assert acc.shape == (0, K + 1) and num.shape == (0,)


def _on_the_block_stream(case):
    """The case with seeds ``seed ^ BV_KEY_XOR``: the token-wise kernel (raw seed as its key) then
    draws the words block verification draws."""
    return types.SimpleNamespace(**{**vars(case), "seeds": case.seeds ^ REF_BV_KEY_XOR})


def test_the_stream_key_is_the_documented_one_and_keyed_apart():
    """The kernel module and the model agree on the key, and it differs from the raw seed (the
    token-wise verify stream, the reference draft sampler) and from the fused draft stream's."""
    assert fbv.BV_KEY_XOR == REF_BV_KEY_XOR
    assert fbv.BV_KEY_XOR not in (0, frs.DRAFT_KEY_XOR)
    assert 0 < fbv.BV_KEY_XOR < 2**63  # fits the int64 seed tensors


def test_k1_equals_the_token_wise_kernel():
    """With one draft token block verification is token-wise verification: the same uniform
    decides acceptance, the same draw picks the correction or bonus token. The token-wise kernel
    runs on the block-verification stream (seed ^ BV_KEY_XOR), which also pins that key."""
    gen = torch.Generator(device=DEV).manual_seed(21)
    g = 24 if ON_GPU else 8
    for trial in range(TRIALS):
        case = _make_case(g, 1, V_RAGGED, 1.0, 1.0, gen, trial)
        a_blk, n_blk = _run(case)
        a_tok, n_tok = _run(_on_the_block_stream(case), frs.fast_rs_verify)
        # Both draw u * q(d) < p(d) vs u <= min(1, p / q): they differ only for a draw within
        # float32 rounding of the threshold.
        p, q, d = _model_inputs(case)
        ratio = p[np.arange(g), 0, d[:, 0]] / q[np.arange(g), 0, d[:, 0]]
        for b in np.flatnonzero(_as_np(n_blk) != _as_np(n_tok)):
            u = bv_uniform(case.seeds_py[b], case.offs_py[b], b, 0)
            assert abs(u - ratio[b]) < 1e-6 * max(ratio[b], 1.0), (b, u, ratio[b])
        same = _as_np(n_blk) == _as_np(n_tok)
        assert torch.equal(a_blk.cpu()[torch.from_numpy(same)], a_tok.cpu()[torch.from_numpy(same)])


def test_identical_models_accept_everything():
    """Proposal == target (up to float32 rounding): the whole block is accepted and the bonus
    token is the inverse-CDF draw from the last target row, as in token-wise verification."""
    gen = torch.Generator(device=DEV).manual_seed(22)
    g = 8
    for k in (1, 3, K):
        case = _make_case(g, k, V_RAGGED, 0.0, 1.0, gen)
        p_all = torch.softmax(case.logits.view(g, k + 1, V_RAGGED).double(), -1)
        case.buf[case.slots, :k] = p_all[:, :k].float()
        case.drafts = torch.multinomial(
            p_all[:, :k].reshape(-1, V_RAGGED).float(), 1, generator=gen
        )
        case.drafts = case.drafts.view(g, k).to(torch.int32).contiguous()
        acc, num = _run(case)
        acc_tok, num_tok = _run(_on_the_block_stream(case), frs.fast_rs_verify)
        assert torch.equal(num.cpu(), torch.full((g,), k + 1, dtype=torch.int32))
        assert torch.equal(num.cpu(), num_tok.cpu())
        assert torch.equal(acc[:, :k].cpu(), case.drafts.cpu())
        # Bonus token: the model's draw on the last target row (within the rounding bar).
        p = _softmax64(case.logits, case.temps).reshape(g, k + 1, V_RAGGED)
        for b in range(g):
            u = bv_uniform(case.seeds_py[b], case.offs_py[b], b, k)
            want = int(inv_cdf(p[b, k][None], np.array([u]))[0])
            got = int(acc[b, k])
            assert got == want or _cdf_gap(p[b, k], u, got, want) < _CDF_TOL, (b, got, want)
            assert (
                got == int(acc_tok[b, k])
                or _cdf_gap(p[b, k], u, got, int(acc_tok[b, k])) < _CDF_TOL
            )


def _chi2_pvalue(counts, probs, min_expected=5.0):
    """Chi-square p-value (Wilson-Hilferty) with every cell below ``min_expected`` pooled."""
    counts = np.asarray(counts, dtype=np.float64)
    n = counts.sum()
    exp = n * np.asarray(probs, dtype=np.float64)
    keep = exp >= min_expected
    obs_k, exp_k = list(counts[keep]), list(exp[keep])
    rest_obs, rest_exp = counts[~keep].sum(), exp[~keep].sum()
    if rest_exp >= min_expected:
        obs_k.append(rest_obs)
        exp_k.append(rest_exp)
    obs_k, exp_k = np.array(obs_k), np.array(exp_k)
    stat = float(((obs_k - exp_k) ** 2 / exp_k).sum())
    dof = max(len(obs_k) - 1, 1)
    z = ((stat / dof) ** (1 / 3) - (1 - 2 / (9 * dof))) / math.sqrt(2 / (9 * dof))
    return stat, dof, 0.5 * math.erfc(z / math.sqrt(2))


def _spread_scenario(rng, k, vocab):
    """One (p, q, drafts) with every accepted length possible (each of probability at least 0.04):
    the first random draw that has that spread."""
    for _ in range(10000):
        zt = rng.normal(0.0, 1.5, (1, k + 1, vocab))
        p = np.exp(zt - zt.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        zq = zt[:, :k] + rng.normal(0.0, 0.8, (1, k, vocab))
        q = np.exp(zq - zq.max(-1, keepdims=True))
        q /= q.sum(-1, keepdims=True)
        d = np.minimum((rng.random((1, k, 1)) > np.cumsum(q, -1)).sum(-1), vocab - 1)
        h, _ = block_accept_probs(p, q, d)
        if block_tau_pmf(h).min() > 0.04:
            return p, q, d
    raise AssertionError("no scenario with a spread-out tau law")


def test_tau_law_matches_the_exact_law():
    """Replicate one (target, proposal, draft block) over many requests (each its own Philox
    stream): the accepted-length histogram follows the exact law and its mean E[tau] + 1."""
    k, vocab = 4, 32
    n = 4096 if ON_GPU else 160
    p, q, d = _spread_scenario(np.random.default_rng(3), k, vocab)
    h, _ = block_accept_probs(p, q, d)
    pmf = block_tau_pmf(h)[0]
    logits = torch.log(torch.from_numpy(p[0]).float()).repeat(n, 1).to(DEV).contiguous()
    buf = torch.zeros((2, K, vocab), dtype=torch.float32, device=DEV)
    buf[0, :k] = torch.from_numpy(q[0]).float()
    drafts = torch.from_numpy(d).int().repeat(n, 1).to(DEV).contiguous()
    nums = []
    for step in range(1 if ON_GPU else 2):
        acc, num = fbv.fast_rs_verify_block(
            logits,
            torch.ones(n * (k + 1), device=DEV),
            drafts,
            torch.zeros(n, dtype=torch.int64, device=DEV),
            -1,
            buf,
            torch.full((n,), 12, dtype=torch.int64, device=DEV),
            torch.full((n,), 5 * step, dtype=torch.int64, device=DEV),
        )
        nums.append(_as_np(num, np.int64) - 1)
    tau = np.concatenate(nums)
    stat, dof, pv = _chi2_pvalue(np.bincount(tau, minlength=k + 1), pmf)
    mean = float(block_expected_tau(h)[0])
    sd = math.sqrt(float((pmf * (np.arange(k + 1) - mean) ** 2).sum()))
    print(f"tau law: chi2={stat:.1f} dof={dof} p={pv:.3f}; mean {tau.mean():.3f} vs {mean:.3f}")
    assert pv > 1e-3
    assert abs(tau.mean() - mean) < 4.0 * sd / math.sqrt(tau.size)


def test_emitted_tokens_follow_the_target_joint():
    """Losslessness at the kernel (Theorem 1): over autoregressive target / draft models, the
    kernel's output completed by the target has the target's joint distribution. The drafts are
    drawn from the proposals, which the verifier reads from the draft-probability buffer."""
    vocab, k = 3, 3
    n = 24000 if ON_GPU else 240
    rng = np.random.default_rng(17)
    target = TreeModel(vocab, k, 1, 1.0)
    draft = TreeModel(vocab, k, 2, 1.0)
    x, q_rows = draft.sample_path(n, k, rng)
    p = np.stack([target.dist(x[:, :i]) for i in range(k + 1)], axis=1)
    logits = torch.from_numpy(np.log(p)).float().reshape(n * (k + 1), vocab).to(DEV).contiguous()
    buf = torch.zeros((n + 1, K, vocab), dtype=torch.float32, device=DEV)
    buf[:n, :k] = torch.from_numpy(q_rows[:, :k]).float().to(DEV)
    acc, num = fbv.fast_rs_verify_block(
        logits,
        torch.ones(n * (k + 1), device=DEV),
        torch.from_numpy(x).int().to(DEV).contiguous(),
        torch.arange(n, dtype=torch.int64, device=DEV),
        -1,
        buf,
        torch.full((n,), 99, dtype=torch.int64, device=DEV),
        torch.zeros(n, dtype=torch.int64, device=DEV),
    )
    out, tau = _as_np(acc, np.int64), _as_np(num, np.int64) - 1
    for j in range(1, k + 1):  # complete the unverified tail with the target
        todo = np.flatnonzero(tau < j)
        dist = target.dist(out[todo, :j])
        out[todo, j] = np.minimum(
            (rng.random((todo.size, 1)) > np.cumsum(dist, -1)).sum(-1), vocab - 1
        )
    counts = np.bincount(
        np.ravel_multi_index(out.T, (vocab,) * (k + 1)), minlength=vocab ** (k + 1)
    )
    stat, dof, pv = _chi2_pvalue(counts, target_joint(target, k).reshape(-1))
    print(f"joint of {n} emitted blocks vs target: chi2={stat:.1f} dof={dof} p={pv:.3f}")
    assert pv > (1e-3 if ON_GPU else 1e-4)


def test_philox_words_are_never_reused():
    """Model of the stream layout over consecutive steps (offset window max_draft_len + 1), with
    every request on one seed: block verification's K + 1 words per step (key seed ^ BV_KEY_XOR,
    subsequence = the request) are never drawn twice, and none of them is a word of the other
    streams that share the seed tensors (the token-wise verify stream, the reference draft
    sampler's, the fused draft stream's). The kernels' use of these words is pinned by the
    comparisons against the float64 model on the same draws."""
    window = K + 1
    seed, m64 = 7, 2**64 - 1
    for k in range(1, K + 1):
        for g in (1, 4, 16):
            mine, others = set(), set()
            for step in range(3):
                for b in range(g):
                    offset = (b % 3 + step) * window
                    for i in range(k + 1):
                        word = ((seed & m64) ^ fbv.BV_KEY_XOR, b, offset + i)
                        assert word not in mine, (k, g, step, b, word)
                        mine.add(word)
                        others.add((seed & m64, b, offset + i))  # token-wise verify
                    for i in range(k):
                        others.add((seed & m64, b * k + i, offset + 1))  # reference draft
                        others.add(((seed & m64) ^ frs.DRAFT_KEY_XOR, b * k + i, offset + 1))
            assert mine.isdisjoint(others), (k, g)


def test_runs_are_deterministic_and_requests_are_independent():
    """Same inputs, same outputs; a request's result is a function of its own rows, proposal,
    drafts and stream (seed, offset, position in the batch) only."""
    gen = torch.Generator(device=DEV).manual_seed(41)
    g, keep = 6, 3
    case = _make_case(g, K, V_RAGGED, 1.0, 1.0, gen)
    first = [x.clone() for x in _run(case)]
    assert all(torch.equal(a, b) for a, b in zip(first, _run(case)))
    other = _make_case(g, K, V_RAGGED, 1.0, 1.0, gen, trial=1)
    rows = case.logits.view(g, K + 1, -1).clone()
    others = [b for b in range(g) if b != keep]
    rows[others] = other.logits.view(g, K + 1, -1)[others]
    case.logits = rows.view(g * (K + 1), -1).contiguous()
    case.drafts[others] = other.drafts[others]
    again = _run(case)
    assert torch.equal(again[0][keep], first[0][keep]) and int(again[1][keep]) == int(
        first[1][keep]
    )


@pytest.mark.skipif(not ON_GPU, reason="statistical independence over thousands of steps: GPU")
def test_consecutive_steps_are_independent():
    """Lossless across steps: the token drawn after a full rejection (tau = 0) must not predict
    whether the next step accepts its first draft token (the next window starts at word K + 1)."""
    gen = torch.Generator(device=DEV).manual_seed(31)
    g, vocab = 16, 4096
    tgt_one = _peaked_logits(K + 1, vocab, gen)
    drf_one = (tgt_one[:K] + 1.5 * torch.randn(K, vocab, generator=gen, device=DEV)).contiguous()
    tgt = tgt_one.repeat(g, 1).contiguous()
    q = torch.softmax(drf_one.double(), -1).float()
    buf = torch.zeros((g + 1, K, vocab), dtype=torch.float32, device=DEV)
    buf[:g] = q
    temps = torch.ones(g * (K + 1), device=DEV)
    slots = torch.arange(g, dtype=torch.int64, device=DEV)
    seeds = torch.full((g,), 11, dtype=torch.int64, device=DEV)
    q_all = q.repeat(g, 1).contiguous()
    rows = []
    for t in range(2500):
        drafts = torch.multinomial(q_all, 1, generator=gen).view(g, K).int().contiguous()
        offs = torch.full((g,), (K + 1) * t, dtype=torch.int64, device=DEV)
        acc, num = fbv.fast_rs_verify_block(tgt, temps, drafts, slots, -1, buf, seeds, offs)
        rows.append((acc[:, 0].cpu().numpy(), num.cpu().numpy()))
    p0 = torch.softmax(tgt_one[0].double(), -1).cpu().numpy()
    resid = np.maximum(p0 - q[0].double().cpu().numpy(), 0.0)
    median = int(np.searchsorted(np.cumsum(resid) / resid.sum(), 0.5))
    pairs = [
        (tok[b] <= median, nxt[b] >= 2)
        for (tok, num), (_, nxt) in zip(rows[:-1], rows[1:])
        for b in range(g)
        if num[b] == 1
    ]
    table = np.zeros((2, 2))
    for a, b in pairs:
        table[int(a), int(b)] += 1
    expected = np.outer(table.sum(1), table.sum(0)) / table.sum()
    stat = float(((table - expected) ** 2 / expected).sum())
    pv = math.erfc(math.sqrt(stat / 2))
    print(f"{len(pairs)} (residual token, next accept) pairs: chi2={stat:.1f} p={pv:.2e}")
    assert pv > 1e-3


def _capture(fn):
    """Warm ``fn`` up on a side stream, then capture it; returns (graph, outputs)."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = fn()
    return graph, outs


@pytest.mark.skipif(not ON_GPU, reason="CUDA graphs")
def test_cuda_graph_replay_matches_eager():
    gen = torch.Generator(device=DEV).manual_seed(9)
    case = _make_case(8, K, V_BIG, 0.5, 1.0, gen)
    static_logits = case.logits.clone()
    case.logits = static_logits
    eager = [x.clone() for x in _run(case)]
    graph, outs = _capture(lambda: _run(case))
    graph.replay()
    torch.cuda.synchronize()
    for a, b in zip(eager, outs):
        assert torch.equal(a, b)
    other = _make_case(8, K, V_BIG, 0.5, 1.0, gen, trial=1)
    static_logits.copy_(other.logits)
    for name in ("buf", "drafts", "slots", "seeds", "offs"):
        getattr(case, name).copy_(getattr(other, name))
    graph.replay()
    torch.cuda.synchronize()
    replay = [x.clone() for x in outs]
    for a, b in zip(_run(case), replay):
        assert torch.equal(a, b)


@pytest.mark.skipif(not ON_GPU, reason="production vocabulary, GPU")
def test_matches_float64_model_at_the_production_vocabulary():
    gen = torch.Generator(device=DEV).manual_seed(5)
    gaps, total = _Gaps(), 0
    for trial in range(4):
        case = _make_case(8, K, V_BIG, 0.8, 1.0, gen, trial, dummy=trial % 2 == 1)
        acc, num = _run(case)
        p, q, d = _model_inputs(case)
        _check_rows(
            _as_np(acc, np.int64),
            _as_np(num, np.int64),
            p,
            q,
            d,
            case.seeds_py,
            case.offs_py,
            K,
            gaps,
        )
        total += 8
    gaps.check(total)


def test_a_token_the_proposal_excludes_has_the_limit_weight():
    """A drawn token has proposal probability > 0. For a stale or mismatched row that excludes
    it the weight is the limit of min(w * p / q, 1) as q goes to 0: 1 where the target allows the
    token and the prefix is possible, 0 otherwise; an impossible prefix stays impossible, so a
    block never emits a token the target excludes."""
    gen = torch.Generator(device=DEV).manual_seed(51)
    g = 6
    case = _make_case(g, K, V_ONE, 0.4, 1.0, gen)
    d = _as_np(case.drafts, np.int64)
    slots = _as_np(case.slots, np.int64)
    rows = case.logits.view(g, K + 1, V_ONE)
    case.buf[slots[0], 1, d[0, 1]] = 0.0  # excluded by q, allowed by p: weight 1
    rows[1, 0, d[1, 0]] = float("-inf")  # excluded by both: weight 0 ...
    case.buf[slots[1], 0, d[1, 0]] = 0.0
    case.buf[slots[1], 1, d[1, 1]] = 0.0  # ... and it stays 0 where q excludes the next token
    rows[2, 0, d[2, 0]] = float("-inf")  # excluded by p only: the ordinary weight 0
    acc, num = _run(case)
    p, q, dm = _model_inputs(case)
    _, w = block_accept_probs(p, q, dm)
    assert w[0, 2] == 1.0  # the model's weight where q excludes d_1 and p allows it
    assert w[1].tolist()[1:3] == [0.0, 0.0] and w[2, 1] == 0.0
    gaps = _Gaps()
    _check_rows(
        _as_np(acc, np.int64), _as_np(num, np.int64), p, q, dm, case.seeds_py, case.offs_py, K, gaps
    )
    gaps.check(g)
    for b in (1, 2):  # no sub-block can be accepted; the correction token has target mass
        assert int(num[b]) == 1, (b, int(num[b]))
        assert float(p[b, 0, int(acc[b, 0])]) > 0.0, b


def test_a_correction_row_without_mass_emits_token_zero():
    """A mismatched proposal set q = 2 p: every sub-block but the full one has acceptance
    probability 0 and the residual of tau = 0 has no mass. The kernel emits token 0 (the
    token-wise kernel's fallback) instead of an invalid token, and a full block keeps its drafts."""
    gen = torch.Generator(device=DEV).manual_seed(52)
    g, k = 24, 3
    case = _make_case(g, k, V_ONE, 0.0, 1.0, gen)
    p_all = torch.softmax(case.logits.view(g, k + 1, V_ONE).double(), -1)
    case.buf[case.slots, :k] = (2.0 * p_all[:, :k]).float()
    case.drafts = p_all[:, :k].argmax(-1).to(torch.int32).contiguous()
    acc, num = _run(case)
    tau = _as_np(num, np.int64) - 1
    assert set(tau.tolist()) <= {0, k}
    for b in range(g):
        if tau[b] == 0:
            assert int(acc[b, 0]) == 0 and all(int(x) == -1 for x in acc[b, 1:])
        else:
            assert (
                torch.equal(acc[b, :k].cpu(), case.drafts[b].cpu()) and 0 <= int(acc[b, k]) < V_ONE
            )


def test_fused_markov_drafts_verified_by_block_verification_follow_the_target():
    """The production pairing end to end: drafts drawn by the fused Markov-feedback chain (which
    stores its proposals), verified by block verification on the same seeds and offsets, then
    completed by the target, have the target's joint distribution."""
    vocab, k, rank = 3, 3, 2
    n = 24000 if ON_GPU else 400
    gen = torch.Generator(device=DEV).manual_seed(24)
    rng = np.random.default_rng(25)
    target = TreeModel(vocab, k, 5, 1.0)
    base = torch.randn(n, k, vocab, generator=gen, device=DEV)
    w1 = torch.randn(vocab, rank, generator=gen, device=DEV)
    w2 = torch.randn(vocab, rank, generator=gen, device=DEV)
    anchors = torch.randint(0, vocab, (n,), generator=gen, device=DEV)
    slots = torch.arange(n, dtype=torch.int64, device=DEV)
    buf = torch.zeros((n + 1, K, vocab), dtype=torch.float32, device=DEV)
    seeds = torch.full((n,), 4242, dtype=torch.int64, device=DEV)  # one seed for every request
    offs = torch.zeros(n, dtype=torch.int64, device=DEV)
    drafts = frs.fast_rs_draft_markov_feedback(
        base, anchors, w1, w2, torch.ones(n, device=DEV), slots, -1, buf, seeds, offs, 1
    )
    x = _as_np(drafts, np.int64)
    p = np.stack([target.dist(x[:, :i]) for i in range(k + 1)], axis=1)
    logits = torch.from_numpy(np.log(p)).float().reshape(n * (k + 1), vocab).to(DEV).contiguous()
    acc, num = fbv.fast_rs_verify_block(
        logits, torch.ones(n * (k + 1), device=DEV), drafts, slots, -1, buf, seeds, offs
    )
    out, tau = _as_np(acc, np.int64), _as_np(num, np.int64) - 1
    for j in range(1, k + 1):  # complete the unverified tail with the target
        todo = np.flatnonzero(tau < j)
        dist = target.dist(out[todo, :j])
        out[todo, j] = np.minimum(
            (rng.random((todo.size, 1)) > np.cumsum(dist, -1)).sum(-1), vocab - 1
        )
    counts = np.bincount(
        np.ravel_multi_index(out.T, (vocab,) * (k + 1)), minlength=vocab ** (k + 1)
    )
    stat, dof, pv = _chi2_pvalue(counts, target_joint(target, k).reshape(-1))
    print(
        f"fused chain + block verification, {n} blocks vs target: chi2={stat:.1f} dof={dof} p={pv:.3f}"
    )
    assert pv > (1e-3 if ON_GPU else 1e-4)


# --- the SpecWorkerBase switch ---------------------------------------------------------------
def _worker_classes():
    itf = pytest.importorskip("tensorrt_llm._torch.speculative.interface")
    mode = pytest.importorskip("tensorrt_llm.llmapi.llm_args").AdvancedSamplingMode

    class Worker(itf.SpecWorkerBase):
        def __init__(self):
            super().__init__()
            self.mapping = None

        @property
        def max_draft_len(self):
            return K

        def _forward_impl(self, *args, **kwargs):
            raise NotImplementedError

    return itf, mode, Worker


def _worker_metadata(itf, mode_enum, mode, case, mbs):
    vocab = case.vocab
    md = itf.SpecMetadata(max_num_requests=mbs, max_draft_len=K, max_total_draft_tokens=K)
    md.runtime_draft_len = K
    md.vocab_size = vocab
    md.is_all_greedy_sample = False
    md.advanced_sampling_mode = getattr(mode_enum, mode)
    md.use_rejection_sampling = True
    rows = case.g * (K + 1)
    md.draft_probs = case.buf
    md.draft_probs_vocab_size = vocab
    md.draft_probs_last_dim = vocab
    md.dummy_slot_row = case.buf.shape[0] - 1
    md.batch_slot_ids = case.slots
    md.temperatures = case.temps
    md.top_ps = torch.ones(rows, device=DEV)
    md.request_seeds = case.seeds
    md.request_offsets = case.offs
    return md


@pytest.mark.parametrize(
    "fast_rs,block,mode,expected",
    [
        ("1", "0", "NO_TOPK_NO_TOPP", "token"),
        ("1", "1", "NO_TOPK_NO_TOPP", "block"),
        ("1", "true", "NO_TOPK_NO_TOPP", "token"),
        ("1", "1", "NO_TOPK", "nucleus"),
        ("1", "0", "NO_TOPK", "nucleus"),
    ],
)
def test_worker_switch(monkeypatch, fast_rs, block, mode, expected):
    """TLLM_SPEC_BLOCK_VERIFY=1 swaps the temperature-only fused verifier for block verification;
    the top-p nucleus verifier is never replaced."""
    itf, mode_enum, worker_cls = _worker_classes()
    monkeypatch.setenv(frs.FAST_RS_ENV_VAR, fast_rs)
    monkeypatch.setenv(fbv.BLOCK_VERIFY_ENV_VAR, block)
    if not ON_GPU:  # fast_rs_supported demands CUDA tensors; the interpreter runs on the CPU
        monkeypatch.setattr(itf, "fast_rs_supported", lambda *a, **k: True)
    gen = torch.Generator(device=DEV).manual_seed(61)
    g = 4
    case = _make_case(g, K, V_RAGGED, 1.0, 1.0, gen)
    calls = []

    def spy(name):
        def run(*args):
            calls.append(name)
            return torch.zeros((g, K + 1), dtype=torch.int32, device=DEV), torch.ones(
                g, dtype=torch.int32, device=DEV
            )

        return run

    monkeypatch.setattr(itf, "fast_rs_verify", spy("token"))
    monkeypatch.setattr(itf, "fast_rs_verify_block", spy("block"))
    monkeypatch.setattr(itf, "fast_topp_rs_verify", spy("nucleus"))
    md = _worker_metadata(itf, mode_enum, mode, case, mbs=g)
    worker_cls()._accept_draft_tokens(case.logits, case.drafts.long(), 0, g, md)
    assert calls == [expected], calls


def test_worker_block_verify_equals_the_kernel(monkeypatch):
    """Through SpecWorkerBase the switch changes only the verifier: with the flag the result is
    exactly ``fast_rs_verify_block``'s, without it ``fast_rs_verify``'s."""
    itf, mode_enum, worker_cls = _worker_classes()
    monkeypatch.setenv(frs.FAST_RS_ENV_VAR, "1")
    if not ON_GPU:
        monkeypatch.setattr(itf, "fast_rs_supported", lambda *a, **k: True)
    gen = torch.Generator(device=DEV).manual_seed(62)
    g = 6
    differ = 0
    for trial in range(max(TRIALS, 3)):
        case = _make_case(g, K, V_RAGGED, 1.0, 1.0, gen, trial, dummy=True)
        md = _worker_metadata(itf, mode_enum, "NO_TOPK_NO_TOPP", case, mbs=g)
        out = {}
        for flag, verify in (("0", frs.fast_rs_verify), ("1", fbv.fast_rs_verify_block)):
            monkeypatch.setenv(fbv.BLOCK_VERIFY_ENV_VAR, flag)
            acc, num = worker_cls()._accept_draft_tokens(case.logits, case.drafts.long(), 0, g, md)
            want_acc, want_num = _run(case, verify)
            assert torch.equal(acc, want_acc) and torch.equal(num, want_num)
            out[flag] = num.clone()
        differ += int((out["0"] != out["1"]).sum())
    assert differ > 0  # the two verifiers are not the same function on this data


def test_worker_notices_are_logged_once_per_worker(monkeypatch):
    """The switch's notices: without TLLM_SPEC_FAST_RS=1 a warning once; with the top-p sampling
    mode a warning once (the nucleus verifier stays); in the temperature mode the info line says
    block verification, and only when the flag is set."""
    itf, mode_enum, worker_cls = _worker_classes()
    logged = []
    for level in ("warning", "info"):
        monkeypatch.setattr(
            itf.logger, level, lambda msg, *a, _level=level, **k: logged.append((_level, str(msg)))
        )
    if not ON_GPU:
        monkeypatch.setattr(itf, "fast_rs_supported", lambda *a, **k: True)
    gen = torch.Generator(device=DEV).manual_seed(63)
    case = _make_case(2, K, V_ONE, 1.0, 1.0, gen)

    def run(fast_rs, block, mode):
        logged.clear()
        monkeypatch.setenv(frs.FAST_RS_ENV_VAR, fast_rs)
        monkeypatch.setenv(fbv.BLOCK_VERIFY_ENV_VAR, block)
        md = _worker_metadata(itf, mode_enum, mode, case, mbs=2)
        worker = worker_cls()
        applies = [worker._fast_rs_applies(md, case.logits, V_ONE, site="verify") for _ in range(3)]
        return applies, list(logged)

    applies, msgs = run("0", "1", "NO_TOPK_NO_TOPP")
    assert applies == [False] * 3
    assert [lvl for lvl, m in msgs if "TLLM_SPEC_BLOCK_VERIFY=1 needs" in m] == ["warning"]
    applies, msgs = run("0", "0", "NO_TOPK_NO_TOPP")
    assert applies == [False] * 3 and not [m for _, m in msgs if "BLOCK_VERIFY" in m]
    applies, msgs = run("1", "1", "NO_TOPK")
    assert applies == [True] * 3
    assert [lvl for lvl, m in msgs if "does not cover top-p" in m] == ["warning"]
    assert not [m for _, m in msgs if "block verification kernels" in m]
    applies, msgs = run("1", "1", "NO_TOPK_NO_TOPP")
    assert applies == [True] * 3
    assert [lvl for lvl, m in msgs if "block verification kernels" in m] == ["info"]
    assert not [m for lvl, m in msgs if lvl == "warning"]
    applies, msgs = run("1", "0", "NO_TOPK_NO_TOPP")
    assert applies == [True] * 3
    assert [lvl for lvl, m in msgs if "temperature kernels" in m] == ["info"]
    assert not [m for _, m in msgs if "block verification" in m]


@pytest.mark.skipif(
    os.environ.get(fbv.BLOCK_VERIFY_ENV_VAR) is not None, reason="env forces the flag"
)
def test_flag_defaults_off(monkeypatch):
    assert not fbv.block_verify_enabled()
    for value, want in (("1", True), ("0", False), ("true", False), ("", False)):
        monkeypatch.setenv(fbv.BLOCK_VERIFY_ENV_VAR, value)
        assert fbv.block_verify_enabled() is want, value
