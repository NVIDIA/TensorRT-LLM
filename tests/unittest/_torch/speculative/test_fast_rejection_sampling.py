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
"""The fused rejection-sampling kernels (TLLM_SPEC_FAST_RS) against the flashinfer reference path.

Exactness, with the reference's Philox seed / offset:
  * draft block sampler vs ``sampling_batch_spec_dec_one_model_for_rejection``;
  * verify vs ``rejection_sampling_one_model`` (``chain_speculative_sampling``);
  * the SpecWorkerBase entry points with the flag on and off;
  * the sampled-token Markov chain vs a float64 model.
Both sides draw the same uniforms, so the decisions agree except where float32 rounding of a
probability or of a running sum decides: every disagreement must sit within ``_CDF_TOL`` of an
inverse-CDF boundary (or ``_ACCEPT_TOL`` of the accept threshold), checked with float64 sums of
the reference's own probabilities and the cuRAND draw recomputed here.

Losslessness: chi-square of the emitted and the draft tokens against their distributions.
"""

import math
import os
import types

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.pyexecutor.sampler.ops.flashinfer import (
    compute_probs_from_logits,
    sampling_batch_spec_dec_one_model_for_rejection,
)
from tensorrt_llm._torch.speculative import fast_rejection_sampling as frs
from tensorrt_llm._torch.speculative import interface as itf
from tensorrt_llm._torch.speculative.interface import rejection_sampling_one_model
from tensorrt_llm.llmapi.llm_args import AdvancedSamplingMode

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

DEV = "cuda"
V = 163840
K = 5
# A token may differ from the float64 model only when the recomputed draw lies within `_CDF_TOL` (of
# the row mass) of an inverse-CDF boundary, and an accept decision only when the draw is within
# `_ACCEPT_TOL` (relative) of the accept threshold. A float32 running sum over V terms is off by a few
# ulps of 1.0 per term group; the tail tokens the boundary falls between carry ~1e-8 each, so a
# disagreement there is decided by rounding.
_CDF_TOL = 2e-5
_ACCEPT_TOL = 1e-5

_M32 = 0xFFFFFFFF


# The float64 models recompute the kernels' cuRAND Philox draws (layout in the comment above
# `frs.DRAFT_KEY_XOR`): request b's verify uses (seed[b], subsequence b, offset[b]), draw i for accept
# test i and draw K for the residual or bonus token; its draft tokens are draw 0 of
# (seed[b] ^ DRAFT_KEY_XOR, subsequence b * K + i, offset[b] + 1).
def _philox4x32_10(ctr, key):
    c0, c1, c2, c3 = ctr
    k0, k1 = key
    for r in range(10):
        p0 = 0xD2511F53 * c0
        p1 = 0xCD9E8D57 * c2
        c0, c1, c2, c3 = (
            ((p1 >> 32) ^ c1 ^ k0) & _M32,
            p1 & _M32,
            ((p0 >> 32) ^ c3 ^ k1) & _M32,
            p0 & _M32,
        )
        if r < 9:
            k0 = (k0 + 0x9E3779B9) & _M32
            k1 = (k1 + 0xBB67AE85) & _M32
    return c0, c1, c2, c3


def curand_uniform(seed, offset, subsequence, draw):
    """Draw ``draw`` of ``curand_init(seed, subsequence, offset)`` + ``curand_uniform``."""
    seed &= (1 << 64) - 1
    g = (offset & 3) + draw
    ctr = (offset >> 2) + (g >> 2)
    out = _philox4x32_10(
        (ctr & _M32, (ctr >> 32) & _M32, subsequence & _M32, (subsequence >> 32) & _M32),
        (seed & _M32, (seed >> 32) & _M32),
    )
    x = np.float32(out[g & 3])
    return float(np.float32(x * np.float32(2.0**-32) + np.float32(2.0**-33)))


def _cdf_gap(weights, u, a, b):
    """Smallest delta for which ``u`` lies within delta of both tokens' inverse-CDF intervals
    (float64 running sum of ``weights``)."""
    c = np.cumsum(np.asarray(weights, dtype=np.float64))

    def dist(t):
        lo = c[t - 1] if t > 0 else 0.0
        hi = c[t]
        if lo <= u < hi:
            return 0.0
        return lo - u if u < lo else u - hi

    return max(dist(a), dist(b))


def peaked_logits(rows, vocab, gen, spread=2.0, ncand=48, top=14.0):
    base = torch.randn(rows, vocab, generator=gen, device=DEV) * spread
    idx = torch.randint(0, vocab, (rows, ncand), generator=gen, device=DEV)
    boost = torch.linspace(top, top - 8.0, ncand, device=DEV).unsqueeze(0).expand(rows, -1)
    base.scatter_add_(1, idx, boost + torch.randn(rows, ncand, generator=gen, device=DEV))
    return base.float().contiguous()


def _correlated_case(num_gens, gen, accept_bias):
    """Target logits [G*(K+1), V] and draft logits [G, K, V] that agree often."""
    tgt = peaked_logits(num_gens * (K + 1), V, gen)
    noise = torch.randn(num_gens, K, V, generator=gen, device=DEV) * accept_bias
    drf = (tgt.view(num_gens, K + 1, V)[:, :K] + noise).contiguous()
    return tgt, drf


def _verify_disagreement(row_args, acc_f, num_f, acc_r, num_r):
    """Size of the rounding that explains one verify row differing between the two paths: the
    relative distance of the accept test to its threshold, or the residual inverse-CDF gap."""
    tprobs, stored, drafts, seed, offset, row = row_args
    pos_f, pos_r = int(num_f) - 1, int(num_r) - 1
    if pos_f != pos_r:
        i = min(pos_f, pos_r)
        d = int(drafts[i])
        p = float(tprobs[i, d])
        q = float(stored[i, d])
        u = curand_uniform(seed, offset, row, i)
        return abs(u * q - p) / max(p, 1e-30), "accept"
    p = tprobs[pos_f].double().cpu().numpy()
    q = stored[pos_f].double().cpu().numpy() if pos_f < K else np.zeros_like(p)
    res = np.maximum(p - q, 0.0)
    total = float(res.sum())
    u = curand_uniform(seed, offset, row, K + int(pos_f < K)) * total
    return _cdf_gap(res, u, int(acc_f[pos_f]), int(acc_r[pos_f])) / max(total, 1e-30), "residual"


@pytest.mark.parametrize("num_gens", [1, 16])
@pytest.mark.parametrize("temperature", [1.0, 0.6])
def test_draft_block_matches_flashinfer(num_gens, temperature):
    """The draft block sampler follows flashinfer's path on the draft key: the stored proposals equal
    flashinfer's softmax and the drawn tokens equal `sampling_from_probs` on the same draws, except for
    tokens within `_CDF_TOL` of an inverse-CDF boundary."""
    gen = torch.Generator(device=DEV).manual_seed(11 + num_gens)
    mism = total = 0
    worst_gap = worst_rel = 0.0
    for trial in range(40):
        logits = peaked_logits(num_gens * K, V, gen)
        temps = torch.full((num_gens,), temperature, dtype=torch.float32, device=DEV)
        seed = 1234 + trial
        seeds = torch.full((num_gens,), seed, dtype=torch.int64, device=DEV)
        offs = torch.full((num_gens,), 6 * trial + trial % 4, dtype=torch.int64, device=DEV)
        ref_tok, ref_probs = sampling_batch_spec_dec_one_model_for_rejection(
            logits,
            temps.repeat_interleave(K),
            None,
            None,
            seed=seeds.repeat_interleave(K),
            offset=(offs + 1).repeat_interleave(K),
        )
        slots = torch.randperm(17, device=DEV)[:num_gens].to(torch.int64)
        buf = torch.zeros((18, K, V), dtype=torch.float32, device=DEV)
        tok = frs.fast_rs_draft_block(logits, temps, slots, -1, buf, seeds, offs, 1)
        stored = buf[slots].reshape(num_gens * K, V)
        big = ref_probs > 1e-6
        rel = ((stored - ref_probs).abs() / ref_probs.clamp_min(1e-30))[big]
        worst_rel = max(worst_rel, float(rel.max()))
        for r in (tok.long() != ref_tok.long()).nonzero().flatten().tolist():
            u = curand_uniform(seed, int(offs[0]) + 1, r, 0)
            gap = _cdf_gap(ref_probs[r].double().cpu().numpy(), u, int(tok[r]), int(ref_tok[r]))
            worst_gap = max(worst_gap, gap)
            mism += 1
        total += tok.numel()
    print(
        f"draft block G={num_gens} T={temperature}: {mism}/{total} tokens differ, all within "
        f"{worst_gap:.2e} of a CDF boundary; max rel prob diff {worst_rel:.2e}"
    )
    assert worst_gap <= _CDF_TOL
    assert mism <= 0.02 * total
    assert worst_rel < 1e-5


@pytest.mark.parametrize("num_gens", [1, 16])
@pytest.mark.parametrize("accept_bias", [0.3, 1.5])
def test_verify_matches_flashinfer(num_gens, accept_bias):
    """The verify accept decisions match `chain_speculative_sampling` on the same draws ``0..K-1``, and
    every row's accept position and drawn token (draw ``K``) match the float64 model. Differences must
    be explained by rounding."""
    gen = torch.Generator(device=DEV).manual_seed(101 + num_gens)
    mism = total = 0
    worst = {"accept": 0.0, "residual": 0.0}
    hist = np.zeros(K + 1, dtype=np.int64)
    temps_req = torch.ones(num_gens, dtype=torch.float32, device=DEV)
    temps_tok = torch.ones(num_gens * (K + 1), dtype=torch.float32, device=DEV)
    for trial in range(60):
        tgt, drf = _correlated_case(num_gens, gen, accept_bias)
        seed, offset = 4321 + trial, 6 * trial + (trial % 4)
        seeds = torch.full((num_gens,), seed, dtype=torch.int64, device=DEV)
        offs = torch.full((num_gens,), offset, dtype=torch.int64, device=DEV)
        slots = torch.randperm(17, device=DEV)[:num_gens].to(torch.int64)
        buf = torch.zeros((18, K, V), dtype=torch.float32, device=DEV)
        drafts = frs.fast_rs_draft_block(drf.view(-1, V), temps_req, slots, -1, buf, seeds, offs, 1)
        drafts = drafts.view(num_gens, K).contiguous()
        # What _sample_and_accept_draft_tokens_rejection does in NO_TOPK_NO_TOPP mode.
        tprobs = compute_probs_from_logits(tgt, temps_tok, None, None).view(num_gens, K + 1, V)
        ref_acc, ref_num = rejection_sampling_one_model(
            buf[slots], drafts, tprobs, deterministic=True, seed=seeds, offset=offs
        )
        acc, num = frs.fast_rs_verify(tgt, temps_tok, drafts, slots, -1, buf, seeds, offs)
        bad = (acc.long() != ref_acc.long()).any(dim=1) | (num.long() != ref_num.long())
        for b in bad.nonzero().flatten().tolist():
            row_args = (tprobs[b], buf[slots[b]], drafts[b].tolist(), seed, offset, b)
            size, kind = _verify_disagreement(row_args, acc[b], num[b], ref_acc[b], ref_num[b])
            worst[kind] = max(worst[kind], size)
            mism += 1
        total += num_gens
        for n in num.tolist():
            hist[n - 1] += 1
    print(
        f"verify G={num_gens} bias={accept_bias}: {mism}/{total} rows differ (worst accept "
        f"{worst['accept']:.2e} rel, residual {worst['residual']:.2e}); accepted-length "
        f"histogram {hist.tolist()}"
    )
    assert worst["accept"] <= _ACCEPT_TOL and worst["residual"] <= _CDF_TOL
    assert mism <= max(2, int(0.02 * total))


class _Worker(itf.SpecWorkerBase):
    """A `SpecWorkerBase` without a model, to call its draft and verify entry points."""

    def __init__(self):
        super().__init__()
        self.mapping = None

    @property
    def max_draft_len(self):
        return K

    def _forward_impl(self, *args, **kwargs):
        raise NotImplementedError


def _worker_metadata(mbs, slots, seed, offset):
    """Spec metadata of ``mbs`` requests in ``NO_TOPK_NO_TOPP`` mode with rejection-sampling buffers,
    temperature 1 and one seed and offset for every request."""
    md = itf.SpecMetadata(max_num_requests=mbs, max_draft_len=K, max_total_draft_tokens=K)
    md.runtime_draft_len = K
    md.vocab_size = V
    md.is_all_greedy_sample = False
    md.advanced_sampling_mode = AdvancedSamplingMode.NO_TOPK_NO_TOPP
    md.use_rejection_sampling = True
    md.prepare_rejection_sampling_buffers()
    rows = mbs * (K + 1)
    md.temperatures = torch.ones(rows, device=DEV)
    md.top_ks = torch.full((rows,), 2**31 - 1, dtype=torch.int32, device=DEV)
    md.top_ps = torch.ones(rows, device=DEV)
    md.request_temperatures = torch.ones(mbs, device=DEV)
    md.request_top_ks = torch.full((mbs,), 2**31 - 1, dtype=torch.int32, device=DEV)
    md.request_top_ps = torch.ones(mbs, device=DEV)
    md.request_seeds = torch.full((mbs,), seed, dtype=torch.int64, device=DEV)
    md.request_offsets = torch.full((mbs,), offset, dtype=torch.int64, device=DEV)
    md.seeds = torch.full((rows,), seed, dtype=torch.int64, device=DEV)
    md.offsets = torch.full((rows,), offset, dtype=torch.int64, device=DEV)
    slots = list(slots)
    slots[-1] = md.dummy_slot_row  # the last generation request is CUDA-graph padding
    md.batch_slot_ids[: len(slots)] = torch.tensor(slots, device=DEV)
    md.draft_probs_last_dim = V
    md.draft_probs.zero_()
    md.draft_probs[:, :, 0] = 1.0
    return md


def test_worker_paths_fast_vs_reference(monkeypatch):
    """SpecWorkerBase entry points with the flag on and off (NO_TOPK_NO_TOPP): a mixed batch
    (2 context rows) and a padding request routed to the dummy slot."""
    mbs, nctx, ngen, seed, offset = 16, 2, 9, 99, 13
    bsz = nctx + ngen
    slots = range(3, 3 + bsz)
    gen = torch.Generator(device=DEV).manual_seed(7)
    tgt_gen, drf = _correlated_case(ngen, gen, 0.5)
    logits = torch.cat([peaked_logits(nctx, V, gen), tgt_gen], dim=0).contiguous()
    drafts, probs = {}, {}
    for flag in ("0", "1"):
        monkeypatch.setenv(frs.FAST_RS_ENV_VAR, flag)
        md = _worker_metadata(mbs, slots, seed, offset)
        out = _Worker().sample_draft_tokens(drf, md, bsz, num_contexts=nctx)
        drafts[flag] = out.view(ngen, K).clone()
        probs[flag] = md.draft_probs.clone()
    real = slice(3 + nctx, 2 + bsz)  # generation slots without the padding request
    assert float((probs["0"][real] - probs["1"][real]).abs().max()) < 1e-6
    mism = 0
    for b in range(ngen - 1):
        for i in range(K):
            t0, t1 = int(drafts["0"][b, i]), int(drafts["1"][b, i])
            if t0 != t1:
                u = curand_uniform(seed, offset + 1, b * K + i, 0)
                weights = probs["0"][3 + nctx + b, i].double().cpu().numpy()
                assert _cdf_gap(weights, u, t0, t1) <= _CDF_TOL
                mism += 1
    # Verify both paths against the same drafts and stored proposals (the reference's).
    out = {}
    for flag in ("0", "1"):
        monkeypatch.setenv(frs.FAST_RS_ENV_VAR, flag)
        md = _worker_metadata(mbs, slots, seed, offset)
        md.draft_probs.copy_(probs["0"])
        md.draft_tokens = drafts["0"].reshape(-1)
        acc, num = _Worker()._accept_draft_tokens(logits, drafts["0"], nctx, bsz, md)
        out[flag] = (acc.clone(), num.clone())
    (a0, n0), (a1, n1) = out["0"], out["1"]
    print(f"worker: {mism} draft disagreements (all at CDF boundaries); counts {n1.tolist()}")
    assert torch.equal(a0[:nctx, 0], a1[:nctx, 0])
    assert torch.equal(a0[nctx:-1], a1[nctx:-1]) and torch.equal(n0[:-1], n1[:-1])


def test_markov_feedback_matches_float64_model():
    """The chain advances on the sampled token: base_i + w2 @ w1[prev], prev_0 = anchor; position
    i of request b draws Philox subsequence b * K + i, draw 0, at offset + 1."""
    gen = torch.Generator(device=DEV).manual_seed(5)
    num_gens, rank = 4, 256
    checked = mism = 0
    worst = 0.0
    for trial in range(6):
        base = peaked_logits(num_gens * K, V, gen).view(num_gens, K, V).contiguous()
        w1 = (torch.randn(V, rank, generator=gen, device=DEV) * 0.2).to(torch.bfloat16)
        w2 = (torch.randn(V, rank, generator=gen, device=DEV) * 0.2).to(torch.bfloat16)
        anchor = torch.randint(0, V, (num_gens,), generator=gen, device=DEV)
        seed, offset = 555 + trial, 6 * trial
        tok = frs.fast_rs_draft_markov_feedback(
            base,
            anchor,
            w1,
            w2,
            torch.ones(num_gens, device=DEV),
            torch.arange(num_gens, dtype=torch.int64, device=DEV),
            -1,
            torch.zeros((8, K, V), dtype=torch.float32, device=DEV),
            torch.tensor([seed], dtype=torch.int64, device=DEV),
            torch.tensor([offset], dtype=torch.int64, device=DEV),
            1,
        )
        # Same batched bias GEMM as the kernel path (F.linear of the [G, rank] embeddings).
        prev = anchor.clone()
        live = [True] * num_gens
        for i in range(K):
            bias = torch.nn.functional.linear(w1[prev], w2).float()
            q = torch.softmax((base[:, i] + bias).double(), dim=-1).cpu().numpy()
            nxt = prev.clone()
            for b in range(num_gens):
                if not live[b]:
                    continue
                u = curand_uniform(seed, offset + 1, b * K + i, 0)
                ref_tok = int(np.searchsorted(np.cumsum(q[b]), u, side="right"))
                checked += 1
                if ref_tok != int(tok[b, i]):
                    # The chains condition on different tokens from here on.
                    worst = max(worst, _cdf_gap(q[b], u, ref_tok, int(tok[b, i])))
                    mism += 1
                    live[b] = False
                nxt[b] = ref_tok
            prev = nxt
    print(f"markov feedback: {mism}/{checked} disagreements, worst CDF gap {worst:.2e}")
    assert worst <= _CDF_TOL


def _small_support_logits(rows, gen, support=96):
    """Rows whose mass sits on ``support`` tokens (others at -30) so chi-square has power."""
    lg = torch.full((rows, V), -30.0, device=DEV)
    idx = torch.stack([torch.randperm(V, generator=gen, device=DEV)[:support] for _ in range(rows)])
    lg.scatter_(1, idx, torch.randn(rows, support, generator=gen, device=DEV) * 1.2)
    return lg.contiguous()


def _chi2_pvalue(tokens, probs, min_expected=20.0):
    """Chi-square goodness-of-fit test of ``tokens`` against ``probs``.

    Tokens expected fewer than ``min_expected`` times share one bin, which is dropped when it is also
    too small. Returns: (statistic, degrees of freedom, p-value from the Wilson-Hilferty approximation).
    """
    n = len(tokens)
    keep = [int(i) for i in np.argsort(-probs) if probs[i] * n >= min_expected]
    obs = np.array([np.sum(tokens == i) for i in keep] + [0], dtype=np.float64)
    exp = np.array([probs[i] * n for i in keep] + [0.0], dtype=np.float64)
    obs[-1] = n - obs[:-1].sum()
    exp[-1] = n - exp[:-1].sum()
    if exp[-1] < min_expected:
        obs, exp = obs[:-1], exp[:-1] * (obs[:-1].sum() / exp[:-1].sum())
    stat = float(((obs - exp) ** 2 / exp).sum())
    dof = len(obs) - 1
    z = ((stat / dof) ** (1 / 3) - (1 - 2 / (9 * dof))) / math.sqrt(2 / (9 * dof))
    return stat, dof, 0.5 * math.erfc(z / math.sqrt(2))  # Wilson-Hilferty


def test_chi2_emitted_tokens_follow_target():
    """P(first emitted = x) = p_0(x) and P(second = y | first accepted) = p_1(y) for a proposal
    that differs from the target (rejection sampling is lossless for any proposal)."""
    gen = torch.Generator(device=DEV).manual_seed(2024)
    g = 16
    tgt_one = _small_support_logits(K + 1, gen)
    drf_one = (tgt_one[:K] + torch.randn(K, V, generator=gen, device=DEV) * 0.8).contiguous()
    p0 = torch.softmax(tgt_one[0].double(), -1).cpu().numpy()
    p1 = torch.softmax(tgt_one[1].double(), -1).cpu().numpy()
    tgt, drf = tgt_one.repeat(g, 1).contiguous(), drf_one.repeat(g, 1).contiguous()
    temps_tok, temps_req = torch.ones(g * (K + 1), device=DEV), torch.ones(g, device=DEV)
    slots = torch.arange(g, dtype=torch.int64, device=DEV)
    buf = torch.zeros((g + 1, K, V), dtype=torch.float32, device=DEV)
    first, second = [], []
    for t in range(1500):
        seeds = torch.full((g,), 7, dtype=torch.int64, device=DEV)
        offs = torch.full((g,), 6 * t, dtype=torch.int64, device=DEV)
        drafts = frs.fast_rs_draft_block(drf, temps_req, slots, -1, buf, seeds, offs, 1)
        acc, num = frs.fast_rs_verify(
            tgt, temps_tok, drafts.view(g, K), slots, -1, buf, seeds, offs
        )
        acc, num = acc.cpu().numpy(), num.cpu().numpy()
        first.append(acc[:, 0])
        second.append(acc[num >= 2, 1])
    s0, d0, pv0 = _chi2_pvalue(np.concatenate(first), p0)
    s1, d1, pv1 = _chi2_pvalue(np.concatenate(second), p1)
    print(
        f"first token chi2={s0:.1f} dof={d0} p={pv0:.3f}; second chi2={s1:.1f} dof={d1} p={pv1:.3f}"
    )
    assert pv0 > 1e-3 and pv1 > 1e-3


def test_chi2_draft_tokens_follow_stored_proposal():
    """The draft tokens at the first and last positions follow the stored proposal (chi-square), with
    the trials on consecutive offset windows."""
    gen = torch.Generator(device=DEV).manual_seed(77)
    g = 16
    drf = _small_support_logits(K, gen).repeat(g, 1).contiguous()
    temps_req = torch.full((g,), 0.8, device=DEV)
    slots = torch.arange(g, dtype=torch.int64, device=DEV)
    buf = torch.zeros((g + 1, K, V), dtype=torch.float32, device=DEV)
    toks = []
    for t in range(1500):
        seeds = torch.full((g,), 3, dtype=torch.int64, device=DEV)
        offs = torch.full((g,), 6 * t, dtype=torch.int64, device=DEV)
        out = frs.fast_rs_draft_block(drf, temps_req, slots, -1, buf, seeds, offs, 1)
        toks.append(out.view(g, K).cpu().numpy())
    toks = np.concatenate(toks)
    q = buf[0].double().cpu().numpy()
    for i in (0, K - 1):
        stat, dof, pv = _chi2_pvalue(toks[:, i], q[i] / q[i].sum())
        print(f"draft position {i}: chi2={stat:.1f} dof={dof} p={pv:.3f}")
        assert pv > 1e-3


def test_cuda_graph_replay_matches_eager():
    """Verify and draft block captured in one CUDA graph replay what eager runs compute, also with new
    logits written into the captured tensors."""
    gen = torch.Generator(device=DEV).manual_seed(9)
    g = 8
    tgt, drf = _correlated_case(g, gen, 0.5)
    temps_tok, temps_req = torch.ones(g * (K + 1), device=DEV), torch.ones(g, device=DEV)
    slots = torch.arange(g, dtype=torch.int64, device=DEV) + 2
    buf = torch.zeros((g + 3, K, V), dtype=torch.float32, device=DEV)
    seeds = torch.full((g,), 5, dtype=torch.int64, device=DEV)
    offs = torch.full((g,), 12, dtype=torch.int64, device=DEV)
    s_tgt, s_drf = tgt.clone(), drf.clone().view(-1, V)
    drafts = frs.fast_rs_draft_block(s_drf, temps_req, slots, -1, buf, seeds, offs, 1)
    drafts = drafts.view(g, K).contiguous()

    def step():
        acc, num = frs.fast_rs_verify(s_tgt, temps_tok, drafts, slots, -1, buf, seeds, offs)
        return acc, num, frs.fast_rs_draft_block(s_drf, temps_req, slots, -1, buf, seeds, offs, 1)

    snapshot = buf.clone()
    eager = [x.clone() for x in step()]
    buf.copy_(snapshot)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step()
    torch.cuda.current_stream().wait_stream(stream)
    buf.copy_(snapshot)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = step()
    buf.copy_(snapshot)
    graph.replay()
    torch.cuda.synchronize()
    for a, b in zip(eager, outs):
        assert torch.equal(a, b)
    tgt2, drf2 = _correlated_case(g, gen, 0.5)
    s_tgt.copy_(tgt2)
    s_drf.copy_(drf2.view(-1, V))
    snapshot = buf.clone()
    graph.replay()
    torch.cuda.synchronize()
    replay = [x.clone() for x in outs]
    buf.copy_(snapshot)
    for a, b in zip(step(), replay):
        assert torch.equal(a, b)


@pytest.mark.skipif(os.environ.get(frs.FAST_RS_ENV_VAR) == "1", reason="env forces the flag")
def test_flag_defaults_off():
    """The fused path is off unless ``TLLM_SPEC_FAST_RS=1``."""
    assert not frs.fast_rs_enabled()
    # Flag off: the gate decides before reading the metadata, so a stand-in without the
    # sampling-mode field (as in test_rejection_buffers_guard) keeps the reference dispatch.
    assert not _Worker()._fast_rs_applies(types.SimpleNamespace(), None, V)
