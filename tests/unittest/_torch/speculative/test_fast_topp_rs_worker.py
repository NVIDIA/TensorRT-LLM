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
"""SpecWorkerBase entry points of the fused top-p path (gate, dispatch, guards, Markov sampled feedback),
losslessness (chi-square of emitted and draft tokens, independence of consecutive steps) and CUDA graphs.
"""

import math
import os
import types

import numpy as np
import pytest
import torch
from topp_rs_test_utils import (
    _CDF_TOL,
    DEV,
    K,
    Nucleus,
    V,
    _cdf_gap,
    _check_drafts,
    _check_verify,
    _correlated_case,
    _draft_key,
    _Gaps,
    _inv_cdf,
    _request_top_ps,
    curand_uniform,
    kernel_kept,
    peaked_logits,
    support_diff_ok,
)

from tensorrt_llm._torch.speculative import fast_rejection_sampling as frs
from tensorrt_llm._torch.speculative import fast_topp_rejection_sampling as ftp
from tensorrt_llm._torch.speculative import interface as itf
from tensorrt_llm.llmapi.llm_args import AdvancedSamplingMode

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


class _Worker(itf.SpecWorkerBase):
    def __init__(self):
        super().__init__()
        self.mapping = None

    @property
    def max_draft_len(self):
        return K

    def _forward_impl(self, *args, **kwargs):
        raise NotImplementedError


def _worker_metadata(mbs, slots, seed, offset, mode=AdvancedSamplingMode.NO_TOPK, top_p=0.95):
    md = itf.SpecMetadata(max_num_requests=mbs, max_draft_len=K, max_total_draft_tokens=K)
    md.runtime_draft_len = K
    md.vocab_size = V
    md.is_all_greedy_sample = False
    md.advanced_sampling_mode = mode
    md.use_rejection_sampling = True
    md.prepare_rejection_sampling_buffers()
    rows = mbs * (K + 1)
    md.temperatures = torch.ones(rows, device=DEV)
    md.top_ks = torch.full((rows,), 2**31 - 1, dtype=torch.int32, device=DEV)
    md.top_ps = torch.full((rows,), top_p, device=DEV)
    md.request_temperatures = torch.ones(mbs, device=DEV)
    md.request_top_ks = torch.full((mbs,), 2**31 - 1, dtype=torch.int32, device=DEV)
    md.request_top_ps = torch.full((mbs,), top_p, device=DEV)
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


def test_worker_paths_fast_topp(monkeypatch):
    """SpecWorkerBase entry points in ``no_topk`` mode (top-p 0.95), flag off vs on: a mixed batch
    (2 context rows) and a padding request routed to the dummy slot."""
    mbs, nctx, ngen, seed, offset = 16, 2, 9, 99, 13
    bsz = nctx + ngen
    slots = range(3, 3 + bsz)
    real = slice(3 + nctx, 2 + bsz)  # generation slots without the padding request
    gen = torch.Generator(device=DEV).manual_seed(7)
    tgt_gen, drf = _correlated_case(ngen, gen, 0.5)
    logits = torch.cat([peaked_logits(nctx, V, gen), tgt_gen], dim=0).contiguous()
    probs = {}
    for flag in ("0", "1"):
        monkeypatch.setenv(frs.FAST_RS_ENV_VAR, flag)
        md = _worker_metadata(mbs, slots, seed, offset)
        out = _Worker().sample_draft_tokens(drf, md, bsz, num_contexts=nctx)
        probs[flag] = md.draft_probs.clone()
    drafts = out.view(ngen, K).contiguous()
    # The fused path stores flashinfer's nucleus proposals (rows may differ within rounding).
    ref, fused = probs["0"][real], probs["1"][real]
    rows_differ = ((ref > 0) != (fused > 0)).flatten(0, 1).any(1).sum()
    assert int(rows_differ) <= 1
    both = (ref > 1e-6) & (fused > 0)
    assert float(((ref - fused).abs() / ref.clamp_min(1e-30))[both].max()) < 1e-5
    q = probs["1"][real].double().cpu().numpy()
    gaps = _Gaps()
    _check_drafts(drafts[: ngen - 1].cpu().numpy(), q, [seed] * ngen, [offset] * ngen, K, gaps)
    out = {}
    for flag in ("0", "1"):
        monkeypatch.setenv(frs.FAST_RS_ENV_VAR, flag)
        md = _worker_metadata(mbs, slots, seed, offset)
        md.draft_probs.copy_(probs["1"])
        md.draft_tokens = drafts.reshape(-1)
        acc, num = _Worker()._accept_draft_tokens(logits, drafts, nctx, bsz, md)
        out[flag] = (acc.clone(), num.clone())
    (a0, n0), (a1, n1) = out["0"], out["1"]
    assert torch.equal(a0[:nctx, 0], a1[:nctx, 0])  # context rows: the same target sampler
    tgt_rows = logits[nctx:]
    temps = torch.ones(ngen * (K + 1), device=DEV)
    tps = torch.full((ngen * (K + 1),), 0.95, device=DEV)
    kept_k, rec = kernel_kept(tgt_rows, temps, tps)
    assert bool((rec[:, 3] == 0).all())
    p = Nucleus(tgt_rows, temps, tps).probs(kept_k).reshape(ngen, K + 1, V).cpu().numpy()
    real_gens = slice(nctx, bsz - 1)
    _check_verify(
        a1[real_gens].cpu().numpy(), n1[real_gens].cpu().numpy(), p[: ngen - 1], q,
        drafts[: ngen - 1].cpu().numpy(), [seed] * ngen, [offset] * ngen, K, gaps,
    )  # fmt: skip
    same = int((n0[real_gens] == n1[real_gens]).sum())
    print(
        f"worker: worst {gaps.worst}; accept counts equal to the reference in {same}/{ngen - 1}"
        f" rows; counts {n1.tolist()}"
    )
    gaps.check(ngen - 1)
    assert same >= ngen - 2


def test_worker_mixed_per_request_temperature_and_top_p(monkeypatch):
    """A batch whose requests carry different temperatures and top-ps, with two context requests ahead of the
    generation ones (they shift the per-token arrays): every request's stored proposals, drawn tokens and
    accept / residual decisions follow its own nucleus. A slicing slip between the request-level arrays of the
    draft step and the token-level arrays of the verify step would hand a request its neighbour's top-p."""
    monkeypatch.setenv(frs.FAST_RS_ENV_VAR, "1")
    mbs, nctx, ngen, seed, offset = 16, 2, 9, 71, 4
    bsz = nctx + ngen
    slots = range(3, 3 + bsz)
    real = slice(3 + nctx, 2 + bsz)  # generation slots without the padding request
    nreal = ngen - 1
    temps_req = torch.tensor(
        [1.0, 1.0, 1.0, 0.6, 1.7, 1.0, 0.8, 1.3, 0.6, 1.7, 1.0], dtype=torch.float32, device=DEV
    )
    tps_req = torch.tensor(
        [0.95, 0.95, 0.95, 1.0, 0.5, 0.9, 0.99, 0.2, 0.95, 1.0, 0.999],
        dtype=torch.float32,
        device=DEV,
    )
    assert temps_req.numel() == tps_req.numel() == bsz
    ntok = torch.tensor([1] * nctx + [K + 1] * ngen, device=DEV)  # tokens per request
    md = _worker_metadata(mbs, slots, seed, offset)
    md.request_temperatures[:bsz] = temps_req
    md.request_top_ps[:bsz] = tps_req
    md.temperatures[: int(ntok.sum())] = temps_req.repeat_interleave(ntok)
    md.top_ps[: int(ntok.sum())] = tps_req.repeat_interleave(ntok)
    temps_gen, tps_gen = temps_req[nctx:], tps_req[nctx:]
    gen = torch.Generator(device=DEV).manual_seed(29)
    tgt_gen, drf = _correlated_case(ngen, gen, 0.5)
    logits = torch.cat([peaked_logits(nctx, V, gen), tgt_gen], dim=0).contiguous()
    out = _Worker().sample_draft_tokens(drf, md, bsz, num_contexts=nctx)
    drafts = out.view(ngen, K).contiguous()
    gaps = _Gaps()
    # The stored proposals: each request's own nucleus of its own draft rows.
    stored = md.draft_probs[real].reshape(nreal * K, V)
    temps_d, tps_d = (
        temps_gen.repeat_interleave(K)[: nreal * K],
        tps_gen.repeat_interleave(K)[: nreal * K],
    )
    nuc_d = Nucleus(drf.reshape(ngen * K, V)[: nreal * K], temps_d, tps_d)
    kept_s = stored > 0
    for r in range(nreal * K):
        assert support_diff_ok(nuc_d, r, kept_s[r], nuc_d.kept[r]), (
            r,
            float(temps_d[r]),
            float(tps_d[r]),
        )
    want = nuc_d.probs(kept_s)
    big = (want > 1e-6) & kept_s
    rel = ((stored.double() - want).abs() / want.clamp_min(1e-300))[big]
    assert float(rel.max()) < 1e-5 + 6e-8 * float(nuc_d.z.abs().max()), float(rel.max())
    q = stored.double().reshape(nreal, K, V).cpu().numpy()
    _check_drafts(drafts[:nreal].cpu().numpy(), q, [seed] * ngen, [offset] * ngen, K, gaps)
    # The verify step: each request's accept / residual decisions against its own target nuclei.
    md.draft_tokens = drafts.reshape(-1)
    acc, num = _Worker()._accept_draft_tokens(logits, drafts, nctx, bsz, md)
    tgt_rows = logits[nctx:]
    temps_t, tps_t = temps_gen.repeat_interleave(K + 1), tps_gen.repeat_interleave(K + 1)
    kept_k, rec = kernel_kept(tgt_rows, temps_t, tps_t)
    assert bool((rec[:, 3] == 0).all())
    nuc_t = Nucleus(tgt_rows, temps_t, tps_t)
    for r in range(nreal * (K + 1)):
        assert support_diff_ok(nuc_t, r, kept_k[r], nuc_t.kept[r]), (
            r,
            float(temps_t[r]),
            float(tps_t[r]),
        )
    p = nuc_t.probs(kept_k).reshape(ngen, K + 1, V).cpu().numpy()
    real_gens = slice(nctx, bsz - 1)
    _check_verify(
        acc[real_gens].cpu().numpy(), num[real_gens].cpu().numpy(), p[:nreal], q,
        drafts[:nreal].cpu().numpy(), [seed] * ngen, [offset] * ngen, K, gaps,
    )  # fmt: skip
    print(f"mixed batch: worst {gaps.worst}; accepted counts {num[real_gens].tolist()}")
    gaps.check(nreal)


@pytest.mark.parametrize(
    "mode,variant",
    [
        (AdvancedSamplingMode.NO_TOPK_NO_TOPP, "temperature"),
        (AdvancedSamplingMode.NO_TOPK, "top-p"),
        (AdvancedSamplingMode.FULL, None),
        (AdvancedSamplingMode.NO_TOPP, None),
    ],
)
def test_gate_and_dispatch_follow_the_sampling_mode(monkeypatch, mode, variant):
    """The fused path covers the top-k-free modes: ``no_topk_no_topp`` runs the temperature-only
    kernels, ``no_topk`` the nucleus kernels; modes that keep top-k use the reference. The other
    disqualifiers (guided decoding, a draft-to-target map, a synthetic acceptance override,
    vocab-sharded logits) and the flag keep the reference in every mode."""
    monkeypatch.setenv(frs.FAST_RS_ENV_VAR, "1")
    calls = []
    for name in (
        "fast_rs_verify",
        "fast_topp_rs_verify",
        "fast_rs_draft_block",
        "fast_topp_rs_draft_block",
    ):
        real = getattr(itf, name)

        def spy(*args, _real=real, _name=name, **kwargs):
            calls.append(_name)
            return _real(*args, **kwargs)

        monkeypatch.setattr(itf, name, spy)
    mbs, ngen = 4, 3
    md = _worker_metadata(mbs, range(mbs), 5, 3, mode=mode)
    worker = _Worker()
    gen = torch.Generator(device=DEV).manual_seed(1)
    tgt, drf = _correlated_case(ngen, gen, 0.5)
    logits = torch.zeros((mbs * (K + 1), V), device=DEV)
    assert bool(worker._fast_rs_applies(md, logits, V)) == (variant is not None)
    drafts = worker.sample_draft_tokens(drf, md, ngen, num_contexts=0)
    drafts = drafts.view(ngen, K).contiguous()
    md.draft_tokens = drafts.reshape(-1)
    worker._accept_draft_tokens(tgt, drafts, 0, ngen, md)
    want = {
        "temperature": ["fast_rs_draft_block", "fast_rs_verify"],
        "top-p": ["fast_topp_rs_draft_block", "fast_topp_rs_verify"],
        None: [],
    }[variant]
    assert calls == want, calls
    if variant is not None:
        for attr, value in (
            ("guided_decoder", types.SimpleNamespace()),
            ("_d2t", torch.zeros(4, dtype=torch.int64)),
            ("force_num_accepted_tokens", 1.0),
        ):
            saved = getattr(worker, attr)
            setattr(worker, attr, value)
            assert not worker._fast_rs_applies(md, logits, V)
            setattr(worker, attr, saved)
        assert not worker._fast_rs_applies(md, itf.VocabShardedLogits(None, logits), V)
        assert worker._fast_rs_applies(md, logits, V)
        monkeypatch.setenv(frs.FAST_RS_ENV_VAR, "0")
        assert not worker._fast_rs_applies(md, logits, V)


@pytest.mark.skipif(os.environ.get(frs.FAST_RS_ENV_VAR) == "1", reason="env forces the flag")
def test_flag_defaults_off():
    assert not frs.fast_rs_enabled()
    # Flag off: the gate decides before reading the metadata, so a stand-in without the
    # sampling-mode field (as in test_rejection_buffers_guard) keeps the reference dispatch.
    assert not _Worker()._fast_rs_applies(types.SimpleNamespace(), None, V)


def test_markov_feedback_matches_float64_model():
    """The chain advances on the sampled token and the proposal is the top-p nucleus of
    ``softmax((base_i + w2 @ w1[prev]) / T)``, prev_0 = anchor; position i of request b draws word
    0 of (seed ^ DRAFT_KEY_XOR, subsequence b * K + i, offset + 1) from it."""
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
        temps = torch.full((num_gens,), 0.9, device=DEV)
        tps = _request_top_ps(num_gens, trial)
        buf = torch.zeros((8, K, V), dtype=torch.float32, device=DEV)
        tok = ftp.fast_topp_rs_draft_markov_feedback(
            base, anchor, w1, w2, temps, tps, torch.arange(num_gens, dtype=torch.int64, device=DEV),
            -1, buf, torch.full((num_gens,), seed, dtype=torch.int64, device=DEV),
            torch.full((num_gens,), offset, dtype=torch.int64, device=DEV), 1,
        )  # fmt: skip
        # Same batched bias GEMM as the kernel path (F.linear of the [G, rank] embeddings).
        prev = anchor.clone()
        live = [True] * num_gens
        for i in range(K):
            bias = torch.nn.functional.linear(w1[prev], w2).float()
            nuc = Nucleus(base[:, i] + bias, temps, tps)
            q = nuc.probs().cpu().numpy()
            # The stored row is the kernel's nucleus of the same logits (up to rounding).
            stored = buf[:num_gens, i]
            for b in range(num_gens):
                if live[b]:
                    assert support_diff_ok(nuc, b, stored[b] > 0, nuc.kept[b])
            nxt = prev.clone()
            for b in range(num_gens):
                if not live[b]:
                    continue
                u = curand_uniform(_draft_key(seed), offset + 1, b * K + i, 0)
                ref_tok = _inv_cdf(q[b], u)
                checked += 1
                if ref_tok != int(tok[b, i]):
                    # The chains condition on different tokens from here on.
                    worst = max(worst, _cdf_gap(q[b], u, ref_tok, int(tok[b, i])))
                    mism += 1
                    live[b] = False
                nxt[b] = ref_tok
            prev = nxt
    print(f"markov feedback (top-p): {mism}/{checked} disagreements, worst CDF gap {worst:.2e}")
    assert worst <= _CDF_TOL


# ----------------------------------------------------------------------------------------------
# Losslessness (distribution tests).
# ----------------------------------------------------------------------------------------------
def _small_support_logits(rows, gen, support=96):
    """Rows whose mass sits on ``support`` tokens (others at -30) so chi-square has power."""
    lg = torch.full((rows, V), -30.0, device=DEV)
    idx = torch.stack([torch.randperm(V, generator=gen, device=DEV)[:support] for _ in range(rows)])
    lg.scatter_(1, idx, torch.randn(rows, support, generator=gen, device=DEV) * 1.2)
    return lg.contiguous()


def _chi2_pvalue(tokens, probs, min_expected=20.0):
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


def _nucleus_probs(logits_row, top_p, temperature=1.0):
    nuc = Nucleus(
        logits_row.view(1, -1),
        torch.full((1,), temperature, device=DEV),
        torch.full((1,), top_p, device=DEV),
    )
    return nuc.probs()[0].cpu().numpy()


def test_chi2_emitted_tokens_follow_the_target_nucleus():
    """P(first emitted = x) = p~_0(x) and P(second = y | first accepted) = p~_1(y), with p~ the
    top-p nuclei of the target, for a proposal that differs from it (rejection sampling is
    lossless for any proposal); the trials are consecutive offset windows of the same requests, as
    in decoding. No token outside the target nucleus is ever emitted."""
    gen = torch.Generator(device=DEV).manual_seed(2024)
    g, top_p = 16, 0.9
    tgt_one = _small_support_logits(K + 1, gen)
    drf_one = (tgt_one[:K] + torch.randn(K, V, generator=gen, device=DEV) * 0.8).contiguous()
    p0, p1 = _nucleus_probs(tgt_one[0], top_p), _nucleus_probs(tgt_one[1], top_p)
    assert int((p0 > 0).sum()) < 90  # the nucleus is a strict subset of the 96-token support
    tgt, drf = tgt_one.repeat(g, 1).contiguous(), drf_one.repeat(g, 1).contiguous()
    temps_tok, temps_req = torch.ones(g * (K + 1), device=DEV), torch.ones(g, device=DEV)
    tps_tok, tps_req = torch.full_like(temps_tok, top_p), torch.full_like(temps_req, top_p)
    slots = torch.arange(g, dtype=torch.int64, device=DEV)
    buf = torch.zeros((g + 1, K, V), dtype=torch.float32, device=DEV)
    first, second = [], []
    for t in range(3000):
        seeds = torch.full((g,), 7, dtype=torch.int64, device=DEV)
        offs = torch.full((g,), (K + 1) * t, dtype=torch.int64, device=DEV)
        drafts = ftp.fast_topp_rs_draft_block(
            drf, temps_req, tps_req, slots, -1, buf, seeds, offs, 1
        )
        acc, num = ftp.fast_topp_rs_verify(
            tgt, temps_tok, tps_tok, drafts.view(g, K), slots, -1, buf, seeds, offs
        )
        acc, num = acc.cpu().numpy(), num.cpu().numpy()
        first.append(acc[:, 0])
        second.append(acc[num >= 2, 1])
    first, second = np.concatenate(first), np.concatenate(second)
    assert bool((p0[first] > 0).all()) and bool((p1[second] > 0).all())
    s0, d0, pv0 = _chi2_pvalue(first, p0)
    s1, d1, pv1 = _chi2_pvalue(second, p1)
    print(
        f"first token chi2={s0:.1f} dof={d0} p={pv0:.3f}; second chi2={s1:.1f} dof={d1} p={pv1:.3f}"
    )
    assert pv0 > 1e-3 and pv1 > 1e-3


def test_chi2_draft_tokens_follow_the_stored_nucleus():
    gen = torch.Generator(device=DEV).manual_seed(77)
    g, top_p = 16, 0.85
    drf_one = _small_support_logits(K, gen)
    drf = drf_one.repeat(g, 1).contiguous()
    temps_req = torch.full((g,), 0.8, device=DEV)
    tps_req = torch.full((g,), top_p, device=DEV)
    slots = torch.arange(g, dtype=torch.int64, device=DEV)
    buf = torch.zeros((g + 1, K, V), dtype=torch.float32, device=DEV)
    toks = []
    for t in range(3000):
        seeds = torch.full((g,), 3, dtype=torch.int64, device=DEV)
        offs = torch.full((g,), (K + 1) * t, dtype=torch.int64, device=DEV)
        out = ftp.fast_topp_rs_draft_block(drf, temps_req, tps_req, slots, -1, buf, seeds, offs, 1)
        toks.append(out.view(g, K).cpu().numpy())
    toks = np.concatenate(toks)
    q = buf[0].double().cpu().numpy()
    for i in (0, K - 1):
        want = _nucleus_probs(drf_one[i], top_p, 0.8)
        assert float(np.abs(q[i] - want).max()) < 1e-5  # the stored row is the nucleus
        assert bool((want[toks[:, i]] > 0).all())  # and no token outside it is drawn
        stat, dof, pv = _chi2_pvalue(toks[:, i], q[i] / q[i].sum())
        print(f"draft position {i}: chi2={stat:.1f} dof={dof} p={pv:.3f}")
        assert pv > 1e-3


def _independence_pvalue(pairs):
    """Chi-square test of independence (1 dof) of a list of (bool, bool) pairs."""
    t = np.zeros((2, 2), dtype=np.float64)
    for x, y in pairs:
        t[int(x), int(y)] += 1
    n = t.sum()
    expected = np.outer(t.sum(1), t.sum(0)) / n
    stat = float(((t - expected) ** 2 / expected).sum())
    return stat, math.erfc(math.sqrt(stat / 2))


def test_consecutive_steps_are_independent():
    """Lossless across steps, not just within one: the token drawn after a rejection at position
    0 must not predict whether the next step (offset advanced by K + 1) accepts its position 0."""
    gen = torch.Generator(device=DEV).manual_seed(31)
    g, top_p = 16, 0.9
    tgt_one = _small_support_logits(K + 1, gen)
    drf_one = (tgt_one[:K] + torch.randn(K, V, generator=gen, device=DEV) * 1.5).contiguous()
    tgt, drf = tgt_one.repeat(g, 1).contiguous(), drf_one.repeat(g, 1).contiguous()
    temps_tok, temps_req = torch.ones(g * (K + 1), device=DEV), torch.ones(g, device=DEV)
    tps_tok, tps_req = torch.full_like(temps_tok, top_p), torch.full_like(temps_req, top_p)
    slots = torch.arange(g, dtype=torch.int64, device=DEV)
    buf = torch.zeros((g + 1, K, V), dtype=torch.float32, device=DEV)
    seeds = torch.full((g,), 11, dtype=torch.int64, device=DEV)
    steps = []
    for t in range(2500):
        offs = torch.full((g,), (K + 1) * t, dtype=torch.int64, device=DEV)
        drafts = ftp.fast_topp_rs_draft_block(
            drf, temps_req, tps_req, slots, -1, buf, seeds, offs, 1
        )
        drafts = drafts.view(g, K).contiguous()
        acc, num = ftp.fast_topp_rs_verify(
            tgt, temps_tok, tps_tok, drafts, slots, -1, buf, seeds, offs
        )
        steps.append((acc[:, 0].cpu().numpy(), num.cpu().numpy()))
    p0 = _nucleus_probs(tgt_one[0], top_p)
    resid = np.maximum(p0 - buf[0, 0].double().cpu().numpy(), 0.0)
    median = int(np.searchsorted(np.cumsum(resid) / resid.sum(), 0.5))
    pairs = [
        (tok[b] <= median, nxt[b] >= 2)
        for (tok, num), (_, nxt) in zip(steps[:-1], steps[1:])
        for b in range(g)
        if num[b] == 1
    ]
    stat, pvalue = _independence_pvalue(pairs)
    print(f"{len(pairs)} (residual token, next accept) pairs, chi2={stat:.1f} p={pvalue:.2e}")
    assert len(pairs) > 5000 and pvalue > 1e-3


# ----------------------------------------------------------------------------------------------
# CUDA graphs.
# ----------------------------------------------------------------------------------------------
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


def test_cuda_graph_replay_matches_eager():
    """Verify, draft block and the Markov chain captured in one graph replay what eager runs
    compute, also with new logits, top-p and seeds written into the captured tensors."""
    gen = torch.Generator(device=DEV).manual_seed(9)
    g, rank = 8, 256
    tgt, drf = _correlated_case(g, gen, 0.5)
    temps_tok, temps_req = torch.ones(g * (K + 1), device=DEV), torch.ones(g, device=DEV)
    tps_tok, tps_req = torch.full_like(temps_tok, 0.95), torch.full_like(temps_req, 0.95)
    slots = torch.arange(g, dtype=torch.int64, device=DEV) + 2
    buf = torch.zeros((g + 3, K, V), dtype=torch.float32, device=DEV)
    seeds = torch.full((g,), 5, dtype=torch.int64, device=DEV)
    offs = torch.full((g,), 12, dtype=torch.int64, device=DEV)
    s_tgt, s_drf = tgt.clone(), drf.clone()
    w1 = (torch.randn(V, rank, generator=gen, device=DEV) * 0.2).to(torch.bfloat16)
    w2 = (torch.randn(V, rank, generator=gen, device=DEV) * 0.2).to(torch.bfloat16)
    anchor = torch.randint(0, V, (g,), generator=gen, device=DEV)
    drafts = ftp.fast_topp_rs_draft_block(
        s_drf.view(-1, V), temps_req, tps_req, slots, -1, buf, seeds, offs, 1
    )
    drafts = drafts.view(g, K).contiguous()

    def step():
        acc, num = ftp.fast_topp_rs_verify(
            s_tgt, temps_tok, tps_tok, drafts, slots, -1, buf, seeds, offs
        )
        blk = ftp.fast_topp_rs_draft_block(
            s_drf.view(-1, V), temps_req, tps_req, slots, -1, buf, seeds, offs, 1
        )
        fb = ftp.fast_topp_rs_draft_markov_feedback(
            s_drf, anchor, w1, w2, temps_req, tps_req, slots, -1, buf, seeds, offs, 1
        )
        return acc, num, blk, fb

    snapshot = buf.clone()
    eager = [x.clone() for x in step()]
    buf.copy_(snapshot)
    graph, outs = _capture(step)
    buf.copy_(snapshot)
    graph.replay()
    torch.cuda.synchronize()
    for a, b in zip(eager, outs):
        assert torch.equal(a, b)
    # New inputs into the same tensors: the replay must follow them (top-p changes the nuclei).
    tgt2, drf2 = _correlated_case(g, gen, 0.5)
    s_tgt.copy_(tgt2)
    s_drf.copy_(drf2)
    tps_tok.fill_(0.6)
    tps_req.fill_(0.6)
    seeds.fill_(6)
    snapshot = buf.clone()
    graph.replay()
    torch.cuda.synchronize()
    replay = [x.clone() for x in outs]
    buf.copy_(snapshot)
    for a, b in zip(step(), replay):
        assert torch.equal(a, b)


def test_workspace_growth_keeps_captured_graph(monkeypatch):
    """A graph captured at a small batch keeps its scratch after an eager step at a larger batch
    grows the buffers: the outgrown buffers are retired, never freed, so memory handed out
    afterwards is not written by a replay (the nucleus scratch included)."""
    workspace = frs.FastRejectionWorkspace()
    monkeypatch.setattr(frs, "_WORKSPACE", workspace)
    monkeypatch.setattr(ftp, "_WORKSPACE", workspace)
    gen = torch.Generator(device=DEV).manual_seed(12)

    def case(g):
        tgt, drf = _correlated_case(g, gen, 0.5)
        slots = torch.arange(g, dtype=torch.int64, device=DEV)
        buf = torch.zeros((g + 1, K, V), dtype=torch.float32, device=DEV)
        seeds = torch.full((g,), 8, dtype=torch.int64, device=DEV)
        offs = torch.full((g,), 24, dtype=torch.int64, device=DEV)
        temps_req, temps_tok = torch.ones(g, device=DEV), torch.ones(g * (K + 1), device=DEV)
        tps_req, tps_tok = torch.full_like(temps_req, 0.95), torch.full_like(temps_tok, 0.95)
        drafts = ftp.fast_topp_rs_draft_block(
            drf.view(-1, V), temps_req, tps_req, slots, -1, buf, seeds, offs, 1
        )
        return (
            tgt,
            temps_tok,
            tps_tok,
            drafts.view(g, K).contiguous(),
            slots,
            -1,
            buf,
            seeds,
            offs,
        )

    small = case(4)
    # Sizes the verify scratch for G=4 on this stream (the capture below reuses it).
    want = [x.clone() for x in ftp.fast_topp_rs_verify(*small)]
    graph, outs = _capture(lambda: ftp.fast_topp_rs_verify(*small))
    ftp.fast_topp_rs_verify(*case(16))  # outgrows every verify scratch buffer
    # Had the G=4 buffers been freed, these (same stream, same size classes) would be handed their
    # blocks and the replay would write into them.
    junk = [
        torch.full((n,), float("nan"), device=DEV)
        for _ in range(3000)
        for n in (8, 160, 960, 4 * 24 * 4, 24 * 4096)
    ]
    graph.replay()
    torch.cuda.synchronize()
    assert bool(torch.cat(junk).isnan().all())
    for a, b in zip(outs, want):
        assert torch.equal(a, b)
