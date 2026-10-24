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
"""The fused rejection-sampling kernels (TLLM_SPEC_FAST_RS) against flashinfer and float64 models.

Random stream (``fast_rejection_sampling`` module docstring): request ``b``'s verify uses the
cuRAND Philox stream (seed[b], subsequence b, offset[b]), draw i for accept test i and draw K for
the residual or bonus token; its draft tokens are draw 0 of (seed[b] ^ DRAFT_KEY_XOR,
subsequence b * K + i, offset[b] + 1).

Exactness:
  * the draft block sampler vs flashinfer's ``sampling_from_probs`` path on the draft key;
  * verify accept decisions vs ``chain_speculative_sampling`` on the same draws, and the drawn
    token vs a float64 inverse-CDF model of draw K;
  * per-request 64-bit seeds and offsets, masked (-inf) logits, a runtime draft length below the
    buffer's, a vocabulary that is not a multiple of the chunk, per-row temperatures;
  * the SpecWorkerBase entry points and the sampled-token Markov chain.
Every disagreement must be explained by float32 rounding: the draw within ``_CDF_TOL`` (of the
row mass) of an inverse-CDF boundary, or within ``_ACCEPT_TOL`` (relative) of the accept
threshold, both computed in float64 with the cuRAND draw recomputed here.

Losslessness: chi-square of the emitted and the draft tokens over consecutive offset windows,
and no Philox draw is used twice for one request across steps (layout test).
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
# A float32 running sum over V terms is off by a few ulps of 1.0 per term group; the tail tokens
# the boundary falls between carry ~1e-8 each, so a disagreement there is decided by rounding.
_CDF_TOL = 2e-5
_ACCEPT_TOL = 1e-5

_M32 = 0xFFFFFFFF
_M64 = (1 << 64) - 1


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
    seed &= _M64
    g = (offset & 3) + draw
    ctr = (offset >> 2) + (g >> 2)
    out = _philox4x32_10(
        (ctr & _M32, (ctr >> 32) & _M32, subsequence & _M32, (subsequence >> 32) & _M32),
        (seed & _M32, (seed >> 32) & _M32),
    )
    x = np.float32(out[g & 3])
    return float(np.float32(x * np.float32(2.0**-32) + np.float32(2.0**-33)))


def _draft_key(seed):
    """The draft stream's Philox key for a request seed (the kernel's int64 XOR)."""
    return (seed & _M64) ^ frs.DRAFT_KEY_XOR


def _as_int64(x):
    x &= _M64
    return x - (1 << 64) if x >> 63 else x


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


def _inv_cdf(weights, u):
    """flashinfer's ``sampling_from_probs`` rule on float64 sums: the first token whose running
    mass exceeds ``u`` (it has nonzero weight), else the last nonzero token."""
    w = np.asarray(weights, dtype=np.float64)
    idx = int(np.searchsorted(np.cumsum(w), u, side="right"))
    return idx if idx < len(w) else int(np.flatnonzero(w > 0)[-1])


def _token_gap(weights, u, got):
    """0 when ``got`` is the model's token for ``u``; else the CDF distance (relative to the row
    mass) that rounding must explain."""
    want = _inv_cdf(weights, u)
    if int(got) == want:
        return 0.0
    return _cdf_gap(weights, u, int(got), want) / max(float(np.sum(weights)), 1e-300)


def _verify_model(p, q, drafts, seed, offset, b, k):
    """Request ``b``'s accept position and drawn token weights from float64 target ``p``
    [k + 1, V] and stored proposals ``q`` [k, V]: (pos, weights, u scaled by their mass)."""
    pos = k
    for i in range(k):
        d = int(drafts[i])
        if not curand_uniform(seed, offset, b, i) * q[i, d] < p[i, d]:
            pos = i
            break
    w = np.maximum(p[pos] - q[pos], 0.0) if pos < k else p[pos]
    return pos, w, curand_uniform(seed, offset, b, k) * float(w.sum())


def _verify_row_gap(acc_row, num, p, q, drafts, seed, offset, b, k):
    """("accept" | "token", size of the rounding that explains the kernel's row vs the model)."""
    pos_m, w, u = _verify_model(p, q, drafts, seed, offset, b, k)
    pos = int(num) - 1
    if pos != pos_m:
        i = min(pos, pos_m)
        d = int(drafts[i])
        u_i = curand_uniform(seed, offset, b, i)
        return "accept", abs(u_i * q[i, d] - p[i, d]) / max(p[i, d], 1e-300)
    row = [int(x) for x in acc_row]
    assert row[:pos] == [int(x) for x in drafts[:pos]]
    assert all(x == -1 for x in row[pos + 1 : k + 1])
    return "token", _token_gap(w, u, row[pos])


def _softmax64(logits, temps):
    """float64 softmax(logits / T) with one temperature per row."""
    return torch.softmax(logits.double() / temps.double().unsqueeze(1), dim=-1).cpu().numpy()


def peaked_logits(rows, vocab, gen, spread=2.0, ncand=48, top=14.0):
    base = torch.randn(rows, vocab, generator=gen, device=DEV) * spread
    idx = torch.randint(0, vocab, (rows, ncand), generator=gen, device=DEV)
    boost = torch.linspace(top, top - 8.0, ncand, device=DEV).unsqueeze(0).expand(rows, -1)
    base.scatter_add_(1, idx, boost + torch.randn(rows, ncand, generator=gen, device=DEV))
    return base.float().contiguous()


def _correlated_case(num_gens, gen, accept_bias, k=K, vocab=V):
    """Target logits [G*(k+1), V] and draft logits [G, k, V] that agree often."""
    tgt = peaked_logits(num_gens * (k + 1), vocab, gen)
    noise = torch.randn(num_gens, k, vocab, generator=gen, device=DEV) * accept_bias
    drf = (tgt.view(num_gens, k + 1, vocab)[:, :k] + noise).contiguous()
    return tgt, drf


class _Gaps:
    """Worst rounding size per kind and the number of rows that needed one."""

    def __init__(self):
        self.worst = {"accept": 0.0, "token": 0.0}
        self.count = 0

    def add(self, kind, size):
        self.worst[kind] = max(self.worst[kind], size)
        self.count += size > 0.0

    def check(self, total):
        assert self.worst["accept"] <= _ACCEPT_TOL and self.worst["token"] <= _CDF_TOL
        assert self.count <= max(2, int(0.02 * total))


def _check_drafts(tokens, stored, seeds, offsets, k, gaps):
    """Draft tokens [G, k] vs the model on the stored proposals [G, k, V] (float64)."""
    for b in range(tokens.shape[0]):
        for i in range(k):
            u = curand_uniform(_draft_key(seeds[b]), offsets[b] + 1, b * k + i, 0)
            gaps.add("token", _token_gap(stored[b, i], u, tokens[b, i]))


def _check_verify(acc, num, p, q, drafts, seeds, offsets, k, gaps):
    for b in range(acc.shape[0]):
        gaps.add(
            *_verify_row_gap(acc[b], num[b], p[b], q[b], drafts[b], seeds[b], offsets[b], b, k)
        )


@pytest.mark.parametrize("num_gens", [1, 16])
@pytest.mark.parametrize("temperature", [1.0, 0.6])
def test_draft_block_matches_flashinfer(num_gens, temperature):
    gen = torch.Generator(device=DEV).manual_seed(11 + num_gens)
    mism = total = 0
    worst_gap = worst_rel = 0.0
    for trial in range(40):
        logits = peaked_logits(num_gens * K, V, gen)
        temps = torch.full((num_gens,), temperature, dtype=torch.float32, device=DEV)
        seed = 1234 + trial
        seeds = torch.full((num_gens,), seed, dtype=torch.int64, device=DEV)
        offs = torch.full((num_gens,), 6 * trial + trial % 4, dtype=torch.int64, device=DEV)
        draft_seeds = torch.full_like(seeds, _as_int64(_draft_key(seed)))
        ref_tok, ref_probs = sampling_batch_spec_dec_one_model_for_rejection(
            logits,
            temps.repeat_interleave(K),
            None,
            None,
            seed=draft_seeds.repeat_interleave(K),
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
            u = curand_uniform(_draft_key(seed), int(offs[0]) + 1, r, 0)
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
    gen = torch.Generator(device=DEV).manual_seed(101 + num_gens)
    model, decisions = _Gaps(), _Gaps()
    total = 0
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
        acc, num = frs.fast_rs_verify(tgt, temps_tok, drafts, slots, -1, buf, seeds, offs)
        # Accept decisions: chain_speculative_sampling draws the same uniforms 0..K-1 (what
        # _sample_and_accept_draft_tokens_rejection runs in NO_TOPK_NO_TOPP mode).
        tprobs = compute_probs_from_logits(tgt, temps_tok, None, None).view(num_gens, K + 1, V)
        _, ref_num = rejection_sampling_one_model(
            buf[slots], drafts, tprobs, deterministic=True, seed=seeds, offset=offs
        )
        p = _softmax64(tgt, temps_tok).reshape(num_gens, K + 1, V)
        q = buf[slots].double().cpu().numpy()
        dr = drafts.cpu().numpy()
        for b in (num.long() != ref_num.long()).nonzero().flatten().tolist():
            i = min(int(num[b]), int(ref_num[b])) - 1
            d = int(dr[b, i])
            u = curand_uniform(seed, offset, b, i)
            decisions.add("accept", abs(u * q[b, i, d] - p[b, i, d]) / max(p[b, i, d], 1e-300))
        # Every row (accept position and drawn token) vs the float64 model.
        _check_verify(
            acc.cpu().numpy(),
            num.cpu().numpy(),
            p,
            q,
            dr,
            [seed] * num_gens,
            [offset] * num_gens,
            K,
            model,
        )
        total += num_gens
        for n in num.tolist():
            hist[n - 1] += 1
    print(
        f"verify G={num_gens} bias={accept_bias}: accept decisions vs flashinfer differ in "
        f"{decisions.count}/{total} rows (worst {decisions.worst['accept']:.2e} rel); vs the "
        f"float64 model {model.count}/{total} rows (worst accept {model.worst['accept']:.2e}, "
        f"token {model.worst['token']:.2e}); accepted-length histogram {hist.tolist()}"
    )
    decisions.check(total)
    model.check(total)


def test_per_request_seeds_and_offsets():
    """Each request keys its own stream: 64-bit (negative) seeds, offsets past 2**32, per-row
    temperatures."""
    gen = torch.Generator(device=DEV).manual_seed(3)
    seeds_py = [1, 2**40 + 7, -5, 0x7FFFFFFFFFFFFFF0, 99, -(2**62), 12345, 31337]
    offs_py = [0, 6, 2**33 + 3, 18, 7, 2**32 - 1, 120, 61]
    g = len(seeds_py)
    seeds = torch.tensor(seeds_py, dtype=torch.int64, device=DEV)
    offs = torch.tensor(offs_py, dtype=torch.int64, device=DEV)
    drafts_gaps, verify_gaps = _Gaps(), _Gaps()
    for _ in range(4):
        tgt, drf = _correlated_case(g, gen, 0.8)
        temps_req = 0.6 + 0.8 * torch.rand(g, generator=gen, device=DEV)
        temps_tok = 0.6 + 0.8 * torch.rand(g * (K + 1), generator=gen, device=DEV)
        slots = torch.randperm(g + 2, device=DEV)[:g].to(torch.int64)
        buf = torch.zeros((g + 2, K, V), dtype=torch.float32, device=DEV)
        drafts = frs.fast_rs_draft_block(drf.view(-1, V), temps_req, slots, -1, buf, seeds, offs, 1)
        drafts = drafts.view(g, K).contiguous()
        q = buf[slots].double().cpu().numpy()
        want_q = _softmax64(drf.view(-1, V), temps_req.repeat_interleave(K)).reshape(g, K, V)
        assert np.abs(q - want_q).max() < 1e-5
        _check_drafts(drafts.cpu().numpy(), q, seeds_py, offs_py, K, drafts_gaps)
        acc, num = frs.fast_rs_verify(tgt, temps_tok, drafts, slots, -1, buf, seeds, offs)
        p = _softmax64(tgt, temps_tok).reshape(g, K + 1, V)
        _check_verify(
            acc.cpu().numpy(),
            num.cpu().numpy(),
            p,
            q,
            drafts.cpu().numpy(),
            seeds_py,
            offs_py,
            K,
            verify_gaps,
        )
    print(f"per-request streams: draft worst {drafts_gaps.worst}, verify worst {verify_gaps.worst}")
    drafts_gaps.check(4 * g * K)
    verify_gaps.check(4 * g)


def test_masked_logits():
    """-inf logits (guided-decoding style masks) get probability 0, also in chunks with no finite
    logit at all, and the outputs stay inside the allowed set."""
    gen = torch.Generator(device=DEV).manual_seed(4)
    g = 4
    chunk = frs._CHUNK
    pool = torch.cat(
        [torch.arange(c * chunk, (c + 1) * chunk) for c in range(V // chunk) if c not in (1, 7, 39)]
    )
    allowed = pool[torch.randperm(pool.numel(), generator=torch.Generator().manual_seed(0))[:100]]
    allowed = allowed.to(DEV)
    keep = torch.zeros(V, dtype=torch.bool, device=DEV)
    keep[allowed] = True
    tgt, drf = _correlated_case(g, gen, 0.8)
    tgt = tgt.masked_fill(~keep, float("-inf"))
    drf = drf.clone()
    drf[: g // 2] = drf[: g // 2].masked_fill(~keep, float("-inf"))  # half the drafts masked too
    seeds = torch.full((g,), 21, dtype=torch.int64, device=DEV)
    offs = torch.full((g,), 30, dtype=torch.int64, device=DEV)
    slots = torch.arange(g, dtype=torch.int64, device=DEV)
    buf = torch.zeros((g + 1, K, V), dtype=torch.float32, device=DEV)
    temps_req = torch.ones(g, device=DEV)
    temps_tok = torch.ones(g * (K + 1), device=DEV)
    drafts = frs.fast_rs_draft_block(drf.view(-1, V), temps_req, slots, -1, buf, seeds, offs, 1)
    drafts = drafts.view(g, K).contiguous()
    assert not torch.isnan(buf).any()
    assert bool(keep[drafts[: g // 2].long()].all())
    assert float(buf[: g // 2][:, :, ~keep].abs().max()) == 0.0
    acc, num = frs.fast_rs_verify(tgt, temps_tok, drafts, slots, -1, buf, seeds, offs)
    for b in range(g):
        emitted = acc[b, : int(num[b])].long()
        assert bool(keep[emitted].all()), (b, emitted.tolist())
    gaps = _Gaps()
    p = _softmax64(tgt, temps_tok).reshape(g, K + 1, V)
    q = buf[slots].double().cpu().numpy()
    _check_verify(
        acc.cpu().numpy(),
        num.cpu().numpy(),
        p,
        q,
        drafts.cpu().numpy(),
        [21] * g,
        [30] * g,
        K,
        gaps,
    )
    gaps.check(g)


def test_short_draft_and_ragged_vocab():
    """Runtime draft length 3 in a 5-step buffer, a vocabulary that is not a chunk multiple,
    per-row temperatures."""
    gen = torch.Generator(device=DEV).manual_seed(6)
    vocab, k, g = 10000, 3, 6
    drafts_gaps, verify_gaps = _Gaps(), _Gaps()
    for trial in range(6):
        tgt, drf = _correlated_case(g, gen, 0.6, k=k, vocab=vocab)
        temps_req = 0.5 + torch.rand(g, generator=gen, device=DEV)
        temps_tok = 0.5 + torch.rand(g * (k + 1), generator=gen, device=DEV)
        seeds_py, offs_py = [77 + trial] * g, [(k + 1) * trial] * g
        seeds = torch.tensor(seeds_py, dtype=torch.int64, device=DEV)
        offs = torch.tensor(offs_py, dtype=torch.int64, device=DEV)
        slots = torch.arange(g, dtype=torch.int64, device=DEV)
        buf = torch.zeros((g + 1, K, vocab), dtype=torch.float32, device=DEV)
        drafts = frs.fast_rs_draft_block(
            drf.view(-1, vocab), temps_req, slots, -1, buf, seeds, offs, 1
        )
        drafts = drafts.view(g, k).contiguous()
        q = buf[slots, :k].double().cpu().numpy()
        _check_drafts(drafts.cpu().numpy(), q, seeds_py, offs_py, k, drafts_gaps)
        acc, num = frs.fast_rs_verify(tgt, temps_tok, drafts, slots, -1, buf, seeds, offs)
        p = _softmax64(tgt, temps_tok).reshape(g, k + 1, vocab)
        _check_verify(
            acc.cpu().numpy(),
            num.cpu().numpy(),
            p,
            q,
            drafts.cpu().numpy(),
            seeds_py,
            offs_py,
            k,
            verify_gaps,
        )
    drafts_gaps.check(6 * g * k)
    verify_gaps.check(6 * g)


def test_philox_layout_never_reuses_a_draw():
    """Over consecutive steps (offset window max_draft_len + 1, as
    _populate_request_rng_state advances it), no (key, subsequence, word) is drawn twice, even
    with every request on one seed."""
    window = K + 1
    seed = 7
    for k in range(1, K + 1):
        for g in (1, 4, 16):
            seen = set()
            for step in range(3):
                for b in range(g):
                    offset = (b % 3 + step) * window
                    draws = [(seed, b, offset + i) for i in range(k + 1)]
                    draws += [(_draft_key(seed), b * k + i, offset + 1) for i in range(k)]
                    for d in draws:
                        assert d not in seen, (k, g, step, b, d)
                        seen.add(d)


class _Worker(itf.SpecWorkerBase):
    def __init__(self):
        super().__init__()
        self.mapping = None

    @property
    def max_draft_len(self):
        return K

    def _forward_impl(self, *args, **kwargs):
        raise NotImplementedError


def _worker_metadata(mbs, slots, seed, offset):
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


def test_worker_paths_fast(monkeypatch):
    """SpecWorkerBase entry points with the flag on (NO_TOPK_NO_TOPP): a mixed batch (2 context
    rows) and a padding request routed to the dummy slot."""
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
    # The fused path stores the reference's proposals and draws its tokens from them.
    assert float((probs["0"][real] - probs["1"][real]).abs().max()) < 1e-6
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
    p = _softmax64(logits[nctx:], torch.ones(ngen * (K + 1), device=DEV)).reshape(ngen, K + 1, V)
    real_gens = slice(nctx, bsz - 1)
    _check_verify(
        a1[real_gens].cpu().numpy(),
        n1[real_gens].cpu().numpy(),
        p,
        q,
        drafts[: ngen - 1].cpu().numpy(),
        [seed] * ngen,
        [offset] * ngen,
        K,
        gaps,
    )
    # The accept decisions draw the reference's uniforms (draws 0..K-1).
    same = int((n0[real_gens] == n1[real_gens]).sum())
    print(
        f"worker: worst {gaps.worst}; accept counts equal to the reference in {same}/{ngen - 1}"
        f" rows; counts {n1.tolist()}"
    )
    gaps.check(ngen - 1)
    assert same >= ngen - 2


def test_guided_decoding_keeps_reference(monkeypatch):
    monkeypatch.setenv(frs.FAST_RS_ENV_VAR, "1")
    md = _worker_metadata(4, range(4), 1, 0)
    worker = _Worker()
    logits = torch.zeros((4 * (K + 1), V), device=DEV)
    assert worker._fast_rs_applies(md, logits, V)
    worker.guided_decoder = types.SimpleNamespace()
    assert not worker._fast_rs_applies(md, logits, V)


def test_markov_feedback_matches_float64_model():
    """The chain advances on the sampled token: base_i + w2 @ w1[prev], prev_0 = anchor; position
    i of request b draws draw 0 of (seed ^ DRAFT_KEY_XOR, subsequence b * K + i, offset + 1)."""
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
            torch.full((num_gens,), seed, dtype=torch.int64, device=DEV),
            torch.full((num_gens,), offset, dtype=torch.int64, device=DEV),
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
    print(f"markov feedback: {mism}/{checked} disagreements, worst CDF gap {worst:.2e}")
    assert worst <= _CDF_TOL


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


def test_chi2_emitted_tokens_follow_target():
    """P(first emitted = x) = p_0(x) and P(second = y | first accepted) = p_1(y) for a proposal
    that differs from the target (rejection sampling is lossless for any proposal); the trials
    are consecutive offset windows of the same requests, as in decoding."""
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
    for t in range(3000):
        seeds = torch.full((g,), 7, dtype=torch.int64, device=DEV)
        offs = torch.full((g,), (K + 1) * t, dtype=torch.int64, device=DEV)
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
    gen = torch.Generator(device=DEV).manual_seed(77)
    g = 16
    drf = _small_support_logits(K, gen).repeat(g, 1).contiguous()
    temps_req = torch.full((g,), 0.8, device=DEV)
    slots = torch.arange(g, dtype=torch.int64, device=DEV)
    buf = torch.zeros((g + 1, K, V), dtype=torch.float32, device=DEV)
    toks = []
    for t in range(3000):
        seeds = torch.full((g,), 3, dtype=torch.int64, device=DEV)
        offs = torch.full((g,), (K + 1) * t, dtype=torch.int64, device=DEV)
        out = frs.fast_rs_draft_block(drf, temps_req, slots, -1, buf, seeds, offs, 1)
        toks.append(out.view(g, K).cpu().numpy())
    toks = np.concatenate(toks)
    q = buf[0].double().cpu().numpy()
    for i in (0, K - 1):
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
    0 must not predict whether the next step (offset advanced by K + 1) accepts its position 0.
    flashinfer's chain draws that residual with draw K + 1, the next window's draw 0, so on the
    reference the two are dependent (printed, not asserted)."""
    gen = torch.Generator(device=DEV).manual_seed(31)
    g = 16
    tgt_one = _small_support_logits(K + 1, gen)
    drf_one = (tgt_one[:K] + torch.randn(K, V, generator=gen, device=DEV) * 1.5).contiguous()
    tgt, drf = tgt_one.repeat(g, 1).contiguous(), drf_one.repeat(g, 1).contiguous()
    temps_tok, temps_req = torch.ones(g * (K + 1), device=DEV), torch.ones(g, device=DEV)
    tprobs = compute_probs_from_logits(tgt, temps_tok, None, None).view(g, K + 1, V)
    slots = torch.arange(g, dtype=torch.int64, device=DEV)
    buf = torch.zeros((g + 1, K, V), dtype=torch.float32, device=DEV)
    seeds = torch.full((g,), 11, dtype=torch.int64, device=DEV)
    steps = {"fused": [], "reference": []}
    for t in range(2500):
        offs = torch.full((g,), (K + 1) * t, dtype=torch.int64, device=DEV)
        drafts = frs.fast_rs_draft_block(drf, temps_req, slots, -1, buf, seeds, offs, 1)
        drafts = drafts.view(g, K).contiguous()
        acc, num = frs.fast_rs_verify(tgt, temps_tok, drafts, slots, -1, buf, seeds, offs)
        steps["fused"].append((acc[:, 0].cpu().numpy(), num.cpu().numpy()))
        acc, num = rejection_sampling_one_model(
            buf[slots], drafts, tprobs, deterministic=True, seed=seeds, offset=offs
        )
        steps["reference"].append((acc[:, 0].cpu().numpy(), num.cpu().numpy()))
    p0 = torch.softmax(tgt_one[0].double(), -1).cpu().numpy()
    resid = np.maximum(p0 - buf[0, 0].double().cpu().numpy(), 0.0)
    median = int(np.searchsorted(np.cumsum(resid) / resid.sum(), 0.5))
    pvalue = {}
    for name, rows in steps.items():
        pairs = [
            (tok[b] <= median, nxt[b] >= 2)
            for (tok, num), (_, nxt) in zip(rows[:-1], rows[1:])
            for b in range(g)
            if num[b] == 1
        ]
        stat, pvalue[name] = _independence_pvalue(pairs)
        print(
            f"{name}: {len(pairs)} (residual token, next accept) pairs, chi2={stat:.1f} "
            f"p={pvalue[name]:.2e}"
        )
    assert pvalue["fused"] > 1e-3


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
    graph, outs = _capture(step)
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


def test_workspace_growth_keeps_captured_graph(monkeypatch):
    """A graph captured at a small batch keeps its scratch after an eager step at a larger batch
    grows the buffers: the outgrown buffers are retired, never freed, so memory handed out
    afterwards is not written by a replay."""
    monkeypatch.setattr(frs, "_WORKSPACE", frs.FastRejectionWorkspace())
    gen = torch.Generator(device=DEV).manual_seed(12)

    def case(g):
        tgt, drf = _correlated_case(g, gen, 0.5)
        slots = torch.arange(g, dtype=torch.int64, device=DEV)
        buf = torch.zeros((g + 1, K, V), dtype=torch.float32, device=DEV)
        seeds = torch.full((g,), 8, dtype=torch.int64, device=DEV)
        offs = torch.full((g,), 24, dtype=torch.int64, device=DEV)
        temps_req, temps_tok = torch.ones(g, device=DEV), torch.ones(g * (K + 1), device=DEV)
        drafts = frs.fast_rs_draft_block(drf.view(-1, V), temps_req, slots, -1, buf, seeds, offs, 1)
        return (tgt, temps_tok, drafts.view(g, K).contiguous(), slots, -1, buf, seeds, offs)

    small = case(4)
    # Sizes the verify scratch for G=4 on this stream (the capture below reuses it).
    want = [x.clone() for x in frs.fast_rs_verify(*small)]
    graph, outs = _capture(lambda: frs.fast_rs_verify(*small))
    frs.fast_rs_verify(*case(16))  # outgrows every verify scratch buffer
    # Had the G=4 buffers been freed, these (same stream, same size classes: pos, rsum / rlast,
    # pmax / psum) would be handed their blocks and the replay would write into them.
    junk = [torch.full((n,), float("nan"), device=DEV) for _ in range(3000) for n in (8, 160, 960)]
    graph.replay()
    torch.cuda.synchronize()
    assert bool(torch.cat(junk).isnan().all())
    for a, b in zip(outs, want):
        assert torch.equal(a, b)


@pytest.mark.skipif(os.environ.get(frs.FAST_RS_ENV_VAR) == "1", reason="env forces the flag")
def test_flag_defaults_off():
    assert not frs.fast_rs_enabled()
    # Flag off: the gate decides before reading the metadata, so a stand-in without the
    # sampling-mode field (as in test_rejection_buffers_guard) keeps the reference dispatch.
    assert not _Worker()._fast_rs_applies(types.SimpleNamespace(), None, V)
