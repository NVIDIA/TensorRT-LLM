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
"""The fused top-p draft block sampler and verify kernels against flashinfer's ``softmax`` +
``top_p_renorm_probs`` + ``sampling_from_probs`` / ``chain_speculative_sampling`` on the same Philox
draws (>= 2,000 verify rows per case), and against float64 inverse-CDF models.
"""

import numpy as np
import pytest
import torch
from topp_rs_test_utils import (
    _ACCEPT_TOL,
    _CDF_TOL,
    DEV,
    K,
    Nucleus,
    V,
    _as_int64,
    _cdf_gap,
    _check_drafts,
    _check_verify,
    _correlated_case,
    _draft_key,
    _Gaps,
    _request_top_ps,
    curand_uniform,
    fi_support_ok,
    kernel_kept,
    make_rows,
    support_diff_ok,
    target_error,
)

from tensorrt_llm._torch.pyexecutor.sampler.ops.flashinfer import (
    compute_probs_from_logits,
    sampling_batch_spec_dec_one_model_for_rejection,
)
from tensorrt_llm._torch.speculative import fast_rejection_sampling as frs
from tensorrt_llm._torch.speculative import fast_topp_rejection_sampling as ftp
from tensorrt_llm._torch.speculative.interface import rejection_sampling_one_model

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@pytest.mark.parametrize("bf16", [False, True], ids=["fp32", "bf16"])
@pytest.mark.parametrize("kind", ["peaked", "flat"])
@pytest.mark.parametrize("num_gens", [1, 16])
@pytest.mark.parametrize("temperature", [1.0, 0.6])
def test_draft_block_matches_flashinfer(kind, num_gens, temperature, bf16):
    """Stored proposals = flashinfer's softmax + top_p_renorm_probs (per-request top-p, including
    1.0); the drawn tokens = flashinfer's sampling_from_probs on the same draft-key draws. ``bf16``
    rows are rounded to bfloat16 first (what the lm_head produces), so exact ties abound."""
    gen = torch.Generator(device=DEV).manual_seed(11 + num_gens)
    mism = total = support_rows = 0
    worst_gap = worst_rel = worst_err = worst_fi_err = 0.0
    for trial in range(30):
        logits = make_rows(kind, num_gens * K, V, gen, bf16)
        temps = torch.full((num_gens,), temperature, dtype=torch.float32, device=DEV)
        tps = _request_top_ps(num_gens, trial)
        seed = 1234 + trial
        seeds = torch.full((num_gens,), seed, dtype=torch.int64, device=DEV)
        offs = torch.full((num_gens,), 6 * trial + trial % 4, dtype=torch.int64, device=DEV)
        draft_seeds = torch.full_like(seeds, _as_int64(_draft_key(seed)))
        ref_tok, ref_probs = sampling_batch_spec_dec_one_model_for_rejection(
            logits,
            temps.repeat_interleave(K),
            None,
            tps.repeat_interleave(K),
            seed=draft_seeds.repeat_interleave(K),
            offset=(offs + 1).repeat_interleave(K),
        )
        slots = torch.randperm(17, device=DEV)[:num_gens].to(torch.int64)
        buf = torch.zeros((18, K, V), dtype=torch.float32, device=DEV)
        tok = ftp.fast_topp_rs_draft_block(logits, temps, tps, slots, -1, buf, seeds, offs, 1)
        stored = buf[slots].reshape(num_gens * K, V)
        nuc = Nucleus(logits, temps.repeat_interleave(K), tps.repeat_interleave(K))
        kept_k, kept_ref = stored > 0, ref_probs > 0
        # The nucleus: the oracle's, within rounding, and flashinfer's likewise.
        differs = (kept_k != nuc.kept).any(1) | (kept_k != kept_ref).any(1)
        tps_rows = tps.repeat_interleave(K)
        for r in differs.nonzero().flatten().tolist():
            assert support_diff_ok(nuc, r, kept_k[r], nuc.kept[r])
            assert fi_support_ok(nuc, r, float(tps_rows[r]), kept_k[r], kept_ref[r])
            worst_err = max(worst_err, target_error(nuc, r, kept_k[r]))
            if float(tps_rows[r]) < 1.0:
                worst_fi_err = max(worst_fi_err, target_error(nuc, r, kept_ref[r]))
        support_rows += int(differs.sum())
        same = ~differs
        big = (ref_probs > 1e-6) & kept_k & same.unsqueeze(1)
        rel = ((stored - ref_probs).abs() / ref_probs.clamp_min(1e-30))[big]
        if rel.numel() > 0:
            worst_rel = max(worst_rel, float(rel.max()))
        for r in (tok.long() != ref_tok.long()).nonzero().flatten().tolist():
            mism += 1
            if bool(differs[r]):
                continue  # a different nucleus gives a different CDF
            u = curand_uniform(_draft_key(seed), int(offs[0]) + 1, r, 0)
            gap = _cdf_gap(ref_probs[r].double().cpu().numpy(), u, int(tok[r]), int(ref_tok[r]))
            worst_gap = max(worst_gap, gap)
        total += tok.numel()
    print(
        f"draft block {kind} bf16={bf16} G={num_gens} T={temperature}: {mism}/{total} tokens differ, all "
        f"within {worst_gap:.2e} of a CDF boundary (or in one of the {support_rows} rows whose "
        f"nucleus differs within rounding); max rel prob diff {worst_rel:.2e}; worst target error: "
        f"kernel {worst_err:.2e}, flashinfer {worst_fi_err:.2e}"
    )
    assert worst_gap <= _CDF_TOL
    assert mism <= 0.02 * total
    assert worst_rel < 1e-5


@pytest.mark.parametrize("kind", ["peaked", "flat"])
def test_draft_and_verify_are_bitwise_reproducible(kind):
    """The draft tokens, the stored proposals and the verify results are pure functions of the
    inputs: two runs agree bit for bit (tensor-parallel ranks hold identical logits and must emit
    identical tokens), and the proposal of a request does not depend on its batch position."""
    num_gens = 8
    gen = torch.Generator(device=DEV).manual_seed(71)
    tgt, drf = _correlated_case(num_gens, gen, 0.8, kind)
    temps_req = torch.full((num_gens,), 1.0, dtype=torch.float32, device=DEV)
    tps_req = _request_top_ps(num_gens, 0)
    tps_tok = tps_req.repeat_interleave(K + 1).contiguous()
    temps_tok = temps_req.repeat_interleave(K + 1).contiguous()
    seeds = torch.arange(num_gens, dtype=torch.int64, device=DEV) + 77
    offs = torch.arange(num_gens, dtype=torch.int64, device=DEV) * 7

    def run(order):
        idx = torch.tensor(order, dtype=torch.int64, device=DEV)
        rows = (idx[:, None] * K + torch.arange(K, device=DEV)).reshape(-1)
        buf = torch.zeros((num_gens + 1, K, V), dtype=torch.float32, device=DEV)
        slots = idx.clone()
        tok = ftp.fast_topp_rs_draft_block(
            drf.view(num_gens * K, V)[rows].contiguous(), temps_req[idx], tps_req[idx], slots, -1,
            buf, seeds[idx], offs[idx], 1,
        )  # fmt: skip
        trows = (idx[:, None] * (K + 1) + torch.arange(K + 1, device=DEV)).reshape(-1)
        acc, num = ftp.fast_topp_rs_verify(
            tgt[trows].contiguous(), temps_tok[trows], tps_tok[trows], tok.view(num_gens, K), slots,
            -1, buf, seeds[idx], offs[idx],
        )  # fmt: skip
        return tok.view(num_gens, K), buf[slots], acc, num

    forward = list(range(num_gens))
    a = run(forward)
    for x, y in zip(a, run(forward)):
        assert torch.equal(x, y)
    # The stored proposals depend on a request's own logits and top-p only, not on its batch position
    # (the draws do: the Philox subsequence is the batch position).
    backward = run(forward[::-1])
    assert torch.equal(a[1], backward[1].flip(0))


@pytest.mark.parametrize("bf16", [False, True], ids=["fp32", "bf16"])
@pytest.mark.parametrize("kind", ["peaked", "flat"])
@pytest.mark.parametrize("accept_bias", [0.3, 1.5])
def test_verify_matches_flashinfer(kind, accept_bias, bf16):
    """>= 2,000 verify rows per case: every row's accept position and drawn token against the
    float64 model of the same Philox draws, and the accept decisions against
    ``chain_speculative_sampling`` fed flashinfer's nuclei (what the reference runs in
    ``no_topk`` mode)."""
    num_gens, trials = 16, 128
    gen = torch.Generator(device=DEV).manual_seed(101 + int(accept_bias * 10))
    model, decisions = _Gaps(), _Gaps()
    total = same_num = support_rows = 0
    worst_err = worst_fi_err = 0.0
    hist = np.zeros(K + 1, dtype=np.int64)
    temps_req = torch.ones(num_gens, dtype=torch.float32, device=DEV)
    temps_tok = torch.ones(num_gens * (K + 1), dtype=torch.float32, device=DEV)
    for trial in range(trials):
        tgt, drf = _correlated_case(num_gens, gen, accept_bias, kind, bf16=bf16)
        tps_req = _request_top_ps(num_gens, trial)
        tps_tok = tps_req.repeat_interleave(K + 1).contiguous()
        seed, offset = 4321 + trial, 6 * trial + (trial % 4)
        seeds = torch.full((num_gens,), seed, dtype=torch.int64, device=DEV)
        offs = torch.full((num_gens,), offset, dtype=torch.int64, device=DEV)
        slots = torch.randperm(17, device=DEV)[:num_gens].to(torch.int64)
        buf = torch.zeros((18, K, V), dtype=torch.float32, device=DEV)
        drafts = ftp.fast_topp_rs_draft_block(
            drf.view(-1, V), temps_req, tps_req, slots, -1, buf, seeds, offs, 1
        )
        drafts = drafts.view(num_gens, K).contiguous()
        acc, num = ftp.fast_topp_rs_verify(
            tgt, temps_tok, tps_tok, drafts, slots, -1, buf, seeds, offs
        )
        # The nucleus the kernel used vs the oracle's (checked) and flashinfer's (counted).
        kept_k, rec = kernel_kept(tgt, temps_tok, tps_tok)
        assert bool((rec[:, 3] == 0).all())
        nuc = Nucleus(tgt, temps_tok, tps_tok)
        tprobs = compute_probs_from_logits(tgt, temps_tok, None, tps_tok)
        kept_ref = tprobs > 0
        for r in (
            ((kept_k != nuc.kept).any(1) | (kept_k != kept_ref).any(1)).nonzero().flatten().tolist()
        ):
            assert support_diff_ok(nuc, r, kept_k[r], nuc.kept[r])
            assert fi_support_ok(nuc, r, float(tps_tok[r]), kept_k[r], kept_ref[r])
            worst_err = max(worst_err, target_error(nuc, r, kept_k[r]))
            if float(tps_tok[r]) < 1.0:
                worst_fi_err = max(worst_fi_err, target_error(nuc, r, kept_ref[r]))
            support_rows += 1
        _, ref_num = rejection_sampling_one_model(
            buf[slots], drafts, tprobs.view(num_gens, K + 1, V), deterministic=True, seed=seeds,
            offset=offs,
        )  # fmt: skip
        p = nuc.probs(kept_k).reshape(num_gens, K + 1, V).cpu().numpy()
        q = buf[slots].double().cpu().numpy()
        dr = drafts.cpu().numpy()
        for b in (num.long() != ref_num.long()).nonzero().flatten().tolist():
            i = min(int(num[b]), int(ref_num[b])) - 1
            d = int(dr[b, i])
            u = curand_uniform(seed, offset, b, i)
            margin = abs(u * q[b, i, d] - p[b, i, d]) / max(p[b, i, d], 1e-300)
            if margin <= _ACCEPT_TOL:
                decisions.add("accept", margin)
                continue
            r = b * (K + 1) + i
            ok = fi_support_ok(nuc, r, float(tps_tok[r]), kept_k[r], kept_ref[r])
            decisions.add("nucleus", 1e-12 if ok else 1.0)
        _check_verify(
            acc.cpu().numpy(), num.cpu().numpy(), p, q, dr, [seed] * num_gens, [offset] * num_gens,
            K, model,
        )  # fmt: skip
        total += num_gens
        same_num += int((num == ref_num).sum())
        for n in num.tolist():
            hist[n - 1] += 1
    print(
        f"verify {kind} bias={accept_bias}: {total} rows; accept counts equal to flashinfer's in "
        f"{same_num}; differences explained by rounding in {decisions.count} rows (worst accept "
        f"{decisions.worst['accept']:.2e} rel); vs the float64 model {model.count} rows (worst "
        f"accept {model.worst['accept']:.2e}, token {model.worst['token']:.2e}); rows with a "
        f"rounding-level nucleus difference {support_rows} (worst target error: kernel {worst_err:.2e}, "
        f"flashinfer {worst_fi_err:.2e}); accepted-length histogram {hist.tolist()}"
    )
    assert total >= 2000
    decisions.check(total)
    model.check(total)


def test_per_request_seeds_offsets_temperatures_and_top_ps():
    """Each request keys its own stream (64-bit negative seeds, offsets past 2**32) and carries its
    own temperature and top-p (per request for the draft, per row for the target)."""
    gen = torch.Generator(device=DEV).manual_seed(3)
    seeds_py = [1, 2**40 + 7, -5, 0x7FFFFFFFFFFFFFF0, 99, -(2**62), 12345, 31337]
    offs_py = [0, 6, 2**33 + 3, 18, 7, 2**32 - 1, 120, 61]
    g = len(seeds_py)
    seeds = torch.tensor(seeds_py, dtype=torch.int64, device=DEV)
    offs = torch.tensor(offs_py, dtype=torch.int64, device=DEV)
    drafts_gaps, verify_gaps = _Gaps(), _Gaps()
    for trial in range(4):
        tgt, drf = _correlated_case(g, gen, 0.8)
        temps_req = 0.6 + 0.8 * torch.rand(g, generator=gen, device=DEV)
        tps_req = _request_top_ps(g, trial)
        temps_tok = temps_req.repeat_interleave(K + 1).contiguous()
        tps_tok = tps_req.repeat_interleave(K + 1).contiguous()
        slots = torch.randperm(g + 2, device=DEV)[:g].to(torch.int64)
        buf = torch.zeros((g + 2, K, V), dtype=torch.float32, device=DEV)
        drafts = ftp.fast_topp_rs_draft_block(
            drf.view(-1, V), temps_req, tps_req, slots, -1, buf, seeds, offs, 1
        )
        drafts = drafts.view(g, K).contiguous()
        q = buf[slots].double().cpu().numpy()
        nuc_d = Nucleus(
            drf.view(-1, V), temps_req.repeat_interleave(K), tps_req.repeat_interleave(K)
        )
        want_q = nuc_d.probs().reshape(g, K, V).cpu().numpy()
        # Equal except where the oracle's pivot is within rounding of a boundary.
        for r in range(g * K):
            got = torch.from_numpy(q[r // K, r % K] > 0).to(DEV)
            same = bool((got == nuc_d.kept[r]).all())
            assert support_diff_ok(nuc_d, r, got, nuc_d.kept[r]), (r, float(nuc_d.gap[r]))
            if same:
                assert np.abs(q[r // K, r % K] - want_q[r // K, r % K]).max() < 1e-5
        _check_drafts(drafts.cpu().numpy(), q, seeds_py, offs_py, K, drafts_gaps)
        acc, num = ftp.fast_topp_rs_verify(
            tgt, temps_tok, tps_tok, drafts, slots, -1, buf, seeds, offs
        )
        kept_k, rec = kernel_kept(tgt, temps_tok, tps_tok)
        assert bool((rec[:, 3] == 0).all())
        p = Nucleus(tgt, temps_tok, tps_tok).probs(kept_k).reshape(g, K + 1, V).cpu().numpy()
        _check_verify(
            acc.cpu().numpy(), num.cpu().numpy(), p, q, drafts.cpu().numpy(), seeds_py, offs_py, K,
            verify_gaps,
        )  # fmt: skip
    print(f"per-request streams: draft worst {drafts_gaps.worst}, verify worst {verify_gaps.worst}")
    drafts_gaps.check(4 * g * K)
    verify_gaps.check(4 * g)


def test_masked_logits_and_padding_requests():
    """-inf logits (guided-decoding style masks) get probability 0 -- also in chunks with no finite
    logit at all -- and the outputs stay inside the allowed set; a padding request (slot ==
    dummy_row) reads as uniform zeros, whose nucleus is every token (one tied group)."""
    gen = torch.Generator(device=DEV).manual_seed(4)
    g = 5
    chunk = ftp._CHUNK
    pool = torch.cat(
        [
            torch.arange(c * chunk, (c + 1) * chunk)
            for c in range(V // chunk)
            if c not in (1, 6, 7, 38, 39)
        ]
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
    dummy = g  # the last request is CUDA-graph padding (slot g)
    slots = torch.arange(g, dtype=torch.int64, device=DEV)
    slots[-1] = dummy
    buf = torch.zeros((g + 1, K, V), dtype=torch.float32, device=DEV)
    temps_req = torch.ones(g, device=DEV)
    temps_tok = torch.ones(g * (K + 1), device=DEV)
    tps_req = torch.full((g,), 0.9, device=DEV)
    tps_tok = torch.full((g * (K + 1),), 0.9, device=DEV)
    drafts = ftp.fast_topp_rs_draft_block(
        drf.view(-1, V), temps_req, tps_req, slots, dummy, buf, seeds, offs, 1
    )
    drafts = drafts.view(g, K).contiguous()
    assert not torch.isnan(buf).any()
    assert bool(keep[drafts[: g // 2].long()].all())
    assert float(buf[: g // 2][:, :, ~keep].abs().max()) == 0.0
    real = slice(0, g - 1)
    # The padding request's proposals are the uniform distribution over the whole vocabulary.
    assert bool(torch.allclose(buf[dummy], torch.full_like(buf[dummy], 1.0 / V), rtol=2e-5))
    acc, num = ftp.fast_topp_rs_verify(
        tgt, temps_tok, tps_tok, drafts, slots, dummy, buf, seeds, offs
    )
    for b in range(g - 1):
        emitted = acc[b, : int(num[b])].long()
        assert bool(keep[emitted].all()), (b, emitted.tolist())
    gaps = _Gaps()
    kept_k, rec = kernel_kept(tgt, temps_tok, tps_tok)
    assert bool((rec[:, 3] == 0).all())
    p = Nucleus(tgt, temps_tok, tps_tok).probs(kept_k).reshape(g, K + 1, V).cpu().numpy()
    q = buf[slots].double().cpu().numpy()
    _check_verify(
        acc[real].cpu().numpy(), num[real].cpu().numpy(), p[real], q[real],
        drafts[real].cpu().numpy(), [21] * (g - 1), [30] * (g - 1), K, gaps,
    )  # fmt: skip
    gaps.check(g - 1)


def test_short_draft_and_ragged_vocab():
    """Runtime draft length 3 in a 5-step buffer, a vocabulary that is not a tile / chunk multiple,
    per-row temperatures and top-p."""
    gen = torch.Generator(device=DEV).manual_seed(6)
    vocab, k, g = 10000, 3, 6
    drafts_gaps, verify_gaps = _Gaps(), _Gaps()
    for trial in range(6):
        tgt, drf = _correlated_case(g, gen, 0.6, k=k, vocab=vocab)
        temps_req = 0.5 + torch.rand(g, generator=gen, device=DEV)
        tps_req = _request_top_ps(g, trial)
        temps_tok = temps_req.repeat_interleave(k + 1).contiguous()
        tps_tok = tps_req.repeat_interleave(k + 1).contiguous()
        seeds_py, offs_py = [77 + trial] * g, [(k + 1) * trial] * g
        seeds = torch.tensor(seeds_py, dtype=torch.int64, device=DEV)
        offs = torch.tensor(offs_py, dtype=torch.int64, device=DEV)
        slots = torch.arange(g, dtype=torch.int64, device=DEV)
        buf = torch.zeros((g + 1, K, vocab), dtype=torch.float32, device=DEV)
        drafts = ftp.fast_topp_rs_draft_block(
            drf.view(-1, vocab), temps_req, tps_req, slots, -1, buf, seeds, offs, 1
        )
        drafts = drafts.view(g, k).contiguous()
        q = buf[slots, :k].double().cpu().numpy()
        _check_drafts(drafts.cpu().numpy(), q, seeds_py, offs_py, k, drafts_gaps)
        acc, num = ftp.fast_topp_rs_verify(
            tgt, temps_tok, tps_tok, drafts, slots, -1, buf, seeds, offs
        )
        kept_k, rec = kernel_kept(tgt, temps_tok, tps_tok)
        assert bool((rec[:, 3] == 0).all())
        p = Nucleus(tgt, temps_tok, tps_tok).probs(kept_k).reshape(g, k + 1, vocab).cpu().numpy()
        _check_verify(
            acc.cpu().numpy(), num.cpu().numpy(), p, q, drafts.cpu().numpy(), seeds_py, offs_py, k,
            verify_gaps,
        )  # fmt: skip
    drafts_gaps.check(6 * g * k)
    verify_gaps.check(6 * g)


def test_top_p_one_reproduces_the_temperature_kernels():
    """With top-p 1 the nucleus kernels draw the same Philox words as the temperature-only
    kernels (accept i -> word i, residual / bonus -> word K, draft -> the draft key's word 0), so
    their outputs agree wherever float rounding does not decide a draw."""
    gen = torch.Generator(device=DEV).manual_seed(8)
    g = 16
    tgt, drf = _correlated_case(g, gen, 0.6)
    temps_req = 0.7 + 0.5 * torch.rand(g, generator=gen, device=DEV)
    temps_tok = temps_req.repeat_interleave(K + 1).contiguous()
    ones_req, ones_tok = torch.ones(g, device=DEV), torch.ones(g * (K + 1), device=DEV)
    seeds = torch.arange(g, dtype=torch.int64, device=DEV) * 7919 + 3
    offs = torch.arange(g, dtype=torch.int64, device=DEV) * (K + 1)
    slots = torch.arange(g, dtype=torch.int64, device=DEV)
    buf_a = torch.zeros((g + 1, K, V), dtype=torch.float32, device=DEV)
    buf_b = torch.zeros_like(buf_a)
    tok_a = frs.fast_rs_draft_block(drf.view(-1, V), temps_req, slots, -1, buf_a, seeds, offs, 1)
    tok_b = ftp.fast_topp_rs_draft_block(
        drf.view(-1, V), temps_req, ones_req, slots, -1, buf_b, seeds, offs, 1
    )
    rel = ((buf_a - buf_b).abs() / buf_a.clamp_min(1e-30))[buf_a > 1e-6].max()
    assert float(rel) < 1e-5
    assert int((tok_a != tok_b).sum()) <= 1
    drafts = tok_a.view(g, K).contiguous()
    acc_a, num_a = frs.fast_rs_verify(tgt, temps_tok, drafts, slots, -1, buf_a, seeds, offs)
    acc_b, num_b = ftp.fast_topp_rs_verify(
        tgt, temps_tok, ones_tok, drafts, slots, -1, buf_a, seeds, offs
    )
    assert int((num_a != num_b).sum()) <= 1
    both = num_a == num_b
    assert int((acc_a[both] != acc_b[both]).any(1).sum()) <= 1


def test_philox_layout_never_reuses_a_draw():
    """Over consecutive steps (offset window max_draft_len + 1, as
    _populate_request_rng_state advances it), no (key, subsequence, word) is drawn twice, even
    with every request on one seed: the layout of the nucleus kernels is that of the temperature-only kernels (see
    test_top_p_one_reproduces_the_temperature_kernels)."""
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
