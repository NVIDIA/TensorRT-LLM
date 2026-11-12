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
"""Tests of the fused block verification (``TRTLLM_SPEC_BLOCK_VERIFICATION=1``).

The kernel tests need a CUDA device, or the Triton interpreter on CPU::

    TRITON_INTERPRET=1 python -m pytest tests/unittest/_torch/speculative/hw_agnostic/\
        test_block_verification_fused.py

What is checked, in the order of the claims:

1. The gate is off unless the environment variable is ``1``, and the engagement poll then does nothing.
2. The fused kernels (``block_verify_one_model``) make the same decisions as the dense torch rule
   of ``block_verify`` from the same random numbers (the in-kernel hash is ``counter_uniforms`` bit
   for bit), on random cases that include zero-mass entries, a proposal supported on a few tokens of
   the vocabulary, drafts the proposal gives no mass, non-finite rows and out-of-range draft ids.
3. Verified blocks completed by sampling the target are distributed exactly as the target's own
   samples (chi-square against the exact sequence probabilities) when the kernels themselves drive
   the draws. A check run on a proposal the drafts were not drawn from must fail.
4. Greedy rows (one-hot target, and one-hot proposal = the drafter's argmax) give the target's
   argmax chain, the same tokens as token-level verification, for every uniform.
5. ``verify_drafts`` routes batches up to ``TRTLLM_SPEC_BLOCK_VERIFICATION_MAX_BATCH`` to the
   fused kernels and larger ones to the token-level callable it is given, untouched.
"""

import math
import os

import pytest
import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.speculative import block_verification as bv

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
DEVICE = "cpu" if INTERPRET else "cuda"
requires_kernels = pytest.mark.skipif(
    not (INTERPRET or torch.cuda.is_available()),
    reason="the fused kernels need a CUDA device or TRITON_INTERPRET=1",
)


def test_gate_is_off_unless_set_to_one(monkeypatch):
    monkeypatch.delenv(bv._ENV, raising=False)
    assert not bv.block_verification_enabled()
    for value, expected in (("0", False), ("", False), ("true", False), ("1", True)):
        monkeypatch.setenv(bv._ENV, value)
        assert bv.block_verification_enabled() is expected


def test_poll_is_a_no_op_with_the_gate_off(monkeypatch):
    monkeypatch.delenv(bv._ENV, raising=False)
    polls, stats = bv._POLLS, dict(bv._STATS)
    bv.poll_block_verification_stats()  # must not touch the device or the counters
    assert bv._POLLS == polls and bv._STATS == stats


@pytest.fixture(autouse=True)
def small_blocks(monkeypatch):
    # Several chunks and several blocks per chunk at test vocabularies.
    monkeypatch.setattr(bv, "_BLOCK", 32)


def _keys(n, base=0):
    return (
        torch.arange(n, dtype=torch.int64) * 7919 + 13 + base,
        torch.arange(n, dtype=torch.int64) * 32 + base * 1000,
    )


def _fused(tokens, dp, tp, seed, offset):
    a, n = bv.block_verify_one_model(
        tokens.int().contiguous().to(DEVICE),
        dp.float().to(DEVICE),
        tp.float().to(DEVICE),
        seed.to(DEVICE),
        offset.to(DEVICE),
    )
    return a.cpu(), n.cpu()


def _ref(tokens, dp, tp, seed, offset):
    return bv.block_verify_reference(tokens.int(), dp.float(), tp.float(), seed, offset)


# ---------------------------------------------------------------------------
# 0. the hash
# ---------------------------------------------------------------------------


@triton.jit
def _draw_kernel(seed_ptr, offset_ptr, out_ptr, K: tl.constexpr):
    b = tl.program_id(0)
    seed = tl.load(seed_ptr + b).to(tl.int64)
    offset = tl.load(offset_ptr + b).to(tl.int64)
    key = bv._mix32((seed & 0xFFFFFFFF).to(tl.uint32) ^ 0x5BD1E995)
    key = bv._mix32(key ^ ((seed >> 32) & 0xFFFFFFFF).to(tl.uint32))
    key = bv._mix32(key ^ (offset & 0xFFFFFFFF).to(tl.uint32))
    key = bv._mix32(key ^ ((offset >> 32) & 0xFFFFFFFF).to(tl.uint32))
    for d in tl.static_range(K + 1):
        tl.store(out_ptr + b * (K + 1) + d, bv._draw(key, d))


@requires_kernels
def test_kernel_hash_is_counter_uniforms_bit_for_bit():
    seed = torch.tensor(
        [0, 1, 42, 2**31 - 1, 2**33 + 5, 2**40 + 123456789, 12345678901234, -1, -(2**40 + 3)],
        dtype=torch.int64,
    )
    offset = torch.tensor(
        [0, 32, 2**32 - 1, 2**32, 2**35 + 7, 10**15, 3, -7, -(2**36)], dtype=torch.int64
    )
    out = torch.empty(len(seed), 8, dtype=torch.float64, device=DEVICE)
    _draw_kernel[(len(seed),)](seed.to(DEVICE), offset.to(DEVICE), out, K=7)
    assert torch.equal(out.cpu(), bv.counter_uniforms(seed, offset, 8))


def test_counter_uniforms_are_uniform_and_distinct_per_key():
    seed, offset = _keys(100_000)
    u = bv.counter_uniforms(seed, offset, 8)
    assert u.min() > 0 and u.max() < 1
    assert abs(u.mean().item() - 0.5) < 3e-3
    # nothing shared between neighbouring requests, steps or draws
    assert (
        abs(
            torch.corrcoef(torch.stack([u[:, 0], u[:, 1], u[:, 7], u[:, 0].roll(1)]))[0, 1:]
            .abs()
            .max()
            .item()
        )
        < 0.02
    )


# ---------------------------------------------------------------------------
# 1. fused == dense, same random numbers
# ---------------------------------------------------------------------------


def _case(generator, batch, k, vocab, support=None, zero_frac=0.2):
    """Target rows with zeros, draft rows (optionally supported on `support` ids), drafts drawn from them."""
    tl_ = 2.0 * torch.randn(batch, k + 1, vocab, generator=generator)
    tl_ = tl_.masked_fill(
        torch.rand(batch, k + 1, vocab, generator=generator) < zero_frac, -float("inf")
    )
    tl_[..., 0] = 0.0
    tp = torch.softmax(tl_.double(), -1)
    dl = tl_[:, :k].nan_to_num(neginf=-4.0) + torch.randn(batch, k, vocab, generator=generator)
    if support is not None:
        keep = torch.zeros(batch, k, vocab, dtype=torch.bool)
        top = torch.rand(batch, k, vocab, generator=generator).topk(support, -1).indices
        keep.scatter_(-1, top, True)
        dl = dl.masked_fill(~keep, -float("inf"))
    dp = torch.softmax(dl.double(), -1)
    tokens = torch.multinomial(dp.reshape(-1, vocab), 1, generator=generator).view(batch, k)
    return tokens, dp.float(), tp.float()


@requires_kernels
@pytest.mark.parametrize("k", [1, 3, 7])
@pytest.mark.parametrize("vocab,support", [(24, None), (200, 16), (1000, 16), (1000, None)])
def test_fused_matches_dense_given_identical_randoms(k, vocab, support):
    generator = torch.Generator().manual_seed(100 + k + vocab)
    batch = 24
    tokens, dp, tp = _case(generator, batch, k, vocab, support)
    seed, offset = _keys(batch, base=k)
    ra, rn = _ref(tokens, dp, tp, seed, offset)
    ka, kn = _fused(tokens, dp, tp, seed, offset)
    assert torch.equal(rn, kn), (rn, kn)
    positions = torch.arange(k + 1)
    accepted = positions[None, :] < (rn - 1)[:, None]
    assert torch.equal(ra[accepted], ka[accepted])  # accepted drafts: exact
    assert torch.equal(ra < 0, ka < 0)  # layout: -1 padding
    emitted = ka.gather(1, (kn - 1).long()[:, None]).squeeze(1).long()
    # the emitted token is always one the residual gives mass to
    last = (rn - 1).long()
    p_row = tp[torch.arange(batch), last].double()
    q_row = dp[torch.arange(batch), last.clamp(max=k - 1)].double()
    log_w = bv.block_acceptance_thresholds(tokens, dp, tp)[0]
    w_prev = torch.cat([torch.zeros(batch, 1), log_w], 1).gather(1, last[:, None]).exp().double()
    weights = torch.where((last == k)[:, None], p_row, (w_prev * p_row - q_row).clamp_min(0))
    assert (weights.gather(1, emitted[:, None]) > 0).all()
    # and it is the dense inverse-CDF draw except at fp32 rounding boundaries
    assert (emitted == ra.gather(1, last[:, None]).squeeze(1).long()).float().mean() >= 0.95


@requires_kernels
def test_fused_matches_dense_many_small_cases():
    generator = torch.Generator().manual_seed(77)
    agree = total = 0
    for trial in range(40):
        k = int(torch.randint(1, 8, (1,), generator=generator))
        vocab = int(torch.randint(2, 60, (1,), generator=generator))
        support = None if vocab < 8 or trial % 2 else min(vocab - 1, 16)
        batch = 16
        tokens, dp, tp = _case(generator, batch, k, vocab, support, zero_frac=0.3)
        seed, offset = _keys(batch, base=trial)
        ra, rn = _ref(tokens, dp, tp, seed, offset)
        ka, kn = _fused(tokens, dp, tp, seed, offset)
        assert torch.equal(rn, kn)
        last = (rn - 1).long()
        agree += int((ra.gather(1, last[:, None]) == ka.gather(1, last[:, None])).sum())
        total += batch
    assert agree / total >= 0.98


@requires_kernels
def test_degenerate_rows_stay_in_vocab_and_in_range():
    generator = torch.Generator().manual_seed(5)
    batch, k, vocab = 8, 3, 50
    tokens, dp, tp = _case(generator, batch, k, vocab, 16)
    tp[0] = float("nan")  # non-finite target rows
    dp[1] = float("nan")  # non-finite draft rows
    tp[2] = 0.0  # no mass anywhere
    dp[3] = 0.0  # the draft's proposal gives it no mass
    tokens[4, 1] = vocab + 40  # out-of-range draft id (padding request)
    tokens[5, 0] = -3
    seed, offset = _keys(batch)
    ka, kn = _fused(tokens, dp, tp, seed, offset)
    assert ((kn >= 1) & (kn <= k + 1)).all()
    assert ((ka >= -1) & (ka < vocab)).all()
    emitted = ka.gather(1, (kn - 1).long()[:, None]).squeeze(1)
    assert ((emitted >= 0) & (emitted < vocab)).all()
    assert kn[3] == 1  # q_x = 0: rejected at the first position
    assert kn[3] == _ref(tokens, dp, tp, seed, offset)[1][3]


@requires_kernels
def test_unaligned_request_vectors_are_copied_and_give_the_same_result():
    # The engine passes views at the first generation row: an odd start is not 16-byte aligned.
    generator = torch.Generator().manual_seed(15)
    batch, k, vocab = 6, 3, 60
    tokens, dp, tp = _case(generator, batch, k, vocab, 16)
    big_seed = torch.arange(batch + 3, dtype=torch.int64) * 7919 + 13
    big_off = torch.arange(batch + 3, dtype=torch.int64) * 32
    big_slots = torch.arange(batch + 3, dtype=torch.int64)
    for start in (0, 1, 3):
        seed, off, slots = (
            big_seed[start : start + batch],
            big_off[start : start + batch],
            big_slots[start : start + batch],
        )
        assert (seed.data_ptr() % 16 == 0) == (start % 2 == 0)
        assert bv._aligned(seed).data_ptr() % 16 == 0 and bv._aligned(slots).data_ptr() % 16 == 0
        got = _fused(tokens, dp, tp, seed, off)
        want = _fused(tokens, dp, tp, seed.clone(), off.clone())
        assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])
    # an aligned vector is passed through, not copied
    assert bv._aligned(big_seed) is big_seed


@requires_kernels
def test_engagement_counters_skip_padding_rows_and_count_zero_mass_drafts():
    generator = torch.Generator().manual_seed(16)
    batch, k, vocab = 8, 3, 40
    tokens, dp, tp = _case(generator, batch, k, vocab, 16)
    dp[2, 1] = 0.0  # a stale row: the drafted token has no mass in it
    dp[5, 0] = 0.0  # a padding row (below): not counted
    slots = torch.tensor([10, 11, 12, 13, 99, 99, 14, 15], dtype=torch.int64)  # 99 = dummy slot
    key = torch.empty(0, device=DEVICE).device  # "cuda:0", not "cuda": the kernels' key
    bv._STATS[key] = torch.zeros(4, dtype=torch.int64, device=DEVICE)
    seed, off = _keys(batch)
    for _ in range(3):
        a, n = bv.block_verify_one_model(
            tokens.int().contiguous().to(DEVICE),
            dp.float().to(DEVICE),
            tp.float().to(DEVICE),
            seed.to(DEVICE),
            off.to(DEVICE),
            slot_ids=slots.to(DEVICE),
            dummy_slot=99,
        )
    steps, rows, accepted, no_mass = bv._STATS.pop(key).tolist()
    keep = slots != 99
    assert (steps, rows) == (3, 3 * int(keep.sum()))
    assert accepted == 3 * int(n.cpu()[keep].sum())
    # row 2: the draft at position 1 has q = 0 (the same key every call, so the same count); the
    # other zero-mass drafts of that row come from its other positions, which are fine
    assert no_mass == 3 * int(
        (dp[keep].gather(-1, tokens[keep].long()[..., None]).squeeze(-1) <= 0).sum()
    )
    assert no_mass >= 3


@requires_kernels
def test_router_joint_up_to_the_limit_token_level_above_and_counters_say_which(monkeypatch):
    monkeypatch.delenv(bv._MAX_BATCH_ENV, raising=False)
    assert bv.block_verification_max_batch() == 48
    monkeypatch.setenv(bv._MAX_BATCH_ENV, "4")
    assert bv.block_verification_max_batch() == 4
    calls = []

    def token_level(draft_probs, draft_token_ids, target_probs, deterministic, seed, offset):
        calls.append(draft_token_ids.shape[0])
        return _ref(
            draft_token_ids, draft_probs, target_probs, seed, offset
        )  # any valid (tokens, num): the stub's rule is not under test

    key = torch.empty(0, device=DEVICE).device  # "cuda:0", not "cuda": the kernels' key
    bv._STATS[key] = torch.zeros(7, dtype=torch.int64, device=DEVICE)
    generator = torch.Generator().manual_seed(23)
    k, vocab = 3, 40
    want = torch.zeros(7, dtype=torch.int64)
    for batch in (3, 4, 5, 7):
        tokens, dp, tp = _case(generator, batch, k, vocab, 16)
        seed, off = _keys(batch)
        slots = torch.arange(batch, dtype=torch.int64)
        slots[-1] = 99  # the dummy slot: not counted
        a, n = bv.verify_drafts(
            tokens.int().contiguous().to(DEVICE),
            dp.float().to(DEVICE),
            tp.float().to(DEVICE),
            seed.to(DEVICE),
            off.to(DEVICE),
            slots.to(DEVICE),
            99,
            token_level=token_level,
        )
        real = batch - 1
        acc = int(n.cpu()[:real].sum())
        if batch <= 4:
            fa, fn = _fused(tokens, dp, tp, seed, off)  # the router's answer is the fused kernels'
            assert torch.equal(a.cpu(), fa) and torch.equal(n.cpu(), fn)
            want[0:3] += torch.tensor([1, real, acc])
        else:
            ra, rn = _ref(
                tokens.int(), dp.float(), tp.float(), seed, off
            )  # the token-level callable's answer, untouched
            assert torch.equal(a.cpu(), ra) and torch.equal(n.cpu(), rn)
            want[4:7] += torch.tensor([1, real, acc])
    assert calls == [5, 7]  # only the batches above the limit reach it
    assert bv._STATS.pop(key).tolist() == want.tolist()


@requires_kernels
def test_identical_draft_and_target_accepts_everything():
    generator = torch.Generator().manual_seed(9)
    batch, k, vocab = 16, 4, 30
    tokens, dp, tp = _case(generator, batch, k, vocab, None, zero_frac=0.0)
    tp = torch.cat([dp, tp[:, k:]], 1)
    ka, kn = _fused(tokens, dp, tp, *_keys(batch))
    assert (kn == k + 1).all()
    assert torch.equal(ka[:, :k], tokens.int())


# ---------------------------------------------------------------------------
# 2. lossless: completed sequences are distributed as the target's
# ---------------------------------------------------------------------------

_VOCAB = 3
_BLOCK = 3


class _PrefixModel:
    """A next-token distribution for every prefix of length 0.._BLOCK; `support` limits a row to that many tokens."""

    def __init__(self, generator, zero_frac=0.0, sharpness=1.0, support=None, vocab=_VOCAB):
        self.vocab = vocab
        n = sum(vocab**m for m in range(_BLOCK + 1))
        logits = sharpness * torch.randn(n, vocab, generator=generator)
        zero = torch.rand(n, vocab, generator=generator) < zero_frac
        if support is not None:
            top = torch.rand(n, vocab, generator=generator).topk(support, -1).indices
            keep = torch.zeros(n, vocab, dtype=torch.bool)
            keep.scatter_(-1, top, True)
            zero |= ~keep
        zero[zero.all(-1), 0] = False
        self.table = torch.softmax(logits.masked_fill(zero, float("-inf")).double(), -1)

    def probs(self, tokens):
        base = sum(self.vocab**m for m in range(tokens.shape[1]))
        ids = torch.zeros(tokens.shape[0], dtype=torch.int64)
        for m in range(tokens.shape[1]):
            ids = ids * self.vocab + tokens[:, m]
        return self.table[ids + base]

    def sequence_probs(self):
        grid = torch.cartesian_prod(*[torch.arange(self.vocab)] * (_BLOCK + 1))
        prob = torch.ones(grid.shape[0], dtype=torch.float64)
        for m in range(_BLOCK + 1):
            prob = prob * self.probs(grid[:, :m]).gather(-1, grid[:, m : m + 1]).squeeze(-1)
        return prob


def _verify_and_complete(verify, target, draft, n, generator, base=0):
    drafts = torch.empty(n, 0, dtype=torch.int64)
    rows = []
    for _ in range(_BLOCK):
        q = draft.probs(drafts)
        rows.append(q)
        drafts = torch.cat([drafts, torch.multinomial(q, 1, generator=generator)], 1)
    target_rows = [target.probs(drafts[:, :j]) for j in range(_BLOCK + 1)]
    dp, tp = torch.stack(rows, 1).float(), torch.stack(target_rows, 1).float()
    seed, offset = _keys(n, base)
    tokens, num = verify(drafts, dp, tp, seed, offset)
    seqs = torch.zeros(n, _BLOCK + 1, dtype=torch.int64)
    emitted = num.long()
    for m in range(_BLOCK + 1):
        cont = torch.multinomial(target.probs(seqs[:, :m]), 1, generator=generator).squeeze(-1)
        seqs[:, m] = torch.where(m < emitted, tokens[:, m].long(), cont)
    ids = torch.zeros(n, dtype=torch.int64)
    for m in range(_BLOCK + 1):
        ids = ids * target.vocab + seqs[:, m]
    return torch.bincount(ids, minlength=target.vocab ** (_BLOCK + 1)), emitted


def _chi_square(counts, probs):
    total = counts.sum().item()
    expected = probs * total
    big = expected >= 5
    observed = torch.cat([counts[big].double(), counts[~big].double().sum().view(1)])
    expected = torch.cat([expected[big], expected[~big].sum().view(1)])
    keep = expected > 0
    stat = (((observed - expected) ** 2)[keep] / expected[keep]).sum().item()
    dof = int(keep.sum()) - 1
    z = 3.719  # one-sided 1e-4 normal quantile, Wilson-Hilferty
    return stat, dof * (1 - 2 / (9 * dof) + z * math.sqrt(2 / (9 * dof))) ** 3


def _dense(drafts, dp, tp, seed, offset):
    return _ref(drafts, dp, tp, seed, offset)


@requires_kernels
def test_fused_kernels_drive_the_draws_losslessly():
    # The kernel itself, on the interpreter: 6000 blocks, 81 sequences (expected ~74 each).
    generator = torch.Generator().manual_seed(41)
    target = _PrefixModel(generator, zero_frac=0.2, sharpness=1.5)
    draft = _PrefixModel(generator, sharpness=1.0, support=2)

    def fused(drafts, dp, tp, seed, offset):
        return _fused(drafts, dp, tp, seed, offset)

    n = 6000 if INTERPRET else 300_000
    counts, _ = _verify_and_complete(fused, target, draft, n, generator)
    exact = target.sequence_probs()
    assert counts[exact == 0].sum() == 0
    stat, critical = _chi_square(counts, exact)
    assert stat < critical, f"chi-square {stat:.1f} >= {critical:.1f}"


def test_guard_a_q_the_drafts_were_not_drawn_from_fails_the_check():
    generator = torch.Generator().manual_seed(3)
    target = _PrefixModel(generator, sharpness=1.5)
    draft = _PrefixModel(generator, sharpness=1.0)

    def wrong_q(drafts, dp, tp, seed, offset):
        return _ref(
            drafts, torch.softmax(torch.log(dp.double()) * 0.3, -1).float(), tp, seed, offset
        )

    counts, _ = _verify_and_complete(wrong_q, target, draft, 300_000, generator)
    stat, critical = _chi_square(counts, target.sequence_probs())
    assert stat > critical


# ---------------------------------------------------------------------------
# 3. greedy rows
# ---------------------------------------------------------------------------


def _token_level_greedy(drafts, tp_argmax):
    """Token-level verification of one-hot rows: accept while the draft is the target's argmax."""
    batch, k = drafts.shape
    out = torch.full((batch, k + 1), -1, dtype=torch.int32)
    num = torch.empty(batch, dtype=torch.int32)
    for b in range(batch):
        length = 0
        while length < k and int(drafts[b, length]) == int(tp_argmax[b, length]):
            out[b, length] = drafts[b, length]
            length += 1
        out[b, length] = tp_argmax[b, length]
        num[b] = length + 1
    return out, num


@requires_kernels
@pytest.mark.parametrize("k", [1, 3, 7])
@pytest.mark.parametrize("temperature_softmax", [False, True])
def test_greedy_rows_give_the_target_argmax_chain(k, temperature_softmax):
    generator = torch.Generator().manual_seed(200 + k)
    batch, vocab = 96, 40
    logits = 3.0 * torch.randn(batch, k + 1, vocab, generator=generator)
    argmax = logits.argmax(-1)
    # drafter's argmax equals the target's for the first m positions, then differs (or all match)
    match = torch.randint(0, k + 2, (batch,), generator=generator)
    drafts = argmax[:, :k].clone()
    for b in range(batch):
        if match[b] <= k - 1:
            drafts[b, match[b]] = (
                argmax[b, match[b]]
                + 1
                + int(torch.randint(0, vocab - 1, (1,), generator=generator))
            ) % vocab
            drafts[b, match[b] + 1 :] = torch.randint(
                0, vocab, (k - match[b] - 1,), generator=generator
            )
    if temperature_softmax:
        # GREEDY_TEMPERATURE_THRESHOLD / 10 = 1e-5: what a greedy row of a mixed batch really carries
        tp = torch.softmax(logits / 1e-5, -1)
        q_logits = torch.zeros(batch, k, vocab).scatter_(-1, drafts[..., None], 5.0)
        dp = torch.softmax(q_logits / 1e-5, -1)
    else:
        tp = torch.zeros_like(logits).scatter_(-1, argmax[..., None], 1.0)
        dp = torch.zeros(batch, k, vocab).scatter_(-1, drafts[..., None], 1.0)
    assert ((tp == 1).sum(-1) == 1).all() and (
        (dp == 1).sum(-1) == 1
    ).all()  # exactly one-hot in fp32
    expected, expected_num = _token_level_greedy(drafts, argmax)
    for base in range(3):  # three different uniform streams
        seed, offset = _keys(batch, base=base)
        for fn in (_fused, _ref):
            ka, kn = fn(drafts, dp, tp, seed, offset)
            assert torch.equal(kn, expected_num)
            assert torch.equal(ka, expected)


@requires_kernels
def test_greedy_one_hot_target_against_sampled_proposal():
    # A greedy request keeps the target one-hot even if the drafter proposed a spread-out q.
    generator = torch.Generator().manual_seed(8)
    batch, k, vocab = 64, 4, 20
    logits = 3.0 * torch.randn(batch, k + 1, vocab, generator=generator)
    argmax = logits.argmax(-1)
    tp = torch.zeros_like(logits).scatter_(-1, argmax[..., None], 1.0)
    dp = torch.softmax(2 * torch.randn(batch, k, vocab, generator=generator), -1)
    drafts = torch.multinomial(dp.reshape(-1, vocab), 1, generator=generator).view(batch, k)
    seed, offset = _keys(batch)
    ka, kn = _fused(drafts, dp, tp, seed, offset)
    # whatever is accepted, the output is a prefix of the target's argmax chain
    for b in range(batch):
        n = int(kn[b])
        assert ka[b, :n].tolist() == argmax[b, :n].tolist()
