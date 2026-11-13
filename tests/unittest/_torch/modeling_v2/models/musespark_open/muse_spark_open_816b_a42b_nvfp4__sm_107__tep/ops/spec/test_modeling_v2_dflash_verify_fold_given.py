# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for ``dflash_verify_fold`` on given accepted tokens and counts (fast path ``rejection_fold``).

A rejection-sampled step hands the fold the built-in rejection sampler's
accepted tokens and counts (``ACCEPTED_GIVEN``); every output, the in-place
``kv_lens_cuda`` bump and the ``_seq_lens_cuda`` copy must be the built-in
chain's (the op's reference) bit for bit, and the forced-acceptance counter
must not move. Then the worker: ``SpecLoopFoldDFlashWorker`` folds a
rejection-sampled step only under ``rejection_fold`` and on a step the fold
takes, from the built-in method's own tensors.
"""

import types

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_107__tep.kernels.verify_fold import (  # noqa: E501
    fold as fold_mod,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_107__tep.ops.spec import (  # noqa: E501
    dflash_verify_fold as vf,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

K = 7
BLOCK = 8
MAX_CTX = 4096
MASK_ID = 200_003


def _inputs(gens, seed, with_buffers=True):
    g = torch.Generator(device="cuda").manual_seed(seed)
    counts = torch.randint(1, K + 2, (gens,), generator=g, device="cuda", dtype=torch.int32)
    tokens = torch.randint(0, 200_000, (gens, K + 1), generator=g, device="cuda", dtype=torch.int32)
    # The rejection kernel leaves -1 past each request's emitted tokens.
    cols = torch.arange(K + 1, device="cuda")
    tokens = torch.where(cols[None, :] < counts[:, None], tokens, -1).to(torch.int32).contiguous()
    drafts = torch.randint(0, 200_000, (gens, K), generator=g, device="cuda", dtype=torch.int32)
    slots = torch.randperm(gens + 5, generator=g, device="cuda")[:gens].contiguous()
    ctx_len = torch.randint(0, MAX_CTX, (gens + 5,), generator=g, device="cuda")
    kv_lens = seq_lens = None
    if with_buffers:
        kv_lens = torch.randint(1, 9000, (gens + 2,), generator=g, device="cuda", dtype=torch.int32)
        seq_lens = torch.randint(
            1, 9000, (gens + 2,), generator=g, device="cuda", dtype=torch.int32
        )
    return tokens, counts, drafts, slots, ctx_len, kv_lens, seq_lens


def _same(a, b):
    if a is None or b is None:
        return a is None and b is None
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)


@pytest.mark.parametrize("with_buffers", [True, False], ids=["buffers", "no-buffers"])
@pytest.mark.parametrize("gens", [1, 3, 16, vf.MAX_REQUESTS])
def test_given_counts_fold_is_the_builtin_chain(gens, with_buffers):
    tokens, counts, drafts, slots, ctx_len, kv_lens, seq_lens = _inputs(gens, gens, with_buffers)
    counter = torch.tensor([41], dtype=torch.int64, device="cuda")
    args = (tokens.reshape(-1), drafts, vf.ACCEPTED_GIVEN, 0, 0, 0.0, None, counter)
    tail = (slots, ctx_len, MAX_CTX, BLOCK, MASK_ID)
    ref_kv = None if kv_lens is None else kv_lens.clone()
    ref_seq = None if seq_lens is None else seq_lens.clone()
    vf.dflash_verify_fold.is_valid(*args, *tail, kv_lens, seq_lens, given_counts=counts)
    got = vf.dflash_verify_fold(*args, *tail, kv_lens, seq_lens, given_counts=counts)
    want = vf.dflash_verify_fold.reference(*args, *tail, ref_kv, ref_seq, given_counts=counts)
    torch.cuda.synchronize()
    for name in vf.VerifyFold._fields:
        assert _same(getattr(got, name), getattr(want, name)), name
    assert _same(kv_lens, ref_kv)
    # The accepted tokens and counts are the given ones; the counter does not move.
    assert torch.equal(got.accepted, tokens) and torch.equal(got.num_accepted, counts)
    assert counter.item() == 41


def _worker(monkeypatch, rejection, applies, accepted, num_accepted):
    """A never-initialized ``SpecLoopFoldDFlashWorker`` over a stubbed built-in rejection sampler."""
    calls, launches = [], []

    def builtin(self, *args):
        calls.append(args)
        return accepted, num_accepted

    def launch(step, target, drafts, nc, bs, mode, *rest, given_counts=None):
        launches.append((target, mode, given_counts))
        return types.SimpleNamespace(
            accepted=target.view_as(accepted).clone(), num_accepted=given_counts.clone()
        )

    monkeypatch.setattr(fold_mod.DFlashWorker, "_sample_and_accept_draft_tokens_rejection", builtin)
    worker = fold_mod.SpecLoopFoldDFlashWorker.__new__(fold_mod.SpecLoopFoldDFlashWorker)
    worker.__dict__.update(
        _fold_rejection=rejection,
        _fold_current=lambda: object(),
        _verify_fold_applies=lambda *args: applies,
        _launch_fold=launch,
    )
    return worker, calls, launches


@pytest.mark.parametrize("rejection, applies", [(True, True), (True, False), (False, True)])
def test_the_worker_folds_rejection_sampled_steps_under_rejection_fold(
    monkeypatch, rejection, applies
):
    tokens, counts, drafts, *_ = _inputs(3, 5)
    logits = torch.zeros(3 * (K + 1), 8, device="cuda")
    worker, calls, launches = _worker(monkeypatch, rejection, applies, tokens, counts)
    accepted, num_accepted = worker._sample_and_accept_draft_tokens_rejection(
        logits, drafts, None, 0, 3, None
    )
    assert len(calls) == 1
    if rejection and applies:
        ((target, mode, given),) = launches
        assert mode == vf.ACCEPTED_GIVEN and given is counts
        assert target.data_ptr() == tokens.data_ptr()
        assert torch.equal(accepted, tokens) and accepted is not tokens
        assert torch.equal(num_accepted, counts)
    else:
        assert not launches
        assert accepted is tokens and num_accepted is counts
