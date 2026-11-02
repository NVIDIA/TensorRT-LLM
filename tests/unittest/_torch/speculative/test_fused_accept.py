# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused strict acceptance and step-0 draft bookkeeping (``speculative/fused_accept.py``) must be bit-identical
to the stock ops."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _worker(force: float):
    from tensorrt_llm._torch.speculative.interface import SpecWorkerBase

    class _W(SpecWorkerBase):
        @property
        def max_draft_len(self) -> int:
            return 6

        def _forward_impl(self, *args, **kwargs):  # pragma: no cover - unused
            raise NotImplementedError

    w = _W()
    w.force_num_accepted_tokens = force
    return w


class _Meta:
    is_all_greedy_sample = True
    is_cuda_graph = False
    is_ragged_verify = False
    sharded_target_logits = False


def _inputs(num_contexts, num_gens, k, match_len, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    vocab = 1000
    target = torch.randint(
        0, vocab, (num_contexts + num_gens * (k + 1),), generator=g, device="cuda"
    )
    tgt_gen = target[num_contexts:].view(num_gens, k + 1)
    draft = torch.randint(0, vocab, (num_gens, k), generator=g, device="cuda", dtype=torch.int32)
    for b in range(num_gens):
        m = int(match_len[b % len(match_len)])
        draft[b, :m] = tgt_gen[b, :m].to(torch.int32)
        if m < k:
            draft[b, m] = (tgt_gen[b, m] + 1) % vocab  # a guaranteed mismatch
    # logits whose argmax is the target
    logits = torch.zeros((target.numel(), vocab), device="cuda")
    logits[torch.arange(target.numel(), device="cuda"), target] = 1.0
    return logits, draft


def _run(worker, logits, draft, num_contexts, batch_size, fused, monkeypatch):
    from tensorrt_llm._torch.speculative import interface

    # MAX_DRAFT_LEN 0 sends every batch down the stock (unfused) path.
    monkeypatch.setattr(interface, "MAX_DRAFT_LEN", 31 if fused else 0)
    acc, n = worker._sample_and_accept_draft_tokens_base(
        logits, draft, num_contexts, batch_size, _Meta()
    )
    torch.cuda.synchronize()
    return acc, n


@pytest.mark.parametrize("force", [0.0, 3.0, 3.8, 9.0])
@pytest.mark.parametrize("num_contexts,num_gens", [(0, 1), (2, 3), (0, 64), (3, 0)])
def test_fused_accept_matches_unfused(force, num_contexts, num_gens, monkeypatch):
    k = 6
    batch = num_contexts + num_gens
    ref_w, fused_w = _worker(force), _worker(force)
    for step in range(4):
        logits, draft = _inputs(
            num_contexts, max(num_gens, 1), k, [0, 1, 3, k, k - 1, 2], seed=step
        )
        draft = draft[:num_gens]
        logits = logits[: num_contexts + num_gens * (k + 1)]
        acc_r, n_r = _run(ref_w, logits, draft, num_contexts, batch, False, monkeypatch)
        acc_f, n_f = _run(fused_w, logits, draft, num_contexts, batch, True, monkeypatch)
        assert torch.equal(n_r, n_f), (step, n_r, n_f)
        # Context rows: only column 0 is defined.
        assert torch.equal(acc_r[:num_contexts, 0], acc_f[:num_contexts, 0])
        assert torch.equal(acc_r[num_contexts:], acc_f[num_contexts:])
        if ref_w._force_accept_rng_counter is not None:
            assert torch.equal(ref_w._force_accept_rng_counter, fused_w._force_accept_rng_counter)


@pytest.mark.parametrize(
    "num_contexts,num_gens,k,rows",
    [(3, 5, 6, 3), (2, 0, 7, 3), (0, 64, 8, 0), (1, 1, 1, 0)],
)
def test_step0_epilogue_matches_stock(num_contexts, num_gens, k, rows):
    # rows = 0: 1-D position ids, else [rows, tokens] MRoPE positions.
    from tensorrt_llm._torch.speculative.fused_accept import step0_epilogue

    g = torch.Generator(device="cuda").manual_seed(num_contexts * 1000 + num_gens * 10 + k)
    b, H = num_contexts + num_gens, 4096
    ctx_lens = torch.randint(1, 300, (num_contexts,), generator=g, device="cuda", dtype=torch.int32)
    seq_lens = torch.cat(
        [ctx_lens, torch.full((num_gens,), k + 1, device="cuda", dtype=torch.int32)]
    )
    num_acc = torch.randint(1, k + 2, (b,), generator=g, device="cuda", dtype=torch.int32)
    num_ctx_tokens = int(ctx_lens.sum())
    n_tok = num_ctx_tokens + num_gens * (k + 1)
    hidden = torch.randn((n_tok, H), generator=g, device="cuda").to(torch.bfloat16)
    pshape = (rows, n_tok) if rows else (n_tok,)
    pos = torch.randint(0, 200000, pshape, generator=g, device="cuda", dtype=torch.int32)
    kv = torch.randint(1000, 5000, (b + 3,), generator=g, device="cuda", dtype=torch.int32)
    # Stock: eagle3._forward_linear_draft_loop, i == 0, plus the later passes' position updates.
    last = torch.cumsum(seq_lens, dim=0, dtype=torch.long) - 1
    gen = (
        (torch.arange(num_gens, device="cuda", dtype=torch.int32) * (k + 1)).long()
        + num_acc[num_contexts:]
        - 1
    )
    ids_ref = torch.concat([last[:num_contexts], gen + num_ctx_tokens], dim=0)
    hid_ref = hidden[ids_ref]
    pos_ref = [pos[..., ids_ref] + j for j in range(1, k)]
    kv_ref = kv.clone()
    kv_ref[num_contexts:b] -= k - num_acc[num_contexts:]
    kv_ref[:num_contexts] += 1
    # Fused: reads the saved copy, writes the live tensors.
    seq_saved, seq_live, kv_live = (
        seq_lens.clone(),
        torch.full((b + 3,), 7, device="cuda", dtype=torch.int32),
        kv.clone(),
    )
    ep = step0_epilogue(
        seq_saved, num_acc, hidden, pos, seq_live, kv_live, num_contexts, b, k, num_ctx_tokens
    )
    torch.cuda.synchronize()
    assert ep is not None
    assert torch.equal(ep.gather_ids, ids_ref)
    assert torch.equal(ep.hidden, hid_ref)
    for j in range(1, k):
        got = ep.positions(j)
        assert got.shape == pos_ref[j - 1].shape and torch.equal(got, pos_ref[j - 1]), j
    assert torch.all(seq_live[:b] == 1) and torch.all(seq_live[b:] == 7)
    assert torch.equal(kv_live, kv_ref)
    assert torch.equal(seq_saved, seq_lens)  # the saved copy is never written
    # Aliasing the input it writes is refused (caller keeps the stock path).
    assert (
        step0_epilogue(
            seq_live, num_acc, hidden, pos, seq_live, kv_live, num_contexts, b, k, num_ctx_tokens
        )
        is None
    )
    # Draft tail: row 0 of the token buffer = stock _prepare_next_new_tokens column 0.
    acc_tok = torch.randint(0, 248320, (b, k + 3), generator=g, device="cuda", dtype=torch.int32)[
        :, : k + 2
    ]
    tok = torch.full((k + 1, b), -5, device="cuda", dtype=torch.int32)
    ep2 = step0_epilogue(
        seq_lens.clone(),
        num_acc,
        hidden,
        pos,
        torch.zeros_like(seq_live),
        kv.clone(),
        num_contexts,
        b,
        k,
        num_ctx_tokens,
        accepted_tokens=acc_tok,
        last_out=tok[0],
    )
    torch.cuda.synchronize()
    assert ep2 is not None
    ref_last = acc_tok[torch.arange(b, device="cuda"), num_acc.long() - 1]
    assert torch.equal(tok[0], ref_last)
    assert torch.all(tok[1:] == -5)  # draft rows untouched (written by the heads)
