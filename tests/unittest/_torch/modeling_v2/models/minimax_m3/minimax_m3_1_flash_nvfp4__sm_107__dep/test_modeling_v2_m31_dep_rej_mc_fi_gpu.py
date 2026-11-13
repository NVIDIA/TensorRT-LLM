# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DEP_REJ_MC_FI on an SM107 GPU: ``rej_mc_fi``'s gathered and slot-indexed forms return FlashInfer's
``chain_speculative_sampling(deterministic=True)`` tokens and emitted + 1 counts bit for bit (as
``rejection_sampling_one_model`` calls it, per-request int64 seed / offset), on served-like rows at DEP's batches, at
every rejection position and the bonus row, on padding rows, on denormal probabilities, at every ctas_per_row and under
CUDA-graph replay. A rejected row whose residual has no mass gets token 0 (FlashInfer reads an uninitialized shared
slot there, so that case is not compared)."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__dep import (  # noqa: E501
    rej_mc_fi,
)


def _sm107() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)


pytestmark = pytest.mark.skipif(not _sm107(), reason="SM107 (Rubin) only")

VOCAB = 200064
K = 7
SLOTS = 64


def _probs(shape, gen, base=None):
    import flashinfer

    logits = torch.randn(*shape, generator=gen, device="cuda") * 2.0
    if base is not None:
        logits[..., : base.shape[-2], :] += base
    p = torch.softmax(logits.view(-1, VOCAB), -1)
    p = flashinfer.sampling.top_p_renorm_probs(p, 0.95)
    return p.view(*shape).contiguous(), logits


def _case(batch, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    draft, dlog = _probs((batch, K, VOCAB), gen)
    target, _ = _probs((batch, K + 1, VOCAB), gen, base=dlog * 1.5)
    ids = torch.multinomial(draft.view(-1, VOCAB), 1, generator=gen).view(batch, K).to(torch.int32)
    buf = torch.zeros(SLOTS, K, VOCAB, device="cuda")
    slot_ids = torch.randperm(SLOTS, generator=gen, device="cuda")[:batch].contiguous()
    buf[slot_ids] = draft
    return dict(draft=draft, ids=ids.contiguous(), target=target, buf=buf, slot_ids=slot_ids)


def _seeds(batch, k):
    gen = torch.Generator(device="cuda").manual_seed(1000 + k)
    s = torch.randint(0, 2**62, (batch,), generator=gen, device="cuda", dtype=torch.int64)
    o = torch.randint(0, 2**40, (batch,), generator=gen, device="cuda", dtype=torch.int64)
    return s, o


def _flashinfer(c, s, o, target=None):
    import flashinfer

    t = c["target"] if target is None else target
    b = c["ids"].shape[0]
    acc = torch.zeros(b, dtype=torch.int32, device="cuda")
    emi = torch.zeros(b, dtype=torch.int32, device="cuda")
    tok, _, emitted = flashinfer.sampling.chain_speculative_sampling(
        c["draft"],
        c["ids"],
        t,
        maybe_output_accepted_token_num=acc,
        maybe_output_emitted_draft_token_num=emi,
        deterministic=True,
        generator=None,
        seed=s,
        offset=o,
    )
    return tok, emitted + 1


def _assert_same(ref, got):
    assert got is not None
    assert torch.equal(got[0], ref[0])
    assert torch.equal(got[1].to(torch.int32), ref[1].to(torch.int32))


def _both_forms(c, s, o, target=None):
    t = c["target"] if target is None else target
    gathered = rej_mc_fi.chain_spec_mc_fi(c["draft"], c["ids"], t, s, o)
    slots = rej_mc_fi.chain_spec_mc_fi_slots(c["buf"], c["slot_ids"], K, c["ids"], t, s, o)
    return gathered, slots


@pytest.mark.parametrize("batch", [1, 12, 48])
def test_both_forms_are_bitwise_flashinfer_on_served_like_rows(batch):
    c = _case(batch, 10 * batch)
    for k in range(8):
        s, o = _seeds(batch, k)
        ref = _flashinfer(c, s, o)
        for got in _both_forms(c, s, o):
            _assert_same(ref, got)


def test_every_rejection_position_and_the_bonus_row():
    c = _case(12, 777)
    rows = torch.arange(12, device="cuda")
    for r in range(K + 1):
        t = c["target"].clone()
        for i in range(K):
            tok = c["ids"][:, i].long()
            if i < r:
                t[rows, i, tok] = c["draft"][rows, i, tok] * 2.0 + 1.0e-3
            elif i == r:
                t[rows, i, tok] = 0.0
        for k in range(2):
            s, o = _seeds(12, 100 + k)
            ref = _flashinfer(c, s, o, target=t)
            assert bool((ref[1] == min(r, K) + 1).all())
            for got in _both_forms(c, s, o, target=t):
                _assert_same(ref, got)


@pytest.mark.parametrize("mode", ["uniform", "onehot"])
def test_padding_rows(mode):
    c = _case(12, 900 + len(mode))
    pad = torch.tensor([3, 7, 10], device="cuda")
    if mode == "uniform":
        c["draft"][pad] = 1.0 / VOCAB
        c["target"][pad] = 1.0 / VOCAB
    else:
        c["draft"][pad] = 0.0
        c["draft"][pad, :, 0] = 1.0
        c["target"][pad] = 0.0
        c["target"][pad, :, 0] = 1.0
    c["ids"][pad] = 0
    c["buf"][c["slot_ids"]] = c["draft"]
    for k in range(4):
        s, o = _seeds(12, 200 + k)
        ref = _flashinfer(c, s, o)
        for got in _both_forms(c, s, o):
            _assert_same(ref, got)


def test_denormal_probabilities():
    c = _case(12, 1234)
    gen = torch.Generator(device="cuda").manual_seed(4321)
    for key in ("draft", "target"):
        c[key][torch.rand(c[key].shape, generator=gen, device="cuda") < 0.01] = 1.0e-39
    rows = torch.arange(12, device="cuda")
    tok = c["ids"][:, 0].long()
    c["draft"][rows, 0, tok] = 1.0e-39
    c["target"][rows, 0, tok] = 2.0e-39
    c["buf"][c["slot_ids"]] = c["draft"]
    for k in range(4):
        s, o = _seeds(12, 300 + k)
        ref = _flashinfer(c, s, o)
        for got in _both_forms(c, s, o):
            _assert_same(ref, got)


def test_the_output_does_not_depend_on_ctas_per_row():
    c = _case(12, 55)
    ws = rej_mc_fi.prebuild(torch.device("cuda", torch.cuda.current_device()))
    op = rej_mc_fi.mc_fi_op()
    s, o = _seeds(12, 5)
    ref = _flashinfer(c, s, o)
    for ctas in (1, 2, 4, 8, 16, 32, 64):
        _assert_same(ref, op(c["draft"], c["ids"], c["target"], s, o, ws, ctas))


def test_the_slot_form_replays_inside_a_cuda_graph():
    c = _case(12, 66)
    s, o = _seeds(12, 6)
    rej_mc_fi.chain_spec_mc_fi_slots(c["buf"], c["slot_ids"], K, c["ids"], c["target"], s, o)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = rej_mc_fi.chain_spec_mc_fi_slots(
            c["buf"], c["slot_ids"], K, c["ids"], c["target"], s, o
        )
    for k in range(3):
        new = _case(12, 600 + k)
        c["buf"].copy_(new["buf"])
        c["slot_ids"].copy_(new["slot_ids"])
        c["ids"].copy_(new["ids"])
        c["target"].copy_(new["target"])
        ns, no = _seeds(12, 60 + k)
        s.copy_(ns)
        o.copy_(no)
        graph.replay()
        torch.cuda.synchronize()
        _assert_same(_flashinfer(new, ns, no), out)


def test_a_rejected_row_without_residual_mass_gets_token_0():
    c = _case(12, 555)
    rows = torch.arange(12, device="cuda")
    r = 2
    t = c["target"].clone()
    for i in range(r):
        tok = c["ids"][:, i].long()
        t[rows, i, tok] = c["draft"][rows, i, tok] * 2.0 + 1.0e-3
    t[:, r] = c["draft"][:, r] * 0.5
    t[rows, r, c["ids"][:, r].long()] = 0.0
    s, o = _seeds(12, 8)
    for got in _both_forms(c, s, o, target=t):
        assert bool((got[1] == r + 1).all())
        assert bool((got[0][:, r] == 0).all())
