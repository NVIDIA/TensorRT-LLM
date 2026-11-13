# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NATIVE_MARKOV_SAMPLING (family ``minimax_m3/ops/markov_sampling``) on a GPU, at the served vocabulary (200064) and
at widths outside the served class that must run the same CUDA path (64 and 4097: one chunk, 4097 not a multiple of
4; 5000; 300000: 147 chunks, more than 128): tokens are in the vocabulary; each stored q_k equals
softmax((base_k + bias(d_{k-1})) / T) of the token drawn before it (fp32 reference, rtol 1e-4); the same seeds draw
the same block and proposals eager and replayed from a CUDA graph on new inputs, bit for bit; a padding row with
non-finite logits leaves the real rows as they are; only invalid operands return None. (Bit for bit the former Triton
sampler on the served class: the native sampler's MB.)"""

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops import (
    markov_sampling as ms,
)
from tensorrt_llm._utils import get_sm_version

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() != 107,
    reason="the DSpark Markov sampling op runs on sm_107 (Rubin) only",
)

K, RANK, CAPACITY, STEP = 7, 64, 64, 32
VOCABS = [200064, 64, 4097, 5000, 300000]


def _inputs(gen, rows, vocab, ldt=torch.float32, wdt=torch.bfloat16, pad_row=None):
    logits = (torch.randn(rows, K, vocab, generator=gen, device="cuda") * 2.0).to(ldt)
    w1 = (torch.randn(vocab, RANK, generator=gen, device="cuda") * 0.05).to(wdt)
    w2 = (torch.randn(vocab, RANK, generator=gen, device="cuda") * 0.05).to(wdt)
    temps = torch.tensor([(0.6, 1.0, 1.3, 0.25, 0.9)[i % 5] for i in range(rows)], device="cuda")
    seeds = torch.randint(-(2**62), 2**62, (rows,), generator=gen, device="cuda", dtype=torch.int64)
    offsets = torch.randint(0, 2**36, (rows,), generator=gen, device="cuda", dtype=torch.int64)
    slots = torch.randperm(CAPACITY - 1, generator=gen, device="cuda")[:rows].to(torch.int64)
    pad_slot = -1
    if pad_row is not None:
        pad_slot = CAPACITY - 1
        slots[pad_row] = pad_slot
        bad = logits[pad_row].float()
        bad[:, ::7] = float("nan")
        bad[:, 3::11] = float("inf")
        logits[pad_row] = bad.to(ldt)
    anchors = torch.randint(0, vocab, (rows,), generator=gen, device="cuda", dtype=torch.int64)
    if rows > 2:
        anchors[1] = -1
        anchors[2] = vocab + 5
    probs = torch.zeros(CAPACITY, K, vocab + 128, device="cuda")
    return [logits, anchors, w1, w2, temps, seeds, offsets, probs, slots, pad_slot]


def _run(args):
    logits, anchors, w1, w2, temps, seeds, offsets, probs, slots, pad_slot = args
    return ms.sample_markov_block(
        logits, anchors, w1, w2, temps, seeds, offsets, STEP, probs, slots, pad_slot
    )


@pytest.mark.parametrize("vocab", VOCABS)
@pytest.mark.parametrize("rows", [1, 11])
@pytest.mark.parametrize("ldt", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_the_proposals_are_the_drawn_distributions(vocab, rows, ldt):
    gen = torch.Generator(device="cuda").manual_seed(rows + vocab)
    args = _inputs(gen, rows, vocab, ldt)
    logits, anchors, w1, w2, temps, _, _, probs, slots, _ = args
    tokens = _run(args)
    assert tokens is not None and tokens.shape == (rows, K) and tokens.dtype == torch.int32
    assert bool(((tokens >= 0) & (tokens < vocab)).all())
    prev = anchors.clone()
    for k in range(K):
        valid = (prev >= 0) & (prev < vocab)
        emb = w1[prev.clamp(0, vocab - 1).long()] * valid[:, None].to(w1.dtype)
        bias = F.linear(emb, w2).float()
        want = torch.softmax((logits[:, k].float() + bias) / temps[:, None], -1)
        got = probs[slots, k, :vocab]
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-7)
        assert bool(
            (got.gather(-1, tokens[:, k : k + 1].long()) > 0).all()
        )  # each token was drawn from its q_k
        prev = tokens[:, k].long()


@pytest.mark.parametrize("vocab", VOCABS)
def test_the_same_seeds_draw_the_same_block_eager_and_in_a_cuda_graph(vocab):
    gen = torch.Generator(device="cuda").manual_seed(5 + vocab)
    args = _inputs(gen, 11, vocab, pad_row=3)
    _run(args)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        got = _run(args)
    new = _inputs(gen, 11, vocab, pad_row=3)
    for dst, src in zip(args[:9], new[:9]):
        dst.copy_(src)
    graph.replay()
    torch.cuda.synchronize()
    eager_probs = torch.zeros_like(args[7])
    eager = _run(args[:7] + [eager_probs] + args[8:])
    assert torch.equal(got, eager)
    assert torch.equal(args[7].view(torch.int32), eager_probs.view(torch.int32))


@pytest.mark.parametrize("vocab", VOCABS)
def test_a_padding_row_leaves_the_real_rows_as_they_are(vocab):
    gen = torch.Generator(device="cuda").manual_seed(9 + vocab)
    args = _inputs(gen, 8, vocab)
    clean = _run(args)
    clean_probs = args[7].clone()
    padded = list(args)
    padded[7] = torch.zeros_like(args[7])
    padded[8] = args[8].clone()
    padded[8][3] = CAPACITY - 1
    padded[9] = CAPACITY - 1
    bad = args[0].clone()
    bad[3] = float("nan")
    padded[0] = bad
    got = _run(padded)
    assert bool(((got[3] >= 0) & (got[3] < vocab)).all())
    real = torch.tensor([i for i in range(8) if i != 3], device="cuda")
    assert torch.equal(got[real], clean[real])
    assert torch.equal(padded[7][args[8][real]], clean_probs[args[8][real]])


def test_only_invalid_operands_return_none():
    gen = torch.Generator(device="cuda").manual_seed(11)
    args = _inputs(gen, 3, 4096)
    args[5] = args[5].int()  # int32 seeds
    assert _run(args) is None
    args = _inputs(gen, 3, 4096)
    args[0] = args[0].half()  # fp16 logits
    assert _run(args) is None
    args = _inputs(gen, 3, 4096)
    args[0] = args[0].transpose(1, 2).contiguous().transpose(1, 2)  # not contiguous
    assert _run(args) is None
    args = _inputs(gen, 3, 4096)
    args[2] = args[2][:, 0].contiguous()  # 1-D markov_w1
    assert _run(args) is None
