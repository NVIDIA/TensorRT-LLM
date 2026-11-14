# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DEP_GUMBEL_DRAFT on a GPU, through ``gumbel_draft.draw_block``:
- every stored proposal q_k equals ``softmax((base_k + bias(d_{k-1})) / T)`` of the token drawn before it (an fp32
  reference, to reduction-order tolerance), at the served geometry (vocab 200064, rank 256, K 7);
- the drawn tokens follow those distributions (a chi-square goodness of fit per position over many seeds);
- the same seeds draw the same block, eager and replayed from a CUDA graph;
- CUDA-graph padding rows (the dummy slot) with non-finite logits leave every real row's tokens and proposals as they
  are without them.
Not bitwise vs the stock loop by design (another random stream)."""

import math
import types

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__dep import (  # noqa: E501
    gumbel_draft,
)
from tensorrt_llm._torch.models.modeling_speculative import dspark_markov_step_bias
from tensorrt_llm._torch.speculative.interface import SpecWorkerBase
from tensorrt_llm._utils import get_sm_version

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() != 107, reason="SM107 (Rubin) only"
)


class _Worker:
    """The worker surface ``draw_block`` reads: the real per-request Philox state."""

    _rng_state_per_request = SpecWorkerBase._rng_state_per_request

    def maybe_gather_sharded_draft_logits(self, gen_logits, spec_metadata):
        return gen_logits


def _case(rows, k, vocab, rank, seed, capacity=64, dummy_rows=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    w1 = (torch.randn(vocab, rank, generator=gen, device="cuda") * 0.05).to(torch.bfloat16)
    w2 = (torch.randn(vocab, rank, generator=gen, device="cuda") * 0.05).to(torch.bfloat16)
    slots = torch.randperm(capacity, generator=gen, device="cuda")[:rows]
    if dummy_rows:
        slots[rows - dummy_rows :] = capacity
    prev = torch.randint(0, vocab, (rows,), generator=gen, device="cuda", dtype=torch.int32)
    prev[0] = -1  # the rejection pad: no bias
    meta = types.SimpleNamespace(
        draft_probs=torch.zeros(capacity + 1, k, vocab + 128, device="cuda"),
        batch_slot_ids=slots,
        request_temperatures=torch.rand(rows, generator=gen, device="cuda") + 0.5,
        request_seeds=torch.randint(
            0, 2**31, (rows,), generator=gen, device="cuda", dtype=torch.int64
        ),
        request_offsets=torch.zeros(rows, dtype=torch.int64, device="cuda"),
        rng_draft_slot=lambda step: 8 + step,
        dummy_slot_row=capacity,
    )
    gen_logits = torch.randn(rows, k, vocab, generator=gen, device="cuda") * 2.0
    draft_model = types.SimpleNamespace(markov_w1=w1, markov_w2=w2)
    return draft_model, gen_logits, {"first_prev_tokens": prev}, meta


def _draw(draft_model, gen_logits, inputs, meta):
    out = gumbel_draft.draw_block(
        _Worker(), draft_model, gen_logits, inputs, meta, 0, gen_logits.shape[0]
    )
    assert out is not None, "the override declined a served-shape case"
    return out


def _reference_q(draft_model, gen_logits, inputs, meta, tokens):
    """q_k = softmax((base_k + bias(d_{k-1})) / T) in fp32, d_{-1} the anchor, from the drawn tokens."""
    rows, k, _ = gen_logits.shape
    prev = inputs["first_prev_tokens"]
    t = meta.request_temperatures.unsqueeze(-1)
    qs = []
    for pos in range(k):
        bias = dspark_markov_step_bias(prev, draft_model.markov_w1, draft_model.markov_w2).float()
        qs.append(torch.softmax((gen_logits[:, pos].float() + bias) / t, dim=-1))
        prev = tokens[:, pos]
    return torch.stack(qs, dim=1)


@pytest.mark.parametrize("rows", [12, 48])
def test_the_stored_proposals_are_the_drawn_distributions(rows):
    vocab, k = 200064, 7
    draft_model, gen_logits, inputs, meta = _case(rows, k, vocab, 256, seed=rows)
    tokens, width = _draw(draft_model, gen_logits, inputs, meta)
    assert width == vocab and meta.draft_probs_last_dim == vocab
    assert tokens.dtype == torch.int32 and tokens.shape == (rows, k)
    assert bool(((tokens >= 0) & (tokens < vocab)).all())
    stored = meta.draft_probs[meta.batch_slot_ids, :, :vocab]
    ref = _reference_q(draft_model, gen_logits, inputs, meta, tokens)
    torch.testing.assert_close(stored, ref, rtol=1e-4, atol=1e-7)
    # Every drawn token has a positive proposal (it was drawn from it).
    picked = stored.gather(-1, tokens.long().unsqueeze(-1))
    assert bool((picked > 0).all())


def test_the_draws_follow_the_proposals():
    """Pooled over draws, each position's token counts match the summed conditional proposals (chi-square)."""
    vocab, k, rows, rank, batches = 64, 3, 48, 16, 100
    draft_model, gen_logits, inputs, meta = _case(rows, k, vocab, rank, seed=7)
    observed = torch.zeros(k, vocab, dtype=torch.float64, device="cuda")
    expected = torch.zeros_like(observed)
    for b in range(batches):
        meta.request_seeds = (
            meta.request_seeds * 0 + 1_000_003 * (b + 1) + torch.arange(rows, device="cuda")
        )
        tokens, _ = _draw(draft_model, gen_logits, inputs, meta)
        ref = _reference_q(draft_model, gen_logits, inputs, meta, tokens).double()
        for pos in range(k):
            observed[pos] += torch.bincount(tokens[:, pos].long(), minlength=vocab).double()
            expected[pos] += ref[:, pos].sum(0)
    for pos in range(k):
        keep = expected[pos] >= 5
        o, e = observed[pos][keep], expected[pos][keep]
        # The bins under 5 expected draws pool into one.
        o = torch.cat([o, observed[pos][~keep].sum().view(1)])
        e = torch.cat([e, expected[pos][~keep].sum().view(1)])
        o, e = o[e > 0], e[e > 0]
        stat = float(((o - e) ** 2 / e).sum())
        df = o.numel() - 1
        assert stat < df + 6 * math.sqrt(2 * df), (pos, stat, df)  # far past p = 1e-6 for these df


def test_the_same_seeds_draw_the_same_block_eager_and_in_a_cuda_graph():
    draft_model, gen_logits, inputs, meta = _case(12, 7, 200064, 256, seed=3)
    eager_tokens, _ = _draw(draft_model, gen_logits, inputs, meta)
    eager_probs = meta.draft_probs.clone()
    again, _ = _draw(draft_model, gen_logits, inputs, meta)
    assert torch.equal(again, eager_tokens)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        _draw(
            draft_model, gen_logits, inputs, meta
        )  # warm up (Triton compiles) outside the capture
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out, _ = _draw(draft_model, gen_logits, inputs, meta)
    meta.draft_probs.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager_tokens)
    assert torch.equal(meta.draft_probs, eager_probs)


def test_padding_rows_leave_the_real_rows_as_they_are():
    rows, dummy = 12, 3
    draft_model, gen_logits, inputs, meta = _case(rows, 7, 200064, 256, seed=11, dummy_rows=dummy)
    real = rows - dummy
    tokens, _ = _draw(draft_model, gen_logits, inputs, meta)
    probs = meta.draft_probs[meta.batch_slot_ids[:real]].clone()
    garbage = gen_logits.clone()
    garbage[real:] = float("nan")
    meta.draft_probs.zero_()
    tokens2, _ = _draw(draft_model, garbage, inputs, meta)
    assert torch.equal(tokens2[:real], tokens[:real])
    assert torch.equal(meta.draft_probs[meta.batch_slot_ids[:real]], probs)
    assert bool(torch.isfinite(meta.draft_probs[meta.dummy_slot_row]).all())
    assert bool(((tokens2[real:] >= 0) & (tokens2[real:] < 200064)).all())
