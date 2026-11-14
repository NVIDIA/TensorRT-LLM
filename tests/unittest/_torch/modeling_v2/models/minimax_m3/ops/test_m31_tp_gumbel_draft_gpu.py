# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The family Gumbel ancestral block sampler (``minimax_m3/ops/gumbel_draft``) at the TP targets' shapes (1..8 requests
of a K 7 block over the full 200064 vocab: TP_GUMBEL_DRAFT), through ``draw_block`` and the DSpark ancestral method:
- every stored proposal q_k equals ``softmax((base_k + bias(d_{k-1})) / T)`` of the token drawn before it (an fp32
  reference, to reduction-order tolerance), at the served geometry (vocab 200064, rank 256, K 7);
- the drawn tokens follow those distributions (a chi-square goodness of fit per position over many seeds);
- the same seeds draw the same block, eager and replayed from a CUDA graph;
- CUDA-graph padding rows (the dummy slot) with non-finite logits leave every real row's tokens and proposals as they
  are without them;
- vocab-sharded drafter logits (TP) are gathered before the draw: a TP8 split draws exactly the full-logits block;
- ``install`` follows the target's switch and puts its tag / switch name on the lines.
Not bitwise vs the stock loop by design (another random stream): the served gates are natural AL and GSM8K."""

import math
import types

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops import gumbel_draft
from tensorrt_llm._torch.models.modeling_speculative import dspark_markov_step_bias
from tensorrt_llm._torch.speculative.interface import SpecWorkerBase
from tensorrt_llm._utils import get_sm_version

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() != 107,
    reason="the tensor-parallel Gumbel block draw runs on sm_107 (Rubin) only",
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


TP_ROWS = [1, 2, 3, 4, 8]
VOCAB, K, RANK = 200064, 7, 256


class _ShardedWorker(_Worker):
    """A TP worker whose drafter logits arrive vocab-sharded and are gathered back (as the in-tree gather does)."""

    def __init__(self, full, world):
        self.full, self.world = full, world

    def maybe_gather_sharded_draft_logits(self, gen_logits, spec_metadata):
        assert gen_logits.shape[-1] * self.world == self.full.shape[-1]
        return self.full


@pytest.mark.parametrize("rows", TP_ROWS)
def test_the_stored_proposals_are_the_drawn_distributions(rows):
    draft_model, gen_logits, inputs, meta = _case(rows, K, VOCAB, RANK, seed=100 + rows)
    tokens, width = _draw(draft_model, gen_logits, inputs, meta)
    assert width == VOCAB and meta.draft_probs_last_dim == VOCAB
    assert tokens.dtype == torch.int32 and tokens.shape == (rows, K)
    assert bool(((tokens >= 0) & (tokens < VOCAB)).all())
    stored = meta.draft_probs[meta.batch_slot_ids, :, :VOCAB]
    ref = _reference_q(draft_model, gen_logits, inputs, meta, tokens)
    torch.testing.assert_close(stored, ref, rtol=1e-4, atol=1e-7)
    picked = stored.gather(-1, tokens.long().unsqueeze(-1))
    assert bool((picked > 0).all())


@pytest.mark.parametrize("rows", [1, 4])
def test_the_draws_follow_the_proposals(rows):
    """Pooled over draws, each position's token counts match the summed conditional proposals (chi-square), at a TP
    batch (the draws come from rows x batches seeds)."""
    vocab, k, rank = 64, 3, 16
    batches = 4800 // rows
    draft_model, gen_logits, inputs, meta = _case(rows, k, vocab, rank, seed=7 + rows)
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
        o = torch.cat([o, observed[pos][~keep].sum().view(1)])
        e = torch.cat([e, expected[pos][~keep].sum().view(1)])
        o, e = o[e > 0], e[e > 0]
        stat = float(((o - e) ** 2 / e).sum())
        df = o.numel() - 1
        assert stat < df + 6 * math.sqrt(2 * df), (pos, stat, df)  # far past p = 1e-6 for these df


@pytest.mark.parametrize("rows", [1, 4])
def test_the_same_seeds_draw_the_same_block_eager_and_in_a_cuda_graph(rows):
    draft_model, gen_logits, inputs, meta = _case(rows, K, VOCAB, RANK, seed=30 + rows)
    eager_tokens, _ = _draw(draft_model, gen_logits, inputs, meta)
    eager_probs = meta.draft_probs.clone()
    again, _ = _draw(draft_model, gen_logits, inputs, meta)
    assert torch.equal(again, eager_tokens)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        _draw(draft_model, gen_logits, inputs, meta)  # warm up outside the capture
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out, _ = _draw(draft_model, gen_logits, inputs, meta)
    meta.draft_probs.zero_()
    for _ in range(3):  # every replay redraws the same block from the same seeds
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, eager_tokens)
        assert torch.equal(meta.draft_probs, eager_probs)


def test_padding_rows_leave_the_real_rows_as_they_are():
    rows, dummy = 4, 1  # a TP batch padded up to a captured graph size
    draft_model, gen_logits, inputs, meta = _case(rows, K, VOCAB, RANK, seed=11, dummy_rows=dummy)
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
    assert bool(((tokens2[real:] >= 0) & (tokens2[real:] < VOCAB)).all())


@pytest.mark.parametrize("world", [2, 8])
def test_sharded_logits_are_gathered_before_the_draw(world):
    rows = 4
    draft_model, gen_logits, inputs, meta = _case(rows, K, VOCAB, RANK, seed=50 + world)
    expected, _ = _draw(draft_model, gen_logits, inputs, meta)
    expected_probs = meta.draft_probs.clone()
    meta.draft_probs.zero_()
    shard = gen_logits[..., : VOCAB // world].contiguous()
    out = gumbel_draft.draw_block(
        _ShardedWorker(gen_logits, world), draft_model, shard, inputs, meta, 0, rows
    )
    assert out is not None
    assert torch.equal(out[0], expected) and out[1] == VOCAB
    assert torch.equal(meta.draft_probs, expected_probs)


def test_a_disabled_install_clears_only_its_own_override(monkeypatch):
    from tensorrt_llm._torch.speculative import dspark

    def setter(hook):
        dspark._ANCESTRAL_BLOCK_OVERRIDE = hook

    monkeypatch.setattr(dspark, "_ANCESTRAL_BLOCK_OVERRIDE", None, raising=False)
    monkeypatch.setattr(dspark, "set_ancestral_block_override", setter, raising=False)
    monkeypatch.setattr(gumbel_draft, "_STATE", dict(gumbel_draft._STATE))
    monkeypatch.setattr(gumbel_draft, "block_sampler", lambda: object())
    assert gumbel_draft.install("[M3.1 TP2]", "TP_GUMBEL_DRAFT", True)
    assert dspark._ANCESTRAL_BLOCK_OVERRIDE is gumbel_draft.draw_block
    assert not gumbel_draft.install("[M3.1 TP2]", "TP_GUMBEL_DRAFT", False)
    assert dspark._ANCESTRAL_BLOCK_OVERRIDE is None
    other = object()
    setter(other)
    assert not gumbel_draft.install("[M3.1 TP2]", "TP_GUMBEL_DRAFT", False)
    assert dspark._ANCESTRAL_BLOCK_OVERRIDE is other
