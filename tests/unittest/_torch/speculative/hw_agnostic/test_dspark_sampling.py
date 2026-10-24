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
"""Sampled DSpark drafting and the rejection path's verification rules.

Under ``use_rejection_sampling`` the DSpark worker draws each block position
from ``softmax((base_k + markov_bias(d_{k-1})) / T)`` with ``d_{k-1}`` the token
it drew before (ancestral sampling of the Markov head), and stores exactly that
distribution as the position's proposal. These tests pin the conditioning, the
stored proposal, the sampling parameters, losslessness with both verification
rules, the acceptance gain over the old greedy-chained proposal, and CUDA
graph capture with the real sampler kernels.
"""

import math
import types
from unittest.mock import patch

import pytest
import torch

from tensorrt_llm._torch.models.modeling_dspark import _DSparkHeadMixin
from tensorrt_llm._torch.models.modeling_speculative import (
    dspark_markov_chain,
    dspark_markov_chain_ancestral,
    dspark_markov_step_bias,
)
from tensorrt_llm._torch.speculative.dspark import DSparkWorker
from tensorrt_llm._torch.speculative.interface import _RNG_SLOT_SPAN, SpecWorkerBase
from tensorrt_llm.llmapi.llm_args import AdvancedSamplingMode, DSparkDecodingConfig, TorchLlmArgs

B, K, V, RANK = 4, 3, 32, 8
INT32_MAX = torch.iinfo(torch.int32).max


class _MarkovDrafter(_DSparkHeadMixin):
    """The DSpark head set alone: the worker's sampled path needs nothing else."""

    def __init__(self, markov_w1, markov_w2):
        self._dspark_markov_rank = markov_w1.shape[1]
        self.markov_w1 = markov_w1
        self.markov_w2 = markov_w2


def _drafter(generator, vocab=V, scale=1.0, device="cpu"):
    w1 = scale * torch.randn(vocab, RANK, generator=generator)
    w2 = torch.randn(vocab, RANK, generator=generator)
    return _MarkovDrafter(w1.to(device), w2.to(device))


def _worker():
    worker = DSparkWorker.__new__(DSparkWorker)
    torch.nn.Module.__init__(worker)
    worker.mapping = None
    worker._d2t = None
    worker.force_num_accepted_tokens = 0.0
    return worker


def _metadata(
    num_gens,
    *,
    num_contexts=0,
    num_slots=None,
    vocab=V,
    draft_len=K,
    temperature=1.0,
    top_k=INT32_MAX,
    top_p=1.0,
    min_p=0.0,
    temperature_only=False,
    mode=None,
    device="cpu",
):
    batch = num_contexts + num_gens
    num_slots = num_slots or batch + 1
    return types.SimpleNamespace(
        request_temperatures=torch.full((batch,), temperature, device=device),
        request_top_ks=torch.full((batch,), top_k, dtype=torch.int32, device=device),
        request_top_ps=torch.full((batch,), top_p, device=device),
        request_min_ps=torch.full((batch,), min_p, device=device),
        request_seeds=torch.arange(batch, dtype=torch.int64, device=device) + 11,
        request_offsets=1000 * torch.arange(batch, dtype=torch.int64, device=device),
        use_rejection_sampling=True,
        max_draft_len=draft_len,
        draft_temperature_only=temperature_only,
        # Slots in reverse batch order, so a batch-order write would be caught.
        batch_slot_ids=torch.arange(batch, dtype=torch.long, device=device).flip(0),
        draft_probs=torch.full((num_slots, draft_len, vocab), 7.0, device=device),
        draft_probs_last_dim=0,
        dummy_slot_row=0,
        advanced_sampling_mode=mode,
        rng_draft_slot=lambda step: draft_len + 1 + step,
        rng_accept_slot=2 * draft_len + 1,
        is_all_greedy_sample=False,
        wants_advanced_draft_sampling=True,
        is_ragged_verify=False,
        draft_vocab_size=vocab,
        vocab_size=vocab,
    )


def _cpu_sampler(generator, calls=None):
    """Stand-in for the seeded sampler: temperature only, returns what it drew from."""

    def sample(mode, logits, temperatures, top_ks, top_ps, min_ps, *, seed, offset):
        if calls is not None:
            calls.append(
                dict(
                    top_ks=top_ks.clone(),
                    top_ps=top_ps.clone(),
                    min_ps=min_ps.clone(),
                    offset=offset.clone(),
                )
            )
        probs = torch.softmax(logits.double() / temperatures.double()[:, None], -1)
        tokens = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
        return tokens.to(torch.int32), probs.float()

    return sample


def _sample_with(sampler, worker, drafter, base, anchors, meta, num_contexts=0):
    inputs = {"first_prev_tokens": anchors}
    with patch.object(SpecWorkerBase, "_sample_from_logits_with_probs", staticmethod(sampler)):
        return worker._sample_block_ancestrally(
            drafter, base, inputs, meta, num_contexts, num_contexts + base.shape[0]
        )


def _expected_rows(drafter, base, anchors, tokens, temperature):
    """softmax((base_k + bias(drawn k-1)) / T) for every position."""
    prev = torch.cat([anchors.view(-1, 1), tokens[:, :-1].long()], dim=1)
    bias = dspark_markov_step_bias(prev.reshape(-1), drafter.markov_w1, drafter.markov_w2)
    logits = base + bias.view(*prev.shape, -1).to(base.dtype)
    return torch.softmax(logits.double() / temperature, -1).float()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_config_field_defaults_to_the_existing_behavior():
    assert DSparkDecodingConfig(max_draft_len=3).draft_temperature_only is False
    config = DSparkDecodingConfig(
        max_draft_len=3, use_rejection_sampling=True, draft_temperature_only=True
    )
    assert config.draft_temperature_only


def test_temperature_only_requires_rejection_sampling():
    off = types.SimpleNamespace(
        speculative_config=DSparkDecodingConfig(max_draft_len=3, draft_temperature_only=True)
    )
    with pytest.raises(ValueError, match="use_rejection_sampling=True"):
        TorchLlmArgs._validate_rejection_sampling_options(off)
    on = types.SimpleNamespace(
        speculative_config=DSparkDecodingConfig(
            max_draft_len=3, use_rejection_sampling=True, draft_temperature_only=True
        )
    )
    TorchLlmArgs._validate_rejection_sampling_options(on)


# ---------------------------------------------------------------------------
# The ancestral chain
# ---------------------------------------------------------------------------


def test_ancestral_chain_conditions_each_position_on_the_drawn_token():
    generator = torch.Generator().manual_seed(1)
    drafter = _drafter(generator)
    base = torch.randn(B, K, V, generator=generator)
    anchors = torch.randint(V, (B,), generator=generator)
    rows = []

    def argmin_step(step, step_logits):  # deliberately not the argmax walk
        rows.append((step, step_logits))
        return step_logits.argmin(-1)

    def step_bias(prev, hidden):
        return dspark_markov_step_bias(prev, drafter.markov_w1, drafter.markov_w2)

    tokens = dspark_markov_chain_ancestral(base, anchors, step_bias, argmin_step)
    prev = anchors
    for step in range(K):
        assert rows[step][0] == step
        torch.testing.assert_close(rows[step][1], base[:, step] + step_bias(prev, None))
        prev = tokens[:, step]
    assert tokens.dtype == torch.long and tokens.shape == (B, K)


def test_ancestral_chain_with_argmax_draws_is_the_greedy_chain():
    generator = torch.Generator().manual_seed(2)
    drafter = _drafter(generator)
    base = torch.randn(B, K, V, generator=generator)
    anchors = torch.randint(V, (B,), generator=generator)

    def step_bias(prev, hidden):
        return dspark_markov_step_bias(prev, drafter.markov_w1, drafter.markov_w2)

    tokens = drafter.sample_markov_chain(base, anchors, lambda step, logits: logits.argmax(-1))
    greedy, _ = dspark_markov_chain(base, anchors, step_bias, cast_bias_to_logits=True)
    assert torch.equal(tokens, greedy)


# ---------------------------------------------------------------------------
# The worker's sampled path
# ---------------------------------------------------------------------------


def test_routing_takes_the_sampled_path_only_for_sampled_drafts_with_a_markov_head():
    generator = torch.Generator().manual_seed(3)
    worker = _worker()
    drafter = _drafter(generator)
    meta = _metadata(B)
    assert worker._samples_block_ancestrally(drafter, meta)
    meta.wants_advanced_draft_sampling = False
    assert not worker._samples_block_ancestrally(drafter, meta)
    meta.wants_advanced_draft_sampling = True
    no_head = types.SimpleNamespace(has_markov_head=False)
    assert not worker._samples_block_ancestrally(no_head, meta)


@pytest.mark.parametrize("num_contexts", [0, 2])
def test_sampled_block_stores_the_distribution_each_draft_was_drawn_from(num_contexts):
    generator = torch.Generator().manual_seed(4)
    worker = _worker()
    drafter = _drafter(generator)
    base = torch.randn(B, K, V, generator=generator)
    anchors = torch.randint(V, (B,), generator=generator)
    meta = _metadata(B, num_contexts=num_contexts, temperature=0.7)

    tokens, vocab = _sample_with(
        _cpu_sampler(generator), worker, drafter, base, anchors, meta, num_contexts
    )

    assert tokens.dtype == torch.int32 and tokens.shape == (B, K)
    assert vocab == meta.draft_probs_last_dim == V
    gen_slots = meta.batch_slot_ids[num_contexts:]
    expected = _expected_rows(drafter, base, anchors, tokens, temperature=0.7)
    torch.testing.assert_close(meta.draft_probs[gen_slots], expected)
    others = torch.ones(meta.draft_probs.shape[0], dtype=torch.bool)
    others[gen_slots] = False
    assert (meta.draft_probs[others] == 7.0).all()  # context and spare slots untouched


@pytest.mark.parametrize("temperature_only", [False, True])
def test_sampled_block_passes_the_draft_sampling_parameters(temperature_only):
    generator = torch.Generator().manual_seed(5)
    worker = _worker()
    drafter = _drafter(generator)
    base = torch.randn(B, K, V, generator=generator)
    anchors = torch.randint(V, (B,), generator=generator)
    meta = _metadata(B, top_k=20, top_p=0.95, min_p=0.05, temperature_only=temperature_only)
    calls = []

    _sample_with(_cpu_sampler(generator, calls), worker, drafter, base, anchors, meta)

    assert len(calls) == K
    for step, call in enumerate(calls):
        if temperature_only:
            assert (call["top_ks"] == INT32_MAX).all()
            assert (call["top_ps"] == 1.0).all() and (call["min_ps"] == 0.0).all()
        else:
            assert (call["top_ks"] == 20).all()
            torch.testing.assert_close(call["top_ps"], torch.full((B,), 0.95))
            torch.testing.assert_close(call["min_ps"], torch.full((B,), 0.05))
        # Each position draws from its own draft slot of the request's window.
        expected_offset = meta.request_offsets + meta.rng_draft_slot(step) * _RNG_SLOT_SPAN
        assert torch.equal(call["offset"], expected_offset)


def test_draft_sampling_params_temperature_only_keeps_the_temperature():
    meta = _metadata(B, temperature=0.6, top_k=20, top_p=0.9, min_p=0.1, temperature_only=True)
    temps, top_ks, top_ps, min_ps = SpecWorkerBase._draft_sampling_params(meta, 1, B, repeat=2)
    torch.testing.assert_close(temps, torch.full((2 * (B - 1),), 0.6))
    assert (top_ks == INT32_MAX).all() and top_ks.dtype == torch.int32
    assert (top_ps == 1.0).all() and (min_ps == 0.0).all()
    meta.draft_temperature_only = False
    _, top_ks, top_ps, min_ps = SpecWorkerBase._draft_sampling_params(meta, 1, B, repeat=2)
    assert (top_ks == 20).all()
    torch.testing.assert_close(top_ps, torch.full((2 * (B - 1),), 0.9))


# ---------------------------------------------------------------------------
# End to end on a toy vocabulary: losslessness and acceptance
# ---------------------------------------------------------------------------

_TOY_VOCAB = 4


def _target_table(generator):
    """Target next-token distributions for every prefix of length 0..K (base-V ids)."""
    num_prefixes = sum(_TOY_VOCAB**m for m in range(K + 1))
    logits = 1.5 * torch.randn(num_prefixes, _TOY_VOCAB, generator=generator)
    return torch.softmax(logits.double(), -1)


def _prefix_ids(tokens):
    length = tokens.shape[1]
    ids = torch.zeros(tokens.shape[0], dtype=torch.int64)
    for m in range(length):
        ids = ids * _TOY_VOCAB + tokens[:, m]
    return ids + sum(_TOY_VOCAB**m for m in range(length))


def _complete_and_count(table, accepted_tokens, num_accepted, generator):
    """Emitted tokens, completed by sampling the target, as base-V sequence ids."""
    num = accepted_tokens.shape[0]
    sequences = torch.zeros(num, K + 1, dtype=torch.int64)
    for m in range(K + 1):
        cont = torch.multinomial(table[_prefix_ids(sequences[:, :m])], 1, generator=generator)
        keep = m < num_accepted.long()
        sequences[:, m] = torch.where(keep, accepted_tokens[:, m].long(), cont.squeeze(-1))
    ids = torch.zeros(num, dtype=torch.int64)
    for m in range(K + 1):
        ids = ids * _TOY_VOCAB + sequences[:, m]
    return torch.bincount(ids, minlength=_TOY_VOCAB ** (K + 1))


def _exact_sequence_probs(table):
    grid = torch.cartesian_prod(*[torch.arange(_TOY_VOCAB)] * (K + 1))
    prob = torch.ones(grid.shape[0], dtype=torch.float64)
    for m in range(K + 1):
        prob = prob * table[_prefix_ids(grid[:, :m])].gather(-1, grid[:, m : m + 1]).squeeze(-1)
    return prob


def _chi_square(counts, probs):
    total = counts.sum().item()
    expected = probs * total
    big = expected >= 5
    observed = torch.cat([counts[big].double(), counts[~big].double().sum().view(1)])
    expected = torch.cat([expected[big], expected[~big].sum().view(1)])
    keep = expected > 0
    stat = (((observed - expected) ** 2)[keep] / expected[keep]).sum().item()
    dof = int(keep.sum()) - 1
    z = 3.719  # 1e-4 one-sided; Wilson-Hilferty approximation of the critical value
    return stat, dof * (1 - 2 / (9 * dof) + z * math.sqrt(2 / (9 * dof))) ** 3


def _token_level(draft_tokens, draft_probs, target_probs, uniforms, resample_uniforms):
    """Token-level rejection sampling (Leviathan et al., 2023), vectorized."""
    num, num_drafts = draft_tokens.shape
    tokens_x = draft_tokens.long().unsqueeze(-1)
    ratio = (
        target_probs[:, :num_drafts].gather(-1, tokens_x).squeeze(-1).double()
        / draft_probs.gather(-1, tokens_x).squeeze(-1).double()
    )
    passed = (uniforms <= ratio).long().cumprod(-1)
    length = passed.sum(-1)
    rows = torch.arange(num)
    p_row = target_probs[rows, length]
    q_row = draft_probs[rows, length.clamp(max=num_drafts - 1)]
    weights = torch.where((length == num_drafts)[:, None], p_row, (p_row - q_row).clamp_min(0))
    cdf = weights.double().cumsum(-1)
    emitted = torch.searchsorted(cdf, resample_uniforms[:, None] * cdf[:, -1:], right=True).squeeze(
        -1
    )
    out = torch.full((num, num_drafts + 1), -1, dtype=torch.int64)
    slots = torch.arange(num_drafts + 1)
    drafts = torch.cat([draft_tokens.long(), torch.zeros(num, 1, dtype=torch.int64)], 1)
    out = torch.where(slots < length[:, None], drafts, out)
    out = torch.where(slots == length[:, None], emitted.clamp(max=_TOY_VOCAB - 1)[:, None], out)
    return out, length + 1


def test_sampled_drafts_verify_losslessly():
    generator = torch.Generator().manual_seed(22)
    num = 100_000
    worker = _worker()
    drafter = _drafter(generator, vocab=_TOY_VOCAB, scale=2.0)
    base = torch.randn(1, K, _TOY_VOCAB, generator=generator).expand(num, -1, -1).contiguous()
    anchors = torch.zeros(num, dtype=torch.int64)
    meta = _metadata(num, vocab=_TOY_VOCAB, num_slots=num + 1)

    drafts, _ = _sample_with(_cpu_sampler(generator), worker, drafter, base, anchors, meta)
    draft_probs = meta.draft_probs[meta.batch_slot_ids]
    table = _target_table(generator)
    target_probs = torch.stack(
        [table[_prefix_ids(drafts[:, :j].long())] for j in range(K + 1)], dim=1
    ).float()
    uniforms = torch.rand(num, K, generator=generator, dtype=torch.float64)
    resample_uniforms = torch.rand(num, generator=generator, dtype=torch.float64)
    accepted_tokens, num_accepted = _token_level(
        drafts, draft_probs, target_probs, uniforms, resample_uniforms
    )

    counts = _complete_and_count(table, accepted_tokens, num_accepted, generator)
    stat, critical = _chi_square(counts, _exact_sequence_probs(table))
    assert stat < critical, f"chi-square {stat:.1f} >= {critical:.1f}"


def test_ancestral_proposal_is_exact_where_the_greedy_chained_one_is_not():
    """Target = the drafter's own Markov chain. Drawing ancestrally proposes exactly
    the target, so every draft is accepted; the old proposal (independent draws from
    rows biased by the argmax chain) is conditioned on tokens the block does not
    contain and is rejected whenever a draw leaves the argmax."""
    generator = torch.Generator().manual_seed(23)
    num = 20_000
    worker = _worker()
    drafter = _drafter(generator, vocab=_TOY_VOCAB, scale=2.0)
    base = torch.randn(1, K, _TOY_VOCAB, generator=generator).expand(num, -1, -1).contiguous()
    anchors = torch.zeros(num, dtype=torch.int64)

    def target_rows(drafts):
        rows = _expected_rows(drafter, base, anchors, drafts, temperature=1.0)
        bonus = torch.full((num, 1, _TOY_VOCAB), 1.0 / _TOY_VOCAB)
        return torch.cat([rows, bonus], dim=1)

    def accepted_lengths(drafts, draft_probs):
        uniforms = torch.rand(num, K, generator=generator, dtype=torch.float64)
        resample = torch.rand(num, generator=generator, dtype=torch.float64)
        _, accepted = _token_level(drafts, draft_probs, target_rows(drafts), uniforms, resample)
        return accepted.double() - 1

    meta = _metadata(num, vocab=_TOY_VOCAB, num_slots=num + 1)
    drafts, _ = _sample_with(_cpu_sampler(generator), worker, drafter, base, anchors, meta)
    ancestral = accepted_lengths(drafts, meta.draft_probs[meta.batch_slot_ids])
    assert (ancestral == K).all()

    # The scheme this replaces: greedy-chained rows, each sampled on its own.
    old_meta = _metadata(num, vocab=_TOY_VOCAB, num_slots=num + 1)
    refined = worker._refine_block_logits(drafter, base, {"first_prev_tokens": anchors}, old_meta)
    with patch.object(
        SpecWorkerBase, "_sample_from_logits_with_probs", staticmethod(_cpu_sampler(generator))
    ):
        old_drafts = worker.sample_draft_tokens(refined, old_meta, num, num_contexts=0)
    old = accepted_lengths(old_drafts, old_meta.draft_probs[old_meta.batch_slot_ids])
    assert old.mean() < K - 0.2, f"old scheme accepted {old.mean():.2f} of {K}"


# ---------------------------------------------------------------------------
# CUDA graphs and the real sampler kernels
# ---------------------------------------------------------------------------

_needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _capture(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn()  # warm up outside the graph
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    return graph, out


@_needs_cuda
@pytest.mark.parametrize("temperature_only", [True, False])
def test_sampled_block_captures_in_a_cuda_graph(temperature_only):
    generator = torch.Generator().manual_seed(31)
    vocab, num_gens = 4096, 6
    worker = _worker()
    drafter = _drafter(generator, vocab=vocab, device="cuda")
    base = torch.randn(num_gens, K, vocab, generator=generator).cuda()
    anchors = torch.randint(vocab, (num_gens,), generator=generator).cuda()
    meta = _metadata(
        num_gens,
        vocab=vocab,
        num_slots=num_gens + 1,
        temperature=0.8,
        top_p=0.9,
        temperature_only=temperature_only,
        mode=AdvancedSamplingMode.FULL,
        device="cuda",
    )
    # The last request is CUDA-graph padding: it routes to the scratch row and
    # its logits are garbage.
    meta.dummy_slot_row = num_gens
    meta.batch_slot_ids[-1] = num_gens
    base[-1] = float("nan")
    inputs = {"first_prev_tokens": anchors}

    graph, (tokens, _) = _capture(
        lambda: worker._sample_block_ancestrally(drafter, base, inputs, meta, 0, num_gens)
    )
    meta.draft_probs.fill_(7.0)
    graph.replay()

    real = meta.batch_slot_ids[:-1]
    stored = meta.draft_probs[real]
    torch.testing.assert_close(stored.sum(-1), torch.ones(num_gens - 1, K, device="cuda"))
    assert (stored.gather(-1, tokens[:-1].long().unsqueeze(-1)) > 0).all()
    assert ((tokens >= 0) & (tokens < vocab)).all()
    if temperature_only:
        expected = _expected_rows(drafter, base[:-1], anchors[:-1], tokens[:-1], temperature=0.8)
        torch.testing.assert_close(stored, expected, atol=1e-5, rtol=1e-4)
    else:
        # top-p 0.9 leaves part of the vocab with no mass.
        assert ((stored == 0).sum(-1) > 0).all()

    # Replays redraw from the new offsets, like successive decode steps.
    first = tokens.clone()
    meta.request_offsets += 1 << 20
    graph.replay()
    assert not torch.equal(first, tokens)
