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
"""Ancestral DSpark draft sampling: config, stored proposals, sampling parameters, verification routing, CUDA graphs."""

import types
from typing import Any, Callable, Optional
from unittest.mock import patch

import pytest
import torch
from _torch.speculative.hw_agnostic.test_block_verification import chi_square

from tensorrt_llm._torch.models.modeling_dspark import _DSparkHeadMixin
from tensorrt_llm._torch.models.modeling_speculative import (
    dspark_markov_chain,
    dspark_markov_step_bias,
)
from tensorrt_llm._torch.speculative.block_verification import block_verify, counter_uniforms
from tensorrt_llm._torch.speculative.dflash import DFlashWorker
from tensorrt_llm._torch.speculative.dspark import DSparkWorker
from tensorrt_llm._torch.speculative.interface import _RNG_SLOT_SPAN, SpecWorkerBase
from tensorrt_llm._torch.speculative.utils import get_spec_metadata
from tensorrt_llm.llmapi.llm_args import AdvancedSamplingMode, MTPDecodingConfig, TorchLlmArgs

B, K, V, RANK = 4, 3, 32, 8
INT32_MAX = torch.iinfo(torch.int32).max


class _MarkovDrafter(_DSparkHeadMixin):
    """The DSpark head set alone: the worker's sampled path needs nothing else."""

    def __init__(self, markov_w1: torch.Tensor, markov_w2: torch.Tensor) -> None:
        self._dspark_markov_rank = markov_w1.shape[1]
        self.markov_w1 = markov_w1
        self.markov_w2 = markov_w2


def _drafter(
    generator: torch.Generator, vocab: int = V, scale: float = 1.0, device: str = "cpu"
) -> _MarkovDrafter:
    w1 = scale * torch.randn(vocab, RANK, generator=generator)
    w2 = torch.randn(vocab, RANK, generator=generator)
    return _MarkovDrafter(w1.to(device), w2.to(device))


def _worker() -> DSparkWorker:
    worker = DSparkWorker.__new__(DSparkWorker)
    torch.nn.Module.__init__(worker)
    worker.mapping = None
    worker._d2t = None
    worker.force_num_accepted_tokens = 0.0
    return worker


def _metadata(
    num_gens: int,
    *,
    num_contexts: int = 0,
    num_slots: Optional[int] = None,
    vocab: int = V,
    draft_len: int = K,
    temperature: float = 1.0,
    top_k: int = INT32_MAX,
    top_p: float = 1.0,
    min_p: float = 0.0,
    temperature_only: bool = False,
    method: str = "standard",
    mode: AdvancedSamplingMode = AdvancedSamplingMode.FULL,
    device: str = "cpu",
) -> types.SimpleNamespace:
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
        rejection_sample_method=method,
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


def _cpu_sampler(
    generator: torch.Generator, calls: Optional[list] = None
) -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
    """Stand-in for the seeded sampler: temperature only, returns what it drew from."""

    def sample(
        mode: Optional[AdvancedSamplingMode],
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_ks: torch.Tensor,
        top_ps: torch.Tensor,
        min_ps: torch.Tensor,
        *,
        seed: torch.Tensor,
        offset: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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


def _sample_with(
    sampler: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    worker: DSparkWorker,
    drafter: _MarkovDrafter,
    base: torch.Tensor,
    anchors: torch.Tensor,
    meta: types.SimpleNamespace,
    num_contexts: int = 0,
) -> tuple[torch.Tensor, int]:
    inputs = {"first_prev_tokens": anchors}
    with patch.object(SpecWorkerBase, "_sample_from_logits_with_probs", staticmethod(sampler)):
        return worker._sample_block_draft_tokens(
            drafter, base, None, inputs, meta, num_contexts, num_contexts + base.shape[0]
        )


def _expected_rows(
    drafter: _MarkovDrafter,
    base: torch.Tensor,
    anchors: torch.Tensor,
    tokens: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """softmax((base_k + bias(drawn k-1)) / T) for every position."""
    prev = torch.cat([anchors.view(-1, 1), tokens[:, :-1].long()], dim=1)
    bias = dspark_markov_step_bias(prev.reshape(-1), drafter.markov_w1, drafter.markov_w2)
    logits = base + bias.view(*prev.shape, -1).to(base.dtype)
    return torch.softmax(logits.double() / temperature, -1).float()


# Configuration


@pytest.mark.parametrize(
    "options",
    [dict(rejection_sample_method="block"), dict(draft_temperature_only=True)],
    ids=["block", "temperature_only"],
)
@pytest.mark.cpu_only
def test_llm_args_reject_the_options_without_rejection_sampling(options: dict) -> None:
    with pytest.raises(ValueError, match="use_rejection_sampling=True"):
        TorchLlmArgs(
            model="/tmp/dummy_model",
            skip_tokenizer_init=True,
            speculative_config=MTPDecodingConfig(num_nextn_predict_layers=1, **options),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="MTP metadata allocates CUDA buffers")
def test_spec_metadata_carries_the_rejection_options() -> None:
    config = MTPDecodingConfig(
        num_nextn_predict_layers=1,
        use_rejection_sampling=True,
        rejection_sample_method="block",
        draft_temperature_only=True,
    )
    model_config = types.SimpleNamespace(
        num_hidden_layers=32, hidden_size=128, vocab_size=1024, torch_dtype=None
    )
    metadata = get_spec_metadata(config, model_config, max_num_requests=4, max_num_tokens=64)
    assert metadata.rejection_sample_method == "block"
    assert metadata.draft_temperature_only is True


@pytest.mark.parametrize(
    "mode, option",
    [
        ("use_dynamic_tree", dict(rejection_sample_method="block")),
        ("use_dynamic_tree", dict(draft_temperature_only=True)),
        ("use_relaxed_acceptance_for_thinking", dict(rejection_sample_method="block")),
    ],
    ids=["tree-block", "tree-temperature_only", "relaxed-block"],
)
@pytest.mark.cpu_only
def test_rejection_options_are_rejected_where_they_would_not_apply(mode: str, option: dict) -> None:
    # The dynamic tree drafts from a softmax top-k of the raw draft logits; relaxed
    # thinking-phase acceptance returns before the rejection rule runs.
    spec_config = types.SimpleNamespace(
        use_rejection_sampling=True,
        use_dynamic_tree=False,
        use_relaxed_acceptance_for_thinking=False,
        rejection_sample_method="standard",
        draft_temperature_only=False,
    )
    setattr(spec_config, mode, True)
    TorchLlmArgs._validate_rejection_sampling_options(
        types.SimpleNamespace(speculative_config=spec_config)
    )
    for name, value in option.items():
        setattr(spec_config, name, value)
    with pytest.raises(ValueError, match=f"{mode}=True"):
        TorchLlmArgs._validate_rejection_sampling_options(
            types.SimpleNamespace(speculative_config=spec_config)
        )


# The ancestral chain


@pytest.mark.cpu_only
def test_ancestral_chain_with_argmax_draws_is_the_greedy_chain() -> None:
    generator = torch.Generator().manual_seed(2)
    worker = _worker()
    drafter = _drafter(generator)
    base = torch.randn(B, K, V, generator=generator)
    anchors = torch.randint(V, (B,), generator=generator)

    def argmax_sampler(
        mode: Optional[AdvancedSamplingMode], logits: torch.Tensor, *params: Any, **rng: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return logits.argmax(-1).to(torch.int32), torch.softmax(logits, -1)

    def step_bias(prev: torch.Tensor, hidden: Optional[torch.Tensor]) -> torch.Tensor:
        return dspark_markov_step_bias(prev, drafter.markov_w1, drafter.markov_w2)

    tokens, _ = _sample_with(argmax_sampler, worker, drafter, base, anchors, _metadata(B))
    greedy, _ = dspark_markov_chain(base, anchors, step_bias, cast_bias_to_logits=True)
    assert torch.equal(tokens.long(), greedy)


# The worker's sampled path


@pytest.mark.cpu_only
def test_only_sampled_drafts_with_a_markov_head_take_the_ancestral_path() -> None:
    """Argmax drafts and drafters without a head keep the DFlash hook: refine, then the block sampler."""
    generator = torch.Generator().manual_seed(3)
    worker = _worker()
    drafter = _drafter(generator)
    base = torch.randn(B, K, V, generator=generator)
    inputs = {"first_prev_tokens": torch.randint(V, (B,), generator=generator)}
    sampled, greedy = _metadata(B), _metadata(B)
    greedy.wants_advanced_draft_sampling = False
    no_head = types.SimpleNamespace(has_markov_head=False)

    # Argmax drafts of a drafter with a head: the DFlash hook, which applies the greedy chain.
    with patch.object(
        DFlashWorker, "_sample_block_draft_tokens", return_value=("dflash", V)
    ) as hook:
        assert worker._sample_block_draft_tokens(drafter, base, None, inputs, greedy, 0, B) == (
            "dflash",
            V,
        )
        hook.assert_called_once()

    # No head, sampled drafts: the real DFlash hook, unrefined logits straight into the block sampler.
    calls = []
    with (
        patch.object(
            SpecWorkerBase,
            "_sample_from_logits_with_probs",
            staticmethod(_cpu_sampler(generator, calls)),
        ),
        patch.object(DSparkWorker, "_apply_dspark_markov_bias") as bias,
    ):
        tokens, vocab = worker._sample_block_draft_tokens(
            no_head, base, None, inputs, sampled, 0, B
        )
    bias.assert_not_called()
    assert len(calls) == 1 and tokens.shape == (B, K) and vocab == V
    assert sampled.draft_probs_last_dim == V

    # Head and sampled drafts: the ancestral walk, one draw per position, no greedy chain.
    calls = []
    with patch.object(DSparkWorker, "_apply_dspark_markov_bias") as bias:
        tokens, _ = _sample_with(
            _cpu_sampler(generator, calls),
            worker,
            drafter,
            base,
            inputs["first_prev_tokens"],
            sampled,
        )
    bias.assert_not_called()
    assert len(calls) == K and tokens.shape == (B, K)


@pytest.mark.parametrize("num_contexts", [0, 2])
@pytest.mark.cpu_only
def test_sampled_block_stores_the_distribution_each_draft_was_drawn_from(num_contexts: int) -> None:
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


@pytest.mark.parametrize("sampler", ["ancestral", "block", "step"])
@pytest.mark.parametrize("temperature_only", [False, True])
@pytest.mark.cpu_only
def test_sampled_drafts_use_the_draft_sampling_parameters(
    temperature_only: bool, sampler: str
) -> None:
    """The ancestral hook and the shared block and step samplers draw with the request's
    filters, or with its temperature alone under draft_temperature_only."""
    generator = torch.Generator().manual_seed(5)
    worker = _worker()
    drafter = _drafter(generator)
    base = torch.randn(B, K, V, generator=generator)
    anchors = torch.randint(V, (B,), generator=generator)
    meta = _metadata(B, top_k=20, top_p=0.95, min_p=0.05, temperature_only=temperature_only)
    calls = []

    if sampler == "ancestral":
        _sample_with(_cpu_sampler(generator, calls), worker, drafter, base, anchors, meta)
        rows = B
    else:
        with patch.object(
            SpecWorkerBase,
            "_sample_from_logits_with_probs",
            staticmethod(_cpu_sampler(generator, calls)),
        ):
            if sampler == "block":
                worker.advanced_sample_draft_block(base, meta, 0, B)
                rows = B * K  # one launch over the flattened block
            else:
                worker.advanced_sample_draft(base[:, 0], meta, B, draft_step=0)
                rows = B

    assert len(calls) == (K if sampler == "ancestral" else 1)
    for step, call in enumerate(calls):
        if temperature_only:
            assert (call["top_ks"] == INT32_MAX).all()
            assert (call["top_ps"] == 1.0).all() and (call["min_ps"] == 0.0).all()
        else:
            assert (call["top_ks"] == 20).all()
            torch.testing.assert_close(call["top_ps"], torch.full((rows,), 0.95))
            torch.testing.assert_close(call["min_ps"], torch.full((rows,), 0.05))
        if sampler == "ancestral":
            # Each position draws from its own draft slot of the request's window.
            expected_offset = meta.request_offsets + meta.rng_draft_slot(step) * _RNG_SLOT_SPAN
            assert torch.equal(call["offset"], expected_offset)


@pytest.mark.parametrize(
    "mode", [AdvancedSamplingMode.FULL, AdvancedSamplingMode.NO_TOPK], ids=["fused", "flashinfer"]
)
@pytest.mark.cpu_only
def test_draft_sampling_params_temperature_only_keeps_the_temperature(
    mode: AdvancedSamplingMode,
) -> None:
    meta = _metadata(
        B, temperature=0.6, top_k=20, top_p=0.9, min_p=0.1, temperature_only=True, mode=mode
    )
    temps, top_ks, top_ps, min_ps = SpecWorkerBase._draft_sampling_params(meta, 1, B, repeat=2)
    torch.testing.assert_close(temps, torch.full((2 * (B - 1),), 0.6))
    if mode.is_fused:
        assert (top_ks == INT32_MAX).all() and top_ks.dtype == torch.int32
        assert (top_ps == 1.0).all() and (min_ps == 0.0).all()
    else:
        # No filter kernel runs on the flashinfer path, top-p's at p = 1 included.
        assert top_ks is None and top_ps is None and min_ps is None
    meta.draft_temperature_only = False
    _, top_ks, top_ps, min_ps = SpecWorkerBase._draft_sampling_params(meta, 1, B, repeat=2)
    assert (top_ks == 20).all()
    torch.testing.assert_close(top_ps, torch.full((2 * (B - 1),), 0.9))


# CUDA graphs and the real sampler kernels

_needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _capture(fn: Callable[[], Any]) -> tuple[torch.cuda.CUDAGraph, Any]:
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
@pytest.mark.parametrize(
    "temperature_only, mode",
    [
        (True, AdvancedSamplingMode.FULL),
        (False, AdvancedSamplingMode.FULL),
        (True, AdvancedSamplingMode.NO_TOPK),
    ],
    ids=["temperature_only", "filtered", "temperature_only-flashinfer"],
)
def test_sampled_block_captures_in_a_cuda_graph(
    temperature_only: bool, mode: AdvancedSamplingMode
) -> None:
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
        mode=mode,
        device="cuda",
    )
    # The last request is CUDA-graph padding: it routes to the scratch row and
    # its logits are garbage.
    meta.dummy_slot_row = num_gens
    meta.batch_slot_ids[-1] = num_gens
    base[-1] = float("nan")
    inputs = {"first_prev_tokens": anchors}

    graph, (tokens, _) = _capture(
        lambda: worker._sample_block_draft_tokens(drafter, base, None, inputs, meta, 0, num_gens)
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


@_needs_cuda
@pytest.mark.parametrize("method", ["standard", "block"])
def test_rejection_acceptance_routes_and_captures(method: str) -> None:
    generator = torch.Generator().manual_seed(41)
    vocab, num_gens = 2048, 5
    worker = _worker()
    meta = _metadata(
        num_gens,
        vocab=vocab,
        num_slots=num_gens + 1,
        method=method,
        mode=AdvancedSamplingMode.FULL,
        device="cuda",
        top_p=0.95,
    )
    rows = num_gens * (K + 1)
    meta.temperatures = torch.ones(rows, device="cuda")
    meta.top_ks = torch.full((rows,), INT32_MAX, dtype=torch.int32, device="cuda")
    meta.top_ps = torch.full((rows,), 0.95, device="cuda")
    meta.min_ps = torch.zeros(rows, device="cuda")
    logits = 3 * torch.randn(rows, vocab, generator=generator).cuda()
    draft_logits = (
        logits.view(num_gens, K + 1, vocab)[:, :K]
        + torch.randn(num_gens, K, vocab, generator=generator).cuda()
    )
    draft_probs = torch.softmax(draft_logits, -1)
    draft_tokens = torch.multinomial(draft_probs.view(-1, vocab), 1).view(num_gens, K).int()

    def accept() -> tuple[torch.Tensor, torch.Tensor]:
        return worker._sample_and_accept_draft_tokens_rejection(
            logits, draft_tokens, draft_probs, 0, num_gens, meta
        )

    eager_tokens, eager_accepted = accept()
    graph, (graph_tokens, graph_accepted) = _capture(accept)
    graph.replay()
    assert torch.equal(graph_accepted, eager_accepted)
    assert torch.equal(graph_tokens, eager_tokens)
    assert ((eager_accepted >= 1) & (eager_accepted <= K + 1)).all()

    # Replays redraw from the new RNG offsets and verify the new drafts.
    draft_tokens.copy_(torch.multinomial(draft_probs.view(-1, vocab), 1).view(num_gens, K).int())
    meta.request_offsets += 1 << 20
    graph.replay()
    eager_tokens, eager_accepted = accept()
    assert torch.equal(graph_accepted, eager_accepted)
    assert torch.equal(graph_tokens, eager_tokens)

    if method == "block":
        # The decisions are block_verify's on the counter uniforms of the
        # acceptance slot; the emitted token comes from the seeded sampler and
        # must carry residual mass.
        target_probs = SpecWorkerBase._compute_probs_from_logits(
            AdvancedSamplingMode.FULL,
            logits,
            meta.temperatures,
            meta.top_ks,
            meta.top_ps,
            meta.min_ps,
        ).view(num_gens, K + 1, vocab)
        seed = meta.request_seeds
        offset = meta.request_offsets + meta.rng_accept_slot * _RNG_SLOT_SPAN
        drawn = {}

        def record(weights: torch.Tensor) -> torch.Tensor:
            drawn["weights"] = weights
            return eager_tokens[torch.arange(num_gens), eager_accepted.long() - 1]

        expected_tokens, expected_accepted = block_verify(
            draft_tokens,
            draft_probs,
            target_probs,
            counter_uniforms(seed, offset, K),
            sample_emitted=record,
        )
        assert torch.equal(eager_accepted, expected_accepted)
        assert torch.equal(eager_tokens, expected_tokens)
        emitted = eager_tokens[torch.arange(num_gens), eager_accepted.long() - 1].long()
        assert (drawn["weights"].gather(1, emitted[:, None]) > 0).all()


@_needs_cuda
@pytest.mark.parametrize(
    "mode", [AdvancedSamplingMode.FULL, AdvancedSamplingMode.NO_TOPK], ids=["fused", "flashinfer"]
)
@pytest.mark.parametrize("case", ["all_rejected", "all_accepted"])
def test_block_verify_emitted_token_follows_the_residual(
    case: str, mode: AdvancedSamplingMode
) -> None:
    """The seeded sampler of either backend draws the emitted token from the normalized residual.

    Each case pins the accepted length regardless of the uniforms, so thousands
    of rows that differ only in their RNG offset sample one known distribution:
    all drafts impossible under the target -> L = 0 and max(p_0 - q_0, 0);
    draft == target -> L = K and the bonus row p_K.
    """
    generator = torch.Generator().manual_seed(51)
    vocab, rows = 64, 20_000
    worker = _worker()
    meta = _metadata(rows, vocab=vocab, mode=mode, device="cuda")
    target = torch.softmax(torch.randn(1, K + 1, vocab, generator=generator), -1)
    if case == "all_rejected":
        target[:, :, 0] = 0.0
        target = target / target.sum(-1, keepdim=True)
        draft = torch.softmax(torch.randn(1, K, vocab, generator=generator), -1)
        draft[:, :, 0] = 0.5
        draft = draft / draft.sum(-1, keepdim=True)
        tokens = torch.zeros(1, K, dtype=torch.int32)
        expected = (target[0, 0] - draft[0, 0]).clamp_min(0)
        position = 0
    else:
        draft = target[:, :K].clone()
        tokens = torch.multinomial(draft[0], 1).view(1, K).int()
        expected = target[0, K]
        position = K
    expected = expected / expected.sum()
    seed = meta.request_seeds
    offset = meta.request_offsets + 7
    out, accepted = worker._block_verify(
        meta,
        tokens.cuda().expand(rows, -1),
        draft.cuda().expand(rows, -1, -1),
        target.cuda().expand(rows, -1, -1),
        seed,
        offset,
    )
    assert (accepted == position + 1).all()
    histogram = torch.bincount(out[:, position].long().cpu(), minlength=vocab).double()
    stat, critical = chi_square(histogram, expected.double())
    assert stat < critical, f"chi-square {stat:.1f} >= {critical:.1f}"
