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
"""Sampled DSpark drafting and the rejection options through _forward_impl, get_spec_metadata and TorchLlmArgs."""

import contextlib
import types
from typing import Any, Callable, Optional
from unittest.mock import patch

import pytest
import torch

from tensorrt_llm._torch.models.modeling_dspark import _DSparkHeadMixin
from tensorrt_llm._torch.models.modeling_speculative import dspark_markov_step_bias
from tensorrt_llm._torch.speculative.block_verification import block_verify, counter_uniforms
from tensorrt_llm._torch.speculative.dspark import DSparkWorker
from tensorrt_llm._torch.speculative.interface import SpecWorkerBase, rejection_sampling_one_model
from tensorrt_llm._torch.speculative.utils import get_spec_metadata
from tensorrt_llm.llmapi.llm_args import (
    AdvancedSamplingMode,
    DSparkDecodingConfig,
    MTPDecodingConfig,
    TorchLlmArgs,
)
from tensorrt_llm.mapping import Mapping

B, K, V, H, RANK = 128, 3, 256, 32, 8
INT32_MAX = torch.iinfo(torch.int32).max
TOP_P = 0.9


class _Drafter(_DSparkHeadMixin):
    """A DSpark drafter whose block forward returns fixed hidden states."""

    def __init__(self, generator: torch.Generator) -> None:
        self._dspark_markov_rank = RANK
        self.markov_w1 = torch.randn(V, RANK, generator=generator).cuda()
        self.markov_w2 = torch.randn(V, RANK, generator=generator).cuda()
        self.proj = (0.5 * torch.randn(H, V, generator=generator)).cuda()
        self.block_hidden = torch.randn(B * K, H, generator=generator).cuda()
        self.lm_head = None

    def dflash_forward(self, **kwargs: Any) -> torch.Tensor:
        return self.block_hidden

    def logits_processor(
        self, hidden: torch.Tensor, lm_head: Any, attn_metadata: Any, return_context_logits: bool
    ) -> torch.Tensor:
        return hidden.float() @ self.proj

    def base_logits(self) -> torch.Tensor:
        """``[B, K, V]`` unbiased block logits."""
        return self.logits_processor(self.block_hidden, None, None, True).view(B, K, V)


def _metadata(
    generator: torch.Generator, target_logits: torch.Tensor, method: str, temperature_only: bool
) -> types.SimpleNamespace:
    """Spec metadata of a generation-only batch whose previous block awaits verification.

    The previous block's proposals are perturbed target rows, so verification accepts
    some drafts and the two rules can disagree.
    """
    rows = B * (K + 1)
    target_rows = target_logits.cpu().view(B, K + 1, V)[:, :K] / 0.8
    noisy = target_rows + 1.5 * torch.randn(B, K, V, generator=generator)
    draft_probs = torch.cat([torch.softmax(noisy, -1), torch.full((1, K, V), 1.0 / V)])
    previous_drafts = torch.multinomial(draft_probs[:B].reshape(-1, V), 1, generator=generator)
    return types.SimpleNamespace(
        runtime_draft_len=K,
        max_draft_len=K,
        is_cuda_graph=False,
        request_ids=None,
        batch_indices_cuda=torch.arange(B, dtype=torch.int32).cuda(),
        is_all_greedy_sample=False,
        wants_advanced_draft_sampling=True,
        use_rejection_sampling=True,
        rejection_sample_method=method,
        draft_temperature_only=temperature_only,
        advanced_sampling_mode=AdvancedSamplingMode.FULL,
        enable_penalty=False,
        is_ragged_verify=False,
        verify_lens=None,
        qo_indptr=None,
        full_draft_probs=None,
        d2t_target_indices=None,
        draft_tokens=previous_drafts.view(-1).int().cuda(),
        draft_probs=draft_probs.cuda(),
        draft_probs_last_dim=V,
        draft_probs_vocab_size=V,
        draft_vocab_size=V,
        vocab_size=V,
        dummy_slot_row=0,
        batch_slot_ids=torch.arange(B, dtype=torch.long).flip(0).cuda(),
        request_temperatures=torch.full((B,), 0.8).cuda(),
        request_top_ks=torch.full((B,), INT32_MAX, dtype=torch.int32).cuda(),
        request_top_ps=torch.full((B,), TOP_P).cuda(),
        request_min_ps=torch.zeros(B).cuda(),
        temperatures=torch.full((rows,), 0.8).cuda(),
        top_ks=torch.full((rows,), INT32_MAX, dtype=torch.int32).cuda(),
        top_ps=torch.full((rows,), TOP_P).cuda(),
        min_ps=torch.zeros(rows).cuda(),
        request_seeds=(torch.arange(B, dtype=torch.int64) + 11).cuda(),
        request_offsets=(1000 * torch.arange(B, dtype=torch.int64)).cuda(),
        rng_draft_slot=lambda step: K + 1 + step,
        rng_accept_slot=2 * K + 1,
    )


def _argmin_sampler(calls: list) -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
    """Draft sampler stand-in: draws the least likely token, returns softmax(logits / T).

    Drawing the argmin keeps every draw off the argmax, which is where the ancestral
    chain and the greedy-chained proposal disagree.
    """

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
        calls.append((top_ks.clone(), top_ps.clone(), min_ps.clone()))
        probs = torch.softmax(logits.float() / temperatures.float()[:, None], -1)
        return logits.argmin(-1).int(), probs

    return sample


def _noop(*args: Any, **kwargs: Any) -> None:
    return None


def _step(
    worker: DSparkWorker,
    drafter: _Drafter,
    meta: types.SimpleNamespace,
    target_logits: torch.Tensor,
    calls: list,
) -> dict:
    """One generation step through ``_forward_impl`` with the attention plumbing stubbed."""
    worker._compute_block_size = K
    worker._ctx_len = torch.zeros(B + 1, dtype=torch.long, device="cuda")
    worker._ctx_len_host = [0] * (B + 1)
    worker._req_ctx_pos = {}
    for stub in (
        "_lazy_init_ctx_buffers",
        "_refresh_ctx_block_tables",
        "_prepare_attn_metadata_for_dflash",
        "_prepare_kv_for_draft_forward",
        "_restore_attn_metadata_from_spec_dec",
        "_apply_kv_rewind_after_draft",
    ):
        setattr(worker, stub, _noop)
    worker.get_draft_kv_cache_manager = lambda resource_manager: None
    worker.draft_kv_cache_context = lambda *a: contextlib.nullcontext()
    worker._draft_slot_ids = lambda model, num_gens, block, k: torch.arange(
        num_gens * k, device="cuda"
    )

    def first_inputs(**kwargs: Any) -> dict:
        accepted, counts = kwargs["accepted_tokens"], kwargs["num_accepted_tokens"]
        bonus = accepted.gather(1, (counts.long() - 1).unsqueeze(1)).squeeze(1)
        names = ("noise_embedding", "query_positions", "num_ctx_per_req", "ctx_k_cache")
        names += ("ctx_v_cache", "ctx_cache_batch_idx", "ctx_kv_cache", "ctx_page_table")
        return {**dict.fromkeys(names + ("block_placement",)), "first_prev_tokens": bonus.long()}

    worker.prepare_1st_drafter_inputs = first_inputs
    attn = types.SimpleNamespace(num_seqs=B, num_contexts=0, kv_cache_manager=None)
    tokens = torch.zeros(B * (K + 1), dtype=torch.long, device="cuda")
    with patch.object(
        SpecWorkerBase, "_sample_from_logits_with_probs", staticmethod(_argmin_sampler(calls))
    ):
        return worker._forward_impl(
            tokens, tokens.clone(), None, target_logits, attn, meta, drafter
        )


def _worker() -> DSparkWorker:
    worker = DSparkWorker(DSparkDecodingConfig(max_draft_len=K), Mapping())
    worker.guided_decoder = None
    return worker


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the worker runs on CUDA")


@pytest.mark.parametrize("temperature_only", [False, True])
def test_forward_draws_the_block_ancestrally_and_stores_each_proposal(
    temperature_only: bool,
) -> None:
    generator = torch.Generator().manual_seed(7)
    drafter = _Drafter(generator)
    target_logits = (2 * torch.randn(B * (K + 1), V, generator=generator)).cuda()
    meta = _metadata(generator, target_logits, "standard", temperature_only)
    calls = []

    out = _step(_worker(), drafter, meta, target_logits, calls)

    drafts = out["next_draft_tokens"].long()
    counts = out["new_tokens_lens"].long()
    anchors = out["new_tokens"].long().gather(1, (counts - 1).unsqueeze(1)).squeeze(1)
    prev = torch.cat([anchors.view(-1, 1), drafts[:, :-1]], dim=1)
    bias = dspark_markov_step_bias(prev.reshape(-1), drafter.markov_w1, drafter.markov_w2)
    logits = drafter.base_logits() + bias.view(B, K, V).to(torch.float32)
    # Each draft is the stand-in's draw from its position's logits, conditioned on the
    # token drawn before it, and that exact distribution is the stored proposal.
    assert torch.equal(drafts, logits.argmin(-1))
    expected = torch.softmax(logits / 0.8, -1)
    stored = meta.draft_probs[meta.batch_slot_ids, :K, :V]
    torch.testing.assert_close(stored, expected, rtol=1e-4, atol=1e-6)
    assert meta.draft_probs_last_dim == V
    # One draw per block position, with the request's filters or with temperature only.
    assert len(calls) == K
    for top_ks, top_ps, min_ps in calls:
        assert torch.equal(top_ks, torch.full_like(top_ks, INT32_MAX))
        assert torch.equal(top_ps, torch.full_like(top_ps, 1.0 if temperature_only else TOP_P))
        assert torch.equal(min_ps, torch.zeros_like(min_ps))


@pytest.mark.parametrize("method", ["standard", "block"])
def test_forward_verifies_the_previous_block_with_the_configured_rule(method: str) -> None:
    generator = torch.Generator().manual_seed(9)
    drafter = _Drafter(generator)
    target_logits = (2 * torch.randn(B * (K + 1), V, generator=generator)).cuda()
    meta = _metadata(generator, target_logits, method, temperature_only=False)
    draft_tokens = meta.draft_tokens.view(B, K).clone()
    draft_probs = meta.draft_probs[meta.batch_slot_ids, :K, :V].clone()
    worker = _worker()
    target_probs = SpecWorkerBase._compute_probs_from_logits(
        AdvancedSamplingMode.FULL,
        target_logits,
        meta.temperatures,
        meta.top_ks,
        meta.top_ps,
        meta.min_ps,
    ).view(B, K + 1, V)
    seed, offset = worker._rng_state_per_request(meta, 0, B, slot=meta.rng_accept_slot)
    _, standard = rejection_sampling_one_model(
        draft_probs, draft_tokens, target_probs, seed, offset
    )
    uniforms = counter_uniforms(seed, offset, K + 1)
    _, block = block_verify(
        draft_tokens, draft_probs, target_probs, uniforms[:, :K], uniforms[:, K]
    )
    # The batch tells the two rules apart.
    assert not torch.equal(standard, block)

    out = _step(worker, drafter, meta, target_logits, [])

    counts = out["new_tokens_lens"]
    assert torch.equal(counts, standard if method == "standard" else block)
    accepted = out["new_tokens"]
    for row in range(B):
        n = int(counts[row]) - 1
        assert torch.equal(accepted[row, :n], draft_tokens[row, :n])


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
    "options",
    [dict(rejection_sample_method="block"), dict(draft_temperature_only=True)],
    ids=["block", "temperature_only"],
)
def test_llm_args_reject_the_options_without_rejection_sampling(options: dict) -> None:
    with pytest.raises(ValueError, match="use_rejection_sampling=True"):
        TorchLlmArgs(
            model="/tmp/dummy_model",
            skip_tokenizer_init=True,
            speculative_config=MTPDecodingConfig(num_nextn_predict_layers=1, **options),
        )
