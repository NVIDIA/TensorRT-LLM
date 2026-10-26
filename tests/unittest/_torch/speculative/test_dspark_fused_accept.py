# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The fused DSpark acceptance / next-token / draft-block-input launches against the eager paths
they replace."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_kernels.blackwell.dspark.generation_prepare import (
    dspark_draft_block_inputs,
    dspark_next_new_tokens,
)
from tensorrt_llm._torch.models.modeling_dspark import (
    DSv4DSparkDraftModel,
    _prepare_dspark_attention_rope,
    build_draft_input_ids,
    precompute_dspark_freqs_cis,
)
from tensorrt_llm._torch.modules.embedding import Embedding
from tensorrt_llm._torch.speculative.dspark import DSv4DSparkWorker
from tensorrt_llm._torch.speculative.interface import SpecWorkerBase

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


class _StubWorker(SpecWorkerBase):
    """Acceptance on given target tokens (the sampler is stubbed out)."""

    _fused_accept = DSv4DSparkWorker._fused_accept

    def __init__(self, target_tokens, force):
        super().__init__()
        self._target_tokens = target_tokens
        self.force_num_accepted_tokens = force

    @property
    def max_draft_len(self) -> int:
        return 8

    def _forward_impl(self, *args, **kwargs):
        raise NotImplementedError

    def _sample_tokens_for_batch(self, logits, spec_metadata, num_contexts, batch_size):
        return self._target_tokens


@pytest.mark.parametrize("force", [0.0, 2.0, 2.51])
def test_fused_accept_matches_eager_path(force):
    """Strict acceptance (with the forced counts) on greedy logits, whose argmax the fused path
    takes itself: ties to the lowest column, -0.0 == 0.0, the first NaN wins, padded rows."""
    K, num_gens, vocab = 5, 4, 129280
    torch.manual_seed(int(force * 10))
    rows = num_gens * (K + 1)
    logits = torch.empty((rows, vocab + 64), dtype=torch.bfloat16, device="cuda")[:, :vocab]
    logits.copy_(torch.randint(-3, 4, (rows, vocab)))
    logits[1] = -1.0
    logits[1, 3], logits[1, 7] = -0.0, 0.0
    logits[2, [9, 300]] = float("nan")
    target = torch.argmax(logits, dim=-1)
    draft = target.view(num_gens, K + 1)[:, :K].clone().to(torch.int32)
    # Partial matches: request g keeps g % (K + 1) leading draft tokens.
    for g in range(num_gens):
        draft[g, g % (K + 1) :] += 1
    spec = SimpleNamespace(
        runtime_draft_len=K,
        draft_tokens=draft.reshape(-1),
        is_ragged_verify=False,
        is_cuda_graph=False,
        use_rejection_sampling=False,
        is_all_greedy_sample=True,
    )
    attn = SimpleNamespace(num_seqs=num_gens, num_contexts=0)
    eager, fused = _StubWorker(target, force), _StubWorker(target, force)
    # Later steps see the RNG counter the earlier ones advanced.
    for _ in range(3):
        want = eager.sample_and_accept_draft_tokens(logits, attn, spec)
        got = fused._fused_accept(logits, attn, spec)
        for g, w in zip(got, want):
            assert g.dtype == w.dtype and torch.equal(g, w)


def test_next_new_tokens_matches_eager_path():
    batch, k = 3, 5
    torch.manual_seed(batch * 100 + k)
    accepted = torch.randint(-(1 << 30), 1 << 30, (batch + 2, k + 1), dtype=torch.int32).cuda()
    draft = torch.randint(-(1 << 30), 1 << 30, (batch, k), dtype=torch.int32).cuda()
    accepted = accepted.t().contiguous().t()  # column-major rows
    draft = torch.cat([draft, draft], dim=1)[:, :k]  # row stride 2k
    # 0 accepted tokens index column -1, which wraps to the last column.
    num_accepted = torch.randint(0, k + 2, (batch,), dtype=torch.int32).cuda()
    orders = (
        torch.arange(batch + 3, dtype=torch.int32),
        torch.randperm(batch + 2)[:batch].to(torch.int32),
    )
    for batch_indices in orders:
        batch_indices = batch_indices.cuda()
        want = SpecWorkerBase._prepare_next_new_tokens(
            None, accepted, draft, batch_indices, batch, num_accepted
        )
        got = dspark_next_new_tokens(accepted, draft, batch_indices, num_accepted)
        assert got.dtype == want.dtype and torch.equal(got, want)


def test_draft_block_inputs_match_eager_path():
    gens, noise, block, hc, vocab, hidden, positions = 5, 64, 5, 4, 64, 7168, 128
    torch.manual_seed(gens)
    embedding = Embedding(vocab, hidden, dtype=torch.bfloat16)
    embedding.weight.data = torch.randn(vocab, hidden, dtype=torch.bfloat16, device="cuda")
    freqs = precompute_dspark_freqs_cis(64, positions, device="cuda")
    # Ids outside the table (and the noise id == vocab) embed to zero rows.
    bonus = torch.randint(-2, vocab + 2, (gens,), device="cuda")
    start_pos = torch.randint(0, positions - block, (gens,), device="cuda")

    ids = build_draft_input_ids(bonus, block_size=block, noise_token_id=noise)
    x = torch.stack([embedding(ids)] * hc, dim=-2)
    want = (
        x,
        ids,
        *_prepare_dspark_attention_rope(start_pos, block, freqs),
        DSv4DSparkDraftModel._initial_pre_mix(SimpleNamespace(uses_lagged_pre=True), x),
    )
    x, ids, (main_freqs, block_freqs), pre_mix = dspark_draft_block_inputs(
        bonus,
        start_pos,
        embedding.weight,
        freqs,
        block_size=block,
        noise_token_id=noise,
        hc_mult=hc,
        pre_mix=True,
    )
    for g, w in zip((x, ids, main_freqs, block_freqs, pre_mix), want):
        assert g.dtype == w.dtype and g.shape == w.shape and torch.equal(g, w)
