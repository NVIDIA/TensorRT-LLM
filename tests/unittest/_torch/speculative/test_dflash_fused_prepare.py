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
"""The fused DFlash drafter-input preparation must match the eager one exactly.

Covers the paged single-latent (MLA) context path: drafter inputs, the stored
context latents, the advanced ``_ctx_len``, ``seq_lens`` and the block-decode
fixup operands.
"""

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.models.modeling_dspark import _build_mla_block_fixup
from tensorrt_llm._torch.speculative.dflash import DFlashWorker
from tensorrt_llm._torch.speculative.spec_tail_kernels import (
    SPEC_TAIL_FUSION_ENV_VAR,
    dflash_noise_embedding,
)
from tensorrt_llm.llmapi import DFlashDecodingConfig
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

K = 6
LAYERS = 5
HEAD_DIM = 72
HIDDEN = 64
VOCAB = 500
PAGE = 4
MAX_BLOCKS = 12
MASK_TOKEN = 499


class _FakeMLADrafter:
    """Deterministic stand-in for the MLA drafter's context-projection hooks."""

    _kv_factor = 1

    def __init__(self):
        gen = torch.Generator(device="cuda").manual_seed(1)
        embed = nn.Embedding(VOCAB, HIDDEN).cuda().to(torch.bfloat16)
        embed.weight.data.copy_(torch.randn(VOCAB, HIDDEN, device="cuda", generator=gen))
        self.draft_model_full = SimpleNamespace(model=SimpleNamespace(embed_tokens=embed))
        self.fc = SimpleNamespace(weight=torch.empty(1, dtype=torch.bfloat16, device="cuda"))
        self.hidden_norm = object()
        self._proj = torch.randn(3 * HIDDEN, LAYERS * HEAD_DIM, device="cuda", generator=gen)
        self._proj = self._proj.to(torch.bfloat16)

    def project_target_hidden(self, hidden):
        return hidden[:, :HIDDEN] * 0.5

    def precompute_context_kv(self, projected, positions):
        latent = projected.repeat(1, 3) @ self._proj
        latent = latent + positions.to(latent.dtype).unsqueeze(1)
        return latent.view(-1, LAYERS, 1, HEAD_DIM).contiguous(), None


def _make_worker(num_slots, counts, ctx_len, gen):
    worker = DFlashWorker(DFlashDecodingConfig(max_draft_len=K), Mapping())
    worker._ctx_paged = True
    worker._ctx_page_size = PAGE
    worker._resolved_block_size = K
    worker._resolved_mask_token_id = MASK_TOKEN
    worker._dummy_slot = num_slots
    worker._max_ctx = MAX_BLOCKS * PAGE - 2
    # Distinct physical pages per (row, block), as a real allocation hands out.
    num_pages = (num_slots + 1) * MAX_BLOCKS
    worker._ctx_kv_buf = [
        torch.randn(num_pages, 1, 1, PAGE, HEAD_DIM, device="cuda", generator=gen).to(
            torch.bfloat16
        )
        for _ in range(LAYERS)
    ]
    pages = torch.randperm(num_pages, device="cuda", generator=gen).to(torch.int32)
    worker._ctx_block_tables = pages.view(num_slots + 1, MAX_BLOCKS)
    worker._ctx_block_counts = counts.clone()
    worker._ctx_len = ctx_len.clone()
    worker._ctx_k_buf = None
    worker._ctx_v_buf = None
    return worker


def _case(num_contexts, num_gens, seed, starved=False):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    batch = num_contexts + num_gens
    num_slots = max(16, batch + 1)
    counts = torch.randint(
        MAX_BLOCKS - 2, MAX_BLOCKS + 1, (num_slots + 1,), device="cuda", generator=gen
    )
    if starved:
        counts[num_contexts : num_contexts + 2] = torch.tensor([0, 2], device="cuda")
    ctx_len = torch.randint(0, 20, (num_slots + 1,), device="cuda", generator=gen)
    worker = _make_worker(num_slots, counts.long(), ctx_len.long(), gen)
    slots = torch.randperm(num_slots, device="cuda", generator=gen)[:batch]
    slots[-1] = num_slots  # a padding request on the dummy slot
    worker._batch_to_slot = slots.long()
    worker._ctx_len[num_slots] = 0
    num_accepted = torch.randint(1, K + 2, (batch,), device="cuda", generator=gen).to(torch.int32)
    accepted = torch.randint(0, VOCAB - 1, (batch, K + 1), device="cuda", generator=gen).to(
        torch.int32
    )
    num_ctx_tokens = 3 * num_contexts
    total_tokens = num_ctx_tokens + num_gens * (K + 1)
    captured = torch.randn(total_tokens, 3 * HIDDEN, device="cuda", generator=gen).to(
        torch.bfloat16
    )
    spec_metadata = SimpleNamespace(
        runtime_draft_len=K, hidden_size=HIDDEN, get_hidden_states=lambda n: captured[:n]
    )

    def attn_metadata():
        return SimpleNamespace(
            num_contexts=num_contexts,
            num_seqs=batch,
            num_ctx_tokens=num_ctx_tokens,
            _seq_lens_cuda=torch.full((batch + 2,), 3, dtype=torch.int32, device="cuda"),
            _seq_lens=torch.full((batch + 2,), 3, dtype=torch.int32),
        )

    return worker, accepted, num_accepted, spec_metadata, attn_metadata, total_tokens


def _run(
    worker, accepted, num_accepted, spec_metadata, attn_metadata, total_tokens, fused, monkeypatch
):
    monkeypatch.setenv(SPEC_TAIL_FUSION_ENV_VAR, "1" if fused else "0")
    draft_model = _FakeMLADrafter()
    md = attn_metadata
    inputs = worker.prepare_1st_drafter_inputs(
        input_ids=None,
        position_ids=None,
        hidden_states=torch.empty(1, HIDDEN, device="cuda"),
        accepted_tokens=accepted,
        num_accepted_tokens=num_accepted,
        attn_metadata=md,
        spec_metadata=spec_metadata,
        draft_model=draft_model,
        total_target_tokens=total_tokens,
    )
    return inputs, md


@pytest.mark.parametrize("num_contexts,num_gens", [(0, 1), (0, 5), (0, 32), (2, 7)])
@pytest.mark.parametrize("starved", [False, True])
def test_fused_prepare_matches_eager(num_contexts, num_gens, starved, monkeypatch):
    if starved and num_gens < 3:
        pytest.skip("needs two starved requests plus the padding one")
    worker, accepted, nacc, spec_md, make_md, total = _case(
        num_contexts, num_gens, num_gens, starved
    )
    ref_worker = copy.deepcopy(worker)
    ref, ref_md = _run(ref_worker, accepted, nacc, spec_md, make_md(), total, False, monkeypatch)
    out, md = _run(worker, accepted, nacc, spec_md, make_md(), total, True, monkeypatch)

    for key in ("noise_embedding", "query_positions", "num_ctx_per_req", "first_prev_tokens"):
        assert torch.equal(out[key], ref[key]), key
    assert torch.equal(out["ctx_cache_batch_idx"], ref["ctx_cache_batch_idx"])
    assert torch.equal(worker._ctx_len, ref_worker._ctx_len)
    assert torch.equal(md._seq_lens_cuda, ref_md._seq_lens_cuda)
    assert torch.equal(md._seq_lens, ref_md._seq_lens)
    if not starved:
        # A starved request clamps several tokens onto one slot; which write
        # lands there is unordered in both paths.
        for layer in range(LAYERS):
            assert torch.equal(worker._ctx_kv_buf[layer], ref_worker._ctx_kv_buf[layer]), layer

    # The fixup handed to dflash_forward equals the one it would build itself.
    rows = ref["ctx_cache_batch_idx"]
    page_tables = ref["ctx_page_table"].index_select(0, rows).long()
    expected = _build_mla_block_fixup(ref["num_ctx_per_req"], page_tables, K, PAGE)
    fixup = out["mla_block_fixup"]
    for field in expected._fields:
        assert torch.equal(getattr(fixup, field), getattr(expected, field)), field
    assert fixup.page_tables_i32.t().stride() == expected.page_tables_i32.t().stride()


@pytest.mark.parametrize("num_gens,block_size,hidden", [(1, 6, 64), (4, 7, 7168), (128, 6, 7168)])
def test_noise_embedding_matches_eager(num_gens, block_size, hidden):
    combined = torch.randn(num_gens + 1, hidden, device="cuda").to(torch.bfloat16)
    expected = combined[num_gens].expand(num_gens, block_size, -1).clone()
    expected[:, 0, :] = combined[:num_gens]
    assert torch.equal(dflash_noise_embedding(combined, block_size), expected)


class _MLAContextProjection:
    """The attributes MLADSparkForCausalLM's context precompute reads."""

    def __init__(self, num_layers=5, seed=0):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        self.kv_lora_rank = 512
        self.qk_rope_head_dim = 64
        self._head_dim = 576
        self._num_attn_layers = num_layers
        self._fused_kv_weight = (
            torch.randn(num_layers * 576, 1024, device="cuda", generator=gen) * 0.05
        ).to(torch.bfloat16)
        self._kv_a_norm_stacked = (
            torch.rand(num_layers, 512, device="cuda", generator=gen) + 0.5
        ).to(torch.bfloat16)
        self._kv_a_norm_eps = 1e-6
        angles = torch.rand(4096, 32, device="cuda", generator=gen) * 6.28
        self._freqs = torch.polar(torch.ones_like(angles), angles)

    def _mla_freqs_cis(self, device):
        return self._freqs

    def _build_fused_kv_buffers(self):
        raise AssertionError("buffers are prebuilt")


@pytest.mark.parametrize("num_tokens", [7, 224, 896])
def test_layer_major_context_latents_match(num_tokens):
    from tensorrt_llm._torch.models.modeling_dspark import MLADSparkForCausalLM

    model = _MLAContextProjection()
    projected = torch.randn(num_tokens, 1024, device="cuda").to(torch.bfloat16)
    positions = torch.randint(0, 4096, (num_tokens,), device="cuda")
    expected, _ = MLADSparkForCausalLM.precompute_context_kv(model, projected, positions)
    out = MLADSparkForCausalLM.precompute_context_kv_layer_major(model, projected, positions)
    if out is None:
        pytest.skip("fused RMSNorm+RoPE kernel unavailable")
    assert torch.equal(out.transpose(0, 1), expected[:, :, 0])
