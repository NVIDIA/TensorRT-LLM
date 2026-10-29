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
"""The DFlash/DSpark drafter cannot tell a decode-only draft pool from the full reservation.

A decode-only draft KV pool (TLLM_DSPARK_DECODE_ONLY_DRAFT_KV) gives a generation request
far fewer pages than the full reservation, and other physical pages. Whenever the allocation
covers the step's stored tokens and the block after them (the invariant that
tests/unittest/_torch/executor/test_decode_only_draft_kv.py replays for the KV manager), the
drafter inputs, the advanced context lengths, every context latent the drafter can attend and
the block-decode geometry must equal the full reservation's, and no write may leave the
request's own pages. One block fewer must show up as a truncated context.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.models.modeling_dspark import _build_mla_block_fixup
from tensorrt_llm._torch.speculative.dflash import DFlashWorker
from tensorrt_llm._torch.speculative.spec_tail_kernels import SPEC_TAIL_FUSION_ENV_VAR
from tensorrt_llm.llmapi import DFlashDecodingConfig
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

K = 3
BLOCK = 3
LAYERS = 5
HEAD_DIM = 72
HIDDEN = 64
VOCAB = 500
MASK_TOKEN = 499
PAGE = 4
MAX_BLOCKS = 96
NUM_GENS = 24
CANARY = 0  # page owned by no request; unallocated table entries point at it


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


def _step_inputs(seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    ctx_len = torch.randint(0, 200, (NUM_GENS,), device="cuda", generator=gen)
    ctx_len[-1] = 0  # a padding row on the dummy slot keeps an empty context
    accepted_count = torch.randint(1, K + 2, (NUM_GENS,), device="cuda", generator=gen)
    accepted = torch.randint(0, VOCAB - 1, (NUM_GENS, K + 1), device="cuda", generator=gen)
    captured = torch.randn(NUM_GENS * (K + 1), 3 * HIDDEN, device="cuda", generator=gen)
    context = torch.randn(
        NUM_GENS, MAX_BLOCKS * PAGE, LAYERS, HEAD_DIM, device="cuda", generator=gen
    )
    return (
        ctx_len.long(),
        accepted_count.to(torch.int32),
        accepted.to(torch.int32),
        captured.to(torch.bfloat16),
        context.to(torch.bfloat16),
    )


def _worker(counts, ctx_len, context, seed):
    """A worker whose rows own ``counts`` randomly placed pages holding ``context``'s prefix."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    worker = DFlashWorker(DFlashDecodingConfig(max_draft_len=K), Mapping())
    worker._ctx_paged = True
    worker._ctx_page_size = PAGE
    worker._resolved_block_size = BLOCK
    worker._resolved_mask_token_id = MASK_TOKEN
    worker._max_ctx = MAX_BLOCKS * PAGE
    num_slots = NUM_GENS
    worker._dummy_slot = num_slots
    num_pages = NUM_GENS * MAX_BLOCKS + 1
    worker._ctx_kv_buf = [
        torch.randn(num_pages, 1, 1, PAGE, HEAD_DIM, device="cuda", generator=gen).to(
            torch.bfloat16
        )
        for _ in range(LAYERS)
    ]
    owned = 1 + torch.randperm(num_pages - 1, device="cuda", generator=gen).view(
        NUM_GENS, MAX_BLOCKS
    )
    blocks = torch.arange(MAX_BLOCKS, device="cuda")
    table = torch.where(blocks < counts.unsqueeze(1), owned, CANARY).to(torch.int32)
    worker._ctx_block_tables = table
    worker._ctx_block_counts = counts.clone()
    slots = torch.arange(NUM_GENS, device="cuda")
    slots[-1] = worker._dummy_slot
    worker._batch_to_slot = slots
    worker._ctx_len = torch.zeros(num_slots + 1, dtype=torch.long, device="cuda")
    worker._ctx_len[slots] = ctx_len
    worker._ctx_k_buf = None
    worker._ctx_v_buf = None
    for row in range(NUM_GENS):
        cols = torch.arange(int(ctx_len[row]), device="cuda")
        pages = table[row, cols // PAGE].long()
        for layer in range(LAYERS):
            worker._ctx_kv_buf[layer][pages, 0, 0, cols % PAGE] = context[row, cols, layer]
    return worker


def _run(worker, accepted, accepted_count, captured, fused, monkeypatch):
    monkeypatch.setenv(SPEC_TAIL_FUSION_ENV_VAR, "1" if fused else "0")
    total_tokens = NUM_GENS * (K + 1)
    attn_metadata = SimpleNamespace(
        num_contexts=0,
        num_seqs=NUM_GENS,
        num_ctx_tokens=0,
        _seq_lens_cuda=torch.full((NUM_GENS + 2,), 3, dtype=torch.int32, device="cuda"),
        _seq_lens=torch.full((NUM_GENS + 2,), 3, dtype=torch.int32),
    )
    spec_metadata = SimpleNamespace(
        runtime_draft_len=K, hidden_size=HIDDEN, get_hidden_states=lambda n: captured[:n]
    )
    canary = [buf[CANARY].clone() for buf in worker._ctx_kv_buf]
    inputs = worker.prepare_1st_drafter_inputs(
        input_ids=None,
        position_ids=None,
        hidden_states=torch.empty(1, HIDDEN, device="cuda"),
        accepted_tokens=accepted,
        num_accepted_tokens=accepted_count,
        attn_metadata=attn_metadata,
        spec_metadata=spec_metadata,
        draft_model=_FakeMLADrafter(),
        total_target_tokens=total_tokens,
    )
    canary_intact = all(
        torch.equal(buf[CANARY], ref) for buf, ref in zip(worker._ctx_kv_buf, canary)
    )
    return inputs, attn_metadata, canary_intact


def _logical(worker, rows, lengths):
    """Each row's first ``lengths[row]`` context latents, read through its block table."""
    out = []
    for row in rows:
        cols = torch.arange(int(lengths[row]), device="cuda")
        pages = worker._ctx_block_tables[row, cols // PAGE].long()
        out.append(
            torch.stack([buf[pages, 0, 0, cols % PAGE] for buf in worker._ctx_kv_buf], dim=1)
        )
    return out


def _fixup(inputs):
    if "mla_block_fixup" in inputs:
        return inputs["mla_block_fixup"]
    rows = inputs["ctx_cache_batch_idx"]
    tables = inputs["ctx_page_table"].index_select(0, rows).long()
    return _build_mla_block_fixup(inputs["num_ctx_per_req"], tables, BLOCK, PAGE)


@pytest.mark.parametrize("fused", [False, True], ids=["eager", "fused"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_decode_only_allocation_is_invisible_to_the_drafter(seed, fused, monkeypatch):
    ctx_len, accepted_count, accepted, captured, context = _step_inputs(seed)
    new_len = ctx_len + accepted_count.long()
    full = torch.full_like(ctx_len, MAX_BLOCKS)
    # Smallest allocation the decode-only invariant guarantees: the step's stores and
    # the block latents past the advanced context.
    needed = torch.maximum(ctx_len + K + 1, new_len + BLOCK)
    decode_only = (needed + PAGE - 1) // PAGE

    ref_worker = _worker(full, ctx_len, context, seed=100 + seed)
    worker = _worker(decode_only, ctx_len, context, seed=200 + seed)
    ref, ref_md, _ = _run(ref_worker, accepted, accepted_count, captured, fused, monkeypatch)
    out, md, canary_intact = _run(worker, accepted, accepted_count, captured, fused, monkeypatch)

    for key in ("noise_embedding", "query_positions", "num_ctx_per_req", "first_prev_tokens"):
        assert torch.equal(out[key], ref[key]), key
    assert torch.equal(out["num_ctx_per_req"][:-1], new_len[:-1])  # nothing truncated
    assert torch.equal(worker._ctx_len, ref_worker._ctx_len)
    assert torch.equal(md._seq_lens_cuda, ref_md._seq_lens_cuda)
    rows = range(NUM_GENS - 1)  # the padding row's context is discarded by design
    for got, want in zip(_logical(worker, rows, new_len), _logical(ref_worker, rows, new_len)):
        assert torch.equal(got, want)
    assert canary_intact

    fixup, ref_fixup = _fixup(out), _fixup(ref)
    for field in ("offsets", "valid", "seq_lens_i32", "kv_bounds_i32"):
        assert torch.equal(getattr(fixup, field), getattr(ref_fixup, field)), field
    assert bool((fixup.pages != CANARY).all())  # every block latent lands in an owned page


@pytest.mark.parametrize("fused", [False, True], ids=["eager", "fused"])
def test_one_block_short_truncates_the_context(fused, monkeypatch):
    ctx_len, accepted_count, accepted, captured, context = _step_inputs(3)
    new_len = ctx_len + accepted_count.long()
    needed = torch.maximum(ctx_len + K + 1, new_len + BLOCK)
    short = (needed + PAGE - 1) // PAGE - 1
    worker = _worker(short, ctx_len, context, seed=300)
    out, _, _ = _run(worker, accepted, accepted_count, captured, fused, monkeypatch)
    assert bool((out["num_ctx_per_req"][:-1] < new_len[:-1]).all())
