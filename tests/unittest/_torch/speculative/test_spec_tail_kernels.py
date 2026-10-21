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
"""The spec-decode step kernels reproduce the eager chains they replace."""

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from tensorrt_llm._torch.models.modeling_dspark import _build_mla_block_fixup
from tensorrt_llm._torch.speculative.dflash import DFlashWorker
from tensorrt_llm._torch.speculative.spec_tail_kernels import (
    decode_draft_block_table,
    strict_accept,
)
from tensorrt_llm.llmapi.llm_args import DFlashDecodingConfig
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels")


def _eager_strict_accept(target, draft, num_ctx, batch):
    k = draft.shape[-1]
    num_gens = batch - num_ctx
    accepted = torch.zeros((batch, k + 1), dtype=torch.int32, device="cuda")
    num_accepted = torch.ones(batch, dtype=torch.int32, device="cuda")
    accepted[:num_ctx, 0] = target[:num_ctx]
    gen_target = target[num_ctx:].reshape(num_gens, k + 1)
    num_accepted[num_ctx:] += torch.cumprod((draft == gen_target[:, :k]).int(), dim=-1).sum(1)
    accepted[num_ctx:] = gen_target
    return accepted, num_accepted


@pytest.mark.parametrize("num_ctx,num_gens", [(0, 1), (3, 0), (2, 5), (1, 130), (200, 7)])
@pytest.mark.parametrize("draft_len", [1, 3, 7])
def test_strict_accept_matches_eager(num_ctx, num_gens, draft_len):
    gen = torch.Generator(device="cuda").manual_seed(num_gens * 10 + draft_len)
    batch = num_ctx + num_gens
    target = torch.randint(
        0, 4, (num_ctx + num_gens * (draft_len + 1),), device="cuda", generator=gen
    )
    # A strided view, as the draft tokens of a wider buffer.
    wide = torch.randint(0, 4, (num_gens, draft_len + 2), device="cuda", generator=gen)
    draft = wide.to(torch.int32)[:, 1 : draft_len + 1]
    accepted, num_accepted = strict_accept(target, draft, num_ctx, batch)
    ref_accepted, ref_num = _eager_strict_accept(target, draft, num_ctx, batch)
    assert torch.equal(num_accepted, ref_num)
    assert torch.equal(accepted[:num_ctx, 0], ref_accepted[:num_ctx, 0])
    assert torch.equal(accepted[num_ctx:], ref_accepted[num_ctx:])


def test_decode_draft_block_table_matches_eager():
    num_pools, rows, n_blocks, divisor = 2, 9, 3000, 6
    gen = torch.Generator(device="cuda").manual_seed(1)
    src = torch.randint(
        -(1 << 20), 1 << 20, (num_pools, rows + 3, 2, n_blocks), device="cuda", generator=gen
    )
    # About a quarter of the entries are the zero offset and a quarter the -1
    # of an unallocated block; besides, one row is all negative and one all zero.
    kind = torch.randint(0, 4, src.shape, device="cuda", generator=gen)
    src = torch.where(kind == 0, 0, torch.where(kind == 1, -1, src)).to(torch.int32)
    view = src[1, :rows, 0]
    view[0] = -torch.randint(1, 1 << 20, (n_blocks,), device="cuda", generator=gen).to(torch.int32)
    view[1] = 0
    assert bool((view < 0).float().mean() > 0.3) and bool((view == 0).float().mean() > 0.2)
    table = torch.full((rows + 3, n_blocks), -7, dtype=torch.int32, device="cuda")
    counts = torch.full((rows + 3,), -7, dtype=torch.long, device="cuda")
    decode_draft_block_table(view, divisor, table, counts)
    encoded = view.to(torch.int64)
    assert torch.equal(counts[:rows], (encoded >= 0).sum(dim=1))
    expected = encoded.clamp(min=0).div(divisor, rounding_mode="floor").to(torch.int32)
    assert torch.equal(table[:rows], expected)
    assert counts[0].item() == 0 and bool((table[0] == 0).all())
    assert counts[1].item() == n_blocks and bool((table[1] == 0).all())
    assert bool((table[rows:] == -7).all()) and bool((counts[rows:] == -7).all())


class _LatentDrafter(nn.Module):
    """The surface of a single-latent (MLA) drafter the worker's preparation reads."""

    _kv_factor = 1
    lilicorr = None

    def __init__(self, vocab, hidden, features, layers, head_dim, mask_token_id):
        super().__init__()
        gen = torch.Generator(device="cuda").manual_seed(5)
        bf16 = dict(device="cuda", dtype=torch.bfloat16)
        self.mask_token_id = mask_token_id
        self.fc = nn.Linear(features, hidden, bias=False, **bf16)
        self.fc.weight.data = torch.randn(hidden, features, generator=gen, **bf16) * 0.05
        self.hidden_norm = nn.RMSNorm(hidden, eps=1e-6, **bf16)
        embed = nn.Embedding(vocab, hidden, **bf16)
        embed.weight.data = torch.randn(vocab, hidden, generator=gen, **bf16)
        self.draft_model_full = SimpleNamespace(model=SimpleNamespace(embed_tokens=embed))
        self.latent_weight = torch.randn(layers * head_dim, hidden, generator=gen, **bf16) * 0.1
        self.layers, self.head_dim = layers, head_dim

    def project_target_hidden(self, hidden_states):
        return self.hidden_norm(self.fc(hidden_states.to(self.fc.weight.dtype)))

    def precompute_context_kv(self, projected, positions):
        latent = F.linear(projected, self.latent_weight).view(-1, self.layers, self.head_dim)
        latent = latent + (positions.view(-1, 1, 1) % 7).to(latent.dtype)
        latent[0, 0, 0] = -latent[0, 0, 0].abs() - 1  # a negative entry for the signed zero
        return latent.unsqueeze(2).contiguous(), None


def _paged_worker(draft_len, page_size, max_blocks, num_slots, layers, head_dim, max_ctx, gen):
    worker = DFlashWorker(DFlashDecodingConfig(max_draft_len=draft_len), Mapping())
    pages = num_slots * max_blocks
    # Nonzero contents, so that a write (even of a masked, zeroed latent) and a
    # missing or stray one are both visible.
    storage = (
        torch.rand((pages, layers, 1, 1, page_size, head_dim), device="cuda", generator=gen) + 0.5
    ).to(torch.bfloat16)
    pools = [storage[:, layer] for layer in range(layers)]
    worker._ctx_paged = True
    worker._ctx_kv_buf = pools
    worker._ctx_kv_ptrs = torch.tensor([p.data_ptr() for p in pools], device="cuda")
    worker._ctx_page_size = page_size
    worker._compute_block_size = draft_len
    worker._max_ctx = max_ctx
    worker._dummy_slot = num_slots - 1
    worker._ctx_len = torch.zeros(num_slots, dtype=torch.long, device="cuda")
    worker._batch_to_slot_buffer = SimpleNamespace(
        tensor=torch.zeros(num_slots - 1, dtype=torch.long, device="cuda")
    )
    perm = torch.randperm(pages, device="cuda", generator=gen).to(torch.int32)
    worker._ctx_block_tables = perm[: num_slots * max_blocks].view(num_slots, max_blocks).clone()
    worker._ctx_block_counts = torch.zeros(num_slots, dtype=torch.long, device="cuda")
    worker._resolved_mask_token_id = 3
    return worker, storage


@pytest.mark.parametrize("draft_len", [3, 7])
def test_paged_latent_prepare_matches_eager(draft_len):
    vocab, hidden, features, layers, head_dim = 100, 64, 128, 3, 96
    page_size, max_blocks, num_slots, max_ctx = 16, 6, 10, 70
    num_ctx_reqs, num_gens = 1, 6
    batch = num_ctx_reqs + num_gens
    width = draft_len + 1
    gen = torch.Generator(device="cuda").manual_seed(draft_len)
    worker, storage = _paged_worker(
        draft_len, page_size, max_blocks, num_slots, layers, head_dim, max_ctx, gen
    )
    drafter = _LatentDrafter(vocab, hidden, features, layers, head_dim, mask_token_id=3)

    # Rows: an ordinary request, one at the max_ctx ceiling, one whose context
    # outruns its pages, one with no pages, and two padding rows on the dummy slot.
    slots = torch.tensor([0, 2, 4, 1, 3, worker._dummy_slot, worker._dummy_slot], device="cuda")
    worker._batch_to_slot_buffer.tensor[:batch] = slots
    worker._ctx_len[:5] = torch.tensor([10, 3, 66, 5, 40], device="cuda")
    worker._ctx_block_counts[:batch] = torch.tensor([0, 3, 6, 2, 0, 1, 1], device="cuda")
    num_accepted = torch.tensor([1, 2, width, 3, 1, 2, 1], dtype=torch.int32, device="cuda")
    accepted = torch.randint(
        0, vocab, (batch, width), dtype=torch.int32, device="cuda", generator=gen
    )
    num_ctx_tokens = 5
    captured = torch.randn(
        num_ctx_tokens + num_gens * width + 4, features, device="cuda", generator=gen
    )
    spec_metadata = SimpleNamespace(
        runtime_draft_len=draft_len,
        hidden_size=0,
        captured_hidden_states=captured,
        get_hidden_states=lambda n: captured[:n],
    )

    def attn_metadata():
        return SimpleNamespace(
            num_contexts=num_ctx_reqs,
            num_seqs=batch,
            num_ctx_tokens=num_ctx_tokens,
            _seq_lens_cuda=torch.full((batch,), 99, dtype=torch.int32, device="cuda"),
            _seq_lens=torch.full((batch,), 99, dtype=torch.int32),
        )

    def prepare(fused):
        w = copy.copy(worker)
        w._ctx_len = worker._ctx_len.clone()
        pool_storage = storage.clone()
        w._ctx_kv_buf = [pool_storage[:, layer] for layer in range(layers)]
        w._ctx_kv_ptrs = torch.tensor([p.data_ptr() for p in w._ctx_kv_buf], device="cuda")
        w._fused_gen_prepare_applies = lambda *_: fused
        md = attn_metadata()
        inputs = w.prepare_1st_drafter_inputs(
            input_ids=None,
            position_ids=None,
            hidden_states=captured,
            accepted_tokens=accepted,
            num_accepted_tokens=num_accepted,
            attn_metadata=md,
            spec_metadata=spec_metadata,
            draft_model=drafter,
            total_target_tokens=captured.shape[0],
        )
        return inputs, w._ctx_len, md, pool_storage

    eager, eager_len, eager_md, eager_pool = prepare(False)
    fused, fused_len, fused_md, fused_pool = prepare(True)

    for key in ("noise_embedding", "query_positions", "num_ctx_per_req", "first_prev_tokens"):
        assert torch.equal(fused[key], eager[key]), key
    assert torch.equal(fused["ctx_cache_batch_idx"], eager["ctx_cache_batch_idx"])
    assert torch.equal(fused_len, eager_len)
    assert torch.equal(fused_md._seq_lens_cuda, eager_md._seq_lens_cuda)
    assert torch.equal(fused_md._seq_lens, eager_md._seq_lens)
    # Tokens whose column is clamped into a short allocation share a
    # destination, and which of them lands last is unordered in both paths;
    # every other pool entry must match bit for bit.
    writers = {}
    old_len = worker._ctx_len.tolist()
    counts = worker._ctx_block_counts.tolist()
    for g in range(num_gens):
        row = num_ctx_reqs + g
        capacity = counts[row] * page_size
        for j in range(width):
            col = max(min(old_len[slots[row].item()] + j, capacity - 1), 0)
            page = worker._ctx_block_tables[row, col // page_size].item()
            writers.setdefault((page, col % page_size), []).append((g, j))
    shared = torch.zeros(storage.shape[0], page_size, dtype=torch.bool)
    written = torch.zeros(storage.shape[0], page_size, dtype=torch.bool)
    for (page, offset), tokens in writers.items():
        shared[page, offset] = len(tokens) > 1
        written[page, offset] = True
    entry_shape = (storage.shape[0], 1, 1, 1, page_size, 1)
    keep = ~shared.view(entry_shape).cuda()
    untouched = ~written.view(entry_shape).cuda()
    fused_bits, eager_bits = fused_pool.view(torch.int16), eager_pool.view(torch.int16)
    assert torch.equal(fused_bits * keep, eager_bits * keep)
    # Every entry outside this step's destinations keeps its contents, and
    # every destination is overwritten (a stored or a masked, zeroed latent).
    init_bits = storage.view(torch.int16)
    for bits in (fused_bits, eager_bits):
        assert torch.equal(bits * untouched, init_bits * untouched)
        changed = (bits != init_bits).any(dim=-1).flatten(1, 3).any(dim=1)
        assert bool(changed[written.cuda()].all())
    assert bool(shared.any()) and bool(untouched.any())

    rows = eager["ctx_cache_batch_idx"]
    fixup = _build_mla_block_fixup(
        eager["num_ctx_per_req"],
        worker._ctx_block_tables.index_select(0, rows).long(),
        draft_len,
        page_size,
    )
    placement = fused["block_placement"]
    assert eager["block_placement"] is None
    assert torch.equal(placement.pages, fixup.pages)
    assert torch.equal(placement.offsets, fixup.offsets)
    assert torch.equal(placement.seq_lens, fixup.seq_lens_i32)
    assert torch.equal(placement.kv_bounds, fixup.kv_bounds_i32)
