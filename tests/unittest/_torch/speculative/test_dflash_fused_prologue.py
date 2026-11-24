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
"""The fused decode-step drafter bookkeeping against the torch code it replaces, bit for bit.

The cases follow NVIDIA/TensorRT-LLM#20550's ``test_dflash_fused_prologue.py``: block-table
refresh, single-latent context scatter, generation prologue, noise block, context-KV store
with the ``ctx_len`` bookkeeping, MLA block fixup and block-latent pool write. Here each
one exercises the kernel of this tree that covers it (``TLLM_SPEC_TAIL_FUSION=1``,
``spec_tail_kernels``), and ``TLLM_DFLASH_FUSED_PROLOGUE=1`` adds the reuse of the
shape-only block-decode operands.
"""

import types

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.models import modeling_dspark
from tensorrt_llm._torch.speculative import dflash
from tensorrt_llm._torch.speculative.spec_tail_kernels import (
    SPEC_TAIL_FUSION_ENV_VAR,
    dflash_gen_prepare,
    dflash_noise_embedding,
    store_mla_context_latents,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

DEVICE = "cuda"
PAGE_SIZE = 32
NUM_LAYERS = 5
HEAD_DIM = 576


def _single_latent_pool(num_pages: int, dtype=torch.bfloat16):
    """Per-layer ``[pages, 1, 1, page_size, head_dim]`` views of one allocation."""
    storage = torch.zeros(
        NUM_LAYERS, num_pages, 1, 1, PAGE_SIZE, HEAD_DIM, dtype=dtype, device=DEVICE
    )
    return storage, [storage[layer] for layer in range(NUM_LAYERS)]


def _block_table(num_rows: int, num_cols: int, num_pages: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    table = torch.randperm(num_pages, generator=gen)[: num_rows * num_cols].view(num_rows, num_cols)
    return table.to(torch.int32).to(DEVICE)


def _pool_ptrs(views) -> torch.Tensor:
    return torch.tensor([v.data_ptr() for v in views], dtype=torch.int64, device=DEVICE)


@pytest.mark.parametrize("num_seqs", [1, 3, 8])
def test_refresh_ctx_block_tables(num_seqs: int, monkeypatch: pytest.MonkeyPatch) -> None:
    max_rows, num_cols, divisor = 8, 40, 2
    gen = torch.Generator().manual_seed(num_seqs)
    src = torch.randint(0, 1000, (2, max_rows, 2, num_cols), generator=gen, dtype=torch.int32)
    src[..., num_cols - 7 :] = -1
    src = src.to(DEVICE)
    attn_metadata = types.SimpleNamespace(draft_kv_cache_block_offsets=src)
    results = []
    for fused in ("0", "1"):
        monkeypatch.setenv(SPEC_TAIL_FUSION_ENV_VAR, fused)
        worker = types.SimpleNamespace(
            _ctx_block_tables=torch.full(
                (max_rows, num_cols), -5, dtype=torch.int32, device=DEVICE
            ),
            _ctx_block_counts=torch.full((max_rows,), -5, dtype=torch.long, device=DEVICE),
            _ctx_pool_idx=1,
            _ctx_block_divisor=divisor,
            _has_unified_draft_cache=lambda: False,
        )
        assert dflash.DFlashWorker._refresh_ctx_block_tables(worker, attn_metadata, num_seqs)
        results.append((worker._ctx_block_tables, worker._ctx_block_counts))
    assert torch.equal(results[0][0], results[1][0])
    assert torch.equal(results[0][1], results[1][1])


@pytest.mark.parametrize("num_tokens", [1, 8, 37])
def test_store_context_kv_paged_single_latent(num_tokens: int) -> None:
    """The single-latent context scatter: the multi-layer store kernel with every token
    kept writes what ``_store_context_kv_paged`` writes."""
    num_rows, num_cols, num_pages = 4, 6, 64
    table = _block_table(num_rows, num_cols, num_pages, seed=num_tokens)
    gen = torch.Generator().manual_seed(num_tokens)
    rows = torch.randint(0, num_rows, (num_tokens,), generator=gen).to(DEVICE)
    # Distinct positions, so no two tokens target the same slot.
    positions = torch.randperm(num_cols * PAGE_SIZE, generator=gen)[:num_tokens].to(DEVICE)
    k = (
        torch.randn(num_tokens, NUM_LAYERS, 1, HEAD_DIM, generator=gen)
        .to(torch.bfloat16)
        .to(DEVICE)
    )

    ref_storage, ref_views = _single_latent_pool(num_pages)
    worker = types.SimpleNamespace(
        _ctx_block_tables=table,
        _ctx_page_table=None,
        _ctx_page_size=PAGE_SIZE,
        _ctx_kv_buf=ref_views,
        _dflash_attention_backend="CUTEDSL",
    )
    dflash.DFlashWorker._store_context_kv_paged(worker, k, None, rows, positions)

    storage, views = _single_latent_pool(num_pages)
    pages = table[rows.long(), positions.long() // PAGE_SIZE].long()
    offsets = positions.long() % PAGE_SIZE
    keep = torch.ones(num_tokens, dtype=torch.int8, device=DEVICE)
    store_mla_context_latents(k[:, :, 0], views, _pool_ptrs(views), pages, offsets, keep)
    assert ref_storage.abs().sum() > 0
    assert torch.equal(storage, ref_storage)


def _gen_case(num_gens: int, block_size: int, max_ctx: int, seed: int):
    gen = torch.Generator().manual_seed(seed)
    k_plus_1 = block_size + 1
    num_slots = num_gens + 3
    slots = torch.randperm(num_slots - 1, generator=gen)[:num_gens]
    ctx_len = torch.randint(0, max_ctx, (num_slots,), generator=gen)
    ctx_len[: num_slots // 3] = max_ctx - 1
    num_accepted = torch.randint(1, k_plus_1 + 1, (num_gens,), generator=gen).to(torch.int32)
    accepted = torch.randint(0, 50000, (num_gens, k_plus_1 + 2), generator=gen).to(torch.int32)
    return (
        slots.to(DEVICE),
        ctx_len.to(DEVICE),
        num_accepted.to(DEVICE),
        accepted.to(DEVICE)[:, :k_plus_1],
    )


def _prepare(slots, ctx_len, num_accepted, accepted, counts, table, block_size, max_ctx, mask_id):
    num_gens = slots.numel()
    seq_lens = torch.full((num_gens + 2,), 3, dtype=torch.int32, device=DEVICE)
    out = dflash_gen_prepare(
        num_accepted,
        accepted,
        slots,
        ctx_len,
        counts,
        table,
        seq_lens,
        0,
        num_gens,
        ctx_len.numel() - 1,
        max_ctx,
        mask_id,
        PAGE_SIZE,
        block_size,
        block_size + 1,
    )
    return out, seq_lens


@pytest.mark.parametrize("num_gens", [1, 4, 13])
def test_gen_prologue_kernel(num_gens: int) -> None:
    block_size, max_ctx, mask_id = 6, 300, 163839
    slots, ctx_len, num_accepted, accepted = _gen_case(num_gens, block_size, max_ctx, seed=num_gens)
    width = block_size + 1
    num_cols = 12
    table = _block_table(num_gens, num_cols, 200, seed=num_gens)
    counts = torch.full((num_gens,), num_cols, dtype=torch.long, device=DEVICE)

    bonus_idx = (num_accepted - 1).clamp_min(0).long().unsqueeze(1)
    bonus = accepted.gather(1, bonus_idx).squeeze(1).long()
    ctx_len_gen = ctx_len[slots]
    ctx_len_now = (ctx_len_gen + num_accepted.long()).clamp(max=max_ctx)
    expected_query = ctx_len_now.unsqueeze(1) + torch.arange(block_size, device=DEVICE).unsqueeze(0)
    expected_ctx = ctx_len_gen.unsqueeze(1) + torch.arange(width, device=DEVICE).unsqueeze(0)

    out, seq_lens = _prepare(
        slots, ctx_len.clone(), num_accepted, accepted, counts, table, block_size, max_ctx, mask_id
    )
    assert torch.equal(out["bonus"], bonus)
    assert torch.equal(out["embed_ids"][:num_gens], bonus)
    assert int(out["embed_ids"][num_gens]) == mask_id
    assert torch.equal(out["query_positions"], expected_query)
    assert torch.equal(out["ctx_positions"], expected_ctx.reshape(-1))
    assert torch.equal(seq_lens[:num_gens], torch.full_like(seq_lens[:num_gens], width))


@pytest.mark.parametrize("num_gens", [1, 5])
def test_noise_block_kernel(num_gens: int) -> None:
    block_size, hidden, vocab = 6, 1000, 128
    gen = torch.Generator().manual_seed(num_gens)
    embed = nn.Embedding(vocab, hidden).to(torch.bfloat16).to(DEVICE)
    bonus = torch.randint(0, vocab - 1, (num_gens,), generator=gen).to(DEVICE)
    mask_id = vocab - 1
    # The eager path of prepare_1st_drafter_inputs.
    combined = embed(torch.cat([bonus, torch.tensor([mask_id], device=DEVICE)]))
    expected = combined[num_gens].expand(num_gens, block_size, -1).clone()
    expected[:, 0, :] = combined[:num_gens]
    assert torch.equal(dflash_noise_embedding(combined, block_size), expected)


@pytest.mark.parametrize("num_gens", [1, 4, 6])
def test_gen_ctx_store_and_ctx_len_kernels(num_gens: int) -> None:
    """The gen tokens' context store and ctx_len bookkeeping vs. the torch path they replace."""
    block_size, max_ctx, num_cols, num_pages = 6, 260, 9, 80
    width = block_size + 1
    slots, ctx_len, num_accepted, accepted = _gen_case(
        num_gens, block_size, max_ctx, seed=10 + num_gens
    )
    dummy_slot = ctx_len.numel() - 1
    rows = torch.arange(num_gens, device=DEVICE)
    table = _block_table(num_gens, num_cols, num_pages, seed=num_gens)
    # Every row holds all its pages: a clamped column would make two tokens race for one slot.
    counts = torch.full((num_gens,), num_cols, dtype=torch.long, device=DEVICE)
    gen = torch.Generator().manual_seed(2)
    k_new = (
        torch.randn(num_gens * width, NUM_LAYERS, 1, HEAD_DIM, generator=gen)
        .to(torch.bfloat16)
        .to(DEVICE)
    )

    ref_storage, ref_views = _single_latent_pool(num_pages)
    ref_ctx_len = ctx_len.clone()
    offsets = torch.arange(width, device=DEVICE)
    col = ref_ctx_len[slots].unsqueeze(1) + offsets.unsqueeze(0)
    write_mask = offsets.unsqueeze(0) < num_accepted.long().unsqueeze(1)
    capacity = counts[rows] * PAGE_SIZE
    col = torch.minimum(col, (capacity - 1).unsqueeze(1)).clamp_(min=0)
    k_ref = k_new.clone().mul_(write_mask.reshape(-1).view(-1, 1, 1, 1).to(k_new.dtype))
    rows_flat = rows.unsqueeze(1).expand(-1, width).reshape(-1)
    pages = table[rows_flat, col.reshape(-1) // PAGE_SIZE].long()
    for layer in range(NUM_LAYERS):
        ref_views[layer][pages, 0, 0, col.reshape(-1) % PAGE_SIZE] = k_ref[:, layer, 0]
    ref_ctx_len[slots] += num_accepted.long()
    ref_ctx_len.clamp_(max=max_ctx)
    ref_ctx_len[dummy_slot].zero_()
    allocated = (counts * PAGE_SIZE - block_size).clamp_(min=0)
    ref_advertised = torch.minimum(ref_ctx_len[slots], allocated[rows])

    storage, views = _single_latent_pool(num_pages)
    fused_ctx_len = ctx_len.clone()
    out, _ = _prepare(
        slots, fused_ctx_len, num_accepted, accepted, counts, table, block_size, max_ctx, 7
    )
    store_mla_context_latents(
        k_new[:, :, 0],
        views,
        _pool_ptrs(views),
        out["store_pages"],
        out["store_offsets"],
        out["store_mask"],
    )
    assert torch.equal(storage, ref_storage)
    assert torch.equal(fused_ctx_len, ref_ctx_len)
    assert torch.equal(out["num_ctx"], ref_advertised)


@pytest.mark.parametrize("batch", [1, 3, 8])
def test_mla_block_fixup_and_block_write(batch: int, monkeypatch: pytest.MonkeyPatch) -> None:
    block_size, num_cols, num_pages, max_ctx = 6, 7, 96, 10000
    slots, ctx_len, num_accepted, accepted = _gen_case(batch, block_size, 160, seed=30 + batch)
    table = _block_table(batch, num_cols, num_pages, seed=batch)
    counts = torch.full((batch,), num_cols, dtype=torch.long, device=DEVICE)
    out, _ = _prepare(
        slots, ctx_len.clone(), num_accepted, accepted, counts, table, block_size, max_ctx, 7
    )
    expected = modeling_dspark._build_mla_block_fixup(
        out["num_ctx"], table.long(), block_size, PAGE_SIZE
    )
    assert torch.equal(out["blk_pages"], expected.pages)
    assert torch.equal(out["blk_offsets"], expected.offsets)
    assert torch.equal(out["blk_seq_lens"], expected.seq_lens_i32)
    assert torch.equal(out["blk_kv_bounds"], expected.kv_bounds_i32)

    gen = torch.Generator().manual_seed(batch)
    blk_kv = torch.randn(batch, block_size, HEAD_DIM, generator=gen).to(torch.bfloat16).to(DEVICE)
    pools = []
    for fused in ("0", "1"):
        monkeypatch.setenv(SPEC_TAIL_FUSION_ENV_VAR, fused)
        storage, views = _single_latent_pool(num_pages)
        modeling_dspark.MLADSparkForCausalLM._write_block_latents(blk_kv, views[2], expected)
        pools.append(storage)
    assert pools[0].abs().sum() > 0
    assert torch.equal(pools[0], pools[1])


@pytest.mark.parametrize("block_size,num_contexts,num_gens", [(6, 0, 1), (5, 0, 8), (6, 2, 3)])
def test_block_decode_shape_operands_are_reused(block_size, num_contexts, num_gens) -> None:
    """TLLM_DFLASH_FUSED_PROLOGUE: the causal mask and batch rows are built once per shape,
    equal to what every step built, and never cached from inside a graph capture."""
    worker = types.SimpleNamespace()
    build = dflash.DFlashWorker._block_decode_shape_operands
    valid, rows = build(worker, block_size, num_contexts, num_gens)
    idx = torch.arange(block_size, device=DEVICE)
    expected_valid = (idx.view(1, block_size) > idx.view(block_size, 1)).view(
        1, block_size, 1, block_size
    )
    assert torch.equal(valid, expected_valid)
    expected_rows = torch.arange(
        num_contexts, num_contexts + num_gens, dtype=torch.long, device=DEVICE
    )
    assert torch.equal(rows, expected_rows) and rows.dtype == torch.long
    again = build(worker, block_size, num_contexts, num_gens)
    assert again[0] is valid and again[1] is rows

    fresh = types.SimpleNamespace()
    graph = torch.cuda.CUDAGraph()
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        with torch.cuda.graph(graph):
            captured = build(fresh, block_size, num_contexts, num_gens)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured[0], expected_valid) and torch.equal(captured[1], expected_rows)
    assert fresh.__dict__["_block_decode_shape_cache"] == {}


def test_fused_prologue_switch_is_read_per_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TLLM_DFLASH_FUSED_PROLOGUE", raising=False)
    assert not dflash._fused_prologue_on()
    monkeypatch.setenv("TLLM_DFLASH_FUSED_PROLOGUE", "1")
    assert dflash._fused_prologue_on()
    monkeypatch.setenv("TLLM_DFLASH_FUSED_PROLOGUE", "0")
    assert not dflash._fused_prologue_on()
