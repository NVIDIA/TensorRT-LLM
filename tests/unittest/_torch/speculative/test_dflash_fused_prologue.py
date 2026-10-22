# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TLLM_DFLASH_FUSED_PROLOGUE: each fused DFlash / MLA DSpark kernel against the torch code it replaces.

Every comparison is bitwise. Runs on any CUDA GPU; with ``TRITON_INTERPRET=1``
it also runs on CPU tensors.
"""

import os
import types

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.models import modeling_dspark
from tensorrt_llm._torch.speculative import dflash

if torch.cuda.is_available():
    DEVICE = "cuda"
elif os.environ.get("TRITON_INTERPRET") == "1":
    DEVICE = "cpu"
else:
    DEVICE = None

pytestmark = pytest.mark.skipif(DEVICE is None, reason="needs a CUDA GPU (or TRITON_INTERPRET=1)")

FUSED_ENV = "TLLM_DFLASH_FUSED_PROLOGUE"
PAGE_SIZE = 32
NUM_LAYERS = 5
HEAD_DIM = 576


def _single_latent_pool(num_pages: int, dtype=torch.bfloat16):
    """Per-layer ``[pages, 1, 1, page_size, head_dim]`` views of one allocation, as the draft KV pool hands out."""
    storage = torch.zeros(
        NUM_LAYERS, num_pages, 1, 1, PAGE_SIZE, HEAD_DIM, dtype=dtype, device=DEVICE
    )
    return storage, [storage[layer] for layer in range(NUM_LAYERS)]


def _fake_worker(**attrs):
    worker = types.SimpleNamespace(_ctx_k_buf=None, **attrs)
    worker._single_latent_pool_geometry = types.MethodType(
        dflash.DFlashWorker._single_latent_pool_geometry, worker
    )
    return worker


def _block_table(num_rows: int, num_cols: int, num_pages: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    table = torch.randperm(num_pages, generator=gen)[: num_rows * num_cols].view(num_rows, num_cols)
    return table.to(torch.int32).to(DEVICE)


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
        monkeypatch.setenv(FUSED_ENV, fused)
        worker = _fake_worker(
            _ctx_block_tables=torch.full(
                (max_rows, num_cols), -5, dtype=torch.int32, device=DEVICE
            ),
            _ctx_block_counts=torch.full((max_rows,), -5, dtype=torch.long, device=DEVICE),
            _ctx_pool_idx=1,
            _ctx_block_divisor=divisor,
        )
        assert dflash.DFlashWorker._refresh_ctx_block_tables(worker, attn_metadata, num_seqs)
        results.append((worker._ctx_block_tables, worker._ctx_block_counts))
    assert torch.equal(results[0][0], results[1][0])
    assert torch.equal(results[0][1], results[1][1])


@pytest.mark.parametrize("num_tokens", [1, 8, 37])
def test_store_context_kv_paged_single_latent(
    num_tokens: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    num_rows, num_cols, num_pages = 4, 6, 64
    table = _block_table(num_rows, num_cols, num_pages, seed=num_tokens)
    gen = torch.Generator().manual_seed(num_tokens)
    rows = torch.randint(0, num_rows, (num_tokens,), generator=gen).to(DEVICE)
    # Distinct positions per row, so no two tokens target the same slot.
    positions = torch.randperm(num_cols * PAGE_SIZE, generator=gen)[:num_tokens].to(DEVICE)
    k = (
        torch.randn(num_tokens, NUM_LAYERS, 1, HEAD_DIM, generator=gen)
        .to(torch.bfloat16)
        .to(DEVICE)
    )
    pools = []
    for fused in ("0", "1"):
        monkeypatch.setenv(FUSED_ENV, fused)
        storage, views = _single_latent_pool(num_pages)
        worker = _fake_worker(
            _ctx_block_tables=table,
            _ctx_page_table=None,
            _ctx_page_size=PAGE_SIZE,
            _ctx_kv_buf=views,
        )
        dflash.DFlashWorker._store_context_kv_paged(worker, k, None, rows, positions)
        pools.append(storage)
    assert pools[0].abs().sum() > 0
    assert torch.equal(pools[0], pools[1])


def _gen_case(num_gens: int, block_size: int, max_ctx: int, seed: int):
    gen = torch.Generator().manual_seed(seed)
    k_plus_1 = block_size
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


@pytest.mark.parametrize("num_gens", [1, 4, 13])
def test_gen_prologue_kernel(num_gens: int) -> None:
    block_size, max_ctx, mask_id = 8, 300, 163839
    slots, ctx_len, num_accepted, accepted = _gen_case(num_gens, block_size, max_ctx, seed=num_gens)
    k_plus_1 = block_size

    bonus_idx = (num_accepted - 1).clamp_min(0).long().unsqueeze(1)
    bonus = accepted.gather(1, bonus_idx).squeeze(1).long()
    ctx_len_gen = ctx_len[slots]
    ctx_len_now = (ctx_len_gen + num_accepted.long()).clamp_(max=max_ctx)
    expected_query = ctx_len_now.unsqueeze(1) + torch.arange(block_size, device=DEVICE).unsqueeze(0)
    expected_ctx = ctx_len_gen.unsqueeze(1) + torch.arange(k_plus_1, device=DEVICE).unsqueeze(0)

    embed_ids = torch.empty(num_gens + 1, dtype=torch.long, device=DEVICE)
    query_positions = torch.empty((num_gens, block_size), dtype=torch.long, device=DEVICE)
    ctx_positions = torch.empty((num_gens, k_plus_1), dtype=torch.long, device=DEVICE)
    dflash._gen_prologue_kernel[(num_gens,)](
        num_accepted,
        accepted,
        accepted.stride(0),
        slots,
        ctx_len,
        embed_ids,
        query_positions,
        ctx_positions,
        num_gens,
        max_ctx,
        mask_id,
        BLK=block_size,
        KP1=k_plus_1,
        P2=dflash._p2(max(block_size, k_plus_1)),
    )
    assert torch.equal(embed_ids[:num_gens], bonus)
    assert int(embed_ids[num_gens]) == mask_id
    assert torch.equal(query_positions, expected_query)
    assert torch.equal(ctx_positions, expected_ctx)


@pytest.mark.parametrize("num_gens", [1, 5])
def test_noise_block_kernel(num_gens: int) -> None:
    block_size, hidden, vocab = 8, 1000, 128
    gen = torch.Generator().manual_seed(num_gens)
    embed = nn.Embedding(vocab, hidden).to(torch.bfloat16).to(DEVICE)
    bonus = torch.randint(0, vocab - 1, (num_gens,), generator=gen).to(DEVICE)
    mask_id = vocab - 1
    expected = dflash.dflash_noise_block_embedding(embed, bonus, mask_id, block_size)

    ids = torch.cat([bonus, torch.tensor([mask_id], device=DEVICE)])
    combined = embed(ids)
    actual = torch.empty((num_gens, block_size, hidden), dtype=combined.dtype, device=DEVICE)
    dflash._noise_block_kernel[(num_gens * block_size,)](
        combined, actual, num_gens, hidden, BLK=block_size, BLOCK=1024
    )
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("num_gens", [1, 4, 6])
def test_gen_ctx_store_and_ctx_len_kernels(num_gens: int) -> None:
    """The fused gen-token context store and ctx_len bookkeeping vs. the torch path they replace."""
    block_size, max_ctx, num_cols, num_pages = 8, 260, 9, 80
    k_plus_1 = block_size
    slots, ctx_len, num_accepted, _ = _gen_case(num_gens, block_size, max_ctx, seed=10 + num_gens)
    dummy_slot = ctx_len.numel() - 1
    rows = torch.arange(num_gens, device=DEVICE).flip(0)
    table = _block_table(num_gens, num_cols, num_pages, seed=num_gens)
    # Every row holds all its pages: a clamped column would make two tokens race for one slot.
    counts = torch.full((num_gens,), num_cols, dtype=torch.long, device=DEVICE)
    gen = torch.Generator().manual_seed(2)
    k_new = (
        torch.randn(num_gens * k_plus_1, NUM_LAYERS, 1, HEAD_DIM, generator=gen)
        .to(torch.bfloat16)
        .to(DEVICE)
    )

    ref_storage, ref_views = _single_latent_pool(num_pages)
    ref_ctx_len = ctx_len.clone()
    offsets = torch.arange(k_plus_1, device=DEVICE)
    col = ref_ctx_len[slots].unsqueeze(1) + offsets.unsqueeze(0)
    write_mask = offsets.unsqueeze(0) < num_accepted.long().unsqueeze(1)
    capacity = counts[rows] * PAGE_SIZE
    col = torch.minimum(col, (capacity - 1).unsqueeze(1)).clamp_(min=0)
    k_ref = k_new.clone().mul_(write_mask.reshape(-1).view(-1, 1, 1, 1).to(k_new.dtype))
    rows_flat = rows.unsqueeze(1).expand(-1, k_plus_1).reshape(-1)
    pages = table[rows_flat, col.reshape(-1) // PAGE_SIZE].long()
    for layer in range(NUM_LAYERS):
        ref_views[layer][pages, 0, 0, col.reshape(-1) % PAGE_SIZE] = k_ref[:, layer, 0]
    ref_ctx_len[slots] += num_accepted.long()
    ref_ctx_len.clamp_(max=max_ctx)
    ref_ctx_len[dummy_slot].zero_()
    allocated = dflash.dflash_allocated_ctx_limit(counts, PAGE_SIZE, block_size)
    ref_advertised = torch.minimum(ref_ctx_len[slots], allocated[rows])

    storage, views = _single_latent_pool(num_pages)
    worker = _fake_worker(_ctx_kv_buf=views)
    base, layer_offsets, page_stride, offset_stride, head_dim = (
        worker._single_latent_pool_geometry()
    )
    fused_ctx_len = ctx_len.clone()
    dflash._gen_ctx_store_kernel[(num_gens * k_plus_1, NUM_LAYERS)](
        k_new,
        k_new.stride(0),
        k_new.stride(1),
        num_accepted,
        slots,
        rows,
        fused_ctx_len,
        counts,
        table,
        table.stride(0),
        base,
        layer_offsets,
        page_stride,
        offset_stride,
        PAGE_SIZE,
        max_ctx,
        USE_COUNTS=True,
        KP1=k_plus_1,
        HD=head_dim,
        HDP2=dflash._p2(head_dim),
    )
    advertised = torch.empty(num_gens, dtype=torch.long, device=DEVICE)
    dflash._gen_ctx_len_kernel[(1,)](
        num_accepted,
        slots,
        rows,
        fused_ctx_len,
        counts,
        advertised,
        num_gens,
        fused_ctx_len.numel(),
        dummy_slot,
        max_ctx,
        PAGE_SIZE,
        block_size,
        USE_COUNTS=True,
        P2=dflash._p2(num_gens),
        S2=dflash._p2(fused_ctx_len.numel()),
    )
    assert torch.equal(storage, ref_storage)
    assert torch.equal(fused_ctx_len, ref_ctx_len)
    assert torch.equal(advertised, ref_advertised)


@pytest.mark.parametrize("batch", [1, 3, 8])
def test_mla_block_fixup_and_block_write(batch: int, monkeypatch: pytest.MonkeyPatch) -> None:
    block_size, num_rows, num_cols, num_pages = 8, 10, 7, 96
    gen = torch.Generator().manual_seed(batch)
    table = _block_table(num_rows, num_cols, num_pages, seed=batch)
    slots = torch.randperm(num_rows, generator=gen)[:batch].to(DEVICE)
    ctx_len = torch.randint(0, (num_cols - 1) * PAGE_SIZE - block_size, (batch,), generator=gen).to(
        DEVICE
    )

    expected = modeling_dspark._build_mla_block_fixup(
        ctx_len, table.index_select(0, slots).long(), block_size, PAGE_SIZE
    )
    model = types.SimpleNamespace()
    actual = modeling_dspark.MLADSparkForCausalLM._build_mla_block_fixup_fused(
        model, ctx_len, slots.contiguous(), table, block_size, PAGE_SIZE
    )
    for name in expected._fields:
        assert torch.equal(getattr(actual, name), getattr(expected, name)), name

    blk_kv = torch.randn(batch, block_size, HEAD_DIM, generator=gen).to(torch.bfloat16).to(DEVICE)
    pools = []
    for fused in ("0", "1"):
        monkeypatch.setenv(FUSED_ENV, fused)
        storage, views = _single_latent_pool(num_pages)
        modeling_dspark.MLADSparkForCausalLM._write_block_latents(model, blk_kv, views[2], actual)
        pools.append(storage)
    assert pools[0].abs().sum() > 0
    assert torch.equal(pools[0], pools[1])
