# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401

BAD_PAGE_INDEX = -1

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")


def _make_common_inputs(
    *,
    num_pools: int = 5,
    table_capacity: int = 9,
    num_layers: int = 4,
    num_attn_types: int = 4,
    num_tables: int = 7,
    max_blocks: int = 17,
):
    torch.manual_seed(0)
    device = "cuda"
    block_offsets = torch.randint(
        0,
        4096,
        (num_pools, table_capacity, 2, max_blocks),
        dtype=torch.int32,
        device=device,
    )
    block_offsets.masked_fill_(
        torch.rand(block_offsets.shape, device=device) < 0.17, BAD_PAGE_INDEX
    )
    copy_idx = (
        torch.randperm(table_capacity, device=device)[:num_tables].to(torch.int32).contiguous()
    )
    pool_ids = torch.randint(
        0, num_pools, (num_layers, num_attn_types), dtype=torch.int64, device=device
    )
    valid_pool = torch.ones((num_layers, num_attn_types), dtype=torch.bool, device=device)
    pool_ids[0, -1] = -1
    valid_pool[0, -1] = False
    valid_pool[-1, 0] = False
    scales = torch.randint(1, 8, (num_layers, num_attn_types), dtype=torch.int32, device=device)
    layer_offsets = torch.randint(
        0, 32, (num_layers, num_attn_types), dtype=torch.int32, device=device
    )
    return block_offsets, copy_idx, pool_ids, valid_pool, scales, layer_offsets


def _reference_sliding_block_tables(
    block_offsets: torch.Tensor,
    copy_idx: torch.Tensor,
    pool_ids: torch.Tensor,
    valid_pool: torch.Tensor,
    scales: torch.Tensor,
    layer_offsets: torch.Tensor,
) -> torch.Tensor:
    base = block_offsets[pool_ids[:, :, None], copy_idx[None, None, :], 0, :]
    scaled_base = torch.where(
        (base == BAD_PAGE_INDEX) | ~(valid_pool[:, :, None, None]),
        BAD_PAGE_INDEX,
        base * scales[:, :, None, None] + layer_offsets[:, :, None, None],
    )
    output = torch.empty_like(scaled_base)
    output.copy_(scaled_base)
    return output


def _reference_sliding_block_tables_with_scratch(
    block_offsets: torch.Tensor,
    copy_idx: torch.Tensor,
    pool_ids: torch.Tensor,
    valid_pool: torch.Tensor,
    scales: torch.Tensor,
    layer_offsets: torch.Tensor,
    scratch_pages: torch.Tensor,
    scratch_begs: torch.Tensor,
    scratch_ends: torch.Tensor,
    scratch_slots: torch.Tensor,
    num_contexts: torch.Tensor,
) -> torch.Tensor:
    base = block_offsets[pool_ids[:, :, None], copy_idx[None, None, :], 0, :]
    scaled_base = torch.where(
        (base == BAD_PAGE_INDEX) | ~(valid_pool[:, :, None, None]),
        BAD_PAGE_INDEX,
        base * scales[:, :, None, None] + layer_offsets[:, :, None, None],
    )
    output = torch.empty_like(scaled_base)
    output.copy_(scaled_base)

    block_positions = torch.arange(
        block_offsets.shape[-1], dtype=torch.int32, device=block_offsets.device
    )
    context_positions = torch.arange(
        scratch_begs.shape[1],
        dtype=torch.int32,
        device=scratch_begs.device,
    )
    active_context = context_positions < num_contexts
    mask = (
        (block_positions >= scratch_begs[:, :, None])
        & (block_positions < scratch_ends[:, :, None])
        & active_context[None, :, None]
    )
    range_index = torch.where(mask, block_positions - scratch_begs[:, :, None], 0)
    total_offset = range_index[pool_ids] * scratch_pages[:, :, None, None]
    slot_idx = (total_offset // scales[:, :, None, None]).clamp(max=scratch_slots.shape[-1] - 1)
    slot_id = scratch_slots[pool_ids].gather(-1, slot_idx.long())
    offset = total_offset % scales[:, :, None, None]
    scratch_index = (
        slot_id * scales[:, :, None, None]
        + (offset + layer_offsets[:, :, None, None]) % scales[:, :, None, None]
    )
    scratch_capacity = scratch_begs.shape[1]
    scratch_rows = output[:, :, :scratch_capacity, :]
    mask = mask[pool_ids] & valid_pool[:, :, None, None]
    output[:, :, :scratch_capacity, :].copy_(torch.where(mask, scratch_index, scratch_rows))
    return output


@pytest.mark.parametrize("max_blocks", [17, 256])
def test_deepseek_v4_compute_sliding_block_tables_matches_reference(max_blocks):
    block_offsets, copy_idx, pool_ids, valid_pool, scales, layer_offsets = _make_common_inputs(
        max_blocks=max_blocks
    )
    expected = _reference_sliding_block_tables(
        block_offsets,
        copy_idx,
        pool_ids,
        valid_pool,
        scales,
        layer_offsets,
    )
    output = torch.empty_like(expected)

    torch.ops.trtllm.deepseek_v4_compute_sliding_block_tables(
        block_offsets,
        copy_idx,
        pool_ids,
        valid_pool,
        scales,
        layer_offsets,
        output,
    )

    assert torch.equal(output, expected)


@pytest.mark.parametrize("max_blocks", [17, 256])
def test_deepseek_v4_compute_sliding_block_tables_with_scratch_matches_reference(max_blocks):
    block_offsets, copy_idx, pool_ids, valid_pool, scales, layer_offsets = _make_common_inputs(
        max_blocks=max_blocks
    )
    num_pools = block_offsets.shape[0]
    scratch_capacity = copy_idx.shape[0]
    max_scratch_slots = 5
    device = block_offsets.device
    torch.manual_seed(1)
    scratch_begs = torch.randint(
        0,
        block_offsets.shape[-1] // 2,
        (num_pools, scratch_capacity),
        dtype=torch.int32,
        device=device,
    )
    scratch_widths = torch.randint(
        1,
        max(2, block_offsets.shape[-1] // 3),
        (num_pools, scratch_capacity),
        dtype=torch.int32,
        device=device,
    )
    scratch_ends = torch.minimum(
        scratch_begs + scratch_widths,
        torch.tensor(block_offsets.shape[-1], dtype=torch.int32, device=device),
    )
    scratch_slots = torch.randint(
        0,
        4096,
        (num_pools, scratch_capacity, max_scratch_slots),
        dtype=torch.int32,
        device=device,
    )
    scratch_pages = torch.randint(1, 8, scales.shape, dtype=torch.int32, device=device)
    num_contexts = torch.tensor(scratch_capacity - 1, dtype=torch.int32, device=device)

    expected = _reference_sliding_block_tables_with_scratch(
        block_offsets,
        copy_idx,
        pool_ids,
        valid_pool,
        scales,
        layer_offsets,
        scratch_pages,
        scratch_begs,
        scratch_ends,
        scratch_slots,
        num_contexts,
    )
    output = torch.empty_like(expected)

    torch.ops.trtllm.deepseek_v4_compute_sliding_block_tables_with_scratch(
        block_offsets,
        copy_idx,
        pool_ids,
        valid_pool,
        scales,
        layer_offsets,
        scratch_pages,
        scratch_begs,
        scratch_ends,
        scratch_slots,
        num_contexts,
        output,
    )

    assert torch.equal(output, expected)
