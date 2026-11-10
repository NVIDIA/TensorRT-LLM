# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Map M3's per-head block selections and token writes into sparse cache arenas."""

import torch
import triton
import triton.language as tl

from ...offload import SparseOffloadState


@triton.jit
def _write_slots_kernel(
    rows,
    active,
    history,
    writes,
    out,
    WRITE_STRIDE: tl.constexpr,
    SLOT: tl.constexpr,
    PAGE: tl.constexpr,
    SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    query = tl.arange(0, BLOCK)
    row = tl.load(rows + query, query < SIZE, other=-1)
    live = (query < tl.load(active)) & (row >= 0) & (query < SIZE)
    length = tl.load(history + row, live, other=0)
    page = tl.load(writes + row * WRITE_STRIDE + SLOT, live, other=-1)
    tl.store(
        out + query, tl.where(live & (page >= 0), page * PAGE + length % PAGE, -1), query < SIZE
    )


def prepare_write_slots(state: SparseOffloadState, layer_idx: int, out: torch.Tensor) -> None:
    """Convert K arena pages to token slots in input order, masking graph padding."""
    descriptor = state.layers[layer_idx]
    writes = state.write_indices[len(descriptor.buffer_ids)]
    _write_slots_kernel[(1,)](
        state.batch_rows,
        state.active_request_count,
        state.history_lengths,
        writes,
        out,
        writes.stride(0),
        state.out_slots[layer_idx][descriptor.data_buffer_index],
        descriptor.unit_tokens,
        out.numel(),
        triton.next_power_of_2(out.numel()),
    )


@triton.jit
def _selection_kernel(
    topk,
    lengths,
    selected,
    units,
    STRIDE_Q: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_K: tl.constexpr,
    QUERIES: tl.constexpr,
    TOPK: tl.constexpr,
    WIDTH: tl.constexpr,
    PAGE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    query = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    block = tl.load(
        topk + query * STRIDE_Q + (lane // TOPK) * STRIDE_H + (lane % TOPK) * STRIDE_K,
        (query < QUERIES) & (lane < WIDTH),
        other=-1,
    )
    # Keep the first occurrence across heads; all heads still read the original
    # selection through the logical-block table filled after the fetch.
    duplicate = (
        tl.sum(
            ((block[:, None] == block[None, :]) & (lane[:, None] > lane[None, :])).to(tl.int32),
            axis=1,
        )
        > 0
    )
    tl.store(selected + query * WIDTH + lane, tl.where(duplicate, -1, block), lane < WIDTH)
    length = tl.load(lengths + query, query < QUERIES, other=0)
    tl.store(units + query, (length + PAGE - 1) // PAGE)


def prepare_selection(
    topk: torch.Tensor,
    seq_lens: torch.Tensor,
    selected: torch.Tensor,
    units: torch.Tensor,
) -> None:
    """Deduplicate [queries, heads, topk] into fixed-width native request selections."""
    width = topk.shape[1] * topk.shape[2]
    if width != selected.shape[1] or topk.shape[0] > selected.shape[0]:
        raise ValueError("MiniMax-M3 selection does not match the sparse cache capacity")
    _selection_kernel[(selected.shape[0],)](
        topk,
        seq_lens,
        selected,
        units,
        *topk.stride(),
        topk.shape[0],
        topk.shape[2],
        width,
        128,
        triton.next_power_of_2(width),
    )


@triton.jit
def _block_table_kernel(
    selected,
    rows,
    active,
    history,
    fetched,
    writes,
    table,
    WIDTH: tl.constexpr,
    FETCH_STRIDE: tl.constexpr,
    WRITE_STRIDE: tl.constexpr,
    SLOT: tl.constexpr,
    COLUMNS: tl.constexpr,
    PAGE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    query = tl.program_id(0)
    row = tl.load(rows + query)
    live = (query < tl.load(active)) & (row >= 0)
    lane = tl.arange(0, BLOCK)
    unit = tl.load(selected + query * WIDTH + lane, lane < WIDTH, other=-1)
    history_units = tl.load(history + row, live, other=0) // PAGE
    valid = live & (lane < WIDTH) & (unit >= 0) & (unit < COLUMNS) & (unit <= history_units)
    page = tl.load(
        fetched + row * FETCH_STRIDE + SLOT * WIDTH + lane, valid & (unit < history_units), other=-1
    )
    write_page = tl.load(writes + row * WRITE_STRIDE + SLOT, live, other=-1)
    page = tl.where(unit == history_units, write_page, page)
    tl.device_assert(~valid | (page >= 0), "MiniMax-M3 selected block has no sparse arena page")
    tl.store(table + query * COLUMNS + unit, page, valid)


def build_block_table(
    state: SparseOffloadState, layer_idx: int, selected: torch.Tensor, out: torch.Tensor
) -> None:
    """Publish fetched K pages and the growing block by original block ordinal."""
    descriptor = state.layers[layer_idx]
    count = len(descriptor.buffer_ids)
    fetched, writes = state.fetched_indices[count], state.write_indices[count]
    _block_table_kernel[(selected.shape[0],)](
        selected,
        state.batch_rows,
        state.active_request_count,
        state.history_lengths,
        fetched,
        writes,
        out,
        selected.shape[1],
        fetched.stride(0),
        writes.stride(0),
        state.out_slots[layer_idx][descriptor.data_buffer_index],
        out.shape[1],
        descriptor.unit_tokens,
        triton.next_power_of_2(selected.shape[1]),
    )
