# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Paged copies between mixed KV pools and the native Attention/codec buffers."""

from typing import Literal

import torch
import triton
import triton.language as tl


@triton.jit
def _page(
    PageTables,
    request,
    position,
    valid,
    role: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    TABLE_SHAPE: tl.constexpr,
    TABLE_STRIDES: tl.constexpr,
):
    logical_page = position // PAGE_SIZE
    valid = valid & (position >= 0) & (logical_page < TABLE_SHAPE[2])
    return tl.load(
        PageTables
        + role * TABLE_STRIDES[0]
        + request * TABLE_STRIDES[1]
        + logical_page.to(tl.int64) * TABLE_STRIDES[2],
        mask=valid,
        other=-1,
    ).to(tl.int64)


@triton.jit
def _offset(page, position, kv, head, column, STRIDES: tl.constexpr):
    return (
        page.to(tl.int64) * STRIDES[0]
        + kv.to(tl.int64) * STRIDES[1]
        + position.to(tl.int64) * STRIDES[2]
        + head.to(tl.int64) * STRIDES[3]
        + column.to(tl.int64) * STRIDES[4]
    )


@triton.jit
def _gather_history_kernel(
    History,
    Scales,
    PageTables,
    Lengths,
    ReadLow,
    ReadScales,
    HISTORY_SHAPE: tl.constexpr,
    HISTORY_STRIDES: tl.constexpr,
    SCALE_SHAPE: tl.constexpr,
    SCALE_STRIDES: tl.constexpr,
    TABLE_SHAPE: tl.constexpr,
    TABLE_STRIDES: tl.constexpr,
    LENGTH_STRIDES: tl.constexpr,
    CAPACITY: tl.constexpr,
    ROWS: tl.constexpr,
    COLUMNS: tl.constexpr,
):
    request = tl.program_id(1).to(tl.int64)
    rows = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
    columns = tl.arange(0, COLUMNS)
    rows_per_token: tl.constexpr = HISTORY_SHAPE[1] * HISTORY_SHAPE[3]
    position = rows // rows_per_token
    kv = rows // HISTORY_SHAPE[3] % HISTORY_SHAPE[1]
    head = rows % HISTORY_SHAPE[3]
    cached = tl.load(Lengths + request * LENGTH_STRIDES[1])
    valid = (position < CAPACITY) & (position < cached)
    high_page = _page(
        PageTables, request, position, valid, 2, HISTORY_SHAPE[2], TABLE_SHAPE, TABLE_STRIDES
    )
    low_page = _page(
        PageTables, request, position, valid, 0, HISTORY_SHAPE[2], TABLE_SHAPE, TABLE_STRIDES
    )
    scale_page = _page(
        PageTables, request, position, valid, 1, HISTORY_SHAPE[2], TABLE_SHAPE, TABLE_STRIDES
    )
    low_valid = valid & (high_page < 0) & (low_page >= 0) & (low_page < HISTORY_SHAPE[0])
    scale_valid = valid & (high_page < 0) & (scale_page >= 0) & (scale_page < SCALE_SHAPE[0])
    within_page = position % HISTORY_SHAPE[2]
    low_offset = _offset(
        low_page[:, None],
        within_page[:, None],
        kv[:, None],
        head[:, None],
        columns[None, :],
        HISTORY_STRIDES,
    )
    scale_offset = _offset(
        scale_page[:, None],
        within_page[:, None],
        kv[:, None],
        head[:, None],
        columns[None, :],
        SCALE_STRIDES,
    )
    low = tl.load(
        History.to(tl.pointer_type(tl.uint8)) + low_offset,
        mask=low_valid[:, None] & (columns[None, :] < HISTORY_SHAPE[4]),
        other=0,
    )
    scales = tl.load(
        Scales.to(tl.pointer_type(tl.uint8)) + scale_offset,
        mask=scale_valid[:, None] & (columns[None, :] < SCALE_SHAPE[4]),
        other=0,
    )
    output_row = request * CAPACITY * rows_per_token + rows
    tl.store(
        ReadLow + output_row[:, None] * HISTORY_SHAPE[4] + columns[None, :],
        low,
        mask=(position[:, None] < CAPACITY) & (columns[None, :] < HISTORY_SHAPE[4]),
    )
    tl.store(
        ReadScales.to(tl.pointer_type(tl.uint8))
        + output_row[:, None] * SCALE_SHAPE[4]
        + columns[None, :],
        scales,
        mask=(position[:, None] < CAPACITY) & (columns[None, :] < SCALE_SHAPE[4]),
    )


@triton.jit
def _restore_view_kernel(
    High,
    Pool,
    PageTables,
    Lengths,
    RestoredLow,
    HIGH_SHAPE: tl.constexpr,
    HIGH_STRIDES: tl.constexpr,
    POOL_SHAPE: tl.constexpr,
    POOL_STRIDES: tl.constexpr,
    TABLE_SHAPE: tl.constexpr,
    TABLE_STRIDES: tl.constexpr,
    LENGTH_STRIDES: tl.constexpr,
    HISTORY_CAPACITY: tl.constexpr,
    PAGES_PER_REQUEST: tl.constexpr,
    STORAGE_BYTES: tl.constexpr,
    FP8: tl.constexpr,
    ROWS: tl.constexpr,
    COLUMNS: tl.constexpr,
):
    request = tl.program_id(1).to(tl.int64)
    rows = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
    columns = tl.arange(0, COLUMNS)
    rows_per_token: tl.constexpr = HIGH_SHAPE[1] * HIGH_SHAPE[3]
    position = rows // rows_per_token
    kv = rows // HIGH_SHAPE[3] % HIGH_SHAPE[1]
    head = rows % HIGH_SHAPE[3]
    capacity: tl.constexpr = PAGES_PER_REQUEST * POOL_SHAPE[2]
    valid = position < capacity
    cached = tl.load(Lengths + request * LENGTH_STRIDES[1])
    high_page = _page(
        PageTables,
        request,
        position,
        valid & (position < cached),
        2,
        HIGH_SHAPE[2],
        TABLE_SHAPE,
        TABLE_STRIDES,
    )
    high_valid = valid & (position < cached) & (high_page >= 0) & (high_page < HIGH_SHAPE[0])
    pool_page = request * PAGES_PER_REQUEST + position // POOL_SHAPE[2]
    pool_offset = _offset(
        pool_page[:, None],
        (position % POOL_SHAPE[2])[:, None],
        kv[:, None],
        head[:, None],
        columns[None, :],
        POOL_STRIDES,
    )
    high_offset = _offset(
        high_page[:, None],
        (position % HIGH_SHAPE[2])[:, None],
        kv[:, None],
        head[:, None],
        columns[None, :],
        HIGH_STRIDES,
    )
    if STORAGE_BYTES == 1:
        bits_type: tl.constexpr = tl.uint8
    else:
        bits_type: tl.constexpr = tl.uint16
    high_bits = tl.load(
        High.to(tl.pointer_type(bits_type)) + high_offset,
        mask=high_valid[:, None] & (columns[None, :] < HIGH_SHAPE[4]),
        other=0,
    )
    tl.store(
        Pool.to(tl.pointer_type(bits_type)) + pool_offset,
        high_bits,
        mask=high_valid[:, None] & (columns[None, :] < HIGH_SHAPE[4]),
    )
    low_valid = valid & (position < cached) & (position < HISTORY_CAPACITY) & (high_page < 0)
    low_row = request * HISTORY_CAPACITY * rows_per_token + rows
    low = tl.load(
        RestoredLow + low_row[:, None] * HIGH_SHAPE[4] + columns[None, :],
        mask=low_valid[:, None] & (columns[None, :] < HIGH_SHAPE[4]),
        other=0,
    )
    if FP8:
        low = tl.minimum(tl.maximum(low.to(tl.float32), -448.0), 448.0)
    tl.store(
        Pool + pool_offset,
        low.to(Pool.dtype.element_ty),
        mask=(valid & ~high_valid)[:, None] & (columns[None, :] < HIGH_SHAPE[4]),
    )


@triton.jit
def _gather_new_kernel(
    Pool,
    High,
    PageTables,
    Lengths,
    CuNewTokens,
    NewLowInput,
    POOL_SHAPE: tl.constexpr,
    POOL_STRIDES: tl.constexpr,
    HIGH_SHAPE: tl.constexpr,
    HIGH_STRIDES: tl.constexpr,
    TABLE_SHAPE: tl.constexpr,
    TABLE_STRIDES: tl.constexpr,
    LENGTH_STRIDES: tl.constexpr,
    CU_STRIDE: tl.constexpr,
    PAGES_PER_REQUEST: tl.constexpr,
    CAPACITY: tl.constexpr,
    OUTPUT_ROWS: tl.constexpr,
    STORAGE_BYTES: tl.constexpr,
    ROWS: tl.constexpr,
    COLUMNS: tl.constexpr,
):
    request = tl.program_id(1).to(tl.int64)
    rows = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
    columns = tl.arange(0, COLUMNS)
    rows_per_token: tl.constexpr = POOL_SHAPE[1] * POOL_SHAPE[3]
    token = rows // rows_per_token
    kv = rows // POOL_SHAPE[3] % POOL_SHAPE[1]
    head = rows % POOL_SHAPE[3]
    cached = tl.load(Lengths + request * LENGTH_STRIDES[1]).to(tl.int64)
    total = tl.load(Lengths + LENGTH_STRIDES[0] + request * LENGTH_STRIDES[1]).to(tl.int64)
    position = cached + token
    valid = (
        (token < CAPACITY)
        & (position >= 0)
        & (position < total)
        & (position < PAGES_PER_REQUEST * POOL_SHAPE[2])
    )
    start = tl.load(CuNewTokens + request * CU_STRIDE).to(tl.int64)
    end = tl.load(CuNewTokens + (request + 1) * CU_STRIDE).to(tl.int64)
    output_row = start * rows_per_token + rows
    valid = valid & (start >= 0) & (start + token < end) & (output_row < OUTPUT_ROWS)
    pool_page = request * PAGES_PER_REQUEST + position // POOL_SHAPE[2]
    pool_offset = _offset(
        pool_page[:, None],
        (position % POOL_SHAPE[2])[:, None],
        kv[:, None],
        head[:, None],
        columns[None, :],
        POOL_STRIDES,
    )
    mask = valid[:, None] & (columns[None, :] < POOL_SHAPE[4])
    if STORAGE_BYTES == 1:
        bits_type: tl.constexpr = tl.uint8
    else:
        bits_type: tl.constexpr = tl.uint16
    bits = tl.load(Pool.to(tl.pointer_type(bits_type)) + pool_offset, mask=mask, other=0)
    values = bits.to(Pool.dtype.element_ty, bitcast=True)
    tl.store(
        NewLowInput + output_row[:, None] * POOL_SHAPE[4] + columns[None, :],
        values.to(NewLowInput.dtype.element_ty),
        mask=mask,
    )
    high_page = _page(
        PageTables, request, position, valid, 2, HIGH_SHAPE[2], TABLE_SHAPE, TABLE_STRIDES
    )
    high_valid = valid & (high_page >= 0) & (high_page < HIGH_SHAPE[0])
    high_offset = _offset(
        high_page[:, None],
        (position % HIGH_SHAPE[2])[:, None],
        kv[:, None],
        head[:, None],
        columns[None, :],
        HIGH_STRIDES,
    )
    tl.store(
        High.to(tl.pointer_type(bits_type)) + high_offset,
        bits,
        mask=high_valid[:, None] & (columns[None, :] < POOL_SHAPE[4]),
    )


@triton.jit
def _scatter_new_kernel(
    History,
    Scales,
    PageTables,
    Lengths,
    CuNewTokens,
    NewLow,
    NewScales,
    HISTORY_SHAPE: tl.constexpr,
    HISTORY_STRIDES: tl.constexpr,
    SCALE_SHAPE: tl.constexpr,
    SCALE_STRIDES: tl.constexpr,
    TABLE_SHAPE: tl.constexpr,
    TABLE_STRIDES: tl.constexpr,
    LENGTH_STRIDES: tl.constexpr,
    CU_STRIDE: tl.constexpr,
    CAPACITY: tl.constexpr,
    INPUT_ROWS: tl.constexpr,
    ROWS: tl.constexpr,
    COLUMNS: tl.constexpr,
):
    request = tl.program_id(1).to(tl.int64)
    rows = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
    columns = tl.arange(0, COLUMNS)
    rows_per_token: tl.constexpr = HISTORY_SHAPE[1] * HISTORY_SHAPE[3]
    token = rows // rows_per_token
    kv = rows // HISTORY_SHAPE[3] % HISTORY_SHAPE[1]
    head = rows % HISTORY_SHAPE[3]
    cached = tl.load(Lengths + request * LENGTH_STRIDES[1]).to(tl.int64)
    total = tl.load(Lengths + LENGTH_STRIDES[0] + request * LENGTH_STRIDES[1]).to(tl.int64)
    position = cached + token
    start = tl.load(CuNewTokens + request * CU_STRIDE).to(tl.int64)
    end = tl.load(CuNewTokens + (request + 1) * CU_STRIDE).to(tl.int64)
    input_row = start * rows_per_token + rows
    valid = (
        (token < CAPACITY)
        & (position >= 0)
        & (position < total)
        & (start >= 0)
        & (start + token < end)
        & (input_row < INPUT_ROWS)
    )
    low_page = _page(
        PageTables, request, position, valid, 0, HISTORY_SHAPE[2], TABLE_SHAPE, TABLE_STRIDES
    )
    scale_page = _page(
        PageTables, request, position, valid, 1, HISTORY_SHAPE[2], TABLE_SHAPE, TABLE_STRIDES
    )
    low_valid = valid & (low_page >= 0) & (low_page < HISTORY_SHAPE[0])
    scale_valid = valid & (scale_page >= 0) & (scale_page < SCALE_SHAPE[0])
    low = tl.load(
        NewLow + input_row[:, None] * HISTORY_SHAPE[4] + columns[None, :],
        mask=low_valid[:, None] & (columns[None, :] < HISTORY_SHAPE[4]),
        other=0,
    )
    scales = tl.load(
        NewScales.to(tl.pointer_type(tl.uint8))
        + input_row[:, None] * SCALE_SHAPE[4]
        + columns[None, :],
        mask=scale_valid[:, None] & (columns[None, :] < SCALE_SHAPE[4]),
        other=0,
    )
    within_page = position % HISTORY_SHAPE[2]
    low_offset = _offset(
        low_page[:, None],
        within_page[:, None],
        kv[:, None],
        head[:, None],
        columns[None, :],
        HISTORY_STRIDES,
    )
    scale_offset = _offset(
        scale_page[:, None],
        within_page[:, None],
        kv[:, None],
        head[:, None],
        columns[None, :],
        SCALE_STRIDES,
    )
    tl.store(
        History.to(tl.pointer_type(tl.uint8)) + low_offset,
        low,
        mask=low_valid[:, None] & (columns[None, :] < HISTORY_SHAPE[4]),
    )
    tl.store(
        Scales.to(tl.pointer_type(tl.uint8)) + scale_offset,
        scales,
        mask=scale_valid[:, None] & (columns[None, :] < SCALE_SHAPE[4]),
    )


def _geometry(pool: torch.Tensor, layout: Literal["HND", "NHD"]) -> tuple[tuple, tuple]:
    """Return shape and strides in page/KV/token/head/channel order."""
    axes = (0, 1, 3, 2, 4) if layout == "HND" else (0, 1, 2, 3, 4)
    return tuple(pool.shape[axis] for axis in axes), tuple(pool.stride(axis) for axis in axes)


def gather_history(
    history: torch.Tensor,
    scales: torch.Tensor,
    page_tables: torch.Tensor,
    lengths: torch.Tensor,
    read_low: torch.Tensor,
    read_scales: torch.Tensor,
    history_capacity: int,
    layout: Literal["HND", "NHD"],
) -> None:
    """Gather low-only history into contiguous codec rows; zero all other rows."""
    if not read_low.numel():
        return
    shape, strides = _geometry(history, layout)
    scale_shape, scale_strides = _geometry(scales, layout)
    grid = (triton.cdiv(history_capacity * shape[1] * shape[3], 4), lengths.shape[1])
    _gather_history_kernel[grid](
        history,
        scales,
        page_tables,
        lengths,
        read_low,
        read_scales,
        shape,
        strides,
        scale_shape,
        scale_strides,
        tuple(page_tables.shape),
        page_tables.stride(),
        lengths.stride(),
        history_capacity,
        4,
        triton.next_power_of_2(shape[4]),
        num_warps=4,
    )


def restore_view(
    high: torch.Tensor,
    pool: torch.Tensor,
    page_tables: torch.Tensor,
    lengths: torch.Tensor,
    restored_low: torch.Tensor,
    history_capacity: int,
    layout: Literal["HND", "NHD"],
) -> None:
    """Restore cached KV and zero remaining native staging rows, preserving high bytes."""
    if not pool.numel():
        return
    high_shape, high_strides = _geometry(high, layout)
    pool_shape, pool_strides = _geometry(pool, layout)
    batch_size = lengths.shape[1]
    pages_per_request = pool_shape[0] // batch_size
    capacity = pages_per_request * pool_shape[2]
    grid = (triton.cdiv(capacity * pool_shape[1] * pool_shape[3], 4), batch_size)
    _restore_view_kernel[grid](
        high,
        pool,
        page_tables,
        lengths,
        restored_low,
        high_shape,
        high_strides,
        pool_shape,
        pool_strides,
        tuple(page_tables.shape),
        page_tables.stride(),
        lengths.stride(),
        history_capacity,
        pages_per_request,
        pool.element_size(),
        pool.dtype == torch.float8_e4m3fn,
        4,
        triton.next_power_of_2(pool_shape[4]),
        num_warps=4,
    )


def gather_new(
    pool: torch.Tensor,
    high: torch.Tensor,
    page_tables: torch.Tensor,
    lengths: torch.Tensor,
    cu_new_tokens: torch.Tensor,
    new_low_input: torch.Tensor,
    max_new_tokens_per_request: int,
    layout: Literal["HND", "NHD"],
) -> None:
    """Gather new KV for the codec and copy its original bytes into live high pages."""
    if not new_low_input.numel():
        return
    pool_shape, pool_strides = _geometry(pool, layout)
    high_shape, high_strides = _geometry(high, layout)
    batch_size = lengths.shape[1]
    grid = (triton.cdiv(max_new_tokens_per_request * pool_shape[1] * pool_shape[3], 4), batch_size)
    _gather_new_kernel[grid](
        pool,
        high,
        page_tables,
        lengths,
        cu_new_tokens,
        new_low_input,
        pool_shape,
        pool_strides,
        high_shape,
        high_strides,
        tuple(page_tables.shape),
        page_tables.stride(),
        lengths.stride(),
        cu_new_tokens.stride(0),
        pool_shape[0] // batch_size,
        max_new_tokens_per_request,
        new_low_input.shape[0],
        pool.element_size(),
        4,
        triton.next_power_of_2(pool_shape[4]),
        num_warps=4,
    )


def scatter_new(
    history: torch.Tensor,
    scales: torch.Tensor,
    page_tables: torch.Tensor,
    lengths: torch.Tensor,
    cu_new_tokens: torch.Tensor,
    new_low: torch.Tensor,
    new_scales: torch.Tensor,
    max_new_tokens_per_request: int,
    layout: Literal["HND", "NHD"],
) -> None:
    """Publish codec bytes for new tokens without touching shared cached history."""
    if not new_low.numel():
        return
    shape, strides = _geometry(history, layout)
    scale_shape, scale_strides = _geometry(scales, layout)
    grid = (triton.cdiv(max_new_tokens_per_request * shape[1] * shape[3], 4), lengths.shape[1])
    _scatter_new_kernel[grid](
        history,
        scales,
        page_tables,
        lengths,
        cu_new_tokens,
        new_low,
        new_scales,
        shape,
        strides,
        scale_shape,
        scale_strides,
        tuple(page_tables.shape),
        page_tables.stride(),
        lengths.stride(),
        cu_new_tokens.stride(0),
        max_new_tokens_per_request,
        new_low.shape[0],
        4,
        triton.next_power_of_2(shape[4]),
        num_warps=4,
    )
