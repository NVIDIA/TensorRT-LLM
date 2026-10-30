# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kernels shared by sparse attention models that offload KV history.

A decode step's selection names units of a request's history; the KV cache
manager fetches them into per-request arena rows. These kernels mask a
selection to the history the Batch published, and assemble the arena rows the
attention reads from the fetch output and the in-flight unit's write row.

Model inputs follow the requests' input positions, while the native Batch keeps
each request in a fixed row, its slot, for as long as the request is scheduled.
The kernels read input-ordered tensors by position and Batch-owned tensors by
row, through a per-step map from each request's position to its Batch row.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# Message of the opt-in device assertion on a merge's coverage flag.
SPARSE_ARENA_COVERAGE_MESSAGE = (
    "Sparse offload: a live selection of a fetched or in-flight unit has no arena row"
)


def check_sparse_offload_tensor(
    tensor: torch.Tensor, name: str, ndim: int, device: torch.device
) -> None:
    """Require a CUDA int32 tensor of ``ndim`` dimensions on ``device``."""
    if tensor.dtype != torch.int32 or tensor.ndim != ndim:
        raise ValueError(f"{name} must be a {ndim}D int32 tensor")
    if not tensor.is_cuda or tensor.device != device:
        raise ValueError(f"{name} must be on CUDA device {device}")


def check_sparse_arena_merge_inputs(
    topk_indices: torch.Tensor,
    unit_lengths: torch.Tensor,
    history_lengths: torch.Tensor,
    batch_rows: torch.Tensor,
    fetched_indices: torch.Tensor,
    write_indices: torch.Tensor,
    active_request_count: torch.Tensor,
    coverage: torch.Tensor | None,
) -> None:
    """Check the inputs every merge of fetched arena rows into a selection reads.

    Selections are [Q, K] in query order; the fetch output [B, K], write rows,
    history lengths and the Batch row map [B] are by Batch row; unit lengths
    cover the queries and fit the Batch; the active count and the coverage
    flag are [1].
    """
    device = fetched_indices.device
    for tensor, name, ndim in (
        (topk_indices, "topk_indices", 2),
        (unit_lengths, "unit_lengths", 1),
        (history_lengths, "history_lengths", 1),
        (batch_rows, "batch_rows", 1),
        (fetched_indices, "fetched_indices", 2),
        (write_indices, "write_indices", 1),
        (active_request_count, "active_request_count", 1),
    ):
        check_sparse_offload_tensor(tensor, name, ndim, device)
    num_rows, width = fetched_indices.shape
    num_queries, topk = topk_indices.shape
    if topk != width:
        raise ValueError("Selections and the fetch output must share the top-k width")
    if active_request_count.shape != (1,):
        raise ValueError("active_request_count must be [1]")
    if (
        history_lengths.shape != (num_rows,)
        or write_indices.shape != (num_rows,)
        or batch_rows.shape != (num_rows,)
    ):
        raise ValueError(
            "history_lengths, write_indices and batch_rows must cover the native Batch"
        )
    if not num_queries <= unit_lengths.numel() <= num_rows:
        raise ValueError("Query and unit-length rows must fit the native Batch capacity")
    if coverage is not None:
        check_sparse_offload_tensor(coverage, "coverage", 1, device)
        if coverage.shape != (1,):
            raise ValueError("coverage must be [1]")


@triton.jit
def sparse_arena_rows(
    unit,
    in_width,
    live,
    length,
    history_units,
    fetched_row_ptr,
    fetched_stride,
    lane,
    write,
    coverage_ptr,
    CHECK_COVERAGE: tl.constexpr,
):
    """Return the arena row each lane's selected unit is read from.

    ``unit`` holds a query's selected units in its lanes, -1 for none, and
    ``in_width`` the lanes that hold a selection. A unit below the request's
    ``history_units`` takes its fetched row, from ``fetched_row_ptr`` at
    ``lane``; the in-flight unit takes ``write``; padding queries (``live``
    false), invalid selections and units past ``length`` take -1.

    With ``CHECK_COVERAGE``, clears the int32 flag at ``coverage_ptr`` when a
    live, valid selection of a history or in-flight unit gets no row.
    """
    valid = live & (unit >= 0) & (unit < length)
    fetched = tl.load(
        fetched_row_ptr + lane * fetched_stride,
        mask=valid & in_width & (unit < history_units),
        other=-1,
    )
    rows = tl.where(valid & (unit == history_units), write, fetched)
    if CHECK_COVERAGE:
        required = valid & in_width & (unit <= history_units)
        covered = tl.min((~required | (rows >= 0)).to(tl.int32), axis=0)
        tl.atomic_and(coverage_ptr, covered, sem="relaxed")
    return rows


@triton.jit(do_not_specialize=["num_lengths"])
def _select_sparse_history_units_kernel(
    topk_ptr,
    request_indices_ptr,
    batch_rows_ptr,
    active_request_count_ptr,
    unit_lengths_ptr,
    history_lengths_ptr,
    out_ptr,
    topk_stride0,
    topk_stride1,
    request_indices_stride,
    batch_rows_stride,
    unit_lengths_stride,
    history_lengths_stride,
    out_stride0,
    out_stride1,
    NUM_ROWS: tl.constexpr,
    num_lengths,
    TOPK: tl.constexpr,
    UNIT_TOKENS: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    query = tl.program_id(0)
    active_count = tl.load(active_request_count_ptr)
    request = tl.load(request_indices_ptr + query * request_indices_stride)
    active = (query < active_count) & (request >= 0) & (request < active_count)
    active = active & (request < NUM_ROWS) & (request < num_lengths)
    row = tl.load(batch_rows_ptr + request * batch_rows_stride, active, other=-1)
    active = active & (row >= 0) & (row < NUM_ROWS)
    length = tl.load(unit_lengths_ptr + request * unit_lengths_stride, active, other=0)
    history = tl.load(history_lengths_ptr + row * history_lengths_stride, active, other=0)
    k = tl.arange(0, BLOCK_K)
    unit = tl.load(topk_ptr + query * topk_stride0 + k * topk_stride1, k < TOPK, other=-1)
    valid = (unit >= 0) & (unit < length) & (unit < history // UNIT_TOKENS)
    # Each live query owns a distinct Batch row. Keep top-k positions, including
    # holes left by in-flight units, so fetch output has the same slot meaning.
    tl.store(
        out_ptr + row * out_stride0 + k * out_stride1,
        tl.where(valid, unit, -1),
        mask=active & (k < TOPK),
    )


def select_sparse_history_units(
    topk_indices: torch.Tensor,
    request_indices: torch.Tensor,
    batch_rows: torch.Tensor,
    active_request_count: torch.Tensor,
    unit_lengths: torch.Tensor,
    history_lengths: torch.Tensor,
    out: torch.Tensor,
    *,
    unit_tokens: int,
) -> None:
    """Mask non-history selections and scatter them into native Batch rows.

    All tensors are CUDA int32 on one device. ``topk_indices`` is [Q, K] in
    selection units, with distinct nonnegative selections per live row.
    ``request_indices`` [Q] gives each query's request by input position, one
    query per request. ``batch_rows`` [B] gives each request's native Batch row,
    or -1 for a position without a live request; live requests have distinct
    rows, in any order and with free rows between them. ``active_request_count``
    [1] excludes graph padding. ``unit_lengths`` [R] counts each request's
    selectable units by input position, and may cover fewer requests than the
    native Batch capacity B. ``history_lengths`` [B] is the Batch-owned,
    published history watermark in original tokens.

    ``out`` is caller-owned [B, K], disjoint from the inputs. Only ordinals below
    both the unit length and history_lengths // unit_tokens are fetched.
    Top-k order and padding positions are preserved; in-flight units and invalid
    selections become -1. Batch rows without a live query are overwritten with
    -1 on every call, including when a graph's Q is smaller than B. No tensor
    allocation, host read, or synchronization is performed.
    """
    device = out.device
    for tensor, name, ndim in (
        (topk_indices, "topk_indices", 2),
        (request_indices, "request_indices", 1),
        (batch_rows, "batch_rows", 1),
        (active_request_count, "active_request_count", 1),
        (unit_lengths, "unit_lengths", 1),
        (history_lengths, "history_lengths", 1),
        (out, "out", 2),
    ):
        check_sparse_offload_tensor(tensor, name, ndim, device)
    num_rows, width = out.shape
    num_queries, topk = topk_indices.shape
    if unit_tokens <= 0:
        raise ValueError("unit_tokens must be positive")
    if topk != width:
        raise ValueError("top-k width must match the configured sparse selection width")
    if request_indices.shape != (num_queries,) or active_request_count.shape != (1,):
        raise ValueError("request_indices must be [Q] and active_request_count must be [1]")
    if history_lengths.shape != (num_rows,) or batch_rows.shape != (num_rows,):
        raise ValueError("history_lengths and batch_rows must cover the native Batch capacity")
    if not num_queries <= unit_lengths.numel() <= num_rows:
        raise ValueError("Query and unit-length rows must fit the native Batch capacity")

    with torch.cuda.device(device):
        # Clear separately: padded queries can carry stale or invalid request IDs,
        # and a smaller graph may leave native Batch rows without any query.
        out.fill_(-1)
        if num_queries == 0 or topk == 0:
            return
        _select_sparse_history_units_kernel[(num_queries,)](
            topk_indices,
            request_indices,
            batch_rows,
            active_request_count,
            unit_lengths,
            history_lengths,
            out,
            *topk_indices.stride(),
            request_indices.stride(0),
            batch_rows.stride(0),
            unit_lengths.stride(0),
            history_lengths.stride(0),
            *out.stride(),
            NUM_ROWS=num_rows,
            num_lengths=unit_lengths.numel(),
            TOPK=topk,
            UNIT_TOKENS=unit_tokens,
            BLOCK_K=triton.next_power_of_2(topk),
        )


@triton.jit(do_not_specialize=["num_lengths"])
def _merge_sparse_arena_indices_kernel(
    topk_ptr,
    request_indices_ptr,
    batch_rows_ptr,
    active_request_count_ptr,
    unit_lengths_ptr,
    history_lengths_ptr,
    fetched_ptr,
    write_ptr,
    out_ptr,
    topk_stride0,
    topk_stride1,
    request_indices_stride,
    batch_rows_stride,
    unit_lengths_stride,
    history_lengths_stride,
    fetched_stride0,
    fetched_stride1,
    write_stride,
    out_stride0,
    out_stride1,
    coverage_ptr,
    NUM_ROWS: tl.constexpr,
    num_lengths,
    TOPK: tl.constexpr,
    UNIT_TOKENS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    CHECK_COVERAGE: tl.constexpr,
):
    query = tl.program_id(0)
    active_count = tl.load(active_request_count_ptr)
    request = tl.load(request_indices_ptr + query * request_indices_stride)
    live = (query < active_count) & (request >= 0) & (request < active_count)
    live = live & (request < NUM_ROWS) & (request < num_lengths)
    row = tl.load(batch_rows_ptr + request * batch_rows_stride, live, other=-1)
    live = live & (row >= 0) & (row < NUM_ROWS)
    length = tl.load(unit_lengths_ptr + request * unit_lengths_stride, live, other=0)
    history_units = (
        tl.load(history_lengths_ptr + row * history_lengths_stride, live, other=0) // UNIT_TOKENS
    )
    k = tl.arange(0, BLOCK_K)
    in_width = k < TOPK
    unit = tl.load(topk_ptr + query * topk_stride0 + k * topk_stride1, in_width, other=-1)
    write = tl.load(write_ptr + row * write_stride, live, other=-1)
    rows = sparse_arena_rows(
        unit,
        in_width,
        live,
        length,
        history_units,
        fetched_ptr + row * fetched_stride0,
        fetched_stride1,
        k,
        write,
        coverage_ptr,
        CHECK_COVERAGE,
    )
    tl.store(out_ptr + query * out_stride0 + k * out_stride1, rows, mask=in_width)


def merge_sparse_arena_indices(
    topk_indices: torch.Tensor,
    request_indices: torch.Tensor,
    batch_rows: torch.Tensor,
    active_request_count: torch.Tensor,
    unit_lengths: torch.Tensor,
    history_lengths: torch.Tensor,
    fetched_indices: torch.Tensor,
    write_indices: torch.Tensor,
    out: torch.Tensor,
    *,
    unit_tokens: int,
    coverage: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return each query's selection as rows of one sparse buffer's arena.

    Inputs follow :func:`select_sparse_history_units`. ``fetched_indices`` is
    the fetch output of one buffer, [B, K] in Batch row order and top-k column
    order; ``write_indices`` [B] is that buffer's in-flight row by Batch row,
    written this step. Both may be strided views of the per-buffer outputs.

    ``out`` is [Q, K] in query order, assembled by :func:`sparse_arena_rows`.
    No tensor allocation, host read, or synchronization is performed. With
    ``coverage``, an int32 [1] diagnostic flag, the merge also resets the flag,
    clears it when a live selection gets no row, and asserts on it on the
    device. A failed assertion makes the CUDA context unusable, so pass it only
    for debugging.
    """
    device = out.device
    check_sparse_arena_merge_inputs(
        topk_indices,
        unit_lengths,
        history_lengths,
        batch_rows,
        fetched_indices,
        write_indices,
        active_request_count,
        coverage,
    )
    for tensor, name, ndim in ((request_indices, "request_indices", 1), (out, "out", 2)):
        check_sparse_offload_tensor(tensor, name, ndim, device)
    num_rows = fetched_indices.shape[0]
    num_queries, topk = topk_indices.shape
    if unit_tokens <= 0:
        raise ValueError("unit_tokens must be positive")
    if out.shape != (num_queries, topk) or request_indices.shape != (num_queries,):
        raise ValueError("out must be [Q, K] and request_indices [Q]")
    if num_queries == 0 or topk == 0:
        return out
    with torch.cuda.device(device):
        if coverage is not None:
            coverage.fill_(1)
        _merge_sparse_arena_indices_kernel[(num_queries,)](
            topk_indices,
            request_indices,
            batch_rows,
            active_request_count,
            unit_lengths,
            history_lengths,
            fetched_indices,
            write_indices,
            out,
            *topk_indices.stride(),
            request_indices.stride(0),
            batch_rows.stride(0),
            unit_lengths.stride(0),
            history_lengths.stride(0),
            *fetched_indices.stride(),
            write_indices.stride(0),
            *out.stride(),
            coverage if coverage is not None else out,
            NUM_ROWS=num_rows,
            num_lengths=unit_lengths.numel(),
            TOPK=topk,
            UNIT_TOKENS=unit_tokens,
            BLOCK_K=triton.next_power_of_2(topk),
            CHECK_COVERAGE=coverage is not None,
        )
        if coverage is not None:
            torch._assert_async(coverage, SPARSE_ARENA_COVERAGE_MESSAGE)
    return out


@triton.jit(do_not_specialize=["num_rows", "num_out"])
def _gather_sparse_batch_rows_kernel(
    values_ptr,
    batch_rows_ptr,
    out_ptr,
    values_stride,
    batch_rows_stride,
    out_stride,
    num_rows,
    num_out,
    BLOCK: tl.constexpr,
):
    position = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    in_range = position < num_out
    row = tl.load(batch_rows_ptr + position * batch_rows_stride, in_range, other=-1)
    valid = in_range & (row >= 0) & (row < num_rows)
    value = tl.load(values_ptr + row * values_stride, valid, other=-1)
    tl.store(out_ptr + position * out_stride, value, mask=in_range)


def gather_sparse_batch_rows(
    values: torch.Tensor,
    batch_rows: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Read a per-Batch-row value for each request, in input order.

    ``values`` [B] holds one int32 per native Batch row, such as one buffer's
    write rows; ``batch_rows`` [N] gives each request's Batch row, or -1 for a
    position without a live request, which reads -1. Both may be strided. ``out``
    [N] receives the result in place, also possibly strided, and is allocated
    when None. One kernel launch; no host read or synchronization.
    """
    device = values.device
    if out is None:
        out = torch.empty(batch_rows.shape, dtype=torch.int32, device=device)
    for tensor, name in ((values, "values"), (batch_rows, "batch_rows"), (out, "out")):
        check_sparse_offload_tensor(tensor, name, 1, device)
    if out.shape != batch_rows.shape:
        raise ValueError("out must have one entry per request")
    num_out = batch_rows.numel()
    if num_out == 0:
        return out
    block = 256
    with torch.cuda.device(device):
        _gather_sparse_batch_rows_kernel[(triton.cdiv(num_out, block),)](
            values,
            batch_rows,
            out,
            values.stride(0),
            batch_rows.stride(0),
            out.stride(0),
            values.numel(),
            num_out,
            BLOCK=block,
        )
    return out
