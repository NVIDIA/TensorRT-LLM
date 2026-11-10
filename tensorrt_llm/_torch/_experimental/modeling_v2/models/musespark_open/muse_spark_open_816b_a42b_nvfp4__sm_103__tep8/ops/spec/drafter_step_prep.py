# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A paged drafter context's decode-step bookkeeping in four launches (``MUSESPARK_OPEN_DRAFTER_FUSED_PREP``).

On a decode step the ``context_kv_fork`` side chain decodes the step's draft page table
(``DFlashWorker._refresh_ctx_block_tables``), stores the ``K + 1`` projected target rows of every
request as the drafter's context K/V (``DFlashWorker._store_step_context``: columns and positions,
``precompute_context_kv``, then one flashinfer paged append per drafter layer, each behind two
``.contiguous()`` copies), and its commit advances the context lengths and reads the lengths the
drafter attends over (``DFlashWorker._commit_context``). Over the KV cache manager's pages (the
served TEP8 gen: the drafter shares the target's manager) that is about fifty torch launches of a
few CTAs each between the target's LM head and the drafter's first layer.

This module runs the integer and copy parts as Triton launches, bit for bit the built-in's bytes:

* ``refresh_block_tables``: a fill of the page counts, then one launch over
  ``(requests, column tiles)`` that decodes the table and takes each row's allocation extent
  (last valid entry + 1) with an atomic max;
* ``store_step_context``: the RoPE positions ``ctx_len[slot] + r`` and the clamped page columns in
  one launch; ``precompute_context_kv`` (the drafter's fused K/V GEMM, norm and RoPE) unchanged; then
  every layer's K and V rows into ``pool_l[table[row, col // page], 0 / 1, h, col % page, :]`` in one
  launch (flashinfer's HND paged append, element for element);
* ``commit_context``: NVIDIA/TensorRT-LLM #21535's ``context_kv_commit_paged`` (``aaba1deabc`` on
  ``feat/musespark-open-agentx``, ``ops/spec/context_kv_commit_paged.py``, verbatim): the length
  advance, the clamp, the dummy slot's reset and the advertised lengths with the allocation check.

Integer arithmetic and copies only; nothing is rounded. Ported from a partner team's GB300 overlay
(its drafter fast path, extended down to one request) and re-implemented on this tree. Default off; any
step or layout the fused path does not take keeps the built-in methods.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import triton
import triton.language as tl

from tensorrt_llm.logger import logger

from .context_kv_commit_paged import context_kv_commit_paged

#: The switch: ``1`` runs the fused bookkeeping on the ``context_kv_fork`` side chain of decode steps
#: of ``MIN_REQUESTS_ENV`` .. ``MAX_REQUESTS`` requests; unset (or anything else) leaves the built-in.
ENV = "MUSESPARK_OPEN_DRAFTER_FUSED_PREP"
#: The fewest generation requests of a step the fused bookkeeping takes (default 1).
MIN_REQUESTS_ENV = "MUSESPARK_OPEN_DRAFTER_FUSED_PREP_MIN_REQUESTS"
DEFAULT_MIN_REQUESTS = 1
#: Generation requests one step may have (``context_kv_commit.MAX_REQUESTS``).
MAX_REQUESTS = 64

#: Drafter layers the store's layer offsets are passed for (scalar arguments, no device table).
MAX_LAYERS = 8
#: Columns of the page table one refresh program decodes.
REFRESH_BLOCK = 1024
#: Stored tokens one positions program writes.
POSITIONS_BLOCK = 256

#: The worker methods the fused path stands in for, as ``DFlashWorker`` defines them; a worker class
#: overriding one keeps the built-in path (``_refresh_ctx_block_tables`` is called as
#: ``DFlashWorker``'s by the fork, which overrides it only to skip a second decode).
REPLACED_METHODS = (
    "_store_step_context",
    "_store_context_kv_paged",
    "_commit_context",
    "_advance_ctx_len",
    "_empty_dummy_slot",
    "_settle_context",
    "_drafter_context_lengths",
    "_check_block_within_allocation",
)


# --------------------------------------------------------------------------
# Kernels
# --------------------------------------------------------------------------
@triton.jit
def _refresh_kernel(src_ptr, src_row, src_col, table_ptr, table_row, counts_ptr, width, offset, divisor,
                    BLOCK: tl.constexpr):
    """``encoded = src - offset``; ``counts[r] = max((encoded >= 0) * (c + 1))``;
    ``table[r, c] = int32(max(encoded, 0) // divisor)``. ``counts[r]`` must be 0 on entry."""
    r = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    inside = c < width
    encoded = tl.load(src_ptr + r * src_row + c * src_col, mask=inside, other=0).to(tl.int64)
    encoded = encoded - offset
    valid = inside & (encoded >= 0)
    extent = tl.max(tl.where(valid, c + 1, 0), axis=0)
    tl.atomic_max(counts_ptr + r, extent)
    decoded = tl.maximum(encoded, 0) // divisor
    tl.store(table_ptr + r * table_row + c, decoded.to(tl.int32), mask=inside)


@triton.jit
def _columns_kernel(ctx_len_ptr, base_ptr, slots_ptr, rows_ptr, counts_ptr, pos_ptr, col_ptr, tokens, page_size,
                    PER_REQUEST: tl.constexpr, HAS_BASE: tl.constexpr, BLOCK: tl.constexpr):
    """Token ``t`` (request ``i = t // PER_REQUEST``, row ``r``): ``pos[t] = ctx_len[slots[i]] + r`` and
    ``col[t] = max(min(ctx_len[slots[i]] - base[slots[i]] + r, counts[rows[i]] * page - 1), 0)``."""
    t = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    inside = t < tokens
    request = t // PER_REQUEST
    r = (t % PER_REQUEST).to(tl.int64)
    slot = tl.load(slots_ptr + request, mask=inside, other=0)
    row = tl.load(rows_ptr + request, mask=inside, other=0)
    length = tl.load(ctx_len_ptr + slot, mask=inside, other=0)
    tl.store(pos_ptr + t, length + r, mask=inside)
    col = length + r
    if HAS_BASE:
        col = length - tl.load(base_ptr + slot, mask=inside, other=0) + r
    capacity = tl.load(counts_ptr + row, mask=inside, other=0) * page_size
    tl.store(col_ptr + t, tl.maximum(tl.minimum(col, capacity - 1), 0), mask=inside)


@triton.jit
def _layer_offset(layer, o0, o1, o2, o3, o4, o5, o6, o7):
    offset = o0
    offset = tl.where(layer == 1, o1, offset)
    offset = tl.where(layer == 2, o2, offset)
    offset = tl.where(layer == 3, o3, offset)
    offset = tl.where(layer == 4, o4, offset)
    offset = tl.where(layer == 5, o5, offset)
    offset = tl.where(layer == 6, o6, offset)
    offset = tl.where(layer == 7, o7, offset)
    return offset


@triton.jit
def _store_kernel(k_ptr, v_ptr, k_token, k_layer, k_head, v_token, v_layer, v_head, col_ptr, rows_ptr, table_ptr,
                  table_row, pool_ptr, pool_page, pool_kv, pool_head, pool_entry, page_size,
                  o0, o1, o2, o3, o4, o5, o6, o7,
                  PER_REQUEST: tl.constexpr, HEADS: tl.constexpr, HEAD_DIM: tl.constexpr,
                  HEADS_P2: tl.constexpr, DIM_P2: tl.constexpr):
    """Token ``t`` (request ``t // PER_REQUEST``), layer ``l``: page ``table[rows[request], col[t] // page]``;
    ``pool_l[page, 0 / 1, h, col[t] % page, :] = k / v[t, l, h, :]`` (flashinfer's HND paged append)."""
    t = tl.program_id(0).to(tl.int64)
    layer = tl.program_id(1)
    row = tl.load(rows_ptr + t // PER_REQUEST).to(tl.int64)
    col = tl.load(col_ptr + t)
    page = tl.load(table_ptr + row * table_row + col // page_size).to(tl.int64)
    entry = col % page_size
    dest = pool_ptr + _layer_offset(layer, o0, o1, o2, o3, o4, o5, o6, o7)
    dest = dest + page * pool_page + entry * pool_entry
    h = tl.arange(0, HEADS_P2)[:, None]
    d = tl.arange(0, DIM_P2)[None, :]
    inside = (h < HEADS) & (d < HEAD_DIM)
    k = tl.load(k_ptr + t * k_token + layer * k_layer + h * k_head + d, mask=inside)
    v = tl.load(v_ptr + t * v_token + layer * v_layer + h * v_head + d, mask=inside)
    tl.store(dest + h * pool_head + d, k, mask=inside)
    tl.store(dest + pool_kv + h * pool_head + d, v, mask=inside)


# --------------------------------------------------------------------------
# The step's bookkeeping
# --------------------------------------------------------------------------
def _p2(n: int) -> int:
    return triton.next_power_of_2(max(int(n), 1))


def pool_geometry(pool) -> Optional[tuple]:
    """``(layer 0 view, per-layer element offsets, page / kv / head / entry strides, heads, head dim)`` of a
    list of per-layer ``[pages, 2, heads, page, head_dim]`` views (or one ``[L, pages, 2, heads, page, dim]``
    tensor), or None.

    Every layer must share the dtype, the shape past the page dimension and the strides, with unit stride
    along the head dimension, at a whole number of elements from layer 0.
    """
    if isinstance(pool, torch.Tensor):
        if pool.dim() != 6:
            return None
        pool = list(pool.unbind(0))
    if not isinstance(pool, (list, tuple)) or not 0 < len(pool) <= MAX_LAYERS:
        return None
    first = pool[0]
    if not isinstance(first, torch.Tensor) or first.dim() != 5 or first.shape[1] != 2:
        return None
    size = first.element_size()
    offsets = []
    for view in pool:
        if (
            not isinstance(view, torch.Tensor)
            or view.dtype != first.dtype
            or view.device != first.device
            or tuple(view.shape[1:]) != tuple(first.shape[1:])
            or tuple(view.stride()) != tuple(first.stride())
            or view.stride(4) != 1
            or (view.data_ptr() - first.data_ptr()) % size
        ):
            return None
        offsets.append((view.data_ptr() - first.data_ptr()) // size)
    offsets += [0] * (MAX_LAYERS - len(offsets))
    return (first, tuple(offsets), first.stride(0), first.stride(1), first.stride(2), first.stride(3),
            first.shape[2], first.shape[4])


def methods_unsupported_reason(worker) -> Optional[str]:
    """Why the fused path cannot stand in for this worker's methods (an override), or None."""
    from tensorrt_llm._torch.speculative.dflash import DFlashWorker

    cls = type(worker)
    for name in REPLACED_METHODS:
        if getattr(cls, name, None) is not getattr(DFlashWorker, name, None):
            return f"the worker class {cls.__name__} overrides {name}"
        if name in vars(worker):
            return f"{name} is replaced on this worker"
    return None


class DrafterStepPrep:
    """The fused bookkeeping of one DFlash worker's decode steps (``context_kv_fork``'s side chain)."""

    def __init__(self, min_requests: int, max_requests: int = MAX_REQUESTS) -> None:
        self.min_requests = min_requests
        self.max_requests = max_requests
        #: Fused commits, stores with the fused K/V launch, stores on the built-in append.
        self.steps = 0
        self.stores = 0
        self.store_fallbacks = 0
        self._geometry_key = None
        self._geometry = None
        self._declined_logged = False

    def geometry(self, worker) -> Optional[tuple]:
        """The worker's paged context pool geometry (``pool_geometry``), cached per pool."""
        pool = getattr(worker, "_ctx_kv_buf", None)
        if isinstance(pool, torch.Tensor):
            key = (pool.data_ptr(), tuple(pool.shape), tuple(pool.stride()), pool.dtype)
        elif isinstance(pool, (list, tuple)):
            key = tuple((v.data_ptr(), tuple(v.shape), tuple(v.stride()), v.dtype) for v in pool)
        else:
            return None
        if key != self._geometry_key:
            self._geometry_key = key
            self._geometry = pool_geometry(pool)
        return self._geometry

    def unsupported_reason(self, worker, num_gens: int, per_request: int) -> Optional[str]:
        """Why a decode step of ``num_gens`` requests keeps the built-in bookkeeping, or None.

        Host-side reads only (per-capture constants under a CUDA graph), the same on every rank.
        """
        if not self.min_requests <= num_gens <= self.max_requests:
            return f"{num_gens} requests, outside {self.min_requests}..{self.max_requests}"
        if not getattr(worker, "_ctx_paged", False):
            return "the drafter context is not paged"
        tables = getattr(worker, "_ctx_block_tables", None)
        counts = getattr(worker, "_ctx_block_counts", None)
        if tables is None or counts is None:
            return "the drafter context is not on the KV cache manager's pages"
        if tables.dtype != torch.int32 or tables.dim() != 2 or tables.stride(1) != 1:
            return "the page table is not a row-major int32 table"
        if counts.dtype != torch.int64 or not counts.is_contiguous() or num_gens > counts.numel() \
                or num_gens > tables.shape[0]:
            return "the page counts are not int64 per request"
        for name in ("_ctx_len", "_batch_to_slot"):
            tensor = getattr(worker, name, None)
            if tensor is None or tensor.dtype != torch.int64 or not tensor.is_contiguous():
                return f"{name} is not a contiguous int64 tensor"
        base = getattr(worker, "_ctx_base", None)
        if base is not None and (base.dtype != torch.int64 or not base.is_contiguous()
                                 or base.shape != worker._ctx_len.shape):
            return "_ctx_base is not a contiguous int64 tensor shaped as _ctx_len"
        if getattr(worker, "_dummy_slot", None) is None:
            return "the worker has no dummy slot"
        if per_request != getattr(worker, "_draft_tokens_per_req", None):
            return f"{per_request} stored rows per request, not the worker's K + 1"
        reason = methods_unsupported_reason(worker)
        if reason is not None:
            return reason
        if self.geometry(worker) is None:
            return "the context pool's layers do not share one geometry"
        return None

    def takes(self, worker, num_gens: int, per_request: int) -> bool:
        reason = self.unsupported_reason(worker, num_gens, per_request)
        if reason is not None and num_gens >= self.min_requests and not self._declined_logged:
            self._declined_logged = True
            logger.warning(f"[musespark_open] drafter_fused_prep: a decode step keeps the built-in drafter "
                           f"bookkeeping: {reason} (logged once)")
        return reason is None

    # -- the side chain --------------------------------------------------------
    def refresh_block_tables(self, worker, attn_metadata, num_seqs: int) -> bool:
        """``DFlashWorker._refresh_ctx_block_tables`` in two launches; the built-in when the step's
        offsets are missing or shaped otherwise (it raises or decodes as it does)."""
        from tensorrt_llm._torch.speculative.dflash import DFlashWorker

        tables = worker._ctx_block_tables
        offsets_name = "kv_cache_block_offsets" if worker._ctx_unified_cache else "draft_kv_cache_block_offsets"
        src = getattr(attn_metadata, offsets_name, None)
        if src is None or num_seqs <= 0 or src.dim() != 4:
            return DFlashWorker._refresh_ctx_block_tables(worker, attn_metadata, num_seqs)
        view = src[worker._ctx_pool_idx, :num_seqs, 0]
        width = view.shape[-1]
        if (
            width != tables.shape[1]
            or worker._ctx_block_cols is None
            or width > worker._ctx_block_cols.numel()
            or view.dtype not in (torch.int32, torch.int64)
        ):
            return DFlashWorker._refresh_ctx_block_tables(worker, attn_metadata, num_seqs)
        counts = worker._ctx_block_counts
        counts[:num_seqs].zero_()
        _refresh_kernel[(num_seqs, triton.cdiv(width, REFRESH_BLOCK))](
            view, view.stride(0), view.stride(1), tables, tables.stride(0), counts, width,
            int(worker._ctx_block_offset), int(worker._ctx_block_divisor), BLOCK=REFRESH_BLOCK, num_warps=4)
        return True

    def store_step_context(self, worker, draft_model, gen_hs, slots, rows, per_request: int) -> None:
        """``DFlashWorker._store_step_context`` without the mask (the fork's side chain) over the manager's
        pages: positions and columns in one launch (from one read of the lengths), the built-in projection
        and ``precompute_context_kv``, then every layer's K/V append in one launch."""
        requests = slots.numel()
        tokens = requests * per_request
        positions = torch.empty(tokens, dtype=torch.long, device=slots.device)
        columns = torch.empty(tokens, dtype=torch.long, device=slots.device)
        base = worker._ctx_base
        _columns_kernel[(triton.cdiv(tokens, POSITIONS_BLOCK),)](
            worker._ctx_len, worker._ctx_len if base is None else base, slots, rows, worker._ctx_block_counts,
            positions, columns, tokens, int(worker._ctx_page_size), PER_REQUEST=per_request,
            HAS_BASE=base is not None, BLOCK=POSITIONS_BLOCK, num_warps=4)
        projected = draft_model.project_target_hidden(gen_hs.reshape(-1, gen_hs.shape[-1]))
        cache_dtype = worker._ctx_kv_buf[0].dtype
        k_new, v_new = draft_model.precompute_context_kv(projected.to(cache_dtype), positions)
        geometry = self.geometry(worker)
        if (
            v_new is None
            or geometry is None
            or k_new.dim() != 4
            or tuple(k_new.shape) != tuple(v_new.shape)
            or k_new.shape[0] != tokens
            or k_new.shape[1] != len(worker._ctx_kv_buf)
            or (k_new.shape[2], k_new.shape[3]) != (geometry[6], geometry[7])
            or k_new.dtype != geometry[0].dtype
            or v_new.dtype != geometry[0].dtype
            or k_new.stride(3) != 1
            or v_new.stride(3) != 1
        ):
            # Any other K/V layout: the built-in paged append, on the same columns.
            self.store_fallbacks += 1
            rows_long = rows.unsqueeze(1).expand(requests, per_request).reshape(-1)
            worker._store_context_kv_paged(k_new, v_new, rows_long, columns)
            return
        first, offsets, s_page, s_kv, s_head, s_entry, heads, head_dim = geometry
        _store_kernel[(tokens, k_new.shape[1])](
            k_new, v_new, k_new.stride(0), k_new.stride(1), k_new.stride(2), v_new.stride(0), v_new.stride(1),
            v_new.stride(2), columns, rows, worker._ctx_block_tables, worker._ctx_block_tables.stride(0), first,
            s_page, s_kv, s_head, s_entry, int(worker._ctx_page_size), *offsets, PER_REQUEST=per_request,
            HEADS=heads, HEAD_DIM=head_dim, HEADS_P2=_p2(heads), DIM_P2=_p2(head_dim),
            num_warps=1 if heads * head_dim <= 256 else 4)
        self.stores += 1

    # -- the commit ------------------------------------------------------------
    def commit_context(self, worker, slots, num_accepted, rows, out) -> None:
        """``out.copy_(DFlashWorker._commit_context(slots, num_accepted.long(), rows))`` over the manager's
        pages, in one launch (#21535's ``context_kv_commit_paged``); the built-in for a layout it does not
        take."""
        requests = slots.numel()
        if requests == 0:
            return
        if (
            num_accepted.dtype not in (torch.int32, torch.int64)
            or num_accepted.dim() != 1
            or num_accepted.numel() != requests
            or not num_accepted.is_contiguous()
            or slots.dtype != torch.int64
            or not slots.is_contiguous()
            or not rows.is_contiguous()
            or worker._dummy_slot is None
        ):
            out.copy_(worker._commit_context(slots, num_accepted.long(), rows))
            return
        context_kv_commit_paged(
            worker._ctx_len, worker._ctx_base, slots, num_accepted, rows,
            worker._ctx_block_counts if worker._ctx_block_tables is not None else None,
            worker._ctx_page_size, worker._compute_block_size, worker._max_ctx, worker._dummy_slot,
            worker._ctx_unified_cache, out)
        self.steps += 1
        if self.steps == 1:
            logger.info(f"[musespark_open] drafter_fused_prep: the first fused decode step ran ({requests} "
                        "requests): page-table refresh, context positions and columns, every layer's context K/V "
                        "store and the commit as Triton launches")


def from_env() -> Optional[DrafterStepPrep]:
    """The fused bookkeeping when ``MUSESPARK_OPEN_DRAFTER_FUSED_PREP`` is ``1``, else None.

    Raises:
        ValueError: on a ``MUSESPARK_OPEN_DRAFTER_FUSED_PREP_MIN_REQUESTS`` that is not a whole number in
            1..MAX_REQUESTS (on every rank alike: they share the environment).
    """
    if os.environ.get(ENV, "0").strip() != "1":
        return None
    raw = os.environ.get(MIN_REQUESTS_ENV, str(DEFAULT_MIN_REQUESTS)).strip()
    try:
        min_requests = int(raw)
    except ValueError as error:
        raise ValueError(f"[musespark_open] {MIN_REQUESTS_ENV}={raw!r} is not a whole number") from error
    if not 1 <= min_requests <= MAX_REQUESTS:
        raise ValueError(f"[musespark_open] {MIN_REQUESTS_ENV}={raw!r} is outside 1..{MAX_REQUESTS}")
    logger.info(f"[musespark_open] {ENV}=1: decode steps of {min_requests}..{MAX_REQUESTS} requests run the "
                "drafter's paged context-K/V bookkeeping (page-table refresh, store, commit) as fused Triton launches")
    return DrafterStepPrep(min_requests, MAX_REQUESTS)
