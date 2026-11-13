# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DFlash context-K/V side chain over a paged drafter context: its integer prologue in one launch.

With the drafter's context in the KV cache manager's pages, the worker's
``_store_step_context`` (``_torch/speculative/dflash.py``) computes around its GEMMs a
handful of small torch kernels: the positions ``ctx_len + r`` of each generation
request's ``K + 1`` rows, their columns clamped to the request's allocation
(``block_counts * page_size - 1``, at least 0), the page-table row of every entry (the
request's batch position) and, inside ``DFlashForCausalLM._fused_rope_inplace``, the
positions as int32 repeated per drafter layer for the RoPE kernel.

``CtxKvPrepPaged`` computes the columns, the row keys and the repeated int32 positions
in one program. Integer arithmetic only: every value is the built-in's.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, OpWrapper

#: Generation requests one call takes (``context_kv_fork.MAX_REQUESTS``).
MAX_REQUESTS = 64


@triton.jit
def _ctx_kv_prep_paged_kernel(
    ctx_len_ptr,
    slots_ptr,
    counts_ptr,
    col_ptr,
    keys_ptr,
    pos_i32_ptr,
    num_gens,
    page_size,
    ROWS: tl.constexpr,
    LAYERS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    r = tl.arange(0, BLOCK)
    rows = r < ROWS
    for g in range(num_gens):
        slot = tl.load(slots_ptr + g)
        ctx = tl.load(ctx_len_ptr + slot)
        pos = ctx + r.to(tl.int64)
        last = tl.load(counts_ptr + g).to(tl.int64) * page_size - 1
        tl.store(col_ptr + g * ROWS + r, tl.maximum(tl.minimum(pos, last), 0), mask=rows)
        tl.store(keys_ptr + g * ROWS + r, (r * 0 + g).to(tl.int64), mask=rows)
        pos32 = pos.to(tl.int32)
        for layer in tl.static_range(LAYERS):
            tl.store(pos_i32_ptr + (g * ROWS + r) * LAYERS + layer, pos32, mask=rows)


class CtxKvPrepPaged(OpWrapper):
    """The paged side chain's columns, row keys and per-layer int32 RoPE positions, in one program.

    ``ctx_len`` int64 ``[slots]``, ``slots`` int64 ``[gens]``, ``counts`` int64 (the
    extent in pages of each batch row's allocation, ``_ctx_block_counts``); writes ``col``
    and ``keys`` int64 ``[gens * rows]`` and ``pos_i32`` int32 ``[gens * rows * layers]``:

    * ``col[g * rows + r] = clamp(min(ctx_len[slots[g]] + r, counts[g] * page_size - 1), min=0)``;
    * ``keys[g * rows + r] = g``;
    * ``pos_i32 = (ctx_len[slots] + r).to(int32).repeat_interleave(layers)``.
    """

    ARCHS = frozenset({Arch.SM_107})

    note = """
    Metadata consumed: the drafter's allocation extents (``_ctx_block_counts``, read).
    One Triton program (grid 1, 1 warp) looping over the requests. Call site:
    ``context_kv_fork``'s paged side chain under ``drafter_kv_side_chain``, after the
    step's page-table decode.
    Numerics: integer arithmetic; int32 is the low word, as ``.to(torch.int32)``.
    Hard limits: rows <= 64, gens <= MAX_REQUESTS.
    """

    def __call__(
        self, ctx_len, slots, counts, col, keys, pos_i32, rows: int, layers: int, page_size: int
    ):
        _ctx_kv_prep_paged_kernel[(1,)](
            ctx_len,
            slots,
            counts,
            col,
            keys,
            pos_i32,
            slots.numel(),
            page_size,
            ROWS=rows,
            LAYERS=layers,
            BLOCK=triton.next_power_of_2(rows),
            num_warps=1,
        )

    def reference(
        self, ctx_len, slots, counts, col, keys, pos_i32, rows: int, layers: int, page_size: int
    ):
        """The worker's ``_store_step_context`` statements (no window base, the manager's table)."""
        gens = slots.numel()
        offsets = torch.arange(rows, dtype=torch.long, device=ctx_len.device)
        pos = ctx_len[slots].unsqueeze(1) + offsets.unsqueeze(0)
        capacity = counts[:gens] * page_size
        col.copy_(torch.minimum(pos, (capacity - 1).unsqueeze(1)).clamp_(min=0).reshape(-1))
        batch = torch.arange(gens, dtype=torch.long, device=ctx_len.device)
        keys.copy_(batch.unsqueeze(1).expand_as(pos).reshape(-1))
        pos_i32.copy_(pos.reshape(-1).to(torch.int32).repeat_interleave(layers))

    def is_valid(
        self, ctx_len, slots, counts, col, keys, pos_i32, rows: int, layers: int, page_size: int
    ):
        if ctx_len.dtype != torch.int64 or slots.dtype != torch.int64:
            raise ValueError("ctx_len and slots must be int64")
        if counts.dtype != torch.int64 or not counts.is_contiguous():
            raise ValueError("counts must be contiguous int64")
        gens = slots.numel()
        if not 1 <= rows <= 64 or not 1 <= gens <= MAX_REQUESTS or counts.numel() < gens:
            raise ValueError(f"1..64 rows, 1..{MAX_REQUESTS} requests, a count for each")
        if page_size < 1 or layers < 1:
            raise ValueError("page_size and layers must be positive")
        n = gens * rows
        for name, out, dtype, numel in (
            ("col", col, torch.int64, n),
            ("keys", keys, torch.int64, n),
            ("pos_i32", pos_i32, torch.int32, n * layers),
        ):
            if out.dtype != dtype or out.numel() != numel or not out.is_contiguous():
                raise ValueError(f"{name} must be contiguous {dtype} of {numel} entries")


ctx_kv_prep_paged = CtxKvPrepPaged()
