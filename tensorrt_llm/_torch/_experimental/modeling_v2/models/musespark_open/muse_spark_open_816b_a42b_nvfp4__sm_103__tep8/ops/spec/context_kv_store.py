# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DFlash context-K/V side chain: its integer prologue and its arena store, one launch each.

The ``context_kv_fork`` side chain (``kernels/context_kv_fork/fork.py``) stores a
decode step's ``K + 1`` captured rows of every generation request into the
drafter's dense context arena, unmasked. Around its GEMMs and norms the
built-in statements it reproduces launch about a dozen small torch kernels:

* in front, ``ctx_len[slots]``, ``arange(K + 1)``, the positions ``ctx_len + r``,
  their clamp to ``max_ctx - 1`` (the columns) and, inside
  ``DFlashForCausalLM._fused_rope_inplace``, the positions as int32 repeated
  per drafter layer for the RoPE kernel;
* behind the RoPE, ``v.contiguous()`` and the two ``index_put`` stores
  ``k_arena[slot, :, col] = k`` / ``v_arena[slot, :, col] = v`` with their
  value copies.

``ContextKvPrep`` computes the positions, the columns and the repeated int32
positions in one program; ``ContextKvScatter`` stores K (post norm and RoPE)
and V (read in place from the fused K/V GEMM's output) into both arenas in
one launch. Integer arithmetic and copies only: every value written is the
built-in's.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: Generation requests one prep call takes (``context_kv_commit.MAX_REQUESTS``).
MAX_REQUESTS = 64


@triton.jit
def _ctx_kv_prep_kernel(
    ctx_len_ptr,
    slots_ptr,
    positions_ptr,
    col_ptr,
    pos_i32_ptr,
    num_gens,
    max_ctx,
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
        tl.store(positions_ptr + g * ROWS + r, pos, mask=rows)
        tl.store(col_ptr + g * ROWS + r, tl.minimum(pos, max_ctx - 1), mask=rows)
        pos32 = pos.to(tl.int32)
        for layer in tl.static_range(LAYERS):
            tl.store(pos_i32_ptr + (g * ROWS + r) * LAYERS + layer, pos32, mask=rows)


class ContextKvPrep(OpWrapper):
    """The side chain's positions, columns and per-layer int32 RoPE positions, in one program.

    ``ctx_len`` int64 ``[slots]``, ``slots`` int64 ``[gens]``; writes ``positions`` and
    ``col`` int64 ``[gens * rows]`` and ``pos_i32`` int32 ``[gens * rows * layers]``:

    * ``positions[g * rows + r] = ctx_len[slots[g]] + r``;
    * ``col = positions.clamp(max=max_ctx - 1)``;
    * ``pos_i32 = positions.to(int32).repeat_interleave(layers)``.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="the served decode step: one request, K + 1 = 8 rows, 5 drafter layers",
            spec=dict(gens=1, rows=8, layers=5, slots=2),
        ),
        Cell(
            why="a context at the arena's end: columns clamp, positions do not",
            spec=dict(gens=1, rows=8, layers=5, slots=2, at_end=True),
        ),
        Cell(
            why="several requests in distinct slots",
            spec=dict(gens=3, rows=4, layers=2, slots=5),
        ),
    )

    note = """
    Metadata consumed: none. One Triton program (grid 1, 1 warp) looping over
    the requests. Call site: ``context_kv_fork.ContextKvFork.side_chain`` under
    ``drafter_kv_side_chain``, first launch of the side stream.
    Numerics: integer arithmetic; int32 is the low word, as ``.to(torch.int32)``.
    Hard limits: rows <= 64, gens <= MAX_REQUESTS.
    """

    def __call__(
        self, ctx_len, slots, positions, col, pos_i32, rows: int, layers: int, max_ctx: int
    ):
        _ctx_kv_prep_kernel[(1,)](
            ctx_len,
            slots,
            positions,
            col,
            pos_i32,
            slots.numel(),
            max_ctx,
            ROWS=rows,
            LAYERS=layers,
            BLOCK=triton.next_power_of_2(rows),
            num_warps=1,
        )

    def reference(
        self, ctx_len, slots, positions, col, pos_i32, rows: int, layers: int, max_ctx: int
    ):
        offsets = torch.arange(rows, dtype=torch.long, device=ctx_len.device)
        pos = ctx_len[slots].unsqueeze(1) + offsets.unsqueeze(0)
        positions.copy_(pos.reshape(-1))
        col.copy_(pos.clamp(max=max_ctx - 1).reshape(-1))
        pos_i32.copy_(pos.reshape(-1).to(torch.int32).repeat_interleave(layers))

    def is_valid(
        self, ctx_len, slots, positions, col, pos_i32, rows: int, layers: int, max_ctx: int
    ):
        assert ctx_len.dtype == torch.int64 and slots.dtype == torch.int64
        n = slots.numel() * rows
        assert (
            positions.dtype == torch.int64 and positions.numel() == n and positions.is_contiguous()
        )
        assert col.dtype == torch.int64 and col.numel() == n and col.is_contiguous()
        assert pos_i32.dtype == torch.int32 and pos_i32.numel() == n * layers
        assert pos_i32.is_contiguous()
        assert 1 <= rows <= 64 and slots.numel() <= MAX_REQUESTS


context_kv_prep = ContextKvPrep()


@triton.jit
def _ctx_kv_scatter_kernel(
    k_ptr,
    kv_ptr,
    slots_ptr,
    col_ptr,
    k_arena_ptr,
    v_arena_ptr,
    max_ctx,
    kv_row_stride,
    k_slot_stride,
    k_layer_stride,
    k_col_stride,
    v_slot_stride,
    v_layer_stride,
    v_col_stride,
    ROWS: tl.constexpr,
    LAYERS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_W: tl.constexpr,
    HAS_V: tl.constexpr,
):
    n = tl.program_id(0)
    g = n // ROWS
    r = n % ROWS
    slot = tl.load(slots_ptr + g)
    col = tl.load(col_ptr + n)
    # Rows whose columns clamp onto one column: the last of them is the one kept, as a
    # deterministic index_put keeps the last duplicate (context_kv_commit's assumption).
    if (r + 1 < ROWS) & (col >= max_ctx - 1):
        return
    layer = tl.arange(0, BLOCK_L)
    w = tl.arange(0, BLOCK_W)
    m = (layer < LAYERS)[:, None] & (w < WIDTH)[None, :]
    layer64 = layer.to(tl.int64)
    k = tl.load(k_ptr + n * (LAYERS * WIDTH) + layer[:, None] * WIDTH + w[None, :], mask=m)
    if HAS_V:
        # The fused K/V GEMM's row: per layer [K | V], each WIDTH wide.
        v = tl.load(
            kv_ptr + n * kv_row_stride + layer[:, None] * (2 * WIDTH) + WIDTH + w[None, :], mask=m
        )
    k_dst = (
        k_arena_ptr + slot * k_slot_stride + layer64[:, None] * k_layer_stride + col * k_col_stride
    )
    tl.store(k_dst + w[None, :], k, mask=m)
    if HAS_V:
        v_dst = (
            v_arena_ptr
            + slot * v_slot_stride
            + layer64[:, None] * v_layer_stride
            + col * v_col_stride
        )
        tl.store(v_dst + w[None, :], v, mask=m)


class ContextKvScatter(OpWrapper):
    """Store a step's context K and V rows into the dense arenas, as the built-in ``index_put`` pair does.

    ``k`` BF16 ``[n, layers, kv_heads, head_dim]`` contiguous (post norm and RoPE);
    ``kv`` the fused K/V GEMM output ``[n, layers * 2 * kv_heads * head_dim]`` whose
    V half of layer ``l`` is the second ``kv_heads * head_dim`` block of the
    layer's ``[K | V]`` pair; ``slots`` int64 ``[gens]``, ``col`` int64
    ``[gens * rows]`` (``n = gens * rows``). Writes, for ``i = g * rows + r``:
    ``k_arena[slots[g], :, col[i]] = k[i]`` and
    ``v_arena[slots[g], :, col[i]] = kv.view(n, layers, 2, kv_heads, head_dim)[i, :, 1]``.
    Where rows of a request clamp onto the arena's last column, the last of them is
    stored (a deterministic ``index_put``'s last-duplicate-wins).
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="the served decode step: one request, 8 rows, 5 layers, 1 kv head x 128",
            spec=dict(gens=1, rows=8, layers=5, kv_heads=1, head_dim=128, slots=2),
        ),
        Cell(
            why="a context at the arena's end: duplicate columns, the last row stored",
            spec=dict(gens=1, rows=8, layers=5, kv_heads=1, head_dim=128, slots=2, at_end=True),
        ),
        Cell(
            why="several requests, 2 kv heads x 64",
            spec=dict(gens=3, rows=4, layers=2, kv_heads=2, head_dim=64, slots=5),
        ),
    )

    note = """
    Metadata consumed: none. Grid ``gens * rows`` Triton programs (1 warp), each
    storing one row's ``layers`` columns of both arenas. Call site:
    ``context_kv_fork.ContextKvFork.side_chain`` under ``drafter_kv_side_chain``,
    in place of ``v.contiguous()`` and the two ``index_put`` stores.
    Numerics: copies. Hard limits: the (kv_heads, head_dim) elements of an arena
    column contiguous; distinct slots across requests.
    """

    def __call__(self, k, kv, slots, col, k_arena, v_arena, rows: int, max_ctx: int):
        n, layers = k.shape[0], k.shape[1]
        width = k.shape[2] * k.shape[3]
        has_v = v_arena is not None
        values = v_arena if has_v else k_arena
        _ctx_kv_scatter_kernel[(n,)](
            k,
            kv,
            slots,
            col,
            k_arena,
            values,
            max_ctx,
            kv.stride(0),
            k_arena.stride(0),
            k_arena.stride(1),
            k_arena.stride(2),
            values.stride(0),
            values.stride(1),
            values.stride(2),
            ROWS=rows,
            LAYERS=layers,
            WIDTH=width,
            BLOCK_L=triton.next_power_of_2(layers),
            BLOCK_W=triton.next_power_of_2(width),
            HAS_V=has_v,
            num_warps=1,
        )

    def reference(self, k, kv, slots, col, k_arena, v_arena, rows: int, max_ctx: int):
        """The built-in statements (``fork.side_chain``), with the stores row by row in order."""
        n, layers, kv_heads, head_dim = k.shape
        slot_flat = slots.unsqueeze(1).expand(-1, rows).reshape(-1)
        v = kv.view(n, layers, 2, kv_heads, head_dim)[:, :, 1].contiguous()
        for i in range(n):
            k_arena[slot_flat[i], :, col[i]] = k[i]
            if v_arena is not None:
                v_arena[slot_flat[i], :, col[i]] = v[i]

    def is_valid(self, k, kv, slots, col, k_arena, v_arena, rows: int, max_ctx: int):
        assert k.dim() == 4 and k.is_contiguous() and k.dtype == torch.bfloat16
        n, layers, kv_heads, head_dim = k.shape
        assert kv.dim() == 2 and kv.shape == (n, layers * 2 * kv_heads * head_dim)
        assert kv.stride(1) == 1 and kv.dtype == torch.bfloat16
        assert slots.dtype == torch.int64 and col.dtype == torch.int64 and col.numel() == n
        assert n == slots.numel() * rows
        for arena in (k_arena, v_arena):
            if arena is None:
                continue
            assert arena.dim() == 5 and arena.dtype == torch.bfloat16
            assert arena.shape[1] == layers and arena.shape[3:] == (kv_heads, head_dim)
            assert arena.stride(4) == 1 and arena.stride(3) == head_dim


context_kv_scatter = ContextKvScatter()
