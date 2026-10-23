# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DFlash prepare over a windowed context arena: the verify-dependent end of the context-K/V store, in two launches.

With ``TRTLLM_DFLASH_WINDOWED_CTX_ARENA`` the built-in DFlash worker keeps only the
last ``keep`` context columns of each slot (``keep`` = the drafter's widest
window): physical column ``c`` of slot ``s`` holds logical token ``base[s] + c``,
so a slot's columns are ``[0, ctx_len[s] - base[s])``. Its
``prepare_1st_drafter_inputs`` (``_torch/speculative/dflash.py``) stores the
``K + 1`` projected rows of a generation request at physical columns
``min(ctx_len - base + r, limit - 1)`` (``limit`` = the arena's capacity), with the
rows at and past the accepted count multiplied by zero; advances ``ctx_len`` by
the accepted count and clamps every slot's length to ``max_ctx``; then compacts
(``DFlashWorker._compact_ctx_window``): each slot whose physical length exceeds
``keep`` has its columns shifted left by the excess (column ``c`` takes column
``min(c + shift, capacity - 1)``, every column of the slot) and its base advanced
by it; the drafter then reads ``ctx_len - base`` (``num_ctx_per_req``).

The ``context_kv_fork`` fast path stores the rows unmasked earlier, on a side
stream that does not wait for the verify. This entry is what is left once the
accepted counts exist, as two kernels:

* the commit (one program, like ``context_kv_commit``'s): the zeroing of the
  masked rows in place at their physical columns, the length update, each
  request's shift, the base update and the new physical lengths;
* the compaction (one program per request, layer, arena and chunk of a
  column's elements): the left shift of every column of the slot, in place,
  column tile by column tile in increasing order (every source column is at
  or right of its destination, and a CTA barrier separates a tile's loads
  from its stores), skipped for a request whose shift is 0 and for a repeated
  slot (CUDA-graph padding rows share the dummy slot; the built-in writes the
  same rows for each copy).

The zeroing is ``context_kv_commit``'s BF16 product with 0.0 and the
compaction moves bytes, so the arena, lengths and bases end up holding the
built-in's bytes.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, OpWrapper

from .context_kv_commit import MAX_REQUESTS, _columns, _times_zero, tile_pairs

#: Columns one compaction tile moves (per program, per loop iteration).
COMPACT_COLUMNS = 512

#: Elements of a column one compaction program moves: a column's ``kv_heads *
#: head_dim`` elements split into chunks of this many.
COMPACT_CHUNK = 16


@triton.jit
def _commit_windowed_kernel(
    k_ptr,
    v_ptr,
    ctx_len_ptr,
    ctx_base_ptr,
    slots_ptr,
    accepted_ptr,
    new_len_ptr,
    new_base_ptr,
    shift_ptr,
    num_ctx_ptr,
    num_gens,
    num_slots,
    max_ctx,
    col_limit,
    keep,
    k_slot_stride,
    k_layer_stride,
    k_col_stride,
    v_slot_stride,
    v_layer_stride,
    v_col_stride,
    ROWS: tl.constexpr,
    LAYERS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    PAIRS: tl.constexpr,
    HAS_V: tl.constexpr,
):
    # The zeroing at the physical columns of the lengths and bases before this call
    # (``_columns`` clamps to its last column, ``col_limit - 1``).
    for g in range(num_gens):
        slot = tl.load(slots_ptr + g).to(tl.int64)
        ctx = tl.load(ctx_len_ptr + slot)
        phys = ctx - tl.load(ctx_base_ptr + slot)
        accepted = tl.load(accepted_ptr + g).to(tl.int64)
        for start in tl.static_range(0, ROWS * LAYERS, PAIRS):
            k_cols, zeroed = _columns(
                k_ptr, slot, phys, accepted, col_limit - 1, k_slot_stride, k_layer_stride,
                k_col_stride, start, ROWS, LAYERS, WIDTH, BLOCK, PAIRS,
            )  # fmt: skip
            if HAS_V:
                v_cols, _ = _columns(
                    v_ptr, slot, phys, accepted, col_limit - 1, v_slot_stride, v_layer_stride,
                    v_col_stride, start, ROWS, LAYERS, WIDTH, BLOCK, PAIRS,
                )  # fmt: skip
                k = tl.load(k_cols, mask=zeroed, other=0.0)
                v = tl.load(v_cols, mask=zeroed, other=0.0)
                tl.store(k_cols, _times_zero(k), mask=zeroed)
                tl.store(v_cols, _times_zero(v), mask=zeroed)
            else:
                k = tl.load(k_cols, mask=zeroed, other=0.0)
                tl.store(k_cols, _times_zero(k), mask=zeroed)
        tl.store(new_len_ptr + g, ctx + accepted)
    tl.debug_barrier()
    # ctx_len[slots] += accepted (gathered before, scattered in request order), then the clamp.
    for g in range(num_gens):
        slot = tl.load(slots_ptr + g).to(tl.int64)
        tl.store(ctx_len_ptr + slot, tl.load(new_len_ptr + g))
    for s in range(num_slots):
        tl.store(ctx_len_ptr + s, tl.minimum(tl.load(ctx_len_ptr + s), max_ctx))
    tl.debug_barrier()
    # The compaction's shift of every request, from the new lengths and the bases before it.
    for g in range(num_gens):
        slot = tl.load(slots_ptr + g).to(tl.int64)
        base = tl.load(ctx_base_ptr + slot)
        shift = tl.where(keep > 0, tl.maximum(tl.load(ctx_len_ptr + slot) - base - keep, 0), 0)
        tl.store(shift_ptr + g, shift)
        tl.store(new_base_ptr + g, base + shift)
    tl.debug_barrier()
    # ctx_base[slots] += shift, then the drafter's physical lengths ctx_len[slots] - ctx_base[slots].
    for g in range(num_gens):
        slot = tl.load(slots_ptr + g).to(tl.int64)
        tl.store(ctx_base_ptr + slot, tl.load(new_base_ptr + g))
    tl.debug_barrier()
    for g in range(num_gens):
        slot = tl.load(slots_ptr + g).to(tl.int64)
        tl.store(num_ctx_ptr + g, tl.load(ctx_len_ptr + slot) - tl.load(ctx_base_ptr + slot))


@triton.jit
def _compact_kernel(
    k_ptr,
    v_ptr,
    slots_ptr,
    shift_ptr,
    capacity,
    k_slot_stride,
    k_layer_stride,
    k_col_stride,
    v_slot_stride,
    v_layer_stride,
    v_col_stride,
    LAYERS: tl.constexpr,
    WIDTH: tl.constexpr,
    CHUNK: tl.constexpr,
    TILE: tl.constexpr,
    GENS: tl.constexpr,
    HAS_V: tl.constexpr,
):
    g = tl.program_id(0)
    which = tl.program_id(1)
    chunk = tl.program_id(2)
    layer = (which % LAYERS).to(tl.int64)
    shift = tl.load(shift_ptr + g)
    slot = tl.load(slots_ptr + g)
    # A slot an earlier request also holds is moved by that request's programs.
    earlier = tl.arange(0, GENS)
    repeats = tl.sum(
        ((earlier < g) & (tl.load(slots_ptr + earlier, mask=earlier < g, other=-1) == slot)).to(
            tl.int32
        ),
        0,
    )
    if (shift > 0) & (repeats == 0):
        if which < LAYERS:
            ptr = k_ptr + slot.to(tl.int64) * k_slot_stride + layer * k_layer_stride
            col_stride = k_col_stride
        else:
            ptr = v_ptr + slot.to(tl.int64) * v_slot_stride + layer * v_layer_stride
            col_stride = v_col_stride
        elems = chunk * CHUNK + tl.arange(0, CHUNK)
        in_col = elems < WIDTH
        for start in range(0, capacity, TILE):
            cols = start + tl.arange(0, TILE)
            dst = cols < capacity
            src = tl.minimum(cols + shift, capacity - 1).to(tl.int64)
            mask = dst[:, None] & in_col[None, :]
            x = tl.load(ptr + src[:, None] * col_stride + elems[None, :], mask=mask)
            # Every thread's loads of this tile before any store of it: a source column
            # of this tile can be another thread's destination column.
            tl.debug_barrier()
            tl.store(ptr + cols.to(tl.int64)[:, None] * col_stride + elems[None, :], x, mask=mask)


class ContextKvCommitWindowed(OpWrapper):
    """``context_kv_commit`` over a windowed arena: zero the masked rows, advance the lengths, compact.

    ``k_arena`` / ``v_arena`` are the windowed arenas ``[slots, layers, capacity,
    kv_heads, head_dim]`` (BF16; ``v_arena`` may be None); the ``rows`` columns
    ``min(ctx_len[slot] - ctx_base[slot] + r, col_limit - 1)``, ``r < rows``, of
    every request's slot hold that step's unmasked context K/V. ``ctx_len`` and
    ``ctx_base`` are int64 ``[slots]`` (logical length, logical index of physical
    column 0), ``slots`` int64 ``[gens]``, ``num_accepted`` int32 or int64
    ``[gens]``, ``num_ctx_out`` int64 ``[gens]``; ``keep`` is the number of
    columns kept (0: no compaction), ``col_limit`` the column clamp. In place:

    * for each request ``g`` and row ``r >= num_accepted[g]``, in every layer of
      both arenas: the column's elements become ``bf16(fp32(x) * 0.0)``;
    * ``ctx_len[slots[g]] += num_accepted[g]`` (gathered before any write,
      scattered in request order), then every slot's length is clamped to
      ``max_ctx``;
    * ``shift[g] = max(ctx_len[slots[g]] - ctx_base[slots[g]] - keep, 0)``
      (``keep > 0``; 0 otherwise); every column ``c`` of the slot takes column
      ``min(c + shift[g], capacity - 1)``; ``ctx_base[slots[g]] += shift[g]``;
    * ``num_ctx_out[g] = ctx_len[slots[g]] - ctx_base[slots[g]]``.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})

    note = """
    Metadata consumed: none. Two Triton launches. The commit is one program
    (4 warps) that loops over the requests as ``context_kv_commit`` does, with
    CTA barriers between the reads of the lengths and bases and their writes.
    The compaction's grid is (requests, 2 x layers, column chunks of
    ``COMPACT_CHUNK`` elements); a program loops over the slot's columns in
    tiles of ``COMPACT_COLUMNS`` in increasing order, loading a tile, a CTA
    barrier, then storing it; it reads its request's shift from the commit's
    output, so its launch follows the commit on the stream.

    Call site: ``context_kv_fork.ContextKvFork.commit`` under the
    ``context_kv_fork`` fast path when the worker's arena is windowed
    (``DFlashWorker._ctx_keep > 0``), on the side stream that stored the unmasked
    rows, behind the main stream's last read of ``ctx_len`` in the prepare; the
    main stream joins before the drafter reads the arena and ``num_ctx_out``.

    Numerics: the zeroing is ``context_kv_commit``'s (the BF16 product with
    0.0); the compaction copies bytes. Bitwise the built-in's arena, lengths and
    bases when the arena held the unmasked rows.

    Hard limits
      Slots must be distinct across requests except for CUDA-graph padding,
      whose repeated slot is moved once (the built-in writes identical rows for
      each copy). ``ctx_len[s] - ctx_base[s]`` must be at most ``keep`` before
      the call on every slot the call names (the built-in's invariant after
      each step's compaction), so the stored columns lie inside the arena.

    Not certified: torch.compile. CUDA-graph capture and replay are exercised
    by the test.
    """

    def __call__(
        self,
        k_arena: torch.Tensor,
        v_arena,
        ctx_len: torch.Tensor,
        ctx_base: torch.Tensor,
        slots: torch.Tensor,
        num_accepted: torch.Tensor,
        num_ctx_out: torch.Tensor,
        rows: int,
        max_ctx: int,
        col_limit: int,
        keep: int,
    ) -> None:
        num_gens = slots.numel()
        if num_gens == 0:
            return
        layers, width = k_arena.shape[1], k_arena.shape[3] * k_arena.shape[4]
        scratch = torch.empty(3, num_gens, dtype=torch.int64, device=ctx_len.device)
        has_v = v_arena is not None
        values = v_arena if has_v else k_arena
        _commit_windowed_kernel[(1,)](
            k_arena,
            values,
            ctx_len,
            ctx_base,
            slots,
            num_accepted,
            scratch[0],
            scratch[1],
            scratch[2],
            num_ctx_out,
            num_gens,
            ctx_len.numel(),
            max_ctx,
            col_limit,
            keep,
            k_arena.stride(0),
            k_arena.stride(1),
            k_arena.stride(2),
            values.stride(0),
            values.stride(1),
            values.stride(2),
            ROWS=rows,
            LAYERS=layers,
            WIDTH=width,
            BLOCK=triton.next_power_of_2(width),
            PAIRS=tile_pairs(rows, layers, width),
            HAS_V=has_v,
            num_warps=4,
        )
        if keep <= 0:
            return
        chunk = min(COMPACT_CHUNK, triton.next_power_of_2(width))
        _compact_kernel[(num_gens, layers * (2 if has_v else 1), triton.cdiv(width, chunk))](
            k_arena,
            values,
            slots,
            scratch[2],
            k_arena.shape[2],
            k_arena.stride(0),
            k_arena.stride(1),
            k_arena.stride(2),
            values.stride(0),
            values.stride(1),
            values.stride(2),
            LAYERS=layers,
            WIDTH=width,
            CHUNK=chunk,
            TILE=COMPACT_COLUMNS,
            GENS=triton.next_power_of_2(max(num_gens, 2)),
            HAS_V=has_v,
            num_warps=4,
        )

    def reference(
        self,
        k_arena: torch.Tensor,
        v_arena,
        ctx_len: torch.Tensor,
        ctx_base: torch.Tensor,
        slots: torch.Tensor,
        num_accepted: torch.Tensor,
        num_ctx_out: torch.Tensor,
        rows: int,
        max_ctx: int,
        col_limit: int,
        keep: int,
    ) -> None:
        """The built-in torch ops, in place: mask the stored rows, update the lengths, compact.

        The masking is one row at a time, as ``context_kv_commit.reference``
        does; the compaction is ``DFlashWorker._compact_ctx_window``'s gather.
        """
        offsets = torch.arange(rows, dtype=torch.long, device=ctx_len.device)
        phys = ctx_len[slots] - ctx_base[slots]
        col = (phys.unsqueeze(1) + offsets.unsqueeze(0)).clamp(max=col_limit - 1)
        mask = offsets.unsqueeze(0) < num_accepted.long().unsqueeze(1)
        for arena in (k_arena, v_arena):
            if arena is None:
                continue
            for r in range(rows):
                stored = arena[slots, :, col[:, r]]
                stored.mul_(mask[:, r].view(-1, 1, 1, 1).to(stored.dtype))
                arena[slots, :, col[:, r]] = stored
        ctx_len[slots] += num_accepted.long()
        ctx_len.clamp_(max=max_ctx)
        if keep > 0:
            phys = ctx_len[slots] - ctx_base[slots]
            shift = (phys - keep).clamp_(min=0)
            capacity = k_arena.size(2)
            cols = torch.arange(capacity, dtype=torch.long, device=slots.device)
            src = (cols.unsqueeze(0) + shift.unsqueeze(1)).clamp_(max=capacity - 1)
            for arena in (k_arena, v_arena):
                if arena is None:
                    continue
                gathered = arena[slots]
                index = src.view(src.size(0), 1, capacity, 1, 1).expand_as(gathered)
                arena[slots] = gathered.gather(2, index)
            ctx_base[slots] += shift
        num_ctx_out.copy_(ctx_len[slots] - ctx_base[slots])

    def is_valid(
        self,
        k_arena: torch.Tensor,
        v_arena,
        ctx_len: torch.Tensor,
        ctx_base: torch.Tensor,
        slots: torch.Tensor,
        num_accepted: torch.Tensor,
        num_ctx_out: torch.Tensor,
        rows: int,
        max_ctx: int,
        col_limit: int,
        keep: int,
    ) -> None:
        for name, arena in (("k_arena", k_arena), ("v_arena", v_arena)):
            if arena is None and name == "v_arena":
                continue
            assert arena.dtype == torch.bfloat16 and arena.dim() == 5, (
                f"{name} must be a BF16 [slots, layers, capacity, kv_heads, head_dim] arena"
            )
            assert arena.stride(4) == 1 and arena.stride(3) == arena.shape[4], (
                f"{name}'s (kv_heads, head_dim) elements of a column must be contiguous"
            )
            assert arena.shape[2] >= col_limit, (
                f"{name} has {arena.shape[2]} columns, fewer than the column limit {col_limit}"
            )
        if v_arena is not None:
            assert v_arena.shape[:3] == k_arena.shape[:3], "k and v arenas differ in shape"
        for name, t in (("ctx_len", ctx_len), ("ctx_base", ctx_base)):
            assert t.dtype == torch.int64 and t.dim() == 1, f"{name} must be int64 [slots]"
            assert t.shape[0] == k_arena.shape[0], f"{name} must have one entry per arena slot"
        assert slots.dtype == torch.int64 and slots.dim() == 1, "slots must be int64 [gens]"
        assert num_accepted.dim() == 1 and num_accepted.numel() == slots.numel(), (
            "num_accepted must hold one count per request"
        )
        assert num_accepted.dtype in (torch.int32, torch.int64), "num_accepted must be int32/int64"
        assert (
            num_ctx_out.dtype == torch.int64
            and num_ctx_out.dim() == 1
            and num_ctx_out.numel() == slots.numel()
        ), "num_ctx_out must be int64 [gens]"
        assert slots.numel() <= MAX_REQUESTS, f"at most {MAX_REQUESTS} requests"
        assert 1 <= rows <= 64, f"rows must be in [1, 64], got {rows}"
        assert keep >= 0 and keep + rows <= k_arena.shape[2], (
            f"keep {keep} + rows {rows} must fit the arena's {k_arena.shape[2]} columns"
        )
        assert bool(((ctx_len[slots] - ctx_base[slots]) <= max(keep, col_limit)).all()), (
            "a slot holds more columns than the window keeps"
        )


context_kv_commit_windowed = ContextKvCommitWindowed()
