# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DFlash prepare: the verify-dependent end of the context-K/V store, in one launch.

The built-in ``DFlashWorker.prepare_1st_drafter_inputs`` (``_torch/speculative/dflash.py``)
projects the ``K + 1`` captured target rows of every generation request into
the drafter's context K/V, multiplies the rows at and past the request's
accepted count by zero, stores all ``K + 1`` rows into the dense context arena
at columns ``min(ctx_len[slot] + r, max_ctx - 1)``, then advances
``ctx_len[slot]`` by the accepted count and clamps every slot's length to
``max_ctx``. The ``context_kv_fork`` fast path stores the rows unmasked earlier, on a side stream
that does not wait for the verify; this entry is what is left once the
accepted counts exist: the zeroing of the masked rows, in place in the arena,
the length update, and the built-in's read of the new lengths
(``_ctx_len[slots]``, the drafter's ``num_ctx_per_req``) into an output.

The zeroing multiplies each stored BF16 element by 0.0 in fp32 and rounds the
product back to BF16, as ``k_new.mul_(mask)`` does on a BF16 tensor, so the
arena ends up holding the built-in's bytes. Columns are computed from each
slot's length before any length is written, as the built-in's ``col_idx`` is
taken before its update: a CTA barrier separates every thread's reads of the
lengths from the first length write.

Over the windowed drafter context arena (``TRTLLM_DFLASH_WINDOWED_CTX_ARENA=1``)
a slot's physical column 0 holds its logical token ``ctx_base[slot]``, and the
built-in stores row ``r`` at column ``min(ctx_len[slot] - ctx_base[slot] + r,
capacity - 1)``. Given ``ctx_base`` and that last column, the entry zeroes those
columns instead; the lengths are updated as before (the built-in's window
compaction and its read of ``ctx_len - ctx_base`` follow the call, on the caller's
side). Without ``ctx_base`` the arithmetic is the dense arena's, unchanged.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, OpWrapper

#: Generation requests one call takes; the single program loops over them.
MAX_REQUESTS = 64

#: Elements of one zeroing tile ((row, layer) columns by a column's elements):
#: the whole of a request's columns in the served shape (8 rows x 5 layers x 128).
TILE_ELEMENTS = 8192


@triton.jit
def _columns(
    arena_ptr,
    slot,
    origin,
    accepted,
    last_col,
    slot_stride,
    layer_stride,
    col_stride,
    START: tl.constexpr,
    ROWS: tl.constexpr,
    LAYERS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    PAIRS: tl.constexpr,
):
    """Pointers to the (row, layer) columns ``START .. START + PAIRS - 1`` of a slot, and which are zeroed.

    Row ``r`` sits at column ``min(origin + r, last_col)``.
    """
    pair = START + tl.arange(0, PAIRS)
    row = pair // LAYERS
    layer = (pair % LAYERS).to(tl.int64)
    col = tl.minimum(origin + row, last_col)
    base = slot * slot_stride + layer * layer_stride + col * col_stride
    offsets = tl.arange(0, BLOCK)
    zeroed = (pair < ROWS * LAYERS) & (row >= accepted)
    pointers = arena_ptr + base[:, None] + offsets[None, :]
    return pointers, zeroed[:, None] & (offsets < WIDTH)[None, :]


@triton.jit
def _times_zero(x):
    return (x.to(tl.float32) * 0.0).to(tl.bfloat16)


@triton.jit
def _context_kv_commit_kernel(
    k_ptr,
    v_ptr,
    ctx_len_ptr,
    base_ptr,
    slots_ptr,
    accepted_ptr,
    new_len_ptr,
    num_ctx_ptr,
    num_gens,
    num_slots,
    max_ctx,
    last_col,
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
    HAS_BASE: tl.constexpr,
):
    # Every request's new length is computed from the lengths as they were
    # before this call (the built-in gathers ``ctx_len[slots]`` once, adds, and
    # scatters), so the zeroing and the new lengths read no length this call wrote.
    for g in range(num_gens):
        slot = tl.load(slots_ptr + g).to(tl.int64)
        ctx = tl.load(ctx_len_ptr + slot)
        # The physical column of the slot's next context token.
        origin = ctx
        if HAS_BASE:
            origin = ctx - tl.load(base_ptr + slot)
        accepted = tl.load(accepted_ptr + g).to(tl.int64)
        # A tile of K columns and its V columns are loaded before either is
        # stored, so their loads are in flight together.
        for start in tl.static_range(0, ROWS * LAYERS, PAIRS):
            k_cols, zeroed = _columns(
                k_ptr, slot, origin, accepted, last_col, k_slot_stride, k_layer_stride,
                k_col_stride, start, ROWS, LAYERS, WIDTH, BLOCK, PAIRS,
            )  # fmt: skip
            if HAS_V:
                v_cols, _ = _columns(
                    v_ptr, slot, origin, accepted, last_col, v_slot_stride, v_layer_stride,
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
    # The lengths are written by one thread below; every thread's reads of
    # them above (the zeroed columns' addresses) come first.
    tl.debug_barrier()
    # Scatter in request order, then clamp every slot, as the built-in's
    # ``ctx_len[slots] += n`` followed by ``ctx_len.clamp_(max=max_ctx)``.
    for g in range(num_gens):
        slot = tl.load(slots_ptr + g).to(tl.int64)
        tl.store(ctx_len_ptr + slot, tl.load(new_len_ptr + g))
    for s in range(num_slots):
        tl.store(ctx_len_ptr + s, tl.minimum(tl.load(ctx_len_ptr + s), max_ctx))
    # The built-in's read of the new lengths after its update (``_ctx_len[slots]``).
    for g in range(num_gens):
        slot = tl.load(slots_ptr + g).to(tl.int64)
        tl.store(num_ctx_ptr + g, tl.load(ctx_len_ptr + slot))


def tile_pairs(rows: int, layers: int, width: int) -> int:
    """(row, layer) columns per zeroing tile: all of them, up to ``TILE_ELEMENTS`` elements."""
    block = triton.next_power_of_2(width)
    return min(triton.next_power_of_2(rows * layers), max(1, TILE_ELEMENTS // block))


class ContextKvCommit(OpWrapper):
    """Zero the context-K/V rows at and past each request's accepted count, then advance its length.

    ``k_arena`` / ``v_arena`` are the dense context arenas ``[slots, layers,
    capacity, kv_heads, head_dim]`` (BF16; ``v_arena`` may be None); the ``rows``
    columns ``min(ctx_len[slot] + r, max_ctx - 1)``, ``r < rows``, of every
    request's slot hold that step's unmasked context K/V. ``ctx_len`` is int64
    ``[slots]``, ``slots`` int64 ``[gens]`` (one per generation request),
    ``num_accepted`` int32 or int64 ``[gens]``, ``num_ctx_out`` int64
    ``[gens]``. In place:

    * for each request ``g`` and row ``r >= num_accepted[g]``, in every layer of
      both arenas: the column's elements become ``bf16(fp32(x) * 0.0)``;
    * ``ctx_len[slots[g]] = ctx_len[slots[g]] + num_accepted[g]`` (gathered
      before any write, scattered in request order), then every slot's length
      is clamped to ``max_ctx``;
    * ``num_ctx_out[g] = ctx_len[slots[g]]`` (the new lengths).

    ``ctx_base`` (int64 ``[slots]``, the windowed arena's logical index of each
    slot's column 0) moves the rows to columns ``min(ctx_len[slot] -
    ctx_base[slot] + r, last_column)``; ``last_column`` defaults to ``max_ctx -
    1`` and must be a column of the arenas. ``ctx_base`` is read, never written.
    """

    ARCHS = frozenset({Arch.SM_103})

    note = """
    Metadata consumed: none. One Triton program (grid 1, 4 warps) that loops
    over the requests; per request it zeroes its masked (row, layer) columns
    of K and V as tiles of up to ``TILE_ELEMENTS`` elements (one tile in the
    served shape), loading a tile of both arenas before storing either, so the
    loads of a request's columns are in flight together rather than one
    column after another. Elements of a (row, layer) column are contiguous
    (``kv_heads * head_dim``, one block). A CTA barrier separates the
    zeroing, whose column addresses every thread computes from the lengths,
    from the length writes, which one thread issues.

    Call site: ``context_kv_fork.ContextKvFork.commit`` (from
    ``ContextKvForkDFlashWorker.prepare_1st_drafter_inputs``) under the
    ``context_kv_fork`` fast path, on the side stream that stored the unmasked
    rows, behind the main stream's last read of ``ctx_len`` in the prepare;
    the main stream joins before the drafter reads the arena and
    ``num_ctx_out``.

    Numerics: the zeroing is the BF16 product with 0.0 (finite x -> +-0 with
    x's sign, inf / NaN -> the canonical NaN), as the built-in's ``mul_`` by a
    BF16 mask; the lengths are integer arithmetic. Bitwise the built-in's
    arena and lengths when the arena held the unmasked rows. The built-in also
    multiplies the kept rows by 1.0, which leaves every value but a
    non-canonical NaN unchanged; the GPU kernels that compute the stored rows
    produce only the canonical NaN.

    The arena's end: rows whose column clamps share column ``max_ctx - 1``,
    and the built-in writes them with one ``index_put``, whose winner among
    duplicate indices is unspecified (a deterministic ``index_put`` keeps the
    last). The entry zeroes that column when any masked row maps to it; the
    masked rows are a suffix and the last row always maps there, so that is
    exactly when the last row is masked -- the last-duplicate-wins result,
    given a store that also left the last row there. A served request is not
    expected to reach the end while ``max_ctx`` is the KV cache manager's
    ``max_seq_len`` (inflated by the speculative slack over what is served);
    it can when the drafter's ``max_position_embeddings`` sets ``max_ctx``,
    which the built-in warns about when it sizes the arena.

    Hard limits
      Slots must be distinct across requests for a defined result, as for the
      built-in's ``index_put`` (CUDA-graph padding rows share the dummy slot,
      whose contents nothing reads).

    Not certified: torch.compile. CUDA-graph capture and replay are exercised
    by the test.
    """

    def __call__(
        self,
        k_arena: torch.Tensor,
        v_arena,
        ctx_len: torch.Tensor,
        slots: torch.Tensor,
        num_accepted: torch.Tensor,
        num_ctx_out: torch.Tensor,
        rows: int,
        max_ctx: int,
        ctx_base=None,
        last_column=None,
    ) -> None:
        num_gens = slots.numel()
        if num_gens == 0:
            return
        last = max_ctx - 1 if last_column is None else last_column
        layers, width = k_arena.shape[1], k_arena.shape[3] * k_arena.shape[4]
        new_len = torch.empty(num_gens, dtype=torch.int64, device=ctx_len.device)
        has_v = v_arena is not None
        values = v_arena if has_v else k_arena
        _context_kv_commit_kernel[(1,)](
            k_arena,
            values,
            ctx_len,
            ctx_len if ctx_base is None else ctx_base,
            slots,
            num_accepted,
            new_len,
            num_ctx_out,
            num_gens,
            ctx_len.numel(),
            max_ctx,
            last,
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
            HAS_BASE=ctx_base is not None,
            num_warps=4,
        )

    def reference(
        self,
        k_arena: torch.Tensor,
        v_arena,
        ctx_len: torch.Tensor,
        slots: torch.Tensor,
        num_accepted: torch.Tensor,
        num_ctx_out: torch.Tensor,
        rows: int,
        max_ctx: int,
        ctx_base=None,
        last_column=None,
    ) -> None:
        """The built-in torch ops, in place: mask the stored rows in row order, then update the lengths.

        One row at a time (slots are distinct, so no ``index_put`` below sees a
        duplicate index): where rows share the arena's last column, the masks
        of all of them multiply it, which zeroes it exactly when the last row
        is masked -- the last-duplicate-wins result of the built-in's one
        ``index_put`` of the masked rows.
        """
        offsets = torch.arange(rows, dtype=torch.long, device=ctx_len.device)
        origin = ctx_len[slots] if ctx_base is None else ctx_len[slots] - ctx_base[slots]
        last = max_ctx - 1 if last_column is None else last_column
        col = (origin.unsqueeze(1) + offsets.unsqueeze(0)).clamp(max=last)
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
        num_ctx_out.copy_(ctx_len[slots])

    def is_valid(
        self,
        k_arena: torch.Tensor,
        v_arena,
        ctx_len: torch.Tensor,
        slots: torch.Tensor,
        num_accepted: torch.Tensor,
        num_ctx_out: torch.Tensor,
        rows: int,
        max_ctx: int,
        ctx_base=None,
        last_column=None,
    ) -> None:
        last = max_ctx - 1 if last_column is None else last_column
        for name, arena in (("k_arena", k_arena), ("v_arena", v_arena)):
            if arena is None and name == "v_arena":
                continue
            assert arena.dtype == torch.bfloat16 and arena.dim() == 5, (
                f"{name} must be a BF16 [slots, layers, capacity, kv_heads, head_dim] arena"
            )
            assert arena.stride(4) == 1 and arena.stride(3) == arena.shape[4], (
                f"{name}'s (kv_heads, head_dim) elements of a column must be contiguous"
            )
            assert 0 <= last < arena.shape[2], (
                f"{name} has {arena.shape[2]} columns; the last column {last} is not one of them"
            )
        if v_arena is not None:
            assert v_arena.shape[:2] == k_arena.shape[:2], (
                "k and v arenas differ in slots or layers"
            )
        assert ctx_len.dtype == torch.int64 and ctx_len.dim() == 1, "ctx_len must be int64 [slots]"
        assert ctx_len.shape[0] == k_arena.shape[0], "ctx_len must have one length per arena slot"
        if ctx_base is not None:
            assert (
                ctx_base.dtype == torch.int64
                and ctx_base.shape == ctx_len.shape
                and ctx_base.is_contiguous()
            ), "ctx_base must be int64 [slots], one base per arena slot"
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


context_kv_commit = ContextKvCommit()
