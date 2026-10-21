# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DFlash windowed context arena: the per-step window compaction, in place, in two launches.

With the windowed drafter context arena (``TRTLLM_DFLASH_WINDOWED_CTX_ARENA=1``)
every generation slot keeps only the last ``keep`` context columns of its dense
arena ``[slots, layers, capacity, kv_heads, head_dim]`` (K, and V unless the
drafter stores one latent). After a decode step folds its accepted tokens in,
the built-in ``DFlashWorker._compact_ctx_window`` (``_torch/speculative/dflash.py``)
shifts each generation slot's columns left by
``shift = max(ctx_len[slot] - ctx_base[slot] - keep, 0)``::

    new[slot, l, c] = old[slot, l, min(c + shift, capacity - 1)]   for every c
    ctx_base[slot] += shift

as a fixed-shape gather of each slot's whole arena (``buf[slots]``), a
``gather`` along the columns and an ``index_put`` back, per buffer: three
passes over every generation slot's window with int64 index arithmetic, about
``2 x 3 x B x 5.3 MB`` per step at TP4. Generation rows that share a slot (the
CUDA-graph padding rows all map to one dummy slot) compute the same rows, so
their writes agree, and the base advances once.

This entry writes the same bytes with one read and one write of each window:

* ``_ctx_window_shift_kernel`` (one program): each generation row's shift,
  zero for a row whose slot an earlier row already owns, and the base advance
  of each owned slot -- before any column moves, so the move reads no base;
* ``_ctx_window_move_kernel``: one program per (generation row, layer and
  buffer, element block of a column) owns a column-strided slice of its slot's
  window and moves it left in place, tile by tile in increasing column order.
  A tile's loads (``min(c + shift, capacity - 1)``) all precede its stores
  (``c``) behind a CTA barrier; a later tile reads only columns past every
  column stored so far (``c + shift > c``), and the clamped last column is
  stored only by the last tile, after its own loads. Programs own disjoint
  slices, and rows with shift 0 (inside the window, or not owning their slot)
  return at once.

The result is the built-in's in every arena byte (the shifted window, the
dead tail and the clamped last column alike) and in every base.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, OpWrapper

#: Generation rows one call takes (the shift kernel compares every pair in one program).
MAX_GENS = 256
#: Elements of one column one move program owns at least (128 bytes of BF16): a column must
#: be a multiple of it.
BLOCK_ELEMENTS = 64
#: Move-kernel launch per number of generation rows: (rows up to, elements per program,
#: columns per tile, warps). Each program walks its slice of a window tile by tile, so a step
#: of few rows wants many narrow programs and a step of many rows fewer, wider ones.
MOVE_LAUNCHES = ((4, 64, 128, 4), (16, 128, 128, 2), (MAX_GENS, 128, 64, 2))
#: Operand dtypes the arena may hold.
SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)


@triton.jit
def _ctx_window_shift_kernel(
    slots_ptr,
    ctx_len_ptr,
    ctx_base_ptr,
    shift_ptr,
    num_gens,
    keep,
    GENS_BLOCK: tl.constexpr,
    LAUNCH_WITH_PDL: tl.constexpr,
):
    if LAUNCH_WITH_PDL:
        # The lengths are the previous kernels' output.
        tl.extra.cuda.gdc_wait()
    g = tl.arange(0, GENS_BLOCK)
    live = g < num_gens
    slot = tl.load(slots_ptr + g, mask=live, other=-1)
    # A row whose slot an earlier row holds does nothing: the slot's first row owns it.
    earlier = (slot[:, None] == slot[None, :]) & (g[None, :] < g[:, None]) & live[None, :]
    owned = live & (tl.max(earlier.to(tl.int32), axis=1) == 0)
    ctx = tl.load(ctx_len_ptr + slot, mask=live, other=0)
    base = tl.load(ctx_base_ptr + slot, mask=live, other=0)
    shift = tl.maximum(ctx - base - keep, 0)
    tl.store(shift_ptr + g, tl.where(owned, shift, 0), mask=live)
    tl.store(ctx_base_ptr + slot, base + shift, mask=owned)


@triton.jit
def _ctx_window_move_kernel(
    k_ptr,
    v_ptr,
    slots_ptr,
    shift_ptr,
    slot_stride,
    layer_stride,
    capacity,
    LAYERS: tl.constexpr,
    COLUMN: tl.constexpr,
    BLOCK: tl.constexpr,
    TILE: tl.constexpr,
    LAUNCH_WITH_PDL: tl.constexpr,
):
    if LAUNCH_WITH_PDL:
        # The shifts are the shift kernel's output; the arena the step's store's.
        tl.extra.cuda.gdc_wait()
    row = tl.program_id(0)
    layer_buffer = tl.program_id(1)
    part = tl.program_id(2)
    shift = tl.load(shift_ptr + row)
    if shift > 0:
        slot = tl.load(slots_ptr + row).to(tl.int64)
        offset = slot * slot_stride + (layer_buffer % LAYERS) * layer_stride + part * BLOCK
        if layer_buffer < LAYERS:
            window = k_ptr + offset
        else:
            window = v_ptr + offset
        e = tl.arange(0, BLOCK)
        for c0 in range(0, capacity, TILE):
            cols = c0 + tl.arange(0, TILE)
            inside = (cols < capacity)[:, None]
            src = tl.minimum(cols + shift, capacity - 1).to(tl.int64)
            values = tl.load(window + src[:, None] * COLUMN + e[None, :], mask=inside)
            # Every thread's loads of this tile before any store of it.
            tl.debug_barrier()
            tl.store(window + cols.to(tl.int64)[:, None] * COLUMN + e[None, :], values, mask=inside)


class CtxWindowCompact(OpWrapper):
    """``DFlashWorker._compact_ctx_window`` on the arena, the lengths and the bases, in place.

    With ``k_arena`` (and ``v_arena``, or None) ``[slots, L, C, Hkv, D]``
    contiguous, ``ctx_len`` and ``ctx_base`` int64 ``[slots]`` and ``slots``
    int64 ``[G]`` (the generation rows' slots, duplicates allowed)::

        for each distinct s in slots:
            shift = max(ctx_len[s] - ctx_base[s] - keep, 0)
            arena[s, l, c] = old arena[s, l, min(c + shift, C - 1)]   for all l, c
            ctx_base[s] += shift

    ``ctx_len`` is read only. ``enable_pdl`` launches both kernels with
    programmatic dependent launch (each waits before its first global read and
    triggers its dependents only by completing).
    """

    ARCHS = frozenset({Arch.SM_103})

    note = """
    Metadata consumed: the generation rows' slots, every touched slot's
    ctx_len and ctx_base (device), the window size keep. Call site: the DFlash
    worker's _compact_ctx_window on every decode step with the windowed dense
    arena (fast path drafter_window_compact): after the step's context-K/V
    store and length update, before the drafter reads the arena. Shift kernel
    grid (1,), move kernel grid (rows, layers x buffers, column / 64 or
    column / 128 elements) with its tile and warps per row count (MOVE_LAUNCHES).

    Bit-identical, gated by its own tests: every arena byte (shifted window,
    dead tail, clamped column, untouched slots) and every base against the
    built-in DFlashWorker._compact_ctx_window on the same arena, including
    rows sharing a dummy slot, rows inside the window, shifts larger than a
    tile and the capacity clamp; eager and replayed from a captured graph.

    Loud preconditions (ValueError before any launch): contiguous 5-D BF16 or
    FP16 arena(s) of one shape whose column (kv_heads x head_dim elements) is
    a multiple of 64 elements, int64 1-D ctx_len and ctx_base over the
    arena's slots, int64 1-D slots of at most 256 rows, keep >= 0.

    Not certified: slot ids outside the arena (read as given), a shift kernel
    not followed by its move kernel.
    """

    def __call__(
        self,
        k_arena: torch.Tensor,
        v_arena: Optional[torch.Tensor],
        ctx_len: torch.Tensor,
        ctx_base: torch.Tensor,
        slots: torch.Tensor,
        keep: int,
        enable_pdl: bool = False,
    ) -> None:
        reason = unsupported_reason(k_arena, v_arena, ctx_len, ctx_base, slots, keep)
        if reason is not None:
            raise ValueError(f"ctx_window_compact: {reason}")
        gens = slots.shape[0]
        if gens == 0:
            return
        _, layers, capacity, kv_heads, head_dim = k_arena.shape
        column = kv_heads * head_dim
        shift = torch.empty(gens, dtype=torch.int64, device=k_arena.device)
        _ctx_window_shift_kernel[(1,)](
            slots,
            ctx_len,
            ctx_base,
            shift,
            gens,
            keep,
            GENS_BLOCK=max(16, triton.next_power_of_2(gens)),
            LAUNCH_WITH_PDL=enable_pdl,
            num_warps=4,
            launch_pdl=enable_pdl,
        )
        buffers = 1 if v_arena is None else 2
        block, tile, warps = _move_launch(gens, column)
        _ctx_window_move_kernel[(gens, layers * buffers, column // block)](
            k_arena,
            k_arena if v_arena is None else v_arena,
            slots,
            shift,
            k_arena.stride(0),
            k_arena.stride(1),
            capacity,
            LAYERS=layers,
            COLUMN=column,
            BLOCK=block,
            TILE=tile,
            LAUNCH_WITH_PDL=enable_pdl,
            num_warps=warps,
            num_stages=1,
            launch_pdl=enable_pdl,
        )

    def reference(
        self,
        k_arena: torch.Tensor,
        v_arena: Optional[torch.Tensor],
        ctx_len: torch.Tensor,
        ctx_base: torch.Tensor,
        slots: torch.Tensor,
        keep: int,
        enable_pdl: bool = False,
    ) -> None:
        """The built-in ``_compact_ctx_window``, statement for statement, in place.

        ``enable_pdl`` changes how the op is launched and is ignored here.
        """
        phys = ctx_len[slots] - ctx_base[slots]
        shift = (phys - keep).clamp_(min=0)
        capacity = k_arena.size(2)
        cols = torch.arange(capacity, dtype=torch.long, device=slots.device)
        src = (cols.unsqueeze(0) + shift.unsqueeze(1)).clamp_(max=capacity - 1)
        for buf in (k_arena, v_arena):
            if buf is None:
                continue
            rows = buf[slots]
            index = src.view(src.size(0), 1, capacity, 1, 1).expand_as(rows)
            buf[slots] = rows.gather(2, index)
        ctx_base[slots] += shift

    def is_valid(
        self,
        k_arena: torch.Tensor,
        v_arena: Optional[torch.Tensor],
        ctx_len: torch.Tensor,
        ctx_base: torch.Tensor,
        slots: torch.Tensor,
        keep: int,
        enable_pdl: bool = False,
    ) -> None:
        """A slot id outside the arena is read and written as given, with no error."""
        if slots.numel() and (int(slots.min()) < 0 or int(slots.max()) >= k_arena.shape[0]):
            raise ValueError(
                f"slots [{int(slots.min())}, {int(slots.max())}] outside the "
                f"{k_arena.shape[0]}-slot arena"
            )


def _move_launch(gens: int, column: int) -> tuple:
    """(elements per program, columns per tile, warps) of the move kernel for ``gens`` rows."""
    for rows, block, tile, warps in MOVE_LAUNCHES:
        if gens <= rows:
            if column % block:
                block = BLOCK_ELEMENTS
            return block, tile, warps
    raise ValueError(f"ctx_window_compact takes at most {MAX_GENS} generation rows, got {gens}")


def unsupported_reason(
    k_arena: torch.Tensor,
    v_arena: Optional[torch.Tensor],
    ctx_len: torch.Tensor,
    ctx_base: Optional[torch.Tensor],
    slots: torch.Tensor,
    keep: int,
) -> Optional[str]:
    """Why ``ctx_window_compact`` does not take these operands, or None. Host-side reads only."""
    checks = (
        (
            not isinstance(k_arena, torch.Tensor)
            or k_arena.dim() != 5
            or k_arena.dtype not in SUPPORTED_DTYPES
            or not k_arena.is_contiguous()
            or not k_arena.is_cuda,
            "the arena is not a contiguous 5-D BF16/FP16 CUDA tensor",
        ),
        (
            v_arena is not None
            and (
                v_arena.shape != k_arena.shape
                or v_arena.dtype != k_arena.dtype
                or not v_arena.is_contiguous()
            ),
            "the V arena is not the K arena's twin",
        ),
        (
            isinstance(k_arena, torch.Tensor)
            and k_arena.dim() == 5
            and (k_arena.shape[3] * k_arena.shape[4]) % BLOCK_ELEMENTS != 0,
            f"a column is not a multiple of {BLOCK_ELEMENTS} elements",
        ),
        (
            ctx_base is None
            or ctx_len.dtype != torch.int64
            or ctx_base.dtype != torch.int64
            or ctx_len.dim() != 1
            or ctx_base.dim() != 1
            or not ctx_len.is_contiguous()
            or not ctx_base.is_contiguous()
            or isinstance(k_arena, torch.Tensor)
            and (ctx_len.shape[0] < k_arena.shape[0] or ctx_base.shape[0] < k_arena.shape[0]),
            "the lengths are not int64 per-slot vectors",
        ),
        (
            slots.dtype != torch.int64
            or slots.dim() != 1
            or not slots.is_contiguous()
            or slots.shape[0] > MAX_GENS,
            f"the slots are not an int64 vector of at most {MAX_GENS} rows",
        ),
        (keep < 0, "a negative window"),
    )
    for failed, reason in checks:
        if failed:
            return reason
    return None


ctx_window_compact = CtxWindowCompact()
