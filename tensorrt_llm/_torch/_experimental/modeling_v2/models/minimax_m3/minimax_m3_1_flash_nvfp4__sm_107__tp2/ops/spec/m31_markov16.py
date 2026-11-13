# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MARKOV16: the one-launch DSpark Markov chain (catalog ``spec_decode/dspark_markov_chain``) for
the TP2 16-row step, on this target's kernel (``kernels/m31_markov16``): every request's chain on each markov_w2 row
load, a caller-chosen grid (at TP2 198 CTAs x 512 rows instead of the catalog's 144 x 704), the streamed markov_w2
blocks held in registers, and the draft head's unpadded shard logits read directly (no pad copy). Bitwise the catalog
chain's tokens; it shares the catalog chain's exchange workspace (``dspark_markov_chain.workspace_args``) and its
buffer rotation, so calls of either kernel may alternate.

Compiled per (grid, rows, block, batch, fold, w2_regs) ahead of any call (``compile_only``) and outside CUDA-graph
capture. The call is a collective, as the catalog chain's: every rank of the group calls it with the same shape and
configuration, in the same order.
"""

from __future__ import annotations

import threading
from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2._native import namespace_of
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.dspark import (
    dspark_markov_chain as chain,
)

# This module's torch namespace (namespace_of): its Python custom ops register there.
_NS = namespace_of(__name__)

MAX_BATCH = 2  # capacity: fixed: kernels/m31_markov16.MAX_BATCH (the 16-row step's requests)
# CTAs per rank at most: every CTA must be co-resident (each takes a whole SM's shared memory), and the PDL dependent
# launched beside the chain takes an SM of its own. SM107 has 212 SMs.
GRID_CAP = 208  # capacity: fixed: an SM count bound with headroom for the PDL dependent

_lock = threading.Lock()
_compiled: Dict[tuple, object] = {}
_dummy: Dict[torch.device, torch.Tensor] = {}


def _kernel():
    """The CuTe DSL kernel module (imported on first use: it pulls in cutlass)."""
    from ...kernels import m31_markov16 as kern

    return kern


def _sm_count(device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def grid_decline(rows: int, grid: int, block: int, batch: int, device=None) -> Optional[str]:
    """Why ``grid`` CTAs cannot run a (``rows`` padded rows, ``block``, ``batch``) call, or None when they can."""
    kern = _kernel()
    sm = _sm_count(device if device is not None else torch.cuda.current_device())
    if grid > min(sm, GRID_CAP):
        return f"grid {grid} past min({sm} SMs, cap {GRID_CAP}): every CTA must be co-resident"
    if not kern.supports(rows, grid, block, batch):
        return (
            f"grid {grid} does not split {rows} rows into even 64-row multiples of at most "
            f"{kern.MAX_ROWS_PER_CTA} rows per CTA within shared memory at batch {batch}"
        )
    return None


def workspace_fits(block: int, batch: int, slots: int, grid: int, buf_words: int) -> bool:
    """Whether a call's exchange fits one buffer of the catalog workspace (sized for the catalog's MAX_GRID CTAs x 16
    requests x 8 positions: 2 requests fit up to GRID_CAP CTAs)."""
    return _kernel().buffer_words(block, batch, slots, grid) <= buf_words


def pick_grid(rows: int, block: int, batch: int, device=None) -> int:
    """The largest CTA count up to min(SM count, GRID_CAP) that runs the call (fewest rows per CTA); 0 if none."""
    sm = _sm_count(device if device is not None else torch.cuda.current_device())
    for grid in range(min(sm, GRID_CAP), 1, -1):
        if grid_decline(rows, grid, block, batch, device) is None:
            return grid
    return 0


def base_stride(base: torch.Tensor) -> Optional[int]:
    """The elements between consecutive (b, k) rows of ``base`` [B, K, C] when its rows are evenly spaced with
    contiguous columns (C for a contiguous ``base``; DHEAD_NOSLICE's [B, K, C] window of the MXFP8 head's padded
    [B x K, Np] output: Np), and its storage holds every row's full stride (the flat view the kernel reads stays inside
    the allocation); None otherwise."""
    if base.dim() != 3:
        return None
    batch, block, cols = base.shape
    if base.is_contiguous():
        return int(cols)
    s0, s1, s2 = base.stride()
    if s2 != 1 or s1 < cols or (batch > 1 and s0 != block * s1):
        return None
    need = (base.storage_offset() + batch * block * s1) * base.element_size()
    if base.untyped_storage().nbytes() < need:
        return None
    return int(s1)


def supports(
    base: torch.Tensor, w1: torch.Tensor, w2_shard: torch.Tensor, grid: int, base_cols: int
) -> bool:
    """Whether the kernel takes these operands: bf16 ``base`` [B <= MAX_BATCH, K, C], rows evenly spaced with
    contiguous columns (``base_stride``: contiguous, or DHEAD_NOSLICE's padded-head window), with C = the padded rows
    (``base_cols`` = C) or the unpadded shard (the fold: ``base_cols`` = C < rows; columns past C read as -inf whatever
    the row's stride holds there), ``w2_shard`` [rows, 256], and ``grid`` CTAs split the rows; a caller that gets False
    runs the catalog chain."""
    kern = _kernel()
    if (
        base.dim() != 3
        or base.dtype != torch.bfloat16
        or not base.is_cuda
        or base.data_ptr() % 16 != 0
    ):
        return False
    stride = base_stride(base)
    if stride is None:
        return False
    batch, block, cols = base.shape
    rows = w2_shard.shape[0] if w2_shard.dim() == 2 else -1
    return (
        w1.dtype == torch.bfloat16
        and w2_shard.dtype == torch.bfloat16
        and w1.dim() == 2
        and w1.shape[1] == kern.MARKOV_RANK
        and tuple(w2_shard.shape) == (rows, kern.MARKOV_RANK)
        and 0 < block <= kern.MAX_BLOCK
        and 0 < batch <= MAX_BATCH
        and 0 < base_cols == cols <= stride <= rows
        and grid_decline(rows, grid, block, batch, base.device) is None
    )


def _arg(t: torch.Tensor, align: int = 16):
    from cutlass.cute.runtime import from_dlpack

    return from_dlpack(t.detach(), assumed_align=align).mark_layout_dynamic(leading_dim=0)


def _launch(
    base: torch.Tensor,
    first_prev: torch.Tensor,
    w1: torch.Tensor,
    w2_shard: torch.Tensor,
    shard_offset: int,
    ws_uc: torch.Tensor,
    ws_mc: torch.Tensor,
    ws_flags: torch.Tensor,
    rank: int,
    slots: int,
    buf_words: int,
    grid: int,
    base_cols: int,
    w2_regs: bool,
    fp32_sum: bool,
    kv_lens: Optional[torch.Tensor] = None,
    rewind: Optional[torch.Tensor] = None,
    rewind_first: int = 0,
    compile_only: bool = False,
) -> Optional[torch.Tensor]:
    """One chain launch (``compile_only``: compile the call's kernel, launch nothing, return None); with ``kv_lens``
    and ``rewind``, also ``kv_lens[rewind_first + i] = max(kv_lens[...] - rewind[i], 0)`` after the grid wait."""
    import cuda.bindings.driver as cuda_driver
    import cutlass.cute as cute

    from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

    kern = _kernel()
    if not supports(base, w1, w2_shard, grid, base_cols):
        raise ValueError(f"m31_markov16: unsupported operands base {tuple(base.shape)} {base.dtype}, "
                         f"w2_shard {tuple(w2_shard.shape)}, grid {grid}, base_cols {base_cols}")  # fmt: skip
    if first_prev.dtype != torch.int64 or first_prev.numel() != base.shape[0]:
        raise ValueError("m31_markov16: first_prev must be int64 [B]")
    batch, block, cols = base.shape
    stride = base_stride(
        base
    )  # the kernel's base stride: cols for a contiguous base (supports() checked it)
    rows = w2_shard.shape[0]
    if not workspace_fits(block, batch, slots, grid, buf_words):
        raise ValueError("m31_markov16: the exchange does not fit the workspace buffers")
    device = base.device
    tokens = torch.empty(batch, block, dtype=torch.int32, device=device)
    next_new = torch.empty(batch, block + 1, dtype=torch.int32, device=device)
    dummy = _dummy.get(device)
    if dummy is None:
        # The kernel's optional outputs and inputs (corrected logits, accepted rows, KV-length rewind) are unused
        # here: one zero word stands for each, and an empty rewind leaves it untouched.
        dummy = _dummy[device] = torch.zeros(4, dtype=torch.int32, device=device)
    rewind_count = 0 if rewind is None else int(rewind.numel())
    if rewind_count:
        if not chain.supports_rewind(kv_lens, rewind, rewind_first) or rewind_count > MAX_BATCH:
            raise ValueError("m31_markov16: unsupported KV-length rewind (see supports_rewind())")
    else:
        kv_lens, rewind = dummy, dummy
    # The catalog chain's read-only accepted inputs (they only feed next_new, which the op discards).
    nacc, rows_idx, acc = chain._accepted_inputs(batch, block, device)
    # The flat base the kernel indexes at (b K + k) x stride + column: the rows' full strides (DHEAD_NOSLICE's padded
    # head output; the kernel reads only columns < base_cols of each row).
    flat_base = (
        base.reshape(-1) if stride == cols else base.as_strided((batch * block * stride,), (1,))
    )
    args = (
        _arg(flat_base.view(torch.int16)),
        _arg(first_prev.contiguous().view(-1), align=8),
        _arg(w1.contiguous().view(-1).view(torch.int32)),
        _arg(w2_shard.contiguous().view(-1).view(torch.int32)),
        _arg(dummy.view(torch.float32)),
        _arg(tokens.view(-1)),
        _arg(next_new.view(-1)),
        _arg(acc.view(-1), align=4),
        _arg(nacc, align=4),
        _arg(rows_idx, align=4),
        _arg(kv_lens, align=4),
        _arg(rewind, align=4),
        _arg(ws_uc.view(-1)),
        _arg(ws_mc.view(-1)),
        _arg(ws_flags.view(-1)),
    )
    scalars = (
        int(w1.shape[0]),
        int(shard_offset),
        int(rank),
        int(block + 1),
        int(rewind_first),
        rewind_count,
    )
    use_pdl = get_env_enable_pdl()
    stream = cuda_driver.CUstream(torch.cuda.current_stream(device).cuda_stream)
    # (grid, RC, S, base stride, base columns, K, B, slots, push copies, buffer words, bf16 base, fp32 sum, w2_regs)
    consts = (grid, rows // grid, rows, stride, int(base_cols), block, batch, slots, 1, buf_words, True,
              bool(fp32_sum), bool(w2_regs))  # fmt: skip
    key = consts + (use_pdl,)
    fn = _compiled.get(key)
    if fn is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"{_NS}::m31_tp2_markov16 must be compiled per shape outside CUDA-graph capture first"
            )
        with _lock:
            fn = _compiled.get(key)
            if fn is None:
                fn = _compiled[key] = cute.compile(
                    kern.m31_markov16, *args, *scalars, *consts, use_pdl, stream
                )
    if compile_only:
        return None
    fn(*args, *scalars, stream)
    return tokens


def is_compiled(rows: int, grid: int, cols: int, base_cols: int, block: int, batch: int, slots: int,
                buf_words: int, fp32_sum: bool, w2_regs: bool, use_pdl: bool) -> bool:  # fmt: skip
    """Whether the kernel of such a call is compiled in this process (``cols``: the base's row stride)."""
    key = (grid, rows // grid, rows, cols, base_cols, block, batch, slots, 1, buf_words, True, bool(fp32_sum),
           bool(w2_regs), bool(use_pdl))  # fmt: skip
    return key in _compiled


@torch.library.custom_op(f"{_NS}::m31_tp2_markov16", mutates_args=())
def _m31_tp2_markov16(
    base: torch.Tensor,
    first_prev: torch.Tensor,
    w1: torch.Tensor,
    w2_shard: torch.Tensor,
    shard_offset: int,
    ws_uc: torch.Tensor,
    ws_mc: torch.Tensor,
    ws_flags: torch.Tensor,
    rank: int,
    slots: int,
    buf_words: int,
    grid: int,
    base_cols: int,
    w2_regs: bool,
    fp32_sum: bool,
) -> torch.Tensor:
    return _launch(base, first_prev, w1, w2_shard, shard_offset, ws_uc, ws_mc, ws_flags, rank, slots, buf_words,
                   grid, base_cols, w2_regs, fp32_sum)  # fmt: skip


@_m31_tp2_markov16.register_fake
def _(base, first_prev, w1, w2_shard, shard_offset, ws_uc, ws_mc, ws_flags, rank, slots, buf_words, grid,
      base_cols, w2_regs, fp32_sum):  # fmt: skip
    return base.new_empty((base.shape[0], base.shape[1]), dtype=torch.int32)


@torch.library.custom_op(f"{_NS}::m31_tp2_markov16_rewind", mutates_args=("kv_lens",))
def _m31_tp2_markov16_rewind(
    base: torch.Tensor,
    first_prev: torch.Tensor,
    w1: torch.Tensor,
    w2_shard: torch.Tensor,
    shard_offset: int,
    ws_uc: torch.Tensor,
    ws_mc: torch.Tensor,
    ws_flags: torch.Tensor,
    rank: int,
    slots: int,
    buf_words: int,
    grid: int,
    base_cols: int,
    w2_regs: bool,
    fp32_sum: bool,
    kv_lens: torch.Tensor,
    rewind: torch.Tensor,
    rewind_first: int,
) -> torch.Tensor:
    return _launch(base, first_prev, w1, w2_shard, shard_offset, ws_uc, ws_mc, ws_flags, rank, slots, buf_words,
                   grid, base_cols, w2_regs, fp32_sum, kv_lens, rewind, rewind_first)  # fmt: skip


@_m31_tp2_markov16_rewind.register_fake
def _(base, first_prev, w1, w2_shard, shard_offset, ws_uc, ws_mc, ws_flags, rank, slots, buf_words, grid,
      base_cols, w2_regs, fp32_sum, kv_lens, rewind, rewind_first):  # fmt: skip
    return base.new_empty((base.shape[0], base.shape[1]), dtype=torch.int32)


def compile_call(
    base: torch.Tensor,
    first_prev: torch.Tensor,
    w1: torch.Tensor,
    w2_shard: torch.Tensor,
    shard_offset: int,
    *,
    ws_uc: torch.Tensor,
    ws_mc: torch.Tensor,
    ws_flags: torch.Tensor,
    rank: int,
    slots: int,
    buf_words: int,
    grid: int,
    base_cols: int,
    w2_regs: bool,
    fp32_sum: bool,
) -> None:
    """Compile (launch nothing) the kernel a call with these operands takes: rank-local, so a rank whose compile
    fails can tell its peers before any of them launches the collective (outside capture)."""
    _launch(base, first_prev, w1, w2_shard, shard_offset, ws_uc, ws_mc, ws_flags, rank, slots, buf_words, grid,
            base_cols, w2_regs, fp32_sum, compile_only=True)  # fmt: skip


def markov16_chain_rewind(
    base: torch.Tensor,
    first_prev: torch.Tensor,
    w1: torch.Tensor,
    w2_shard: torch.Tensor,
    shard_offset: int,
    kv_lens: torch.Tensor,
    rewind: torch.Tensor,
    rewind_first: int,
    *,
    ws_uc: torch.Tensor,
    ws_mc: torch.Tensor,
    ws_flags: torch.Tensor,
    rank: int,
    slots: int,
    buf_words: int,
    grid: int,
    base_cols: int,
    w2_regs: bool,
    fp32_sum: bool,
) -> torch.Tensor:
    """``m31_markov16``'s tokens plus the target's KV-length rewind after the draft forward in the same launch (the
    catalog ``dspark_markov_chain_rewind``'s contract; check ``dspark_markov_chain.supports_rewind`` first)."""
    return _m31_tp2_markov16_rewind(
        base, first_prev, w1, w2_shard, int(shard_offset), ws_uc, ws_mc, ws_flags, int(rank), int(slots),
        int(buf_words), int(grid), int(base_cols), bool(w2_regs), bool(fp32_sum), kv_lens, rewind, int(rewind_first),
    )  # fmt: skip


class M31Markov16Chain(OpWrapper):
    """Greedy draft tokens [B, K] (int32) of the DSpark vanilla-Markov head from this rank's vocab shard of the block
    logits, at 1..MAX_BATCH requests: the catalog ``dspark_markov_chain``'s tokens, bit for bit.

    ``base`` [B, K, C] bf16, contiguous (or rows evenly spaced with contiguous columns, ``base_stride``: DHEAD_NOSLICE's
    window of the MXFP8 head's padded output): this rank's shard logits, either in the catalog's padded buffer (C = the
    padded rows, padding columns -inf, ``base_cols`` = C) or unpadded straight from the draft head (C = the shard,
    ``base_cols`` = C: the columns past it read as -inf). ``first_prev`` [B] int64: the anchors. ``w1`` [V, 256] bf16.
    ``w2_shard`` [rows, 256] bf16: this rank's rows of markov_w2, zero-padded to the catalog's padded rows.
    ``shard_offset``: this rank's first vocabulary index. ``ws_*``, ``rank``, ``slots``, ``buf_words``: the catalog
    workspace (``dspark_markov_chain.workspace_args``). ``grid``: CTAs per rank (``pick_grid``). ``w2_regs``: hold the
    streamed markov_w2 blocks in registers. ``fp32_sum``: the catalog chain's ``enable_fp32_sum`` setting.

    The call is a collective: every rank of the group calls it with the same (B, K, grid, fold, w2_regs) in the same
    order (and may alternate with the catalog chain's calls on the same workspace).
    """

    ARCHS = frozenset({Arch.SM_107})

    # MiniMax-M3.1 TP2 on SM107 (212 SMs): vocabulary 200064, shard 100032 padded to the catalog's 101376 rows, K = 7.
    _M31_TP2 = dict(world=2, vocab=200064, block=7)

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="the shipped default: 198 CTAs x 512 rows, 2 streamed blocks per warp in registers, unpadded logits",
            spec=dict(**_M31_TP2, batches=(1, 2), grid=198, fold=True, w2_regs=True),
        ),
        Cell(
            why="the streamed blocks from global memory (w2_regs off) at the default grid",
            spec=dict(**_M31_TP2, batches=(2,), grid=198, fold=True, w2_regs=False),
        ),
        Cell(
            why="the catalog's geometry (144 x 704: 5 streamed blocks per warp, past REG_BLOCKS_MAX) on the padded "
            "buffer: the per-request layout alone",
            spec=dict(**_M31_TP2, batches=(2,), grid=144, fold=False, w2_regs=True),
        ),
        Cell(
            why="176 x 576 (3 streamed blocks, from global memory), unpadded logits",
            spec=dict(**_M31_TP2, batches=(2,), grid=176, fold=True, w2_regs=True),
        ),
    )

    note = """
    Greedy only, the vanilla Markov head (as the catalog chain). The kernel also writes next_new_tokens and applies
    the target's KV-length rewind (``markov16_chain_rewind``); this entry discards next_new and returns the tokens.
    Grids past the catalog's MAX_GRID (152) are taken: the catalog workspace is sized for 152 CTAs x 16 requests x
    8 positions per buffer, which holds 2 requests at up to GRID_CAP CTAs.
    """

    def __call__(
        self,
        base: torch.Tensor,
        first_prev: torch.Tensor,
        w1: torch.Tensor,
        w2_shard: torch.Tensor,
        shard_offset: int,
        ws_uc: torch.Tensor,
        ws_mc: torch.Tensor,
        ws_flags: torch.Tensor,
        rank: int,
        slots: int,
        buf_words: int,
        grid: int,
        base_cols: int,
        w2_regs: bool,
        fp32_sum: bool,
    ) -> torch.Tensor:
        return _m31_tp2_markov16(
            base,
            first_prev,
            w1,
            w2_shard,
            int(shard_offset),
            ws_uc,
            ws_mc,
            ws_flags,
            int(rank),
            int(slots),
            int(buf_words),
            int(grid),
            int(base_cols),
            bool(w2_regs),
            bool(fp32_sum),
        )

    def reference(
        self,
        base_in_rank_order: Sequence[torch.Tensor],
        first_prev: torch.Tensor,
        w1: torch.Tensor,
        w2_shard_in_rank_order: Sequence[torch.Tensor],
        shard: int,
        fp32_sum: bool = True,
    ) -> torch.Tensor:
        """The DSpark worker's in-tree chain on the gathered full-vocabulary logits (the catalog entry's reference):
        per position ``bias = F.linear(w1[prev], w2)`` in bf16, ``logits + bias`` (fp32 logits under fp32_sum),
        ``torch.argmax`` (first maximum). Takes every rank's ``base`` and ``w2_shard`` in rank order and ``shard``,
        the unpadded columns of each (the documented collective exception, as the catalog entry's)."""
        full = torch.cat([b[..., :shard] for b in base_in_rank_order], dim=-1)
        if fp32_sum:
            full = full.float()
        w2 = torch.cat([w[:shard] for w in w2_shard_in_rank_order], dim=0)
        prev = first_prev
        tokens = []
        for k in range(full.shape[1]):
            bias = F.linear(w1[prev], w2).to(full.dtype)
            prev = torch.argmax(full[:, k] + bias, dim=-1)
            tokens.append(prev)
        return torch.stack(tokens, 1).to(torch.int32)

    def is_valid(
        self,
        base: torch.Tensor,
        first_prev: torch.Tensor,
        w1: torch.Tensor,
        w2_shard: torch.Tensor,
        shard_offset: int,
        ws_uc: torch.Tensor,
        ws_mc: torch.Tensor,
        ws_flags: torch.Tensor,
        rank: int,
        slots: int,
        buf_words: int,
        grid: int,
        base_cols: int,
        w2_regs: bool,
        fp32_sum: bool,
    ) -> None:
        # As the catalog entry: a padding column that is not -inf can win and return an index past this rank's
        # shard, a wrong offset returns another rank's ids, an anchor outside the vocabulary reads w1 past its end.
        # Here also: base_cols must be base's width (a narrower value would hide live columns as -inf).
        vocab = w1.shape[0]
        assert vocab % slots == 0, (vocab, slots)
        shard = vocab // slots
        assert shard_offset == rank * shard, (shard_offset, rank, shard)
        assert base.shape[-1] == base_cols >= shard, (base.shape, base_cols, shard)
        assert bool(torch.isneginf(base[..., shard:]).all()), (
            "the padding columns of base must hold -inf"
        )
        assert bool(((first_prev >= 0) & (first_prev < vocab)).all()), (
            "first_prev outside [0, vocab)"
        )


m31_markov16 = M31Markov16Chain()
