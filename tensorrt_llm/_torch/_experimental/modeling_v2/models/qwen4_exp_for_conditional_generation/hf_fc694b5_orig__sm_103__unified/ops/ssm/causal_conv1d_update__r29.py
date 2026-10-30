# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-step causal depthwise conv1d against a slot-indexed conv-state pool.

The decode step's shape (one token a row, SiLU, no bias, a width-4 conv over a
3-column bf16 pool, slot indices, a channel count that is a multiple of 1024)
from `_CUTE_MIN_ROWS` rows runs a CuTe DSL kernel (written for the tp1 target)
that is bitwise the TRT-LLM kernel it replaces there; every other call is
`torch.ops.trtllm.causal_conv1d_update`.
"""

from __future__ import annotations

from typing import Optional

import cutlass
import cutlass.cute as cute
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.kernels.blackwell.utils import make_ptr

#: This checkpoint's GDN `[q|k|v]` width; the conv is width 4 over a 3-column pool.
CKPT_DIM = 10240
#: Largest conv-pool slot count the cache manager was seen to allocate for this
#: checkpoint (10 .. 461 over twelve runtime-value dumps, batch never above 32).
#: The decode step hands the whole pool over unsliced.
ENGINE_SLOTS = 461

#: A tp4/tep4 rank's GDN conv width (`[q | k | v]` of 4 key and 12 value heads of 128) and
#: the row stride of its input-projection output, `[q | k | v | z | a | b]` = 2560 + 1536 +
#: 12 + 12; at tp1 and dep4 the projection is 16480 wide.
RANK_DIM = 2560
RANK_X_STRIDE = 4120
CKPT_X_STRIDE = 16480
#: Per-rank conv-pool slot counts of the stock engine at core's config (bf16 KV and
#: recurrent state, KV fraction 0.6, `max_batch_size` 128, `max_seq_len` 9216), read off
#: the V2 hybrid manager's "PLE state views bound" line on all four ranks of tep4 and dep4
#: boots (the PLE and GDN pools share one slot space). The count is memory-derived, so a
#: dep4 rank, which keeps every GDN head, holds fewer than a tep4 rank. No engine has run
#: tp4 (the stock MoE refuses its 160-row expert slice); its rank holds tep4's conv geometry,
#: so tep4's count stands in for it.
TEP4_SLOTS = 1874
DEP4_SLOTS = 850
TP4_SLOTS = TEP4_SLOTS
#: GDN layers whose conv states share one slot's page in the V2 manager: a layer's view
#: `[slots, dim, 3]` has slot stride 36 * dim * 3 (276480 at 2560, 1105920 at 10240) and
#: layer `i` sits `i * dim * 3` elements into the page (measured on the stock manager at
#: tp4, tep4 and dep4).
GDN_PAGE_LAYERS = 36


# --------------------------------------------------------------------------------------
# The decode shape in CuTe DSL. TRT-LLM's seqlen-1 kernel runs a thread per (row,
# channel) with 2-byte accesses -- at 128 rows 1.3 M threads, four waves of one load
# latency each (7.5 us a layer on GB300). Here a thread owns 8 channels of one row: the
# weights (64 B), its state (8 channels x 3 columns, 48 contiguous bytes) and x (16 B)
# by 16-byte loads, then the TRT-LLM kernel's arithmetic instruction for instruction
# (`__fmaf_rn` over the taps from 0, `v * __frcp_rn(1 + __expf(-v))`, round to bf16),
# x written in place and the state shifted by the new token.
# --------------------------------------------------------------------------------------

#: Rows from which the decode shape runs the CuTe kernel (GB300, graph replay, 8
#: rotating layers, us: 128 rows 7.46 -> 5.41, 121 rows 7.40 -> 5.13; at 33 rows
#: and below the TRT-LLM kernel's one wave is faster: 3.10 vs 3.39).
_CUTE_MIN_ROWS = 64
_CH = 8
_THREADS = 128


def _iv(x, loc, ip):
    return x.ir_value(loc=loc, ip=ip)


def _asm(ret, args, text, constraints, loc, ip, side_effects=False):
    return llvm.inline_asm(
        ret,
        [_iv(a, loc, ip) for a in args],
        text,
        constraints,
        has_side_effects=side_effects,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _asm4(args, text, constraints, loc, ip, side_effects=False):
    ty = T.i32()
    res = _asm(
        llvm.StructType.get_literal([ty] * 4), args, text, constraints, loc, ip, side_effects
    )
    return [cutlass.Int32(llvm.extractvalue(ty, res, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def _ldg_v4(addr, *, loc=None, ip=None):
    """16 bytes; volatile so that no load moves across the pool's in-place update."""
    return _asm4(
        [cutlass.Int64(addr)],
        "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def _ldg_nc_v4(addr, *, loc=None, ip=None):
    """16 bytes of a read-only operand (the weights), non-coherent path."""
    return _asm4(
        [cutlass.Int64(addr)],
        "ld.global.nc.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        loc,
        ip,
    )


@dsl_user_op
def _stg_v4(addr, a, b, c, d, *, loc=None, ip=None):
    _asm(
        T.i32(),
        [cutlass.Int64(addr)] + [cutlass.Int32(v) for v in (a, b, c, d)],
        "{ st.global.v4.b32 [$1], {$2, $3, $4, $5}; mov.u32 $0, 0; }",
        "=r,l,r,r,r,r",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def _bf16_lo(w, *, loc=None, ip=None):
    """The low bf16 of a word as fp32 (exact)."""
    return cutlass.Float32(
        _asm(
            T.f32(),
            [cutlass.Int32(w)],
            "{ .reg .b32 t; shl.b32 t, $1, 16; mov.b32 $0, t; }",
            "=f,r",
            loc,
            ip,
        )
    )


@dsl_user_op
def _bf16_hi(w, *, loc=None, ip=None):
    """The high bf16 of a word as fp32 (exact)."""
    return cutlass.Float32(
        _asm(
            T.f32(),
            [cutlass.Int32(w)],
            "{ .reg .b32 t; and.b32 t, $1, 0xFFFF0000; mov.b32 $0, t; }",
            "=f,r",
            loc,
            ip,
        )
    )


@dsl_user_op
def _bf16x2(lo, hi, *, loc=None, ip=None):
    """`cvt.rn.bf16x2.f32`: `lo` in the low half (exact for values already in bf16)."""
    return cutlass.Int32(
        _asm(
            T.i32(),
            [cutlass.Float32(hi), cutlass.Float32(lo)],
            "cvt.rn.bf16x2.f32 $0, $1, $2;",
            "=r,f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def _conv_silu(w0, w1, w2, w3, s0, s1, s2, x, *, loc=None, ip=None):
    """`__fmaf_rn(w[i], s[i], v)` over the four taps from 0, then `v * __frcp_rn(1 +
    __expf(-v))` (`__expf` is `ex2.approx.ftz` of the `mul.ftz` by log2e)."""
    return cutlass.Float32(
        _asm(
            T.f32(),
            [cutlass.Float32(v) for v in (w0, w1, w2, w3, s0, s1, s2, x)],
            "{ .reg .f32 v, t, e; fma.rn.f32 v, $1, $5, 0f00000000; fma.rn.f32 v, $2, $6, v; "
            "fma.rn.f32 v, $3, $7, v; fma.rn.f32 v, $4, $8, v; neg.f32 t, v; "
            "mul.ftz.f32 t, t, 0f3FB8AA3B; ex2.approx.ftz.f32 e, t; add.f32 e, e, 0f3F800000; "
            "rcp.rn.f32 e, e; mul.rn.f32 $0, v, e; }",
            "=f,f,f,f,f,f,f,f,f",
            loc,
            ip,
        )
    )


class _DecodeConvKernel:
    """One token a row: `x[row, c] = silu(conv)` in place and the row's slot shifted."""

    def __init__(self, dim: int, x_stride: int):
        assert dim % (_CH * _THREADS) == 0
        self.dim, self.sx = dim, x_stride

    @cute.jit
    def __call__(
        self,
        x_t: cute.Tensor,
        w_ptr: cute.Pointer,
        st_ptr: cute.Pointer,
        slot_ptr: cute.Pointer,
        rows: cutlass.Int32,
        slot_stride: cutlass.Int32,
        pad_slot_id: cutlass.Int32,
        stream,
    ):
        slots = cute.make_tensor(slot_ptr, cute.make_layout((rows,), stride=(1,)))
        # `x` arrives as a tensor: the TVM-FFI entry takes its stream from a GPU tensor.
        self.kernel(
            x_t.iterator.toint(), w_ptr.toint(), st_ptr.toint(), slots, slot_stride, pad_slot_id
        ).launch(
            grid=[rows, self.dim // (_CH * _THREADS), 1], block=[_THREADS, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        x_base: cutlass.Int64,
        w_base: cutlass.Int64,
        st_base: cutlass.Int64,
        mSlot: cute.Tensor,
        slot_stride: cutlass.Int32,
        pad_slot_id: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        row, cb, _ = cute.arch.block_idx()
        c0 = (cb * _THREADS + tidx) * _CH
        # The weights: 8 channels x 4 taps, 32 contiguous bf16.
        wv = []
        for q in cutlass.range_constexpr(4):
            wv += _ldg_nc_v4(w_base + cutlass.Int64(c0 * 4 + q * 8) * 2)
        slot = mSlot[row]
        if slot != pad_slot_id:
            sb = st_base + (cutlass.Int64(slot) * slot_stride + c0 * 3) * 2
            sv = []
            for q in cutlass.range_constexpr(3):
                sv += _ldg_v4(sb + q * 16)
            xa = x_base + (cutlass.Int64(row) * self.sx + c0) * 2
            xv = _ldg_v4(xa)

            def elem(words, e):
                return _bf16_lo(words[e // 2]) if e % 2 == 0 else _bf16_hi(words[e // 2])

            outs, shifted = [], []
            for ch in cutlass.range_constexpr(_CH):
                w = [elem(wv, ch * 4 + k) for k in range(4)]
                st = [elem(sv, ch * 3 + k) for k in range(3)]
                xc = elem(xv, ch)
                outs.append(_conv_silu(*w, *st, xc))
                shifted += [st[1], st[2], xc]
            _stg_v4(xa, *[_bf16x2(outs[2 * i], outs[2 * i + 1]) for i in range(4)])
            for q in cutlass.range_constexpr(3):
                _stg_v4(
                    sb + q * 16,
                    *[
                        _bf16x2(shifted[8 * q + 2 * i], shifted[8 * q + 2 * i + 1])
                        for i in range(4)
                    ],
                )


_DECODE_CONV: dict = {}


def _decode_conv_ok(x, conv_state, weight, bias, silu_activation, cache_seqlens, indices) -> bool:
    """The decode shape this kernel certifies; anything else runs the TRT-LLM op."""
    return (
        x.dim() == 2
        and x.shape[0] >= _CUTE_MIN_ROWS
        and bias is None
        and silu_activation
        and cache_seqlens is None
        and indices is not None
        and indices.dtype == torch.int32
        and indices.is_contiguous()
        and indices.shape == (x.shape[0],)
        and x.dtype == conv_state.dtype == weight.dtype == torch.bfloat16
        and weight.shape == (x.shape[1], 4)
        and weight.is_contiguous()
        and conv_state.dim() == 3
        and conv_state.shape[1:] == (x.shape[1], 3)
        and conv_state.stride()[1:] == (3, 1)
        and x.shape[1] % (_CH * _THREADS) == 0
        and x.stride(1) == 1
        and (x.stride(0) * 2) % 16 == 0
        and (conv_state.stride(0) * 2) % 16 == 0
        and x.data_ptr() % 16 == 0
        and conv_state.data_ptr() % 16 == 0
        and weight.data_ptr() % 16 == 0
    )


def _decode_conv(x, conv_state, weight, indices, pad_slot_id) -> None:
    dim = x.shape[1]
    key = (dim, x.stride(0))
    args = (x, weight.data_ptr(), conv_state.data_ptr(), indices.data_ptr())
    fn = _DECODE_CONV.get(key)
    if fn is None:
        gm = cute.AddressSpace.gmem
        cargs = (
            from_dlpack(x, assumed_align=16).mark_layout_dynamic(leading_dim=1),
            make_ptr(cutlass.BFloat16, args[1], gm, assumed_align=16),
            make_ptr(cutlass.BFloat16, args[2], gm, assumed_align=16),
            make_ptr(cutlass.Int32, args[3], gm, assumed_align=4),
        )
        fn = cute.compile(
            _DecodeConvKernel(dim, x.stride(0)),
            *cargs,
            cutlass.Int32(1),
            cutlass.Int32(1),
            cutlass.Int32(-1),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--opt-level 2 --enable-tvm-ffi",
        )
        _DECODE_CONV[key] = fn
    fn(*args, x.shape[0], conv_state.stride(0), pad_slot_id)


def _pinned(batch: int) -> tuple[int, ...]:
    """`batch` distinct slots of the engine pool, slot 0 and the last pinned in."""
    return (0,) + tuple(i * 14 + 3 for i in range(1, batch - 1)) + (ENGINE_SLOTS - 1,)


def _spread(batch: int, pool_slots: int, pad_every: int = 0) -> tuple[int, ...]:
    """`batch` distinct slots of a `pool_slots` pool: the top slot (the page's highest
    addresses) and slot 0 first, then a stride-37 walk over the interior (37 is coprime to
    every `pool_slots - 2` here), so batch row `b` is not slot `b`. With `pad_every`, rows
    `5, 5 + pad_every, ...` take the pad slot -1 instead."""
    rest = tuple((i * 37 + 4) % (pool_slots - 2) + 1 for i in range(max(batch - 2, 0)))
    slots = ((pool_slots - 1, 0) + rest)[:batch]
    assert len(set(slots)) == batch
    if pad_every:
        slots = tuple(-1 if i % pad_every == 5 else s for i, s in enumerate(slots))
    return slots


def _rank(batch: int, pool_slots: int, pad_every: int = 0) -> dict:
    """A tp4/tep4 rank's call: 2560 conv channels at x row stride 4120, the layer's view the
    last of the manager's 36-layer conv page."""
    return dict(
        dim=RANK_DIM,
        x_row_stride=RANK_X_STRIDE,
        batch=batch,
        pool_slots=pool_slots,
        slots=_spread(batch, pool_slots, pad_every),
        pool_layers=GDN_PAGE_LAYERS,
    )


def _dep4(batch: int) -> dict:
    """A dep4 rank's call: tp1's 10240 channels at row stride 16480, the layer's view the
    last of the manager's 36-layer conv page."""
    return dict(
        dim=CKPT_DIM,
        x_row_stride=CKPT_X_STRIDE,
        batch=batch,
        pool_slots=DEP4_SLOTS,
        slots=_spread(batch, DEP4_SLOTS),
        pool_layers=GDN_PAGE_LAYERS,
    )


class CausalConv1dUpdate(OpWrapper):
    """One causal depthwise conv step per batch row against that row's cached
    window, then the window's advance -- `x` and the pool both written in place.

    Row `b`, slot `s = conv_state_indices[b]` (or `b`), new tokens `i < seqlen`:
    `history = [conv_state[s, c, :], x[b, c, :]]`, the window is its last
    `W - 1 + seqlen` columns, `x[b, c, i] = act(bias[c] + sum_j weight[c, j] *
    window[i + j])` in fp32 rounded once, and `conv_state[s] = history[-state_len:]`
    with `state_len = conv_state.shape[2]`. With `cache_seqlens` the pool is a
    ring instead: position `p` lives at `p % state_len`, `cache_seqlens[b]` is
    the first new token's absolute position, and nothing shifts.

    Fusion boundary: window read, convolution, bias, SiLU and state write are
    inside. The projection producing `x`, slot allocation and the q/k/v split
    are the caller's.

    Returns `x` -- the same object, written in place. A 2-D `[batch, dim]` `x`
    is handed to the op as an `unsqueeze(-1)` view (metadata only; the op
    writes through it). `conv_state` is advanced in place and not returned.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec keys: dim, x_row_stride (default 16480), batch, pool_slots, slots
    # ("random" | tuple), pool_layers (the pool the last layer's view of a buffer
    # this many layers deep, default 1). Every cell is the call site: x the
    # q|k|v column view of the decode rows of a [rows, x_row_stride] in_proj
    # output, bf16, W 4, a 3-column pool, no bias, SiLU, no ring, pad_slot_id -1.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="x the q|k|v column view of the [rows, 16480] in_proj, batch 32",
            spec=dict(dim=CKPT_DIM, batch=32, pool_slots=ENGINE_SLOTS, slots=_pinned(32)),
        ),
        Cell(
            why="x the q|k|v column view of the [rows, 16480] in_proj, batch 128",
            spec=dict(dim=CKPT_DIM, batch=128, pool_slots=256, slots="random"),
        ),
        Cell(
            why="64 rows (the CuTe kernel's first) over a layer view, pad -1 rows untouched",
            spec=dict(
                dim=CKPT_DIM,
                batch=64,
                pool_slots=ENGINE_SLOTS,
                slots=tuple(-1 if i % 21 == 5 else i * 7 for i in range(64)),
                pool_layers=2,
            ),
        ),
        # This target's members, each over its rank's measured conv page. Every
        # 2560-channel call runs the TRT-LLM kernel (2560 is not a multiple of the
        # CuTe kernel's 1024 channels a block row).
        Cell(
            why="tep4 decode, 1 row at 2560 / stride 4120: the top slot of the 1874-slot page",
            spec=_rank(1, TEP4_SLOTS),
        ),
        Cell(
            why="tep4, 64 rows at 2560 / stride 4120 over the 1874-slot page",
            spec=_rank(64, TEP4_SLOTS),
        ),
        Cell(
            why="tep4, 128 rows (max_batch_size) at 2560 / stride 4120 over the 1874-slot page",
            spec=_rank(128, TEP4_SLOTS),
        ),
        Cell(
            why="tp4, 128 rows at 2560 / stride 4120 over tp4's page, pad rows (-1) on the "
            "TRT-LLM kernel",
            spec=_rank(128, TP4_SLOTS, pad_every=21),
        ),
        Cell(
            why="dep4 decode, 32 rows at 10240 / stride 16480: the top slot of the 850-slot page",
            spec=_dep4(32),
        ),
        Cell(
            why="dep4, 128 rows (max_batch_size, the CuTe kernel) over the 850-slot page",
            spec=_dep4(128),
        ),
    )

    note = """
    Metadata consumed: none.

    Call site: the GDN decode step -- 2-D `[num_decodes, 10240]` x, the q|k|v
    column view (row stride 16480) of the decode rows of the layer's one
    `[T, 16480]` input-projection output (the op writes those columns only);
    the layer's whole conv pool `[slots, 10240, 3]`, W 4, no bias, SiLU on,
    `cache_seqlens` None, int32 slots, `pad_slot_id` -1.
    That exact shape (silu, no ring, `state_len == W - 1`) runs TRT-LLM's
    dedicated seqlen-1 kernel; any other shape runs its general one, which the
    target never reaches and which is not certified. From `_CUTE_MIN_ROWS`
    (64) rows, with bf16 operands, a contiguous weight, a pool whose slot is
    `[dim, 3]` contiguous and 16-byte aligned rows, it runs
    `_DecodeConvKernel` (CuTe DSL) instead: 8 channels a thread by 16-byte
    loads, the TRT-LLM kernel's instructions, bitwise its result (x and pool,
    pad rows untouched); 128 rows 7.46 -> 5.41 us a layer (GB300, graph
    replay over 8 rotating layers). Certified at that call: 32 rows (the
    TRT-LLM kernel) over the 461-slot pool, 128 rows (the CuTe kernel) over a
    256-slot pool, and 64 rows (the CuTe kernel's first row count) with pad
    rows over the 461-slot pool seen as one layer of a two-layer buffer.

    The op checks metadata itself, loudly: width 2..4, `state_len >= W - 1`,
    one dtype across x/conv_state/weight (bias matches weight), weight shape,
    int32 contiguous length-batch `conv_state_indices`, int32
    `cache_seqlens`, a batch-sized pool when indices are absent. So no guard.

    Not guarded because it is content, not metadata: two rows naming one slot
    race -- the slot ended up matching neither row's expected state in 10 of
    10 runs while both outputs matched. Slot uniqueness is the caller's.
    Out-of-range slot values are an illegal access, not probed.

    A row whose index equals `pad_slot_id` (-1 at the call site) is skipped:
    its x rows and the pool stay bitwise unchanged (certified on both kernels:
    the CuTe one at 64 rows of 10240, the TRT-LLM one at tp4's 128 rows).

    This target's call site, per rank. tp4 / tep4: x `[num_decodes, 2560]`,
    the q|k|v view (row stride 4120) of the rank's `[T, 4120]` projection; the
    pool one layer's view `[slots, 2560, 3]` of the V2 manager's conv page,
    36 GDN layers a slot (slot stride 276480). dep4: tp1's 10240 channels at
    stride 16480, page slot stride 1105920. Certified at the measured per-rank
    slot counts (`TEP4_SLOTS` 1874, `DEP4_SLOTS` 850; tp4 has none measured and
    takes tep4's, `TP4_SLOTS`), the view the page's last layer and the top slot
    named, so a cell reaches the pool's highest address: tep4 / tp4 at 1, 64
    and 128 rows, dep4 at 32 and 128 rows.

    Which kernel runs: 2560 is not a multiple of `_CH * _THREADS` (1024), so
    every tp4 / tep4 call runs TRT-LLM's seqlen-1 kernel
    (`causal_conv1d_update_kernel_sl1`, one thread per (row, channel), 20
    blocks of 128 channels a row), whatever the row count; the CuTe kernel's
    gain at 64+ rows is not had there. dep4 runs the CuTe kernel from 64 rows,
    as tp1 does.

    Sources: the TRT-LLM kernel is compiled into the wheel
    (`cpp/tensorrt_llm/kernels/causalConv1d/causalConv1d.cu`, registered by
    `cpp/tensorrt_llm/thop/causalConv1dOp.cpp`); the CuTe kernel is JIT-compiled
    by `cute.compile` on its first call for the running device, sm_103 here.

    Hard limits (from source, not probed: the measured pools stay far below)
      The TRT-LLM kernel forms a slot's offset as `int slot * uint32_t
      conv_state_batch_stride`, 32-bit unsigned elements: it wraps from slot
      15535 at stride 276480 (tp4 / tep4) and slot 3884 at 1105920 (dep4).
      The measured pools reach 518,123,519 (tep4) and 940,031,999 (dep4)
      elements, under 2^31 elements and bytes. The CuTe kernel forms slot and
      row offsets in 64 bits.
    """

    def __call__(
        self,
        x: torch.Tensor,
        conv_state: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        silu_activation: bool,
        cache_seqlens: Optional[torch.Tensor],
        conv_state_indices: Optional[torch.Tensor],
        pad_slot_id: int,
    ) -> torch.Tensor:
        if _decode_conv_ok(
            x, conv_state, weight, bias, silu_activation, cache_seqlens, conv_state_indices
        ):
            _decode_conv(x, conv_state, weight, conv_state_indices, pad_slot_id)
            return x
        torch.ops.trtllm.causal_conv1d_update(
            x.unsqueeze(-1) if x.dim() == 2 else x,
            conv_state,
            weight,
            bias,
            silu_activation,
            cache_seqlens,
            conv_state_indices,
            pad_slot_id,
        )
        return x

    def reference(
        self,
        x: torch.Tensor,
        conv_state: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        silu_activation: bool,
        cache_seqlens: Optional[torch.Tensor],
        conv_state_indices: Optional[torch.Tensor],
        pad_slot_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`(x, conv_state)` as the call leaves them; mutates nothing."""
        x3 = x.unsqueeze(-1) if x.dim() == 2 else x
        batch, dim, seqlen = x3.shape
        width = weight.shape[1]
        state_len = conv_state.shape[2]
        out = x3.clone()
        new_state = conv_state.clone()
        slots = None if conv_state_indices is None else conv_state_indices.tolist()
        starts = None if cache_seqlens is None else cache_seqlens.tolist()
        for b in range(batch):
            slot = b if slots is None else slots[b]
            if slots is not None and slot == pad_slot_id:
                continue
            state = conv_state[slot].float()
            tokens = x3[b].float()
            if starts is None:
                history = torch.cat([state, tokens], dim=1)
                window = history[:, state_len - (width - 1) :]
                new_state[slot] = history[:, -state_len:].to(conv_state.dtype)
            else:
                start = starts[b] % state_len
                read = [(start - (width - 1) + i) % state_len for i in range(width - 1)]
                window = torch.cat([state[:, read], tokens], dim=1)
                advanced = state.clone()
                for i in range(seqlen):
                    advanced[:, (start + i) % state_len] = tokens[:, i]
                new_state[slot] = advanced.to(conv_state.dtype)
            acc = torch.zeros(dim, seqlen, dtype=torch.float32, device=x.device)
            for j in range(width):
                acc = acc + weight[:, j].float().unsqueeze(1) * window[:, j : j + seqlen]
            if bias is not None:
                acc = acc + bias.float().unsqueeze(1)
            if silu_activation:
                acc = acc * torch.sigmoid(acc)
            out[b] = acc.to(x.dtype)
        return (out.squeeze(-1) if x.dim() == 2 else out), new_state


causal_conv1d_update = CausalConv1dUpdate()
