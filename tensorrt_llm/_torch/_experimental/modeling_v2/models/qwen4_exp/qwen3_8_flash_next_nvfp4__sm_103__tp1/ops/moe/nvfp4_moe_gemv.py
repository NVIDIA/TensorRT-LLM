# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small-row NVFP4 MoE layer entered with router logits, left unfinalized: Renormalize
routing + FC1 + SwiGLU + NVFP4 requantization + FC2 over the trtllm-gen expert layout, in
two PDL-chained CuTe DSL launches on the tcgen05 block-scaled tensor core
(`kind::mxf4nvf4.block_scale.block16`: e2m1 operands, ue4m3 16-element scales, fp32 in
TMEM), for 1 .. MAX_ROWS rows. Same outputs as `fp4_block_scale_moe_renorm_permuted`: the
permuted FC2 rows, the routing weights and the slot -> row map the next hyper-connection
kernel gathers.

Geometry (asserted): 512 experts, top-10, hidden 2560, intermediate 640.

Routing (both launches, every CTA, one warp per row): the runner's Renormalize policy --
the 10 largest bf16 logits ranked by the packed key (order-preserving bf16 bits << 16 |
65535 - expert), ties to the lower expert, then an fp32 softmax over the 10 with the routing
kernel's arithmetic (libdevice `expf`, a 32-lane xor-butterfly sum, IEEE division), each
weight rounded once to bf16. The distinct experts of all rows, in ascending id order, are the
work list; a row's slot k is its k-th ranked expert.

MMA: M 128 weight rows x N 8 rows x K 64 a step; A (weights) arrives by 3D TMA boxes in the
128- (FC1) / 64-byte (FC2) swizzled K-major layout, B (the rows) sits in 16-byte core
matrices, and both scale operands are 512-byte 128x4 atoms (the checkpoint's own for the
weights) moved into TMEM by `tcgen05.cp.32x128b.warpx4`. The TMA descriptors of an expert
stack are built once (`register_weights`) and read from global memory.

Launch 1 (`_Fc1`): after `griddepcontrol.wait` it releases its dependents and routes; a
producer thread streams its 128-row FC1 tiles (5 K stages of 36 KB) into a shared-memory
ring, one thread issues the MMAs (the whole K of a tile in one TMEM accumulator), and four
epilogue warps read the accumulator: TMEM lane quarter q of tile t is the 32-row block
4 t + q, whose rows are the up and gate rows of 16 intermediate columns (row d's gate is
row d + 8, a shuffle away) -- one NVFP4 block. `up * output1_scale_scalar` times
`silu(gate * output1_scale_gate_scalar)` is requantized (scale `e4m3(amax / 6)`, codes
`e2m1(v / scale)`) into a per-slot record in global memory.

Launch 2 (`_Fc2`): before its wait it routes again (from the logits, which launch 1 read
after its own wait, so they are complete), loads its 128-row FC2 tiles and their weight
scales into shared memory and the scales into TMEM -- while launch 1 still runs. After the
wait it copies each unit's rows of the slot records into its B operand, issues the MMAs,
scales by `output2_scale_scalar` and stores bf16 rows. Launch 1 trips no barrier launch 2
waits on: the only cross-launch dependency is `griddepcontrol.wait`, so residency of either
grid is never required.

The two grids do not share an SM (launch 2's CTAs start where no launch-1 CTA is resident),
so at one row, where the FC1 work is always 100 tiles, launch 1 takes 100 SMs (its whole
tile in flight) and launch 2 two CTAs on each of the other 52; at more rows each takes every
SM and launch 2 starts as launch 1 leaves.

Outputs: `rows [10 T, 2560]` bf16 with slot (t, k) at row 10 t + k, `topk_weights [T, 10]`
bf16, `topk_rows [T, 10]` int32 = 10 t + k (a constant per T).
"""

from __future__ import annotations

import functools

import cutlass
import cutlass.cute as cute
import torch
from cutlass import const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import from_dlpack, make_ptr
from cutlass.cutlass_dsl import T, dsl_user_op

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ..gemm.fp8_gemv import (
    _asm,
    _bar_sync,
    _bulk_g2s,
    _bulk_prefetch_l2,
    _env_pdl,
    _ir,
    _lds4,
    _lds_f32,
    _mbar_arrive,
    _mbar_expect_tx,
    _mbar_init,
    _mbar_init_fence,
    _mbar_wait,
    _sms,
    _sts_f32,
    _unpack,
)
from ..gemm.nvfp4_gemv import _lds32, _sts4
from .fp4_block_scale_moe_renorm_permuted import finalize
from .fp4_block_scale_moe_renorm_routed import CERTIFIED_PAIRS, _ckpt
from .fp4_block_scale_moe_renorm_routed import fp4_block_scale_moe_renorm_routed as _routed

NUM_EXPERTS = 512
TOP_K = 10
HIDDEN = 2560
INTER = 640
#: Most rows a launch takes (the n8 MMA tile).
MAX_ROWS = 8
#: Compiled row buckets: a call with T rows runs the smallest bucket >= T.
ROW_BUCKETS = (1, 2, 4, 8)

BLK = 32  # rows of one requantization block (FC1) / output block (FC2) of a tile
N1, K1 = 2 * INTER, HIDDEN  # FC1: 1280 rows of 1280 code bytes, 160 scales
N2, K2 = HIDDEN, INTER  # FC2: 2560 rows of 320 code bytes, 40 scales
NT1, NT2 = N1 // 128, N2 // 128  # 128-row tiles per expert: 10 and 20
KQ1 = 5  # FC1 K stages a tile (512 columns each)
A1_BYTES = 128 * 256  # one FC1 stage: two 128-byte swizzled chunks of 128 rows
SLOT1 = A1_BYTES + 8 * 512  # + its 8 scale atoms
A2_BYTES = 128 * (K2 // 2)  # one FC2 tile: five 64-byte swizzled chunks of 128 rows
SLOT2 = A2_BYTES + (K2 // 64) * 512
NB_ROW = 8 * 16  # B operand: one 16-byte k-chunk of the 8 rows
B2_BYTES = 4352  # FC2 B (20 chunks) and its 10 scale atoms (+ the 384-byte read overhang)
TC_THREADS = 256  # producer, MMA issuer, 4 epilogue warps, 2 helpers
#: Per-slot record of the requantized intermediate: 320 code bytes, 40 e4m3 scales, pad.
REC = 384
#: Most units one CTA takes in a call.
MAX_ITEMS = 64
#: TMEM columns per CTA (both launches together use all 512 of an SM). FC1: two 8-column
#: accumulators, the weight scales of 2 stages (32 each), the activation scales (160); FC2:
#: two accumulators, the weight scales of 2 tiles (40 each), the B scales of 2 units (40 each).
TMEM_COLS = 256
SFA1_COL = 16  # FC1 activation scales follow the slots' weight scales
SFA2_COL = 16  # FC2 B scales follow the slots' weight scales
#: Instruction descriptor: E2M1 A and B, UE4M3 16-element block scales, N 8, M 128.
IDESC = (1 << 7) | (1 << 10) | ((8 >> 3) << 17) | ((128 >> 4) << 24)


def SFB_BYTES(k: int) -> int:  # noqa: N802
    """Bytes of the activation's scale atoms over K (128 a 64-column step + the overhang a
    512-byte atom copy reads past the last)."""
    return (k // 64) * 128 + 384


_SMEM_BUDGET = 227 << 10
_COMPILE_OPTIONS = "--opt-level 2 --enable-tvm-ffi"


# ------------------------------------------------------------------------------------------
# inline PTX helpers
# ------------------------------------------------------------------------------------------


@dsl_user_op
def _ldg_cg4(gaddr, *, loc=None, ip=None):
    """16 bytes from global through L2 only (data another grid wrote this step)."""
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_ir(gaddr, cutlass.Int64, loc, ip)],
        "ld.global.cg.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        loc,
        ip,
        side=True,
    )
    return _unpack(res, 4, cutlass.Int32, loc, ip)


@dsl_user_op
def _ldg_nc_f32(gaddr, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            T.f32(),
            [_ir(gaddr, cutlass.Int64, loc, ip)],
            "ld.global.nc.f32 $0, [$1];",
            "=f,l",
            loc,
            ip,
            side=True,
        )
    )


@dsl_user_op
def _lds_u8(saddr, *, loc=None, ip=None):
    return cutlass.Int32(
        _asm(
            T.i32(),
            [_ir(saddr, cutlass.Int32, loc, ip)],
            "ld.shared.u8 $0, [$1];",
            "=r,r",
            loc,
            ip,
            side=True,
        )
    )


@dsl_user_op
def _sts32(saddr, v, *, loc=None, ip=None):
    _asm(
        T.i32(),
        [_ir(saddr, cutlass.Int32, loc, ip), _ir(v, cutlass.Int32, loc, ip)],
        "{ st.shared.b32 [$1], $2; mov.u32 $0, 0; }",
        "=r,r,r",
        loc,
        ip,
        side=True,
    )


@dsl_user_op
def _sts_u8(saddr, v, *, loc=None, ip=None):
    _asm(
        T.i32(),
        [_ir(saddr, cutlass.Int32, loc, ip), _ir(v, cutlass.Int32, loc, ip)],
        "{ .reg .b16 h; cvt.u16.u32 h, $2; st.shared.u8 [$1], h; mov.u32 $0, 0; }",
        "=r,r,r",
        loc,
        ip,
        side=True,
    )


@dsl_user_op
def _atom_or_shared(saddr, v, *, loc=None, ip=None):
    _asm(
        T.i32(),
        [_ir(saddr, cutlass.Int32, loc, ip), _ir(v, cutlass.Int32, loc, ip)],
        "{ .reg .b32 t; atom.shared.or.b32 t, [$1], $2; mov.u32 $0, 0; }",
        "=r,r,r",
        loc,
        ip,
        side=True,
    )


@dsl_user_op
def _stg_b16(gaddr, v, *, loc=None, ip=None):
    _asm(
        T.i32(),
        [_ir(gaddr, cutlass.Int64, loc, ip), _ir(v, cutlass.Int32, loc, ip)],
        "{ .reg .b16 h; cvt.u16.u32 h, $2; st.global.b16 [$1], h; mov.u32 $0, 0; }",
        "=r,l,r",
        loc,
        ip,
        side=True,
    )


@dsl_user_op
def _stg_v2(gaddr, v0, v1, *, loc=None, ip=None):
    _asm(
        T.i32(),
        [_ir(gaddr, cutlass.Int64, loc, ip)] + [_ir(v, cutlass.Int32, loc, ip) for v in (v0, v1)],
        "{ st.global.v2.b32 [$1], {$2, $3}; mov.u32 $0, 0; }",
        "=r,l,r,r",
        loc,
        ip,
        side=True,
    )


@dsl_user_op
def _stg_u8(gaddr, v, *, loc=None, ip=None):
    _asm(
        T.i32(),
        [_ir(gaddr, cutlass.Int64, loc, ip), _ir(v, cutlass.Int32, loc, ip)],
        "{ .reg .b16 h; cvt.u16.u32 h, $2; st.global.u8 [$1], h; mov.u32 $0, 0; }",
        "=r,l,r",
        loc,
        ip,
        side=True,
    )


@dsl_user_op
def _cp_async16(saddr, gaddr, *, loc=None, ip=None):
    _asm(
        T.i32(),
        [_ir(saddr, cutlass.Int32, loc, ip), _ir(gaddr, cutlass.Int64, loc, ip)],
        "{ cp.async.cg.shared.global [$1], [$2], 16; mov.u32 $0, 0; }",
        "=r,r,l",
        loc,
        ip,
        side=True,
    )


@dsl_user_op
def _cp_async_wait_all(*, loc=None, ip=None):
    _asm(
        T.i32(),
        [],
        "{ cp.async.commit_group; cp.async.wait_all; mov.u32 $0, 0; }",
        "=r",
        loc,
        ip,
        side=True,
    )


@dsl_user_op
def _popc(x, *, loc=None, ip=None):
    return cutlass.Int32(
        _asm(T.i32(), [_ir(x, cutlass.Int32, loc, ip)], "popc.b32 $0, $1;", "=r,r", loc, ip)
    )


@dsl_user_op
def _keys2(w, e0, *, loc=None, ip=None):
    """A word of two bf16 logits (experts e0, e0 + 1) -> their ranking keys: the bits made
    order-preserving as unsigned (positive: flip the sign bit; negative: flip all) in the
    high half, 65535 - expert in the low half (ties rank the lower expert first)."""
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 2),
        [_ir(w, cutlass.Int32, loc, ip), _ir(e0, cutlass.Int32, loc, ip)],
        "{ .reg .b32 lo, hi, m, i; "
        "and.b32 lo, $2, 65535; shr.b32 hi, $2, 16; "
        "shr.b32 m, lo, 15; mul.lo.u32 m, m, 32767; or.b32 m, m, 32768; xor.b32 lo, lo, m; "
        "shr.b32 m, hi, 15; mul.lo.u32 m, m, 32767; or.b32 m, m, 32768; xor.b32 hi, hi, m; "
        "sub.u32 i, 65535, $3; shl.b32 lo, lo, 16; or.b32 $0, lo, i; "
        "sub.u32 i, i, 1; shl.b32 hi, hi, 16; or.b32 $1, hi, i; }",
        "=r,=r,r,r",
        loc,
        ip,
    )
    return _unpack(res, 2, cutlass.Int32, loc, ip)


@dsl_user_op
def _umax(a, b, *, loc=None, ip=None):
    return cutlass.Int32(
        _asm(
            T.i32(),
            [_ir(a, cutlass.Int32, loc, ip), _ir(b, cutlass.Int32, loc, ip)],
            "max.u32 $0, $1, $2;",
            "=r,r,r",
            loc,
            ip,
        )
    )


@dsl_user_op
def _redux_umax(a, *, loc=None, ip=None):
    return cutlass.Int32(
        _asm(
            T.i32(),
            [_ir(a, cutlass.Int32, loc, ip)],
            "redux.sync.max.u32 $0, $1, 0xffffffff;",
            "=r,r",
            loc,
            ip,
            side=True,
        )
    )


@dsl_user_op
def _clear_eq(k, win, *, loc=None, ip=None):
    """`k`, or 0 where it equals `win`."""
    return cutlass.Int32(
        _asm(
            T.i32(),
            [_ir(k, cutlass.Int32, loc, ip), _ir(win, cutlass.Int32, loc, ip)],
            "{ .reg .pred p; setp.eq.b32 p, $1, $2; selp.b32 $0, 0, $1, p; }",
            "=r,r,r",
            loc,
            ip,
        )
    )


@dsl_user_op
def _sel_eq(a, b, x, y, *, loc=None, ip=None):
    """`x` where `a == b`, else `y`."""
    return cutlass.Int32(
        _asm(
            T.i32(),
            [_ir(v, cutlass.Int32, loc, ip) for v in (a, b, x, y)],
            "{ .reg .pred p; setp.eq.b32 p, $1, $2; selp.b32 $0, $3, $4, p; }",
            "=r,r,r,r,r",
            loc,
            ip,
        )
    )


@dsl_user_op
def _key_value(key, *, loc=None, ip=None):
    """The fp32 value of a ranking key's bf16 logit."""
    return cutlass.Float32(
        _asm(
            T.f32(),
            [_ir(key, cutlass.Int32, loc, ip)],
            "{ .reg .b32 o, m; shr.b32 o, $1, 16; shr.b32 m, o, 15; mul.lo.u32 m, m, 32767; "
            "xor.b32 m, m, 65535; xor.b32 o, o, m; shl.b32 $0, o, 16; }",
            "=f,r",
            loc,
            ip,
        )
    )


@dsl_user_op
def _shfl_idx_f32(v, src, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            T.f32(),
            [_ir(v, cutlass.Float32, loc, ip), _ir(src, cutlass.Int32, loc, ip)],
            "shfl.sync.idx.b32 $0, $1, $2, 31, 0xffffffff;",
            "=f,f,r",
            loc,
            ip,
            side=True,
        )
    )


@dsl_user_op
def _shfl_bfly_f32(v, m, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            T.f32(),
            [_ir(v, cutlass.Float32, loc, ip), _ir(m, cutlass.Int32, loc, ip)],
            "shfl.sync.bfly.b32 $0, $1, $2, 31, 0xffffffff;",
            "=f,f,r",
            loc,
            ip,
            side=True,
        )
    )


@dsl_user_op
def _expf(x, *, loc=None, ip=None):
    """libdevice `expf` (the sequence nvcc emits without fast math)."""
    return cutlass.Float32(
        _asm(
            T.f32(),
            [_ir(x, cutlass.Float32, loc, ip)],
            "{ .reg .f32 a, b, c, d, e, f; .reg .b32 s; "
            "fma.rn.f32 a, $1, 0f3BBB989D, 0f3F000000; cvt.sat.f32.f32 a, a; "
            "fma.rm.f32 b, a, 0f437C0000, 0f4B400001; add.f32 c, b, 0fCB40007F; neg.f32 c, c; "
            "fma.rn.f32 d, $1, 0f3FB8AA3B, c; fma.rn.f32 d, $1, 0f32A57060, d; "
            "mov.b32 s, b; shl.b32 s, s, 23; mov.b32 f, s; "
            "ex2.approx.ftz.f32 e, d; mul.f32 $0, e, f; }",
            "=f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def _div_rn(a, b, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            T.f32(),
            [_ir(a, cutlass.Float32, loc, ip), _ir(b, cutlass.Float32, loc, ip)],
            "div.rn.f32 $0, $1, $2;",
            "=f,f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def _bf16_bits_f32(x, *, loc=None, ip=None):
    """`x` rounded to bf16: (bits in the low half, the rounded value as fp32)."""
    res = _asm(
        llvm.StructType.get_literal([T.i32(), T.f32()]),
        [_ir(x, cutlass.Float32, loc, ip)],
        "{ .reg .b16 h; cvt.rn.bf16.f32 h, $2; cvt.u32.u16 $0, h; cvt.f32.bf16 $1, h; }",
        "=r,=f,f",
        loc,
        ip,
    )
    return (
        cutlass.Int32(llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)),
        cutlass.Float32(llvm.extractvalue(T.f32(), res, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def _e4m3_sf(amax, *, loc=None, ip=None):
    """The NVFP4 block scale of a block whose largest magnitude is `amax`: (e4m3 byte of
    amax / 6, rounded to nearest and saturated, and its value as fp32)."""
    res = _asm(
        llvm.StructType.get_literal([T.i32(), T.f32()]),
        [_ir(amax, cutlass.Float32, loc, ip)],
        "{ .reg .f32 q; .reg .b16 h, lo, hi; .reg .b32 f; "
        "div.rn.f32 q, $2, 0f40C00000; cvt.rn.satfinite.e4m3x2.f32 h, 0f00000000, q; "
        "cvt.u32.u16 $0, h; cvt.rn.f16x2.e4m3x2 f, h; mov.b32 {lo, hi}, f; "
        "cvt.f32.f16 $1, lo; }",
        "=r,=f,f",
        loc,
        ip,
    )
    return (
        cutlass.Int32(llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)),
        cutlass.Float32(llvm.extractvalue(T.f32(), res, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def _pos_or_one(x, *, loc=None, ip=None):
    """`x` if positive, else 1 (a zero block scale divides nothing: its block is all zero
    codes either way)."""
    return cutlass.Float32(
        _asm(
            T.f32(),
            [_ir(x, cutlass.Float32, loc, ip)],
            "{ .reg .pred p; setp.gt.f32 p, $1, 0f00000000; selp.f32 $0, $1, 0f3F800000, p; }",
            "=f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def _fabs(x, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(T.f32(), [_ir(x, cutlass.Float32, loc, ip)], "abs.f32 $0, $1;", "=f,f", loc, ip)
    )


@dsl_user_op
def _fmax(a, b, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            T.f32(),
            [_ir(a, cutlass.Float32, loc, ip), _ir(b, cutlass.Float32, loc, ip)],
            "max.f32 $0, $1, $2;",
            "=f,f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def _shfl_down_i32(v, d, *, loc=None, ip=None):
    return cutlass.Int32(
        _asm(
            T.i32(),
            [_ir(v, cutlass.Int32, loc, ip), _ir(d, cutlass.Int32, loc, ip)],
            "shfl.sync.down.b32 $0, $1, $2, 31, 0xffffffff;",
            "=r,r,r",
            loc,
            ip,
            side=True,
        )
    )


@dsl_user_op
def _e2m1_lo(x, *, loc=None, ip=None):
    """`x` rounded (nearest even, saturating) to e2m1: its 4-bit code."""
    return cutlass.Int32(
        _asm(
            T.i32(),
            [_ir(x, cutlass.Float32, loc, ip)],
            "{ .reg .b8 b; .reg .b16 h; cvt.rn.satfinite.e2m1x2.f32 b, 0f00000000, $1; "
            "cvt.u16.u8 h, b; cvt.u32.u16 $0, h; and.b32 $0, $0, 15; }",
            "=r,f",
            loc,
            ip,
        )
    )


# ---- tcgen05 / TMA helpers -----------------------------------------------------------------


def _desc_asm(reg: str, saddr: str, lbo: int, sbo: int, layout: int) -> str:
    """PTX building the 64-bit shared-memory matrix descriptor `reg` of the shared address in
    register `saddr` (start >> 4, LBO >> 4, SBO >> 4, version 1, layout type)."""
    hi = (sbo >> 4) | (1 << 14) | (layout << 29)
    return (
        f"shr.b32 {reg}lo, {saddr}, 4; and.b32 {reg}lo, {reg}lo, 16383; "
        f"or.b32 {reg}lo, {reg}lo, {(lbo >> 4) << 16}; mov.b32 {reg}hi, {hi}; "
        f"mov.b64 {reg}, {{{reg}lo, {reg}hi}}; "
    )


@dsl_user_op
def _tma3d(saddr, tmap, x, y, z, mbar, *, loc=None, ip=None):
    _asm(
        T.i32(),
        [_ir(saddr, cutlass.Int32, loc, ip), _ir(tmap, cutlass.Int64, loc, ip)]
        + [_ir(v, cutlass.Int32, loc, ip) for v in (x, y, z, mbar)],
        "{ cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[$1], [$2, {$3, $4, $5}], [$6]; mov.u32 $0, 0; }",
        "=r,r,l,r,r,r,r",
        loc,
        ip,
        side=True,
    )


@dsl_user_op
def _prefetch_tmap(tmap, *, loc=None, ip=None):
    _asm(
        T.i32(),
        [_ir(tmap, cutlass.Int64, loc, ip)],
        "{ prefetch.tensormap [$1]; mov.u32 $0, 0; }",
        "=r,l",
        loc,
        ip,
        side=True,
    )


@dsl_user_op
def _tc_alloc(saddr, ncols=256, *, loc=None, ip=None):
    _asm(
        T.i32(),
        [_ir(saddr, cutlass.Int32, loc, ip)],
        f"{{ tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [$1], {ncols}; "
        "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned; mov.u32 $0, 0; }",
        "=r,r",
        loc,
        ip,
        side=True,
    )


@dsl_user_op
def _tc_dealloc(taddr, ncols=256, *, loc=None, ip=None):
    _asm(
        T.i32(),
        [_ir(taddr, cutlass.Int32, loc, ip)],
        f"{{ tcgen05.dealloc.cta_group::1.sync.aligned.b32 $1, {ncols}; mov.u32 $0, 0; }}",
        "=r,r",
        loc,
        ip,
        side=True,
    )


@dsl_user_op
def _tc_fence(after, *, loc=None, ip=None):
    op = "after_thread_sync" if after else "before_thread_sync"
    _asm(T.i32(), [], f"{{ tcgen05.fence::{op}; mov.u32 $0, 0; }}", "=r", loc, ip, side=True)


@dsl_user_op
def _fence_proxy_async(*, loc=None, ip=None):
    _asm(T.i32(), [], "{ fence.proxy.async.shared::cta; mov.u32 $0, 0; }", "=r", loc, ip, side=True)


def _make_desc(lbo: int, sbo: int, layout: int):
    @dsl_user_op
    def _desc(saddr, *, loc=None, ip=None):
        """The 64-bit shared-memory matrix descriptor of `saddr`."""
        return cutlass.Int64(
            _asm(
                T.i64(),
                [_ir(saddr, cutlass.Int32, loc, ip)],
                "{ .reg .b32 dlo, dhi; "
                + _desc_asm("d", "$1", lbo, sbo, layout).replace("mov.b64 d,", "mov.b64 $0,")
                + "}",
                "=l,r",
                loc,
                ip,
            )
        )

    return _desc


_desc_sf = _make_desc(0, 128, 0)  # a scale atom (tcgen05.cp)
_desc_b = _make_desc(8 * 16, 128, 0)  # B: the 8 rows in 16-byte core matrices
_desc_a1 = _make_desc(16, 1024, 2)  # FC1 A: 128-byte swizzle
_desc_a2 = _make_desc(16, 512, 4)  # FC2 A: 64-byte swizzle


@dsl_user_op
def _tc_cp_d(taddr, desc, off, *, loc=None, ip=None):
    """tcgen05.cp of the scale atom `off` bytes past descriptor `desc` (off % 16 == 0)."""
    _asm(
        T.i32(),
        [_ir(taddr, cutlass.Int32, loc, ip), _ir(desc, cutlass.Int64, loc, ip)],
        "{ .reg .b64 d; add.s64 d, $2, " + str(off >> 4) + "; "
        "tcgen05.cp.cta_group::1.32x128b.warpx4 [$1], d; mov.u32 $0, 0; }",
        "=r,r,l",
        loc,
        ip,
        side=True,
    )


@dsl_user_op
def _mma_d(tacc, adesc, aoff, bdesc, boff, sfa, sfb, acc, *, loc=None, ip=None):
    """tcgen05 block-scaled NVFP4 MMA (M 128, N 8, K 64) on descriptors `adesc` / `bdesc`
    advanced by `aoff` / `boff` bytes."""
    _asm(
        T.i32(),
        [
            _ir(tacc, cutlass.Int32, loc, ip),
            _ir(adesc, cutlass.Int64, loc, ip),
            _ir(bdesc, cutlass.Int64, loc, ip),
            _ir(sfa, cutlass.Int32, loc, ip),
            _ir(sfb, cutlass.Int32, loc, ip),
            _ir(acc, cutlass.Int32, loc, ip),
        ],
        "{ .reg .pred p; .reg .b64 a, b; "
        f"add.s64 a, $2, {aoff >> 4}; add.s64 b, $3, {boff >> 4}; setp.ne.b32 p, $6, 0; "
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 "
        f"[$1], a, b, {IDESC}, [$4], [$5], p; mov.u32 $0, 0; }}",
        "=r,r,l,l,r,r,r",
        loc,
        ip,
        side=True,
    )


@dsl_user_op
def _cp_async4g(saddr, gaddr, *, loc=None, ip=None):
    _asm(
        T.i32(),
        [_ir(saddr, cutlass.Int32, loc, ip), _ir(gaddr, cutlass.Int64, loc, ip)],
        "{ cp.async.ca.shared.global [$1], [$2], 4; mov.u32 $0, 0; }",
        "=r,r,l",
        loc,
        ip,
        side=True,
    )


@dsl_user_op
def _tc_commit(mbar, *, loc=None, ip=None):
    _asm(
        T.i32(),
        [_ir(mbar, cutlass.Int32, loc, ip)],
        "{ tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [$1]; "
        "mov.u32 $0, 0; }",
        "=r,r",
        loc,
        ip,
        side=True,
    )


@dsl_user_op
def _tc_ld8(taddr, *, loc=None, ip=None):
    """Eight fp32 TMEM columns of this thread's lane (32x32b shape), waited."""
    res = _asm(
        llvm.StructType.get_literal([T.f32()] * 8),
        [_ir(taddr, cutlass.Int32, loc, ip)],
        "{ tcgen05.ld.sync.aligned.32x32b.x8.b32 {$0, $1, $2, $3, $4, $5, $6, $7}, [$8]; "
        "tcgen05.wait::ld.sync.aligned; }",
        "=f,=f,=f,=f,=f,=f,=f,=f,r",
        loc,
        ip,
        side=True,
    )
    return _unpack(res, 8, cutlass.Float32, loc, ip)


@dsl_user_op
def _ldg_nc4p(gaddr, pred, *, loc=None, ip=None):
    """16 bytes from global (L2-coherent) if `pred` != 0, else zeros."""
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_ir(gaddr, cutlass.Int64, loc, ip), _ir(pred, cutlass.Int32, loc, ip)],
        "{ .reg .pred p; setp.ne.b32 p, $5, 0; mov.b32 $0, 0; mov.b32 $1, 0; mov.b32 $2, 0; "
        "mov.b32 $3, 0; @p ld.global.cg.v4.b32 {$0, $1, $2, $3}, [$4]; }",
        "=r,=r,=r,=r,l,r",
        loc,
        ip,
        side=True,
    )
    return _unpack(res, 4, cutlass.Int32, loc, ip)


@dsl_user_op
def _ldg_cg32p(gaddr, pred, *, loc=None, ip=None):
    return cutlass.Int32(
        _asm(
            T.i32(),
            [_ir(gaddr, cutlass.Int64, loc, ip), _ir(pred, cutlass.Int32, loc, ip)],
            "{ .reg .pred p; setp.ne.b32 p, $2, 0; mov.b32 $0, 0; @p ld.global.cg.b32 $0, [$1]; }",
            "=r,l,r",
            loc,
            ip,
            side=True,
        )
    )


@dsl_user_op
def _sel_f(flag, a, b, *, loc=None, ip=None):
    """`a` where `flag` != 0, else `b`."""
    return cutlass.Float32(
        _asm(
            T.f32(),
            [
                _ir(flag, cutlass.Int32, loc, ip),
                _ir(a, cutlass.Float32, loc, ip),
                _ir(b, cutlass.Float32, loc, ip),
            ],
            "{ .reg .pred p; setp.ne.b32 p, $1, 0; selp.f32 $0, $2, $3, p; }",
            "=f,r,f,f",
            loc,
            ip,
        )
    )


# ------------------------------------------------------------------------------------------
# kernels
# ------------------------------------------------------------------------------------------


class _Routed:
    """Routing and the work list, shared by both launches. Shared-memory routing area
    (`o_rt`): slot keys' experts `[8][10]` int32, weights `[8][10]` fp32, the selected-expert
    bitmap (16 words), the work list `[80]` int32, per-entry row masks `[80]` and per-entry
    row slots `[80][8]` bytes."""

    RT_IDS, RT_WTS, RT_MAP, RT_LIST, RT_MASK, RT_SLOT = 0, 320, 640, 704, 1024, 1344
    RT_BYTES = 1344 + 640

    @cute.jit
    def _route(self, lg_base, n, lane, rt, topw_base, write_topw):
        """Row `n`'s top-10 and softmax (one warp): slot k on lane k; results to shared
        memory, the weights also to `topw_base` when `write_topw`."""
        row = lg_base + cutlass.Int64(n) * (NUM_EXPERTS * 2) + cutlass.Int64(lane * 32)
        va = _ldg_cg4(row)
        vb = _ldg_cg4(row + 16)
        words = va + vb
        keys = []
        for i in cutlass.range_constexpr(8):
            k2 = _keys2(words[i], lane * 16 + 2 * i)
            keys.append(k2[0])
            keys.append(k2[1])
        mine = cutlass.Int32(0)
        for r in cutlass.range_constexpr(TOP_K):
            m8 = [_umax(keys[2 * i], keys[2 * i + 1]) for i in range(8)]
            m4 = [_umax(m8[2 * i], m8[2 * i + 1]) for i in range(4)]
            m2 = [_umax(m4[0], m4[1]), _umax(m4[2], m4[3])]
            win = _redux_umax(_umax(m2[0], m2[1]))
            keys = [_clear_eq(k, win) for k in keys]
            mine = _sel_eq(lane, cutlass.Int32(r), win, mine)
        v = _key_value(mine)
        vmax = _shfl_idx_f32(v, cutlass.Int32(0))
        e = cutlass.Float32(0.0)
        if lane < TOP_K:
            e = _expf(v - vmax)
        s = e
        for m in cutlass.range_constexpr(5):
            s = s + _shfl_bfly_f32(s, cutlass.Int32(16 >> m))
        if lane < TOP_K:
            wbits, wf = _bf16_bits_f32(_div_rn(e, s))
            expert = 65535 - (mine & 65535)
            _sts32(rt + self.RT_IDS + (n * TOP_K + lane) * 4, expert)
            _sts_f32(rt + self.RT_WTS + (n * TOP_K + lane) * 4, wf)
            _atom_or_shared(rt + self.RT_MAP + (expert >> 5) * 4, cutlass.Int32(1) << (expert & 31))
            if write_topw:
                _stg_b16(topw_base + cutlass.Int64(n * TOP_K + lane) * 2, wbits)

    @cute.jit
    def _list(self, tidx, rows, rt):
        """Thread i < 10 rows: slot (i // 10, i % 10) enters the work list at its expert's
        rank among the selected experts (ascending id)."""
        if tidx < rows * TOP_K:
            n = tidx // TOP_K
            k = tidx - n * TOP_K
            e = cutlass.Int32(0)
            e = _lds32(rt + self.RT_IDS + tidx * 4)
            wd = e >> 5
            rank = _popc(_lds32(rt + self.RT_MAP + wd * 4) & ((cutlass.Int32(1) << (e & 31)) - 1))
            for w in cutlass.range_constexpr(16):
                if w < wd:
                    rank = rank + _popc(_lds32(rt + self.RT_MAP + w * 4))
            _sts32(rt + self.RT_LIST + rank * 4, e)
            _atom_or_shared(rt + self.RT_MASK + rank * 4, cutlass.Int32(1) << n)
            _sts_u8(rt + self.RT_SLOT + rank * 8 + n, k)

    @cute.jit
    def _ndist(self, rt):
        nd = cutlass.Int32(0)
        for w in cutlass.range_constexpr(4):
            q = _lds4(rt + self.RT_MAP + w * 16)
            nd = nd + _popc(q[0]) + _popc(q[1]) + _popc(q[2]) + _popc(q[3])
        return nd

    @cute.jit
    def _setup_rt(self, tidx, rt):
        """Zero the bitmap and the row masks."""
        if tidx < 16:
            _sts32(rt + self.RT_MAP + tidx * 4, cutlass.Int32(0))
        if tidx < 80:
            _sts32(rt + self.RT_MASK + tidx * 4, cutlass.Int32(0))


class _Fc1(_Routed):
    """Launch 1: routing, FC1 on the tensor core (tcgen05, accumulated in TMEM over all of
    K), SwiGLU and the requantization into the slot records."""

    def __init__(self, *, mt, slots, grid, pdl):
        assert mt in ROW_BUCKETS
        self.mt, self.slots, self.grid, self.pdl = mt, slots, grid, pdl
        self.o_ring = 0
        self.o_b = slots * SLOT1
        self.o_sfb = self.o_b + (K1 // 32) * NB_ROW
        self.o_rt = self.o_sfb + SFB_BYTES(K1)
        self.o_scal = self.o_rt + -(-self.RT_BYTES // 128) * 128
        self.o_bar = self.o_scal + MAX_ITEMS * 8
        self.o_tm = self.o_bar + (2 * slots + 4) * 8
        self.smem = self.o_tm + 16 + 1024
        assert self.smem <= _SMEM_BUDGET, (self.smem, mt, slots)
        # TMEM: two 8-column accumulators, 32 weight-scale columns a slot, the 160
        # activation-scale columns.
        self.sfb_col = SFA1_COL + 32 * slots
        self.tcols = 256 if self.sfb_col + (K1 // 64) * 4 <= 256 else 512

    @cute.jit
    def __call__(
        self,
        rec_t: cute.Tensor,
        lg_ptr: cute.Pointer,
        xq_ptr: cute.Pointer,
        xsf_ptr: cute.Pointer,
        tmap_ptr: cute.Pointer,
        s_ptr: cute.Pointer,
        o1_ptr: cute.Pointer,
        o1g_ptr: cute.Pointer,
        topw_ptr: cute.Pointer,
        rows: cutlass.Int32,
        stream,
    ):
        self.kernel(
            rec_t.iterator.toint(),
            lg_ptr.toint(),
            xq_ptr.toint(),
            xsf_ptr.toint(),
            tmap_ptr.toint(),
            s_ptr.toint(),
            o1_ptr.toint(),
            o1g_ptr.toint(),
            topw_ptr.toint(),
            rows,
        ).launch(
            grid=[self.grid, 1, 1],
            block=[TC_THREADS, 1, 1],
            smem=self.smem,
            stream=stream,
            use_pdl=self.pdl,
            preferred_smem_carveout=100,
        )

    @cute.kernel
    def kernel(
        self,
        rec_base: cutlass.Int64,
        lg_base: cutlass.Int64,
        xq_base: cutlass.Int64,
        xsf_base: cutlass.Int64,
        tmap: cutlass.Int64,
        s_base: cutlass.Int64,
        o1_base: cutlass.Int64,
        o1g_base: cutlass.Int64,
        topw_base: cutlass.Int64,
        rows: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        mt: cutlass.Constexpr = self.mt
        slots: cutlass.Constexpr = self.slots
        grid: cutlass.Constexpr = self.grid

        smem = cutlass.utils.SmemAllocator()
        base_t = smem.allocate_tensor(
            cutlass.Int8, cute.make_layout((self.smem - 1024,), stride=(1,)), byte_alignment=1024
        )
        sbase = cutlass.Int32(base_t.iterator.toint())
        ring = sbase + self.o_ring
        bq = sbase + self.o_b  # B: k-chunk c (16 B) of row n at c * 128 + n * 16
        sfb = sbase + self.o_sfb  # SFB: k-step j's atom at j * 128, row n's 4 scales at n * 16
        rt = sbase + self.o_rt
        scal = sbase + self.o_scal  # [item] (output1_scale_scalar, output1_scale_gate_scalar)
        full = sbase + self.o_bar
        empty = full + 8 * slots
        accf = empty + 8 * slots
        acce = accf + 16
        tmw = sbase + self.o_tm

        if warp == 0:
            if lane == 0:
                for sl in cutlass.range_constexpr(slots):
                    _mbar_init(full + 8 * sl, 1)
                    _mbar_init(empty + 8 * sl, 1)
                for ab in cutlass.range_constexpr(2):
                    _mbar_init(accf + 8 * ab, 1)
                    _mbar_init(acce + 8 * ab, 4)
                _mbar_init_fence()
        if warp == 1:
            _tc_alloc(tmw, self.tcols)
        if tidx == 0:
            _prefetch_tmap(tmap)
        self._setup_rt(tidx, rt)
        _tc_fence(False)
        cute.arch.sync_threads()
        _tc_fence(True)
        tm = _lds32(tmw)
        if const_expr(self.pdl):
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()

        # The activation as the MMA's B operand (8 rows, rows past `rows` zero) and its scale
        # atoms, loaded before the routing so the two latencies overlap.
        n_bi: cutlass.Constexpr = -(-((K1 // 32) * 8) // TC_THREADS)
        n_si: cutlass.Constexpr = -(-((K1 // 64) * 8) // TC_THREADS)
        bv = []
        for it in cutlass.range_constexpr(n_bi):
            q = tidx + it * TC_THREADS
            n = q % 8
            c = q // 8
            ok = cutlass.Int32((q < (K1 // 32) * 8) & (n < rows))
            bv.append(_ldg_nc4p(xq_base + cutlass.Int64(n) * (K1 // 2) + c * 16, ok))
        sv = []
        for it in cutlass.range_constexpr(n_si):
            q = tidx + it * TC_THREADS
            n = q % 8
            j = q // 8
            ok = cutlass.Int32((q < (K1 // 64) * 8) & (n < rows))
            sv.append(_ldg_cg32p(xsf_base + cutlass.Int64(n) * (K1 // 16) + j * 4, ok))

        if warp < rows:
            self._route(lg_base, warp, lane, rt, topw_base, bidx == 0)
        cute.arch.sync_threads()
        self._list(tidx, rows, rt)
        cute.arch.sync_threads()
        n_units = self._ndist(rt) * NT1
        n_items = (n_units - bidx + grid - 1) // grid

        for it in cutlass.range_constexpr(n_bi):
            q = tidx + it * TC_THREADS
            if q < (K1 // 32) * 8:
                v = bv[it]
                _sts4(bq + q * 16, v[0], v[1], v[2], v[3])
        for it in cutlass.range_constexpr(n_si):
            q = tidx + it * TC_THREADS
            if q < (K1 // 64) * 8:
                _sts32(sfb + q * 16, sv[it])
        if tidx < n_items:
            e_i = _lds32(rt + self.RT_LIST + ((bidx + grid * tidx) // NT1) * 4)
            _sts_f32(scal + tidx * 8, _ldg_nc_f32(o1_base + cutlass.Int64(e_i) * 4))
            _sts_f32(scal + tidx * 8 + 4, _ldg_nc_f32(o1g_base + cutlass.Int64(e_i) * 4))
        _fence_proxy_async()
        _tc_fence(False)
        cute.arch.sync_threads()
        _tc_fence(True)

        if warp == 0:
            # ---- producer: tile (expert rank u // 10, tile u % 10), five K stages ----
            if lane == 0:
                for i in cutlass.range(n_items):
                    u = bidx + grid * i
                    rank = u // NT1
                    tile = u - rank * NT1
                    e = _lds32(rt + self.RT_LIST + rank * 4)
                    for kq in cutlass.range_constexpr(KQ1):
                        st = i * KQ1 + kq
                        sl = st % slots
                        if st >= slots:
                            _mbar_wait(empty + 8 * sl, ((st // slots) - 1) & 1)
                        slot = ring + sl * SLOT1
                        _mbar_expect_tx(full + 8 * sl, SLOT1)
                        _tma3d(slot, tmap, 0, e * N1 + tile * 128, 2 * kq, full + 8 * sl)
                        _bulk_g2s(
                            slot + A1_BYTES,
                            s_base
                            + cutlass.Int64(e) * (N1 * (K1 // 16))
                            + tile * (128 * (K1 // 16))
                            + kq * (8 * 512),
                            8 * 512,
                            full + 8 * sl,
                        )
        elif warp == 1:
            # ---- MMA issuer ----
            if lane == 0:
                dsfb = _desc_sf(sfb)
                dbq = _desc_b(bq)
                for c in cutlass.range_constexpr(K1 // 64):
                    _tc_cp_d(tm + self.sfb_col + 4 * c, dsfb, c * 128)
                for i in cutlass.range(n_items):
                    ab = i % 2
                    if i >= 2:
                        _mbar_wait(acce + 8 * ab, ((i // 2) - 1) & 1)
                        _tc_fence(True)
                    for kq in cutlass.range_constexpr(KQ1):
                        st = i * KQ1 + kq
                        sl = st % slots
                        _mbar_wait(full + 8 * sl, (st // slots) & 1)
                        _tc_fence(True)
                        slot = ring + sl * SLOT1
                        tsfa = tm + SFA1_COL + sl * 32
                        dsfa = _desc_sf(slot + A1_BYTES)
                        da = _desc_a1(slot)
                        for q in cutlass.range_constexpr(8):
                            _tc_cp_d(tsfa + 4 * q, dsfa, q * 512)
                        for q in cutlass.range_constexpr(8):
                            _mma_d(
                                tm + ab * 8,
                                da,
                                (q // 4) * 16384 + (q % 4) * 32,
                                dbq,
                                (2 * (8 * kq + q)) * NB_ROW,
                                tsfa + 4 * q,
                                tm + self.sfb_col + 4 * (8 * kq + q),
                                cutlass.Int32(0 if kq + q == 0 else 1),
                            )
                        _tc_commit(empty + 8 * sl)
                    _tc_commit(accf + 8 * ab)
        elif warp < 6:
            # ---- epilogue: TMEM lane quarter = one 32-row block = one requant block ----
            quarter = warp % 4
            trow = tm + ((quarter * 32) << 16)
            up_row = cutlass.Int32(((lane >> 3) & 1) == 0)
            for i in cutlass.range(n_items):
                ab = i % 2
                _mbar_wait(accf + 8 * ab, (i // 2) & 1)
                _tc_fence(True)
                r = _tc_ld8(trow + ab * 8)
                _tc_fence(False)
                cute.arch.sync_warp()
                if lane == 0:
                    _mbar_arrive(acce + 8 * ab)
                u = bidx + grid * i
                rank = u // NT1
                b = (u - rank * NT1) * 4 + quarter
                s_up = _lds_f32(scal + i * 8)
                s_gt = _lds_f32(scal + i * 8 + 4)
                tmask = _lds32(rt + self.RT_MASK + rank * 4)
                for n in cutlass.range_constexpr(mt):
                    # Unit row d = lane: up rows have bit 3 clear, their gate row is d + 8.
                    gt = _shfl_bfly_f32(r[n], cutlass.Int32(8)) * s_gt
                    act = (r[n] * s_up) * _div_rn(gt, 1.0 + _expf(-gt))
                    amax = _sel_f(up_row, _fabs(act), 0.0)
                    for sh in cutlass.range_constexpr(5):
                        amax = _fmax(amax, _shfl_bfly_f32(amax, cutlass.Int32(16 >> sh)))
                    sfb_, sff = _e4m3_sf(amax)
                    nib = _e2m1_lo(_div_rn(act, _pos_or_one(sff)))
                    # Intermediate column j = 2 (d % 8) + d // 16: byte d % 8, high nibble
                    # from rows 16 .. 23.
                    byte = nib | (_shfl_down_i32(nib, cutlass.Int32(16)) << 4)
                    w = (
                        byte
                        | (_shfl_down_i32(byte, cutlass.Int32(1)) << 8)
                        | (_shfl_down_i32(byte, cutlass.Int32(2)) << 16)
                        | (_shfl_down_i32(byte, cutlass.Int32(3)) << 24)
                    )
                    w1 = _shfl_down_i32(w, cutlass.Int32(4))
                    if (lane == 0) & (n < rows) & (((tmask >> n) & 1) == 1):
                        slot_k = _lds_u8(rt + self.RT_SLOT + rank * 8 + n)
                        recp = rec_base + cutlass.Int64(n * TOP_K + slot_k) * REC
                        _stg_v2(recp + b * 8, w, w1)
                        _stg_u8(recp + 320 + b, sfb_)
        _tc_fence(False)
        cute.arch.sync_threads()
        if warp == 1:
            _tc_dealloc(tm, self.tcols)


class _Fc2(_Routed):
    """Launch 2: routing, its FC2 tiles and their scales into shared memory / TMEM before the
    wait; after it the B operand from the slot records, the MMAs and the bf16 rows."""

    def __init__(self, *, mt, slots, grid, pdl):
        assert mt in ROW_BUCKETS
        self.mt, self.slots, self.grid, self.pdl = mt, slots, grid, pdl
        # Batch mode (4+ slots: every unit of the CTA in shared memory before the wait, the CTA
        # alone on its SM): one B buffer, accumulator and TMEM scale range per unit, all 512
        # TMEM columns; otherwise two of each, double-buffered.
        self.batch = slots >= 4
        self.nb = slots if self.batch else 2
        self.tcols = 512 if self.batch else TMEM_COLS
        self.o_ring = 0
        self.o_b = slots * SLOT2
        self.o_rt = self.o_b + self.nb * B2_BYTES
        self.o_s2 = self.o_rt + -(-self.RT_BYTES // 128) * 128
        self.o_bar = self.o_s2 + MAX_ITEMS * 4
        self.o_tm = self.o_bar + (2 * slots + 4 + 4 * self.nb) * 8
        self.smem = self.o_tm + 16 + 1024
        self.acc_cols = 8 * self.nb
        self.sfa_col = self.acc_cols
        self.sfb_col = self.sfa_col + 40 * slots
        assert self.sfb_col + 40 * self.nb <= self.tcols, slots
        assert self.smem <= _SMEM_BUDGET, (self.smem, mt, slots)

    @cute.jit
    def __call__(
        self,
        out_t: cute.Tensor,
        lg_ptr: cute.Pointer,
        rec_ptr: cute.Pointer,
        tmap_ptr: cute.Pointer,
        s_ptr: cute.Pointer,
        o2_ptr: cute.Pointer,
        w_ptr: cute.Pointer,
        rows: cutlass.Int32,
        stream,
    ):
        self.kernel(
            out_t.iterator.toint(),
            lg_ptr.toint(),
            rec_ptr.toint(),
            tmap_ptr.toint(),
            s_ptr.toint(),
            o2_ptr.toint(),
            w_ptr.toint(),
            rows,
        ).launch(
            grid=[self.grid, 1, 1],
            block=[TC_THREADS, 1, 1],
            smem=self.smem,
            stream=stream,
            use_pdl=self.pdl,
            preferred_smem_carveout=100,
        )

    @cute.kernel
    def kernel(
        self,
        out_base: cutlass.Int64,
        lg_base: cutlass.Int64,
        rec_base: cutlass.Int64,
        tmap: cutlass.Int64,
        s_base: cutlass.Int64,
        o2_base: cutlass.Int64,
        w_base: cutlass.Int64,
        rows: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        mt: cutlass.Constexpr = self.mt
        slots: cutlass.Constexpr = self.slots
        grid: cutlass.Constexpr = self.grid

        smem = cutlass.utils.SmemAllocator()
        base_t = smem.allocate_tensor(
            cutlass.Int8, cute.make_layout((self.smem - 1024,), stride=(1,)), byte_alignment=1024
        )
        sbase = cutlass.Int32(base_t.iterator.toint())
        ring = sbase + self.o_ring
        bb = sbase + self.o_b  # [2] (B 2560 B, SFB atoms)
        rt = sbase + self.o_rt
        s2b = sbase + self.o_s2  # [item] output2_scale_scalar
        nb: cutlass.Constexpr = self.nb
        full = sbase + self.o_bar
        empty = full + 8 * slots
        accf = empty + 8 * slots
        acce = accf + 8 * nb
        bready = acce + 8 * nb
        bfree = bready + 8 * nb
        tmw = sbase + self.o_tm

        if warp == 0:
            if lane == 0:
                for sl in cutlass.range_constexpr(slots):
                    _mbar_init(full + 8 * sl, 1)
                    _mbar_init(empty + 8 * sl, 1)
                for pb in cutlass.range_constexpr(nb):
                    _mbar_init(accf + 8 * pb, 1)
                    _mbar_init(acce + 8 * pb, 4)
                    _mbar_init(bready + 8 * pb, 1)
                    _mbar_init(bfree + 8 * pb, 1)
                _mbar_init_fence()
        if warp == 1:
            _tc_alloc(tmw, self.tcols)
        if tidx == 0:
            _prefetch_tmap(tmap)
        self._setup_rt(tidx, rt)
        _tc_fence(False)
        cute.arch.sync_threads()
        _tc_fence(True)
        tm = _lds32(tmw)

        # Routing before the wait: the logits are complete (launch 1 released this grid
        # only after its own wait on the router GEMM), and are read through L2.
        if warp < rows:
            self._route(lg_base, warp, lane, rt, lg_base, False)
        cute.arch.sync_threads()
        self._list(tidx, rows, rt)
        cute.arch.sync_threads()
        n_units = self._ndist(rt) * NT2
        n_items = (n_units - bidx + grid - 1) // grid
        if (warp >= 2) & (tidx - 64 < n_items):
            e_i = _lds32(rt + self.RT_LIST + ((bidx + grid * (tidx - 64)) // NT2) * 4)
            _sts_f32(s2b + (tidx - 64) * 4, _ldg_nc_f32(o2_base + cutlass.Int64(e_i) * 4))

        if warp == 0:
            # ---- producer: the first `slots` tiles now, L2 prefetch of the rest ----
            if lane == 0:
                for i in cutlass.range(n_items):
                    u = bidx + grid * i
                    rank = u // NT2
                    tile = u - rank * NT2
                    e = _lds32(rt + self.RT_LIST + rank * 4)
                    sl = i % slots
                    sa = s_base + cutlass.Int64(e) * (N2 * (K2 // 16)) + tile * (128 * (K2 // 16))
                    if i < slots:
                        slot = ring + sl * SLOT2
                        _mbar_expect_tx(full + 8 * sl, SLOT2)
                        _tma3d(slot, tmap, 0, e * N2 + tile * 128, 0, full + 8 * sl)
                        _bulk_g2s(slot + A2_BYTES, sa, 10 * 512, full + 8 * sl)
                    else:
                        _bulk_prefetch_l2(
                            w_base + (cutlass.Int64(e) * N2 + tile * 128) * (K2 // 2),
                            128 * (K2 // 2),
                        )
                        _bulk_prefetch_l2(sa, 10 * 512)
                for i in cutlass.range(slots, n_items):
                    u = bidx + grid * i
                    rank = u // NT2
                    tile = u - rank * NT2
                    e = _lds32(rt + self.RT_LIST + rank * 4)
                    sl = i % slots
                    _mbar_wait(empty + 8 * sl, ((i // slots) - 1) & 1)
                    slot = ring + sl * SLOT2
                    _mbar_expect_tx(full + 8 * sl, SLOT2)
                    _tma3d(slot, tmap, 0, e * N2 + tile * 128, 0, full + 8 * sl)
                    _bulk_g2s(
                        slot + A2_BYTES,
                        s_base + cutlass.Int64(e) * (N2 * (K2 // 16)) + tile * (128 * (K2 // 16)),
                        10 * 512,
                        full + 8 * sl,
                    )
        elif warp == 1:
            # ---- MMA issuer: weight scales to TMEM as tiles land (before the wait too) ----
            if lane == 0:
                # The weight scales of the units already in their slots, before the wait.
                for i in cutlass.range(cutlass.min(n_items, slots)):
                    slot = ring + i * SLOT2
                    _mbar_wait(full + 8 * i, 0)
                    _tc_fence(True)
                    dsfa = _desc_sf(slot + A2_BYTES)
                    for q in cutlass.range_constexpr(K2 // 64):
                        _tc_cp_d(tm + self.sfa_col + i * 40 + 4 * q, dsfa, q * 512)
                for i in cutlass.range(n_items):
                    sl = i % slots
                    pb = i % nb
                    slot = ring + sl * SLOT2
                    _mbar_wait(full + 8 * sl, (i // slots) & 1)
                    _tc_fence(True)
                    tsfa = tm + self.sfa_col + sl * 40
                    if i >= slots:
                        dsfa = _desc_sf(slot + A2_BYTES)
                        for q in cutlass.range_constexpr(K2 // 64):
                            _tc_cp_d(tsfa + 4 * q, dsfa, q * 512)
                    if const_expr(self.batch):
                        _mbar_wait(bready, 0)
                    else:
                        _mbar_wait(bready + 8 * pb, (i // 2) & 1)
                    _tc_fence(True)
                    tsfb = tm + self.sfb_col + pb * 40
                    bp = bb + pb * B2_BYTES
                    dsfb = _desc_sf(bp + (K2 // 32) * NB_ROW)
                    dbp = _desc_b(bp)
                    da = _desc_a2(slot)
                    for q in cutlass.range_constexpr(K2 // 64):
                        _tc_cp_d(tsfb + 4 * q, dsfb, q * 128)
                    if const_expr(not self.batch):
                        if i >= 2:
                            _mbar_wait(acce + 8 * pb, ((i // 2) - 1) & 1)
                            _tc_fence(True)
                    for q in cutlass.range_constexpr(K2 // 64):
                        _mma_d(
                            tm + pb * 8,
                            da,
                            (q // 2) * 8192 + (q % 2) * 32,
                            dbp,
                            (2 * q) * NB_ROW,
                            tsfa + 4 * q,
                            tsfb + 4 * q,
                            cutlass.Int32(0 if q == 0 else 1),
                        )
                    if const_expr(not self.batch):
                        _tc_commit(empty + 8 * sl)
                        _tc_commit(bfree + 8 * pb)
                    _tc_commit(accf + 8 * pb)
        elif warp >= 6:
            # ---- B builders: B zeroed before the wait; after it each valid row's record
            # pieces (20 x 16-byte code chunks, 10 scale words) by cp.async from global ----
            ptid = tidx - 192
            for it in cutlass.range_constexpr(-(-(nb * B2_BYTES // 16) // 64)):
                q = ptid + it * 64
                if q < nb * B2_BYTES // 16:
                    z = cutlass.Int32(0)
                    _sts4(bb + q * 16, z, z, z, z)
            _bar_sync(2, 64)
            if const_expr(self.pdl):
                cute.arch.griddepcontrol_wait()
                cute.arch.griddepcontrol_launch_dependents()
            n_pc: cutlass.Constexpr = mt * ((K2 // 32) + (K2 // 64))
            for i in cutlass.range(n_items):
                pb = i % nb
                u = bidx + grid * i
                rank = u // NT2
                tmask = _lds32(rt + self.RT_MASK + rank * 4)
                if const_expr(not self.batch):
                    if i >= 2:
                        _mbar_wait(bfree + 8 * pb, ((i // 2) - 1) & 1)
                        for it in cutlass.range_constexpr(-(-(B2_BYTES // 16) // 64)):
                            q = ptid + it * 64
                            if q < B2_BYTES // 16:
                                z = cutlass.Int32(0)
                                _sts4(bb + pb * B2_BYTES + q * 16, z, z, z, z)
                        _bar_sync(2, 64)
                bp = bb + pb * B2_BYTES
                for it in cutlass.range_constexpr(-(-n_pc // 64)):
                    q = ptid + it * 64
                    n = q // ((K2 // 32) + (K2 // 64))
                    pc = q - n * ((K2 // 32) + (K2 // 64))
                    if (q < n_pc) & (n < rows):
                        if ((tmask >> n) & 1) == 1:
                            sk = _lds_u8(rt + self.RT_SLOT + rank * 8 + n)
                            rec = rec_base + cutlass.Int64(n * TOP_K + sk) * REC
                            if pc < K2 // 32:
                                _cp_async16(bp + pc * NB_ROW + n * 16, rec + pc * 16)
                            else:
                                j = pc - K2 // 32
                                _cp_async4g(
                                    bp + (K2 // 32) * NB_ROW + j * 128 + n * 16, rec + 320 + j * 4
                                )
                if const_expr(not self.batch):
                    _cp_async_wait_all()
                    _fence_proxy_async()
                    _bar_sync(2, 64)
                    if ptid == 0:
                        _mbar_arrive(bready + 8 * pb)
            if const_expr(self.batch):
                _cp_async_wait_all()
                _fence_proxy_async()
                _bar_sync(2, 64)
                if ptid == 0:
                    _mbar_arrive(bready)
        else:
            # ---- epilogue: TMEM lane quarter = 32 output rows ----
            if const_expr(self.pdl):
                cute.arch.griddepcontrol_wait()
                cute.arch.griddepcontrol_launch_dependents()
            quarter = warp % 4
            trow = tm + ((quarter * 32) << 16)
            for i in cutlass.range(n_items):
                pb = i % nb
                _mbar_wait(accf + 8 * pb, (i // nb) & 1)
                _tc_fence(True)
                r = _tc_ld8(trow + pb * 8)
                _tc_fence(False)
                cute.arch.sync_warp()
                if lane == 0:
                    _mbar_arrive(acce + 8 * pb)
                u = bidx + grid * i
                rank = u // NT2
                b = (u - rank * NT2) * 4 + quarter
                # Unit row d = lane is output column 32 b + 4 (d % 8) + d // 8.
                col = b * BLK + 4 * (lane % 8) + lane // 8
                s2 = _lds_f32(s2b + i * 4)
                tmask = _lds32(rt + self.RT_MASK + rank * 4)
                for n in cutlass.range_constexpr(mt):
                    if (n < rows) & (((tmask >> n) & 1) == 1):
                        sk = _lds_u8(rt + self.RT_SLOT + rank * 8 + n)
                        _stg_b16(
                            out_base + (cutlass.Int64(n * TOP_K + sk) * N2 + col) * 2,
                            _bf16_bits_f32(r[n] * s2)[0],
                        )
        _tc_fence(False)
        cute.arch.sync_threads()
        if warp == 1:
            _tc_dealloc(tm, self.tcols)


# ------------------------------------------------------------------------------------------
# host side
# ------------------------------------------------------------------------------------------


def _bucket(rows: int) -> int:
    for b in ROW_BUCKETS:
        if rows <= b:
            return b
    raise AssertionError(f"{rows} rows > MAX_ROWS ({MAX_ROWS})")


#: Launch plan by row bucket: (launch 1 slots, launch 2 slots, launch 1 grid, launch 2 grid).
#: One FC1 stage (36 KB) or FC2 tile (45 KB) a slot. At one row the FC1 work is exactly 100
#: tiles (10 experts x 10), so launch 1 takes 100 SMs and launch 2 runs two CTAs on each of
#: the other 52, where it routes and loads all of its FC2 tiles while launch 1 runs; a grid
#: of 0 means one CTA per SM.
_PLANS = {1: (5, 2, 100, 104), 2: (5, 2, 0, 0), 4: (5, 2, 0, 0), 8: (4, 2, 0, 0)}


def _dl(t: torch.Tensor):
    return from_dlpack(t, assumed_align=16).mark_layout_dynamic(leading_dim=t.dim() - 1)


def _gptr(dtype, buf, align=16):
    return make_ptr(dtype, buf.data_ptr(), cute.AddressSpace.gmem, assumed_align=align)


@functools.lru_cache(maxsize=None)
def _compiled(mt: int, slots1: int, slots2: int, grid1: int, grid2: int, pdl: bool):
    """(launch 1, launch 2) for row bucket `mt`."""
    buf = torch.empty(64, dtype=torch.int32, device="cuda")
    rec = torch.empty(mt * TOP_K, REC, dtype=torch.uint8, device="cuda")
    out = torch.empty(mt * TOP_K, N2, dtype=torch.bfloat16, device="cuda")
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    f1 = cute.compile(
        _Fc1(mt=mt, slots=slots1, grid=grid1, pdl=pdl),
        _dl(rec),
        _gptr(cutlass.BFloat16, buf),
        _gptr(cutlass.Uint8, buf),
        _gptr(cutlass.Uint8, buf),
        _gptr(cutlass.Uint8, buf),
        _gptr(cutlass.Uint8, buf),
        _gptr(cutlass.Float32, buf),
        _gptr(cutlass.Float32, buf),
        _gptr(cutlass.BFloat16, buf),
        cutlass.Int32(1),
        stream,
        options=_COMPILE_OPTIONS,
    )
    f2 = cute.compile(
        _Fc2(mt=mt, slots=slots2, grid=grid2, pdl=pdl),
        _dl(out),
        _gptr(cutlass.BFloat16, buf),
        _gptr(cutlass.Uint8, buf),
        _gptr(cutlass.Uint8, buf),
        _gptr(cutlass.Uint8, buf),
        _gptr(cutlass.Float32, buf),
        _gptr(cutlass.Uint8, buf),
        cutlass.Int32(1),
        stream,
        options=_COMPILE_OPTIONS,
    )
    return f1, f2


def _plan_key(mt: int, pdl: bool) -> tuple:
    slots1, slots2, grid1, grid2 = _PLANS[mt]
    sms = _sms()
    grid1 = grid1 if grid1 > 0 else sms
    grid2 = sms + grid2 if grid2 < 0 else (grid2 or sms)
    return (mt, slots1, slots2, grid1, grid2, bool(pdl and _env_pdl()))


def precompile(max_rows: int = MAX_ROWS, pdl: bool = True) -> None:
    """Compile every row bucket up to `max_rows` now: a first call mid-serving would stall on
    the compile."""
    for mt in ROW_BUCKETS:
        if mt <= _bucket(max_rows):
            _compiled(*_plan_key(mt, pdl))


#: TMA descriptors of registered expert stacks, 128 bytes each in one device buffer (the
#: kernels read them from global memory), keyed by (data_ptr, operand).
_TMAP_ROWS = 1024
_tmaps: dict = {}
_tmap_buf: list = []


def _encode(ptr: int, fc: int) -> bytes:
    from cuda.bindings import driver as cu

    if fc == 1:  # FC1: (128 B, rows, 10 chunks), box (128, 128, 2), 128-byte swizzle
        ch, rows, kp, row_bytes, box_kp = 128, NUM_EXPERTS * N1, 10, K1 // 2, 2
        sw = cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B
    else:  # FC2: (64 B, rows, 5 chunks), box (64, 128, 5), 64-byte swizzle
        ch, rows, kp, row_bytes, box_kp = 64, NUM_EXPERTS * N2, 5, K2 // 2, 5
        sw = cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_64B
    err, tm = cu.cuTensorMapEncodeTiled(
        cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
        3,
        ptr,
        [cu.cuuint64_t(ch), cu.cuuint64_t(rows), cu.cuuint64_t(kp)],
        [cu.cuuint64_t(row_bytes), cu.cuuint64_t(ch)],
        [cu.cuuint32_t(ch), cu.cuuint32_t(128), cu.cuuint32_t(box_kp)],
        [cu.cuuint32_t(1)] * 3,
        cu.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        sw,
        cu.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        cu.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    )
    assert int(err) == 0, f"cuTensorMapEncodeTiled: {err}"
    return b"".join(int(v).to_bytes(8, "little") for v in tm.opaque)


def _tmap(t: torch.Tensor, fc: int) -> int:
    """Device address of `t`'s TMA descriptor (built and uploaded on first use, which must not
    be inside a CUDA-graph capture: `register_weights` does it ahead)."""
    key = (t.data_ptr(), fc)
    row = _tmaps.get(key)
    if row is None:
        assert not torch.cuda.is_current_stream_capturing(), (
            "nvfp4_moe_gemv: expert stack not registered before CUDA-graph capture "
            "(call register_weights at load)"
        )
        if not _tmap_buf:
            _tmap_buf.append(torch.zeros(_TMAP_ROWS, 128, dtype=torch.uint8, device=t.device))
        row = len(_tmaps)
        assert row < _TMAP_ROWS, "TMA descriptor table full"
        src = torch.frombuffer(bytearray(_encode(t.data_ptr(), fc)), dtype=torch.uint8)
        _tmap_buf[0][row].copy_(src)
        torch.cuda.synchronize(t.device)
        _tmaps[key] = row
    return _tmap_buf[0].data_ptr() + row * 128


def register_weights(gemm1_weights: torch.Tensor, gemm2_weights: torch.Tensor) -> None:
    """Build the TMA descriptors of one layer's expert stacks (trtllm-gen layout) now."""
    _tmap(gemm1_weights, 1)
    _tmap(gemm2_weights, 2)


@functools.lru_cache(maxsize=None)
def _topk_rows(rows: int, device: torch.device) -> torch.Tensor:
    return torch.arange(rows * TOP_K, dtype=torch.int32, device=device).view(rows, TOP_K)


def nvfp4_moe_gemv(
    routing_logits: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    output1_scale_scalar: torch.Tensor,
    output1_scale_gate_scalar: torch.Tensor,
    output2_scale_scalar: torch.Tensor,
    pdl: bool = True,
) -> tuple:
    """`(rows [10 T, 2560] bf16, topk_weights [T, 10] bf16, topk_rows [T, 10] int32)`; see
    the module docstring. Every operand dense, the expert operands in the trtllm-gen layout."""
    rows = routing_logits.shape[0]
    dev = routing_logits.device
    f1, f2 = _compiled(*_plan_key(_bucket(rows), pdl))
    t1 = _tmap(gemm1_weights, 1)
    t2 = _tmap(gemm2_weights, 2)
    rec = torch.empty(rows * TOP_K, REC, dtype=torch.uint8, device=dev)
    out = torch.empty(rows * TOP_K, N2, dtype=torch.bfloat16, device=dev)
    topw = torch.empty(rows, TOP_K, dtype=torch.bfloat16, device=dev)
    lg = routing_logits.data_ptr()
    f1(
        rec,
        lg,
        hidden_states.data_ptr(),
        hidden_states_scale.data_ptr(),
        t1,
        gemm1_weights_scale.data_ptr(),
        output1_scale_scalar.data_ptr(),
        output1_scale_gate_scalar.data_ptr(),
        topw.data_ptr(),
        rows,
    )
    f2(
        out,
        lg,
        rec.data_ptr(),
        t2,
        gemm2_weights_scale.data_ptr(),
        output2_scale_scalar.data_ptr(),
        gemm2_weights.data_ptr(),
        rows,
    )
    return out, topw, _topk_rows(rows, dev)


class Nvfp4MoeGemv(OpWrapper):
    """`fp4_block_scale_moe_renorm_permuted` for 1 .. MAX_ROWS rows, in two PDL launches.

    Same arguments and the same three outputs: `rows` `[10 T, H]` bf16 (slot `(t, k)` at row
    `10 t + k`), `topk_weights` `[T, 10]` bf16 in the runner's slot order, `topk_rows`
    `[T, 10]` int32. `finalize(*outputs)` is the layer's combined output.

    Fusion boundary: Renormalize routing, FC1, SwiGLU, the intermediate's NVFP4
    requantization and FC2; the finalize is the consumer's (the hyper-connection fold).
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(why="STP decode, one row (con=1): the 1-row bucket", spec=_ckpt(1)),
        Cell(why="2 rows: the 2-row bucket", spec=_ckpt(2)),
        Cell(why="3 rows: a partial 4-row bucket", spec=_ckpt(3)),
        Cell(why="MTP3 verify of one request (4 rows): the 4-row bucket", spec=_ckpt(4)),
        Cell(why="8 rows: the largest bucket", spec=_ckpt(8)),
        Cell(why="coarse 3-level logits: ties decide most slots", spec=_ckpt(4, dist="coarse")),
    )

    note = """
    Metadata consumed: none. Call site: `DecodeTarget._moe_block` under
    `TRTLLM_MODELING_V2_MOE_GEMV=1` at the rows in `_MOE_GEMV_ROWS` (1 .. 4) when the finalize
    folds into the next site. Preconditions: every operand contiguous; the expert stacks in
    the trtllm-gen layout (FC1 rows interleaved and 32-row block shuffled, FC2 rows block
    shuffled, both scale stacks 128x4 swizzled) and registered (`register_weights`) before a
    CUDA-graph capture; `hidden_states_scale` the linear e4m3 layout. Two PDL launches;
    CUDA-graph capturable; one compile per row bucket (`precompile` builds them all), the row
    count a launch argument.

    Measured on GB300, us a layer, 48 distinct layers each behind its router projection in
    one CUDA graph, rows correlated 0.85 (runner -> this): 14.42 -> 12.64 at 1 row, 20.59 ->
    17.99 at 2, 23.37 -> 22.72 at 4, 28.41 -> 31.59 at 8 (slower: not dispatched).

    Numerics: the routing is the runner's (bitwise ids and weights). The products of e2m1
    codes and e4m3 scales are exact and accumulate in fp32 on the tensor core; the SwiGLU is
    `up * (gate / (1 + expf(-gate)))` in fp32 and the intermediate is requantized as
    `quantize_nvfp4` defines it. Measured bitwise equal to the runner's slot rows on the
    cells and on real checkpoint layers.

    Hard limits
      T            1 .. MAX_ROWS (8); buckets 1, 2, 4, 8
      geometry     512 experts, top-10, hidden 2560, intermediate 640
      TMA          `_TMAP_ROWS` registered expert stacks per process
    """

    def __call__(
        self,
        routing_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor,
        gemm1_weights: torch.Tensor,
        gemm1_weights_scale: torch.Tensor,
        gemm2_weights: torch.Tensor,
        gemm2_weights_scale: torch.Tensor,
        output1_scale_scalar: torch.Tensor,
        output1_scale_gate_scalar: torch.Tensor,
        output2_scale_scalar: torch.Tensor,
        num_experts: int,
        top_k: int,
        intermediate_size: int,
    ) -> tuple:
        return nvfp4_moe_gemv(
            routing_logits,
            hidden_states,
            hidden_states_scale,
            gemm1_weights,
            gemm1_weights_scale,
            gemm2_weights,
            gemm2_weights_scale,
            output1_scale_scalar,
            output1_scale_gate_scalar,
            output2_scale_scalar,
        )

    def reference(self, *args, **kwargs) -> torch.Tensor:
        """The finalized entry's fp32 reference `[T, H]`: what `finalize(*self(...))` must
        answer (same arguments as that entry's `reference`)."""
        return _routed.reference(*args, **kwargs)

    def is_valid(
        self,
        routing_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor,
        gemm1_weights: torch.Tensor,
        gemm1_weights_scale: torch.Tensor,
        gemm2_weights: torch.Tensor,
        gemm2_weights_scale: torch.Tensor,
        output1_scale_scalar: torch.Tensor,
        output1_scale_gate_scalar: torch.Tensor,
        output2_scale_scalar: torch.Tensor,
        num_experts: int,
        top_k: int,
        intermediate_size: int,
    ) -> None:
        for name, tensor in (
            ("routing_logits", routing_logits),
            ("hidden_states", hidden_states),
            ("hidden_states_scale", hidden_states_scale),
            ("gemm1_weights", gemm1_weights),
            ("gemm1_weights_scale", gemm1_weights_scale),
            ("gemm2_weights", gemm2_weights),
            ("gemm2_weights_scale", gemm2_weights_scale),
            ("output1_scale_scalar", output1_scale_scalar),
            ("output1_scale_gate_scalar", output1_scale_gate_scalar),
            ("output2_scale_scalar", output2_scale_scalar),
        ):
            assert tensor.is_contiguous(), (
                f"{name} must be contiguous; the kernels address it from data_ptr() as dense"
            )
        assert (num_experts, top_k) in CERTIFIED_PAIRS and num_experts == NUM_EXPERTS
        assert intermediate_size == INTER and hidden_states.shape[1] * 2 == HIDDEN
        assert routing_logits.dtype == torch.bfloat16, "bf16 router logits"
        assert 1 <= routing_logits.shape[0] <= MAX_ROWS, f"1 .. {MAX_ROWS} rows"


nvfp4_moe_gemv_op = Nvfp4MoeGemv()

__all__ = ["finalize", "nvfp4_moe_gemv", "nvfp4_moe_gemv_op", "precompile", "MAX_ROWS"]
