# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sparse MLA attention for 64 query heads on sm_103 (its row maxima use ``tcgen05.ld.red``), CuTe DSL.

One query token (64 heads) at a time per CTA; persistent CTAs walk tokens ``b, b + grid, ...``. A token's
selected bank rows (``indices[t, :lengths[t]]``) are visited in tiles of 64 keys; every 512-byte E4M3 row is
gathered once into a ``[128-byte column chunk][key][128 bytes]`` stage with the 128-byte swizzle, which is the
K-major layout of K (keys x 512) and the MN-major layout of V^T (512 x keys) over the same bytes.

- QK: ``S[64 heads x 64 keys] = Q K^T``, tcgen05 M=64 N=64 K=32 (Q and K from shared memory).
- PV, swapped: ``O^T[512 dims x 64 heads] += V^T[dims x keys] P^T[keys x heads]``, M=128 (one 128-dim block)
  N=64 K=32; P is the softmax's E4M3 ``P[head][key]`` (K-major B operand, no swizzle).

TMEM (512 columns): S in columns [0, 64), two buffers in the two 16-lane halves of the M=64 accumulator
(lanes 0-15 / 16-31 of every 32-lane sub-partition); O^T in 7 slots of 64 columns [64, 512): block c of the
CTA's it-th token lives in slot (4 it + c) % 7, so the epilogue drains a token while the next one accumulates.

Softmax: lazy running maximum per head (O^T is rescaled only when a tile's maximum exceeds the reference by more
than 2^RESCALE_LOG2), P carries a 2^P_OFFSET_LOG2 offset (P <= 2^8.8 stays inside E4M3), the denominator sums the
E4M3-rounded P that the MMA accumulates, and the attention sink joins the final normalization.

Warps: 0-3 gather KV rows (16-byte cp.async), 4 QK MMA + TMEM allocation, 5 PV MMA, 6 Q (TMA)
and index lists (bulk copies) one token ahead, 8-11 softmax (sub-partition q: heads 16q..16q+15, one half row of
32 keys per thread), 12-15 O^T correction + epilogue (sub-partition q: dims 32q..32q+31 of every block).
"""

from __future__ import annotations

import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import BFloat16, Float8E4M3FN, Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.nvgpu.common import OperandMajorMode
from cutlass.cutlass_dsl import T, dsl_user_op

from .kernel import (
    _asm,
    _tma_desc,
    bulk_load,
    cp_async_16,
    cp_async_16_zero,
    cp_async_arrive_noinc,
    e4m3_sum16,
    exp2_f16x2_fma2,
    pack_e4m3x4,
    pack_e4m3x4_f16x2,
    pack_f32x2,
    scale_bf16x2,
    st_shared_v4,
    tcgen05_after_thread_sync,
    tcgen05_before_thread_sync,
    tma_load_tile,
    tma_store_tile,
    tree_max,
)

try:
    from cutlass.memory import SmemAllocator
except ImportError:  # older DSL releases
    from cutlass.utils import SmemAllocator

HEADS = 64
DIM = 512
KEYS = 64
# KV ring depth: 64-key stages of 32 KB, which leave room for one 16 KB output staging buffer.
KV_STAGES = 4
# S buffers (b: lane half b of the M=64 accumulator in TMEM columns [0, 64)) and P buffers (4 KB of shared memory
# each); O^T takes the 7 slots of 64 columns after the S columns.
S_BUFS = 2
P_BUFS = 2
O_SLOTS = 7
O_COL0 = 512 - 64 * O_SLOTS
# Longest per-token index list the shared-memory list buffers hold (128 window rows + 512 selected rows).
LIST_COLS = 640
# Per-tile row-factor slots (softmax -> correction warps): the softmax may run this many non-first tiles ahead.
STAT_SLOTS = 8
THREADS = 512
GATHER_WARPS = 4
QK_WARP = 4
PV_WARP = 5
Q_WARP = 6
SM_WARP0 = 8
CO_WARP0 = 12
# Named barriers: the four correction warps; the two warps writing one 64-dim output box.
CO_BAR = 1
BOX_BAR0 = 2
RESCALE_LOG2 = 4.8
P_OFFSET_LOG2 = 4.0
LOG2E = 1.4426950408889634
STAGE_BYTES = KEYS * DIM
Q_BYTES = HEADS * DIM
P_BYTES = HEADS * KEYS
# Epilogue staging: one 128-dim block of BF16 O (64 heads x 128 dims) as two 64 x 64 boxes, one buffer.
OBUF_BYTES = HEADS * 128 * 2

# mbarrier slots
Q_FULL = 0
Q_EMPTY = Q_FULL + 2
LIST_FULL = Q_EMPTY + 2
LIST_EMPTY = LIST_FULL + 2
KV_FULL = LIST_EMPTY + 2
KV_EMPTY = KV_FULL + KV_STAGES
S_FULL = KV_EMPTY + KV_STAGES
S_EMPTY = S_FULL + S_BUFS
P_FULL = S_EMPTY + S_BUFS
P_EMPTY = P_FULL + P_BUFS
STATS_FULL = P_EMPTY + P_BUFS
STATS_EMPTY = STATS_FULL + STAT_SLOTS
O_DONE = STATS_EMPTY + STAT_SLOTS
O_READY = O_DONE + 1
O_FULL = O_READY + 1
O_FREE = O_FULL + 1
EPI_FULL = O_FREE + O_SLOTS
EPI_EMPTY = EPI_FULL + 2
NUM_BARS = EPI_EMPTY + 2


@dsl_user_op
def tmem_ld_32x32b_x8_x4(a0, a1, a2, a3, *, loc=None, ip=None):
    """Four 32x32b.x8 loads (thread t: lane taddr.lane + t, 8 columns at each address) and one wait: 32 values,
    address i's columns at [8 i, 8 i + 8)."""
    f32 = T.f32()
    text = (
        "\n".join(
            "tcgen05.ld.sync.aligned.32x32b.x8.b32 {"
            + ", ".join(f"${8 * i + k}" for k in range(8))
            + "}, [$"
            + str(32 + i)
            + "];"
            for i in range(4)
        )
        + "\ntcgen05.wait::ld.sync.aligned;"
    )
    res = _asm(
        [Int32(a).ir_value(loc=loc, ip=ip) for a in (a0, a1, a2, a3)],
        text,
        ",".join(["=f"] * 32 + ["r"] * 4),
        result=llvm.StructType.get_literal([f32] * 32),
        loc=loc,
        ip=ip,
    )
    return [Float32(llvm.extractvalue(f32, res, [k], loc=loc, ip=ip)) for k in range(32)]


@dsl_user_op
def tmem_st_32x32b_x8_x4(a0, a1, a2, a3, vals, *, loc=None, ip=None):
    """Four 32x32b.x8 stores (values [8 i, 8 i + 8) to address i) without a wait (``tmem_wait_st`` before the
    stores are relied on)."""
    text = "\n".join(
        "tcgen05.st.sync.aligned.32x32b.x8.b32 [$"
        + str(i)
        + "], {"
        + ", ".join(f"${4 + 8 * i + k}" for k in range(8))
        + "};"
        for i in range(4)
    )
    _asm(
        [Int32(a).ir_value(loc=loc, ip=ip) for a in (a0, a1, a2, a3)]
        + [Float32(v).ir_value(loc=loc, ip=ip) for v in vals],
        text,
        ",".join(["r"] * 4 + ["f"] * 32),
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def tmem_ld_16x256b_x8_pair(a0, a1, *, loc=None, ip=None):
    """Two 16x256b.x8 loads (16 lanes from each address, 64 columns) and one wait: 64 values; load i's register
    4 k + e (column group k = 0..7, e = 0..3) holds lane (t / 4) + 8 (e / 2), column 8 k + 2 (t % 4) + e % 2."""
    f32 = T.f32()
    text = (
        "\n".join(
            "tcgen05.ld.sync.aligned.16x256b.x8.b32 {"
            + ", ".join(f"${32 * i + k}" for k in range(32))
            + "}, [$"
            + str(64 + i)
            + "];"
            for i in range(2)
        )
        + "\ntcgen05.wait::ld.sync.aligned;"
    )
    res = _asm(
        [Int32(a).ir_value(loc=loc, ip=ip) for a in (a0, a1)],
        text,
        ",".join(["=f"] * 64 + ["r"] * 2),
        result=llvm.StructType.get_literal([f32] * 64),
        loc=loc,
        ip=ip,
    )
    return [Float32(llvm.extractvalue(f32, res, [k], loc=loc, ip=ip)) for k in range(64)]


@dsl_user_op
def stmatrix_x4_trans(addr, w0, w1, w2, w3, *, loc=None, ip=None):
    """Four 8x8 b16 matrices (register i = matrix i: row t / 4, columns 2 (t % 4), +1), stored transposed: thread
    8 i + r gives the shared address of row r of matrix i's transpose (16 contiguous bytes)."""
    _asm(
        [Int32(x).ir_value(loc=loc, ip=ip) for x in (addr, w0, w1, w2, w3)],
        "stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [$0], {$1, $2, $3, $4};",
        "r, r, r, r, r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def tmem_ld_red_16x32bx2_x32(taddr, *, loc=None, ip=None):
    """``tcgen05.ld.red.16x32bx2.x32 .max`` (split offset 32 columns) and its wait in one asm block: thread t < 16
    receives lane taddr.lane + t, columns [col, col + 32), thread t >= 16 lane t - 16, columns [col + 32, col + 64).
    Returns the 32 values and their maximum."""
    f32 = T.f32()
    regs = ", ".join(f"${k}" for k in range(32))
    text = (
        f"tcgen05.ld.red.sync.aligned.16x32bx2.x32.f32.max {{{regs}}}, $32, [$33], 32;\n"
        "tcgen05.wait::ld.sync.aligned;"
    )
    res = _asm(
        [Int32(taddr).ir_value(loc=loc, ip=ip)],
        text,
        ",".join(["=f"] * 33 + ["r"]),
        result=llvm.StructType.get_literal([f32] * 33),
        loc=loc,
        ip=ip,
    )
    vals = [Float32(llvm.extractvalue(f32, res, [k], loc=loc, ip=ip)) for k in range(32)]
    return vals, Float32(llvm.extractvalue(f32, res, [32], loc=loc, ip=ip))


@dsl_user_op
def ld_shared_bf16x8(addr, *, loc=None, ip=None):
    """Eight consecutive BF16 values of shared memory at ``addr`` (16-byte aligned) as FP32 (exact), in order."""
    f32 = T.f32()
    text = (
        "{\n.reg .b32 w0, w1, w2, w3, t;\nld.shared.v4.b32 {w0, w1, w2, w3}, [$8];\n"
        + "".join(
            f"shl.b32 t, w{i}, 16;\nmov.b32 ${2 * i}, t;\nand.b32 t, w{i}, 0xffff0000;\nmov.b32 ${2 * i + 1}, t;\n"
            for i in range(4)
        )
        + "}"
    )
    res = _asm(
        [Int32(addr).ir_value(loc=loc, ip=ip)],
        text,
        ",".join(["=f"] * 8 + ["r"]),
        result=llvm.StructType.get_literal([f32] * 8),
        loc=loc,
        ip=ip,
    )
    return [Float32(llvm.extractvalue(f32, res, [k], loc=loc, ip=ip)) for k in range(8)]


@dsl_user_op
def f32_op(text, *args, loc=None, ip=None) -> Float32:
    """One FP32 PTX instruction ``text`` ("$0" result, "$1".. operands) on FP32 ``args``, as written
    (no contraction)."""
    return Float32(
        _asm(
            [Float32(a).ir_value(loc=loc, ip=ip) for a in args],
            text,
            ",".join(["=f"] + ["f"] * len(args)),
            result=T.f32(),
            loc=loc,
            ip=ip,
            pure=True,
        )
    )


def rope_inv_pair(x0, x1, c, sn):
    """Inverse GPT-J rotation of one pair: r0 = fma(x0, c, x1 * s), r1 = fma(x1, c, -(x0 * s)), the products
    rounded. Two single-result asm blocks: a pure inline asm with several results crashes this DSL build's
    compiler."""
    r0 = f32_op(
        "{\n.reg .f32 a;\nmul.rn.f32 a, $2, $4;\nfma.rn.f32 $0, $1, $3, a;\n}", x0, x1, c, sn
    )
    r1 = f32_op(
        "{\n.reg .f32 b;\nmul.rn.f32 b, $1, $4;\nneg.f32 b, b;\nfma.rn.f32 $0, $2, $3, b;\n}",
        x0,
        x1,
        c,
        sn,
    )
    return r0, r1


@dsl_user_op
def select_f32(flag, a, b, *, loc=None, ip=None) -> Float32:
    """``a`` where ``flag`` != 0, else ``b`` (selp, no branch)."""
    return Float32(
        _asm(
            [
                Int32(flag).ir_value(loc=loc, ip=ip),
                Float32(a).ir_value(loc=loc, ip=ip),
                Float32(b).ir_value(loc=loc, ip=ip),
            ],
            "{\n.reg .pred p;\nsetp.ne.s32 p, $1, 0;\nselp.f32 $0, $2, $3, p;\n}",
            "=f, r, f, f",
            result=T.f32(),
            loc=loc,
            ip=ip,
            pure=True,
        )
    )


def absmax_tree(vals):
    """max |v| over ``vals`` with max.f32 (a NaN operand yields the other one), as a balanced tree."""
    parts = [f32_op("abs.f32 $0, $1;", v) for v in vals]
    while len(parts) > 1:
        parts = [
            f32_op("max.f32 $0, $1, $2;", parts[i], parts[i + 1])
            for i in range(0, len(parts) - 1, 2)
        ] + ([parts[-1]] if len(parts) % 2 else [])
    return parts[0]


@dsl_user_op
def ld_global_f32(addr, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.f32 $0, [$1];",
            "=f, l",
            result=T.f32(),
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def st_global_b32(addr, w, *, loc=None, ip=None):
    _asm(
        [Int64(addr).ir_value(loc=loc, ip=ip), Int32(w).ir_value(loc=loc, ip=ip)],
        "st.global.b32 [$0], $1;",
        "l, r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_global_b16(addr, w, *, loc=None, ip=None):
    """The low 16 bits of ``w`` to global ``addr``."""
    _asm(
        [Int64(addr).ir_value(loc=loc, ip=ip), Int32(w).ir_value(loc=loc, ip=ip)],
        "{\n.reg .b16 l, h;\nmov.b32 {l, h}, $1;\nst.global.b16 [$0], l;\n}",
        "l, r",
        loc=loc,
        ip=ip,
    )


def bf16_round(x):
    """``x`` rounded to BF16 (nearest even) and back to FP32."""
    return f32_op("{\n.reg .b16 h;\ncvt.rn.bf16.f32 h, $1;\ncvt.f32.bf16 $0, h;\n}", x)


@dsl_user_op
def mx_exponent(amax, *, loc=None, ip=None) -> Int32:
    """UE8M0 exponent of the MXFP8 block with |max| ``amax``, as ``projection._mxfp8_blocks`` computes it:
    scale = amax * (1 / 448) in FP32, rounded up to a power of two, 0 at or below 2^-127, at most 254."""
    return Int32(
        _asm(
            [Float32(amax).ir_value(loc=loc, ip=ip)],
            "{\n.reg .b32 b, e, m;\n.reg .f32 s;\n.reg .pred p, z;\nmul.rn.f32 s, $1, 0f3B124925;\n"
            "mov.b32 b, s;\nbfe.u32 e, b, 23, 8;\nand.b32 m, b, 0x7fffff;\nsetp.ne.u32 p, m, 0;\n"
            "@p add.u32 e, e, 1;\nmin.u32 e, e, 254;\nsetp.le.f32 z, s, 0f00400000;\nselp.u32 $0, 0, e, z;\n}",
            "=r, f",
            result=T.i32(),
            loc=loc,
            ip=ip,
            pure=True,
        )
    )


@dsl_user_op
def mx_reciprocal(exponent, amax, *, loc=None, ip=None) -> Float32:
    """2^(127 - exponent), or 0 for an all-zero block: the factor that maps a block onto E4M3."""
    return Float32(
        _asm(
            [Int32(exponent).ir_value(loc=loc, ip=ip), Float32(amax).ir_value(loc=loc, ip=ip)],
            "{\n.reg .b32 r;\n.reg .pred z;\nmov.u32 r, 254;\nsub.u32 r, r, $1;\nshl.b32 r, r, 23;\n"
            "setp.eq.f32 z, $2, 0f00000000;\nselp.b32 r, 0, r, z;\nmov.b32 $0, r;\n}",
            "=f, r, f",
            result=T.f32(),
            loc=loc,
            ip=ip,
            pure=True,
        )
    )


def mx_scale_offset(g, row, kb, scale_m):
    """Byte offset of UE8M0 scale ``kb`` (of 128 per row) of ``row`` in group ``g``, CUTLASS 128x4 layout."""
    return (
        Int64(g) * scale_m * 128
        + (row // 128) * 16384
        + (row % 32) * 16
        + ((row % 128) // 32) * 4
        + (kb // 4) * 512
        + kb % 4
    )


@dsl_user_op
def griddep_wait(*, loc=None, ip=None):
    """PDL: wait until the preceding grid completed and its memory is visible (no-op without a programmatic edge)."""
    _asm([], "griddepcontrol.wait;", "", loc=loc, ip=ip)


@dsl_user_op
def griddep_launch_dependents(*, loc=None, ip=None):
    """PDL: this thread is done; the dependent grid may launch once every CTA released or exited."""
    _asm([], "griddepcontrol.launch_dependents;", "", loc=loc, ip=ip)


@dsl_user_op
def warp_sync(*, loc=None, ip=None):
    """bar.warp.sync over the full warp (orders the lanes' shared-memory accesses)."""
    _asm([], "bar.warp.sync 0xffffffff;", "", loc=loc, ip=ip)


@dsl_user_op
def tma_store_tile3(atom, src, c0, c1, c2, *, loc=None, ip=None):
    """One 3D TMA tile (the atom's box) from this CTA's shared memory to global (bulk group)."""
    _asm(
        [
            _tma_desc(atom, loc, ip),
            Int32(src).ir_value(loc=loc, ip=ip),
            Int32(c0).ir_value(loc=loc, ip=ip),
            Int32(c1).ir_value(loc=loc, ip=ip),
            Int32(c2).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [$0, {$2, $3, $4}], [$1];",
        "l, r, r, r, r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def keep_branch(*, loc=None, ip=None):
    """An empty side-effecting asm: the branch around it stays a branch (no if-conversion into selects that would
    run on every pass)."""
    _asm([], "", "", loc=loc, ip=ip)


@dsl_user_op
def tmem_wait_st(*, loc=None, ip=None):
    _asm([], "tcgen05.wait::st.sync.aligned;", "", loc=loc, ip=ip)


@dsl_user_op
def mbar_wait(mbar, phase, *, loc=None, ip=None):
    """Wait for phase ``phase`` (parity) of the shared-memory mbarrier at ``mbar``: mbarrier.try_wait with a
    suspend-time hint in a branch loop, without the backoff sleep of the generic lowering (the thread resumes when
    the phase completes)."""
    _asm(
        [Int32(mbar).ir_value(loc=loc, ip=ip), Int32(phase).ir_value(loc=loc, ip=ip)],
        "{\n.reg .pred p;\nWAIT:\nmbarrier.try_wait.parity.shared::cta.b64 p, [$0], $1, 10000000;\n@!p bra WAIT;\n}",
        "r, r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ld_shared_v2_f32x2(addr, *, loc=None, ip=None) -> Int64:
    """Two consecutive FP32 values of shared memory as one f32x2 operand (the lower address in the low half)."""
    return Int64(
        _asm(
            [Int32(addr).ir_value(loc=loc, ip=ip)],
            "{\n.reg .f32 a, b;\nld.shared.v2.f32 {a, b}, [$1];\nmov.b64 $0, {a, b};\n}",
            "=l, r",
            result=T.i64(),
            loc=loc,
            ip=ip,
        )
    )


class SparseMlaAttentionGB300:
    """``out[t, h] = softmax_sink(Q[t, h] K_t^T * scale) V_t`` over each token's selected bank rows (K = V).

    Inputs: Q ``[T, 64, 512]`` E4M3, the bank ``[R, 512]`` E4M3, ``indices`` ``[T, width]`` int32 (bank rows,
    the first ``lengths[t]`` valid, 1 <= lengths[t] <= 640), ``lengths`` ``[T]``, ``sinks`` ``[64]`` FP32 logits,
    ``scale`` FP32 = q_dequant * kv_dequant * softmax_scale * log2(e) and ``vscale`` FP32 = kv_dequant. Output:
    ``[T, 64, 512]`` BF16, or with ``epi="rope_mxfp8"`` the grouped MXFP8 O-A input.
    """

    def __init__(self, arch: str = "sm_103", epi: str = "stm", pdl: bool = False):
        # Both epilogues stage each 128-dim block of BF16 O as two 64 x 64 boxes (16x256b TMEM loads, transposing
        # stmatrix); "stm" stores them with TMA, "rope_mxfp8" quantizes them as ``inverse_rope_mxfp8`` does.
        if epi not in ("stm", "rope_mxfp8"):
            raise ValueError("epi must be 'stm' or 'rope_mxfp8'")
        self.arch = arch
        self.rope = epi == "rope_mxfp8"
        self.pdl = bool(pdl)
        if cute.arch.get_max_tmem_alloc_cols(arch) < 512:
            raise ValueError("needs 512 TMEM columns")

    @cute.jit
    def __call__(
        self,
        q_ptr: cute.Pointer,
        bank_ptr: cute.Pointer,
        idx_ptr: cute.Pointer,
        len_ptr: cute.Pointer,
        sinks_ptr: cute.Pointer,
        scale_ptr: cute.Pointer,
        vscale_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        pos_ptr: cute.Pointer,
        rtab_ptr: cute.Pointer,
        f8_ptr: cute.Pointer,
        fsc_ptr: cute.Pointer,
        tokens: Int32,
        width: Int32,
        scale_m: Int32,
        grid: Int32,
        stream: cuda.CUstream,
    ):
        q = cute.make_tensor(q_ptr, cute.make_layout((tokens * HEADS, DIM), stride=(DIM, 1)))
        q_box = cute.make_composed_layout(
            cute.make_swizzle(3, 4, 3), 0, cute.make_layout((HEADS, 128), stride=(128, 1))
        )
        q_atom, _ = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), q, q_box, (HEADS, 128)
        )
        out = cute.make_tensor(out_ptr, cute.make_layout((tokens * HEADS, DIM), stride=(DIM, 1)))
        o_box = cute.tile_to_shape(
            tcgen05.make_smem_layout_atom(tcgen05.SmemLayoutAtomKind.K_SW128, BFloat16),
            (HEADS, 64),
            (0, 1),
        )
        o_atom, _ = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), out, o_box, (HEADS, 64)
        )
        f8_atom = o_atom
        if cutlass.const_expr(self.rope):
            # wo_a's input [8 g][T][4096] as (512 bytes of a head, T * 8 rows (token, head of the group), 8 groups):
            # one (token, block) is the box (128, 8, 8), i.e. 64 head rows of 128 bytes.
            f8 = cute.make_tensor(
                f8_ptr, cute.make_layout((512, tokens * 8, 8), stride=(1, 512, tokens * 4096))
            )
            f8_box = cute.make_composed_layout(
                cute.make_swizzle(3, 4, 3), 0, cute.make_layout((128, 8, 8), stride=(1, 128, 1024))
            )
            f8_atom, _ = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(), f8, f8_box, (128, 8, 8)
            )
        K = OperandMajorMode.K
        qk_mma = utils.sm100.make_trivial_tiled_mma(
            Float8E4M3FN, Float8E4M3FN, K, K, Float32, tcgen05.CtaGroup.ONE, (HEADS, KEYS)
        )
        pv_mma = utils.sm100.make_trivial_tiled_mma(
            Float8E4M3FN,
            Float8E4M3FN,
            OperandMajorMode.MN,
            K,
            Float32,
            tcgen05.CtaGroup.ONE,
            (128, HEADS),
        )
        self.kernel(
            q_atom,
            o_atom,
            f8_atom,
            qk_mma,
            pv_mma,
            bank_ptr,
            idx_ptr,
            len_ptr,
            sinks_ptr,
            scale_ptr,
            vscale_ptr,
            pos_ptr,
            rtab_ptr,
            fsc_ptr,
            tokens,
            width,
            scale_m,
        ).launch(grid=[grid, 1, 1], block=[THREADS, 1, 1], stream=stream, use_pdl=self.pdl)

    @cute.kernel
    def kernel(
        self,
        q_atom: cute.CopyAtom,
        o_atom: cute.CopyAtom,
        f8_atom: cute.CopyAtom,
        qk_mma: cute.TiledMma,
        pv_mma: cute.TiledMma,
        bank_ptr: cute.Pointer,
        idx_ptr: cute.Pointer,
        len_ptr: cute.Pointer,
        sinks_ptr: cute.Pointer,
        scale_ptr: cute.Pointer,
        vscale_ptr: cute.Pointer,
        pos_ptr: cute.Pointer,
        rtab_ptr: cute.Pointer,
        fsc_ptr: cute.Pointer,
        tokens: Int32,
        width: Int32,
        scale_m: Int32,
    ):
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()

        smem = SmemAllocator()
        bars = smem.allocate_array(Int64, NUM_BARS, byte_alignment=8)
        tmem_holder = smem.allocate_array(Int32, 1, byte_alignment=4)
        lists = smem.allocate_array(Int32, 2 * LIST_COLS, byte_alignment=128)
        cfac = smem.allocate_array(Float32, STAT_SLOTS * HEADS, byte_alignment=16)
        cflag = smem.allocate_array(Int32, STAT_SLOTS * 4, byte_alignment=16)
        estat = smem.allocate_array(Float32, 2 * 2 * HEADS, byte_alignment=16)
        # Per-token epilogue factors and (rope_mxfp8) cos / sin, double-buffered by token parity: a warp may write
        # token it + 1's while another still reads token it's (only the per-token barrier orders them two tokens
        # apart).
        einv = smem.allocate_array(Float32, 2 * HEADS, byte_alignment=16)
        # rope_mxfp8: the token's 32 cos + 32 sin of the inverse rotation.
        ecs = smem.allocate_array(Float32, 2 * 64, byte_alignment=16)
        k128 = tcgen05.make_smem_layout_atom(tcgen05.SmemLayoutAtomKind.K_SW128, Float8E4M3FN)
        mn128 = tcgen05.make_smem_layout_atom(tcgen05.SmemLayoutAtomKind.MN_SW128, Float8E4M3FN)
        kint = tcgen05.make_smem_layout_atom(tcgen05.SmemLayoutAtomKind.K_INTER, Float8E4M3FN)
        q_layout = cute.tile_to_shape(k128, (HEADS, DIM, 2), (0, 1, 2))
        k_layout = cute.tile_to_shape(k128, (KEYS, DIM, KV_STAGES), (0, 1, 2))
        # V^T block c of stage s is "stage" 4 s + c of a (128 dims, 64 keys) MN-major view of the same bytes.
        vt_layout = cute.tile_to_shape(mn128, (128, KEYS, KV_STAGES * 4), (1, 0, 2))
        p_layout = cute.tile_to_shape(kint, (HEADS, KEYS, P_BUFS), (0, 1, 2))
        sQ = smem.allocate_tensor(Float8E4M3FN, q_layout.outer, 1024, q_layout.inner)
        sK = smem.allocate_tensor(Float8E4M3FN, k_layout.outer, 1024, k_layout.inner)
        sVt = cute.make_tensor(sK.iterator, vt_layout.outer)
        sP = smem.allocate_tensor(Float8E4M3FN, p_layout.outer, 1024, p_layout.inner)
        sO = smem.allocate_array(cutlass.Uint8, OBUF_BYTES, byte_alignment=1024)

        if warp == 0:
            if lane == 0:
                for b in cutlass.range_constexpr(2):
                    cute.arch.mbarrier_init(bars + Q_FULL + b, 1)
                    cute.arch.mbarrier_init(bars + Q_EMPTY + b, 1)
                    cute.arch.mbarrier_init(bars + LIST_FULL + b, 1)
                    cute.arch.mbarrier_init(bars + LIST_EMPTY + b, GATHER_WARPS)
                    cute.arch.mbarrier_init(bars + EPI_FULL + b, 4)
                    cute.arch.mbarrier_init(bars + EPI_EMPTY + b, 4)
                for s in cutlass.range_constexpr(KV_STAGES):
                    cute.arch.mbarrier_init(bars + KV_FULL + s, 128)
                    cute.arch.mbarrier_init(bars + KV_EMPTY + s, 1)
                for b in cutlass.range_constexpr(S_BUFS):
                    cute.arch.mbarrier_init(bars + S_FULL + b, 1)
                    cute.arch.mbarrier_init(bars + S_EMPTY + b, 4)
                for b in cutlass.range_constexpr(P_BUFS):
                    cute.arch.mbarrier_init(bars + P_FULL + b, 4)
                    cute.arch.mbarrier_init(bars + P_EMPTY + b, 1)
                for b in cutlass.range_constexpr(STAT_SLOTS):
                    cute.arch.mbarrier_init(bars + STATS_FULL + b, 4)
                    cute.arch.mbarrier_init(bars + STATS_EMPTY + b, 4)
                cute.arch.mbarrier_init(bars + O_DONE, 1)
                cute.arch.mbarrier_init(bars + O_READY, 1)
                cute.arch.mbarrier_init(bars + O_FULL, 1)
                for b in cutlass.range_constexpr(O_SLOTS):
                    cute.arch.mbarrier_init(bars + O_FREE + b, 4)
                cute.arch.mbarrier_init_fence()
        if warp == QK_WARP:
            cute.arch.alloc_tmem(512, tmem_holder, arch=self.arch)
            cute.arch.relinquish_tmem_alloc_permit()
            tcgen05_before_thread_sync()
        if cutlass.const_expr(self.pdl):
            if (warp == Q_WARP) & (lane == 0):
                cpasync.prefetch_descriptor(q_atom)
                cpasync.prefetch_descriptor(o_atom)
                if cutlass.const_expr(self.rope):
                    cpasync.prefetch_descriptor(f8_atom)
        cute.arch.sync_threads()
        tcgen05_after_thread_sync()
        if cutlass.const_expr(self.pdl):
            # Everything above is this CTA's own state; the predecessor's data is read only below.
            griddep_wait()
        tmem_ptr = cute.arch.retrieve_tmem_ptr(Float32, 16, tmem_holder)
        tmem_base = cute.make_tensor(tmem_holder, cute.make_layout(1))[0]

        thr_qk = qk_mma.get_slice(0)
        thr_pv = pv_mma.get_slice(0)
        tCrQ = qk_mma.make_fragment_A(thr_qk.partition_A(sQ))
        tCrK = qk_mma.make_fragment_B(thr_qk.partition_B(sK))
        tCrVt = pv_mma.make_fragment_A(thr_pv.partition_A(sVt))
        tCrPt = pv_mma.make_fragment_B(thr_pv.partition_B(sP))
        tS_layout = qk_mma.make_fragment_C(qk_mma.partition_shape_C((HEADS, KEYS))).layout
        tO_layout = pv_mma.make_fragment_C(pv_mma.partition_shape_C((128, HEADS))).layout

        lens = cute.make_tensor(len_ptr, cute.make_layout(tokens))
        if warp < GATHER_WARPS:
            self.gather(sK, lists, bars, bank_ptr, lens, tokens)
        if warp == QK_WARP:
            self.mma_qk(qk_mma, tCrQ, tCrK, tmem_ptr, tS_layout, bars, lens, tokens)
        if warp == PV_WARP:
            self.mma_pv(pv_mma, tCrVt, tCrPt, tmem_ptr, tO_layout, cflag, bars, lens, tokens)
        if warp == Q_WARP:
            self.load_q(q_atom, sQ, lists, bars, idx_ptr, lens, tokens, width)
        if (warp >= SM_WARP0) & (warp < CO_WARP0):
            self.softmax(tmem_base, sP, cfac, cflag, estat, bars, scale_ptr, lens, tokens)
        if warp >= CO_WARP0:
            self.correction(
                tmem_base,
                cfac,
                cflag,
                estat,
                einv,
                ecs,
                sO,
                o_atom,
                f8_atom,
                pos_ptr,
                rtab_ptr,
                fsc_ptr,
                scale_m,
                bars,
                sinks_ptr,
                vscale_ptr,
                lens,
                tokens,
            )

        cute.arch.sync_threads()
        if warp == QK_WARP:
            cute.arch.dealloc_tmem(tmem_ptr, 512, arch=self.arch)
        if cutlass.const_expr(self.pdl):
            # Every store of this CTA was issued before the barrier above.
            griddep_launch_dependents()

    @cute.jit
    def num_tiles(self, n):
        """64-key tiles of a token with ``n`` keys; a token without keys still runs one fully masked tile (its
        output is zero), so every role sees at least one tile per token."""
        nt = (n + KEYS - 1) // KEYS
        if nt < 1:
            nt = Int32(1)
        return nt

    @cute.jit
    def tile_of(self, j, ntiles):
        """List tile visited at position ``j`` of a token with ``ntiles`` tiles: 1, 2, 0, 3, 4, ... (two tiles: 1, 0).
        The list is the 128-row window (oldest first) then the selected rows by bank row, so tiles 1 (the newest
        window rows) and 2 (the first selected rows) hold a token's maximum for most heads."""
        jj = j
        if ntiles >= 3:
            if j < 3:
                jj = (j + 1) % 3
        if ntiles == 2:
            jj = 1 - j
        return jj

    @cute.jit
    def wait(self, bar, phase):
        """Wait for parity ``phase`` of the mbarrier ``bar``."""
        mbar_wait(bar.toint(), phase)

    @cute.jit
    def warp_arrive(self, bar):
        """One arrival per warp (lane 0, after a warp barrier that orders the warp's earlier writes)."""
        cute.arch.sync_warp()
        if cute.arch.lane_idx() == 0:
            cute.arch.mbarrier_arrive(bar)

    # ------------------------------------------------------------------------------------------------------
    # Producers: index lists + Q (warp 6), KV rows (warps 0-3)
    # ------------------------------------------------------------------------------------------------------

    @cute.jit
    def load_q(self, q_atom, sQ, lists, bars, idx_ptr, lens, tokens, width):
        """Per token: its index list into list buffer it % 2 (bulk copy) once the gather warps released it, then Q
        (64 heads x 512, four TMA tiles) into Q buffer it % 2 once the QK warp is done with token it - 2."""
        lane = cute.arch.lane_idx()
        bid, _, _ = cute.arch.block_idx()
        nblk, _, _ = cute.arch.grid_dim()
        if lane == 0:
            cpasync.prefetch_descriptor(q_atom)
            q_base = sQ.iterator.toint()
            it = Int32(0)
            for t in cutlass.range(bid, tokens, nblk, unroll=1):
                b = it % 2
                cols = self.num_tiles(lens[t]) * KEYS
                if it >= 2:
                    self.wait(bars + LIST_EMPTY + b, ((it - 2) // 2) & 1)
                cute.arch.mbarrier_arrive_and_expect_tx(bars + LIST_FULL + b, cols * 4)
                bulk_load(
                    lists.toint() + b * (LIST_COLS * 4),
                    idx_ptr.toint() + Int64(t) * width * 4,
                    cols * 4,
                    (bars + LIST_FULL + b).toint(),
                )
                if it >= 2:
                    self.wait(bars + Q_EMPTY + b, ((it - 2) // 2) & 1)
                cute.arch.mbarrier_arrive_and_expect_tx(bars + Q_FULL + b, Q_BYTES)
                for c in cutlass.range_constexpr(DIM // 128):
                    tma_load_tile(
                        q_atom,
                        q_base + b * Q_BYTES + c * (HEADS * 128),
                        (bars + Q_FULL + b).toint(),
                        c * 128,
                        t * HEADS,
                    )
                it = it + 1

    @cute.jit
    def gather(self, sK, lists, bars, bank_ptr, lens, tokens):
        """KV rows of every tile into stage g % KV_STAGES: warp w fetches keys [16 w, 16 w + 16) of the tile; keys
        past the token's length are zero-filled."""
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        bid, _, _ = cute.arch.block_idx()
        nblk, _, _ = cute.arch.grid_dim()
        rpw = KEYS // GATHER_WARPS
        lk = cute.make_tensor(lists, cute.make_layout(2 * LIST_COLS))
        bank = bank_ptr.toint()
        k_base = sK.iterator.toint()
        chunk = lane // 8
        unit = lane % 8
        g = Int32(0)
        it = Int32(0)
        for t in cutlass.range(bid, tokens, nblk, unroll=1):
            lb = it % 2
            n = lens[t]
            ntiles = self.num_tiles(n)
            self.wait(bars + LIST_FULL + lb, (it // 2) & 1)
            for j in cutlass.range(ntiles, unroll=1):
                s = g % KV_STAGES
                if g >= KV_STAGES:
                    self.wait(bars + KV_EMPTY + s, (g // KV_STAGES - 1) & 1)
                k0 = self.tile_of(j, ntiles) * KEYS + warp * rpw
                mine = Int32(-1)
                if lane < rpw:
                    if k0 + lane < n:
                        mine = lk[lb * LIST_COLS + k0 + lane]
                stage = k_base + s * STAGE_BYTES
                for i in cutlass.range_constexpr(rpw):
                    key = warp * rpw + i
                    row = cute.arch.shuffle_sync(mine, i)
                    dst = stage + chunk * (KEYS * 128) + key * 128 + ((unit ^ (key % 8)) * 16)
                    if row >= 0:
                        cp_async_16(dst, bank + Int64(row) * DIM + lane * 16)
                    else:
                        cp_async_16_zero(dst, bank)
                cp_async_arrive_noinc((bars + KV_FULL + s).toint())
                g = g + 1
            # This warp has read the token's list: release the buffer for token it + 2.
            self.warp_arrive(bars + LIST_EMPTY + lb)
            it = it + 1

    # ------------------------------------------------------------------------------------------------------
    # MMA warps
    # ------------------------------------------------------------------------------------------------------

    @cute.jit
    def mma_qk(self, qk_mma, tCrQ, tCrK, tmem_ptr, tS_layout, bars, lens, tokens):
        """QK of tile g into S buffer g % S_BUFS."""
        bid, _, _ = cute.arch.block_idx()
        nblk, _, _ = cute.arch.grid_dim()
        g = Int32(0)
        it = Int32(0)
        for t in cutlass.range(bid, tokens, nblk, unroll=1):
            qb = it % 2
            ntiles = self.num_tiles(lens[t])
            self.wait(bars + Q_FULL + qb, (it // 2) & 1)
            for j in cutlass.range(ntiles, unroll=1):
                s = g % KV_STAGES
                b = g % S_BUFS
                self.wait(bars + KV_FULL + s, (g // KV_STAGES) & 1)
                if g >= S_BUFS:
                    self.wait(bars + S_EMPTY + b, (g // S_BUFS - 1) & 1)
                tcgen05_after_thread_sync()
                tS = cute.make_tensor(tmem_ptr + b * (16 * 65536), tS_layout)
                qk_mma = self.qk(qk_mma, tCrQ, tCrK, tS, qb, s)
                with cute.arch.elect_one():
                    tcgen05.commit(bars + S_FULL + b)
                    if j == ntiles - 1:
                        tcgen05.commit(bars + Q_EMPTY + qb)
                g = g + 1
            it = it + 1

    @cute.jit
    def qk(self, qk_mma, tCrQ, tCrK, tS, qb, s):
        qk_mma.set(tcgen05.Field.ACCUMULATE, False)
        for kb in cutlass.range_constexpr(DIM // 32):
            cute.gemm(qk_mma, tS, tCrQ[None, None, kb, qb], tCrK[None, None, kb, s], tS)
            qk_mma.set(tcgen05.Field.ACCUMULATE, True)
        return qk_mma

    @cute.jit
    def mma_pv(self, pv_mma, tCrVt, tCrPt, tmem_ptr, tO_layout, cflag, bars, lens, tokens):
        """PV of tile g into the token's four O^T slots once its P is in shared memory; a token's first tile waits
        for its slots to be drained, the others for the correction warps when a row factor of the tile is not 1."""
        bid, _, _ = cute.arch.block_idx()
        nblk, _, _ = cute.arch.grid_dim()
        flags = cute.make_tensor(cflag, cute.make_layout(STAT_SLOTS * 4))
        g = Int32(0)
        it = Int32(0)
        nf = Int32(0)
        nr = Int32(0)
        for t in cutlass.range(bid, tokens, nblk, unroll=1):
            ntiles = self.num_tiles(lens[t])
            for j in cutlass.range(ntiles, unroll=1):
                pv_mma, nf, nr = self.pv_tile(
                    pv_mma, tCrVt, tCrPt, tmem_ptr, tO_layout, flags, bars, g, j, ntiles, it, nf, nr
                )
                g = g + 1
            it = it + 1

    @cute.jit
    def pv_tile(
        self, pv_mma, tCrVt, tCrPt, tmem_ptr, tO_layout, flags, bars, g, j, ntiles, it, nf, nr
    ):
        b = g % P_BUFS
        s = g % KV_STAGES
        self.wait(bars + P_FULL + b, (g // P_BUFS) & 1)
        if j > 0:
            fs = nf % STAT_SLOTS
            need = flags[fs * 4] | flags[fs * 4 + 1] | flags[fs * 4 + 2] | flags[fs * 4 + 3]
            if need != 0:
                self.wait(bars + O_READY, nr & 1)
                nr = nr + 1
            nf = nf + 1
        for c in cutlass.range_constexpr(4):
            u = it * 4 + c
            slot = u % O_SLOTS
            if j == 0:
                if u >= O_SLOTS:
                    self.wait(bars + O_FREE + slot, (u // O_SLOTS - 1) & 1)
            tcgen05_after_thread_sync()
            tO = cute.make_tensor(tmem_ptr + O_COL0 + 64 * slot, tO_layout)
            pv_mma.set(tcgen05.Field.ACCUMULATE, j > 0)
            for kk in cutlass.range_constexpr(KEYS // 32):
                cute.gemm(
                    pv_mma, tO, tCrVt[None, None, kk, s * 4 + c], tCrPt[None, None, kk, b], tO
                )
                pv_mma.set(tcgen05.Field.ACCUMULATE, True)
        with cute.arch.elect_one():
            tcgen05.commit(bars + KV_EMPTY + s)
            tcgen05.commit(bars + P_EMPTY + b)
            tcgen05.commit(bars + O_DONE)
            if j == ntiles - 1:
                tcgen05.commit(bars + O_FULL)
        return pv_mma, nf, nr

    # ------------------------------------------------------------------------------------------------------
    # Softmax (warps 8-11)
    # ------------------------------------------------------------------------------------------------------

    @cute.jit
    def softmax(self, tmem_base, sP, cfac, cflag, estat, bars, scale_ptr, lens, tokens):
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        bid, _, _ = cute.arch.block_idx()
        nblk, _, _ = cute.arch.grid_dim()
        neg_inf = Float32(-math.inf)
        q = warp - SM_WARP0
        kh = lane // 16
        row = q * 16 + lane % 16
        scale_log2 = cute.make_tensor(scale_ptr, cute.make_layout(1))[0]
        scale2 = pack_f32x2(scale_log2, scale_log2)
        fac = cute.make_tensor(cfac, cute.make_layout(STAT_SLOTS * HEADS))
        flags = cute.make_tensor(cflag, cute.make_layout(STAT_SLOTS * 4))
        es = cute.make_tensor(estat, cute.make_layout(4 * HEADS))
        # P[head][key], K-major without swizzle: 16 keys (16 bytes) of row h at (k / 16) * 1024 + (h / 8) * 128 +
        # (h % 8) * 16; this thread's 32 keys are units 2 kh and 2 kh + 1.
        p_row = sP.iterator.toint() + (row // 8) * 128 + (row % 8) * 16 + (2 * kh) * (HEADS * 16)
        rS = cute.make_rmem_tensor((32,), Float32)
        rT = cute.make_rmem_tensor((32,), Float32)
        g = Int32(0)
        it = Int32(0)
        nf = Int32(0)
        for t in cutlass.range(bid, tokens, nblk, unroll=1):
            n = lens[t]
            ntiles = self.num_tiles(n)
            m_ref = neg_inf
            l_sum = Float32(0.0)
            for j in cutlass.range(ntiles, unroll=1):
                b = g % S_BUFS
                pb = g % P_BUFS
                self.wait(bars + S_FULL + b, (g // S_BUFS) & 1)
                tcgen05_after_thread_sync()
                vals, raw_max = tmem_ld_red_16x32bx2_x32(tmem_base + ((q * 32 + b * 16) << 16))
                for k in cutlass.range_constexpr(32):
                    rS[k] = vals[k]
                tcgen05_before_thread_sync()
                self.warp_arrive(bars + S_EMPTY + b)
                kbase = self.tile_of(j, ntiles) * KEYS + kh * 32
                if kbase + 32 > n:
                    # Partial tile (one per token at most): mask the keys past the length. A real branch, so full
                    # tiles do not pay the 32 compares and selects.
                    keep_branch()
                    for k in cutlass.range_constexpr(32):
                        v = rS[k]
                        if kbase + k >= n:
                            v = neg_inf
                        rS[k] = v
                    raw_max = tree_max([rS[k] for k in range(32)])
                if (j == 0) & (ntiles > 1):
                    # The token's second tile joins the starting maximum (its S buffer stays held until
                    # its own turn).
                    b1 = (g + 1) % S_BUFS
                    self.wait(bars + S_FULL + b1, ((g + 1) // S_BUFS) & 1)
                    tcgen05_after_thread_sync()
                    v1, red1 = tmem_ld_red_16x32bx2_x32(tmem_base + ((q * 32 + b1 * 16) << 16))
                    kb1 = self.tile_of(1, ntiles) * KEYS + kh * 32
                    if kb1 + 32 > n:
                        for k in cutlass.range_constexpr(32):
                            rT[k] = v1[k]
                        for k in cutlass.range_constexpr(32):
                            v = rT[k]
                            if kb1 + k >= n:
                                v = neg_inf
                            rT[k] = v
                        red1 = tree_max([rT[k] for k in range(32)])
                    raw_max = cute.arch.fmax(raw_max, red1)
                raw_max = cute.arch.fmax(raw_max, cute.arch.shuffle_sync_bfly(raw_max, offset=16))
                tile_max = raw_max * scale_log2
                m_new = m_ref
                factor = Float32(1.0)
                if tile_max > m_ref + RESCALE_LOG2:
                    m_new = tile_max
                    if m_ref != neg_inf:
                        factor = cute.arch.exp2(m_ref - m_new)
                l_sum = l_sum * factor
                m_ref = m_new
                # P buffer b (and every factor slot an earlier tile used) is free once PV(g - 2) completed.
                if g >= P_BUFS:
                    self.wait(bars + P_EMPTY + pb, (g // P_BUFS - 1) & 1)
                if j > 0:
                    fs = nf % STAT_SLOTS
                    if nf >= STAT_SLOTS:
                        self.wait(bars + STATS_EMPTY + fs, (nf // STAT_SLOTS - 1) & 1)
                    if kh == 0:
                        fac[fs * HEADS + row] = factor
                    moved = cute.arch.vote_any_sync(factor != Float32(1.0))
                    mv = Int32(0)
                    if moved:
                        mv = Int32(1)
                    if lane == 0:
                        flags[fs * 4 + q] = mv
                    self.warp_arrive(bars + STATS_FULL + fs)
                    nf = nf + 1
                m_use = m_ref - P_OFFSET_LOG2
                if m_ref == neg_inf:
                    m_use = Float32(0.0)
                neg_m = Float32(0.0) - m_use
                neg2 = pack_f32x2(neg_m, neg_m)
                row_sum = Float32(0.0)
                for u in cutlass.range_constexpr(2):
                    hp = [
                        exp2_f16x2_fma2(rS[16 * u + 2 * i], rS[16 * u + 2 * i + 1], scale2, neg2)
                        for i in range(8)
                    ]
                    w = [pack_e4m3x4_f16x2(hp[2 * i], hp[2 * i + 1]) for i in range(4)]
                    st_shared_v4(p_row + pb * P_BYTES + u * (HEADS * 16), w[0], w[1], w[2], w[3])
                    row_sum = row_sum + e4m3_sum16(w)
                l_sum = l_sum + row_sum
                cute.arch.fence_view_async_shared()
                tcgen05_before_thread_sync()
                self.warp_arrive(bars + P_FULL + pb)
                g = g + 1
            # The token's final maximum and row sum (both halves of the row) for the epilogue.
            l_tot = l_sum + cute.arch.shuffle_sync_bfly(l_sum, offset=16)
            eb = it % 2
            if it >= 2:
                self.wait(bars + EPI_EMPTY + eb, ((it - 2) // 2) & 1)
            if kh == 0:
                es[eb * 2 * HEADS + row] = m_ref
                es[eb * 2 * HEADS + HEADS + row] = l_tot
            self.warp_arrive(bars + EPI_FULL + eb)
            it = it + 1

    # ------------------------------------------------------------------------------------------------------
    # O^T correction and epilogue (warps 12-15)
    # ------------------------------------------------------------------------------------------------------

    @cute.jit
    def correction(
        self,
        tmem_base,
        cfac,
        cflag,
        estat,
        einv,
        ecs,
        sO,
        o_atom,
        f8_atom,
        pos_ptr,
        rtab_ptr,
        fsc_ptr,
        scale_m,
        bars,
        sinks_ptr,
        vscale_ptr,
        lens,
        tokens,
    ):
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        bid, _, _ = cute.arch.block_idx()
        nblk, _, _ = cute.arch.grid_dim()
        neg_inf = Float32(-math.inf)
        q = warp - CO_WARP0
        tid = q * 32 + lane
        fac = cute.make_tensor(cfac, cute.make_layout(STAT_SLOTS * HEADS))
        flags = cute.make_tensor(cflag, cute.make_layout(STAT_SLOTS * 4))
        es = cute.make_tensor(estat, cute.make_layout(4 * HEADS))
        inv_t = cute.make_tensor(einv, cute.make_layout(2 * HEADS))
        sinks = cute.make_tensor(sinks_ptr, cute.make_layout(HEADS))
        vscale = cute.make_tensor(vscale_ptr, cute.make_layout(1))[0]
        lane_addr = (q * 32) << 16
        # Output staging: box bx = dims [64 bx, 64 bx + 64) of the block, written by warps 2 bx and 2 bx + 1 (32 dims
        # each), rows = heads with the 128-byte swizzle.
        bx = q // 2
        inv_base = einv.toint()
        o_base = sO.toint()
        is_issuer = (q % 2 == 0) & (lane == 0)
        # This thread addresses row r = lane % 8 of matrix mi = lane / 8 of every stmatrix.x4: for lane half u
        # (dims 16 u .. of the warp's 32) and head-group pair kp, matrix mi is heads 8 (2 kp + mi / 2) + r, dims
        # 8 (mi % 2) .. +8 of the half -- the 16-byte unit (q % 2) 4 + 2 u + mi % 2 of the 128-byte head row.
        st_r = lane % 8
        st_m = lane // 8
        # rope_mxfp8: box 1 rows use the 128-byte swizzle with row ^ 4, so that the two threads of a head (one per
        # box, adjacent lanes) read different banks in ``rope_mxfp8_block``.
        sw1 = 4 if self.rope else 0
        st_off = [
            bx * (OBUF_BYTES // 2)
            + (2 * kp + st_m // 2) * 1024
            + st_r * 128
            + ((((q % 2) * 4 + 2 * u + st_m % 2) ^ st_r ^ (bx * sw1)) * 16)
            for u in range(2)
            for kp in range(4)
        ]
        rope = self.rope
        if (q == 0) & (lane == 0):
            cpasync.prefetch_descriptor(o_atom)
            if cutlass.const_expr(rope):
                cpasync.prefetch_descriptor(f8_atom)
        if cutlass.const_expr(rope):
            # The UE8M0 scales of the padding rows [tokens, scale_m) are zero (the GEMM reads them). They lie in the
            # last 128-row block, whose 512-byte chunks (8 groups x 32 four-scale columns) the CTAs share: thread tid
            # zeroes word tid, i.e. row 32 (tid % 4) + tid / 4 of the block, so a warp's stores are contiguous.
            row = scale_m - 128 + (tid % 4) * 32 + tid // 4
            if row >= tokens:
                for chunk in cutlass.range(bid, 256, nblk, unroll=1):
                    st_global_b32(
                        fsc_ptr.toint()
                        + mx_scale_offset(chunk // 32, row, (chunk % 32) * 4, scale_m),
                        Int32(0),
                    )
        g = Int32(0)
        it = Int32(0)
        nf = Int32(0)
        posv = cute.make_tensor(pos_ptr, cute.make_layout(tokens))
        for t in cutlass.range(bid, tokens, nblk, unroll=1):
            n = lens[t]
            ntiles = self.num_tiles(n)
            g = g + 1
            for _j in cutlass.range(1, ntiles, unroll=1):
                fs = nf % STAT_SLOTS
                self.wait(bars + STATS_FULL + fs, (nf // STAT_SLOTS) & 1)
                need = flags[fs * 4] | flags[fs * 4 + 1] | flags[fs * 4 + 2] | flags[fs * 4 + 3]
                if need != 0:
                    # O^T holds PV(.. g - 1): rescale the heads whose reference moved, then release PV(g).
                    self.wait(bars + O_DONE, (g - 1) & 1)
                    tcgen05_after_thread_sync()
                    cb = [
                        tmem_base + O_COL0 + 64 * ((it * 4 + c) % O_SLOTS) + lane_addr
                        for c in range(4)
                    ]
                    for ch in cutlass.range_constexpr(HEADS // 8):
                        f = [fac[fs * HEADS + ch * 8 + i] for i in range(8)]
                        moved = (
                            (f[0] != Float32(1.0))
                            | (f[1] != Float32(1.0))
                            | (f[2] != Float32(1.0))
                            | (f[3] != Float32(1.0))
                            | (f[4] != Float32(1.0))
                            | (f[5] != Float32(1.0))
                            | (f[6] != Float32(1.0))
                            | (f[7] != Float32(1.0))
                        )
                        if moved:
                            v = tmem_ld_32x32b_x8_x4(
                                cb[0] + ch * 8, cb[1] + ch * 8, cb[2] + ch * 8, cb[3] + ch * 8
                            )
                            tmem_st_32x32b_x8_x4(
                                cb[0] + ch * 8,
                                cb[1] + ch * 8,
                                cb[2] + ch * 8,
                                cb[3] + ch * 8,
                                [v[8 * c + i] * f[i] for c in range(4) for i in range(8)],
                            )
                    tmem_wait_st()
                    tcgen05_before_thread_sync()
                    cute.arch.barrier(barrier_id=CO_BAR, number_of_threads=128)
                    if tid == 0:
                        cute.arch.mbarrier_arrive(bars + O_READY)
                # The factor slot has been read.
                self.warp_arrive(bars + STATS_EMPTY + fs)
                g = g + 1
                nf = nf + 1

            # Epilogue of the token: per-head 1 / (sum + sink term) x kv_dequant, then the four O^T blocks.
            eb = it % 2
            self.wait(bars + EPI_FULL + eb, (it // 2) & 1)
            if tid < HEADS:
                m_ref = es[eb * 2 * HEADS + tid]
                l_sum = es[eb * 2 * HEADS + HEADS + tid]
                m_use = m_ref - P_OFFSET_LOG2
                if m_ref == neg_inf:
                    m_use = Float32(0.0)
                inv_t[eb * HEADS + tid] = vscale / (
                    l_sum + cute.arch.exp2(sinks[tid] * LOG2E - m_use)
                )
                if cutlass.const_expr(rope):
                    cs_t = cute.make_tensor(ecs, cute.make_layout(2 * 64))
                    pos = posv[t]
                    cs_t[eb * 64 + tid] = ld_global_f32(
                        rtab_ptr.toint() + (Int64(pos) * 64 + tid) * 4
                    )
            cute.arch.barrier(barrier_id=CO_BAR, number_of_threads=128)
            # This thread's head pairs 8 k + 2 (lane % 4), +1 of every column group k.
            inv2 = [
                ld_shared_v2_f32x2(inv_base + (eb * HEADS + 8 * k + 2 * (lane % 4)) * 4)
                for k in range(8)
            ]
            self.warp_arrive(bars + EPI_EMPTY + eb)
            self.wait(bars + O_FULL, it & 1)
            tcgen05_after_thread_sync()
            for c in cutlass.range_constexpr(4):
                slot = (it * 4 + c) % O_SLOTS
                col = O_COL0 + 64 * slot
                v = tmem_ld_16x256b_x8_pair(
                    tmem_base + col + lane_addr, tmem_base + col + lane_addr + (16 << 16)
                )
                tcgen05_before_thread_sync()
                # O^T slot drained: the next token that maps a block to it may accumulate.
                self.warp_arrive(bars + O_FREE + slot)
                # rope_mxfp8 reads whole 128-dim head rows of the block (both boxes) back: its buffer waits and
                # hand-offs span the four warps.
                bar_id = CO_BAR if rope else BOX_BAR0 + bx
                bar_n = 128 if rope else 64
                if cutlass.const_expr(c > 0):
                    # The stores issued from the staging buffer for the previous block have read it.
                    if is_issuer:
                        cute.arch.cp_async_bulk_wait_group(0, read=True)
                    cute.arch.barrier(barrier_id=bar_id, number_of_threads=bar_n)
                else:
                    if it > 0:
                        if is_issuer:
                            cute.arch.cp_async_bulk_wait_group(0, read=True)
                        cute.arch.barrier(barrier_id=bar_id, number_of_threads=bar_n)
                box = o_base + bx * (OBUF_BYTES // 2)
                for u in cutlass.range_constexpr(2):
                    for kp in cutlass.range_constexpr(4):
                        w = []
                        for kk in (2 * kp, 2 * kp + 1):
                            r0 = 32 * u + 4 * kk
                            w.append(scale_bf16x2(v[r0], v[r0 + 1], inv2[kk]))
                            w.append(scale_bf16x2(v[r0 + 2], v[r0 + 3], inv2[kk]))
                        stmatrix_x4_trans(o_base + st_off[4 * u + kp], w[0], w[1], w[2], w[3])
                if cutlass.const_expr(not rope):
                    cute.arch.fence_view_async_shared()
                cute.arch.barrier(barrier_id=bar_id, number_of_threads=bar_n)
                if cutlass.const_expr(not rope):
                    if is_issuer:
                        tma_store_tile(o_atom, box, c * 128 + bx * 64, t * HEADS)
                        cute.arch.cp_async_bulk_commit_group()
                if cutlass.const_expr(rope):
                    self.rope_mxfp8_block(o_base, ecs, eb, fsc_ptr, scale_m, t, c, tid)
                    cute.arch.fence_view_async_shared()
                    cute.arch.barrier(barrier_id=CO_BAR, number_of_threads=128)
                    if tid == 0:
                        tma_store_tile3(f8_atom, o_base, c * 128, t * 8, 0)
                        cute.arch.cp_async_bulk_commit_group()
            it = it + 1
        if is_issuer:
            # The TMA stores' global writes (not only their shared-memory reads) complete before the CTA exits:
            # the kernel's consumer may start as soon as every CTA has exited.
            cute.arch.cp_async_bulk_wait_group(0, read=False)

    @cute.jit
    def rope_mxfp8_block(self, blk, ecs, eb, fsc_ptr, scale_m, t, c: cutlass.Constexpr, tid):
        """Block ``c`` of token ``t`` from the BF16 O rows staged at ``blk`` (two 64-dim boxes, box 1 with row ^ 4):
        thread ``tid`` takes head h = tid / 2, dims 64 (tid % 2) .. +64 of the block; ``eb`` selects the token's
        cos / sin buffer. The rotated channels (block 3: 448..511) round to BF16, then each 32-dim block gets a UE8M0
        scale (into the CUTLASS-layout scales at ``fsc_ptr``) and its E4M3 values, which replace box 0 as 64 head
        rows of 128 bytes."""
        h = tid // 2
        half = tid % 2
        hr = h % 8
        rowb = blk + half * (OBUF_BYTES // 2) + (h // 8) * 1024 + hr * 128
        hs = hr ^ (half * 4)
        x = []
        for k in cutlass.range_constexpr(8):
            x += ld_shared_bf16x8(rowb + ((hs ^ k) * 16))
        if cutlass.const_expr(c == 3):
            cs = cute.make_tensor(ecs, cute.make_layout(2 * 64))
            for i in cutlass.range_constexpr(32):
                r0, r1 = rope_inv_pair(
                    x[2 * i], x[2 * i + 1], cs[eb * 64 + i], cs[eb * 64 + 32 + i]
                )
                x[2 * i] = select_f32(half, bf16_round(r0), x[2 * i])
                x[2 * i + 1] = select_f32(half, bf16_round(r1), x[2 * i + 1])
        exponents = []
        words = []
        for b in cutlass.range_constexpr(2):
            amax = absmax_tree(x[32 * b : 32 * b + 32])
            e = mx_exponent(amax)
            inv = mx_reciprocal(e, amax)
            exponents.append(e)
            words += [
                pack_e4m3x4(
                    f32_op("mul.rn.f32 $0, $1, $2;", x[32 * b + 4 * i], inv),
                    f32_op("mul.rn.f32 $0, $1, $2;", x[32 * b + 4 * i + 1], inv),
                    f32_op("mul.rn.f32 $0, $1, $2;", x[32 * b + 4 * i + 2], inv),
                    f32_op("mul.rn.f32 $0, $1, $2;", x[32 * b + 4 * i + 3], inv),
                )
                for i in range(8)
            ]
        row8 = blk + (h // 8) * 1024 + hr * 128
        for u in cutlass.range_constexpr(4):
            if cutlass.const_expr(u == 0):
                # The upper-half thread of head h writes into the bytes its lower-half neighbor read.
                warp_sync()
            st_shared_v4(
                row8 + (((half * 4 + u) ^ hr) * 16),
                words[4 * u],
                words[4 * u + 1],
                words[4 * u + 2],
                words[4 * u + 3],
            )
        # This thread's two blocks are 16 (h % 8) + 4 c + 2 half and the next of group h / 8's 128 per row.
        st_global_b16(
            fsc_ptr.toint() + mx_scale_offset(h // 8, t, (h % 8) * 16 + c * 4 + half * 2, scale_m),
            exponents[0] | (exponents[1] << 8),
        )
