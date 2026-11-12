# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``swa_block_multi_attention`` of a 2..4-request decode step in one launch: a CTA cluster per (request, KV head).

The multi-request sliding-window block (``swa_block``, the 64-key x 64-row tile) launches two dependent grids per
layer: a split grid (one program per 64-key block of the window, 34 per request and KV head) that writes FP32
partials to global memory, and a combine grid that reads them back, merges them and applies the output norm-gate.
Here the 34 splits of one (request, KV head) are the 8 CTAs of one cluster: CTA ``r`` owns key blocks
``r, r + 8, r + 16, r + 24 (, r + 32)``, runs each block as the split program does, and the merge goes through
distributed shared memory instead of global memory and a second grid:

1. Before ``griddepcontrol.wait``: the sequence length, the block table, the cached K/V of the CTA's blocks (FP8,
   widened to BF16 in shared memory), the positions and RoPE rows, the KV scale. Tensor memory is allocated only
   after the wait and before the dependents are triggered (no allocation can wait on a dependent).
2. After it: every CTA runs the fused prep of the 64 query rows of its KV head (8 tokens x 8 heads) and of the
   step's 8 keys; cluster rank 0 writes the step's K/V into the paged cache (the split's writer program); the CTA
   that owns a block holding a step key patches its tile.
3. Per block: S = Q K^T (tcgen05, M64 N64, four K16 steps, fresh accumulator), the split's softmax (scale, window
   mask, row max, ``ex2.approx.ftz``, its row-sum tree), P to tensor memory in BF16, O = P V (four K16 steps onto
   a zeroed accumulator). Each block's (max, sum) per row goes to every CTA of the cluster (``st.async``).
4. With every block's (max, sum) in: each CTA weighs its own blocks' O (``exp2(m - m_max)``, zero for an empty
   split) and sums them in the combine's tree for its split class (splits ``r mod 8``: the combine's thread holds
   exactly those), then sends the sums to the CTA owning the token (cluster rank ``t`` owns token ``t``'s 8 heads).
5. The owner finishes the combine's tree across the 8 classes, divides by the denominator (the combine's
   butterfly over its 64 split lanes), and applies the output norm-gate.

Numerics: each statement is the served Triton build's instruction (``_swa_split_kernel`` / ``_swa_combine_kernel``
at ``MULTI_KEYS = MULTI_ROWS = 64`` with the fused prep, Q in FP8 and the fused gate, as Triton 3.8 compiles them
for sm_103a): the same tcgen05 instruction descriptors, operand layouts and K order, ``add.rn.ftz`` /
``rsqrt.approx.ftz`` / ``div.full`` / ``ex2.approx(.ftz)`` / ``cvt.rn.satfinite.e4m3x2`` / ``mul.bf16`` where it
has them, its packed-add trees in the same pairing, and its contracted FMA (the denominator's first butterfly
level). Every arithmetic statement is inline PTX with an explicit rounding mode, so nothing is re-contracted. The
KV cache bytes and the gated output are the two-grid build's.

Only ``swa_block.SwaBlockMultiAttention`` calls this module, under ``TRTLLM_MUSESPARK_SWA_BLOCK_CLUSTER=1`` and the
conditions it checks (``fits``).
"""

import math
from typing import Optional

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

#: CTAs per cluster, one cluster per (request, KV head). The 34 splits go round-robin, so CTA ``r`` holds every
#: split of the combine's tree class ``r mod 8``.
CLUSTER = 8
#: Query rows of one KV head: 8 tokens x 8 query heads (row = token * 8 + head in group).
ROWS = 64
TOKENS = 8
GROUP = 8
D = 64
#: Keys per block (the multi-request tile), the window and its block count at that tile.
BLOCK = 64
WINDOW = 2048
NUM_SPLITS = 34
#: Most blocks one CTA owns (ranks 0 and 1 own five, the others four).
NB_MAX = 5
#: One warpgroup per owned block.
THREADS = 128 * NB_MAX

# Shared memory (bytes; the tiles 1024-aligned, as the 128-byte swizzle needs).
TILE_BYTES = ROWS * D * 2
Q_OFF = 0
K_OFF = TILE_BYTES
V_OFF = K_OFF + NB_MAX * TILE_BYTES
ML_OFF = V_OFF + NB_MAX * TILE_BYTES  # [NUM_SPLITS][ROWS] x (max, sum) fp32
ML_BYTES = NUM_SPLITS * ROWS * 8
T_OFF = ML_OFF + ML_BYTES  # [CLUSTER classes][8 heads][64 dims] fp32: the owner's class sums
T_BYTES = CLUSTER * GROUP * D * 4
BAR_OFF = T_OFF + T_BYTES  # s_bar[5], o_bar[5], ml_bar, t_bar
SLOT_OFF = BAR_OFF + 16 * 8  # the TMEM base address
DEN_OFF = SLOT_OFF + 16  # per head, the two halves' denominators
SMEM_BYTES = DEN_OFF + 64
# Tensor memory: block k at columns 96 k: O (lanes 0-15 of each subpartition, 64 columns), S (lanes 16-31, the
# same columns), P in BF16 (lanes 0-15, columns 64-95): the split's own placement, once per block.
TMEM_COLS = 512
COLS_PER_BLOCK = 96

#: tcgen05 kind::f16 instruction descriptors of the split (BF16 x BF16 -> FP32, M64 N64; PV with B MN-major).
IDESC_QK = 0x04100490
IDESC_PV = 0x04110490
#: High word of the split's shared-memory descriptors: SBO 1024 B, version 1, 128-byte swizzle.
DESC_HI = 0x40004040

NEG_BIG = -1.0e30

# Debug dump (``debug=`` tensor, uint8): Q tiles, then each split's (max, sum), then each split's O * dequant.
DBG_REQS = 8


STAMPS = 16


def dbg_layout(kv_heads: int) -> dict:
    q = DBG_REQS * kv_heads * ROWS * D * 2
    ml = DBG_REQS * NUM_SPLITS * kv_heads * ROWS * 8
    o = DBG_REQS * NUM_SPLITS * kv_heads * ROWS * D * 4
    ts = DBG_REQS * kv_heads * CLUSTER * STAMPS * 8
    return dict(q=0, ml=q, o=q + ml, ts=q + ml + o, bytes=q + ml + o + ts)


def owned_splits(rank: int) -> list:
    """The splits (key blocks) cluster rank ``rank`` runs, in its block-slot order."""
    return [rank + CLUSTER * k for k in range(NB_MAX) if rank + CLUSTER * k < NUM_SPLITS]


def fits(
    rows: int,
    reqs: int,
    window: int,
    num_splits: int,
    fused_prep: bool,
    q_fp8: bool,
    has_gate: bool,
    keys: int,
    tile_rows: int,
) -> Optional[str]:
    """``None`` when a multi-request call is one this block computes bit for bit, else why not."""
    if keys != BLOCK or tile_rows != ROWS:
        return f"tile {keys} keys x {tile_rows} rows (needs {BLOCK} x {ROWS})"
    if rows != TOKENS:
        return f"{rows} rows per request (needs {TOKENS})"
    if not 2 <= reqs <= 4:
        return f"{reqs} requests (needs 2..4)"
    if window != WINDOW or num_splits != NUM_SPLITS:
        return f"window {window} / {num_splits} splits (needs {WINDOW} / {NUM_SPLITS})"
    if not (fused_prep and q_fp8 and has_gate):
        return "needs the fused prep, an FP8 Q and the fused gate"
    return None


AD = llvm.AsmDialect.AD_ATT


def _v(x, loc, ip):
    return x.ir_value(loc=loc, ip=ip)


def _asm(ret, args, text, cons, side):
    return llvm.inline_asm(
        ret, args, text, cons, has_side_effects=side, is_align_stack=False, asm_dialect=AD
    )


# ------------------------------------------------------------------------------------------ PTX helpers
@dsl_user_op
def f1(text, a, *, loc=None, ip=None):
    return Float32(_asm(T.f32(), [_v(Float32(a), loc, ip)], text, "=f,f", False))


@dsl_user_op
def f2(text, a, b, *, loc=None, ip=None):
    return Float32(
        _asm(T.f32(), [_v(Float32(a), loc, ip), _v(Float32(b), loc, ip)], text, "=f,f,f", False)
    )


@dsl_user_op
def f3(text, a, b, c, *, loc=None, ip=None):
    return Float32(
        _asm(
            T.f32(),
            [_v(Float32(a), loc, ip), _v(Float32(b), loc, ip), _v(Float32(c), loc, ip)],
            text,
            "=f,f,f,f",
            False,
        )
    )


def add(a, b):
    return f2("add.rn.f32 $0, $1, $2;", a, b)


def add_ftz(a, b):
    return f2("add.rn.ftz.f32 $0, $1, $2;", a, b)


def sub(a, b):
    return f2("sub.rn.f32 $0, $1, $2;", a, b)


def mul(a, b):
    return f2("mul.rn.f32 $0, $1, $2;", a, b)


def mul_ftz(a, b):
    return f2("mul.rn.ftz.f32 $0, $1, $2;", a, b)


def fma(a, b, c):
    return f3("fma.rn.f32 $0, $1, $2, $3;", a, b, c)


def fma_ftz(a, b, c):
    return f3("fma.rn.ftz.f32 $0, $1, $2, $3;", a, b, c)


def div_full(a, b):
    return f2("div.full.f32 $0, $1, $2;", a, b)


def fmax(a, b):
    return f2("max.f32 $0, $1, $2;", a, b)


def rsqrt_ftz(a):
    return f1("rsqrt.approx.ftz.f32 $0, $1;", a)


def ex2_ftz(a):
    return f1("ex2.approx.ftz.f32 $0, $1;", a)


def neg(a):
    return f1("neg.f32 $0, $1;", a)


def bf16r(a):
    """fp32 -> BF16 (round to nearest even) -> fp32."""
    return f1("{\n\t.reg .b16 h;\n\tcvt.rn.bf16.f32 h, $1;\n\tcvt.f32.bf16 $0, h;\n\t}", a)


def bf16_mul(a, b):
    """``mul.bf16`` of two BF16-exact fp32 values, as fp32 (the served ``(x * s).to(bf16)``)."""
    return f2(
        "{\n\t.reg .b16 x, y, z;\n\tcvt.rn.bf16.f32 x, $1;\n\tcvt.rn.bf16.f32 y, $2;\n\t"
        "mul.rn.bf16 z, x, y;\n\tcvt.f32.bf16 $0, z;\n\t}",
        a,
        b,
    )


def sigmoid_bf16(g):
    """The combine's ``tl.sigmoid(g).to(bf16)``: ``1 / (1 + ex2.approx(g * -log2 e))``, ``div.full``."""
    return f1(
        "{\n\t.reg .f32 x, e;\n\tmul.rn.f32 x, $1, 0fBFB8AA3B;\n\tex2.approx.f32 e, x;\n\t"
        "add.rn.f32 e, e, 0f3F800000;\n\tdiv.full.f32 x, 0f3F800000, e;\n\t.reg .b16 h;\n\t"
        "cvt.rn.bf16.f32 h, x;\n\tcvt.f32.bf16 $0, h;\n\t}",
        g,
    )


@dsl_user_op
def e4m3x2(hi, lo, *, loc=None, ip=None):
    """``cvt.rn.satfinite.e4m3x2.f32`` (``hi`` in the upper byte) as the low 16 bits of an Int32."""
    return Int32(
        _asm(
            T.i32(),
            [_v(Float32(hi), loc, ip), _v(Float32(lo), loc, ip)],
            "{\n\t.reg .b16 q;\n\tcvt.rn.satfinite.e4m3x2.f32 q, $1, $2;\n\tcvt.u32.u16 $0, q;\n\t}",
            "=r,f,f",
            False,
        )
    )


@dsl_user_op
def e4m3x2_to_bf16x2(q, *, loc=None, ip=None):
    """Two E4M3 (the low 16 bits of ``q``, element 0 in the low byte) -> BF16x2 (exact)."""
    return Int32(
        _asm(
            T.i32(),
            [_v(Int32(q), loc, ip)],
            "{\n\t.reg .b16 q, a, b;\n\t.reg .b32 h;\n\t.reg .f32 x, y;\n\tcvt.u16.u32 q, $1;\n\t"
            "cvt.rn.f16x2.e4m3x2 h, q;\n\tmov.b32 {a, b}, h;\n\tcvt.f32.f16 x, a;\n\t"
            "cvt.f32.f16 y, b;\n\tcvt.rn.bf16x2.f32 $0, y, x;\n\t}",
            "=r,r",
            False,
        )
    )


@dsl_user_op
def lo_bf16(w, *, loc=None, ip=None):
    return Float32(_asm(T.f32(), [_v(Int32(w), loc, ip)], "shl.b32 $0, $1, 16;", "=f,r", False))


@dsl_user_op
def hi_bf16(w, *, loc=None, ip=None):
    return Float32(
        _asm(T.f32(), [_v(Int32(w), loc, ip)], "and.b32 $0, $1, 0xFFFF0000;", "=f,r", False)
    )


@dsl_user_op
def pack_bf16x2(hi, lo, *, loc=None, ip=None):
    """``cvt.rn.bf16x2.f32`` (``lo`` in the low half)."""
    return Int32(
        _asm(
            T.i32(),
            [_v(Float32(hi), loc, ip), _v(Float32(lo), loc, ip)],
            "cvt.rn.bf16x2.f32 $0, $1, $2;",
            "=r,f,f",
            False,
        )
    )


@dsl_user_op
def bits(x, *, loc=None, ip=None):
    return Int32(_asm(T.i32(), [_v(Float32(x), loc, ip)], "mov.b32 $0, $1;", "=r,f", False))


@dsl_user_op
def as_f32(x, *, loc=None, ip=None):
    return Float32(_asm(T.f32(), [_v(Int32(x), loc, ip)], "mov.b32 $0, $1;", "=f,r", False))


@dsl_user_op
def selp(a, b, p, *, loc=None, ip=None):
    """``p != 0 ? a : b`` (fp32)."""
    return Float32(
        _asm(
            T.f32(),
            [_v(Float32(a), loc, ip), _v(Float32(b), loc, ip), _v(Int32(p), loc, ip)],
            "{\n\t.reg .pred q;\n\tsetp.ne.s32 q, $3, 0;\n\tselp.f32 $0, $1, $2, q;\n\t}",
            "=f,f,f,r",
            False,
        )
    )


@dsl_user_op
def isel(a, b, p, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_v(Int32(a), loc, ip), _v(Int32(b), loc, ip), _v(Int32(p), loc, ip)],
            "{\n\t.reg .pred q;\n\tsetp.ne.s32 q, $3, 0;\n\tselp.b32 $0, $1, $2, q;\n\t}",
            "=r,r,r,r",
            False,
        )
    )


@dsl_user_op
def lt(a, b, *, loc=None, ip=None):
    """``a < b`` (signed) as 0 / 1."""
    return Int32(
        _asm(
            T.i32(),
            [_v(Int32(a), loc, ip), _v(Int32(b), loc, ip)],
            "{\n\t.reg .pred q;\n\tsetp.lt.s32 q, $1, $2;\n\tselp.u32 $0, 1, 0, q;\n\t}",
            "=r,r,r",
            False,
        )
    )


@dsl_user_op
def gt0(x, *, loc=None, ip=None):
    """``x > 0`` (fp32, ordered) as 0 / 1."""
    return Int32(
        _asm(
            T.i32(),
            [_v(Float32(x), loc, ip)],
            "{\n\t.reg .pred q;\n\tsetp.gt.f32 q, $1, 0f00000000;\n\tselp.u32 $0, 1, 0, q;\n\t}",
            "=r,f",
            False,
        )
    )


@dsl_user_op
def keep_sel(s, lo_e, hi_e, e, *, loc=None, ip=None):
    """The split's window mask on one score: ``lo_e <= e <= hi_e ? s : -inf``."""
    return Float32(
        _asm(
            T.f32(),
            [_v(Float32(s), loc, ip), _v(Int32(lo_e), loc, ip), _v(Int32(hi_e), loc, ip)],
            "{\n\t.reg .pred a, b;\n\tsetp.le.s32 a, $2, "
            + str(e)
            + ";\n\tsetp.ge.s32 b, $3, "
            + str(e)
            + ";\n\tand.pred a, a, b;\n\tselp.f32 $0, $1, 0fFF800000, a;\n\t}",
            "=f,f,r,r",
            False,
        )
    )


@dsl_user_op
def shfl_bfly(x, m, *, loc=None, ip=None):
    return Float32(
        _asm(
            T.f32(),
            [_v(Float32(x), loc, ip)],
            f"shfl.sync.bfly.b32 $0, $1, {m}, 31, -1;",
            "=f,f",
            True,
        )
    )


@dsl_user_op
def shfl_idx(x, src, *, loc=None, ip=None):
    return Float32(
        _asm(
            T.f32(),
            [_v(Float32(x), loc, ip)],
            f"shfl.sync.idx.b32 $0, $1, {src}, 31, -1;",
            "=f,f",
            True,
        )
    )


# ---- memory
@dsl_user_op
def ld_g_v4(addr, pred, *, loc=None, ip=None):
    """16 bytes from global memory as 4 Int32 (zeros where ``pred`` is 0)."""
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_v(Int64(addr), loc, ip), _v(Int32(pred), loc, ip)],
        "{\n\t.reg .pred q;\n\tsetp.ne.s32 q, $5, 0;\n\tmov.b32 $0, 0;\n\tmov.b32 $1, 0;\n\t"
        "mov.b32 $2, 0;\n\tmov.b32 $3, 0;\n\t@q ld.global.v4.b32 {$0, $1, $2, $3}, [$4];\n\t}",
        "=r,=r,=r,=r,l,r",
        True,
    )
    return [Int32(llvm.extractvalue(T.i32(), res, [i])) for i in range(4)]


@dsl_user_op
def ld_g_v2(addr, pred, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 2),
        [_v(Int64(addr), loc, ip), _v(Int32(pred), loc, ip)],
        "{\n\t.reg .pred q;\n\tsetp.ne.s32 q, $3, 0;\n\tmov.b32 $0, 0;\n\tmov.b32 $1, 0;\n\t"
        "@q ld.global.v2.b32 {$0, $1}, [$2];\n\t}",
        "=r,=r,l,r",
        True,
    )
    return [Int32(llvm.extractvalue(T.i32(), res, [i])) for i in range(2)]


@dsl_user_op
def ld_g_b32(addr, pred, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_v(Int64(addr), loc, ip), _v(Int32(pred), loc, ip)],
            "{\n\t.reg .pred q;\n\tsetp.ne.s32 q, $2, 0;\n\tmov.b32 $0, 0;\n\t"
            "@q ld.global.b32 $0, [$1];\n\t}",
            "=r,l,r",
            True,
        )
    )


@dsl_user_op
def st_g_v2(addr, a, b, pred, *, loc=None, ip=None):
    _asm(
        None,
        [
            _v(Int64(addr), loc, ip),
            _v(Int32(a), loc, ip),
            _v(Int32(b), loc, ip),
            _v(Int32(pred), loc, ip),
        ],
        "{\n\t.reg .pred q;\n\tsetp.ne.s32 q, $3, 0;\n\t@q st.global.v2.b32 [$0], {$1, $2};\n\t}",
        "l,r,r,r",
        True,
    )


@dsl_user_op
def st_g_b32(addr, a, pred, *, loc=None, ip=None):
    _asm(
        None,
        [_v(Int64(addr), loc, ip), _v(Int32(a), loc, ip), _v(Int32(pred), loc, ip)],
        "{\n\t.reg .pred q;\n\tsetp.ne.s32 q, $2, 0;\n\t@q st.global.b32 [$0], $1;\n\t}",
        "l,r,r",
        True,
    )


@dsl_user_op
def st_s_v4(addr, w, pred, *, loc=None, ip=None):
    _asm(
        None,
        [_v(Int32(addr), loc, ip)]
        + [_v(Int32(x), loc, ip) for x in w]
        + [_v(Int32(pred), loc, ip)],
        "{\n\t.reg .pred q;\n\tsetp.ne.s32 q, $5, 0;\n\t@q st.shared.v4.b32 [$0], {$1, $2, $3, $4};\n\t}",
        "r,r,r,r,r,r",
        True,
    )


def isel4(a, b, p):
    """Element-wise ``p ? a : b`` of two lists of Int32."""
    return [isel(x, y, p) for x, y in zip(a, b)]


@dsl_user_op
def st_s_b32_pred(addr, a, pred, *, loc=None, ip=None):
    _asm(
        None,
        [_v(Int32(addr), loc, ip), _v(Int32(a), loc, ip), _v(Int32(pred), loc, ip)],
        "{\n\t.reg .pred q;\n\tsetp.ne.s32 q, $2, 0;\n\t@q st.shared.b32 [$0], $1;\n\t}",
        "r,r,r",
        True,
    )


@dsl_user_op
def bar_named(bar_id, count, *, loc=None, ip=None):
    """``bar.sync`` on named barrier ``bar_id`` (a register) for ``count`` threads."""
    _asm(None, [_v(Int32(bar_id), loc, ip)], f"bar.sync $0, {count};", "r", True)


@dsl_user_op
def stamp(addr, pred, *, loc=None, ip=None):
    """``%clock64`` to global memory (the debug build's phase stamps)."""
    _asm(
        None,
        [_v(Int64(addr), loc, ip), _v(Int32(pred), loc, ip)],
        "{\n\t.reg .pred q;\n\t.reg .b64 c;\n\tsetp.ne.s32 q, $1, 0;\n\tmov.u64 c, %clock64;\n\t"
        "@q st.global.b64 [$0], c;\n\t}",
        "l,r",
        True,
    )


@dsl_user_op
def st_s_b32(addr, a, *, loc=None, ip=None):
    _asm(
        None,
        [_v(Int32(addr), loc, ip), _v(Int32(a), loc, ip)],
        "st.shared.b32 [$0], $1;",
        "r,r",
        True,
    )


@dsl_user_op
def ld_s_b32(addr, *, loc=None, ip=None):
    return Int32(_asm(T.i32(), [_v(Int32(addr), loc, ip)], "ld.shared.b32 $0, [$1];", "=r,r", True))


@dsl_user_op
def ld_s_v2f(addr, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.f32()] * 2),
        [_v(Int32(addr), loc, ip)],
        "ld.shared.v2.f32 {$0, $1}, [$2];",
        "=f,=f,r",
        True,
    )
    return [Float32(llvm.extractvalue(T.f32(), res, [i])) for i in range(2)]


@dsl_user_op
def ld_s_v4f(addr, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [_v(Int32(addr), loc, ip)],
        "ld.shared.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,r",
        True,
    )
    return [Float32(llvm.extractvalue(T.f32(), res, [i])) for i in range(4)]


@dsl_user_op
def mapa(addr, rank, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_v(Int32(addr), loc, ip), _v(Int32(rank), loc, ip)],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            False,
        )
    )


@dsl_user_op
def st_async(raddr, vals, rbar, pred, *, loc=None, ip=None):
    """``st.async`` of 2 or 4 fp32 into a cluster peer's shared memory, crediting its mbarrier."""
    n = len(vals)
    regs = ", ".join(f"${i + 1}" for i in range(n))
    _asm(
        None,
        [_v(Int32(raddr), loc, ip)]
        + [_v(bits(x), loc, ip) for x in vals]
        + [_v(Int32(rbar), loc, ip), _v(Int32(pred), loc, ip)],
        "{\n\t.reg .pred q;\n\tsetp.ne.s32 q, $" + str(n + 2) + ", 0;\n\t"
        f"@q st.async.shared::cluster.mbarrier::complete_tx::bytes.v{n}.b32 [$0], {{{regs}}}, [${n + 1}];\n\t}}",
        ",".join(["r"] * (n + 3)),
        True,
    )


# ---- barriers
@dsl_user_op
def bar_init(addr, pred, *, loc=None, ip=None):
    _asm(
        None,
        [_v(Int32(addr), loc, ip), _v(Int32(pred), loc, ip)],
        "{\n\t.reg .pred q;\n\tsetp.ne.s32 q, $1, 0;\n\t@q mbarrier.init.shared::cta.b64 [$0], 1;\n\t}",
        "r,r",
        True,
    )


@dsl_user_op
def bar_expect(addr, tx, pred, *, loc=None, ip=None):
    _asm(
        None,
        [_v(Int32(addr), loc, ip), _v(Int32(pred), loc, ip)],
        "{\n\t.reg .pred q;\n\tsetp.ne.s32 q, $1, 0;\n\t"
        f"@q mbarrier.arrive.expect_tx.shared::cta.b64 _, [$0], {tx};\n\t}}",
        "r,r",
        True,
    )


@dsl_user_op
def bar_arrive(addr, pred, *, loc=None, ip=None):
    """``mbarrier.arrive`` (release, CTA scope) when ``pred``."""
    _asm(
        None,
        [_v(Int32(addr), loc, ip), _v(Int32(pred), loc, ip)],
        "{\n\t.reg .pred q;\n\tsetp.ne.s32 q, $1, 0;\n\t@q mbarrier.arrive.shared::cta.b64 _, [$0];\n\t}",
        "r,r",
        True,
    )


@dsl_user_op
def bar_wait(addr, pred, cluster_scope, *, loc=None, ip=None):
    """Wait for phase 0 of the mbarrier at ``addr`` when ``pred``; acquire at cluster scope for peer data."""
    scope = ".acquire.cluster" if cluster_scope else ""
    _asm(
        None,
        [_v(Int32(addr), loc, ip), _v(Int32(pred), loc, ip)],
        "{\n\t.reg .pred q, done;\n\t.reg .b32 ph;\n\tmov.b32 ph, 0;\n\tsetp.ne.s32 q, $1, 0;\n\t"
        "@!q bra.uni WDONE;\n\tWLOOP:\n\t"
        "mbarrier.try_wait.parity" + scope + ".shared::cta.b64 done, [$0], ph;\n\t"
        "@!done bra.uni WLOOP;\n\tWDONE:\n\t}",
        "r,r",
        True,
    )


@dsl_user_op
def raw(text, *, loc=None, ip=None):
    _asm(None, [], text, "", True)


# ---- tensor memory and the MMA
@dsl_user_op
def tmem_alloc(slot, pred, *, loc=None, ip=None):
    _asm(
        None,
        [_v(Int32(slot), loc, ip), _v(Int32(pred), loc, ip)],
        "{\n\t.reg .pred q;\n\tsetp.ne.s32 q, $1, 0;\n\t"
        f"@q tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [$0], {TMEM_COLS};\n\t"
        "@q tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n\t}",
        "r,r",
        True,
    )


@dsl_user_op
def tmem_free(taddr, pred, *, loc=None, ip=None):
    _asm(
        None,
        [_v(Int32(taddr), loc, ip), _v(Int32(pred), loc, ip)],
        "{\n\t.reg .pred q;\n\tsetp.ne.s32 q, $1, 0;\n\t"
        f"@q tcgen05.dealloc.cta_group::1.sync.aligned.b32 $0, {TMEM_COLS};\n\t}}",
        "r,r",
        True,
    )


@dsl_user_op
def tmem_ld(taddr, n, *, loc=None, ip=None):
    """``tcgen05.ld.16x32bx2.x{n}`` (half split 32): thread l gets lane l % 16, columns 32 (l / 16) + [0, n)."""
    regs = ", ".join(f"${i}" for i in range(n))
    res = _asm(
        llvm.StructType.get_literal([T.f32()] * n),
        [_v(Int32(taddr), loc, ip)],
        f"tcgen05.ld.sync.aligned.16x32bx2.x{n}.b32 {{{regs}}}, [${n}], 32;\n\t"
        "tcgen05.wait::ld.sync.aligned;",
        ",".join(["=f"] * n) + ",r",
        True,
    )
    return [Float32(llvm.extractvalue(T.f32(), res, [i])) for i in range(n)]


@dsl_user_op
def tmem_st_zero32(taddr, *, loc=None, ip=None):
    zeros = ", ".join(["$1"] * 32)
    _asm(
        None,
        [_v(Int32(taddr), loc, ip), _v(Int32(0), loc, ip)],
        "tcgen05.st.sync.aligned.16x32bx2.x32.b32 [$0], 32, {" + zeros + "};",
        "r,r",
        True,
    )


@dsl_user_op
def tmem_st16(taddr, words, *, loc=None, ip=None):
    regs = ", ".join(f"${i + 1}" for i in range(16))
    _asm(
        None,
        [_v(Int32(taddr), loc, ip)] + [_v(Int32(w), loc, ip) for w in words],
        "tcgen05.st.sync.aligned.16x32bx2.x16.b32 [$0], 16, {" + regs + "};",
        "r" + ",r" * 16,
        True,
    )


@dsl_user_op
def mma4(
    d_taddr,
    a_lo,
    b_lo,
    bar,
    pred,
    a_tmem,
    accumulate_first,
    idesc,
    a_step,
    b_step,
    *,
    loc=None,
    ip=None,
):
    """Four tcgen05 kind::f16 K16 steps as the split issues them, then a commit to ``bar``: D at ``d_taddr``;
    A in shared memory (descriptor low word ``a_lo`` + ``a_step`` per step) or in tensor memory (column
    ``a_lo`` + 8 per step); B descriptor low word ``b_lo`` + ``b_step`` per step; the first step accumulates iff
    ``accumulate_first``, the others always."""
    lines = [
        "{",
        ".reg .pred q, acc0, acc1;",
        ".reg .b64 a, b;",
        ".reg .b32 al, bl, hi, id;",
        "setp.ne.s32 q, $4, 0;",
        f"mov.pred acc0, {1 if accumulate_first else 0};",
        "mov.pred acc1, 1;",
        f"mov.b32 hi, {DESC_HI};",
        f"mov.b32 id, {idesc};",
    ]
    for kk in range(4):
        acc = "acc0" if kk == 0 else "acc1"
        lines.append(f"add.s32 bl, $2, {b_step * kk};")
        lines.append("mov.b64 b, {bl, hi};")
        if a_tmem:
            lines.append(f"add.s32 al, $1, {8 * kk};")
            lines.append(f"@q tcgen05.mma.cta_group::1.kind::f16 [$0], [al], b, id, {acc};")
        else:
            lines.append(f"add.s32 al, $1, {a_step * kk};")
            lines.append("mov.b64 a, {al, hi};")
            lines.append(f"@q tcgen05.mma.cta_group::1.kind::f16 [$0], a, b, id, {acc};")
    lines.append("@q tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [$3];")
    lines.append("}")
    _asm(
        None,
        [
            _v(Int32(d_taddr), loc, ip),
            _v(Int32(a_lo), loc, ip),
            _v(Int32(b_lo), loc, ip),
            _v(Int32(bar), loc, ip),
            _v(Int32(pred), loc, ip),
        ],
        "\n\t".join(lines),
        "r,r,r,r,r",
        True,
    )


@dsl_user_op
def pair2(text, a0, a1, b0, b1, *, loc=None, ip=None):
    """A packed ``.f32x2`` op on (a0, a1), (b0, b1) (``text`` uses x, y -> z); two fp32."""
    res = _asm(
        llvm.StructType.get_literal([T.f32()] * 2),
        [
            _v(Float32(a0), loc, ip),
            _v(Float32(a1), loc, ip),
            _v(Float32(b0), loc, ip),
            _v(Float32(b1), loc, ip),
        ],
        "{\n\t.reg .b64 x, y, z;\n\tmov.b64 x, {$2, $3};\n\tmov.b64 y, {$4, $5};\n\t"
        + text
        + "\n\tmov.b64 {$0, $1}, z;\n\t}",
        "=f,=f,f,f,f,f",
        False,
    )
    return [Float32(llvm.extractvalue(T.f32(), res, [i])) for i in range(2)]


def add2(a, b):
    """``add.rn.f32x2`` of two (lo, hi) pairs: the split's and the combine's packed adds."""
    return pair2("add.rn.f32x2 z, x, y;", a[0], a[1], b[0], b[1])


def max3(a, b, c):
    return f3("max.f32 $0, $1, $2, $3;", a, b, c)


@dsl_user_op
def scale_mask2(a0, a1, qk_scale, nlo, span, e, *, loc=None, ip=None):
    """Scores ``e``, ``e + 1`` of a row half: ``mul.rn.f32x2`` by the QK scale, then the split's window mask as one
    unsigned range test (``e - lo <= span``) per score."""
    res = _asm(
        llvm.StructType.get_literal([T.f32()] * 2),
        [
            _v(Float32(a0), loc, ip),
            _v(Float32(a1), loc, ip),
            _v(Float32(qk_scale), loc, ip),
            _v(Int32(nlo), loc, ip),
            _v(Int32(span), loc, ip),
        ],
        "{\n\t.reg .b64 x, y, z;\n\t.reg .f32 s0, s1;\n\t.reg .b32 t;\n\t.reg .pred p;\n\t"
        "mov.b64 x, {$2, $3};\n\tmov.b64 y, {$4, $4};\n\tmul.rn.f32x2 z, x, y;\n\tmov.b64 {s0, s1}, z;\n\t"
        f"add.s32 t, $5, {e};\n\tsetp.le.u32 p, t, $6;\n\tselp.f32 $0, s0, 0fFF800000, p;\n\t"
        f"add.s32 t, $5, {e + 1};\n\tsetp.le.u32 p, t, $6;\n\tselp.f32 $1, s1, 0fFF800000, p;\n\t}}",
        "=f,=f,f,f,f,r,r",
        False,
    )
    return [Float32(llvm.extractvalue(T.f32(), res, [i])) for i in range(2)]


@dsl_user_op
def tmem_ld8x(taddrs, *, loc=None, ip=None):
    """``tcgen05.ld.16x32bx2.x8`` (half split 32) at each address, one ``wait::ld`` for all."""
    n = len(taddrs)
    lines = []
    for b in range(n):
        regs = ", ".join(f"${8 * b + i}" for i in range(8))
        lines.append(f"tcgen05.ld.sync.aligned.16x32bx2.x8.b32 {{{regs}}}, [${8 * n + b}], 32;")
    lines.append("tcgen05.wait::ld.sync.aligned;")
    res = _asm(
        llvm.StructType.get_literal([T.f32()] * (8 * n)),
        [_v(Int32(a), loc, ip) for a in taddrs],
        "\n\t".join(lines),
        ",".join(["=f"] * (8 * n)) + "," + ",".join(["r"] * n),
        True,
    )
    vals = [Float32(llvm.extractvalue(T.f32(), res, [i])) for i in range(8 * n)]
    return [vals[8 * b : 8 * b + 8] for b in range(n)]


def desc_lo(saddr):
    """Low word of the split's shared-memory descriptor: the start address >> 4 (14 bits)."""
    return (saddr >> 4) & 0x3FFF


# ------------------------------------------------------------------------------------------ math blocks
def unpack8(words):
    out = []
    for w in words:
        out.append(lo_bf16(w))
        out.append(hi_bf16(w))
    return out


def prep8(words, cosv, sinv, eps, scale, rope):
    """``_prep_rows_math`` of this lane's 8 elements (``words``: BF16x2 of elements 8 l .. 8 l + 7 of a head;
    8 lanes per head): the squares (``mul.f32``) summed in order with ``add.rn.ftz``, the lanes paired over
    bits 0, 1, 2 (``add.f32``), ``rsqrt.approx.ftz(total / 64 + eps)`` with ``div.full``, the two BF16
    roundings, then (RoPE) the pair rotation with ``mul.rn.ftz`` / ``fma.rn.ftz``. 8 BF16-exact fp32."""
    x = unpack8(words)
    s = add_ftz(mul(x[0], x[0]), mul(x[1], x[1]))
    for i in range(2, 8):
        s = add_ftz(s, mul(x[i], x[i]))
    s = add(s, shfl_bfly(s, 1))
    s = add(s, shfl_bfly(s, 2))
    s = add(s, shfl_bfly(s, 4))
    inv = rsqrt_ftz(add(eps, div_full(s, Float32(64.0))))
    y = [bf16r(mul(bf16r(mul(v, inv)), scale)) for v in x]
    if rope:
        out = []
        for i in range(4):
            ev, od = y[2 * i], y[2 * i + 1]
            out.append(bf16r(fma_ftz(ev, cosv[i], mul_ftz(neg(od), sinv[i]))))
            out.append(bf16r(fma_ftz(od, cosv[i], mul_ftz(ev, sinv[i]))))
        y = out
    return y


def to_cache_fp8(vals, scale_b):
    """``_to_cache_fp8`` of 8 BF16-exact values: ``mul.bf16`` by the BF16 scale, then E4M3 (RN, satfinite).
    4 Int32, each two E4M3 in its low 16 bits (element 2 i in the low byte)."""
    y = [bf16_mul(v, scale_b) for v in vals]
    return [e4m3x2(y[2 * i + 1], y[2 * i]) for i in range(4)]


def fp8_to_bf16(pairs):
    return [e4m3x2_to_bf16x2(p) for p in pairs]


def fp8_words_to_bf16(words):
    """4 Int32 of 4 E4M3 each (16 elements) -> 8 BF16x2 words."""
    out = []
    for w in words:
        out.append(e4m3x2_to_bf16x2(w & 0xFFFF))
        out.append(e4m3x2_to_bf16x2((w >> 16) & 0xFFFF))
    return out


def store_tile_half(tile_row, key, kh, words8, valid):
    """32 BF16 (16 words) of a 64-wide tile row into chunks 4 kh .. 4 kh + 3 of the 128-byte swizzle."""
    for c in range(4):
        chunk = kh * 4 + c
        st_s_v4(tile_row + ((chunk ^ (key % 8)) * 16), words8[4 * c : 4 * c + 4], valid)


def rope_row(g_cos, pos, l8):
    """``cos[pos, 4 l8 .. 4 l8 + 3]`` and ``sin[...]`` of the fp32 ``[positions, 2, 32]`` table."""
    row = g_cos + Int64(pos) * 256 + Int64(l8 * 16)
    return [as_f32(x) for x in ld_g_v4(row, Int32(1))], [
        as_f32(x) for x in ld_g_v4(row + 128, Int32(1))
    ]


def softmax(acc, qk_scale, nlo, span):
    """The split's online-softmax step on one block (one row half per thread, 32 scores): the scale, the window
    mask (``nlo`` = -lo_e, ``span`` = hi_e - lo_e, or an empty range), the row max (exact in any order), ``ex2``,
    the row sum in the split's packed tree (pairs (p 2k, p 2k+1) added over k = 0..15 pairwise, then even + odd,
    then the other half). Returns (m_new, l_new, P as 16 BF16x2 words)."""
    sc = []
    for e in range(0, 32, 2):
        sc += scale_mask2(acc[e], acc[e + 1], qk_scale, nlo, span, e)
    mx = max_tree(sc)
    m_new = max3(Float32(NEG_BIG), mx, shfl_bfly(mx, 16))
    alpha = ex2_ftz(sub(Float32(NEG_BIG), m_new))
    p = [ex2_ftz(sub(x, m_new)) for x in sc]
    v = [(p[2 * k], p[2 * k + 1]) for k in range(16)]
    while len(v) > 1:
        v = [add2(v[2 * i], v[2 * i + 1]) for i in range(len(v) // 2)]
    ssum = add(v[0][0], v[0][1])
    ssum = add(ssum, shfl_bfly(ssum, 16))
    l_new = fma(Float32(0.0), alpha, ssum)
    return m_new, l_new, [pack_bf16x2(p[2 * i + 1], p[2 * i]) for i in range(16)]


def max_tree(v):
    """Max of a list (exact in any order), three at a time."""
    v = list(v)
    while len(v) > 1:
        nxt = [max3(v[i], v[i + 1], v[i + 2]) for i in range(0, len(v) - 2, 3)]
        if len(v) % 3 == 1:
            nxt.append(v[-1])
        elif len(v) % 3 == 2:
            nxt.append(fmax(v[-2], v[-1]))
        v = nxt
    return v[0]


def row_max(sml, row, half):
    """Max over the splits' maxima of ``row`` and -1e30 (the padded lanes'), the two lanes of a row (``half`` 0
    and 1, lanes l and l ^ 16) each loading every other split first, then a tree and one shuffle."""
    v = [ld_s_v2f(sml + ((2 * i + half) * ROWS + row) * 8)[0] for i in range(NUM_SPLITS // 2)]
    m = max_tree([Float32(NEG_BIG)] + v)
    return fmax(m, shfl_bfly(m, 16))


def class_sums(xs):
    """The combine's in-thread tree over the splits ``r + 8 k``, k = 0..7 (absent ones +0):
    ((x0 + x1) + (x2 + x3)) + ((x4 + x5) + (x6 + x7)), on column pairs with ``add.rn.f32x2``."""
    out = []
    zero = (Float32(0.0), Float32(0.0))
    for i in range(0, len(xs[0]), 2):
        x = [(xs[k][i], xs[k][i + 1]) for k in range(len(xs))] + [zero] * (8 - len(xs))
        out += add2(
            add2(add2(x[0], x[1]), add2(x[2], x[3])), add2(add2(x[4], x[5]), add2(x[6], x[7]))
        )
    return out


def denominators(sml, rows, lane, den_lane):
    """The combine's denominator of each row in ``rows`` over its 64 split lanes (two 32-lane halves): ``w l``
    (``mul.f32``) shuffled 16, contracted into ``fma(w, l, partner)``, then lanes 8, 4, 2, 1 (``add.f32``); per
    half the value of lane ``den_lane`` (the one the combine's shared-memory store keeps); the two halves added.
    The chains are emitted level by level, interleaved."""
    mm = []
    for row in rows:
        lo_ = ld_s_v2f(sml + (lane * ROWS + row) * 8)[0]
        hi_ = selp(
            ld_s_v2f(
                sml
                + ((isel(lane + 32, Int32(0), lt(lane, Int32(NUM_SPLITS - 32)))) * ROWS + row) * 8
            )[0],
            Float32(NEG_BIG),
            lt(lane, Int32(NUM_SPLITS - 32)),
        )
        mm.append(max3(Float32(NEG_BIG), lo_, hi_))
    for m in (16, 8, 4, 2, 1):
        mm = [fmax(x, shfl_bfly(x, m)) for x in mm]
    chains = []
    for i, row in enumerate(rows):
        for hh in range(2):
            s = 32 * hh + lane
            sok = lt(s, Int32(NUM_SPLITS))
            ml = ld_s_v2f(sml + (isel(s, Int32(0), sok) * ROWS + row) * 8)
            chains.append((i, sok, ml))

    ws = []
    for i, sok, ml in chains:
        m_s = selp(ml[0], Float32(NEG_BIG), sok)
        l_s = selp(ml[1], Float32(0.0), sok)
        ws.append(
            (selp(selp(ex2_ftz(sub(m_s, mm[i])), Float32(0.0), gt0(l_s)), Float32(0.0), sok), l_s)
        )
    parts = [shfl_bfly(mul(w, lv), 16) for w, lv in ws]
    a = [fma(w, lv, q) for (w, lv), q in zip(ws, parts)]
    for m in (8, 4, 2, 1):
        sh = [shfl_bfly(x, m) for x in a]
        a = [add(x, y) for x, y in zip(a, sh)]
    win = [shfl_idx(x, den_lane) for x in a]
    return [add(win[2 * i], win[2 * i + 1]) for i in range(len(rows))]


def gate_rows(num, den, sig, gate_eps):
    """The combine's ``(num / den).to(bf16)`` and its output norm-gate on this thread's 4 dims (16 lanes per
    head, dims 4 (lane % 16) + [0, 4)): the square tree over dims bits 4, 3, 2 (lanes 4, 2, 1), then 1, 0
    (in thread), then 5 (lane 8); the BF16 sigmoid; ``mul.bf16``. 4 BF16-exact fp32."""
    att = [bf16r(div_full(n, den)) for n in num]
    lv = [add(mul(a, a), shfl_bfly(mul(a, a), 4)) for a in att]
    lv = [add(x, shfl_bfly(x, 2)) for x in lv]
    lv = [add(x, shfl_bfly(x, 1)) for x in lv]
    tot = add(add(lv[0], lv[2]), add(lv[1], lv[3]))
    tot = add(tot, shfl_bfly(tot, 8))
    rs = rsqrt_ftz(add(gate_eps, div_full(tot, Float32(64.0))))
    return [bf16_mul(bf16r(mul(rs, att[i])), sig[i]) for i in range(4)]


class SwaClusterBlockKernel:
    """The kernel of one build: PDL, the rank's KV heads, the cache page, the QKV layout, RoPE, the debug output
    (0 none, 1 the partials dump, 2 per-phase clock stamps) and the denominator lane are compile-time constants.

    Thread roles (``THREADS`` = one warpgroup per owned block): warpgroup ``g`` widens block ``g``'s cached K/V,
    runs its softmax and issues its PV; warpgroups 0-3 run the query prep (one 64-wide head per 8 lanes) and the
    class sums (8 of each half's 32 O columns each); warpgroup 4 runs the step keys' prep and the cache write, then
    the owner's denominators; warpgroup 0 finishes the owner's rows."""

    def __init__(
        self,
        enable_pdl: bool,
        kv_heads: int,
        page: int,
        permuted: bool,
        rope: bool,
        debug: int,
        den_lane: int,
    ):
        self.enable_pdl = enable_pdl
        self.hkv = kv_heads
        self.page = page
        self.rope = rope
        self.debug = debug
        self.den_lane = den_lane
        self.q_stride = D if permuted else 2 * D
        self.k_col0 = (GROUP * kv_heads if permuted else 2 * GROUP * kv_heads) * D
        self.v_col0 = self.k_col0 + kv_heads * D
        self.dbg = dbg_layout(kv_heads)

    @cute.jit
    def __call__(
        self,
        qkv_ptr: cute.Pointer,
        pool_ptr: cute.Pointer,
        bt_ptr: cute.Pointer,
        seq_ptr: cute.Pointer,
        scale_ptr: cute.Pointer,
        cos_ptr: cute.Pointer,
        pos_ptr: cute.Pointer,
        gate_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        dbg_ptr: cute.Pointer,
        reqs: Int32,
        qkv_stride: Int32,
        bt_stride_req: Int32,
        bt_stride_kv: Int32,
        gate_stride: Int32,
        gate_hstride: Int32,
        out_stride: Int32,
        prep_eps: Float32,
        q_scale: Float32,
        gate_eps: Float32,
        sm_scale_log2: Float32,
        stream,
    ):
        self.kernel(
            qkv_ptr,
            pool_ptr,
            bt_ptr,
            seq_ptr,
            scale_ptr,
            cos_ptr,
            pos_ptr,
            gate_ptr,
            out_ptr,
            dbg_ptr,
            qkv_stride,
            bt_stride_req,
            bt_stride_kv,
            gate_stride,
            gate_hstride,
            out_stride,
            prep_eps,
            q_scale,
            gate_eps,
            sm_scale_log2,
        ).launch(
            grid=(CLUSTER, self.hkv, reqs),
            block=[THREADS, 1, 1],
            cluster=(CLUSTER, 1, 1),
            smem=SMEM_BYTES + 1024,
            stream=stream,
            use_pdl=self.enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        qkv_ptr: cute.Pointer,
        pool_ptr: cute.Pointer,
        bt_ptr: cute.Pointer,
        seq_ptr: cute.Pointer,
        scale_ptr: cute.Pointer,
        cos_ptr: cute.Pointer,
        pos_ptr: cute.Pointer,
        gate_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        dbg_ptr: cute.Pointer,
        qkv_stride: Int32,
        bt_stride_req: Int32,
        bt_stride_kv: Int32,
        gate_stride: Int32,
        gate_hstride: Int32,
        out_stride: Int32,
        prep_eps: Float32,
        q_scale: Float32,
        gate_eps: Float32,
        sm_scale_log2: Float32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        _, h, b = cute.arch.block_idx()
        wg = tid // 128  # warpgroup = this thread's block slot
        wt = tid % 128
        warp = wt // 32  # TMEM subpartition
        lane = tid % 32
        l16 = lane % 16
        half = lane // 16
        l8 = tid % 8
        is_t0 = lt(tid, Int32(1))
        # warp 16 (warpgroup 4's first) owns the tensor memory: it is idle while warpgroup 0 finishes the rows
        is_w16 = lt(tid, Int32(544)) * (1 - lt(tid, Int32(512)))
        lead = lt(wt, Int32(1))
        prep_q = lt(tid, Int32(512))  # warpgroups 0-3
        one = Int32(1)

        smem = utils.SmemAllocator()
        sbase = smem.allocate(SMEM_BYTES, byte_alignment=1024).toint()
        g_qkv = qkv_ptr.toint()
        g_pool = pool_ptr.toint()
        g_bt = bt_ptr.toint() + Int64(b) * Int64(bt_stride_req) * 4
        g_cos = cos_ptr.toint()
        g_pos = pos_ptr.toint()
        g_out = out_ptr.toint()
        g_dbg = dbg_ptr.toint()
        g_ts = (
            g_dbg
            + Int64(self.dbg["ts"])
            + Int64(((b * self.hkv + h) * CLUSTER + rank) * STAMPS * 8)
        )
        sml = sbase + ML_OFF
        s_bar = sbase + BAR_OFF
        o_bar = s_bar + 40
        ml_bar = s_bar + 80
        t_bar = s_bar + 88
        if cutlass.const_expr(self.debug == 2):
            stamp(g_ts, is_t0)
        for k in cutlass.range_constexpr(NB_MAX):
            bar_init(s_bar + 8 * k, is_t0)
            bar_init(o_bar + 8 * k, is_t0)
        bar_init(ml_bar, is_t0)
        bar_init(t_bar, is_t0)
        raw("fence.mbarrier_init.release.cluster;")
        bar_expect(ml_bar, ML_BYTES, is_t0)
        bar_expect(t_bar, T_BYTES, is_t0)

        # ------------------------------------------------ before the wait: graph inputs and the cached K/V
        # round trip 1: the length, the scale, this thread's RoPE position (query token tid / 64, or the step key
        # (tid % 128) / 8 of warpgroup 4)
        L = ld_g_b32(seq_ptr.toint() + Int64(b) * 4, one)
        scale = as_f32(ld_g_b32(scale_ptr.toint(), one))
        tq = isel(tid // 64, (wt // 8) % TOKENS, prep_q)
        tw = (wt // 8) % TOKENS
        if cutlass.const_expr(self.rope):
            pos = ld_g_b32(g_pos + Int64(b * TOKENS + tq) * 4, one)
        past = L - TOKENS
        lo = cutlass.max(past - (WINDOW - 1), Int32(0))
        base = (lo // BLOCK) * BLOCK
        nblk = (L - base + (BLOCK - 1)) // BLOCK
        scale_b = bf16r(scale)
        dequant = div_full(Float32(1.0), scale)
        qk_scale = mul(mul(sm_scale_log2, dequant), dequant)
        s = rank + CLUSTER * wg  # this warpgroup's split
        live = lt(s, nblk)
        valid = lt(s, Int32(NUM_SPLITS))
        # round trip 2: the block's page ids, the writer's page ids, the RoPE row
        key = wt // 2
        kh = wt % 2
        j = base + s * BLOCK + key
        cached = live * lt(j, past) * (1 - lt(j, lo))
        pg = j // self.page
        pk = ld_g_b32(g_bt + Int64(pg) * 4, cached)
        pv = ld_g_b32(g_bt + (Int64(bt_stride_kv) + Int64(pg)) * 4, cached)
        jw = past + tw
        writer = lt(rank, one) * (1 - prep_q) * lt(wt, Int32(64))
        wpk = ld_g_b32(g_bt + Int64(jw // self.page) * 4, writer)
        wpv = ld_g_b32(g_bt + (Int64(bt_stride_kv) + Int64(jw // self.page)) * 4, writer)
        if cutlass.const_expr(self.rope):
            cosv, sinv = rope_row(g_cos, pos, l8)
        else:
            cosv = None
            sinv = None
        # round trip 3: the block's cached K/V (FP8, 32 bytes of a key row per thread)
        inpage = Int64((j % self.page) * D + kh * 32)
        ka = g_pool + (Int64(pk) * self.hkv + Int64(h)) * (self.page * D) + inpage
        va = g_pool + (Int64(pv) * self.hkv + Int64(h)) * (self.page * D) + inpage
        kw = ld_g_v4(ka, cached) + ld_g_v4(ka + 16, cached)
        vw = ld_g_v4(va, cached) + ld_g_v4(va + 16, cached)
        tile_off = wg * TILE_BYTES + key * 128
        store_tile_half(sbase + K_OFF + tile_off, key, kh, fp8_words_to_bf16(kw), valid)
        store_tile_half(sbase + V_OFF + tile_off, key, kh, fp8_words_to_bf16(vw), valid)

        cute.arch.cluster_arrive_relaxed()
        cute.arch.cluster_wait()
        if cutlass.const_expr(self.debug == 2):
            stamp(g_ts + 8, is_t0)

        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()
        if cutlass.const_expr(self.debug == 2):
            stamp(g_ts + 16, is_t0)

        # ------------------------------------------------ after the wait: the projection, one round trip
        qr = tid // 8  # query row of warpgroups 0-3 (token qr / 8, head qr % 8)
        qa = (
            g_qkv
            + (
                Int64(b * TOKENS + qr // 8) * Int64(qkv_stride)
                + Int64((h * GROUP + qr % 8) * self.q_stride + l8 * 8)
            )
            * 2
        )
        row_k = Int64(b * TOKENS + tw) * Int64(qkv_stride)
        step = 1 - prep_q
        qw = ld_g_v4(qa, prep_q)
        kraw = ld_g_v4(g_qkv + (row_k + Int64(self.k_col0 + h * D + l8 * 8)) * 2, step)
        vraw = ld_g_v4(g_qkv + (row_k + Int64(self.v_col0 + h * D + l8 * 8)) * 2, step)
        og = wt // 16
        od = wt % 16
        ga = (
            gate_ptr.toint()
            + (
                Int64(b * TOKENS + rank) * Int64(gate_stride)
                + Int64((h * GROUP + og) * gate_hstride + od * 4)
            )
            * 2
        )
        gw = ld_g_v2(ga, lt(tid, Int32(128)))

        # Tensor memory only now, with the projection loads in flight: after the wait every predecessor has
        # completed (and freed its columns), and before the trigger no dependent can hold any, so the allocation
        # never waits on a kernel that waits on this one. The dependents launch once every CTA holds its columns.
        tmem_alloc(sbase + SLOT_OFF, is_w16)
        raw("tcgen05.fence::before_thread_sync;")
        cute.arch.barrier()
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_launch_dependents()
        raw("tcgen05.fence::after_thread_sync;")
        tmem = ld_s_b32(sbase + SLOT_OFF)

        # warpgroups 0-3: the query rows; warpgroup 4: the step keys (one prep each: 8 lanes per head)
        y = prep8(
            isel4(qw, kraw, prep_q),
            cosv,
            sinv,
            prep_eps,
            selp(q_scale, Float32(1.0), prep_q),
            self.rope,
        )
        qb = fp8_to_bf16(to_cache_fp8(y, scale_b))
        st_s_v4(sbase + Q_OFF + qr * 128 + ((l8 ^ (qr % 8)) * 16), qb, prep_q)
        if cutlass.const_expr(self.debug == 1):
            dq = (
                g_dbg
                + Int64(self.dbg["q"])
                + Int64(((b * self.hkv + h) * ROWS + qr) * D + l8 * 8) * 2
            )
            st_g_v2(dq, qb[0], qb[1], prep_q)
            st_g_v2(dq + 8, qb[2], qb[3], prep_q)
        k8 = to_cache_fp8(y, scale_b)
        v8 = to_cache_fp8(unpack8(vraw), scale_b)
        inw = Int64((jw % self.page) * D + l8 * 8)
        st_g_v2(
            g_pool + (Int64(wpk) * self.hkv + Int64(h)) * (self.page * D) + inw,
            k8[0] | (k8[1] << 16),
            k8[2] | (k8[3] << 16),
            writer,
        )
        st_g_v2(
            g_pool + (Int64(wpv) * self.hkv + Int64(h)) * (self.page * D) + inw,
            v8[0] | (v8[1] << 16),
            v8[2] | (v8[3] << 16),
            writer,
        )
        sj = (jw - base) // BLOCK
        kj = (jw - base) % BLOCK
        mine = (
            step * lt(wt, Int32(64)) * (1 - lt(sj % CLUSTER, rank)) * (1 - lt(rank, sj % CLUSTER))
        )
        prow = (sj // CLUSTER) * TILE_BYTES + kj * 128 + ((l8 ^ (kj % 8)) * 16)
        st_s_v4(sbase + K_OFF + prow, qb, mine)
        st_s_v4(sbase + V_OFF + prow, fp8_to_bf16(v8), mine)
        raw("fence.proxy.async.shared::cta;")
        cute.arch.barrier()
        if cutlass.const_expr(self.debug == 2):
            stamp(g_ts + 24, is_t0)

        # ------------------------------------------------ QK: each warpgroup's lead thread issues its block's
        mma4(
            tmem + (16 << 16) + COLS_PER_BLOCK * wg,
            desc_lo(sbase + Q_OFF),
            desc_lo(sbase + K_OFF + wg * TILE_BYTES),
            s_bar + 8 * wg,
            lead * live,
            False,
            False,
            IDESC_QK,
            2,
            2,
        )

        # ------------------------------------------------ each warpgroup: its block's softmax, PV, (max, sum) out
        row = 16 * warp + l16
        q_pos = past + row // 8
        lane_tm = (32 * warp) << 16
        if valid != 0:
            bar_wait(s_bar + 8 * wg, live, False)
            raw("tcgen05.fence::after_thread_sync;")
            col = tmem + lane_tm + COLS_PER_BLOCK * wg
            jb = base + s * BLOCK + 32 * half
            lo_e = cutlass.max(lo, q_pos - (WINDOW - 1)) - jb
            hi_e = q_pos - jb
            empty = lt(hi_e, lo_e)
            m_new, l_new, pw = softmax(
                tmem_ld(col + (16 << 16), 32),
                qk_scale,
                isel(Int32(-64), 0 - lo_e, empty),
                isel(Int32(0), hi_e - lo_e, empty),
            )
            m_k = selp(m_new, Float32(NEG_BIG), live)
            l_k = selp(l_new, Float32(0.0), live)
            tmem_st16(col + 64, pw)
            tmem_st_zero32(col)
            raw("tcgen05.wait::st.sync.aligned;")
            raw("tcgen05.fence::before_thread_sync;")
            bar_named(1 + wg, 128)
            raw("tcgen05.fence::after_thread_sync;")
            mma4(
                tmem + COLS_PER_BLOCK * wg,
                tmem + COLS_PER_BLOCK * wg + 64,
                desc_lo(sbase + V_OFF + wg * TILE_BYTES),
                o_bar + 8 * wg,
                lead * live,
                True,
                True,
                IDESC_PV,
                0,
                128,
            )
            # a split with no keys: O stays the zeros just stored (the split's acc * dequant of an empty loop)
            bar_arrive(o_bar + 8 * wg, lead * (1 - live))
            slot = sml + (s * ROWS + row) * 8
            for dd in cutlass.range_constexpr(4):
                st_async(mapa(slot, half * 4 + dd), [m_k, l_k], mapa(ml_bar, half * 4 + dd), valid)
            if cutlass.const_expr(self.debug == 1):
                dml = (
                    g_dbg
                    + Int64(self.dbg["ml"])
                    + Int64((((b * NUM_SPLITS + s) * self.hkv + h) * ROWS + row) * 8)
                )
                st_g_v2(dml, bits(m_k), bits(l_k), valid * (1 - half))
            if cutlass.const_expr(self.debug == 2):
                stamp(g_ts + 32, is_t0)

        if valid == 0:
            # an absent block slot (rank >= 2, slot 4): O = 0, so the class sums read it as an empty split
            tmem_st_zero32(tmem + lane_tm + COLS_PER_BLOCK * wg)
            raw("tcgen05.wait::st.sync.aligned;")
            raw("tcgen05.fence::before_thread_sync;")
            bar_named(1 + wg, 128)
            bar_arrive(o_bar + 8 * wg, lead)

        # ------------------------------------------------ every split's (max, sum) in
        sig = [sigmoid_bf16(x) for x in unpack8(gw)]
        bar_wait(ml_bar, one, True)
        if cutlass.const_expr(self.debug == 2):
            stamp(g_ts + 40, is_t0)
        # warpgroups 0-3: the class sums of 8 of each half's 32 O columns, to the token's owner
        if wg < 4:
            chunk = wg % 4
            cls_ok = prep_q
            xs = []
            mls = []
            for k in cutlass.range_constexpr(NB_MAX):
                sk = rank + CLUSTER * k
                mls.append(
                    ld_s_v2f(sml + (isel(sk, Int32(0), lt(sk, Int32(NUM_SPLITS))) * ROWS + row) * 8)
                )
            m_max = row_max(sml, row, half)
            wks = []
            for k in cutlass.range_constexpr(NB_MAX):
                wks.append(selp(ex2_ftz(sub(mls[k][0], m_max)), Float32(0.0), gt0(mls[k][1])))
            for k in cutlass.range_constexpr(NB_MAX):
                bar_wait(o_bar + 8 * k, cls_ok, False)
            raw("tcgen05.fence::after_thread_sync;")
            if cutlass.const_expr(self.debug == 2):
                stamp(g_ts + 72, is_t0)
            os_ = tmem_ld8x(
                [tmem + lane_tm + COLS_PER_BLOCK * k + 8 * chunk for k in range(NB_MAX)]
            )
            for k in cutlass.range_constexpr(NB_MAX):
                sk = rank + CLUSTER * k
                lk = lt(sk, nblk)
                w_k = wks[k]
                o = os_[k]
                xk = []
                for i in cutlass.range_constexpr(0, 8, 2):
                    okp = pair2("mul.rn.f32x2 z, x, y;", o[i], o[i + 1], dequant, dequant)
                    xk += pair2("mul.rn.f32x2 z, x, y;", w_k, w_k, okp[0], okp[1])
                for i in cutlass.range_constexpr(8):
                    ok = mul(o[i], dequant)
                    if cutlass.const_expr(self.debug == 1):
                        dob = (
                            g_dbg
                            + Int64(self.dbg["o"])
                            + Int64(
                                (
                                    (((b * NUM_SPLITS + sk) * self.hkv + h) * ROWS + row) * D
                                    + 32 * half
                                    + 8 * chunk
                                    + i
                                )
                                * 4
                            )
                        )
                        st_g_b32(
                            dob,
                            bits(selp(ok, Float32(0.0), lk)),
                            lt(sk, Int32(NUM_SPLITS)) * cls_ok,
                        )
                xs.append(xk)
            tsum = class_sums(xs)
            dst = sbase + T_OFF + ((rank * GROUP + row % 8) * D + 32 * half + 8 * chunk) * 4
            owner = row // 8
            st_async(mapa(dst, owner), tsum[0:4], mapa(t_bar, owner), cls_ok)
            st_async(mapa(dst + 16, owner), tsum[4:8], mapa(t_bar, owner), cls_ok)
            if cutlass.const_expr(self.debug == 2):
                stamp(g_ts + 88, is_t0)
        # warpgroup 4: the owner's denominators (warp i: heads 2 i, 2 i + 1), the combine's butterfly
        if wg == 4:
            dens = denominators(
                sml, [rank * GROUP + 2 * warp, rank * GROUP + 2 * warp + 1], lane, self.den_lane
            )
            st_s_b32_pred(sbase + DEN_OFF + (warp * 2) * 4, bits(dens[0]), step)
            st_s_b32_pred(sbase + DEN_OFF + (warp * 2 + 1) * 4, bits(dens[1]), step)
            if cutlass.const_expr(self.debug == 2):
                stamp(g_ts + 80, lt(wt, one) * step)
        raw("tcgen05.fence::before_thread_sync;")
        cute.arch.barrier()
        raw("tcgen05.fence::after_thread_sync;")
        tmem_free(tmem, is_w16)  # every tensor-memory read is done (the barrier above)
        if cutlass.const_expr(self.debug == 2):
            stamp(g_ts + 48, is_t0)

        # ------------------------------------------------ warpgroup 0, the owner (token = rank): finish the rows
        fin = lt(tid, Int32(128))
        bar_wait(t_bar, fin, True)
        if cutlass.const_expr(self.debug == 2):
            stamp(g_ts + 56, is_t0)
        den = as_f32(ld_s_b32(sbase + DEN_OFF + og * 4))
        tv = []
        for cls in cutlass.range_constexpr(CLUSTER):
            tv.append(ld_s_v4f(sbase + T_OFF + ((cls * GROUP + og) * D + od * 4) * 4))
        num = []
        for i in cutlass.range_constexpr(4):
            pw4 = []
            for w2 in cutlass.range_constexpr(4):
                pw4.append(add(tv[2 * w2][i], tv[2 * w2 + 1][i]))
            num.append(add(add(pw4[0], pw4[2]), add(pw4[1], pw4[3])))
        res = gate_rows(num, den, sig, gate_eps)
        oa = (
            g_out
            + (Int64(b * TOKENS + rank) * Int64(out_stride) + Int64((h * GROUP + og) * D + od * 4))
            * 2
        )
        st_g_v2(oa, pack_bf16x2(res[1], res[0]), pack_bf16x2(res[3], res[2]), fin)
        if cutlass.const_expr(self.debug == 2):
            stamp(g_ts + 64, is_t0)


_compiled: dict = {}


def compiled_key(
    device: int,
    enable_pdl: bool,
    kv_heads: int,
    page: int,
    permuted: bool,
    rope: bool,
    debug: int = 0,
    den_lane: int = 0,
) -> tuple:
    return (
        device,
        bool(enable_pdl),
        int(kv_heads),
        int(page),
        bool(permuted),
        bool(rope),
        int(debug),
        int(den_lane),
    )


def _compile(key: tuple):
    fn = _compiled.get(key)
    if fn is not None:
        return fn
    import cuda.bindings.driver as cuda
    from cutlass.cute.runtime import make_ptr

    device, pdl, hkv, page, permuted, rope, debug, den_lane = key
    kernel = SwaClusterBlockKernel(pdl, hkv, page, permuted, rope, debug, den_lane)
    ptrs = [
        make_ptr(dt, 16, cute.AddressSpace.gmem, assumed_align=16)
        for dt in (
            cutlass.BFloat16,
            cutlass.Uint8,
            cutlass.Int32,
            cutlass.Int32,
            cutlass.Float32,
            cutlass.Float32,
            cutlass.Int32,
            cutlass.BFloat16,
            cutlass.BFloat16,
            cutlass.Uint8,
        )
    ]
    with torch.cuda.device(device):
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        fn = cute.compile(
            kernel,
            *ptrs,
            *[cutlass.Int32(1)] * 7,
            *[cutlass.Float32(0.0)] * 4,
            stream,
            options="--enable-tvm-ffi",
        )
    _compiled[key] = fn
    return fn


def swa_cluster_block(
    qkv: torch.Tensor,
    kv_pool: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    kv_scale: torch.Tensor,
    num_kv_heads: int,
    cos_sin: Optional[torch.Tensor],
    positions: Optional[torch.Tensor],
    prep_eps: float,
    q_scale: float,
    qkv_permuted: bool,
    gate: torch.Tensor,
    gate_head_stride: int,
    gate_eps: float,
    enable_pdl: bool,
    debug: Optional[torch.Tensor] = None,
    den_lane: int = 0,
    stamps: bool = False,
) -> torch.Tensor:
    """The block of ``seq_lens.numel()`` sequences of 8 rows each; the caller checked ``fits`` and the multi-request
    op's preconditions. ``debug`` (uint8, ``dbg_layout(G)["bytes"]``) receives the Q tiles and the split partials
    in the split kernel's layout; ``den_lane`` exists for the build's tests."""
    reqs = seq_lens.numel()
    out = torch.empty(
        (qkv.shape[0], GROUP * num_kv_heads * D), dtype=torch.bfloat16, device=qkv.device
    )
    rope = cos_sin is not None
    key = compiled_key(
        qkv.device.index,
        enable_pdl,
        num_kv_heads,
        kv_pool.shape[2],
        qkv_permuted,
        rope,
        0 if debug is None else (2 if stamps else 1),
        den_lane,
    )
    if torch.cuda.is_current_stream_capturing() and key not in _compiled:
        raise RuntimeError(
            "swa_cluster_block compiles on its first call per device and build; make that call "
            "before capturing a CUDA graph"
        )
    fn = _compile(key)
    import cuda.bindings.driver as cuda

    fn(
        qkv.data_ptr(),
        kv_pool.data_ptr(),
        block_tables.data_ptr(),
        seq_lens.data_ptr(),
        kv_scale.data_ptr(),
        (cos_sin if rope else kv_scale).data_ptr(),
        (positions if rope else seq_lens).data_ptr(),
        gate.data_ptr(),
        out.data_ptr(),
        (debug if debug is not None else out).data_ptr(),
        reqs,
        qkv.stride(0),
        block_tables.stride(0),
        block_tables.stride(1),
        gate.stride(0),
        gate_head_stride,
        out.stride(0),
        float(prep_eps),
        float(q_scale),
        float(gate_eps),
        math.log2(math.e) / math.sqrt(D),
        cuda.CUstream(torch.cuda.current_stream(qkv.device).cuda_stream),
    )
    return out


__all__ = [
    "CLUSTER",
    "NUM_SPLITS",
    "dbg_layout",
    "fits",
    "owned_splits",
    "swa_cluster_block",
    "compiled_key",
]
