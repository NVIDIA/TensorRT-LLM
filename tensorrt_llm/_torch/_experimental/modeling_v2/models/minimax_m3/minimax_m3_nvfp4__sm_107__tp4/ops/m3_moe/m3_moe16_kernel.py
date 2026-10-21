# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MiniMax-M3 routed + shared experts for 9..16 decode tokens on one TP rank: ``m3_moe16`` (v2).

Same math as the 1..8-token engine (m3_moe_kernel.py), operation for operation for the routed part (bitwise equal to
TRTLLM-Gen's fused MoE on the fp32 router-GEMV logits); a different schedule for 9..16 rows:

- Front: the hidden states land in smem by bulk copies in K chunks (the idle ring); all 16 warps run the router
  GEMV (m3_router_gemv's per-thread order) and the NVFP4 quantization of B1 from smem; the logits cross CTAs as
  tagged words; top-k one warp per token.
- Ring: FC1 / FC2 weight stages by TMA; their scale words go by 4-byte cp.async straight into the instruction atom
  layout (SFA_CP; no raw scale box, no splicer warp), completing on the stage's full barrier.
- FC1 -> FC2 hand-off (B2IMG): the FC1 epilogue writes each unit's FP4 values and scale byte into a per-group global
  image laid out as the FC2 B tile + its scale atoms; one counter per group (release add per finished unit, every
  counter gains exactly 192 per call); warp 5 loads a completed image with two bulk copies.
- Shared expert on NSH more CTAs (MXFP8 tcgen05), as the 1..8 engine.
Configuration via M3_CONFIG (m_max, num_ctas, stages, nb2_cap, lag, shared, sfa_cp, b2buf, ...).
"""

from __future__ import annotations

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda
from cutlass._mlir.dialects import llvm as _llvm
from cutlass._mlir.extras import types as _T
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.experimental import primitives as prims

_CFG = globals().get("M3_CONFIG") or {}


def _cfg(key: str, default):
    return type(default)(_CFG.get(key, default))


H = 6144  # capacity: fixed: the checkpoint's hidden size
I_TP = 768
TWO_I = 2 * I_TP
E_LOCAL = 128
TOP_K = 4  # capacity: fixed: the checkpoint's top-k
ROUTE_SCALE = 2.0
SUM_EPS = 1e-20
M_MAX = _cfg("m_max", 12)
assert 9 <= M_MAX <= 16, M_MAX
NCTA = _cfg("num_ctas", 128)
FUSED_ROUTER = True
NROUTE = E_LOCAL
NGEMV = (NROUTE + NCTA - 1) // NCTA
assert NGEMV == 1, "m3_moe16: one router expert per CTA (num_ctas >= 128)"
TIMING = bool(_cfg("timing", 0))
HANG_DBG = bool(_cfg("hang_dbg", 0))
LAG = _cfg("lag", 3)
G_CAP = TOP_K * M_MAX
NG_CAP = M_MAX
THREADS = 512  # capacity: fixed: the CTA size
NWARPS = THREADS // 32
GEMV_THREADS = 256
SHARED = bool(_cfg("shared", 0))
LATE_TRIGGER = bool(_cfg("late_trigger", 1))
SFA_CP = bool(
    _cfg("sfa_cp", 1)
)  # A scales by 4-byte cp.async into the atom layout (else raw box + splicer warp)
B2IMG = True
PERM = bool(_cfg("perm", 1))
PERM_MIN_NG = max(
    3, _cfg("perm_min_ng", 5)
)  # (measured: helps at random routing G >= 23, hurts at spec G 13-14)
# a partial last slot group becomes group 1 (the plan must match: op16)
DBG_DELAY = bool(
    _cfg("dbg_delay", 0)
)  # (stress builds) pseudo-random producer / epilogue / release stalls
DBG_NOSTORE = bool(_cfg("dbg_nostore", 0))  # (timing experiment only: wrong results)
EPI_FENCE = bool(
    _cfg("epi_fence", 0)
)  # (rel_warp 0) writer lanes' fence.acq_rel.gpu before the release add
EPI_PFENCE = bool(
    _cfg("epi_pfence", 1)
)  # producer-side fence.proxy.async.global before the release add
REL_WARP = bool(
    _cfg("rel_warp", 1)
)  # warp 8 does the per-group release adds (off the epilogue's path)
NSH = 24 if SHARED else 0
SH_I = 768
SH_KB = H // 32  # capacity: fixed: a kernel tile constant
SH_DKB = SH_I // 32  # capacity: fixed: a kernel tile constant
SH_SPLIT = 2
SH_GT = SH_KB // 4 // SH_SPLIT  # capacity: fixed: a kernel tile constant
SH_STAGES_GU = SH_GT
SH_STAGES = SH_GT + 2 * (SH_I // 128)  # capacity: fixed: pipeline stages
SHX_BAR = 6
SH_PRETILE = False
SH_STAGE_BYTES = 16384 + 512
MMA_M = 128
KT_BYTES = 128
INC_K = 2
NUM_KB = 4
# FC1
N1 = _cfg("n1", 48 if _cfg("m_max", 12) <= 12 else 64)  # B rows QS q + t
N1_LD = 1 << (N1 - 1).bit_length()  # tcgen05.ld .num (a power of two) covering N1 columns
QS = N1 // 4
assert QS >= M_MAX
R1 = 32
C1 = 4
U1 = TWO_I // R1
K1_TILES = H // 256
S1 = K1_TILES // C1  # capacity: fixed: pipeline stages
NUM_BYTES_A = MMA_M * KT_BYTES  # capacity: fixed: a stage's operand bytes
NUM_BYTES_BOX1 = R1 * KT_BYTES
SF1_KA = H // 64
SF1_RB = TWO_I // 128
NUM_BYTES_SFA_RAW = 16 * 512
NUM_WORDS_SFA_RAW = NUM_BYTES_SFA_RAW // 4
NUM_BYTES_SFA = NUM_KB * 512
NUM_WORDS_SFA = NUM_BYTES_SFA // 4
NUM_BYTES_B1 = N1 * KT_BYTES
NUM_SFB1_ATOMS = S1 * NUM_KB
# FC2
R2 = 32
NB2 = H // R2
KT2 = I_TP // 256
SF2_KA = I_TP // 64
SF2_RB = H // 128
N2 = _cfg("n2", (4 * M_MAX + 15) // 16 * 16)  # B rows j M + t
NUM_BYTES_B2 = N2 * KT_BYTES
IMG_B2 = KT2 * NUM_BYTES_B2  # one group's B2 tile image (global and smem)
IMG_SF = KT2 * NUM_KB * 512  # its scale atoms
IMG_BYTES = IMG_B2 + IMG_SF
CNT_PER_CALL = 4 * (TWO_I // R1)  # every group counter gains 192 per call
B1IMG_OFF = None  # set below (after N1): the activation image after the NG_CAP group images
TK_WARPS = [0, 1, 2, 3, 6, 8, 9, 10, 11, 12, 13, 14, 5, 7, 15, 4][
    : _cfg("m_max", 12)
]  # top-k warp of token t (4 last: early0)
B1_WARP = 15 if _cfg("m_max", 12) <= 14 else -1  # (-1: warp 0 after the routing)
NB2_CAP = _cfg("nb2_cap", 2)
B1IMG_OFF = M_MAX * IMG_BYTES
B1IMG_BYTES = (S1 * N1 * 128 + M_MAX * (H // 16) + 15) // 16 * 16
IMG_TOTAL = B1IMG_OFF + B1IMG_BYTES
UNITS_ROW = TWO_I // R1
PARTIAL_EMPTY = -1  # capacity: fixed: a sentinel, not a size
EPI_THREADS = 128
EPI_BAR = 1
ROUTE_BAR = 2
GEMV_BAR = 3
TOPK_BAR = 4
ALL_BAR = 5
POLL_NS = _cfg("poll_ns", 150)
B2_POLL_NS = _cfg("b2_poll_ns", 16)
LG_REPL = _cfg("lg_repl", 64)
LG_STRIDE = M_MAX * E_LOCAL + 32
W_WARP = 4  # capacity: fixed: a warp role index
X_WARP = 5  # B2 image loader; slots + plan  # capacity: fixed: a warp role index
S_WARP = 6  # (SFA_CP = 0) scale splicer  # capacity: fixed: a warp role index
MMA_WARP = 7  # capacity: fixed: a warp role index
EARLY0 = bool(_cfg("early0", 1))
EVICT_FIRST = 0x12F0000000000000
EVICT_LAST = 0x14F0000000000000
_LOG2E = 1.4426950408889634
NB2BUF = _cfg("b2buf", 2)
SPB = 1
# x staging: K chunks of XCK elements, NXB smem buffers (rotating if fewer than the chunks)
XCK = _cfg("xck", 2048)
NXC = H // XCK
XCB = M_MAX * XCK * 2  # bytes of one chunk (all tokens)
STAGE_A = NUM_BYTES_A
STAGE_SF = NUM_BYTES_SFA if SFA_CP else NUM_BYTES_SFA_RAW + NUM_BYTES_SFA
_FIXED_SMEM = (
    S1 * NUM_BYTES_B1
    + NB2BUF * IMG_B2
    + IMG_SF
    + NB2_CAP * M_MAX * TOP_K * R2 * 2
    + SPB * M_MAX * C1 * R1 * 4
    + (G_CAP + 1) * (NG_CAP + 1) * 4
    + 6 * 1024
)
_STAGES_FIT = (227 * 1024 - _FIXED_SMEM - (128 if SHARED else 0)) // (STAGE_A + STAGE_SF)
ACC1_COL = 0
ACC2_COL = 2 * N1
ACC2_STRIDE = 1 << (N2 - 1).bit_length()
SFB1_COL = ACC2_COL + 2 * ACC2_STRIDE
SFB1_WSTEP = 1
SFB2_COL = (
    SFB1_COL + NUM_SFB1_ATOMS * 4
)  # one TMEM slot (loaded after the previous group's FC2 MMAs)
SFA1_COL = SFB2_COL + KT2 * NUM_KB * 4
STAGES = min(_cfg("stages", 8), _STAGES_FIT, (512 - SFA1_COL) // 16)
TMEM_COLS = 512
assert STAGES >= 3, (STAGES, _STAGES_FIT, SFA1_COL)
# smem: [ring A | ring SF | B2 bufs | B2 SF staging | B1]; x chunks alias [0, _OFF_B1) during the front
_OFF_SFA = STAGES * NUM_BYTES_A
_OFF_B2 = (_OFF_SFA + STAGES * STAGE_SF + 1023) // 1024 * 1024
_OFF_SFB2 = _OFF_B2 + NB2BUF * IMG_B2
_OFF_B1 = (_OFF_SFB2 + IMG_SF + 1023) // 1024 * 1024
BIG = _OFF_B1 + S1 * NUM_BYTES_B1
NXB = min(NXC, _OFF_B1 // XCB)
assert NXB >= 2, (NXB, _OFF_B1, XCB)
# shared CTA layout in BIG: one pass over all M tokens; B rows SHN q + t (gate_up, 2 K slices), t (down)
SHN = 16  # capacity: fixed: a kernel tile constant
SH_NPASS = 1
SH_HINT = EVICT_FIRST
SH_PF_POST = bool(
    _cfg("sh_pf_post", 1)
)  # L2 prefetch of the shared weights after the grid dependency (else before)
_SB_A = 0
_SH_FIX = (
    SH_GT * 2 * SHN * 128
    + 6 * SHN * 128
    + M_MAX * SH_KB
    + M_MAX * SH_DKB
    + 64 * SHN * 4
    + 2 * 32 * SHN * 4
    + 2048
)
SH_NST = min(8, (BIG - _SH_FIX) // (NUM_BYTES_A + 4 * 512 + 512)) if SHARED else 1
_SB_RAW = SH_NST * NUM_BYTES_A
_SB_SPL = _SB_RAW + SH_NST * 2048
_SB_BX = (_SB_SPL + SH_NST * 512 + 1023) // 1024 * 1024
_SB_BH = _SB_BX + SH_GT * 2 * SHN * 128
_SB_XC = _SB_BH + 6 * SHN * 128
_SB_HC = _SB_XC + M_MAX * SH_KB
_SB_XCH = (_SB_HC + M_MAX * SH_DKB + 15) // 16 * 16
_SB_GU = _SB_XCH + 64 * SHN * 4
_SB_END = _SB_GU + 2 * 32 * SHN * 4
SH_ACC_GU = 0
SH_ACC_DN = 2 * SHN
SH_SFA = SH_ACC_DN + 2 * SHN
SH_SFB_GU = SH_SFA + 4 * 8
SH_SFB_DN = SH_SFB_GU + 4 * SH_GT  # capacity: fixed: a kernel tile constant
SH_TMEM = 256  # capacity: fixed: TMEM columns
if SHARED:
    assert SH_STAGES - SH_NST <= 64, SH_STAGES
    assert SH_NST >= 3 and _SB_END <= BIG and SH_SFB_DN + 24 <= SH_TMEM, (SH_NST, _SB_END, BIG)
TD_STRIDE = 64
PFD = _cfg("pfd", 0)  # L2 prefetch distance in ring stages (0: off)
PF_WARP = 12  # capacity: fixed: a warp role index
TICKET = bool(_cfg("ticket", 0))  # CTA roles by entry order
# priority tokens: token 0 and PT1 (M / 2: the second request's first row at spec c2) get their logits published first
NPT = _cfg("npt", 0)
PT1 = M_MAX // 2
PF_PRIO = bool(
    _cfg("pf_prio", 0)
)  # L2 prefetch of the priority tokens' experts (this CTA's 1/NCTA of each)
PF_W1 = TWO_I * (H // 2) // NCTA
PF_S1 = TWO_I * (H // 16) // NCTA
PF_W2 = H * (I_TP // 2) // NCTA
PF_S2 = H * (I_TP // 16) // NCTA
assert PF_W1 % 16 == 0 and PF_S1 % 16 == 0 and PF_W2 % 16 == 0 and PF_S2 % 16 == 0


# =============================================================================
# Inline-PTX helpers.
# =============================================================================
def _asm(ret, args, ptx, cons, side=False, loc=None, ip=None):
    return _llvm.inline_asm(
        ret,
        [a.ir_value(loc=loc, ip=ip) for a in args],
        ptx,
        cons,
        has_side_effects=side,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


# ---- all-reduce push (the MoE boundary's multicast exchange, set A):
# arp = int64 [enable, mc, flags, rank, world, maxt]
@dsl_user_op
def _ld_volatile_s32(addr, *, loc=None, ip=None):
    return cutlass.Int32(
        _llvm.inline_asm(
            _T.i32(), [cutlass.Int64(addr).ir_value(loc=loc, ip=ip)], "ld.volatile.global.s32 $0, [$1];", "=r,l",
            has_side_effects=True, is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip


@dsl_user_op
def _st_global_b32_if(addr, value, pred, *, loc=None, ip=None):
    """st.global.b32 when pred != 0."""
    _llvm.inline_asm(
        None, [cutlass.Int64(addr).ir_value(loc=loc, ip=ip), cutlass.Int32(value).ir_value(loc=loc, ip=ip),
               cutlass.Int32(pred).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, $2, 0;\n\t@p st.global.b32 [$0], $1;\n\t}", "l,r,r",
        has_side_effects=True, is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )  # fmt: skip


@dsl_user_op
def _bf16x2_word(lo, hi, *, loc=None, ip=None):
    """bf16(lo) | bf16(hi) << 16 (RN), with -0.0 halves turned into +0.0 (0x80000000 marks an empty exchange word)."""
    return cutlass.Int32(
        _llvm.inline_asm(
            _T.i32(), [cutlass.Float32(hi).ir_value(loc=loc, ip=ip), cutlass.Float32(lo).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b32 w, l, h;\n\t.reg .pred pl, ph;\n\tcvt.rn.bf16x2.f32 w, $1, $2;\n\t"
            "and.b32 l, w, 0xFFFF;\n\tsetp.eq.u32 pl, l, 0x8000;\n\t@pl and.b32 w, w, 0xFFFF0000;\n\t"
            "shr.u32 h, w, 16;\n\tsetp.eq.u32 ph, h, 0x8000;\n\t@ph and.b32 w, w, 0x0000FFFF;\n\t"
            "mov.b32 $0, w;\n\t}", "=r,f,f",
            has_side_effects=False, is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip


def _ar_push(arp, v, low, t, cl):
    """Push bf16 channel pair (cl, cl + 1) of token t (v = channel cl's bf16 value on this lane, its neighbour's on
    lane ^ 8) into this rank's set-A slot of the exchange; the slot half is the all-reduce CTA (t, cl // 768)'s call
    parity. Only lanes with ``low`` store; every lane of the warp must call it (shuffle)."""
    hi = cute.arch.shuffle_sync_bfly(v, 8)
    pred = cutlass.select_(low, cutlass.Int32(1), cutlass.Int32(0))
    cs = cutlass.select_(low, cl, cutlass.Int32(0))
    world = arp.load(idx=4)
    parity = cutlass.Int64(
        _ld_volatile_s32(
            arp.load(idx=2)
            + cutlass.Int64(4) * cutlass.Int64(t * cutlass.Int32(8) + cs // cutlass.Int32(H // 8))
        )
        & cutlass.Int32(1)
    )
    off = (
        parity * arp.load(idx=5) * world * cutlass.Int64(H // 2)
        + (cutlass.Int64(t) * world + arp.load(idx=3)) * (cutlass.Int64(H // 2))
        + cutlass.Int64(cs // cutlass.Int32(2))
    )
    _st_global_b32_if(arp.load(idx=1) + off * cutlass.Int64(4), _bf16x2_word(v, hi), pred)


@dsl_user_op
def _bar_arrive(bar_id: int, threads: int, *, loc=None, ip=None):
    """bar.arrive: signal a named barrier without waiting (its memory ordering is bar.sync's)."""
    _llvm.inline_asm(
        None,
        [],
        f"bar.arrive {bar_id}, {threads};",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_relaxed_u32(addr, val, *, loc=None, ip=None):
    _asm(
        None,
        [cutlass.Int64(addr), cutlass.Int32(val)],
        "st.relaxed.gpu.global.u32 [$0], $1;",
        "l,r",
        True,
        loc,
        ip,
    )


@dsl_user_op
def _ld_relaxed_u32(addr, *, loc=None, ip=None):
    return cutlass.Int32(
        _asm(
            _T.i32(),
            [cutlass.Int64(addr)],
            "ld.relaxed.gpu.global.u32 $0, [$1];",
            "=r,l",
            True,
            loc,
            ip,
        )
    )


@dsl_user_op
def _st_relaxed_u64(addr, val, *, loc=None, ip=None):
    _asm(
        None,
        [cutlass.Int64(addr), cutlass.Int64(val)],
        "st.relaxed.gpu.global.u64 [$0], $1;",
        "l,l",
        True,
        loc,
        ip,
    )


@dsl_user_op
def _ld_relaxed_u64(addr, *, loc=None, ip=None):
    return cutlass.Int64(
        _asm(
            _T.i64(),
            [cutlass.Int64(addr)],
            "ld.relaxed.gpu.global.u64 $0, [$1];",
            "=l,l",
            True,
            loc,
            ip,
        )
    )


@dsl_user_op
def _nanosleep(ns: int, *, loc=None, ip=None):
    _llvm.inline_asm(
        None,
        [],
        f"nanosleep.u32 {ns};",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _gtime(*, loc=None, ip=None):
    return cutlass.Int64(
        _llvm.inline_asm(
            _T.i64(),
            [],
            "mov.u64 $0, %globaltimer;",
            "=l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=_llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _smid(*, loc=None, ip=None):
    return cutlass.Int32(
        _llvm.inline_asm(
            _T.i32(),
            [],
            "mov.u32 $0, %smid;",
            "=r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=_llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def _stamp(tdbg: cutlass.Array, bx, i: cutlass.Constexpr):
    if cutlass.const_expr(TIMING):
        tdbg.store(_gtime(), idx=bx * cutlass.Int32(TD_STRIDE) + cutlass.Int32(i))


@dsl_user_op
def _rcp(a, *, loc=None, ip=None):
    """rcp.approx.ftz.f32 (the reciprocal TRT-LLM's FP4 quantization uses)."""
    return cutlass.Float32(
        _asm(_T.f32(), [cutlass.Float32(a)], "rcp.approx.ftz.f32 $0, $1;", "=f,f", False, loc, ip)
    )


@dsl_user_op
def _div_rn(a, b, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            _T.f32(),
            [cutlass.Float32(a), cutlass.Float32(b)],
            "div.rn.f32 $0, $1, $2;",
            "=f,f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _fma_rn(a, b, c, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            _T.f32(),
            [cutlass.Float32(a), cutlass.Float32(b), cutlass.Float32(c)],
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _add_rn(a, b, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            _T.f32(),
            [cutlass.Float32(a), cutlass.Float32(b)],
            "add.rn.f32 $0, $1, $2;",
            "=f,f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _sub_rn(a, b, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            _T.f32(),
            [cutlass.Float32(a), cutlass.Float32(b)],
            "sub.rn.f32 $0, $1, $2;",
            "=f,f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _mul_rn(a, b, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            _T.f32(),
            [cutlass.Float32(a), cutlass.Float32(b)],
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _e4m3_bits(x, *, loc=None, ip=None):
    """e4m3 (round to nearest even, saturating) of x as an int 0..255."""
    return cutlass.Int32(
        _asm(
            _T.i32(),
            [cutlass.Float32(x)],
            "{\n.reg .b16 t;\ncvt.rn.satfinite.e4m3x2.f32 t, 0f00000000, $1;\ncvt.u32.u16 $0, t;\n}",
            "=r,f",
            False,
            loc,
            ip,
        )
    ) & cutlass.Int32(0xFF)


@dsl_user_op
def _e4m3_to_f32(b, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            _T.f32(),
            [cutlass.Int32(b)],
            "{\n.reg .b16 t, l, u;\n.reg .b32 h;\ncvt.u16.u32 t, $1;\ncvt.rn.f16x2.e4m3x2 h, t;\nmov.b32 {l, u}, h;\n"
            "cvt.f32.f16 $0, l;\n}",
            "=f,r",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _e2m1x2(hi, lo, *, loc=None, ip=None):
    """One byte of two E2M1 values (lo in the low nibble), round to nearest even, saturating."""
    return cutlass.Int32(
        _asm(
            _T.i32(),
            [cutlass.Float32(hi), cutlass.Float32(lo)],
            "{\n.reg .b8 t;\ncvt.rn.satfinite.e2m1x2.f32 t, $1, $2;\ncvt.u32.u8 $0, t;\n}",
            "=r,f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _ex2(x, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(_T.f32(), [cutlass.Float32(x)], "ex2.approx.ftz.f32 $0, $1;", "=f,f", False, loc, ip)
    )


@dsl_user_op
def _bf16_round(x, *, loc=None, ip=None):
    """x rounded to bf16 (nearest even), as f32."""
    return cutlass.Float32(
        _asm(
            _T.f32(),
            [cutlass.Float32(x)],
            "{\n.reg .b16 t;\ncvt.rn.bf16.f32 t, $1;\nmov.b32 $0, {0, t};\n}",
            "=f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _redux_max_u32(v, *, loc=None, ip=None):
    return cutlass.Int32(
        _asm(
            _T.i32(),
            [cutlass.Int32(v)],
            "redux.sync.max.u32 $0, $1, 0xffffffff;",
            "=r,r",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _bf16_bits(x, *, loc=None, ip=None):
    """bf16 (nearest even) bits of x in the low 16 bits."""
    return cutlass.Int32(
        _asm(
            _T.i32(),
            [cutlass.Float32(x)],
            "{\n.reg .b16 t;\ncvt.rn.bf16.f32 t, $1;\ncvt.u32.u16 $0, t;\n}",
            "=r,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _ue8m0_rp(x, *, loc=None, ip=None):
    """UE8M0 code of x rounded toward +inf, saturating (__nv_cvt_float_to_e8m0(x, SATFINITE, RoundPosInf))."""
    return cutlass.Int32(
        _asm(
            _T.i32(),
            [cutlass.Float32(x)],
            "{\n.reg .b16 t;\ncvt.rp.satfinite.ue8m0x2.f32 t, $1, $1;\ncvt.u32.u16 $0, t;\n}",
            "=r,f",
            False,
            loc,
            ip,
        )
    ) & cutlass.Int32(0xFF)


@dsl_user_op
def _e4m3x2(hi, lo, *, loc=None, ip=None):
    """Two e4m3 (nearest even, saturating) in 16 bits, lo in the low byte."""
    return cutlass.Int32(
        _asm(
            _T.i32(),
            [cutlass.Float32(hi), cutlass.Float32(lo)],
            "{\n.reg .b16 t;\ncvt.rn.satfinite.e4m3x2.f32 t, $1, $2;\ncvt.u32.u16 $0, t;\n}",
            "=r,f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _min_xorsign_abs(a, b, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            _T.f32(),
            [cutlass.Float32(a), cutlass.Float32(b)],
            "min.xorsign.abs.f32 $0, $1, $2;",
            "=f,f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _div_full(a, b, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            _T.f32(),
            [cutlass.Float32(a), cutlass.Float32(b)],
            "div.full.f32 $0, $1, $2;",
            "=f,f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _ex2_full(x, *, loc=None, ip=None):
    """ex2.approx.f32 (no flush to zero), Triton's tl.exp."""
    return cutlass.Float32(
        _asm(_T.f32(), [cutlass.Float32(x)], "ex2.approx.f32 $0, $1;", "=f,f", False, loc, ip)
    )


@dsl_user_op
def _l2_prefetch(addr, nbytes, *, loc=None, ip=None):
    """cp.async.bulk.prefetch.L2 of nbytes (a multiple of 16) at global address addr (16-byte aligned)."""
    _llvm.inline_asm(
        None,
        [
            cutlass.Int64(addr).ir_value(loc=loc, ip=ip),
            cutlass.Int32(nbytes).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.bulk.prefetch.L2.global [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _mx_osc(amax):
    """mxfp8 quantization of a block with absolute max amax: (UE8M0 code, the element multiplier)."""
    code = _ue8m0_rp(_mul_rn(amax, _rcp(cutlass.Float32(448.0))))
    cf = cutlass.Int32(
        cutlass.select_(
            code == cutlass.Int32(255),
            cutlass.Int32(0x7FFFFFFF),
            cutlass.select_(
                code == cutlass.Int32(0), cutlass.Int32(0x00400000), code << cutlass.Int32(23)
            ),
        )
    ).bitcast(cutlass.Float32)
    return code, cutlass.select_(amax != cutlass.Float32(0.0), _rcp(cf), cutlass.Float32(0.0))


def _mx_words(v, osc):
    """e4m3 words of len(v) (a multiple of 4) values times osc, in element order."""
    words = []
    for t in range(len(v) // 4):
        lo = _e4m3x2(_mul_rn(v[4 * t + 1], osc), _mul_rn(v[4 * t], osc))
        hi = _e4m3x2(_mul_rn(v[4 * t + 3], osc), _mul_rn(v[4 * t + 2], osc))
        words.append(lo | (hi << cutlass.Int32(16)))
    return words


def _twiddle(score):
    """Float bits mapped so that unsigned order is float order (cub TwiddleIn)."""
    bits = (
        score.bitcast(cutlass.Int32)
        if hasattr(score, "bitcast")
        else cutlass.Float32(score).bitcast(cutlass.Int32)
    )
    return cutlass.select_(bits < cutlass.Int32(0), ~bits, bits | cutlass.Int32(-2147483648))


@cute.jit
def _wait(bar: cutlass.Array, parity):
    while not cute.arch.mbarrier_try_wait(bar.data_ptr(), parity):
        pass
    cute.arch.sync_warp()


def _tptr(base_row, col, dtype=cutlass.Int32):
    return cutlass.inttoptr((base_row << 16) | col, 6, dtype)


@dsl_user_op
def _match_any(value, *, loc=None, ip=None):
    """The mask of the warp's lanes holding the same 32-bit value."""
    return cutlass.Int32(
        _asm(
            _T.i32(),
            [cutlass.Int32(value)],
            "match.any.sync.b32 $0, $1, 0xffffffff;",
            "=r,r",
            False,
            loc,
            ip,
        )
    )


@cute.jit
def _waitd(bar: cutlass.Array, parity, tdbg: cutlass.Array, site: cutlass.Constexpr):
    """_wait, or (HANG_DBG) give up after ~2^22 polls and record the site in tdbg[bx, 24 + warp]."""
    if cutlass.const_expr(HANG_DBG):
        n = cutlass.Int32(0)
        while (not cute.arch.mbarrier_try_wait(bar.data_ptr(), parity)) & (
            n < cutlass.Int32(1 << 22)
        ):
            n = n + cutlass.Int32(1)
        if n >= cutlass.Int32(1 << 22):
            bx, _, _ = cute.arch.block_idx()
            w = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            tdbg.store(
                cutlass.Int64(site), idx=bx * cutlass.Int32(TD_STRIDE) + cutlass.Int32(24) + w
            )
    else:
        while not cute.arch.mbarrier_try_wait(bar.data_ptr(), parity):
            pass
    cute.arch.sync_warp()


@cute.jit
def _adv(stage, phase):
    """The next ring position's (stage, phase)."""
    stage = stage + cutlass.Int32(1)
    wrap = stage == cutlass.Int32(STAGES)
    return cutlass.select_(wrap, cutlass.Int32(0), stage), cutlass.select_(
        wrap, phase ^ cutlass.Int32(1), phase
    )


@cute.jit
def _seg(i0, i1, k):
    """Segment k of the FC1 item range [i0, i1) (items = unit-major stages): (unit, first stage, end stage). Order:
    the last unit (its head, the part a split unit's next CTA does not cover), then the first unit, then the rest."""
    uf = i0 // cutlass.Int32(S1)
    ul = (i1 - cutlass.Int32(1)) // cutlass.Int32(S1)
    unit = uf + k - cutlass.Int32(1)
    unit = cutlass.select_(k == cutlass.Int32(0), ul, unit)
    unit = cutlass.select_(ul == uf, uf, unit)
    j0 = i0 - unit * cutlass.Int32(S1)
    j0 = cutlass.select_(j0 < cutlass.Int32(0), cutlass.Int32(0), j0)
    j1 = i1 - unit * cutlass.Int32(S1)
    j1 = cutlass.select_(j1 > cutlass.Int32(S1), cutlass.Int32(S1), j1)
    return unit, j0, j1


@cute.jit
def _nsegs(i0, i1):
    n = (i1 - cutlass.Int32(1)) // cutlass.Int32(S1) - i0 // cutlass.Int32(S1) + cutlass.Int32(1)
    return cutlass.select_(i1 > i0, n, cutlass.Int32(0))


@cute.jit
def _sfb1_to_tmem(warp, lane, base_row, base_col, src: cutlass.Array, b1_ready: cutlass.Array):
    """B1 scales to TMEM: instruction (j, kb)'s column holds, at lane n = QS q + t, token t's 4 scales of k-tile
    4 j + q, K 64 kb .. (src: int32 words [M, 96] of the token scales); each warp writes its sub-partition."""
    # B rows n >= 32 (N1 = 64) go to the atom's next 32-bit column (row n: lane n % 32, column n // 32)
    for w in cutlass.range_constexpr((N1 + 31) // 32):
        n = lane + cutlass.Int32(32 * w)
        q = n // cutlass.Int32(QS)
        tl = n - q * cutlass.Int32(QS)
        sf_ok = (n < cutlass.Int32(N1)) & (tl < cutlass.Int32(M_MAX))
        sfv = []
        for a in cutlass.range_constexpr(NUM_SFB1_ATOMS):
            j = a // NUM_KB
            kb = a % NUM_KB
            wsf = src.load(
                idx=cutlass.select_(
                    sf_ok,
                    tl * cutlass.Int32(H // 64)
                    + (cutlass.Int32(4 * j) + q) * cutlass.Int32(4)
                    + cutlass.Int32(kb),
                    cutlass.Int32(0),
                )
            )
            sfv.append(cutlass.select_(sf_ok, cutlass.Int32(wsf), cutlass.Int32(0)))
        for a in cutlass.range_constexpr(NUM_SFB1_ATOMS):
            prims.tcgen05_st(
                "32x32b",
                _tptr(
                    base_row + warp * cutlass.Int32(32),
                    base_col + SFB1_COL + a * 4 + w * SFB1_WSTEP,
                ),
                sfv[a],
            )
    prims.tcgen05_wait(prims.Tcgen05Wait.STORE)
    prims.fence_proxy("async_shared", space=prims.SharedSpace.shared_cta)
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    prims.mbarrier_arrive(b1_ready)


@cute.jit
def _b1_addr(t, kb16):
    """Word index in the resident B tiles of the 8 FP4 bytes of token t's 16-value block kb16."""
    kt = kb16 >> cutlass.Int32(4)
    j = kt >> cutlass.Int32(2)
    n = (kt & cutlass.Int32(3)) * cutlass.Int32(QS) + t
    cc = (kb16 >> cutlass.Int32(1)) & cutlass.Int32(7)
    return (
        j * cutlass.Int32(NUM_BYTES_B1)
        + n * cutlass.Int32(KT_BYTES)
        + (cc ^ (n & cutlass.Int32(7))) * cutlass.Int32(16)
        + (kb16 & cutlass.Int32(1)) * cutlass.Int32(8)
    ) // cutlass.Int32(4)


@cute.jit
def _swiglu_quant(xv, gate_raw, is_up, a, c, alpha, beta, lim):
    """TRTLLM-Gen's FC1 epilogue (the swiGlu Bmm E2m1 output), operation for operation: lane value xv is the up row,
    gate_raw its gate row (fp32 sums). Returns (q, the scale's e4m3 bits) of the 16-row NVFP4 block (lanes r, r ^ 16
    and r ^ 8 cover it; amax over the warp's up lanes of a half). a = fc31_alpha, c = fc31_scale_c."""
    neg_max = cutlass.Float32(-3.40282346638528859812e38)
    gc = cutlass.select_(gate_raw > lim, lim, gate_raw)
    gc = cutlass.select_(gate_raw < neg_max, neg_max, gc)
    uc = cutlass.select_(xv > lim, lim, xv)
    uc = cutlass.select_(xv < -lim, -lim, uc)
    c1 = _mul_rn(a, _mul_rn(alpha, cutlass.Float32(1.4426950216293334961)))
    sgm = _rcp(_add_rn(_ex2(-_mul_rn(gc, c1)), cutlass.Float32(1.0)))
    gs = _mul_rn(gc, sgm)
    h = _mul_rn(_fma_rn(uc, a, _mul_rn(a, beta)), gs)
    h = cutlass.select_(is_up, h, cutlass.Float32(0.0))
    amax = cute.arch.fmax(_mul_rn(cute.math.abs(h), c), cutlass.Float32(0.0))
    for off in (16, 8, 4, 2, 1):
        amax = cute.arch.fmax(amax, cute.arch.shuffle_sync_bfly(amax, off))
    sfb = _e4m3_bits(_mul_rn(amax, cutlass.Float32(0.16666667163372039795)))
    sff = _e4m3_to_f32(sfb)
    osc = cutlass.select_(sff != cutlass.Float32(0.0), _rcp(sff), cutlass.Float32(0.0))
    return _mul_rn(_mul_rn(h, osc), c), sfb


@cute.jit
def _plan_range(s_plan: cutlass.Array, p):
    w = s_plan.load(idx=p)
    return w & cutlass.Int32(0xFFFF), w >> cutlass.Int32(16)


@cute.jit
def _sh_issue(
    lane,
    sid,
    g,
    s_big: cutlass.Array,
    sh_full: cutlass.Array,
    tma_gu,
    tma_dn,
    gusf: cutlass.Array,
    dnsf: cutlass.Array,
    gpos,
):
    """Stage g of the shared CTA's stream into ring slot g % SH_NST (warp 0). g < SH_GT: gate_up k-tile g of both
    slices: quadrant Q = 2 q + u (q slice, u 0 gate / 1 up) = rows 32 sid .. (+ 768 for up) at k-tile SH_GT q + g, its
    raw scale atom at raw slot Q. Else down tile j = (g - SH_GT) / 6 (rows 256 sid + 128 j ..), k-tile
    (g - SH_GT) % 6."""
    st = gpos % cutlass.Int32(SH_NST)  # ring position (stage g of this pass)
    bar = sh_full.subview(st)
    if cutlass.const_expr(SH_PRETILE):
        if lane == 0:
            src = (sid * cutlass.Int32(SH_STAGES) + g) * cutlass.Int32(SH_STAGE_BYTES)
            prims.mbarrier_arrive_expect_tx(bar, SH_STAGE_BYTES)
            prims.cp_async_bulk_shared_cluster_global(
                s_big.subview(st * cutlass.Int32(NUM_BYTES_A)), gusf.subview(src), bar, 16384
            )
            prims.cp_async_bulk_shared_cluster_global(
                s_big.subview(cutlass.Int32(_SB_SPL) + st * cutlass.Int32(512)),
                gusf.subview(src + cutlass.Int32(16384)),
                bar,
                512,
            )
        cute.arch.sync_warp()
    else:
        if g < cutlass.Int32(SH_GT):
            if lane == 0:
                prims.mbarrier_arrive_expect_tx(bar, NUM_BYTES_A + 4 * 512)
            cute.arch.sync_warp()
            if lane < 4:
                q = lane >> cutlass.Int32(1)
                u = lane & cutlass.Int32(1)
                row0 = sid * cutlass.Int32(32) + u * cutlass.Int32(SH_I)
                kt = q * cutlass.Int32(SH_GT) + g
                prims.cp_async_bulk_tensor_shared_cta_global(
                    s_big.subview(st * cutlass.Int32(NUM_BYTES_A) + lane * cutlass.Int32(4096)),
                    tma_gu,
                    (kt * cutlass.Int32(128), row0),
                    bar,
                    l2_cache_hint=SH_HINT,
                )
                prims.cp_async_bulk_shared_cluster_global(
                    s_big.subview(
                        cutlass.Int32(_SB_RAW)
                        + st * cutlass.Int32(2048)
                        + lane * cutlass.Int32(512)
                    ),
                    gusf.subview(
                        ((row0 >> cutlass.Int32(7)) * cutlass.Int32(SH_KB // 4) + kt)
                        * cutlass.Int32(512)
                    ),
                    bar,
                    512,
                )
        else:
            gd = g - cutlass.Int32(SH_GT)
            jt = gd // cutlass.Int32(6)
            kt = gd - jt * cutlass.Int32(6)
            row0 = sid * cutlass.Int32(256) + jt * cutlass.Int32(128)
            if lane == 0:
                prims.mbarrier_arrive_expect_tx(bar, NUM_BYTES_A + 512)
                prims.cp_async_bulk_tensor_shared_cta_global(
                    s_big.subview(st * cutlass.Int32(NUM_BYTES_A)),
                    tma_dn,
                    (kt * cutlass.Int32(128), row0),
                    bar,
                    l2_cache_hint=SH_HINT,
                )
                prims.cp_async_bulk_shared_cluster_global(
                    s_big.subview(cutlass.Int32(_SB_RAW) + st * cutlass.Int32(2048)),
                    dnsf.subview(
                        ((row0 >> cutlass.Int32(7)) * cutlass.Int32(SH_DKB // 4) + kt)
                        * cutlass.Int32(512)
                    ),
                    bar,
                    512,
                )
            cute.arch.sync_warp()


@cute.jit
def _shared_cta(
    tidx,
    warp,
    lane,
    sid,
    s_big: cutlass.Array,
    sh_bar: cutlass.Array,
    epochs: cutlass.Array,
    tdbg: cutlass.Array,
    bx,
    xb32: cutlass.Array,
    tma_gu,
    tma_dn,
    gusf: cutlass.Array,
    dnsf: cutlass.Array,
    hx: cutlass.Array,
    dnx: cutlass.Array,
    arp: cutlass.Array,
    tmem_ptr_i32: cutlass.Array,
    gu8: cutlass.Array,
    dn8: cutlass.Array,
):
    """Shared CTA sid (see SHARED): tcgen05 MXF8 (E4M3 x E4M3, UE8M0 per 32 K), m3_mx_gemv's tile order. gate_up: one
    128-row MMA tile = 2 K slices x (32 gate + 32 up rows), B rows 8 q + t = token t's slice-q k-tile; down: 2 tiles
    of 128 rows, full K. Warps: 0 weights (TMA), 1 TMEM + scale splice + MMA, 4-7 epilogue; 2.. activation quant."""
    big32 = cutlass.Array(s_big.data_ptr(0), shape=(BIG // 4,), dtype=cutlass.Int32, alignment=16)
    bigf = cutlass.Array(s_big.data_ptr(0), shape=(BIG // 4,), dtype=cutlass.Float32, alignment=16)
    big8 = cutlass.Array(s_big.data_ptr(0), shape=(BIG,), dtype=cutlass.Int8, alignment=16)
    sh_full = sh_bar  # [0, SH_NST): stage full
    sh_empty = cutlass.Array(sh_bar.data_ptr(8), shape=(8,), dtype=cutlass.Int64, alignment=8)
    b_gu_rdy = sh_bar.subview(16)
    b_dn_rdy = sh_bar.subview(17)
    acc_gu = sh_bar.subview(18)
    acc_dn = sh_bar.subview(19)
    if warp == 0:
        prims.prefetch_tensormap(tma_gu)
        prims.prefetch_tensormap(tma_dn)
        if lane < SH_NST:
            prims.mbarrier_init(sh_full.subview(lane), 1)
            prims.mbarrier_init(
                sh_empty.subview(lane), 33
            )  # the MMAs' commit + each splicer lane's reads done
        if lane == 0:
            prims.mbarrier_init(b_gu_rdy, 1)
            prims.mbarrier_init(b_dn_rdy, 1)
            prims.mbarrier_init(acc_gu, 1)
            prims.mbarrier_init(acc_dn, 1)
    if warp == 1:
        prims.tcgen05_alloc(tmem_ptr_i32, SH_TMEM)
        prims.tcgen05_relinquish_alloc_permit()
    prims.fence_mbarrier_init()
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    cute.arch.sync_warp()
    prims.barrier_cta_sync(0)
    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
    tmem_raw = tmem_ptr_i32.load()
    base_col = tmem_raw & 0xFFFF
    base_row = tmem_raw >> 16
    if warp == 0:
        for g in cutlass.range_constexpr(SH_NST):  # layer constants: before the grid dependency
            _sh_issue(
                lane,
                sid,
                cutlass.Int32(g),
                s_big,
                sh_full,
                tma_gu,
                tma_dn,
                gusf,
                dnsf,
                cutlass.Int32(g),
            )
        # the rest of this CTA's weights into L2 (64 gate / up rows of 6 KB, 256 down rows of 768 B)
        if cutlass.const_expr(SH_PRETILE):
            # this CTA's stages beyond the ring, 4 per lane
            for r in cutlass.range_constexpr(2):
                pg = cutlass.Int32(SH_NST) + lane * cutlass.Int32(2) + cutlass.Int32(r)
                if pg < cutlass.Int32(SH_STAGES):
                    _l2_prefetch(
                        gusf.subview(
                            (sid * cutlass.Int32(SH_STAGES) + pg) * cutlass.Int32(SH_STAGE_BYTES)
                        )
                        .data_ptr()
                        .toint(),
                        SH_STAGE_BYTES,
                    )
        elif cutlass.const_expr(not SH_PF_POST):
            for r in cutlass.range_constexpr(2):
                prw = sid * cutlass.Int32(32) + cutlass.Int32(r * SH_I) + lane
                _l2_prefetch(gu8.subview(prw * cutlass.Int32(H)).data_ptr().toint(), H)
            if lane == 0:
                _l2_prefetch(
                    dn8.subview(sid * cutlass.Int32(256 * SH_I)).data_ptr().toint(), 256 * SH_I
                )
    cute.arch.griddepcontrol_wait()
    if cutlass.const_expr(SH_PF_POST):
        if warp == 0:  # after the grid dependency: DRAM is idle during the routed front
            for r in cutlass.range_constexpr(2):
                prw = sid * cutlass.Int32(32) + cutlass.Int32(r * SH_I) + lane
                _l2_prefetch(gu8.subview(prw * cutlass.Int32(H)).data_ptr().toint(), H)
            if lane == 0:
                _l2_prefetch(
                    dn8.subview(sid * cutlass.Int32(256 * SH_I)).data_ptr().toint(), 256 * SH_I
                )
    if cutlass.const_expr(not LATE_TRIGGER):
        if arp.load(idx=0) == cutlass.Int64(0):
            cute.arch.griddepcontrol_launch_dependents()
    if tidx == 0:
        _stamp(tdbg, bx, 1)
    ep = epochs.load(idx=bx)
    tag16 = (ep & cutlass.Int32(0x7FFF)) | cutlass.Int32(0x8000)
    NQ = THREADS - 64  # activation quant threads (warps 2 ..)
    for ps in cutlass.range_constexpr(SH_NPASS):
        TT0 = 0  # one pass: B rows SHN q + t (gate_up), t (down)
        MS = M_MAX
        GB = ps * SH_STAGES  # ring positions before this pass
        PAR = ps & 1  # parity of the per-pass barriers (B ready, accumulators)
        if warp == 0:
            for g in cutlass.range(SH_NST if ps == 0 else 0, SH_STAGES, unroll=1):
                gg = g + cutlass.Int32(GB)  # ring position over the passes
                _wait(
                    sh_empty.subview(gg % cutlass.Int32(SH_NST)),
                    ((gg // cutlass.Int32(SH_NST)) - cutlass.Int32(1)) & cutlass.Int32(1),
                )
                _sh_issue(lane, sid, g, s_big, sh_full, tma_gu, tma_dn, gusf, dnsf, gg)
        elif warp == 1:
            # ---- MMA: gate_up (scale atoms spliced to the tile's row order), then down
            idesc_gu = prims.Tcgen05MxInstrDesc.build(
                a_dtype=cutlass.Float8E4M3FN,
                b_dtype=cutlass.Float8E4M3FN,
                scale_format=1,
                n_dim=2 * SHN,
                m_dim=128,
            )
            idesc_dn = prims.Tcgen05MxInstrDesc.build(
                a_dtype=cutlass.Float8E4M3FN,
                b_dtype=cutlass.Float8E4M3FN,
                scale_format=1,
                n_dim=SHN,
                m_dim=128,
            )
            s2t_shape, s2t_mc = prims.S2TCopyMode.S2T_32x128b_WARPX4
            desc_a0 = prims.Tcgen05SmemDesc.build(
                s_big.subview(0),
                leading_byte_offset=16,
                stride_byte_offset=1024,
                base_offset=0,
                layout=2,
            )
            desc_bx = prims.Tcgen05SmemDesc.build(
                s_big.subview(_SB_BX),
                leading_byte_offset=16,
                stride_byte_offset=1024,
                base_offset=0,
                layout=2,
            )
            desc_bh = prims.Tcgen05SmemDesc.build(
                s_big.subview(_SB_BH),
                leading_byte_offset=16,
                stride_byte_offset=1024,
                base_offset=0,
                layout=2,
            )
            tcol = (base_row << 16) | base_col
            sq = sid & cutlass.Int32(3)
            _wait(b_gu_rdy, PAR)
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            for g in cutlass.range(SH_STAGES, unroll=1):
                gg = g + cutlass.Int32(GB)
                st = gg % cutlass.Int32(SH_NST)
                if g == cutlass.Int32(SH_GT):
                    _wait(b_dn_rdy, PAR)
                    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                _wait(sh_full.subview(st), (gg // cutlass.Int32(SH_NST)) & cutlass.Int32(1))
                raw = cutlass.Int32(_SB_RAW // 4) + st * cutlass.Int32(512)
                if (g < cutlass.Int32(SH_GT)) & cutlass.Boolean(not SH_PRETILE):
                    # MMA row 32 Q + i = quadrant Q's row i: word 4 i + Q <- raw atom Q's word 4 i + sid % 4
                    w4 = [
                        big32.load(idx=raw + cutlass.Int32(128 * qq) + lane * cutlass.Int32(4) + sq)
                        for qq in range(4)
                    ]
                    big32.store(
                        (
                            cutlass.Int32(w4[0]),
                            cutlass.Int32(w4[1]),
                            cutlass.Int32(w4[2]),
                            cutlass.Int32(w4[3]),
                        ),
                        idx=cutlass.Int32(_SB_SPL // 4)
                        + st * cutlass.Int32(128)
                        + lane * cutlass.Int32(4),
                        alignment=16,
                    )
                    prims.fence_proxy("async_shared", space=prims.SharedSpace.shared_cta)
                cute.arch.sync_warp()
                prims.mbarrier_arrive(
                    sh_empty.subview(st)
                )  # this lane's generic reads of the slot are done
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                if prims.elect_sync():
                    if g < cutlass.Int32(SH_GT):
                        dsf = prims.Tcgen05SmemDesc.build(
                            s_big.subview(cutlass.Int32(_SB_SPL) + st * cutlass.Int32(512)),
                            leading_byte_offset=16,
                            stride_byte_offset=128,
                            base_offset=0,
                            layout=0,
                        )
                        prims.tcgen05_cp(
                            s2t_shape,
                            _tptr(base_row, base_col + SH_SFA + st * cutlass.Int32(4)),
                            dsf,
                            multicast=s2t_mc,
                        )
                        da = desc_a0 + cutlass.Int64(st * cutlass.Int32(NUM_BYTES_A >> 4))
                        db = desc_bx + cutlass.Int64(g * cutlass.Int32((2 * SHN * 128) >> 4))
                        for kb in cutlass.range_constexpr(4):
                            prims.tcgen05_mma_block_scale(
                                prims.MMABlockScaleKind.MXF8F6F4,
                                prims.CTAGroup.CTA_1,
                                cutlass.inttoptr(tcol + SH_ACC_GU, 6, cutlass.Float32),
                                da + 2 * kb,
                                db + 2 * kb,
                                idesc_gu.set_sf_ids(a_sf_id=kb, b_sf_id=kb),
                                (g > cutlass.Int32(0)) | cutlass.Boolean(kb > 0),
                                cutlass.inttoptr(
                                    tcol + SH_SFA + st * cutlass.Int32(4), 6, cutlass.Int32
                                ),
                                cutlass.inttoptr(
                                    tcol + SH_SFB_GU + g * cutlass.Int32(4), 6, cutlass.Int32
                                ),
                            )
                        prims.tcgen05_commit(sh_empty.subview(st))
                        if g == cutlass.Int32(SH_GT - 1):
                            prims.tcgen05_commit(acc_gu)
                    else:
                        gd = g - cutlass.Int32(SH_GT)
                        jt = gd // cutlass.Int32(6)
                        kt = gd - jt * cutlass.Int32(6)
                        dsf = prims.Tcgen05SmemDesc.build(
                            s_big.subview(
                                cutlass.Int32(_SB_SPL if SH_PRETILE else _SB_RAW)
                                + st * cutlass.Int32(512 if SH_PRETILE else 2048)
                            ),
                            leading_byte_offset=16,
                            stride_byte_offset=128,
                            base_offset=0,
                            layout=0,
                        )
                        prims.tcgen05_cp(
                            s2t_shape,
                            _tptr(base_row, base_col + SH_SFA + st * cutlass.Int32(4)),
                            dsf,
                            multicast=s2t_mc,
                        )
                        da = desc_a0 + cutlass.Int64(st * cutlass.Int32(NUM_BYTES_A >> 4))
                        db = desc_bh + cutlass.Int64(kt * cutlass.Int32((SHN * 128) >> 4))
                        for kb in cutlass.range_constexpr(4):
                            prims.tcgen05_mma_block_scale(
                                prims.MMABlockScaleKind.MXF8F6F4,
                                prims.CTAGroup.CTA_1,
                                cutlass.inttoptr(
                                    tcol + SH_ACC_DN + jt * cutlass.Int32(SHN), 6, cutlass.Float32
                                ),
                                da + 2 * kb,
                                db + 2 * kb,
                                idesc_dn.set_sf_ids(a_sf_id=kb, b_sf_id=kb),
                                (kt > cutlass.Int32(0)) | cutlass.Boolean(kb > 0),
                                cutlass.inttoptr(
                                    tcol + SH_SFA + st * cutlass.Int32(4), 6, cutlass.Int32
                                ),
                                cutlass.inttoptr(
                                    tcol + SH_SFB_DN + kt * cutlass.Int32(4), 6, cutlass.Int32
                                ),
                            )
                        prims.tcgen05_commit(sh_empty.subview(st))
                        if g == cutlass.Int32(SH_STAGES - 1):
                            prims.tcgen05_commit(acc_dn)
                cute.arch.sync_warp()
        if warp >= 2:
            if cutlass.const_expr(ps > 0):
                _wait(
                    acc_dn, PAR ^ 1
                )  # the previous pass's MMAs are done: its B tiles and TMEM scales are free
            # ---- the activation, MXFP8 per 32-K block (native mxfp8_quantize), into B (row 8 q + t of stage tile i =
            # token t's k-tile SH_GT q + i, 128-byte swizzle) and its codes
            qt = tidx - cutlass.Int32(64)
            for r in cutlass.range_constexpr((MS * SH_KB + NQ - 1) // NQ):
                q_ = qt + cutlass.Int32(r * NQ)
                if q_ < cutlass.Int32(MS * SH_KB):
                    t = q_ // cutlass.Int32(SH_KB)
                    b = q_ - t * cutlass.Int32(SH_KB)
                    wv = [
                        xb32.load(
                            idx=(t + cutlass.Int32(TT0)) * cutlass.Int32(H // 2)
                            + b * cutlass.Int32(16)
                            + cutlass.Int32(4 * z),
                            vector_size=4,
                            alignment=16,
                        )
                        for z in range(4)
                    ]
                    v = []
                    for z in cutlass.range_constexpr(4):
                        for zz in cutlass.range_constexpr(4):
                            w = cutlass.Int32(wv[z][zz])
                            v.append((w << cutlass.Int32(16)).bitcast(cutlass.Float32))
                            v.append((w & cutlass.Int32(-65536)).bitcast(cutlass.Float32))
                    amax = cutlass.Float32(0.0)
                    for z in cutlass.range_constexpr(32):
                        amax = cute.arch.fmax(amax, cute.math.abs(v[z]))
                    code, osc = _mx_osc(amax)
                    words = _mx_words(v, osc)
                    kt = b >> cutlass.Int32(2)
                    sl = kt // cutlass.Int32(SH_GT)
                    it = kt - sl * cutlass.Int32(SH_GT)
                    n = sl * cutlass.Int32(SHN) + t
                    c0 = (b & cutlass.Int32(3)) * cutlass.Int32(2)
                    rw = (
                        cutlass.Int32(_SB_BX)
                        + it * cutlass.Int32(2 * SHN * 128)
                        + n * cutlass.Int32(128)
                    ) >> cutlass.Int32(2)
                    big32.store(
                        (words[0], words[1], words[2], words[3]),
                        idx=rw + ((c0 ^ (n & cutlass.Int32(7))) << cutlass.Int32(2)),
                        alignment=16,
                    )
                    big32.store(
                        (words[4], words[5], words[6], words[7]),
                        idx=rw
                        + (((c0 + cutlass.Int32(1)) ^ (n & cutlass.Int32(7))) << cutlass.Int32(2)),
                        alignment=16,
                    )
                    big8.store(cutlass.Int8(code), idx=cutlass.Int32(_SB_XC) + q_)
            prims.fence_proxy("async_shared", space=prims.SharedSpace.shared_cta)
            cute.arch.sync_warp()
            cute.arch.barrier(barrier_id=SHX_BAR, number_of_threads=NQ)
        if warp >= 4:
            if warp < 8:
                qd = warp - cutlass.Int32(4)
                # B scales of gate_up to TMEM: column of stage i, lane n = 8 q + t: token t's codes of
                # k-tile SH_GT q + i
                n = lane
                t = n & cutlass.Int32(SHN - 1)
                ok = (n < cutlass.Int32(2 * SHN)) & (t < cutlass.Int32(MS))
                for i in cutlass.range_constexpr(SH_GT):
                    kt = (n // cutlass.Int32(SHN)) * cutlass.Int32(SH_GT) + cutlass.Int32(i)
                    wsf = big32.load(
                        idx=(
                            cutlass.Int32(_SB_XC)
                            + cutlass.select_(ok, t, cutlass.Int32(0)) * cutlass.Int32(SH_KB)
                            + kt * cutlass.Int32(4)
                        )
                        >> cutlass.Int32(2)
                    )
                    prims.tcgen05_st(
                        "32x32b",
                        _tptr(base_row + qd * cutlass.Int32(32), base_col + SH_SFB_GU + i * 4),
                        cutlass.select_(ok, cutlass.Int32(wsf), cutlass.Int32(0)),
                    )
                prims.tcgen05_wait(prims.Tcgen05Wait.STORE)
                prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                cute.arch.sync_warp()
                cute.arch.barrier(barrier_id=SHX_BAR + 1, number_of_threads=128)
                et = tidx - cutlass.Int32(128)
                if et == 0:
                    prims.mbarrier_arrive(b_gu_rdy)
                    _stamp(tdbg, bx, 54)
                # ---- gate_up out: quadrant qd lane l = MMA row 32 qd + l; slice q = qd / 2 holds token t
                # at column 8 q + t
                _wait(acc_gu, PAR)
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                av = prims.tcgen05_ld(
                    "32x32b",
                    _tptr(base_row + qd * cutlass.Int32(32), base_col + SH_ACC_GU, cutlass.Float32),
                    num=2 * SHN,
                )
                prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                if qd >= cutlass.Int32(2):
                    for tt in cutlass.range_constexpr(MS):
                        bigf.store(
                            cutlass.Float32(av[SHN + tt]),
                            idx=cutlass.Int32(_SB_XCH // 4)
                            + ((qd - cutlass.Int32(2)) * cutlass.Int32(32) + lane)
                            * cutlass.Int32(SHN)
                            + cutlass.Int32(tt),
                        )
                cute.arch.sync_warp()
                cute.arch.barrier(barrier_id=SHX_BAR + 1, number_of_threads=128)
                if qd < cutlass.Int32(2):
                    # m3_mx_gemv's owner sum: (0 + rank 0) + rank 1, then bf16
                    for tt in cutlass.range_constexpr(MS):
                        tot = _add_rn(cutlass.Float32(0.0), cutlass.Float32(av[tt]))
                        tot = _add_rn(
                            tot,
                            bigf.load(
                                idx=cutlass.Int32(_SB_XCH // 4)
                                + (qd * cutlass.Int32(32) + lane) * cutlass.Int32(SHN)
                                + cutlass.Int32(tt)
                            ),
                        )
                        bigf.store(
                            _bf16_round(tot),
                            idx=cutlass.Int32(_SB_GU // 4)
                            + (qd * cutlass.Int32(32) + lane) * cutlass.Int32(SHN)
                            + cutlass.Int32(tt),
                        )
                cute.arch.sync_warp()
                cute.arch.barrier(barrier_id=SHX_BAR + 1, number_of_threads=128)
                # h = silu_and_mul(gate, up) of positions 32 sid + p, published under the call's tag
                for r in cutlass.range_constexpr((32 * MS + 127) // 128):
                    pq = et + cutlass.Int32(r * 128)
                    if pq < cutlass.Int32(32 * MS):
                        p = pq // cutlass.Int32(MS)
                        tt = pq - p * cutlass.Int32(MS)
                        ga = bigf.load(idx=cutlass.Int32(_SB_GU // 4) + p * cutlass.Int32(SHN) + tt)
                        up = bigf.load(
                            idx=cutlass.Int32(_SB_GU // 4)
                            + (cutlass.Int32(32) + p) * cutlass.Int32(SHN)
                            + tt
                        )
                        q_a = cute.arch.fmin(cutlass.Float32(ga), cutlass.Float32(7.0))
                        q_bu = _min_xorsign_abs(cutlass.Float32(up), cutlass.Float32(7.0))
                        q_ub = _add_rn(q_bu, cutlass.Float32(1.0))
                        q_e2 = _ex2_full(
                            _mul_rn(
                                _mul_rn(q_a, cutlass.Float32(-1.70200002193450927734)),
                                cutlass.Float32(1.4426950216293334961),
                            )
                        )
                        q_sg = _div_full(cutlass.Float32(1.0), _add_rn(q_e2, cutlass.Float32(1.0)))
                        q_h = _mul_rn(q_ub, _mul_rn(q_sg, q_a))
                        _st_relaxed_u32(
                            hx.subview(
                                (tt + cutlass.Int32(TT0)) * cutlass.Int32(SH_I)
                                + sid * cutlass.Int32(32)
                                + p
                            )
                            .data_ptr()
                            .toint(),
                            (tag16 << cutlass.Int32(16)) | _bf16_bits(q_h),
                        )
                if et == 0:
                    _stamp(tdbg, bx, 50)
                # every shared CTA's h: (token, K block) per thread, 32 tagged words, MXFP8 into B (h) and its codes
                for rr in cutlass.range_constexpr((MS * SH_DKB + 127) // 128):
                    q_ = et + cutlass.Int32(rr * 128)
                    if q_ < cutlass.Int32(MS * SH_DKB):
                        t = q_ // cutlass.Int32(SH_DKB)
                        kb = q_ - t * cutlass.Int32(SH_DKB)
                        base = (t + cutlass.Int32(TT0)) * cutlass.Int32(SH_I) + kb * cutlass.Int32(
                            32
                        )
                        hv = [cutlass.Int32(0) for _ in range(32)]
                        pend = cutlass.Int32(1)
                        while pend != cutlass.Int32(0):
                            for q4 in cutlass.range_constexpr(8):
                                v4 = prims.load_ext(
                                    hx.subview(base + cutlass.Int32(4 * q4)),
                                    dtype=cutlass.Int32,
                                    count=4,
                                    order="relaxed",
                                    scope="gpu",
                                )
                                for z in cutlass.range_constexpr(4):
                                    hv[4 * q4 + z] = cutlass.Int32(v4[z])
                            okw = cutlass.Boolean(True)
                            for z in cutlass.range_constexpr(32):
                                okw = okw & (
                                    ((hv[z] >> cutlass.Int32(16)) & cutlass.Int32(0xFFFF)) == tag16
                                )
                            pend = cutlass.select_(okw, cutlass.Int32(0), cutlass.Int32(1))
                        vals = [
                            (hv[z] << cutlass.Int32(16)).bitcast(cutlass.Float32) for z in range(32)
                        ]
                        amax = cutlass.Float32(0.0)
                        for z in cutlass.range_constexpr(32):
                            amax = cute.arch.fmax(amax, cute.math.abs(vals[z]))
                        code, osc = _mx_osc(amax)
                        words = _mx_words(vals, osc)
                        kt = kb >> cutlass.Int32(2)
                        c0 = (kb & cutlass.Int32(3)) * cutlass.Int32(2)
                        rw = (
                            cutlass.Int32(_SB_BH)
                            + kt * cutlass.Int32(SHN * 128)
                            + t * cutlass.Int32(128)
                        ) >> cutlass.Int32(2)
                        big32.store(
                            (words[0], words[1], words[2], words[3]),
                            idx=rw + ((c0 ^ (t & cutlass.Int32(7))) << cutlass.Int32(2)),
                            alignment=16,
                        )
                        big32.store(
                            (words[4], words[5], words[6], words[7]),
                            idx=rw
                            + (
                                ((c0 + cutlass.Int32(1)) ^ (t & cutlass.Int32(7)))
                                << cutlass.Int32(2)
                            ),
                            alignment=16,
                        )
                        big8.store(cutlass.Int8(code), idx=cutlass.Int32(_SB_HC) + q_)
                prims.fence_proxy("async_shared", space=prims.SharedSpace.shared_cta)
                cute.arch.sync_warp()
                cute.arch.barrier(barrier_id=SHX_BAR + 1, number_of_threads=128)
                okh = lane < cutlass.Int32(MS)
                for kt in cutlass.range_constexpr(6):
                    wsf = big32.load(
                        idx=(
                            cutlass.Int32(_SB_HC)
                            + cutlass.select_(okh, lane, cutlass.Int32(0)) * cutlass.Int32(SH_DKB)
                            + cutlass.Int32(4 * kt)
                        )
                        >> cutlass.Int32(2)
                    )
                    prims.tcgen05_st(
                        "32x32b",
                        _tptr(base_row + qd * cutlass.Int32(32), base_col + SH_SFB_DN + kt * 4),
                        cutlass.select_(okh, cutlass.Int32(wsf), cutlass.Int32(0)),
                    )
                prims.tcgen05_wait(prims.Tcgen05Wait.STORE)
                prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                cute.arch.sync_warp()
                cute.arch.barrier(barrier_id=SHX_BAR + 1, number_of_threads=128)
                if et == 0:
                    prims.mbarrier_arrive(b_dn_rdy)
                    _stamp(tdbg, bx, 51)
                # ---- down out: tile j row 32 qd + l -> y_sh row 256 sid + 128 j + 32 qd + l
                _wait(acc_dn, PAR)
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                for jt in cutlass.range_constexpr(2):
                    dv = prims.tcgen05_ld(
                        "32x32b",
                        _tptr(
                            base_row + qd * cutlass.Int32(32),
                            base_col + SH_ACC_DN + jt * SHN,
                            cutlass.Float32,
                        ),
                        num=SHN,
                    )
                    prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                    rowo = (
                        sid * cutlass.Int32(256)
                        + cutlass.Int32(128 * jt)
                        + qd * cutlass.Int32(32)
                        + lane
                    )
                    for tt in cutlass.range_constexpr(MS):
                        _st_relaxed_u32(
                            dnx.subview(cutlass.Int32((tt + TT0) * H) + rowo).data_ptr().toint(),
                            (tag16 << cutlass.Int32(16)) | _bf16_bits(cutlass.Float32(dv[tt])),
                        )
                prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                if et == 0:
                    _stamp(tdbg, bx, 52)
    cute.arch.sync_warp()
    prims.barrier_cta_sync(0)
    if warp == 1:
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        prims.tcgen05_dealloc(cutlass.inttoptr(tmem_raw, 6, cutlass.Float32), SH_TMEM)
    if tidx == 0:
        epochs.store(ep + cutlass.Int32(1), idx=bx)


# =============================================================================
# v2 helpers: global release / acquire, proxy fences.
# =============================================================================
@dsl_user_op
def _red_release_add(addr, val, *, loc=None, ip=None):
    _asm(
        None,
        [cutlass.Int64(addr), cutlass.Int32(val)],
        "red.release.gpu.global.add.u32 [$0], $1;",
        "l,r",
        True,
        loc,
        ip,
    )


@dsl_user_op
def _atom_add_u64(addr, val, *, loc=None, ip=None):
    return cutlass.Int64(
        _asm(
            _T.i64(),
            [cutlass.Int64(addr), cutlass.Int64(val)],
            "atom.relaxed.gpu.global.add.u64 $0, [$1], $2;",
            "=l,l,l",
            True,
            loc,
            ip,
        )
    )


@dsl_user_op
def _ld_acquire_u32(addr, *, loc=None, ip=None):
    return cutlass.Int32(
        _asm(
            _T.i32(),
            [cutlass.Int64(addr)],
            "ld.acquire.gpu.global.u32 $0, [$1];",
            "=r,l",
            True,
            loc,
            ip,
        )
    )


@dsl_user_op
def _fence_proxy_async_global(*, loc=None, ip=None):
    _llvm.inline_asm(
        None,
        [],
        "fence.proxy.async.global;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _fence_acq_rel_gpu(*, loc=None, ip=None):
    _llvm.inline_asm(
        None,
        [],
        "fence.acq_rel.gpu;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.jit
def _dbg_delay(salt):
    """(DBG_DELAY stress builds) a pseudo-random ~0..4 us stall: protocol robustness under skew."""
    if cutlass.const_expr(DBG_DELAY):
        r = cutlass.Int32(_gtime() >> cutlass.Int64(6)) * cutlass.Int32(1103515245) + salt
        if ((r >> cutlass.Int32(12)) & cutlass.Int32(7)) == cutlass.Int32(0):
            for _ in cutlass.range((r >> cutlass.Int32(16)) & cutlass.Int32(31), unroll=1):
                _nanosleep(128)


@cute.jit
def _waitt(bar: cutlass.Array, parity, acc):
    """_wait that (TIMING) adds the waited ns to acc."""
    t0 = _gtime() if cutlass.const_expr(TIMING) else cutlass.Int64(0)
    while not cute.arch.mbarrier_try_wait(bar.data_ptr(), parity):
        pass
    cute.arch.sync_warp()
    if cutlass.const_expr(TIMING):
        acc = acc + (_gtime() - t0)
    return acc


@cute.jit
def _tstore(tdbg: cutlass.Array, bx, i: cutlass.Constexpr, v):
    if cutlass.const_expr(TIMING):
        tdbg.store(cutlass.Int64(v), idx=bx * cutlass.Int32(TD_STRIDE) + cutlass.Int32(i))


@cute.jit
def _issue_fc1(
    lane,
    stage,
    e,
    c,
    j,
    sA: cutlass.Array,
    sSF: cutlass.Array,
    ab_full: cutlass.Array,
    tma_a1,
    tma_s1,
    s1w: cutlass.Array,
):
    """FC1 stage j of unit c of expert e into ring stage `stage` (the caller waited its empty barrier): 4 TMA boxes
    (rows 32 c .., k-tiles 4 j + q) and the stage's scale atoms (SFA_CP: word [r][q] of atom kb <- raw atom
    (k-tile 4 j + q, kb) word [r][c % 4], by this warp's 32 lanes)."""
    row = c * cutlass.Int32(R1)
    kx = j * cutlass.Int32(C1 * KT_BYTES)
    if prims.elect_sync():
        prims.mbarrier_arrive_expect_tx(
            ab_full.subview(stage), NUM_BYTES_A if SFA_CP else NUM_BYTES_A + NUM_BYTES_SFA_RAW
        )
        for q in cutlass.range_constexpr(C1):
            prims.cp_async_bulk_tensor_shared_cta_global(
                sA.subview(stage * cutlass.Int32(NUM_BYTES_A) + cutlass.Int32(q * NUM_BYTES_BOX1)),
                tma_a1,
                (kx + cutlass.Int32(q * KT_BYTES), row, e),
                ab_full.subview(stage),
                l2_cache_hint=EVICT_FIRST,
            )
        if cutlass.const_expr(not SFA_CP):
            sf_line = (
                (e * cutlass.Int32(SF1_RB) + (c >> cutlass.Int32(2))) * cutlass.Int32(SF1_KA)
                + j * cutlass.Int32(C1 * NUM_KB)
            ) * cutlass.Int32(4)
            prims.cp_async_bulk_tensor_shared_cta_global(
                sSF.subview(stage * cutlass.Int32(NUM_WORDS_SFA_RAW)),
                tma_s1,
                (cutlass.Int32(0), sf_line),
                ab_full.subview(stage),
            )
    if cutlass.const_expr(SFA_CP):
        atom0 = (e * cutlass.Int32(SF1_RB) + (c >> cutlass.Int32(2))) * cutlass.Int32(
            SF1_KA
        ) + j * cutlass.Int32(16)
        src0 = atom0 * cutlass.Int32(128) + lane * cutlass.Int32(4) + (c & cutlass.Int32(3))
        dst0 = stage * cutlass.Int32(NUM_WORDS_SFA) + lane * cutlass.Int32(4)
        for q in cutlass.range_constexpr(C1):
            for kb in cutlass.range_constexpr(NUM_KB):
                prims.cp_async_shared_global(
                    sSF.subview(dst0 + cutlass.Int32(kb * 128 + q)),
                    s1w.subview(src0 + cutlass.Int32((4 * q + kb) * 128)),
                    4,
                    "ca",
                )
        prims.cp_async_mbarrier_arrive(ab_full.subview(stage), noinc=True)
        # one group per ring stage (racecheck: the issuer waits it before the slot's reuse)
        prims.cp_async_commit_group()
    cute.arch.sync_warp()


@cute.jit
def _issue_fc2(
    lane,
    stage,
    g,
    blk,
    t2: cutlass.Constexpr,
    n_here,
    s_el: cutlass.Array,
    sA: cutlass.Array,
    sSF: cutlass.Array,
    ab_full: cutlass.Array,
    tma_a2,
    tma_s2,
    s2w: cutlass.Array,
):
    """FC2 k-tile t2 of output block blk for the group's n_here slots: box jj = rows 32 blk .. of slot jj's w2 at
    k-tile t2; scale word [r][jj] of atom kb <- raw atom (rows / 128, K block 4 t2 + kb) word [r][blk % 4]."""
    if prims.elect_sync():
        prims.mbarrier_arrive_expect_tx(
            ab_full.subview(stage),
            n_here
            * cutlass.Int32(NUM_BYTES_BOX1 if SFA_CP else NUM_BYTES_BOX1 + NUM_BYTES_SFA_RAW // 4),
        )
        for jj in cutlass.range_constexpr(4):
            if cutlass.Int32(jj) < n_here:
                e = s_el.load(idx=g * cutlass.Int32(4) + cutlass.Int32(jj))
                prims.cp_async_bulk_tensor_shared_cta_global(
                    sA.subview(
                        stage * cutlass.Int32(NUM_BYTES_A) + cutlass.Int32(jj * NUM_BYTES_BOX1)
                    ),
                    tma_a2,
                    (cutlass.Int32(t2 * KT_BYTES), blk * cutlass.Int32(R2), e),
                    ab_full.subview(stage),
                    l2_cache_hint=EVICT_FIRST,
                )
                if cutlass.const_expr(not SFA_CP):
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        sSF.subview(
                            stage * cutlass.Int32(NUM_WORDS_SFA_RAW)
                            + cutlass.Int32(jj * NUM_WORDS_SFA_RAW // 4)
                        ),
                        tma_s2,
                        (
                            cutlass.Int32(0),
                            (
                                (e * cutlass.Int32(SF2_RB) + (blk >> cutlass.Int32(2)))
                                * cutlass.Int32(SF2_KA)
                                + cutlass.Int32(4 * t2)
                            )
                            * cutlass.Int32(4),
                        ),
                        ab_full.subview(stage),
                    )
    if cutlass.const_expr(SFA_CP):
        dst0 = stage * cutlass.Int32(NUM_WORDS_SFA) + lane * cutlass.Int32(4)
        for jj in cutlass.range_constexpr(4):
            if cutlass.Int32(jj) < n_here:
                e = s_el.load(idx=g * cutlass.Int32(4) + cutlass.Int32(jj))
                src0 = (
                    (
                        (e * cutlass.Int32(SF2_RB) + (blk >> cutlass.Int32(2)))
                        * cutlass.Int32(SF2_KA)
                        + cutlass.Int32(4 * t2)
                    )
                    * cutlass.Int32(128)
                    + lane * cutlass.Int32(4)
                    + (blk & cutlass.Int32(3))
                )
                for kb in cutlass.range_constexpr(NUM_KB):
                    prims.cp_async_shared_global(
                        sSF.subview(dst0 + cutlass.Int32(kb * 128 + jj)),
                        s2w.subview(src0 + cutlass.Int32(kb * 128)),
                        4,
                        "ca",
                    )
        prims.cp_async_mbarrier_arrive(ab_full.subview(stage), noinc=True)
        # one group per ring stage (racecheck: the issuer waits it before the slot's reuse)
        prims.cp_async_commit_group()
    cute.arch.sync_warp()


@cute.jit
def _quant_words(w, sc, rsc, rsix):
    """NVFP4 of one 16-value block given as 8 bf16-pair words: (lo word, hi word, scale e4m3 bits)."""
    xs = []
    for z in cutlass.range_constexpr(8):
        xs.append((w[z] << cutlass.Int32(16)).bitcast(cutlass.Float32))
        xs.append((w[z] & cutlass.Int32(-65536)).bitcast(cutlass.Float32))
    vmax = cute.math.abs(xs[0])
    for z in cutlass.range_constexpr(1, 16):
        vmax = cute.arch.fmax(vmax, cute.math.abs(xs[z]))
    sfb = _e4m3_bits(_mul_rn(sc, _mul_rn(vmax, rsix)))
    osc = cutlass.select_(
        vmax != cutlass.Float32(0.0), _rcp(_mul_rn(_e4m3_to_f32(sfb), rsc)), cutlass.Float32(0.0)
    )
    by = [
        _e2m1x2(_mul_rn(xs[2 * z + 1], osc), _mul_rn(xs[2 * z], osc)) & cutlass.Int32(0xFF)
        for z in range(8)
    ]
    lo = (
        by[0]
        | (by[1] << cutlass.Int32(8))
        | (by[2] << cutlass.Int32(16))
        | (by[3] << cutlass.Int32(24))
    )
    hi = (
        by[4]
        | (by[5] << cutlass.Int32(8))
        | (by[6] << cutlass.Int32(16))
        | (by[7] << cutlass.Int32(24))
    )
    return lo, hi, sfb


@cute.jit
def _quant_block(xs32: cutlass.Array, widx, sc, rsc, rsix):
    """NVFP4 of one 16-value block (8 bf16 words at widx): (lo word, hi word, scale e4m3 bits), fp4_quantize's ops."""
    w = []
    for h in cutlass.range_constexpr(2):
        v4 = xs32.load(idx=widx + cutlass.Int32(4 * h), vector_size=4, alignment=16)
        for z in cutlass.range_constexpr(4):
            w.append(cutlass.Int32(v4[z]))
    xs = []
    for z in cutlass.range_constexpr(8):
        xs.append((w[z] << cutlass.Int32(16)).bitcast(cutlass.Float32))
        xs.append((w[z] & cutlass.Int32(-65536)).bitcast(cutlass.Float32))
    vmax = cute.math.abs(xs[0])
    for z in cutlass.range_constexpr(1, 16):
        vmax = cute.arch.fmax(vmax, cute.math.abs(xs[z]))
    sfb = _e4m3_bits(_mul_rn(sc, _mul_rn(vmax, rsix)))
    osc = cutlass.select_(
        vmax != cutlass.Float32(0.0), _rcp(_mul_rn(_e4m3_to_f32(sfb), rsc)), cutlass.Float32(0.0)
    )
    by = [
        _e2m1x2(_mul_rn(xs[2 * z + 1], osc), _mul_rn(xs[2 * z], osc)) & cutlass.Int32(0xFF)
        for z in range(8)
    ]
    lo = (
        by[0]
        | (by[1] << cutlass.Int32(8))
        | (by[2] << cutlass.Int32(16))
        | (by[3] << cutlass.Int32(24))
    )
    hi = (
        by[4]
        | (by[5] << cutlass.Int32(8))
        | (by[6] << cutlass.Int32(16))
        | (by[7] << cutlass.Int32(24))
    )
    return lo, hi, sfb


@cute.kernel
def m3_moe_kernel(
    tma_a1_desc: cutlass.GridConstant[cuda.TensorMap],
    tma_a2_desc: cutlass.GridConstant[cuda.TensorMap],
    tma_s1_desc: cutlass.GridConstant[cuda.TensorMap],
    tma_s2_desc: cutlass.GridConstant[cuda.TensorMap],
    tma_gu_desc: cutlass.GridConstant[cuda.TensorMap],
    tma_dn_desc: cutlass.GridConstant[cuda.TensorMap],
    bias: cutlass.Array,  # f32 [128]
    escal: cutlass.Array,  # f32 [6, 128]
    plan: cutlass.Array,  # int32 [4 M + 1, NCTA, M + 1]
    partials: cutlass.Array,  # int32 bits of f32 [G_CAP * 48, M, 32]: split units' head sums
    img32: cutlass.Array,  # int32 [NG_CAP, IMG_BYTES / 4]: per-group FC2 B image (tile + scale atoms)
    cnt: cutlass.Array,  # int32 [NG_CAP * 8]: per-group unit counters (index 8 g)
    s1w: cutlass.Array,  # int32 words of w3_w1_weight_scale
    s2w: cutlass.Array,  # int32 words of w2_weight_scale
    w1b: cutlass.Array,  # int8 bytes of w3_w1_weight (L2 prefetch)
    w2b: cutlass.Array,  # int8 bytes of w2_weight (L2 prefetch)
    epochs: cutlass.Array,  # int32 [NCTA + NSH]
    out: cutlass.Array,  # bf16 [M, 6144]
    route_dbg: cutlass.Array,  # int32 [M, 8]
    tdbg: cutlass.Array,  # int64 [NCTA + NSH, 64]
    xb32: cutlass.Array,  # int32 words (bf16 pairs) of the hidden states [M, 6144]
    rw: cutlass.Array,  # f32 router weight [128, 6144]
    sfsc: cutlass.Array,  # f32 [1] fc31_input_scale
    lgbuf: cutlass.Array,  # int64 [LG_REPL, LG_STRIDE]
    guw: cutlass.Array,
    gusf: cutlass.Array,
    dnw: cutlass.Array,
    dnsf: cutlass.Array,
    hx: cutlass.Array,
    dnx: cutlass.Array,
    arp: cutlass.Array,
):
    tidx, _, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = tidx % 32
    if cutlass.const_expr(TICKET):
        # role by entry order: the CTAs that become resident last (behind the predecessor's CTAs) take the shared
        # expert roles (most slack), the first NCTA the routed ones (the router GEMV / logits exchange)
        s_tk = cutlass.Array(cutlass.Int32, 4, space=cutlass.AddressSpace.smem, alignment=16)
        if tidx == 0:
            old = _atom_add_u64(
                cnt.subview(cutlass.Int32(NG_CAP * 8 + 8)).data_ptr().toint(), cutlass.Int64(1)
            )
            s_tk.store(cutlass.Int32(old % cutlass.Int64(NCTA + NSH)), idx=0)
        prims.barrier_cta_sync(0)
        bx = s_tk.load(idx=0)
    if tidx == 0:
        _stamp(tdbg, bx, 0)
        if cutlass.const_expr(TIMING):
            tdbg.store(
                cutlass.Int64(_smid()), idx=bx * cutlass.Int32(TD_STRIDE) + cutlass.Int32(63)
            )

    s_big = cutlass.Array(cutlass.Int8, BIG, space=cutlass.AddressSpace.smem, alignment=1024)
    sA = cutlass.Array(
        s_big.data_ptr(0), shape=(NUM_BYTES_A * STAGES,), dtype=cutlass.Int8, alignment=1024
    )
    # ring scale words: SFA_CP: the instruction atoms [STAGES][4 x 512 B]; else raw boxes [STAGES][8 KB] + spliced
    sSF = cutlass.Array(
        s_big.data_ptr(_OFF_SFA),
        shape=(STAGES * STAGE_SF // 4,),
        dtype=cutlass.Int32,
        alignment=1024,
    )
    sSFA = cutlass.Array(
        s_big.data_ptr(_OFF_SFA + (0 if SFA_CP else STAGES * NUM_BYTES_SFA_RAW)),
        shape=(NUM_WORDS_SFA * STAGES,),
        dtype=cutlass.Int32,
        alignment=512,
    )
    sB1 = cutlass.Array(
        s_big.data_ptr(_OFF_B1), shape=(NUM_BYTES_B1 * S1,), dtype=cutlass.Int8, alignment=1024
    )
    sB2 = cutlass.Array(
        s_big.data_ptr(_OFF_B2), shape=(NB2BUF * IMG_B2,), dtype=cutlass.Int8, alignment=1024
    )
    sSFB2 = cutlass.Array(
        s_big.data_ptr(_OFF_SFB2), shape=(IMG_SF // 4,), dtype=cutlass.Int32, alignment=1024
    )
    ys = cutlass.Array(
        cutlass.BFloat16,
        NB2_CAP * M_MAX * TOP_K * R2,
        space=cutlass.AddressSpace.smem,
        alignment=16,
    )  # bf16 values
    s_part = cutlass.Array(
        cutlass.Float32, SPB * M_MAX * C1 * R1, space=cutlass.AddressSpace.smem, alignment=16
    )
    s_bias = cutlass.Array(cutlass.Float32, E_LOCAL, space=cutlass.AddressSpace.smem, alignment=16)
    s_red = cutlass.Array(
        cutlass.Float32, 9 * M_MAX + 16, space=cutlass.AddressSpace.smem, alignment=16
    )
    # the activation's scale codes [M, 384] until they are in TMEM: aliases ys (first written by the FC2 epilogue)
    s_xsf = cutlass.Array(
        ys.data_ptr(0), shape=(M_MAX * (H // 64),), dtype=cutlass.Int32, alignment=16
    )
    ab_full = cutlass.Array(cutlass.Int64, STAGES, space=cutlass.AddressSpace.smem, alignment=8)
    ab_empty = cutlass.Array(cutlass.Int64, STAGES, space=cutlass.AddressSpace.smem, alignment=8)
    sfa_ready = cutlass.Array(cutlass.Int64, STAGES, space=cutlass.AddressSpace.smem, alignment=8)
    acc_full = cutlass.Array(cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8)
    acc_empty = cutlass.Array(cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8)
    acc2_full = cutlass.Array(cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8)
    acc2_empty = cutlass.Array(cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8)
    b2_ready = cutlass.Array(cutlass.Int64, NB2BUF, space=cutlass.AddressSpace.smem, alignment=8)
    b2_free = cutlass.Array(cutlass.Int64, NB2BUF, space=cutlass.AddressSpace.smem, alignment=8)
    img_bar = cutlass.Array(cutlass.Int64, NB2BUF, space=cutlass.AddressSpace.smem, alignment=8)
    sfb2_free = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    b1_ready = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    x_full = cutlass.Array(cutlass.Int64, NXB, space=cutlass.AddressSpace.smem, alignment=8)
    x_empty = cutlass.Array(cutlass.Int64, NXB, space=cutlass.AddressSpace.smem, alignment=8)
    route_bar = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    ys_done = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    grp_done = cutlass.Array(cutlass.Int64, 4, space=cutlass.AddressSpace.smem, alignment=8)
    b1_load = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    topk_bar = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    tok0_bar = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    px_full = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    s_prog = cutlass.Array(cutlass.Int32, 4, space=cutlass.AddressSpace.smem, alignment=16)
    tmem_ptr_i32 = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem)
    s_el = cutlass.Array(cutlass.Int32, G_CAP + 1, space=cutlass.AddressSpace.smem, alignment=16)
    s_kpos = cutlass.Array(
        cutlass.Int32, G_CAP * M_MAX, space=cutlass.AddressSpace.smem, alignment=16
    )
    s_w = cutlass.Array(
        cutlass.Float32, M_MAX * TOP_K, space=cutlass.AddressSpace.smem, alignment=16
    )
    s_ids = cutlass.Array(
        cutlass.Int32, M_MAX * TOP_K, space=cutlass.AddressSpace.smem, alignment=16
    )
    s_plan = cutlass.Array(cutlass.Int32, NG_CAP + 1, space=cutlass.AddressSpace.smem, alignment=16)
    s_gn = cutlass.Array(
        cutlass.Int32, NG_CAP, space=cutlass.AddressSpace.smem, alignment=16
    )  # slots per group
    s_emap = cutlass.Array(cutlass.Int32, E_LOCAL, space=cutlass.AddressSpace.smem, alignment=16)
    s_misc = cutlass.Array(cutlass.Int32, 4, space=cutlass.AddressSpace.smem, alignment=16)
    s_plan_all = cutlass.Array(
        cutlass.Int32, (G_CAP + 1) * (NG_CAP + 1), space=cutlass.AddressSpace.smem, alignment=16
    )
    s_esc = cutlass.Array(cutlass.Float32, 8 * G_CAP, space=cutlass.AddressSpace.smem, alignment=32)

    is_routed = cutlass.Boolean(True)
    if cutlass.const_expr(SHARED):
        sh_full = cutlass.Array(cutlass.Int64, 32, space=cutlass.AddressSpace.smem, alignment=8)
        is_routed = bx < cutlass.Int32(NCTA)
        if bx >= cutlass.Int32(NCTA):
            _shared_cta(
                tidx,
                warp,
                lane,
                bx - cutlass.Int32(NCTA),
                s_big,
                sh_full,
                epochs,
                tdbg,
                bx,
                xb32,
                tma_gu_desc.get_ptr(),
                tma_dn_desc.get_ptr(),
                gusf,
                dnsf,
                hx,
                dnx,
                arp,
                tmem_ptr_i32,
                guw,
                dnw,
            )
    if is_routed:
        # ---- before the grid dependency
        if warp == 0:
            if tidx < STAGES:
                prims.mbarrier_init(ab_full.subview(tidx), 33 if SFA_CP else 1)
                prims.mbarrier_init(ab_empty.subview(tidx), 1)
                prims.mbarrier_init(sfa_ready.subview(tidx), 1)
            if tidx < 2:
                prims.mbarrier_init(acc_full.subview(tidx), 1)
                prims.mbarrier_init(acc_empty.subview(tidx), EPI_THREADS)
                prims.mbarrier_init(acc2_full.subview(tidx), 1)
                prims.mbarrier_init(acc2_empty.subview(tidx), EPI_THREADS)
            if tidx < NB2BUF:
                prims.mbarrier_init(b2_ready.subview(tidx), 1)
                prims.mbarrier_init(b2_free.subview(tidx), 1)
                prims.mbarrier_init(img_bar.subview(tidx), 1)
            if tidx < NXB:
                prims.mbarrier_init(x_full.subview(tidx), 1)
                prims.mbarrier_init(x_empty.subview(tidx), NWARPS)
            if tidx == 0:
                prims.mbarrier_init(b1_ready.subview(0), EPI_THREADS)
                prims.mbarrier_init(sfb2_free.subview(0), 1)
                prims.mbarrier_init(route_bar.subview(0), 32)
                prims.mbarrier_init(ys_done.subview(0), EPI_THREADS)
                prims.mbarrier_init(b1_load.subview(0), 1)
                prims.mbarrier_init(topk_bar.subview(0), M_MAX * TOP_K)
                prims.mbarrier_init(tok0_bar.subview(0), TOP_K)
                prims.mbarrier_init(px_full.subview(0), 1)
            if tidx < 4:
                prims.mbarrier_init(grp_done.subview(tidx), EPI_THREADS)
        if tidx == 0:
            s_prog.store(cutlass.Int32(0), idx=0)
        if warp == X_WARP:  # slot tables: empty (the slot warp fills them after the top-k)
            for z in cutlass.range_constexpr((G_CAP * M_MAX + 31) // 32):
                if lane + cutlass.Int32(32 * z) < cutlass.Int32(G_CAP * M_MAX):
                    s_kpos.store(cutlass.Int32(-1), idx=lane + cutlass.Int32(32 * z))
            for z in cutlass.range_constexpr(E_LOCAL // 32):
                s_emap.store(cutlass.Int32(-1), idx=lane + cutlass.Int32(32 * z))
        if warp == MMA_WARP:
            prims.tcgen05_alloc(tmem_ptr_i32, TMEM_COLS)
            prims.tcgen05_relinquish_alloc_permit()
        if warp == W_WARP:
            prims.prefetch_tensormap(tma_a1_desc.get_ptr())
            prims.prefetch_tensormap(tma_a2_desc.get_ptr())
            if cutlass.const_expr(not SFA_CP):
                prims.prefetch_tensormap(tma_s1_desc.get_ptr())
                prims.prefetch_tensormap(tma_s2_desc.get_ptr())
            for i in cutlass.range_constexpr(E_LOCAL // 32):
                s_bias.store(cutlass.Float32(bias.load(idx=lane + i * 32)), idx=lane + i * 32)
            for i in cutlass.range_constexpr(((G_CAP + 1) * (NG_CAP + 1) + 31) // 32):
                q = lane + cutlass.Int32(32 * i)
                if q < cutlass.Int32((G_CAP + 1) * (NG_CAP + 1)):
                    gq = q // cutlass.Int32(NG_CAP + 1)
                    s_plan_all.store(
                        plan.load(
                            idx=(gq * cutlass.Int32(NCTA) + bx) * cutlass.Int32(NG_CAP + 1)
                            + q
                            - gq * cutlass.Int32(NG_CAP + 1)
                        ),
                        idx=q,
                    )
        # router weights (expert bx) in registers: thread tg = tid % 256 owns elements 4 tg + 1024 j
        tg = tidx % cutlass.Int32(GEMV_THREADS)
        half = tidx // cutlass.Int32(GEMV_THREADS)
        rwv = [
            rw.load(
                idx=bx * cutlass.Int32(H) + tg * cutlass.Int32(4) + cutlass.Int32(1024 * j),
                vector_size=4,
                alignment=16,
            )
            for j in range(H // 1024)
        ]
        prims.fence_mbarrier_init()
        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
        cute.arch.sync_warp()
        prims.barrier_cta_sync(0)
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        tmem_raw = tmem_ptr_i32.load()
        base_col = tmem_raw & 0xFFFF
        base_row = tmem_raw >> 16

        cute.arch.griddepcontrol_wait()
        # ---- x chunks into the idle ring (warp 6): chunk c = K [XCK c, XCK (c + 1)) of every token
        if warp == S_WARP:
            if cutlass.const_expr(NPT > 0):
                # the priority tokens' rows first (their logits are published ahead of the others)
                if lane == 0:
                    prims.mbarrier_arrive_expect_tx(px_full.subview(0), NPT * H * 2)
                cute.arch.sync_warp()
                if lane < NPT:
                    prims.cp_async_bulk_shared_cluster_global(
                        sB1.subview(lane * cutlass.Int32(H * 2)),
                        xb32.subview(
                            cutlass.select_(
                                lane == cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(PT1)
                            )
                            * cutlass.Int32(H // 2)
                        ),
                        px_full.subview(0),
                        H * 2,
                    )
            for c in cutlass.range_constexpr(NXB):
                if lane == 0:
                    prims.mbarrier_arrive_expect_tx(x_full.subview(c), XCB)
                cute.arch.sync_warp()
                if lane < M_MAX:
                    prims.cp_async_bulk_shared_cluster_global(
                        s_big.subview(cutlass.Int32(c * XCB) + lane * cutlass.Int32(XCK * 2)),
                        xb32.subview(lane * cutlass.Int32(H // 2) + cutlass.Int32(c * XCK // 2)),
                        x_full.subview(c),
                        XCK * 2,
                    )
        ar_on = arp.load(idx=0) != cutlass.Int64(0)
        if cutlass.const_expr(not LATE_TRIGGER):
            if arp.load(idx=0) == cutlass.Int64(0):
                cute.arch.griddepcontrol_launch_dependents()
        if tidx == 0:
            _stamp(tdbg, bx, 1)
        ep = epochs.load(idx=bx)
        tag16 = (ep & cutlass.Int32(0x7FFF)) | cutlass.Int32(0x8000)
        ltag = cutlass.Int64(ep + cutlass.Int32(1)) << cutlass.Int64(32)
        cnt_target = (ep + cutlass.Int32(1)) * cutlass.Int32(CNT_PER_CALL)

        sc = cutlass.Float32(sfsc.load(idx=0))
        rsc = _rcp(sc)
        rsix = _rcp(cutlass.Float32(6.0))
        img8f = cutlass.Array(
            img32.data_ptr(0), shape=(IMG_TOTAL,), dtype=cutlass.Int8, alignment=16
        )
        if cutlass.const_expr(NPT > 0):
            # ---- priority tokens (0 and PT1): their logits first (m3_router_gemv's per-thread / warp order)
            _wait(px_full.subview(0), 0)
            ph = cutlass.select_(half < cutlass.Int32(NPT), half, cutlass.Int32(0))
            pacc = cutlass.Float32(0.0)
            px32 = cutlass.Array(
                sB1.data_ptr(0), shape=(NPT * H // 2,), dtype=cutlass.Int32, alignment=16
            )
            for j in cutlass.range_constexpr(H // 1024):
                pxw = px32.load(
                    idx=ph * cutlass.Int32(H // 2) + cutlass.Int32(512 * j) + tg * cutlass.Int32(2),
                    vector_size=2,
                    alignment=8,
                )
                px0 = cutlass.Int32(pxw[0])
                px1 = cutlass.Int32(pxw[1])
                pacc = _fma_rn(
                    (px0 << cutlass.Int32(16)).bitcast(cutlass.Float32),
                    cutlass.Float32(rwv[j][0]),
                    pacc,
                )
                pacc = _fma_rn(
                    (px0 & cutlass.Int32(-65536)).bitcast(cutlass.Float32),
                    cutlass.Float32(rwv[j][1]),
                    pacc,
                )
                pacc = _fma_rn(
                    (px1 << cutlass.Int32(16)).bitcast(cutlass.Float32),
                    cutlass.Float32(rwv[j][2]),
                    pacc,
                )
                pacc = _fma_rn(
                    (px1 & cutlass.Int32(-65536)).bitcast(cutlass.Float32),
                    cutlass.Float32(rwv[j][3]),
                    pacc,
                )
            for off in (16, 8, 4, 2, 1):
                pacc = _add_rn(pacc, cute.arch.shuffle_sync_bfly(pacc, off))
            if lane == 0:
                s_red.store(pacc, idx=cutlass.Int32(9 * M_MAX) + warp)  # [half][8 warps]
            cute.arch.sync_warp()
            cute.arch.barrier(barrier_id=ALL_BAR, number_of_threads=THREADS)
            if tidx < cutlass.Int32(NPT * LG_REPL):
                hh = tidx // cutlass.Int32(LG_REPL)
                rr = tidx - hh * cutlass.Int32(LG_REPL)
                sacc = cutlass.Float32(0.0)
                for w in cutlass.range_constexpr(8):
                    sacc = _add_rn(
                        sacc, s_red.load(idx=cutlass.Int32(9 * M_MAX + w) + hh * cutlass.Int32(8))
                    )
                tp = cutlass.select_(hh == cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(PT1))
                _st_relaxed_u64(
                    lgbuf.subview(rr * cutlass.Int32(LG_STRIDE) + tp * cutlass.Int32(E_LOCAL) + bx)
                    .data_ptr()
                    .toint(),
                    ltag | (cutlass.Int64(sacc.bitcast(cutlass.Int32)) & cutlass.Int64(0xFFFFFFFF)),
                )
            if tidx == 0:
                _stamp(tdbg, bx, 43)
        # ---- this CTA's share of the NVFP4 activation (QB blocks, from global) for the global B1 image: loads first
        QB = (M_MAX * (H // 16) + NCTA - 1) // NCTA
        qb = bx * cutlass.Int32(QB) + tidx
        q_ok = (tidx < cutlass.Int32(QB)) & (qb < cutlass.Int32(M_MAX * (H // 16)))
        qb = cutlass.select_(q_ok, qb, cutlass.Int32(0))
        qtok = qb // cutlass.Int32(H // 16)
        qkb = qb - qtok * cutlass.Int32(H // 16)
        xq_v = [
            xb32.load(
                idx=qtok * cutlass.Int32(H // 2) + qkb * cutlass.Int32(8) + cutlass.Int32(4 * h),
                vector_size=4,
                alignment=16,
            )
            for h in range(2)
        ]
        # ---- router GEMV from the staged chunks, all 512 threads (rolled: the front runs once per call)
        MH = (M_MAX + 1) // 2  # tokens per GEMV half
        acc_r = cute.make_rmem_tensor((MH,), cutlass.Float32)
        for i in cutlass.range_constexpr(MH):
            acc_r[i] = cutlass.Float32(0.0)
        big32 = cutlass.Array(
            s_big.data_ptr(0), shape=(BIG // 4,), dtype=cutlass.Int32, alignment=16
        )
        JPC = XCK // 1024  # router j-steps per chunk
        for xc in cutlass.range(NXC, unroll=1):
            buf = xc % cutlass.Int32(NXB)
            _wait(x_full.subview(buf), (xc // cutlass.Int32(NXB)) & cutlass.Int32(1))
            if (tidx == 0) & (xc == cutlass.Int32(0)):
                _stamp(tdbg, bx, 41)
            xw0 = buf * cutlass.Int32(XCB // 4)  # chunk word base in s_big
            for jj in cutlass.range_constexpr(JPC):
                jg = xc * cutlass.Int32(JPC) + cutlass.Int32(jj)
                wsel = [cutlass.Float32(rwv[0][z]) for z in range(4)]
                for jx in cutlass.range_constexpr(1, H // 1024):
                    for z in cutlass.range_constexpr(4):
                        wsel[z] = cutlass.select_(
                            jg == cutlass.Int32(jx), cutlass.Float32(rwv[jx][z]), wsel[z]
                        )
                for i in cutlass.range_constexpr(MH):
                    t = half * cutlass.Int32(MH) + cutlass.Int32(i)
                    t = cutlass.select_(t < cutlass.Int32(M_MAX), t, cutlass.Int32(M_MAX - 1))
                    xw = big32.load(
                        idx=xw0
                        + t * cutlass.Int32(XCK // 2)
                        + cutlass.Int32(512 * jj)
                        + tg * cutlass.Int32(2),
                        vector_size=2,
                        alignment=8,
                    )
                    x0 = cutlass.Int32(xw[0])
                    x1 = cutlass.Int32(xw[1])
                    acc_g = acc_r[i]
                    acc_g = _fma_rn(
                        (x0 << cutlass.Int32(16)).bitcast(cutlass.Float32), wsel[0], acc_g
                    )
                    acc_g = _fma_rn(
                        (x0 & cutlass.Int32(-65536)).bitcast(cutlass.Float32), wsel[1], acc_g
                    )
                    acc_g = _fma_rn(
                        (x1 << cutlass.Int32(16)).bitcast(cutlass.Float32), wsel[2], acc_g
                    )
                    acc_g = _fma_rn(
                        (x1 & cutlass.Int32(-65536)).bitcast(cutlass.Float32), wsel[3], acc_g
                    )
                    acc_r[i] = acc_g
            if cutlass.const_expr(NXB < NXC):
                if xc + cutlass.Int32(NXB) < cutlass.Int32(NXC):
                    # every thread's reads of this chunk are done (CTA barrier, then a generic -> async proxy fence by
                    # the refilling warp: the bulk copy overwrites what the generic proxy read); racecheck-clean
                    cute.arch.sync_warp()
                    cute.arch.barrier(barrier_id=ALL_BAR, number_of_threads=THREADS)
                    if warp == S_WARP:
                        prims.fence_proxy("async_shared", space=prims.SharedSpace.shared_cta)
                        if lane == 0:
                            prims.mbarrier_arrive_expect_tx(x_full.subview(buf), XCB)
                        cute.arch.sync_warp()
                        if lane < M_MAX:
                            prims.cp_async_bulk_shared_cluster_global(
                                s_big.subview(
                                    buf * cutlass.Int32(XCB) + lane * cutlass.Int32(XCK * 2)
                                ),
                                xb32.subview(
                                    lane * cutlass.Int32(H // 2)
                                    + (xc + cutlass.Int32(NXB)) * cutlass.Int32(XCK // 2)
                                ),
                                x_full.subview(buf),
                                XCK * 2,
                            )
        if tidx == 0:
            _stamp(tdbg, bx, 42)
        for i in cutlass.range_constexpr(MH):
            v = acc_r[i]
            for off in (16, 8, 4, 2, 1):
                v = _add_rn(v, cute.arch.shuffle_sync_bfly(v, off))
            t = half * cutlass.Int32(MH) + cutlass.Int32(i)
            if (lane == 0) & (t < cutlass.Int32(M_MAX)):
                s_red.store(v, idx=(warp % cutlass.Int32(8)) * cutlass.Int32(M_MAX) + t)
        # this CTA's quantization blocks into the global B1 image (B1 tile bytes, then the [M, 384] scale codes)
        if q_ok:
            qw = []
            for h in cutlass.range_constexpr(2):
                for z in cutlass.range_constexpr(4):
                    qw.append(cutlass.Int32(xq_v[h][z]))
            qlo, qhi, qsf = _quant_words(qw, sc, rsc, rsix)
            img32.store(
                (qlo, qhi), idx=cutlass.Int32(B1IMG_OFF // 4) + _b1_addr(qtok, qkb), alignment=8
            )
            img8f.store(cutlass.Int8(qsf), idx=cutlass.Int32(B1IMG_OFF + S1 * NUM_BYTES_B1) + qb)
        cute.arch.sync_warp()
        cute.arch.barrier(barrier_id=ALL_BAR, number_of_threads=THREADS)
        if warp == cutlass.Int32(0):
            _dbg_delay(bx * cutlass.Int32(3))
        if tidx < LG_REPL:
            for t in cutlass.range_constexpr(M_MAX):
                sacc = cutlass.Float32(0.0)
                for w in cutlass.range_constexpr(8):
                    sacc = _add_rn(sacc, s_red.load(idx=cutlass.Int32(w * M_MAX + t)))
                lw = ltag | (cutlass.Int64(sacc.bitcast(cutlass.Int32)) & cutlass.Int64(0xFFFFFFFF))
                _st_relaxed_u64(
                    lgbuf.subview(tidx * cutlass.Int32(LG_STRIDE) + cutlass.Int32(t * E_LOCAL) + bx)
                    .data_ptr()
                    .toint(),
                    lw,
                )
        if tidx == 0:
            _stamp(tdbg, bx, 21)
        if tidx == cutlass.Int32(
            THREADS - 32
        ):  # B1 share released after the logits (its gpu fence stalls warp 15 only)
            if cutlass.const_expr(EPI_PFENCE):
                _fence_proxy_async_global()  # once, after the join (ALL_BAR) of the writer threads
            _red_release_add(
                cnt.subview(cutlass.Int32(NG_CAP * 8)).data_ptr().toint(), cutlass.Int32(1)
            )
        # ---- the B1 image (every CTA's share) into smem (warp 13; the top-k warps are 0 .. M - 1)
        if cutlass.const_expr(B1_WARP >= 0):
            if warp == cutlass.Int32(B1_WARP):
                if lane == 0:
                    b1addr = cnt.subview(cutlass.Int32(NG_CAP * 8)).data_ptr().toint()
                    b1tgt = (ep + cutlass.Int32(1)) * cutlass.Int32(NCTA)
                    cv = _ld_acquire_u32(b1addr)
                    while (cv - b1tgt) < cutlass.Int32(0):
                        _nanosleep(64)
                        cv = _ld_acquire_u32(b1addr)
                    _fence_proxy_async_global()
                    prims.mbarrier_arrive_expect_tx(
                        b1_load.subview(0), S1 * NUM_BYTES_B1 + M_MAX * (H // 16)
                    )
                    prims.cp_async_bulk_shared_cluster_global(
                        sB1.subview(0),
                        img32.subview(cutlass.Int32(B1IMG_OFF // 4)),
                        b1_load.subview(0),
                        S1 * NUM_BYTES_B1,
                    )
                    prims.cp_async_bulk_shared_cluster_global(
                        s_xsf.subview(0),
                        img32.subview(cutlass.Int32((B1IMG_OFF + S1 * NUM_BYTES_B1) // 4)),
                        b1_load.subview(0),
                        M_MAX * (H // 16),
                    )
                    _stamp(tdbg, bx, 23)
                cute.arch.sync_warp()

        # ---- top-k: token t on warp t (keys sigmoid + bias, ties to the lower id), TRTLLM-Gen MiniMax2 ops
        tk_t = cutlass.Int32(-1)  # this warp's top-k token (TK_WARPS[t] = warp)
        for i_tk in cutlass.range_constexpr(M_MAX):
            tk_t = cutlass.select_(warp == cutlass.Int32(TK_WARPS[i_tk]), cutlass.Int32(i_tk), tk_t)
        if tk_t >= cutlass.Int32(0):
            t = tk_t
            want = ltag >> cutlass.Int64(32)
            rbase = (
                (bx % cutlass.Int32(LG_REPL)) * cutlass.Int32(LG_STRIDE)
                + t * cutlass.Int32(E_LOCAL)
                + lane
            )
            done = cutlass.Int32(0)
            res = [cutlass.Int64(0) for _ in range(4)]
            while done == cutlass.Int32(0):
                ok = cutlass.Boolean(True)
                for z in cutlass.range_constexpr(4):
                    res[z] = _ld_relaxed_u64(
                        lgbuf.subview(cutlass.Int32(z * 32) + rbase).data_ptr().toint()
                    )
                    ok = ok & ((res[z] >> cutlass.Int64(32)) == want)
                if prims.vote_sync(0xFFFFFFFF, ok, prims.VoteSync.BALLOT) == cutlass.Int32(-1):
                    done = cutlass.Int32(1)
                else:
                    _nanosleep(POLL_NS)
            if (lane == 0) & (t == cutlass.Int32(0)):
                _stamp(tdbg, bx, 22)
            lgv = [cutlass.Int32(res[z]).bitcast(cutlass.Float32) for z in range(4)]
            keys = []
            for i in cutlass.range_constexpr(4):
                sg = _fma_rn(
                    cutlass.Float32(0.5),
                    cute.math.tanh(_mul_rn(cutlass.Float32(0.5), lgv[i])),
                    cutlass.Float32(0.5),
                )
                keys.append(_add_rn(sg, s_bias.load(idx=i * 32 + lane)))
            hk = [_twiddle(keys[i]) for i in range(4)]
            taken = [cutlass.Boolean(False) for _ in range(4)]
            ids_t = []
            kt_t = []
            for k in cutlass.range_constexpr(TOP_K):
                bh = cutlass.Int32(0)
                bl = cutlass.Int32(0)
                for i in cutlass.range_constexpr(4):
                    lo_i = cutlass.Int32(65535 - i * 32) - lane
                    hv = cutlass.select_(taken[i], cutlass.Int32(0), hk[i])
                    lv = cutlass.select_(taken[i], cutlass.Int32(0), lo_i)
                    better = (cutlass.Uint32(hv) > cutlass.Uint32(bh)) | (
                        (hv == bh) & (cutlass.Uint32(lv) > cutlass.Uint32(bl))
                    )
                    bh = cutlass.select_(better, hv, bh)
                    bl = cutlass.select_(better, lv, bl)
                mh = _redux_max_u32(bh)
                ml = _redux_max_u32(cutlass.select_(bh == mh, bl, cutlass.Int32(0)))
                e_k = cutlass.Int32(65535) - ml
                for i in cutlass.range_constexpr(4):
                    taken[i] = taken[i] | (e_k == cutlass.Int32(i * 32) + lane)
                ids_t.append(e_k)
                kv = cutlass.Float32(0.0)
                for i in cutlass.range_constexpr(4):
                    kv = cutlass.select_(e_k == cutlass.Int32(i * 32) + lane, keys[i], kv)
                kt_t.append(cute.arch.shuffle_sync(kv, e_k % cutlass.Int32(32)))
            sig = cutlass.Float32(0.0)
            e_l = cutlass.Int32(0)
            for k in cutlass.range_constexpr(TOP_K):
                sk = _sub_rn(kt_t[k], s_bias.load(idx=ids_t[k]))
                sig = cutlass.select_(lane == cutlass.Int32(k), sk, sig)
                e_l = cutlass.select_(lane == cutlass.Int32(k), ids_t[k], e_l)
            total = sig
            for off in (16, 8, 4, 2, 1):
                total = _add_rn(total, cute.arch.shuffle_sync_bfly(total, off))
            wv = _bf16_round(
                _div_rn(
                    _mul_rn(sig, cutlass.Float32(ROUTE_SCALE)),
                    _add_rn(total, cutlass.Float32(SUM_EPS)),
                )
            )
            if lane < TOP_K:
                s_w.store(wv, idx=t * cutlass.Int32(TOP_K) + lane)
                s_ids.store(e_l, idx=t * cutlass.Int32(TOP_K) + lane)
                if bx == 0:
                    route_dbg.store(
                        wv.bitcast(cutlass.Int32),
                        idx=t * cutlass.Int32(8) + cutlass.Int32(4) + lane,
                    )
            if (lane == 0) & (t == cutlass.Int32(0)):
                _stamp(tdbg, bx, 20)
            if cutlass.const_expr(PF_PRIO):
                if (t == cutlass.Int32(0)) | (t == cutlass.Int32(PT1)):
                    if lane < TOP_K:
                        # this CTA's 1/NCTA of each of the token's experts into L2 (DRAM is idle until the routing is
                        # complete; every byte is streamed later anyway)
                        _l2_prefetch(
                            w1b.subview(
                                e_l * cutlass.Int32(TWO_I * H // 2) + bx * cutlass.Int32(PF_W1)
                            )
                            .data_ptr()
                            .toint(),
                            PF_W1,
                        )
                        _l2_prefetch(
                            s1w.subview(
                                (e_l * cutlass.Int32(TWO_I * H // 16) + bx * cutlass.Int32(PF_S1))
                                // cutlass.Int32(4)
                            )
                            .data_ptr()
                            .toint(),
                            PF_S1,
                        )
                        _l2_prefetch(
                            w2b.subview(
                                e_l * cutlass.Int32(H * I_TP // 2) + bx * cutlass.Int32(PF_W2)
                            )
                            .data_ptr()
                            .toint(),
                            PF_W2,
                        )
                        _l2_prefetch(
                            s2w.subview(
                                (e_l * cutlass.Int32(H * I_TP // 16) + bx * cutlass.Int32(PF_S2))
                                // cutlass.Int32(4)
                            )
                            .data_ptr()
                            .toint(),
                            PF_S2,
                        )
            cute.arch.sync_warp()
            if lane < TOP_K:  # every writer lane arrives (its own release of s_ids / s_w)
                prims.mbarrier_arrive(topk_bar.subview(0))  # this token's ids / weights are in smem
                if t == cutlass.Int32(0):
                    prims.mbarrier_arrive(
                        tok0_bar.subview(0)
                    )  # token 0's experts: the early group-0 stages
            cute.arch.sync_warp()

        if cutlass.const_expr(EARLY0):
            if warp == W_WARP:
                # slots 0..3 = token 0's four distinct experts; group 0's stage ranges are the same for every G
                _wait(tok0_bar.subview(0), 0)
                n0 = cutlass.Int32(0)
                w0 = s_plan_all.load(idx=G_CAP * (NG_CAP + 1))
                ei0 = w0 & cutlass.Int32(0xFFFF)
                ei1 = w0 >> cutlass.Int32(16)
                for k in range(_nsegs(ei0, ei1)):
                    ug, j0, j1 = _seg(ei0, ei1, k)
                    c = ug - (ug // cutlass.Int32(U1)) * cutlass.Int32(U1)
                    e = s_ids.load(idx=ug // cutlass.Int32(U1))
                    for j in cutlass.range(j0, j1, unroll=1):
                        if n0 < cutlass.Int32(STAGES):
                            _issue_fc1(
                                lane,
                                n0,
                                e,
                                c,
                                j,
                                sA,
                                sSF,
                                ab_full,
                                tma_a1_desc.get_ptr(),
                                tma_s1_desc.get_ptr(),
                                s1w,
                            )
                            n0 = n0 + cutlass.Int32(1)
                if lane == 0:
                    s_misc.store(n0, idx=0)
                cute.arch.sync_warp()
        if warp == X_WARP:
            # slots: entry i < 4 M is (token i / 4, top-k i % 4); distinct experts in entry order
            NCH = (TOP_K * M_MAX + 31) // 32
            _wait(topk_bar.subview(0), 0)
            gcnt = cutlass.Int32(0)
            rec = []  # per chunk: (first, rank, slot rank of the entry, expert, entry valid)
            for ch in cutlass.range_constexpr(NCH):
                ei = lane + cutlass.Int32(32 * ch)
                valid = ei < cutlass.Int32(TOP_K * M_MAX)
                my_e = cutlass.select_(
                    valid,
                    s_ids.load(idx=cutlass.select_(valid, ei, cutlass.Int32(0))),
                    cutlass.Int32(-1) - lane,
                )
                known = cutlass.Boolean(False)
                prev = cutlass.Int32(-1)
                key = my_e
                if cutlass.const_expr(ch > 0):
                    prev = cutlass.select_(
                        valid,
                        s_emap.load(idx=cutlass.select_(valid, my_e, cutlass.Int32(0))),
                        cutlass.Int32(-1),
                    )
                    known = prev >= cutlass.Int32(0)
                    key = cutlass.select_(known, cutlass.Int32(-1000) - lane, my_e)
                    cute.arch.sync_warp()  # every lane's s_emap read before any lane's write
                same = _match_any(key)
                leader = cute.arch.popc((same & (cutlass.Int32(0) - same)) - cutlass.Int32(1))
                first = valid & (known == cutlass.Boolean(False)) & (lane == leader)
                bal = prims.vote_sync(0xFFFFFFFF, first, prims.VoteSync.BALLOT)
                rank = gcnt + cute.arch.popc(bal & cutlass.Int32(cute.arch.lanemask_lt()))
                slot_l = cute.arch.shuffle_sync(rank, leader)
                slot_l = cutlass.select_(known, prev, slot_l)
                if first:
                    s_emap.store(rank, idx=my_e)
                if valid:
                    if bx == 0:
                        route_dbg.store(
                            my_e,
                            idx=(ei // cutlass.Int32(TOP_K)) * cutlass.Int32(8)
                            + ei % cutlass.Int32(TOP_K),
                        )
                rec.append((first, rank, slot_l, my_e, valid))
                cute.arch.sync_warp()
                gcnt = gcnt + cute.arch.popc(bal)
            # slot of rank r: with a partial last group (tail < 4) and >= 3 groups, that group becomes group 1 (its
            # FC2 then is not the last hand-off behind a short FC1 phase); the full groups 1.. shift up by one
            ngr = (gcnt + cutlass.Int32(3)) >> cutlass.Int32(2)
            tail = gcnt - (ngr - cutlass.Int32(1)) * cutlass.Int32(4)
            do_perm = (
                cutlass.Boolean(PERM)
                & (tail < cutlass.Int32(4))
                & (ngr >= cutlass.Int32(PERM_MIN_NG))
            )
            for z in cutlass.range_constexpr((G_CAP + 31) // 32):
                if lane + cutlass.Int32(32 * z) < cutlass.Int32(G_CAP):
                    s_el.store(cutlass.Int32(-1), idx=lane + cutlass.Int32(32 * z))
            cute.arch.sync_warp()
            for ch in cutlass.range_constexpr(NCH):
                first, rank, slot_r, my_e, valid = rec[ch]
                ei = lane + cutlass.Int32(32 * ch)
                for which in cutlass.range_constexpr(2):
                    r = rank if which == 0 else slot_r
                    sl_m = cutlass.select_(
                        r < cutlass.Int32(4),
                        r,
                        cutlass.select_(
                            r >= (ngr - cutlass.Int32(1)) * cutlass.Int32(4),
                            r - (ngr - cutlass.Int32(2)) * cutlass.Int32(4),
                            r + cutlass.Int32(4),
                        ),
                    )
                    sl_m = cutlass.select_(do_perm, sl_m, r)
                    if cutlass.const_expr(which == 0):
                        if first:
                            s_el.store(my_e, idx=sl_m)
                    else:
                        if valid:
                            s_kpos.store(
                                ei % cutlass.Int32(TOP_K),
                                idx=sl_m * cutlass.Int32(M_MAX) + ei // cutlass.Int32(TOP_K),
                            )
            # slots per group
            if lane < cutlass.Int32(NG_CAP):
                nh = gcnt - lane * cutlass.Int32(4)
                nh = cutlass.select_(nh > cutlass.Int32(4), cutlass.Int32(4), nh)
                nh = cutlass.select_(nh < cutlass.Int32(0), cutlass.Int32(0), nh)
                nh = cutlass.select_(do_perm & (lane == cutlass.Int32(1)), tail, nh)
                nh = cutlass.select_(
                    do_perm & (lane == ngr - cutlass.Int32(1)), cutlass.Int32(4), nh
                )
                s_gn.store(nh, idx=lane)
            cute.arch.sync_warp()
            if lane == 0:
                s_el.store(gcnt, idx=G_CAP)
            if lane < cutlass.Int32(NG_CAP + 1):
                s_plan.store(s_plan_all.load(idx=gcnt * cutlass.Int32(NG_CAP + 1) + lane), idx=lane)
            cute.arch.sync_warp()
            if lane == 0:
                _stamp(tdbg, bx, 24)
            prims.mbarrier_arrive(
                route_bar.subview(0)
            )  # every lane: its slots / kpos / plan-row writes are in smem
            cute.arch.sync_warp()
            # CTA 0 pads every group counter to exactly CNT_PER_CALL for this call (absent slots, unused groups)
            if (bx == 0) & (lane < cutlass.Int32(NG_CAP)):
                nh = s_gn.load(idx=lane)
                pad = cutlass.Int32(CNT_PER_CALL) - nh * cutlass.Int32(UNITS_ROW)
                if pad > cutlass.Int32(0):
                    _red_release_add(cnt.subview(lane * cutlass.Int32(8)).data_ptr().toint(), pad)
        else:
            _wait(route_bar.subview(0), 0)
        if tidx == 0:
            _stamp(tdbg, bx, 2)
        if cutlass.const_expr(B1_WARP < 0):
            if warp == cutlass.Int32(0):
                if lane == 0:
                    b1addr = cnt.subview(cutlass.Int32(NG_CAP * 8)).data_ptr().toint()
                    b1tgt = (ep + cutlass.Int32(1)) * cutlass.Int32(NCTA)
                    cv = _ld_acquire_u32(b1addr)
                    while (cv - b1tgt) < cutlass.Int32(0):
                        _nanosleep(64)
                        cv = _ld_acquire_u32(b1addr)
                    _fence_proxy_async_global()
                    prims.mbarrier_arrive_expect_tx(
                        b1_load.subview(0), S1 * NUM_BYTES_B1 + M_MAX * (H // 16)
                    )
                    prims.cp_async_bulk_shared_cluster_global(
                        sB1.subview(0),
                        img32.subview(cutlass.Int32(B1IMG_OFF // 4)),
                        b1_load.subview(0),
                        S1 * NUM_BYTES_B1,
                    )
                    prims.cp_async_bulk_shared_cluster_global(
                        s_xsf.subview(0),
                        img32.subview(cutlass.Int32((B1IMG_OFF + S1 * NUM_BYTES_B1) // 4)),
                        b1_load.subview(0),
                        M_MAX * (H // 16),
                    )
                    _stamp(tdbg, bx, 23)
                cute.arch.sync_warp()
        if warp < 4:  # B1 scales to TMEM (the MMA needs them only with the first ring stage)
            _wait(b1_load.subview(0), 0)
            _sfb1_to_tmem(warp, lane, base_row, base_col, s_xsf, b1_ready)

        g_n = s_el.load(idx=G_CAP)
        ng = (g_n + cutlass.Int32(3)) >> cutlass.Int32(2)
        pr2 = s_plan.load(idx=NG_CAP)
        blk0 = pr2 & cutlass.Int32(0xFFFF)
        nb = pr2 >> cutlass.Int32(16)

        # ---- weights producer (warp 4): phase p = FC1 stages of group p, then FC2 tiles of group p - LAG
        if warp == W_WARP:
            stage = cutlass.Int32(0)
            phase = cutlass.Int32(1)
            skip = cutlass.Int32(0)
            n_iss = cutlass.Int32(
                0
            )  # stages issued (early ones included): the L2-prefetch warp's throttle
            w_emp = cutlass.Int64(0)
            if cutlass.const_expr(EARLY0):
                skip = s_misc.load(idx=0)
                stage = cutlass.select_(skip == cutlass.Int32(STAGES), cutlass.Int32(0), skip)
                phase = cutlass.select_(
                    skip == cutlass.Int32(STAGES), cutlass.Int32(0), cutlass.Int32(1)
                )
            for p in range(ng + cutlass.Int32(LAG)):
                if p < ng:
                    i0, i1 = _plan_range(s_plan, p)
                    for k in range(_nsegs(i0, i1)):
                        ug, j0, j1 = _seg(i0, i1, k)
                        slot = p * cutlass.Int32(4) + ug // cutlass.Int32(U1)
                        c = ug - (ug // cutlass.Int32(U1)) * cutlass.Int32(U1)
                        e = s_el.load(idx=slot)
                        for j in cutlass.range(j0, j1, unroll=1):
                            if skip > cutlass.Int32(0):
                                skip = skip - cutlass.Int32(1)
                            else:
                                w_emp = _waitt(ab_empty.subview(stage), phase, w_emp)
                                prims.cp_async_wait_group(STAGES - 1)
                                _dbg_delay(bx)
                                _issue_fc1(
                                    lane,
                                    stage,
                                    e,
                                    c,
                                    j,
                                    sA,
                                    sSF,
                                    ab_full,
                                    tma_a1_desc.get_ptr(),
                                    tma_s1_desc.get_ptr(),
                                    s1w,
                                )
                                stage, phase = _adv(stage, phase)
                            if cutlass.const_expr(PFD > 0):
                                n_iss = n_iss + cutlass.Int32(1)
                                if lane == 0:
                                    prims.store_ext(
                                        n_iss.ir_value(),
                                        s_prog.subview(0),
                                        order="relaxed",
                                        scope="cta",
                                    )
                if p >= cutlass.Int32(LAG):
                    g = p - cutlass.Int32(LAG)
                    n_here = s_gn.load(idx=g)
                    for bb in range(nb):
                        blk = blk0 + bb
                        for t2 in cutlass.range_constexpr(KT2):
                            w_emp = _waitt(ab_empty.subview(stage), phase, w_emp)
                            prims.cp_async_wait_group(STAGES - 1)
                            _issue_fc2(
                                lane,
                                stage,
                                g,
                                blk,
                                t2,
                                n_here,
                                s_el,
                                sA,
                                sSF,
                                ab_full,
                                tma_a2_desc.get_ptr(),
                                tma_s2_desc.get_ptr(),
                                s2w,
                            )
                            stage, phase = _adv(stage, phase)
                            if cutlass.const_expr(PFD > 0):
                                n_iss = n_iss + cutlass.Int32(1)
                                if lane == 0:
                                    prims.store_ext(
                                        n_iss.ir_value(),
                                        s_prog.subview(0),
                                        order="relaxed",
                                        scope="cta",
                                    )
            if lane == 0:
                _stamp(tdbg, bx, 3)
                _tstore(tdbg, bx, 30, w_emp)

        # ---- (SFA_CP = 0) scale splicer (warp 6): atom kb word [r][q] = raw atom 4 q + kb word [r][m1]
        if cutlass.const_expr(not SFA_CP):
            if warp == S_WARP:
                s2t_shape, s2t_mc = prims.S2TCopyMode.S2T_32x128b_WARPX4
                stage = cutlass.Int32(0)
                phase = cutlass.Int32(0)
                for p in range(ng + cutlass.Int32(LAG)):
                    i0 = cutlass.Int32(0)
                    i1 = cutlass.Int32(0)
                    for part in cutlass.range_constexpr(2):
                        cnt_ = cutlass.Int32(0)
                        m1 = cutlass.Int32(0)
                        nk = cutlass.Int32(0)
                        if cutlass.const_expr(part == 0):
                            if p < ng:
                                i0, i1 = _plan_range(s_plan, p)
                                nk = _nsegs(i0, i1)
                        else:
                            if p >= cutlass.Int32(LAG):
                                nk = nb
                        for k in range(nk):
                            if cutlass.const_expr(part == 0):
                                ug, j0, j1 = _seg(i0, i1, k)
                                m1 = (
                                    ug - (ug // cutlass.Int32(U1)) * cutlass.Int32(U1)
                                ) & cutlass.Int32(3)
                                cnt_ = j1 - j0
                            else:
                                m1 = (blk0 + k) & cutlass.Int32(3)
                                cnt_ = cutlass.Int32(KT2)
                            for _j in cutlass.range(cnt_, unroll=1):
                                _waitd(ab_full.subview(stage), phase, tdbg, 3)
                                raw0 = (
                                    stage * cutlass.Int32(NUM_WORDS_SFA_RAW)
                                    + lane * cutlass.Int32(4)
                                    + m1
                                )
                                dst0 = stage * cutlass.Int32(NUM_WORDS_SFA) + lane * cutlass.Int32(
                                    4
                                )
                                for kb in cutlass.range_constexpr(NUM_KB):
                                    w0 = sSF.load(idx=raw0 + cutlass.Int32((0 * 4 + kb) * 128))
                                    w1 = sSF.load(idx=raw0 + cutlass.Int32((1 * 4 + kb) * 128))
                                    w2 = sSF.load(idx=raw0 + cutlass.Int32((2 * 4 + kb) * 128))
                                    w3 = sSF.load(idx=raw0 + cutlass.Int32((3 * 4 + kb) * 128))
                                    sSFA.store(
                                        (
                                            cutlass.Int32(w0),
                                            cutlass.Int32(w1),
                                            cutlass.Int32(w2),
                                            cutlass.Int32(w3),
                                        ),
                                        idx=dst0 + cutlass.Int32(kb * 128),
                                        alignment=16,
                                    )
                                prims.fence_proxy(
                                    "async_shared", space=prims.SharedSpace.shared_cta
                                )
                                cute.arch.sync_warp()
                                if prims.elect_sync():
                                    sd0 = prims.Tcgen05SmemDesc.build(
                                        sSFA.subview(stage * cutlass.Int32(NUM_WORDS_SFA)),
                                        leading_byte_offset=16,
                                        stride_byte_offset=128,
                                        base_offset=0,
                                        layout=0,
                                    )
                                    for kb in cutlass.range_constexpr(NUM_KB):
                                        prims.tcgen05_cp(
                                            s2t_shape,
                                            _tptr(
                                                base_row,
                                                base_col
                                                + SFA1_COL
                                                + stage * cutlass.Int32(16)
                                                + kb * 4,
                                            ),
                                            sd0 + (kb * 512 >> 4),
                                            multicast=s2t_mc,
                                        )
                                    prims.tcgen05_commit(sfa_ready.subview(stage))
                                stage, phase = _adv(stage, phase)

        # ---- FC2 B loader (warp 5): group g's image once its counter is complete
        if warp == X_WARP:
            s2t_shape, s2t_mc = prims.S2TCopyMode.S2T_32x128b_WARPX4
            for g in range(ng):
                buf = g % cutlass.Int32(NB2BUF)
                if g >= cutlass.Int32(NB2BUF):
                    _waitd(
                        b2_free.subview(buf),
                        ((g // cutlass.Int32(NB2BUF)) - cutlass.Int32(1)) & cutlass.Int32(1),
                        tdbg,
                        101,
                    )
                if (lane == 0) & (g == ng - cutlass.Int32(1)):
                    _stamp(tdbg, bx, 29)
                caddr = cnt.subview(g * cutlass.Int32(8)).data_ptr().toint()
                if lane == 0:
                    cv = _ld_acquire_u32(caddr)
                    while (cv - cnt_target) < cutlass.Int32(0):
                        _nanosleep(B2_POLL_NS)
                        cv = _ld_acquire_u32(caddr)
                cute.arch.sync_warp()
                if (lane == 0) & (g == ng - cutlass.Int32(1)):
                    _stamp(tdbg, bx, 51)
                _fence_proxy_async_global()
                if g >= cutlass.Int32(1):
                    _waitd(
                        sfb2_free.subview(0), (g - cutlass.Int32(1)) & cutlass.Int32(1), tdbg, 107
                    )
                if lane == 0:
                    prims.mbarrier_arrive_expect_tx(img_bar.subview(buf), IMG_BYTES)
                    prims.cp_async_bulk_shared_cluster_global(
                        sB2.subview(buf * cutlass.Int32(IMG_B2)),
                        img32.subview(g * cutlass.Int32(IMG_BYTES // 4)),
                        img_bar.subview(buf),
                        IMG_B2,
                    )
                    prims.cp_async_bulk_shared_cluster_global(
                        sSFB2.subview(0),
                        img32.subview(
                            g * cutlass.Int32(IMG_BYTES // 4) + cutlass.Int32(IMG_B2 // 4)
                        ),
                        img_bar.subview(buf),
                        IMG_SF,
                    )
                cute.arch.sync_warp()
                _waitd(
                    img_bar.subview(buf), (g // cutlass.Int32(NB2BUF)) & cutlass.Int32(1), tdbg, 108
                )
                if (lane == 0) & (g == ng - cutlass.Int32(1)):
                    _stamp(tdbg, bx, 52)
                if g >= cutlass.Int32(1):
                    # the single TMEM scale slot: the previous group's FC2 MMAs are done
                    gp = g - cutlass.Int32(1)
                    _waitd(
                        b2_free.subview(gp % cutlass.Int32(NB2BUF)),
                        (gp // cutlass.Int32(NB2BUF)) & cutlass.Int32(1),
                        tdbg,
                        109,
                    )
                if (lane == 0) & (g == ng - cutlass.Int32(1)):
                    _stamp(tdbg, bx, 53)
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                if prims.elect_sync():
                    sd0 = prims.Tcgen05SmemDesc.build(
                        sSFB2.subview(0),
                        leading_byte_offset=16,
                        stride_byte_offset=128,
                        base_offset=0,
                        layout=0,
                    )
                    for aa in cutlass.range_constexpr(KT2 * NUM_KB):
                        prims.tcgen05_cp(
                            s2t_shape,
                            _tptr(base_row, base_col + SFB2_COL + aa * 4),
                            sd0 + cutlass.Int64(aa * (512 >> 4)),
                            multicast=s2t_mc,
                        )
                    prims.tcgen05_commit(b2_ready.subview(buf))
                    prims.tcgen05_commit(sfb2_free.subview(0))
                cute.arch.sync_warp()
                if lane == 0:
                    _stamp(tdbg, bx, 25)

        # ---- (PFD) L2 prefetch warp (12): the same stage sequence as the producer, PFD stages ahead of it
        if cutlass.const_expr(PFD > 0):
            if warp == cutlass.Int32(PF_WARP):
                pf = cutlass.Int32(0)
                for p in range(ng + cutlass.Int32(LAG)):
                    if p < ng:
                        i0, i1 = _plan_range(s_plan, p)
                        for k in range(_nsegs(i0, i1)):
                            ug, j0, j1 = _seg(i0, i1, k)
                            slot = p * cutlass.Int32(4) + ug // cutlass.Int32(U1)
                            c = ug - (ug // cutlass.Int32(U1)) * cutlass.Int32(U1)
                            e = s_el.load(idx=slot)
                            for j in cutlass.range(j0, j1, unroll=1):
                                if lane == 0:
                                    prg = cutlass.Int32(
                                        prims.load_ext(
                                            s_prog.subview(0),
                                            dtype=cutlass.Int32,
                                            order="relaxed",
                                            scope="cta",
                                        )
                                    )
                                    while prg + cutlass.Int32(PFD) <= pf:
                                        _nanosleep(100)
                                        prg = cutlass.Int32(
                                            prims.load_ext(
                                                s_prog.subview(0),
                                                dtype=cutlass.Int32,
                                                order="relaxed",
                                                scope="cta",
                                            )
                                        )
                                    if pf >= prg:
                                        kx = j * cutlass.Int32(C1 * KT_BYTES)
                                        for q in cutlass.range_constexpr(C1):
                                            prims.cp_async_bulk_tensor_prefetch(
                                                tma_a1_desc.get_ptr(),
                                                [
                                                    kx + cutlass.Int32(q * KT_BYTES),
                                                    c * cutlass.Int32(R1),
                                                    e,
                                                ],
                                                [],
                                            )
                                        atom0 = (
                                            e * cutlass.Int32(SF1_RB) + (c >> cutlass.Int32(2))
                                        ) * cutlass.Int32(SF1_KA) + j * cutlass.Int32(16)
                                        _l2_prefetch(
                                            s1w.subview(atom0 * cutlass.Int32(128))
                                            .data_ptr()
                                            .toint(),
                                            8192,
                                        )
                                cute.arch.sync_warp()
                                pf = pf + cutlass.Int32(1)
                    if p >= cutlass.Int32(LAG):
                        g = p - cutlass.Int32(LAG)
                        n_here = s_gn.load(idx=g)
                        for bb in range(nb):
                            blk = blk0 + bb
                            for t2 in cutlass.range_constexpr(KT2):
                                if lane == 0:
                                    prg = cutlass.Int32(
                                        prims.load_ext(
                                            s_prog.subview(0),
                                            dtype=cutlass.Int32,
                                            order="relaxed",
                                            scope="cta",
                                        )
                                    )
                                    while prg + cutlass.Int32(PFD) <= pf:
                                        _nanosleep(100)
                                        prg = cutlass.Int32(
                                            prims.load_ext(
                                                s_prog.subview(0),
                                                dtype=cutlass.Int32,
                                                order="relaxed",
                                                scope="cta",
                                            )
                                        )
                                    if pf >= prg:
                                        for jj in cutlass.range_constexpr(4):
                                            if cutlass.Int32(jj) < n_here:
                                                e = s_el.load(
                                                    idx=g * cutlass.Int32(4) + cutlass.Int32(jj)
                                                )
                                                prims.cp_async_bulk_tensor_prefetch(
                                                    tma_a2_desc.get_ptr(),
                                                    [
                                                        cutlass.Int32(t2 * KT_BYTES),
                                                        blk * cutlass.Int32(R2),
                                                        e,
                                                    ],
                                                    [],
                                                )
                                                a2 = (
                                                    e * cutlass.Int32(SF2_RB)
                                                    + (blk >> cutlass.Int32(2))
                                                ) * cutlass.Int32(SF2_KA) + cutlass.Int32(4 * t2)
                                                _l2_prefetch(
                                                    s2w.subview(a2 * cutlass.Int32(128))
                                                    .data_ptr()
                                                    .toint(),
                                                    2048,
                                                )
                                cute.arch.sync_warp()
                                pf = pf + cutlass.Int32(1)

        # ---- (REL_WARP) group releases (warp 8): this CTA's finished units of group p, one release add per phase
        if cutlass.const_expr(REL_WARP):
            if warp == cutlass.Int32(8):
                for p in range(ng):
                    i0, i1 = _plan_range(s_plan, p)
                    n_done = cutlass.Int32(0)
                    for k in range(_nsegs(i0, i1)):
                        ug, j0, j1 = _seg(i0, i1, k)
                        n_done = n_done + cutlass.select_(
                            (j0 != cutlass.Int32(0)) | (j1 == cutlass.Int32(S1)),
                            cutlass.Int32(1),
                            cutlass.Int32(0),
                        )
                    _wait(
                        grp_done.subview(p & cutlass.Int32(3)),
                        (p >> cutlass.Int32(2)) & cutlass.Int32(1),
                    )
                    _dbg_delay(bx * cutlass.Int32(13) + p)
                    if (lane == 0) & (n_done > cutlass.Int32(0)):
                        if cutlass.const_expr(EPI_PFENCE):
                            _fence_proxy_async_global()
                        _red_release_add(
                            cnt.subview(p * cutlass.Int32(8)).data_ptr().toint(), n_done
                        )
                    if (lane == 0) & (p == ng - cutlass.Int32(1)):
                        _stamp(tdbg, bx, 54)
                    cute.arch.sync_warp()

        # ---- MMA (warp 7)
        if warp == MMA_WARP:
            s2t_shape, s2t_mc = prims.S2TCopyMode.S2T_32x128b_WARPX4
            idesc1 = prims.Tcgen05MxOmmaInstrDesc.build(
                a_dtype=cutlass.Float4E2M1FN,
                b_dtype=cutlass.Float4E2M1FN,
                scale_format=0,
                n_dim=N1,
                m_dim=MMA_M,
                sparsity_version=0,
            )
            idesc2 = prims.Tcgen05MxOmmaInstrDesc.build(
                a_dtype=cutlass.Float4E2M1FN,
                b_dtype=cutlass.Float4E2M1FN,
                scale_format=0,
                n_dim=N2,
                m_dim=MMA_M,
                sparsity_version=0,
            )
            desc_a0 = prims.Tcgen05SmemDesc.build(
                sA.subview(0),
                leading_byte_offset=16,
                stride_byte_offset=1024,
                base_offset=0,
                layout=2,
            )
            desc_b10 = prims.Tcgen05SmemDesc.build(
                sB1.subview(0),
                leading_byte_offset=16,
                stride_byte_offset=1024,
                base_offset=0,
                layout=2,
            )
            desc_b20 = prims.Tcgen05SmemDesc.build(
                sB2.subview(0),
                leading_byte_offset=16,
                stride_byte_offset=1024,
                base_offset=0,
                layout=2,
            )
            tcol = (base_row << 16) | base_col
            _waitd(b1_ready, 0, tdbg, 5)
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            stage = cutlass.Int32(0)
            phase = cutlass.Int32(0)
            k1 = cutlass.Int32(0)
            k2 = cutlass.Int32(0)
            m_b2 = cutlass.Int64(0)
            m_full = cutlass.Int64(0)
            m_acc = cutlass.Int64(0)
            for p in range(ng + cutlass.Int32(LAG)):
                if p < ng:
                    i0, i1 = _plan_range(s_plan, p)
                    for k in range(_nsegs(i0, i1)):
                        ug, j0, j1 = _seg(i0, i1, k)
                        ab = k1 & cutlass.Int32(1)
                        m_acc = _waitt(
                            acc_empty.subview(ab),
                            ((k1 >> cutlass.Int32(1)) & cutlass.Int32(1)) ^ cutlass.Int32(1),
                            m_acc,
                        )
                        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                        acc_ptr = cutlass.inttoptr(
                            tcol + ACC1_COL + ab * cutlass.Int32(N1), 6, cutlass.Float32
                        )
                        for j in cutlass.range(j0, j1, unroll=1):
                            if cutlass.const_expr(SFA_CP):
                                m_full = _waitt(ab_full.subview(stage), phase, m_full)
                                prims.fence_proxy(
                                    "async_shared", space=prims.SharedSpace.shared_cta
                                )
                            else:
                                _waitd(sfa_ready.subview(stage), phase, tdbg, 6)
                            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                            da = desc_a0 + cutlass.Int64(stage * cutlass.Int32(NUM_BYTES_A >> 4))
                            db = desc_b10 + cutlass.Int64(j * cutlass.Int32(NUM_BYTES_B1 >> 4))
                            sfa_c = tcol + SFA1_COL + stage * cutlass.Int32(16)
                            sfb_c = tcol + SFB1_COL + j * cutlass.Int32(NUM_KB * 4)
                            if prims.elect_sync():
                                if cutlass.const_expr(SFA_CP):
                                    sd0 = prims.Tcgen05SmemDesc.build(
                                        sSFA.subview(stage * cutlass.Int32(NUM_WORDS_SFA)),
                                        leading_byte_offset=16,
                                        stride_byte_offset=128,
                                        base_offset=0,
                                        layout=0,
                                    )
                                    for kb in cutlass.range_constexpr(NUM_KB):
                                        prims.tcgen05_cp(
                                            s2t_shape,
                                            _tptr(
                                                base_row,
                                                base_col
                                                + SFA1_COL
                                                + stage * cutlass.Int32(16)
                                                + kb * 4,
                                            ),
                                            sd0 + (kb * 512 >> 4),
                                            multicast=s2t_mc,
                                        )
                                for kb in cutlass.range_constexpr(NUM_KB):
                                    prims.tcgen05_mma_block_scale(
                                        prims.MMABlockScaleKind.MXF4NVF4,
                                        prims.CTAGroup.CTA_1,
                                        acc_ptr,
                                        da + INC_K * kb,
                                        db + INC_K * kb,
                                        idesc1,
                                        (j > j0) | cutlass.Boolean(kb > 0),
                                        cutlass.inttoptr(sfa_c + kb * 4, 6, cutlass.Int32),
                                        cutlass.inttoptr(sfb_c + kb * 4, 6, cutlass.Int32),
                                        scale_vec_size=prims.Tcgen05MMABlockScale.BLOCK16,
                                    )
                                prims.tcgen05_commit(ab_empty.subview(stage))
                            cute.arch.sync_warp()
                            stage, phase = _adv(stage, phase)
                        if prims.elect_sync():
                            prims.tcgen05_commit(acc_full.subview(ab))
                        cute.arch.sync_warp()
                        k1 = k1 + cutlass.Int32(1)
                if p >= cutlass.Int32(LAG):
                    g = p - cutlass.Int32(LAG)
                    buf = g % cutlass.Int32(NB2BUF)
                    m_b2_0 = m_b2
                    m_b2 = _waitt(
                        b2_ready.subview(buf), (g // cutlass.Int32(NB2BUF)) & cutlass.Int32(1), m_b2
                    )
                    if cutlass.const_expr(TIMING):
                        if (lane == 0) & (g < cutlass.Int32(4)):
                            tdbg.store(
                                m_b2 - m_b2_0,
                                idx=bx * cutlass.Int32(TD_STRIDE) + cutlass.Int32(44) + g,
                            )
                    if (lane == 0) & (g == ng - cutlass.Int32(1)):
                        _stamp(tdbg, bx, 4)
                    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                    for bb in range(nb):
                        a2 = k2 & cutlass.Int32(1)
                        m_acc = _waitt(
                            acc2_empty.subview(a2),
                            ((k2 >> cutlass.Int32(1)) & cutlass.Int32(1)) ^ cutlass.Int32(1),
                            m_acc,
                        )
                        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                        acc2_ptr = cutlass.inttoptr(
                            tcol + ACC2_COL + a2 * cutlass.Int32(ACC2_STRIDE), 6, cutlass.Float32
                        )
                        for t2 in cutlass.range_constexpr(KT2):
                            if cutlass.const_expr(SFA_CP):
                                m_full = _waitt(ab_full.subview(stage), phase, m_full)
                                prims.fence_proxy(
                                    "async_shared", space=prims.SharedSpace.shared_cta
                                )
                            else:
                                _waitd(sfa_ready.subview(stage), phase, tdbg, 7)
                            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                            da = desc_a0 + cutlass.Int64(stage * cutlass.Int32(NUM_BYTES_A >> 4))
                            db = desc_b20 + cutlass.Int64(
                                (buf * cutlass.Int32(KT2) + cutlass.Int32(t2))
                                * cutlass.Int32(NUM_BYTES_B2 >> 4)
                            )
                            sfa_c = tcol + SFA1_COL + stage * cutlass.Int32(16)
                            sfb_c = tcol + SFB2_COL + cutlass.Int32(t2 * NUM_KB * 4)
                            if prims.elect_sync():
                                if cutlass.const_expr(SFA_CP):
                                    sd0 = prims.Tcgen05SmemDesc.build(
                                        sSFA.subview(stage * cutlass.Int32(NUM_WORDS_SFA)),
                                        leading_byte_offset=16,
                                        stride_byte_offset=128,
                                        base_offset=0,
                                        layout=0,
                                    )
                                    for kb in cutlass.range_constexpr(NUM_KB):
                                        prims.tcgen05_cp(
                                            s2t_shape,
                                            _tptr(
                                                base_row,
                                                base_col
                                                + SFA1_COL
                                                + stage * cutlass.Int32(16)
                                                + kb * 4,
                                            ),
                                            sd0 + (kb * 512 >> 4),
                                            multicast=s2t_mc,
                                        )
                                for kb in cutlass.range_constexpr(NUM_KB):
                                    prims.tcgen05_mma_block_scale(
                                        prims.MMABlockScaleKind.MXF4NVF4,
                                        prims.CTAGroup.CTA_1,
                                        acc2_ptr,
                                        da + INC_K * kb,
                                        db + INC_K * kb,
                                        idesc2,
                                        cutlass.Boolean(t2 > 0 or kb > 0),
                                        cutlass.inttoptr(sfa_c + kb * 4, 6, cutlass.Int32),
                                        cutlass.inttoptr(sfb_c + kb * 4, 6, cutlass.Int32),
                                        scale_vec_size=prims.Tcgen05MMABlockScale.BLOCK16,
                                    )
                                prims.tcgen05_commit(ab_empty.subview(stage))
                            cute.arch.sync_warp()
                            stage, phase = _adv(stage, phase)
                        if prims.elect_sync():
                            prims.tcgen05_commit(acc2_full.subview(a2))
                        cute.arch.sync_warp()
                        k2 = k2 + cutlass.Int32(1)
                    if prims.elect_sync():
                        prims.tcgen05_commit(
                            b2_free.subview(buf)
                        )  # this group's B2 tile and TMEM scales are free
                    cute.arch.sync_warp()
            if lane == 0:
                _tstore(tdbg, bx, 26, m_b2)
                _tstore(tdbg, bx, 27, m_full)
                _tstore(tdbg, bx, 28, m_acc)

        # ---- epilogue warps (0-3)
        if warp < 4:
            row_w = base_row + warp * 32
            img8 = cutlass.Array(
                img32.data_ptr(0), shape=(NG_CAP * IMG_BYTES,), dtype=cutlass.Int8, alignment=16
            )
            for r in cutlass.range_constexpr((6 * G_CAP + EPI_THREADS - 1) // EPI_THREADS):
                qi = tidx + cutlass.Int32(r * EPI_THREADS)
                sl_e = qi // cutlass.Int32(6)
                if qi < cutlass.Int32(6 * G_CAP):
                    e_sl = s_el.load(idx=sl_e)
                    if e_sl >= cutlass.Int32(0):
                        fld = qi - sl_e * cutlass.Int32(6)
                        s_esc.store(
                            cutlass.Float32(escal.load(idx=fld * cutlass.Int32(E_LOCAL) + e_sl)),
                            idx=sl_e * cutlass.Int32(8) + fld,
                        )
            cute.arch.sync_warp()
            cute.arch.barrier(barrier_id=EPI_BAR, number_of_threads=EPI_THREADS)
            k1 = cutlass.Int32(0)
            k2 = cutlass.Int32(0)
            e_wait = cutlass.Int64(0)
            e_busy = cutlass.Int64(0)
            e_a = cutlass.Int64(0)
            e_pw = cutlass.Int64(0)
            for p in range(ng + cutlass.Int32(LAG)):
                if p < ng:
                    i0, i1 = _plan_range(s_plan, p)
                    for k in range(_nsegs(i0, i1)):
                        ug, j0, j1 = _seg(i0, i1, k)
                        ab = k1 & cutlass.Int32(1)
                        slot = p * cutlass.Int32(4) + ug // cutlass.Int32(U1)
                        cblk = ug - (ug // cutlass.Int32(U1)) * cutlass.Int32(U1)
                        unit = slot * cutlass.Int32(U1) + cblk
                        unit_done = (j0 != cutlass.Int32(0)) | (j1 == cutlass.Int32(S1))
                        e_wait = _waitt(
                            acc_full.subview(ab),
                            (k1 >> cutlass.Int32(1)) & cutlass.Int32(1),
                            e_wait,
                        )
                        tb0 = _gtime() if cutlass.const_expr(TIMING) else cutlass.Int64(0)
                        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                        acc = prims.tcgen05_ld(
                            "32x32b",
                            _tptr(
                                row_w, base_col + ACC1_COL + ab * cutlass.Int32(N1), cutlass.Float32
                            ),
                            num=N1_LD,
                        )
                        prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                        prims.mbarrier_arrive(acc_empty.subview(ab))
                        for t in cutlass.range_constexpr(M_MAX):
                            part = cutlass.Float32(0.0)
                            for w in cutlass.range_constexpr(4):
                                part = cutlass.select_(
                                    warp == cutlass.Int32(w), cutlass.Float32(acc[QS * w + t]), part
                                )
                            s_part.store(part, idx=(t * C1 + warp) * R1 + lane)
                        cute.arch.sync_warp()
                        cute.arch.barrier(barrier_id=EPI_BAR, number_of_threads=EPI_THREADS)
                        tc0 = _gtime() if cutlass.const_expr(TIMING) else cutlass.Int64(0)
                        if cutlass.const_expr(TIMING):
                            e_a = e_a + (tc0 - tb0)
                        g_u = slot >> cutlass.Int32(2)
                        jrow = slot & cutlass.Int32(3)
                        # this warp's tokens t = warp + 4 tr, all in flight together (latency-bound otherwise)
                        NT = (M_MAX + 3) // 4
                        xr = cute.make_rmem_tensor((NT,), cutlass.Float32)
                        for tr in cutlass.range_constexpr(NT):
                            tq = cutlass.select_(
                                warp + cutlass.Int32(4 * tr) < cutlass.Int32(M_MAX),
                                warp + cutlass.Int32(4 * tr),
                                cutlass.Int32(0),
                            )
                            xr[tr] = _add_rn(
                                _add_rn(
                                    _add_rn(
                                        s_part.load(idx=(tq * C1 + 0) * R1 + lane),
                                        s_part.load(idx=(tq * C1 + 1) * R1 + lane),
                                    ),
                                    s_part.load(idx=(tq * C1 + 2) * R1 + lane),
                                ),
                                s_part.load(idx=(tq * C1 + 3) * R1 + lane),
                            )
                        if (j0 == cutlass.Int32(0)) & (j1 != cutlass.Int32(S1)):
                            # head of a unit split with the next CTA: its partial sums
                            for tr in cutlass.range_constexpr(NT):
                                t = warp + cutlass.Int32(4 * tr)
                                if t < cutlass.Int32(M_MAX):
                                    _st_relaxed_u32(
                                        partials.subview(
                                            (unit * cutlass.Int32(M_MAX) + t) * cutlass.Int32(R1)
                                            + lane
                                        )
                                        .data_ptr()
                                        .toint(),
                                        xr[tr].bitcast(cutlass.Int32),
                                    )
                        else:
                            if j0 != cutlass.Int32(0):
                                # tail: the head's partial sums (head first, as TRTLLM-Gen's K order)
                                for tr in cutlass.range_constexpr(NT):
                                    t = warp + cutlass.Int32(4 * tr)
                                    if t < cutlass.Int32(M_MAX):
                                        paddr = (
                                            partials.subview(
                                                (unit * cutlass.Int32(M_MAX) + t)
                                                * cutlass.Int32(R1)
                                                + lane
                                            )
                                            .data_ptr()
                                            .toint()
                                        )
                                        pbits = _ld_relaxed_u32(paddr)
                                        while pbits == cutlass.Int32(PARTIAL_EMPTY):
                                            pbits = _ld_relaxed_u32(paddr)
                                        _st_relaxed_u32(paddr, cutlass.Int32(PARTIAL_EMPTY))
                                        xr[tr] = _add_rn(pbits.bitcast(cutlass.Float32), xr[tr])
                                cute.arch.sync_warp()
                            _dbg_delay(bx * cutlass.Int32(7) + warp)
                            es = slot * cutlass.Int32(8)
                            ea = s_esc.load(idx=es)
                            ec = s_esc.load(idx=es + cutlass.Int32(1))
                            eal = s_esc.load(idx=es + cutlass.Int32(2))
                            ebe = s_esc.load(idx=es + cutlass.Int32(3))
                            eli = s_esc.load(idx=es + cutlass.Int32(4))
                            is_up = (
                                (lane >> cutlass.Int32(3)) & cutlass.Int32(1)
                            ) == cutlass.Int32(0)
                            n0 = jrow * cutlass.Int32(M_MAX)
                            t2 = cblk >> cutlass.Int32(4)
                            cc = (cblk >> cutlass.Int32(1)) & cutlass.Int32(7)
                            gbase = g_u * cutlass.Int32(IMG_BYTES)
                            sfoff = (
                                gbase
                                + cutlass.Int32(IMG_B2)
                                + (
                                    (
                                        t2 * cutlass.Int32(NUM_KB)
                                        + ((cblk & cutlass.Int32(15)) >> cutlass.Int32(2))
                                    )
                                    * cutlass.Int32(512)
                                )
                                + (cblk & cutlass.Int32(3))
                            )
                            for tr in cutlass.range_constexpr(NT):
                                # (skip) only tokens routed to this slot (warp-uniform); other B2 rows are never used
                                if s_kpos.load(
                                    idx=slot * cutlass.Int32(M_MAX)
                                    + cutlass.select_(
                                        warp + cutlass.Int32(4 * tr) < cutlass.Int32(M_MAX),
                                        warp + cutlass.Int32(4 * tr),
                                        cutlass.Int32(0),
                                    )
                                ) >= cutlass.Int32(0):
                                    xv = xr[tr]
                                    gate_raw = cute.arch.shuffle_sync_bfly(xv, 8)
                                    qv, sfb = _swiglu_quant(
                                        xv, gate_raw, is_up, ea, ec, eal, ebe, eli
                                    )
                                    byte = _e2m1x2(cute.arch.shuffle_sync_down(qv, 16), qv)
                                    bt = [
                                        cute.arch.shuffle_sync(byte, i) & cutlass.Int32(0xFF)
                                        for i in range(8)
                                    ]
                                    lo = (
                                        bt[0]
                                        | (bt[1] << cutlass.Int32(8))
                                        | (bt[2] << cutlass.Int32(16))
                                        | (bt[3] << cutlass.Int32(24))
                                    )
                                    hi = (
                                        bt[4]
                                        | (bt[5] << cutlass.Int32(8))
                                        | (bt[6] << cutlass.Int32(16))
                                        | (bt[7] << cutlass.Int32(24))
                                    )
                                    t = warp + cutlass.Int32(4 * tr)
                                    if (
                                        (lane == 0)
                                        & (t < cutlass.Int32(M_MAX))
                                        & cutlass.Boolean(not DBG_NOSTORE)
                                    ):
                                        # B2 row n = j M + t, k-tile c / 16, 16-byte chunk (c / 2) % 8
                                        # (swizzled), half c % 2; scale: atom (t2, kb = (c % 16) / 4), row n
                                        # at byte 16 (n % 32) + 4 (n / 32), byte c % 4
                                        n = n0 + t
                                        img32.store(
                                            (lo, hi),
                                            idx=(
                                                gbase
                                                + t2 * cutlass.Int32(NUM_BYTES_B2)
                                                + n * cutlass.Int32(KT_BYTES)
                                                + (cc ^ (n & cutlass.Int32(7))) * cutlass.Int32(16)
                                                + (cblk & cutlass.Int32(1)) * cutlass.Int32(8)
                                            )
                                            // cutlass.Int32(4),
                                            alignment=8,
                                        )
                                        img8.store(
                                            cutlass.Int8(sfb & cutlass.Int32(0xFF)),
                                            idx=sfoff
                                            + (n & cutlass.Int32(31)) * cutlass.Int32(16)
                                            + (n >> cutlass.Int32(5)) * cutlass.Int32(4),
                                        )
                        tc1 = _gtime() if cutlass.const_expr(TIMING) else cutlass.Int64(0)
                        if cutlass.const_expr(TIMING):
                            e_pw = e_pw + (tc1 - tc0)
                        if cutlass.const_expr(not REL_WARP):
                            if unit_done:
                                if lane == 0:
                                    if cutlass.const_expr(EPI_PFENCE):
                                        _fence_proxy_async_global()
                                    if cutlass.const_expr(EPI_FENCE):
                                        _fence_acq_rel_gpu()
                        cute.arch.sync_warp()
                        cute.arch.barrier(
                            barrier_id=EPI_BAR, number_of_threads=EPI_THREADS
                        )  # s_part reads done
                        if cutlass.const_expr(not REL_WARP):
                            if unit_done & (tidx == 0):
                                _red_release_add(
                                    cnt.subview(g_u * cutlass.Int32(8)).data_ptr().toint(),
                                    cutlass.Int32(1),
                                )
                        if cutlass.const_expr(TIMING):
                            e_busy = e_busy + (_gtime() - tb0)
                        k1 = k1 + cutlass.Int32(1)
                    if cutlass.const_expr(REL_WARP):
                        prims.mbarrier_arrive(
                            grp_done.subview(p & cutlass.Int32(3))
                        )  # this CTA's group-p units are out
                    if tidx == 0:
                        _stamp(tdbg, bx, 10)
                if p >= cutlass.Int32(LAG):
                    g = p - cutlass.Int32(LAG)
                    for bb in range(nb):
                        a2 = k2 & cutlass.Int32(1)
                        e_wait = _waitt(
                            acc2_full.subview(a2),
                            (k2 >> cutlass.Int32(1)) & cutlass.Int32(1),
                            e_wait,
                        )
                        if cutlass.const_expr(TIMING):
                            if (
                                (tidx == 0)
                                & (g == ng - cutlass.Int32(1))
                                & (bb == cutlass.Int32(0))
                            ):
                                _stamp(tdbg, bx, 8)
                        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                        yv = []
                        ya = prims.tcgen05_ld(
                            "32x32b",
                            _tptr(
                                row_w,
                                base_col + ACC2_COL + a2 * cutlass.Int32(ACC2_STRIDE),
                                cutlass.Float32,
                            ),
                            num=ACC2_STRIDE,
                        )
                        for t in cutlass.range_constexpr(M_MAX):
                            v = cutlass.Float32(0.0)
                            for w in cutlass.range_constexpr(4):
                                v = cutlass.select_(
                                    warp == cutlass.Int32(w), cutlass.Float32(ya[w * M_MAX + t]), v
                                )
                            yv.append(v)
                        prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                        prims.mbarrier_arrive(acc2_empty.subview(a2))
                        sl = g * cutlass.Int32(4) + warp
                        if warp < s_gn.load(idx=g):
                            a2s = s_esc.load(idx=sl * cutlass.Int32(8) + cutlass.Int32(5))
                            for t in cutlass.range_constexpr(M_MAX):
                                kp = s_kpos.load(idx=sl * cutlass.Int32(M_MAX) + cutlass.Int32(t))
                                if kp >= cutlass.Int32(0):
                                    ys.store(
                                        cutlass.BFloat16(
                                            _bf16_round(_mul_rn(cutlass.Float32(yv[t]), a2s))
                                        ),
                                        idx=(
                                            (bb * cutlass.Int32(M_MAX) + cutlass.Int32(t))
                                            * cutlass.Int32(TOP_K)
                                            + kp
                                        )
                                        * cutlass.Int32(R2)
                                        + lane,
                                    )
                        k2 = k2 + cutlass.Int32(1)
            if tidx == 0:
                _stamp(tdbg, bx, 6)
                _tstore(tdbg, bx, 31, e_wait)
                _tstore(tdbg, bx, 34, e_busy)
                _tstore(tdbg, bx, 36, e_a)
                _tstore(tdbg, bx, 38, e_pw)
            prims.mbarrier_arrive(ys_done.subview(0))  # this thread's ys values are in smem

        # ---- finalize on all 16 warps: (block bb, token t) pairs, the fma chain over t's top-4 in order, + y_sh
        NPAIR = NB2_CAP * M_MAX
        NPW = (NPAIR + NWARPS - 1) // NWARPS
        wyl = []
        for bt in cutlass.range_constexpr(NPW):
            pair = warp + cutlass.Int32(bt * NWARPS)
            bb = pair // cutlass.Int32(M_MAX)
            t = pair - bb * cutlass.Int32(M_MAX)
            ok = (pair < cutlass.Int32(NPAIR)) & (bb < nb)
            ch = cutlass.select_(
                ok,
                t * cutlass.Int32(H)
                + (blk0 + bb) * cutlass.Int32(R2)
                + cutlass.Int32(4) * (lane & cutlass.Int32(7))
                + (lane >> cutlass.Int32(3)),
                cutlass.Int32(0),
            )
            wyl.append(
                _ld_relaxed_u32(dnx.subview(ch).data_ptr().toint())
                if cutlass.const_expr(SHARED)
                else cutlass.Int32(0)
            )
        _wait(ys_done.subview(0), 0)
        if tidx == 0:
            _stamp(tdbg, bx, 19)
        for bt in cutlass.range_constexpr(NPW):
            pair = warp + cutlass.Int32(bt * NWARPS)
            bb = pair // cutlass.Int32(M_MAX)
            t = pair - bb * cutlass.Int32(M_MAX)
            if (pair < cutlass.Int32(NPAIR)) & (bb < nb):
                acc_o = cutlass.Float32(0.0)
                for k in cutlass.range_constexpr(TOP_K):
                    acc_o = _fma_rn(
                        s_w.load(idx=t * cutlass.Int32(TOP_K) + cutlass.Int32(k)),
                        cutlass.Float32(
                            ys.load(
                                idx=(
                                    (bb * cutlass.Int32(M_MAX) + t) * cutlass.Int32(TOP_K)
                                    + cutlass.Int32(k)
                                )
                                * cutlass.Int32(R2)
                                + lane
                            )
                        ),
                        acc_o,
                    )
                ch = (
                    t * cutlass.Int32(H)
                    + (blk0 + bb) * cutlass.Int32(R2)
                    + cutlass.Int32(4) * (lane & cutlass.Int32(7))
                    + (lane >> cutlass.Int32(3))
                )
                if cutlass.const_expr(SHARED):
                    wy = wyl[bt]
                    while ((wy >> cutlass.Int32(16)) & cutlass.Int32(0xFFFF)) != tag16:
                        wy = _ld_relaxed_u32(dnx.subview(ch).data_ptr().toint())
                    acc_o = _add_rn(
                        (wy << cutlass.Int32(16)).bitcast(cutlass.Float32), _bf16_round(acc_o)
                    )
                out.store(cutlass.BFloat16(acc_o), idx=ch)
                if ar_on:
                    cl = (
                        (blk0 + bb) * cutlass.Int32(R2)
                        + cutlass.Int32(4) * (lane & cutlass.Int32(7))
                        + (lane >> cutlass.Int32(3))
                    )
                    _ar_push(
                        arp,
                        _bf16_round(acc_o),
                        ((lane >> cutlass.Int32(3)) & cutlass.Int32(1)) == cutlass.Int32(0),
                        t,
                        cl,
                    )

        # ---- teardown
        if tidx == 0:
            _stamp(tdbg, bx, 7)
            epochs.store(ep + cutlass.Int32(1), idx=bx)
        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
        cute.arch.sync_warp()
        prims.barrier_cta_sync(0)
        if ar_on | cutlass.Boolean(LATE_TRIGGER):
            cute.arch.griddepcontrol_launch_dependents()
        if warp == MMA_WARP:
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            prims.tcgen05_dealloc(
                cutlass.inttoptr(tmem_ptr_i32.load(), 6, cutlass.Float32), TMEM_COLS
            )


@cute.jit
def m3_moe(
    a1_tensor: cute.Tensor,
    sfa1_tensor: cute.Tensor,
    a2_tensor: cute.Tensor,
    sfa2_tensor: cute.Tensor,
    bias: cute.Tensor,
    escal: cute.Tensor,
    plan: cute.Tensor,
    partials: cute.Tensor,
    img32: cute.Tensor,
    cnt: cute.Tensor,
    s1w: cute.Tensor,
    s2w: cute.Tensor,
    w1b: cute.Tensor,
    w2b: cute.Tensor,
    epochs: cute.Tensor,
    out: cute.Tensor,
    route_dbg: cute.Tensor,
    tdbg: cute.Tensor,
    xb32: cute.Tensor,
    rw: cute.Tensor,
    sfsc: cute.Tensor,
    lgbuf: cute.Tensor,
    guw: cute.Tensor,
    gusf: cute.Tensor,
    dnw: cute.Tensor,
    dnsf: cute.Tensor,
    hx: cute.Tensor,
    dnx: cute.Tensor,
    arp: cute.Tensor,
    stream: cuda_driver.CUstream,
):
    tma_a1_desc = cuda.create_tensor_map_tiled(
        global_address=a1_tensor.iterator.toint(),
        dtype=cutlass.Uint8,
        global_dims=[H // 2, TWO_I, E_LOCAL],
        global_strides=[H // 2 // 16, TWO_I * (H // 2) // 16],
        box_dims=(KT_BYTES, R1, 1),
        swizzle=cuda.TensorMapSwizzle.s128b,
    )
    tma_a2_desc = cuda.create_tensor_map_tiled(
        global_address=a2_tensor.iterator.toint(),
        dtype=cutlass.Uint8,
        global_dims=[I_TP // 2, H, E_LOCAL],
        global_strides=[I_TP // 2 // 16, H * (I_TP // 2) // 16],
        box_dims=(KT_BYTES, R2, 1),
        swizzle=cuda.TensorMapSwizzle.s128b,
    )
    tma_s1_desc = cuda.create_tensor_map_tiled(
        global_address=sfa1_tensor.iterator.toint(),
        dtype=cutlass.Uint8,
        global_dims=[128, E_LOCAL * TWO_I * (H // 16) // 128],
        global_strides=[128 // 16],
        box_dims=(128, NUM_BYTES_SFA_RAW // 128),
        swizzle=cuda.TensorMapSwizzle.none,
    )
    tma_s2_desc = cuda.create_tensor_map_tiled(
        global_address=sfa2_tensor.iterator.toint(),
        dtype=cutlass.Uint8,
        global_dims=[128, E_LOCAL * H * (I_TP // 16) // 128],
        global_strides=[128 // 16],
        box_dims=(128, NUM_BYTES_SFA_RAW // 4 // 128),
        swizzle=cuda.TensorMapSwizzle.none,
    )
    tma_gu_desc = tma_a1_desc
    tma_dn_desc = tma_a2_desc
    if cutlass.const_expr(SHARED):
        tma_gu_desc = cuda.create_tensor_map_tiled(
            global_address=guw.iterator.toint(),
            dtype=cutlass.Uint8,
            global_dims=[H, 2 * SH_I],
            global_strides=[H // 16],
            box_dims=(KT_BYTES, 32),
            swizzle=cuda.TensorMapSwizzle.s128b,
        )
        tma_dn_desc = cuda.create_tensor_map_tiled(
            global_address=dnw.iterator.toint(),
            dtype=cutlass.Uint8,
            global_dims=[SH_I, H],
            global_strides=[SH_I // 16],
            box_dims=(KT_BYTES, 128),
            swizzle=cuda.TensorMapSwizzle.s128b,
        )
    m3_moe_kernel(
        tma_a1_desc,
        tma_a2_desc,
        tma_s1_desc,
        tma_s2_desc,
        tma_gu_desc,
        tma_dn_desc,
        bias,
        escal,
        plan,
        partials,
        img32,
        cnt,
        s1w,
        s2w,
        w1b,
        w2b,
        epochs,
        out,
        route_dbg,
        tdbg,
        xb32,
        rw,
        sfsc,
        lgbuf,
        guw,
        gusf,
        dnw,
        dnsf,
        hx,
        dnx,
        arp,
    ).launch(
        grid=(NCTA + NSH, 1, 1),
        block=(THREADS, 1, 1),
        cluster=(1, 1, 1),
        stream=stream,
        use_pdl=True,
    )
