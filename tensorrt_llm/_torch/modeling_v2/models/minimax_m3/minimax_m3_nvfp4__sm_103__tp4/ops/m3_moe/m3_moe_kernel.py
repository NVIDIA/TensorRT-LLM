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
"""MiniMax-M3 routed experts for up to 8 decode tokens on one TP rank: ``m3_moe`` (group-phased weight stream).

One persistent kernel replaces TRTLLM-Gen's MiniMax2 routing, FC1 (NVFP4 x NVFP4 + SwiGLU-OAI + NVFP4 requantization),
FC2 and finalize, optionally with the router GEMV and the activation's NVFP4 quantization folded in (FUSED_ROUTER). It
writes the rank's routed partial [M, 6144] bf16, bit for bit what the TRTLLM-Gen op returns.

- Routing (warps 4..7, token t on warp 4 + t % 4): TRTLLM-Gen's MiniMax2 top-4, operation for operation; warp 4 then
  makes the distinct experts slots in (token, top-k) order (match.any), slot s in group s / 4.
- Work: the groups' FC1 stages and FC2 tiles stream through one ring per CTA in phases: phase p = FC1 of group p (a
  stream-K range of the group's unit-major stages), then FC2 of group p - LAG (this CTA's 32-row output blocks, 4
  slots' slices stacked in one 128-row MMA, B row j M + t). LAG phases of streaming hide the hand-off of group
  p - LAG's intermediate between CTAs.
- FC1 stage: a 32-row unit's rows at 4 consecutive 256-K tiles (rows 32 q + r) against resident B rows QS q + t.
  FC2 stage: 4 slots' 32 rows at one 256-K tile. Both carry 8 KB of raw scale atoms, spliced into the 4 instruction
  atoms by warp 6 with one formula (word [r][q] <- raw atom 4 q + kb, word [r][m1]).
- FC1 epilogue: k-quarters summed in order, SwiGLU-OAI, NVFP4 requant into tagged 16-byte records (3 words of 3 data
  bytes + the call's tag); a unit split between two CTAs is summed by its tail's holder (Lamport slot).
- Warps 5 and 8.. (XW warps) build each group's FC2 B (tiles + scale atoms) from the records once they carry the
  call's tag; warp 5 copies the scales to TMEM.
- Finalize: bf16(fc2_alpha * acc) per (slot, token) into its top-k position, then per token the fma chain over its
  top-4 in order, bf16 out. With an all-reduce descriptor, every output word is also pushed to the exchange
  (AR_PRE: call parities and descriptor loaded during the last group's FC2).

Configuration (trace-time constants) via M3_CONFIG = {"m_max", "num_ctas", "stages", "nb2_cap", "fused_router", "lag",
"xw", ...}; the plan (op.make_phase_plan) must be built for the same lag.
Launched with PDL: barrier setup, TMEM allocation, tensormap prefetch, bias / plan / router weight loads before
griddepcontrol.wait; every producer-output read and every global write after it.
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


# =============================================================================
# Problem shape (trace-time constants).
# =============================================================================
H = 6144  # capacity: fixed: the checkpoint's hidden size
I_TP = 768
TWO_I = 2 * I_TP
E_LOCAL = 128
TOP_K = 4  # capacity: fixed: the checkpoint's top-k
ROUTE_SCALE = 2.0
SUM_EPS = 1e-20
M_MAX = _cfg("m_max", 1)
NCTA = _cfg("num_ctas", 152)
FUSED_ROUTER = bool(_cfg("fused_router", 0))
NROUTE = E_LOCAL if FUSED_ROUTER else 0  # experts whose logits the router GEMV computes
NGEMV = (NROUTE + NCTA - 1) // NCTA  # experts per CTA: bx, bx + NCTA, ..
TIMING = bool(_cfg("timing", 0))
HANG_DBG = bool(_cfg("hang_dbg", 0))
XVDUMP = bool(
    _cfg("xvdump", 0)
)  # debug only (needs a plan without split units): FC1 sums into `partials`
YDUMP = bool(
    _cfg("ydump", 0)
)  # debug only: FC2 outputs per (token, k) into `partials` (breaks its protocol)
LAG = _cfg("lag", 2)  # phase p runs FC1 of group p and FC2 of group p - LAG
G_CAP = TOP_K * M_MAX  # distinct experts per call
NG_CAP = M_MAX  # groups of 4 slots
XW = _cfg("xw", 5 if M_MAX > 4 else (4 if M_MAX > 2 else 1))  # FC2 B builder warps: 5, then 8 ..
THREADS = 256 + 32 * (XW - 1)
GEMV_THREADS = 256  # capacity: fixed: the CTA size
# SHARED: the MoE's shared expert (MXFP8 gate_up [1536, 6144] -> SwiGLU-OAI -> MXFP8 down [6144, 768]) on NSH more
# CTAs (warps 0-3), its output added in finalize: out = bf16(shared + routed) (shared_output.add_(routed_output)).
# The math is mxfp8_low_m_gemm's (native MXFP8 activation quant, mma.sync m16n8k32 e4m3 per 32-K block, block scales
# as one fma, its partial-sum order, bf16) and the Triton silu_and_mul's, operation for operation. Shared CTA s owns
# intermediate positions 32 s .. 32 s + 31 (gate and up rows: MMA tiles = warps 0-3) and down rows 256 s .. + 255;
# 16-row x 768-byte weight chunks (+ their scale atoms) stream through NBW buffers per warp, the first NBW before
# the grid dependency. h crosses the shared CTAs, y_sh reaches the routed CTAs' finalize (both tagged words).
SHARED = bool(_cfg("shared", 0))
# LATE_TRIGGER: release the dependents at teardown, not after the grid dependency. An early release lets the next
# kernels' CTAs (the all-reduce, then the attention it releases at entry) become resident during this kernel, packed
# onto the SMs it leaves free, where the attention then runs on a fraction of the GPU.
LATE_TRIGGER = bool(_cfg("late_trigger", 1))
# AR_PRE: the finalize's all-reduce push uses the call parities and descriptor fields loaded while the last group's
# FC2 runs (not two dependent global loads per push on the kernel's tail).
AR_PRE = bool(_cfg("ar_pre", 1))
# XSTAGE (fused router): the hidden states land in the (still idle) ring by one bulk copy per token right after the
# grid dependency; the NVFP4 quantization of B1 reads them from shared memory (one HBM / L2 round trip, not ~8).
XSTAGE = bool(_cfg("xstage", 1)) and bool(_cfg("fused_router", 0))
NSH = 24 if SHARED else 0
SH_I = 768
SH_KB = H // 32  # gate_up K blocks  # capacity: fixed: a kernel tile constant
SH_DKB = SH_I // 32  # down K blocks  # capacity: fixed: a kernel tile constant
SH_SPLIT = 2  # gate_up K slices (= m3_mx_gemv's cluster ranks, contiguous K ranges): MMA rows 64 q + .. = slice q
SH_GT = (
    SH_KB // 4 // SH_SPLIT
)  # gate_up k-tiles (128 K) per slice  # capacity: fixed: a kernel tile constant
SH_STAGES_GU = SH_GT
SH_STAGES = SH_GT + 2 * (
    SH_I // 128
)  # + 2 down tiles x 6 k-tiles  # capacity: fixed: pipeline stages
SHX_BAR = 6
# SH_PRETILE: the shared weights come pre-tiled (op._pretile_shared): per shared CTA its 36 stages, each the 16 KB
# shared-memory image of its 128-row MMA tile (128-byte swizzle) followed by the stage's 512-byte instruction scale
# atom, so a stage is two contiguous bulk copies (not 128 row segments of 128 bytes) and needs no splicing.
SH_PRETILE = bool(
    _cfg("sh_pretile", 1 if M_MAX <= 2 else 0)
)  # (in-chain: a win at M <= 2, not at 4..8)
SH_STAGE_BYTES = 16384 + 512
MMA_M = 128
KT_BYTES = 128
INC_K = 2
NUM_KB = 4
# FC1
N1 = 16 if M_MAX <= 4 else 32  # B rows: QS q + t (token t at the stage's k-tile 4 j + q)
QS = N1 // 4
R1 = 32
C1 = 4
U1 = TWO_I // R1  # 48 units per slot
K1_TILES = H // 256
S1 = K1_TILES // C1  # 6 stages per unit  # capacity: fixed: pipeline stages
NUM_BYTES_A = MMA_M * KT_BYTES  # capacity: fixed: a stage's operand bytes
NUM_BYTES_BOX1 = R1 * KT_BYTES
SF1_KA = H // 64
SF1_RB = TWO_I // 128
NUM_BYTES_SFA_RAW = 16 * 512
NUM_WORDS_SFA_RAW = NUM_BYTES_SFA_RAW // 4
NUM_BYTES_SFA = NUM_KB * 512  # capacity: fixed: a stage's scale bytes
NUM_WORDS_SFA = NUM_BYTES_SFA // 4
NUM_BYTES_B1 = N1 * KT_BYTES
NUM_SFB1_ATOMS = S1 * NUM_KB
# FC2
R2 = 32
NB2 = H // R2  # 192
KT2 = I_TP // 256  # 3
SF2_KA = I_TP // 64
SF2_RB = H // 128
N2 = max(8, (4 * M_MAX + 7) // 8 * 8)  # B rows j M + t
NUM_BYTES_B2 = N2 * KT_BYTES
NB2_CAP = _cfg("nb2_cap", 2)
UNITS_ROW = TWO_I // R1  # 48 records per (slot, token) row
H_ROW = UNITS_ROW * 16
PARTIAL_EMPTY = -1  # capacity: fixed: a sentinel, not a size
EPI_THREADS = 128
EPI_BAR = 1
ROUTE_BAR = 2
GEMV_BAR = 3
B2_BAR = 5
TD_STRIDE = 64  # tdbg columns per CTA
TOPK_BAR = 4
B1_AFTER_TOPK = bool(
    _cfg("b1_after_topk", 0)
)  # (experiment) the B1 build waits for the routing warps' top-k
TOPK2_BAR = 7
EARLY0 = bool(
    _cfg("early0", 1)
)  # warp 4 streams group 0 (token 0's experts) right after its own top-k
EARLY0_N = _cfg("early0_n", 99)  # stages issued early (capped at STAGES)
# routing warps: 4..7, then the FC2 B builder warps 8.. (idle until the routing is done) up to one warp per token
RW_EXTRA = max(0, min(M_MAX - 4, (THREADS // 32) - 8)) if bool(_cfg("rw_extra", 1)) else 0
NRW0 = (
    min(4, M_MAX) if bool(_cfg("rw_trim", 1)) else 4
)  # warps 4.. with a token (no idle pollers in the barrier)
NRW = NRW0 + RW_EXTRA
LG_REPL = _cfg("lg_repl", 32 if M_MAX > 4 else 128)
GEMV_SMEM = bool(
    _cfg("gemv_smem", 1)
)  # (XSTAGE) the router GEMV reads the staged hidden states from smem
LG_STRIDE = M_MAX * E_LOCAL + 32
POLL_NS = _cfg("poll_ns", 150)
B2_POLL_NS = _cfg("b2_poll_ns", 200)
# B2IMG (needs SFA_CP: warp 6 is free): the FC1 epilogue writes each finished unit's FP4 bytes and scale byte straight
# into a per-group global image laid out as the FC2 B tile + its scale atoms (exactly what the builder warps assembled
# from the tagged records); per group, warp 6 joins the epilogue (fc1_done), fences the async proxy and does one
# release add of this CTA's finished units to the group counter; warp 5 acquires the counter (every group gains
# CNT_PER_CALL per call, CTA 0 pads absent slots), fences, and bulk-copies the image into the B2 buffer.
B2IMG = bool(_cfg("b2img", 0)) and bool(_cfg("sfa_cp", 0))
IMG_B2 = KT2 * NUM_BYTES_B2
IMG_SF = KT2 * NUM_KB * 512
IMG_BYTES = IMG_B2 + IMG_SF
CNT_PER_CALL = 4 * (TWO_I // R1)
B2IMG_NS = _cfg("b2img_ns", 16)  # (B2IMG) counter poll back-off
B2_DLY = _cfg(
    "b2_dly", 0
)  # (stress) the release warp spins up to this many ns (CTA- and group-dependent) first
W_WARP = 4  # capacity: fixed: a warp role index
X_WARP = 5  # capacity: fixed: a warp role index
S_WARP = 6  # capacity: fixed: a warp role index
MMA_WARP = 7  # capacity: fixed: a warp role index
# (M <= 3) the early stages are issued by the idle MMA warp, not by warp 4 (routing's critical path)
E0_WARP = MMA_WARP  # capacity: fixed: a warp role index
E0_BAR = 8
E1_BAR = 9
E0_OFFLOAD = EARLY0 and NRW0 <= 3 and bool(_cfg("e0_off", 1))
EVICT_FIRST = 0x12F0000000000000
_LOG2E = 1.4426950408889634
LD_POW2 = (
    1 if M_MAX == 1 else 2 if M_MAX == 2 else 4 if M_MAX <= 4 else 8
)  # tcgen05.ld .num covering M columns

_FIXED_SMEM = (
    S1 * NUM_BYTES_B1
    + 2 * KT2 * NUM_BYTES_B2
    + 2 * KT2 * NUM_KB * 512
    + NB2_CAP * M_MAX * TOP_K * R2 * 4
    + 2 * M_MAX * C1 * R1 * 4
    + (M_MAX * (H // 16) if FUSED_ROUTER else 0)
    + 6 * 1024
)
# SFA_CP: the A scale words go by 4-byte cp.async straight into the instruction atom layout (no raw scale box, no
# splicer warp; the MMA warp tcgen05.cp's them), completing on the stage's full barrier: 18 KB stages instead of 26 KB
SFA_CP = bool(_cfg("sfa_cp", 0))
_RAW_STAGE = 0 if SFA_CP else NUM_BYTES_SFA_RAW
_STAGE_SMEM = NUM_BYTES_A + _RAW_STAGE + NUM_BYTES_SFA
_STAGES_FIT = (227 * 1024 - _FIXED_SMEM - (128 if SHARED else 0)) // _STAGE_SMEM
ACC1_COL = 0
ACC2_COL = 2 * N1
SFB1_COL = ACC2_COL + 2 * N2
SFB2_COL = SFB1_COL + NUM_SFB1_ATOMS * 4
SFA1_COL = SFB2_COL + 2 * KT2 * NUM_KB * 4
STAGES = min(_cfg("stages", 12 if SFA_CP else 8), _STAGES_FIT, (512 - SFA1_COL) // 16)
TMEM_COLS = 512
assert 1 <= M_MAX <= 8, M_MAX
assert STAGES >= 3, (STAGES, _STAGES_FIT, SFA1_COL)
assert M_MAX * H * 2 <= STAGES * (NUM_BYTES_A + _RAW_STAGE + NUM_BYTES_SFA), (
    "x staging must fit the ring"
)
# one shared-memory region for the ring and the FC1 / FC2 B operands (a shared CTA reuses all of it)
_OFF_SFAR = STAGES * NUM_BYTES_A
_OFF_SFA = _OFF_SFAR + STAGES * _RAW_STAGE
_OFF_B1 = _OFF_SFA + STAGES * NUM_BYTES_SFA
_OFF_B2 = _OFF_B1 + S1 * NUM_BYTES_B1
_OFF_SFB2 = _OFF_B2 + 2 * KT2 * NUM_BYTES_B2
BIG = _OFF_SFB2 + 2 * KT2 * NUM_KB * 512
# shared CTA layout in BIG: weight buffers, x quarters (2), x scales [M, 192], gate / up [4, 16, M], h, h scales
# shared CTA layout in BIG: A ring (16 KB stages), raw scale atoms (4 per stage), spliced atoms, B (x: [stage][16 rows]
# [128 B]; h: [6][8][128 B]), x / h scale codes, gate_up partial exchange
_SB_A = 0
_SB_RAW = None
_SH_FIX = (
    SH_GT * 16 * 128
    + 6 * 8 * 128
    + M_MAX * SH_KB
    + M_MAX * SH_DKB
    + 64 * 8 * 4
    + 2 * 32 * 8 * 4
    + 1024
)
SH_NST = min(8, (BIG - _SH_FIX) // (NUM_BYTES_A + 4 * 512 + 512)) if SHARED else 1
_SB_RAW = SH_NST * NUM_BYTES_A
_SB_SPL = _SB_RAW + SH_NST * 2048
_SB_BX = (_SB_SPL + SH_NST * 512 + 1023) // 1024 * 1024
_SB_BH = _SB_BX + SH_GT * 16 * 128
_SB_XC = _SB_BH + 6 * 8 * 128
_SB_HC = _SB_XC + M_MAX * SH_KB
_SB_XCH = (_SB_HC + M_MAX * SH_DKB + 15) // 16 * 16
_SB_GU = _SB_XCH + 64 * 8 * 4
_SB_END = _SB_GU + 2 * 32 * 8 * 4
# shared CTA TMEM columns: gate_up accumulator (N 16), down accumulators (2 x N 8), A scales per stage, B scales
SH_ACC_GU = 0
SH_ACC_DN = 16
SH_SFA = 32
SH_SFB_GU = SH_SFA + 4 * 8
SH_SFB_DN = SH_SFB_GU + 4 * SH_GT  # capacity: fixed: a kernel tile constant
SH_TMEM = 256  # capacity: fixed: TMEM columns
if SHARED:
    assert FUSED_ROUTER, "shared expert: fused router only"
    assert SH_STAGES - SH_NST <= 64, SH_STAGES
    assert SH_NST >= 3 and _SB_END <= BIG and SH_SFB_DN + 24 <= SH_TMEM, (SH_NST, _SB_END, BIG)


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


def _ar_parity(arp, t, cl):
    """The call parity _ar_push reads for token t's channel cl (fixed between this kernel's grid dependency and the
    next all-reduce, which this kernel releases at teardown): loaded ahead of the finalize (AR_PRE)."""
    return _ld_volatile_s32(
        arp.load(idx=2)
        + cutlass.Int64(4) * cutlass.Int64(t * cutlass.Int32(8) + cl // cutlass.Int32(H // 8))
    )


def _ar_push_pre(ard, parity32, v, low, t, cl):
    """_ar_push with the parity (_ar_parity) and the descriptor fields ard = (mc, rank, world, maxt) preloaded."""
    hi = cute.arch.shuffle_sync_bfly(v, 8)
    pred = cutlass.select_(low, cutlass.Int32(1), cutlass.Int32(0))
    cs = cutlass.select_(low, cl, cutlass.Int32(0))
    mc, rk, world, maxt = ard
    parity = cutlass.Int64(parity32 & cutlass.Int32(1))
    off = (
        parity * maxt * world * cutlass.Int64(H // 2)
        + (cutlass.Int64(t) * world + rk) * (cutlass.Int64(H // 2))
        + cutlass.Int64(cs // cutlass.Int32(2))
    )
    _st_global_b32_if(mc + off * cutlass.Int64(4), _bf16x2_word(v, hi), pred)


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
    q = lane // cutlass.Int32(QS)
    tl = lane - q * cutlass.Int32(QS)
    sf_ok = (lane < cutlass.Int32(N1)) & (tl < cutlass.Int32(M_MAX))
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
            _tptr(base_row + warp * cutlass.Int32(32), base_col + SFB1_COL + a * 4),
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
def _build_b1(
    tidx,
    warp,
    lane,
    base_row,
    base_col,
    x32: cutlass.Array,
    xsf32: cutlass.Array,
    sB1: cutlass.Array,
    b1_ready: cutlass.Array,
):
    """B (the M tokens' whole K) into the resident tiles from the NVFP4 activation, its scales to TMEM."""
    sB1_32 = cutlass.Array(
        sB1.data_ptr(0), shape=(S1 * NUM_BYTES_B1 // 4,), dtype=cutlass.Int32, alignment=16
    )
    for r in cutlass.range_constexpr((M_MAX * (H // 16) + EPI_THREADS - 1) // EPI_THREADS):
        b = tidx + cutlass.Int32(r * EPI_THREADS)
        if b < cutlass.Int32(M_MAX * (H // 16)):
            t = b // cutlass.Int32(H // 16)
            kb = b - t * cutlass.Int32(H // 16)
            v2 = x32.load(idx=b * cutlass.Int32(2), vector_size=2, alignment=8)
            sB1_32.store(
                (cutlass.Int32(v2[0]), cutlass.Int32(v2[1])), idx=_b1_addr(t, kb), alignment=8
            )
    _sfb1_to_tmem(warp, lane, base_row, base_col, xsf32, b1_ready)


@cute.jit
def _build_b1_fused(
    tidx,
    warp,
    lane,
    base_row,
    base_col,
    xb32: cutlass.Array,
    sfsc: cutlass.Array,
    sB1: cutlass.Array,
    s_xsf: cutlass.Array,
    b1_ready: cutlass.Array,
):
    """_build_b1 from the bf16 hidden states, quantized exactly as fp4_quantize(x, sf_scale, 16)."""
    sB1_32 = cutlass.Array(
        sB1.data_ptr(0), shape=(S1 * NUM_BYTES_B1 // 4,), dtype=cutlass.Int32, alignment=16
    )
    s_xsf8 = cutlass.Array(
        s_xsf.data_ptr(0), shape=(M_MAX * (H // 16),), dtype=cutlass.Int8, alignment=16
    )
    sc = cutlass.Float32(sfsc.load(idx=0))
    rsc = _rcp(sc)
    rsix = _rcp(cutlass.Float32(6.0))
    NBLK = M_MAX * (H // 16) // EPI_THREADS
    QB = (
        3 if NBLK % 3 == 0 else (2 if NBLK % 2 == 0 else 1)
    )  # blocks whose loads are in flight together
    # a rolled loop over groups of QB blocks (a fully unrolled one is ~50 KB of SASS at M = 8: instruction fetch bound)
    for rg in cutlass.range(NBLK // QB, unroll=1):
        wall = []
        for rr in cutlass.range_constexpr(QB):
            bq = tidx + (rg * cutlass.Int32(QB) + cutlass.Int32(rr)) * cutlass.Int32(EPI_THREADS)
            for h in cutlass.range_constexpr(2):
                wall.append(
                    xb32.load(
                        idx=bq * cutlass.Int32(8) + cutlass.Int32(4 * h),
                        vector_size=4,
                        alignment=16,
                    )
                )
        for rr in cutlass.range_constexpr(QB):
            b = tidx + (rg * cutlass.Int32(QB) + cutlass.Int32(rr)) * cutlass.Int32(EPI_THREADS)
            t = b // cutlass.Int32(H // 16)
            kb = b - t * cutlass.Int32(H // 16)
            w = []
            for h in cutlass.range_constexpr(2):
                v4 = wall[rr * 2 + h]
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
                vmax != cutlass.Float32(0.0),
                _rcp(_mul_rn(_e4m3_to_f32(sfb), rsc)),
                cutlass.Float32(0.0),
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
            sB1_32.store((lo, hi), idx=_b1_addr(t, kb), alignment=8)
            s_xsf8.store(cutlass.Int8(sfb), idx=b)
    cute.arch.sync_warp()
    cute.arch.barrier(barrier_id=EPI_BAR, number_of_threads=EPI_THREADS)
    _sfb1_to_tmem(warp, lane, base_row, base_col, s_xsf, b1_ready)


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
):
    """Stage g of the shared CTA's stream into ring slot g % SH_NST (warp 0). g < SH_GT: gate_up k-tile g of both
    slices: quadrant Q = 2 q + u (q slice, u 0 gate / 1 up) = rows 32 sid .. (+ 768 for up) at k-tile SH_GT q + g, its
    raw scale atom at raw slot Q. Else down tile j = (g - SH_GT) / 6 (rows 256 sid + 128 j ..), k-tile
    (g - SH_GT) % 6."""
    st = g % cutlass.Int32(SH_NST)
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
                    l2_cache_hint=EVICT_FIRST,
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
                    l2_cache_hint=EVICT_FIRST,
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
    _asm(None, [], "fence.proxy.async.global;", "", True, loc, ip)


@cute.jit
def _sfa_to_tmem(sSFA: cutlass.Array, stage, base_row, base_col):
    """(SFA_CP, elected MMA thread) the stage's 4 scale atoms smem -> TMEM (what the splicer warp does otherwise)."""
    s2t_shape, s2t_mc = prims.S2TCopyMode.S2T_32x128b_WARPX4
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
            _tptr(base_row, base_col + SFA1_COL + stage * cutlass.Int32(16) + kb * 4),
            sd0 + (kb * 512 >> 4),
            multicast=s2t_mc,
        )


@cute.jit
def _sf1_cp(
    lane, stage, sf_line, c, sSFA: cutlass.Array, s1w: cutlass.Array, ab_full: cutlass.Array
):
    """(SFA_CP) FC1 stage scale atoms: word [r][q] of atom kb <- raw atom 4 q + kb (raw line sf_line = 4 x atom)
    word [r][c % 4] (what the splicer copies out of the raw box), 4-byte cp.async by the warp's 32 lanes; their
    arrivals (noinc) complete the stage's full barrier together with the A boxes' bytes."""
    src0 = (
        (sf_line >> cutlass.Int32(2)) * cutlass.Int32(128)
        + lane * cutlass.Int32(4)
        + (c & cutlass.Int32(3))
    )
    dst0 = stage * cutlass.Int32(NUM_WORDS_SFA) + lane * cutlass.Int32(4)
    for q in cutlass.range_constexpr(C1):
        for kb in cutlass.range_constexpr(NUM_KB):
            prims.cp_async_shared_global(
                sSFA.subview(dst0 + cutlass.Int32(kb * 128 + q)),
                s1w.subview(src0 + cutlass.Int32((4 * q + kb) * 128)),
                4,
                "ca",
            )
    prims.cp_async_mbarrier_arrive(ab_full.subview(stage), noinc=True)
    cute.arch.sync_warp()


@cute.jit
def _sf2_cp(
    lane,
    stage,
    g,
    blk,
    t2: cutlass.Constexpr,
    n_here,
    s_el: cutlass.Array,
    sSFA: cutlass.Array,
    s2w: cutlass.Array,
    ab_full: cutlass.Array,
):
    """(SFA_CP) FC2 k-tile t2 scale atoms: word [r][jj] of atom kb <- raw atom (rows / 128, K block 4 t2 + kb) of slot
    jj's expert, word [r][blk % 4]."""
    dst0 = stage * cutlass.Int32(NUM_WORDS_SFA) + lane * cutlass.Int32(4)
    for jj in cutlass.range_constexpr(4):
        if cutlass.Int32(jj) < n_here:
            e = s_el.load(idx=g * cutlass.Int32(4) + cutlass.Int32(jj))
            src0 = (
                (
                    (e * cutlass.Int32(SF2_RB) + (blk >> cutlass.Int32(2))) * cutlass.Int32(SF2_KA)
                    + cutlass.Int32(4 * t2)
                )
                * cutlass.Int32(128)
                + lane * cutlass.Int32(4)
                + (blk & cutlass.Int32(3))
            )
            for kb in cutlass.range_constexpr(NUM_KB):
                prims.cp_async_shared_global(
                    sSFA.subview(dst0 + cutlass.Int32(kb * 128 + jj)),
                    s2w.subview(src0 + cutlass.Int32(kb * 128)),
                    4,
                    "ca",
                )
    prims.cp_async_mbarrier_arrive(ab_full.subview(stage), noinc=True)
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
            _sh_issue(lane, sid, cutlass.Int32(g), s_big, sh_full, tma_gu, tma_dn, gusf, dnsf)
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
        else:
            for r in cutlass.range_constexpr(2):
                prw = sid * cutlass.Int32(32) + cutlass.Int32(r * SH_I) + lane
                _l2_prefetch(gu8.subview(prw * cutlass.Int32(H)).data_ptr().toint(), H)
            if lane == 0:
                _l2_prefetch(
                    dn8.subview(sid * cutlass.Int32(256 * SH_I)).data_ptr().toint(), 256 * SH_I
                )
    cute.arch.griddepcontrol_wait()
    if cutlass.const_expr(not LATE_TRIGGER):
        if arp.load(idx=0) == cutlass.Int64(0):
            cute.arch.griddepcontrol_launch_dependents()
    if tidx == 0:
        _stamp(tdbg, bx, 1)
    ep = epochs.load(idx=bx)
    tag16 = (ep & cutlass.Int32(0x7FFF)) | cutlass.Int32(0x8000)
    NQ = THREADS - 64  # activation quant threads (warps 2 ..)
    if warp == 0:
        for g in cutlass.range(SH_NST, SH_STAGES, unroll=1):
            _wait(
                sh_empty.subview(g % cutlass.Int32(SH_NST)),
                ((g // cutlass.Int32(SH_NST)) - cutlass.Int32(1)) & cutlass.Int32(1),
            )
            _sh_issue(lane, sid, g, s_big, sh_full, tma_gu, tma_dn, gusf, dnsf)
    elif warp == 1:
        # ---- MMA: gate_up (scale atoms spliced to the tile's row order), then down
        idesc_gu = prims.Tcgen05MxInstrDesc.build(
            a_dtype=cutlass.Float8E4M3FN,
            b_dtype=cutlass.Float8E4M3FN,
            scale_format=1,
            n_dim=16,
            m_dim=128,
        )
        idesc_dn = prims.Tcgen05MxInstrDesc.build(
            a_dtype=cutlass.Float8E4M3FN,
            b_dtype=cutlass.Float8E4M3FN,
            scale_format=1,
            n_dim=8,
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
        _wait(b_gu_rdy, 0)
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        for g in cutlass.range(SH_STAGES, unroll=1):
            st = g % cutlass.Int32(SH_NST)
            if g == cutlass.Int32(SH_GT):
                _wait(b_dn_rdy, 0)
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            _wait(sh_full.subview(st), (g // cutlass.Int32(SH_NST)) & cutlass.Int32(1))
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
                    db = desc_bx + cutlass.Int64(g * cutlass.Int32(2048 >> 4))
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
                    db = desc_bh + cutlass.Int64(kt * cutlass.Int32(1024 >> 4))
                    for kb in cutlass.range_constexpr(4):
                        prims.tcgen05_mma_block_scale(
                            prims.MMABlockScaleKind.MXF8F6F4,
                            prims.CTAGroup.CTA_1,
                            cutlass.inttoptr(
                                tcol + SH_ACC_DN + jt * cutlass.Int32(8), 6, cutlass.Float32
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
        # ---- the activation, MXFP8 per 32-K block (native mxfp8_quantize), into B (row 8 q + t of stage tile i =
        # token t's k-tile SH_GT q + i, 128-byte swizzle) and its codes
        qt = tidx - cutlass.Int32(64)
        for r in cutlass.range_constexpr((M_MAX * SH_KB + NQ - 1) // NQ):
            q_ = qt + cutlass.Int32(r * NQ)
            if q_ < cutlass.Int32(M_MAX * SH_KB):
                t = q_ // cutlass.Int32(SH_KB)
                b = q_ - t * cutlass.Int32(SH_KB)
                wv = [
                    xb32.load(
                        idx=t * cutlass.Int32(H // 2)
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
                n = sl * cutlass.Int32(8) + t
                c0 = (b & cutlass.Int32(3)) * cutlass.Int32(2)
                rw = (
                    cutlass.Int32(_SB_BX) + it * cutlass.Int32(2048) + n * cutlass.Int32(128)
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
            # B scales of gate_up to TMEM: column of stage i, lane n = 8 q + t: token t's codes of k-tile SH_GT q + i
            n = lane
            t = n & cutlass.Int32(7)
            ok = (n < cutlass.Int32(16)) & (t < cutlass.Int32(M_MAX))
            for i in cutlass.range_constexpr(SH_GT):
                kt = (n >> cutlass.Int32(3)) * cutlass.Int32(SH_GT) + cutlass.Int32(i)
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
            # ---- gate_up out: quadrant qd lane l = MMA row 32 qd + l; slice q = qd / 2 holds token t at column 8 q + t
            _wait(acc_gu, 0)
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            av = prims.tcgen05_ld(
                "32x32b",
                _tptr(base_row + qd * cutlass.Int32(32), base_col + SH_ACC_GU, cutlass.Float32),
                num=16,
            )
            prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
            if qd >= cutlass.Int32(2):
                for tt in cutlass.range_constexpr(8):
                    bigf.store(
                        cutlass.Float32(av[8 + tt]),
                        idx=cutlass.Int32(_SB_XCH // 4)
                        + ((qd - cutlass.Int32(2)) * cutlass.Int32(32) + lane) * cutlass.Int32(8)
                        + cutlass.Int32(tt),
                    )
            cute.arch.sync_warp()
            cute.arch.barrier(barrier_id=SHX_BAR + 1, number_of_threads=128)
            if qd < cutlass.Int32(2):
                # m3_mx_gemv's owner sum: (0 + rank 0) + rank 1, then bf16
                for tt in cutlass.range_constexpr(8):
                    tot = _add_rn(cutlass.Float32(0.0), cutlass.Float32(av[tt]))
                    tot = _add_rn(
                        tot,
                        bigf.load(
                            idx=cutlass.Int32(_SB_XCH // 4)
                            + (qd * cutlass.Int32(32) + lane) * cutlass.Int32(8)
                            + cutlass.Int32(tt)
                        ),
                    )
                    bigf.store(
                        _bf16_round(tot),
                        idx=cutlass.Int32(_SB_GU // 4)
                        + (qd * cutlass.Int32(32) + lane) * cutlass.Int32(8)
                        + cutlass.Int32(tt),
                    )
            cute.arch.sync_warp()
            cute.arch.barrier(barrier_id=SHX_BAR + 1, number_of_threads=128)
            # h = silu_and_mul(gate, up) of positions 32 sid + p, published under the call's tag
            for r in cutlass.range_constexpr((32 * M_MAX + 127) // 128):
                pq = et + cutlass.Int32(r * 128)
                if pq < cutlass.Int32(32 * M_MAX):
                    p = pq // cutlass.Int32(M_MAX)
                    tt = pq - p * cutlass.Int32(M_MAX)
                    ga = bigf.load(idx=cutlass.Int32(_SB_GU // 4) + p * cutlass.Int32(8) + tt)
                    up = bigf.load(
                        idx=cutlass.Int32(_SB_GU // 4)
                        + (cutlass.Int32(32) + p) * cutlass.Int32(8)
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
                        hx.subview(tt * cutlass.Int32(SH_I) + sid * cutlass.Int32(32) + p)
                        .data_ptr()
                        .toint(),
                        (tag16 << cutlass.Int32(16)) | _bf16_bits(q_h),
                    )
            if et == 0:
                _stamp(tdbg, bx, 50)
            # every shared CTA's h: (token, K block) per thread, 32 tagged words, MXFP8 into B (h) and its codes
            for rr in cutlass.range_constexpr((M_MAX * SH_DKB + 127) // 128):
                q_ = et + cutlass.Int32(rr * 128)
                if q_ < cutlass.Int32(M_MAX * SH_DKB):
                    t = q_ // cutlass.Int32(SH_DKB)
                    kb = q_ - t * cutlass.Int32(SH_DKB)
                    base = t * cutlass.Int32(SH_I) + kb * cutlass.Int32(32)
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
                        cutlass.Int32(_SB_BH) + kt * cutlass.Int32(1024) + t * cutlass.Int32(128)
                    ) >> cutlass.Int32(2)
                    big32.store(
                        (words[0], words[1], words[2], words[3]),
                        idx=rw + ((c0 ^ t) << cutlass.Int32(2)),
                        alignment=16,
                    )
                    big32.store(
                        (words[4], words[5], words[6], words[7]),
                        idx=rw + (((c0 + cutlass.Int32(1)) ^ t) << cutlass.Int32(2)),
                        alignment=16,
                    )
                    big8.store(cutlass.Int8(code), idx=cutlass.Int32(_SB_HC) + q_)
            prims.fence_proxy("async_shared", space=prims.SharedSpace.shared_cta)
            cute.arch.sync_warp()
            cute.arch.barrier(barrier_id=SHX_BAR + 1, number_of_threads=128)
            okh = lane < cutlass.Int32(M_MAX)
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
            _wait(acc_dn, 0)
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            for jt in cutlass.range_constexpr(2):
                dv = prims.tcgen05_ld(
                    "32x32b",
                    _tptr(
                        base_row + qd * cutlass.Int32(32),
                        base_col + SH_ACC_DN + jt * 8,
                        cutlass.Float32,
                    ),
                    num=8,
                )
                prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                rowo = (
                    sid * cutlass.Int32(256)
                    + cutlass.Int32(128 * jt)
                    + qd * cutlass.Int32(32)
                    + lane
                )
                for tt in cutlass.range_constexpr(M_MAX):
                    _st_relaxed_u32(
                        dnx.subview(cutlass.Int32(tt * H) + rowo).data_ptr().toint(),
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


@cute.kernel
def m3_moe_kernel(
    tma_a1_desc: cutlass.GridConstant[cuda.TensorMap],
    tma_a2_desc: cutlass.GridConstant[cuda.TensorMap],
    tma_s1_desc: cutlass.GridConstant[cuda.TensorMap],  # w3_w1_weight_scale as 128-byte lines
    tma_s2_desc: cutlass.GridConstant[cuda.TensorMap],  # w2_weight_scale as 128-byte lines
    tma_gu_desc: cutlass.GridConstant[
        cuda.TensorMap
    ],  # SHARED: gate_up e4m3 [1536, 6144], box 128 B x 32 rows
    tma_dn_desc: cutlass.GridConstant[
        cuda.TensorMap
    ],  # SHARED: down e4m3 [6144, 768], box 128 B x 128 rows
    logits: cutlass.Array,  # f32 [M, 128] (not FUSED_ROUTER)
    bias: cutlass.Array,  # f32 [128]
    x32: cutlass.Array,  # int32 words of x_fp4 [M, 3072 B] (not FUSED_ROUTER)
    xsf32: cutlass.Array,  # int32 words of x_sf [M, 384 B] (not FUSED_ROUTER)
    escal: cutlass.Array,  # f32 [6, 128]: fc31_alpha, fc31_scale_c, act_alpha, act_beta, act_clamp, fc2_alpha
    plan: cutlass.Array,  # int32 [4 M + 1, NCTA, NG_CAP + 1] by G: per group i0 | i1 << 16, then b0 | nb << 16
    partials: cutlass.Array,  # int32 bits of f32 [G_CAP * 48, M, 32]: split units' head sums
    hbuf32: cutlass.Array,  # int32 [G_CAP * M, 48, 4]: tagged intermediate records
    epochs: cutlass.Array,  # int32 [NCTA]
    out: cutlass.Array,  # bf16 [M, 6144]
    route_dbg: cutlass.Array,  # int32 [M, 8]: ids, weight bits (CTA 0)
    tdbg: cutlass.Array,  # int64 [NCTA, 32] timing stamps (TIMING builds)
    xb32: cutlass.Array,  # FUSED_ROUTER: int32 words (bf16 pairs) of the hidden states [M, 6144]
    rw: cutlass.Array,  # FUSED_ROUTER: f32 router weight [128, 6144]
    sfsc: cutlass.Array,  # FUSED_ROUTER: f32 [1] fc31_input_scale
    lgbuf: cutlass.Array,  # FUSED_ROUTER: int64 [LG_REPL, LG_STRIDE] logits: tag << 32 | f32 bits
    guw: cutlass.Array,  # SHARED: uint8 gate_up e4m3 [1536, 6144]
    gusf: cutlass.Array,  # SHARED: uint8 its swizzled UE8M0 scales
    dnw: cutlass.Array,  # SHARED: uint8 down e4m3 [6144, 768]
    dnsf: cutlass.Array,  # SHARED: uint8 its swizzled UE8M0 scales
    hx: cutlass.Array,  # SHARED: int32 [M, 768]: h as (call tag << 16 | bf16), tag 0 = never written
    dnx: cutlass.Array,  # SHARED: int32 [M, 6144]: y_sh as (call tag << 16 | bf16), tag 0 = never written
    arp: cutlass.Array,  # int64 [8]: all-reduce push (enable, mc, flags, rank, world, maxt); enable 0 = off
    s1w: cutlass.Array,  # SFA_CP: int32 words of w3_w1_weight_scale
    s2w: cutlass.Array,  # SFA_CP: int32 words of w2_weight_scale
    img32: cutlass.Array,  # B2IMG: int32 [NG_CAP, IMG_BYTES / 4] per-group FC2 B image (tile + scale atoms)
    b2cnt: cutlass.Array,  # B2IMG: int32 [NG_CAP * 8 + 8] per-group unit counters (index 8 g)
):
    tidx, _, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = tidx % 32
    if tidx == 0:
        _stamp(tdbg, bx, 0)
        if cutlass.const_expr(TIMING):
            if tidx == 0:
                tdbg.store(
                    cutlass.Int64(_smid()), idx=bx * cutlass.Int32(TD_STRIDE) + cutlass.Int32(63)
                )

    s_big = cutlass.Array(cutlass.Int8, BIG, space=cutlass.AddressSpace.smem, alignment=1024)
    sA = cutlass.Array(
        s_big.data_ptr(0), shape=(NUM_BYTES_A * STAGES,), dtype=cutlass.Int8, alignment=1024
    )
    sSFAraw = cutlass.Array(
        s_big.data_ptr(_OFF_SFAR),
        shape=(max(1, _RAW_STAGE // 4) * STAGES,),
        dtype=cutlass.Int32,
        alignment=1024,
    )
    sSFA = cutlass.Array(
        s_big.data_ptr(_OFF_SFA),
        shape=(NUM_WORDS_SFA * STAGES,),
        dtype=cutlass.Int32,
        alignment=1024,
    )
    sB1 = cutlass.Array(
        s_big.data_ptr(_OFF_B1), shape=(NUM_BYTES_B1 * S1,), dtype=cutlass.Int8, alignment=1024
    )
    sB2 = cutlass.Array(
        s_big.data_ptr(_OFF_B2), shape=(2 * KT2 * NUM_BYTES_B2,), dtype=cutlass.Int8, alignment=1024
    )
    sSFB2 = cutlass.Array(
        s_big.data_ptr(_OFF_SFB2),
        shape=(2 * KT2 * NUM_KB * 128,),
        dtype=cutlass.Int32,
        alignment=1024,
    )
    ys = cutlass.Array(
        cutlass.Float32, NB2_CAP * M_MAX * TOP_K * R2, space=cutlass.AddressSpace.smem, alignment=16
    )
    s_part = cutlass.Array(
        cutlass.Float32, 2 * M_MAX * C1 * R1, space=cutlass.AddressSpace.smem, alignment=16
    )
    s_bias = cutlass.Array(cutlass.Float32, E_LOCAL, space=cutlass.AddressSpace.smem, alignment=16)
    s_red = cutlass.Array(cutlass.Float32, 9 * M_MAX, space=cutlass.AddressSpace.smem, alignment=16)
    s_xsf = cutlass.Array(
        cutlass.Int32,
        M_MAX * (H // 64) if FUSED_ROUTER else 4,
        space=cutlass.AddressSpace.smem,
        alignment=16,
    )
    ab_full = cutlass.Array(cutlass.Int64, STAGES, space=cutlass.AddressSpace.smem, alignment=8)
    ab_empty = cutlass.Array(cutlass.Int64, STAGES, space=cutlass.AddressSpace.smem, alignment=8)
    sfa_ready = cutlass.Array(cutlass.Int64, STAGES, space=cutlass.AddressSpace.smem, alignment=8)
    acc_full = cutlass.Array(cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8)
    acc_empty = cutlass.Array(cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8)
    acc2_full = cutlass.Array(cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8)
    acc2_empty = cutlass.Array(cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8)
    b2_ready = cutlass.Array(cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8)
    b2_free = cutlass.Array(cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8)
    # this CTA's FC1 share of group g is out: one barrier per group parity (a single barrier could complete twice
    # while warp 5 still works on an earlier group, and its parity wait would then miss them)
    fc1_done = cutlass.Array(cutlass.Int64, 4, space=cutlass.AddressSpace.smem, alignment=8)
    b1_ready = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    x_bar = cutlass.Array(
        cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8
    )  # x landed, x consumed
    img_bar = cutlass.Array(
        cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8
    )  # (B2IMG) image landed
    tmem_ptr_i32 = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem)
    s_el = cutlass.Array(
        cutlass.Int32, G_CAP + 1, space=cutlass.AddressSpace.smem, alignment=16
    )  # slots, then G
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
    # this CTA's plan rows for every G (static data: loaded before the grid dependency)
    s_misc = cutlass.Array(cutlass.Int32, 4, space=cutlass.AddressSpace.smem, alignment=16)
    s_plan_all = cutlass.Array(
        cutlass.Int32, (G_CAP + 1) * (NG_CAP + 1), space=cutlass.AddressSpace.smem, alignment=16
    )
    s_esc = cutlass.Array(
        cutlass.Float32, 8 * G_CAP, space=cutlass.AddressSpace.smem, alignment=32
    )  # slot s at 8 s

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
                prims.mbarrier_init(b2_ready.subview(tidx), 1)
                prims.mbarrier_init(b2_free.subview(tidx), EPI_THREADS)
                if cutlass.const_expr(B2IMG):
                    prims.mbarrier_init(img_bar.subview(tidx), 1)
            if tidx < 4:
                prims.mbarrier_init(fc1_done.subview(tidx), EPI_THREADS)
            if tidx == 0:
                prims.mbarrier_init(b1_ready.subview(0), EPI_THREADS)
                if cutlass.const_expr(XSTAGE):
                    prims.mbarrier_init(x_bar.subview(0), 1)
                    prims.mbarrier_init(x_bar.subview(1), EPI_THREADS)
        if warp == MMA_WARP:
            prims.tcgen05_alloc(tmem_ptr_i32, TMEM_COLS)
            prims.tcgen05_relinquish_alloc_permit()
        if warp == W_WARP:
            prims.prefetch_tensormap(tma_a1_desc.get_ptr())
            prims.prefetch_tensormap(tma_a2_desc.get_ptr())
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
        rwv = []
        if cutlass.const_expr(FUSED_ROUTER):
            tg = cutlass.select_(tidx < cutlass.Int32(GEMV_THREADS), tidx, cutlass.Int32(0))
            for rg in cutlass.range_constexpr(NGEMV):
                eg = bx + cutlass.Int32(rg * NCTA)
                rbx = cutlass.select_(eg < cutlass.Int32(NROUTE), eg, cutlass.Int32(0))
                rwv.append(
                    [
                        rw.load(
                            idx=rbx * cutlass.Int32(H)
                            + tg * cutlass.Int32(4)
                            + cutlass.Int32(1024 * j),
                            vector_size=4,
                            alignment=16,
                        )
                        for j in range(H // 1024)
                    ]
                )
        if cutlass.const_expr(TIMING):
            if tidx == 0:
                s_el.store(cutlass.Int32(-7), idx=G_CAP)
        prims.fence_mbarrier_init()
        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
        cute.arch.sync_warp()
        prims.barrier_cta_sync(0)
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        tmem_raw = tmem_ptr_i32.load()
        base_col = tmem_raw & 0xFFFF
        base_row = tmem_raw >> 16

        cute.arch.griddepcontrol_wait()
        if cutlass.const_expr(XSTAGE):
            if warp == 0:
                if lane == 0:
                    prims.mbarrier_arrive_expect_tx(x_bar.subview(0), M_MAX * H * 2)
                cute.arch.sync_warp()
                if lane < M_MAX:
                    prims.cp_async_bulk_shared_cluster_global(
                        s_big.subview(lane * cutlass.Int32(H * 2)),
                        xb32.subview(lane * cutlass.Int32(H // 2)),
                        x_bar.subview(0),
                        H * 2,
                    )
        ar_on = arp.load(idx=0) != cutlass.Int64(0)
        ar_off = arp.load(idx=0) == cutlass.Int64(0)
        if cutlass.const_expr(not LATE_TRIGGER):
            if ar_off:
                cute.arch.griddepcontrol_launch_dependents()
        if tidx == 0:
            _stamp(tdbg, bx, 1)

        ep = epochs.load(idx=bx)
        tagv = ((ep & cutlass.Int32(0x7F)) | cutlass.Int32(0x80)) << cutlass.Int32(24)
        tag16 = (ep & cutlass.Int32(0x7FFF)) | cutlass.Int32(0x8000)
        ltag = cutlass.Int64(ep + cutlass.Int32(1)) << cutlass.Int64(32)
        if cutlass.const_expr(FUSED_ROUTER):
            # ---- router GEMV (experts bx, bx + NCTA, .., threads 0..255), m3_router_gemv's order exactly
            for rg in cutlass.range_constexpr(NGEMV):
                eg = bx + cutlass.Int32(rg * NCTA)
                if (eg < cutlass.Int32(NROUTE)) & (tidx < cutlass.Int32(GEMV_THREADS)):
                    accs = [cutlass.Float32(0.0) for _ in range(M_MAX)]
                    xsrc = xb32
                    if cutlass.const_expr(GEMV_SMEM and XSTAGE):
                        _wait(x_bar.subview(0), 0)
                        xsrc = cutlass.Array(
                            s_big.data_ptr(0),
                            shape=(M_MAX * H // 2,),
                            dtype=cutlass.Int32,
                            alignment=16,
                        )
                    for j in cutlass.range_constexpr(H // 1024):
                        for t in cutlass.range_constexpr(M_MAX):
                            xw = xsrc.load(
                                idx=cutlass.Int32(t * (H // 2) + 512 * j) + tidx * cutlass.Int32(2),
                                vector_size=2,
                                alignment=8,
                            )
                            x0 = cutlass.Int32(xw[0])
                            x1 = cutlass.Int32(xw[1])
                            a = accs[t]
                            a = _fma_rn(
                                (x0 << cutlass.Int32(16)).bitcast(cutlass.Float32),
                                cutlass.Float32(rwv[rg][j][0]),
                                a,
                            )
                            a = _fma_rn(
                                (x0 & cutlass.Int32(-65536)).bitcast(cutlass.Float32),
                                cutlass.Float32(rwv[rg][j][1]),
                                a,
                            )
                            a = _fma_rn(
                                (x1 << cutlass.Int32(16)).bitcast(cutlass.Float32),
                                cutlass.Float32(rwv[rg][j][2]),
                                a,
                            )
                            a = _fma_rn(
                                (x1 & cutlass.Int32(-65536)).bitcast(cutlass.Float32),
                                cutlass.Float32(rwv[rg][j][3]),
                                a,
                            )
                            accs[t] = a
                    for t in cutlass.range_constexpr(M_MAX):
                        v = accs[t]
                        for off in (16, 8, 4, 2, 1):
                            v = _add_rn(v, cute.arch.shuffle_sync_bfly(v, off))
                        if lane == 0:
                            s_red.store(v, idx=warp * cutlass.Int32(M_MAX) + cutlass.Int32(t))
                    cute.arch.sync_warp()
                    cute.arch.barrier(barrier_id=GEMV_BAR, number_of_threads=GEMV_THREADS)
                    if tidx < M_MAX:
                        sacc = cutlass.Float32(0.0)
                        for w in cutlass.range_constexpr(8):
                            sacc = _add_rn(sacc, s_red.load(idx=cutlass.Int32(w * M_MAX) + tidx))
                        s_red.store(sacc, idx=cutlass.Int32(8 * M_MAX) + tidx)
                    cute.arch.sync_warp()
                    cute.arch.barrier(barrier_id=GEMV_BAR, number_of_threads=GEMV_THREADS)
                    if tidx < LG_REPL:
                        for t in cutlass.range_constexpr(M_MAX):
                            lw = ltag | (
                                cutlass.Int64(
                                    cutlass.Float32(s_red.load(idx=8 * M_MAX + t)).bitcast(
                                        cutlass.Int32
                                    )
                                )
                                & cutlass.Int64(0xFFFFFFFF)
                            )
                            _st_relaxed_u64(
                                lgbuf.subview(
                                    tidx * cutlass.Int32(LG_STRIDE)
                                    + cutlass.Int32(t * E_LOCAL)
                                    + eg
                                )
                                .data_ptr()
                                .toint(),
                                lw,
                            )
                    if cutlass.const_expr(NGEMV > 1):
                        cute.arch.sync_warp()
                        cute.arch.barrier(
                            barrier_id=GEMV_BAR, number_of_threads=GEMV_THREADS
                        )  # s_red reused next
                    if tidx == 0:
                        _stamp(tdbg, bx, 21)

        # ---- routing (warps 4..7, token t on warp 4 + t % 4); the epilogue warps build B meanwhile
        if ((warp >= W_WARP) & (warp < cutlass.Int32(W_WARP + NRW0))) | (
            (warp >= cutlass.Int32(8)) & (warp < cutlass.Int32(8 + RW_EXTRA))
        ):
            rw_i = cutlass.select_(
                warp < cutlass.Int32(8), warp - cutlass.Int32(W_WARP), warp - cutlass.Int32(4)
            )
            # ILP: poll every token's logits first, then run the independent top-k chains back to back
            lgv_all = []
            t_all = []
            for tr in cutlass.range_constexpr((M_MAX + NRW - 1) // NRW):
                t = rw_i + cutlass.Int32(NRW * tr)
                if cutlass.const_expr(M_MAX % NRW != 0):
                    t = cutlass.select_(t < cutlass.Int32(M_MAX), t, cutlass.Int32(M_MAX - 1))
                t_all.append(t)
                lgv = []
                if cutlass.const_expr(FUSED_ROUTER):
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
                        if prims.vote_sync(0xFFFFFFFF, ok, prims.VoteSync.BALLOT) == cutlass.Int32(
                            -1
                        ):
                            done = cutlass.Int32(1)
                        else:
                            _nanosleep(POLL_NS)
                    for z in cutlass.range_constexpr(4):
                        lgv.append(cutlass.Int32(res[z]).bitcast(cutlass.Float32))
                    if (lane == 0) & (warp == cutlass.Int32(W_WARP)):
                        _stamp(tdbg, bx, 22)
                else:
                    for i in cutlass.range_constexpr(4):
                        lgv.append(
                            cutlass.Float32(
                                logits.load(
                                    idx=t * cutlass.Int32(E_LOCAL) + cutlass.Int32(i * 32) + lane
                                )
                            )
                        )
                lgv_all.append(lgv)
            if (lane == 0) & (warp == cutlass.Int32(W_WARP)):
                _stamp(tdbg, bx, 61)
            res_all = []
            for tr in cutlass.range_constexpr((M_MAX + NRW - 1) // NRW):
                t = t_all[tr]
                lgv = lgv_all[tr]
                own = (rw_i + cutlass.Int32(NRW * tr)) < cutlass.Int32(M_MAX)
                # top-4 (keys sigmoid + bias, ties to the lower id)
                keys = []
                for i in cutlass.range_constexpr(4):
                    sg = _fma_rn(
                        cutlass.Float32(0.5),
                        cute.math.tanh(_mul_rn(cutlass.Float32(0.5), lgv[i])),
                        cutlass.Float32(0.5),
                    )
                    keys.append(_add_rn(sg, s_bias.load(idx=i * 32 + lane)))
                hi = [_twiddle(keys[i]) for i in range(4)]
                taken = [cutlass.Boolean(False) for _ in range(4)]
                ids_t = []
                kt_t = []
                for k in cutlass.range_constexpr(TOP_K):
                    bh = cutlass.Int32(0)
                    bl = cutlass.Int32(0)
                    for i in cutlass.range_constexpr(4):
                        lo_i = cutlass.Int32(65535 - i * 32) - lane
                        hv = cutlass.select_(taken[i], cutlass.Int32(0), hi[i])
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
                res_all.append((t, own, wv, e_l))
            if (lane == 0) & (warp == cutlass.Int32(W_WARP)):
                _stamp(tdbg, bx, 62)
            for tr in cutlass.range_constexpr(
                (M_MAX + NRW - 1) // NRW
            ):  # stores after every chain (one branch-free block)
                t, own, wv, e_l = res_all[tr]
                if (lane < TOP_K) & own:
                    s_w.store(wv, idx=t * cutlass.Int32(TOP_K) + lane)
                    s_ids.store(e_l, idx=t * cutlass.Int32(TOP_K) + lane)
                    if bx == 0:
                        route_dbg.store(
                            wv.bitcast(cutlass.Int32),
                            idx=t * cutlass.Int32(8) + cutlass.Int32(4) + lane,
                        )
            cute.arch.sync_warp()
            if cutlass.const_expr(EARLY0 and not E0_OFFLOAD):
                if warp == cutlass.Int32(W_WARP):
                    # slots 0..3 are always token 0's four (distinct) experts and group 0's stage ranges are the same
                    # for every G (the plan guarantees it): fill the ring with this CTA's first group-0 stages now
                    if cutlass.const_expr(XSTAGE):
                        _wait(x_bar.subview(1), 0)
                    if lane == 0:
                        _stamp(tdbg, bx, 56)
                    n0 = cutlass.Int32(0)
                    w0 = s_plan_all.load(idx=G_CAP * (NG_CAP + 1))
                    ei0 = w0 & cutlass.Int32(0xFFFF)
                    ei1 = w0 >> cutlass.Int32(16)
                    for k in range(_nsegs(ei0, ei1)):
                        ug, j0, j1 = _seg(ei0, ei1, k)
                        c = ug - (ug // cutlass.Int32(U1)) * cutlass.Int32(U1)
                        e = s_ids.load(idx=ug // cutlass.Int32(U1))
                        sf_line = (
                            (e * cutlass.Int32(SF1_RB) + (c >> cutlass.Int32(2)))
                            * cutlass.Int32(SF1_KA)
                            + j0 * cutlass.Int32(C1 * NUM_KB)
                        ) * cutlass.Int32(4)
                        kx = j0 * cutlass.Int32(C1 * KT_BYTES)
                        row = c * cutlass.Int32(R1)
                        for j in cutlass.range(j0, j1, unroll=1):
                            if n0 < cutlass.Int32(min(STAGES, EARLY0_N)):
                                if prims.elect_sync():
                                    prims.mbarrier_arrive_expect_tx(
                                        ab_full.subview(n0),
                                        NUM_BYTES_A if SFA_CP else NUM_BYTES_A + NUM_BYTES_SFA_RAW,
                                    )
                                    for q in cutlass.range_constexpr(C1):
                                        prims.cp_async_bulk_tensor_shared_cta_global(
                                            sA.subview(
                                                n0 * cutlass.Int32(NUM_BYTES_A)
                                                + cutlass.Int32(q * NUM_BYTES_BOX1)
                                            ),
                                            tma_a1_desc.get_ptr(),
                                            (kx + cutlass.Int32(q * KT_BYTES), row, e),
                                            ab_full.subview(n0),
                                            l2_cache_hint=EVICT_FIRST,
                                        )
                                    if cutlass.const_expr(not SFA_CP):
                                        prims.cp_async_bulk_tensor_shared_cta_global(
                                            sSFAraw.subview(n0 * cutlass.Int32(NUM_WORDS_SFA_RAW)),
                                            tma_s1_desc.get_ptr(),
                                            (cutlass.Int32(0), sf_line),
                                            ab_full.subview(n0),
                                        )
                                if cutlass.const_expr(SFA_CP):
                                    _sf1_cp(lane, n0, sf_line, c, sSFA, s1w, ab_full)
                                n0 = n0 + cutlass.Int32(1)
                            kx = kx + cutlass.Int32(C1 * KT_BYTES)
                            sf_line = sf_line + cutlass.Int32(NUM_BYTES_SFA_RAW // 128)
                    if lane == 0:
                        s_misc.store(n0, idx=0)
                    cute.arch.sync_warp()
            if cutlass.const_expr(E0_OFFLOAD):
                if warp == cutlass.Int32(W_WARP):
                    _bar_arrive(
                        E0_BAR, 64
                    )  # token 0's experts are in s_ids: warp 7 issues the early stages
            if cutlass.const_expr(B1_AFTER_TOPK):
                _bar_arrive(TOPK2_BAR, 128 + 32 * NRW)
            cute.arch.barrier(barrier_id=TOPK_BAR, number_of_threads=32 * NRW)
        if cutlass.const_expr(E0_OFFLOAD):
            if warp == cutlass.Int32(E0_WARP):
                cute.arch.barrier(barrier_id=E0_BAR, number_of_threads=64)
                # slots 0..3 are always token 0's four (distinct) experts and group 0's stage ranges are the same
                # for every G (the plan guarantees it): fill the ring with this CTA's first group-0 stages now
                if cutlass.const_expr(XSTAGE):
                    _wait(x_bar.subview(1), 0)
                if lane == 0:
                    _stamp(tdbg, bx, 56)
                n0 = cutlass.Int32(0)
                w0 = s_plan_all.load(idx=G_CAP * (NG_CAP + 1))
                ei0 = w0 & cutlass.Int32(0xFFFF)
                ei1 = w0 >> cutlass.Int32(16)
                for k in range(_nsegs(ei0, ei1)):
                    ug, j0, j1 = _seg(ei0, ei1, k)
                    c = ug - (ug // cutlass.Int32(U1)) * cutlass.Int32(U1)
                    e = s_ids.load(idx=ug // cutlass.Int32(U1))
                    sf_line = (
                        (e * cutlass.Int32(SF1_RB) + (c >> cutlass.Int32(2)))
                        * cutlass.Int32(SF1_KA)
                        + j0 * cutlass.Int32(C1 * NUM_KB)
                    ) * cutlass.Int32(4)
                    kx = j0 * cutlass.Int32(C1 * KT_BYTES)
                    row = c * cutlass.Int32(R1)
                    for j in cutlass.range(j0, j1, unroll=1):
                        if n0 < cutlass.Int32(min(STAGES, EARLY0_N)):
                            if prims.elect_sync():
                                prims.mbarrier_arrive_expect_tx(
                                    ab_full.subview(n0),
                                    NUM_BYTES_A if SFA_CP else NUM_BYTES_A + NUM_BYTES_SFA_RAW,
                                )
                                for q in cutlass.range_constexpr(C1):
                                    prims.cp_async_bulk_tensor_shared_cta_global(
                                        sA.subview(
                                            n0 * cutlass.Int32(NUM_BYTES_A)
                                            + cutlass.Int32(q * NUM_BYTES_BOX1)
                                        ),
                                        tma_a1_desc.get_ptr(),
                                        (kx + cutlass.Int32(q * KT_BYTES), row, e),
                                        ab_full.subview(n0),
                                        l2_cache_hint=EVICT_FIRST,
                                    )
                                if cutlass.const_expr(not SFA_CP):
                                    prims.cp_async_bulk_tensor_shared_cta_global(
                                        sSFAraw.subview(n0 * cutlass.Int32(NUM_WORDS_SFA_RAW)),
                                        tma_s1_desc.get_ptr(),
                                        (cutlass.Int32(0), sf_line),
                                        ab_full.subview(n0),
                                    )
                            if cutlass.const_expr(SFA_CP):
                                _sf1_cp(lane, n0, sf_line, c, sSFA, s1w, ab_full)
                            n0 = n0 + cutlass.Int32(1)
                        kx = kx + cutlass.Int32(C1 * KT_BYTES)
                        sf_line = sf_line + cutlass.Int32(NUM_BYTES_SFA_RAW // 128)
                if lane == 0:
                    s_misc.store(n0, idx=0)
                cute.arch.sync_warp()
                _bar_arrive(E1_BAR, 64)
        if warp == W_WARP:
            if lane == 0:
                _stamp(tdbg, bx, 20)
            # slots: lane l < 4 M is (token l / 4, top-k l % 4); the distinct experts in lane order (match.any)
            valid = lane < cutlass.Int32(TOP_K * M_MAX)
            my_e = cutlass.select_(
                valid,
                s_ids.load(idx=cutlass.select_(valid, lane, cutlass.Int32(0))),
                cutlass.Int32(-1) - lane,
            )
            same = _match_any(my_e)
            leader = cute.arch.popc((same & (cutlass.Int32(0) - same)) - cutlass.Int32(1))
            first = valid & (lane == leader)
            bal = prims.vote_sync(0xFFFFFFFF, first, prims.VoteSync.BALLOT)
            rank = cute.arch.popc(bal & cutlass.Int32(cute.arch.lanemask_lt()))
            slot_l = cute.arch.shuffle_sync(rank, leader)
            for z in cutlass.range_constexpr((G_CAP * M_MAX + 31) // 32):
                if lane + cutlass.Int32(32 * z) < cutlass.Int32(G_CAP * M_MAX):
                    s_kpos.store(cutlass.Int32(-1), idx=lane + cutlass.Int32(32 * z))
            cute.arch.sync_warp()
            if first:
                s_el.store(my_e, idx=rank)
            if valid:
                s_kpos.store(
                    lane % cutlass.Int32(TOP_K),
                    idx=slot_l * cutlass.Int32(M_MAX) + lane // cutlass.Int32(TOP_K),
                )
                if bx == 0:
                    route_dbg.store(
                        my_e,
                        idx=(lane // cutlass.Int32(TOP_K)) * cutlass.Int32(8)
                        + lane % cutlass.Int32(TOP_K),
                    )
            gcnt = cute.arch.popc(bal)
            if lane == 0:
                s_el.store(gcnt, idx=G_CAP)
            if lane < cutlass.Int32(NG_CAP + 1):  # the plan row of this G (preloaded)
                s_plan.store(s_plan_all.load(idx=gcnt * cutlass.Int32(NG_CAP + 1) + lane), idx=lane)
            cute.arch.sync_warp()
            if cutlass.const_expr(B2IMG):
                # CTA 0 pads every group counter to exactly CNT_PER_CALL for this call (absent slots, unused groups)
                if (bx == 0) & (lane < cutlass.Int32(NG_CAP)):
                    nh = gcnt - lane * cutlass.Int32(4)
                    nh = cutlass.select_(nh > cutlass.Int32(4), cutlass.Int32(4), nh)
                    nh = cutlass.select_(nh < cutlass.Int32(0), cutlass.Int32(0), nh)
                    pad = cutlass.Int32(CNT_PER_CALL) - nh * cutlass.Int32(UNITS_ROW)
                    if pad > cutlass.Int32(0):
                        _red_release_add(
                            b2cnt.subview(lane * cutlass.Int32(8)).data_ptr().toint(), pad
                        )
                cute.arch.sync_warp()
            if lane == 0:
                _stamp(tdbg, bx, 24)
            cute.arch.sync_warp()
            _bar_arrive(ROUTE_BAR, THREADS)
        if warp < 4:
            if cutlass.const_expr(B1_AFTER_TOPK):
                cute.arch.barrier(barrier_id=TOPK2_BAR, number_of_threads=128 + 32 * NRW)
            if cutlass.const_expr(FUSED_ROUTER):
                if tidx == 0:
                    _stamp(tdbg, bx, 57)
                if cutlass.const_expr(XSTAGE):
                    _wait(x_bar.subview(0), 0)
                    if tidx == 0:
                        _stamp(tdbg, bx, 58)
                    xs32 = cutlass.Array(
                        s_big.data_ptr(0),
                        shape=(M_MAX * H // 2,),
                        dtype=cutlass.Int32,
                        alignment=16,
                    )
                    _build_b1_fused(
                        tidx, warp, lane, base_row, base_col, xs32, sfsc, sB1, s_xsf, b1_ready
                    )
                    prims.mbarrier_arrive(x_bar.subview(1))  # the ring is free for the weights
                else:
                    _build_b1_fused(
                        tidx, warp, lane, base_row, base_col, xb32, sfsc, sB1, s_xsf, b1_ready
                    )
                if tidx == 0:
                    _stamp(tdbg, bx, 23)
            else:
                _build_b1(tidx, warp, lane, base_row, base_col, x32, xsf32, sB1, b1_ready)
        if warp != W_WARP:  # one sync site for every thread but warp 4's (which arrived)
            cute.arch.sync_warp()
            cute.arch.barrier(barrier_id=ROUTE_BAR, number_of_threads=THREADS)
        if tidx == 0:
            _stamp(tdbg, bx, 2)
            if cutlass.const_expr(TIMING):
                tdbg.store(
                    cutlass.Int64(s_el.load(idx=G_CAP)),
                    idx=bx * cutlass.Int32(TD_STRIDE) + cutlass.Int32(28),
                )

        g_n = s_el.load(idx=G_CAP)
        ng = (g_n + cutlass.Int32(3)) >> cutlass.Int32(2)
        pr2 = s_plan.load(idx=NG_CAP)
        blk0 = pr2 & cutlass.Int32(0xFFFF)
        nb = pr2 >> cutlass.Int32(16)

        # ---- weights producer (warp 4): phase p = FC1 stages of group p, then FC2 tiles of group p - LAG
        if warp == W_WARP:
            if cutlass.const_expr(XSTAGE):
                _wait(x_bar.subview(1), 0)  # the staged hidden states are quantized
            stage = cutlass.Int32(0)
            phase = cutlass.Int32(1)
            skip = cutlass.Int32(0)
            if cutlass.const_expr(E0_OFFLOAD):
                cute.arch.barrier(barrier_id=E1_BAR, number_of_threads=64)
            if cutlass.const_expr(EARLY0):
                skip = s_misc.load(idx=0)  # group-0 stages already in the ring
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
                        sf_line = (
                            (e * cutlass.Int32(SF1_RB) + (c >> cutlass.Int32(2)))
                            * cutlass.Int32(SF1_KA)
                            + j0 * cutlass.Int32(C1 * NUM_KB)
                        ) * cutlass.Int32(4)
                        kx = j0 * cutlass.Int32(C1 * KT_BYTES)
                        row = c * cutlass.Int32(R1)
                        for j in cutlass.range(j0, j1, unroll=1):
                            if skip > cutlass.Int32(0):
                                skip = skip - cutlass.Int32(1)
                            else:
                                _waitd(ab_empty.subview(stage), phase, tdbg, 1)
                                if prims.elect_sync():
                                    prims.mbarrier_arrive_expect_tx(
                                        ab_full.subview(stage),
                                        NUM_BYTES_A if SFA_CP else NUM_BYTES_A + NUM_BYTES_SFA_RAW,
                                    )
                                    for q in cutlass.range_constexpr(C1):
                                        prims.cp_async_bulk_tensor_shared_cta_global(
                                            sA.subview(
                                                stage * cutlass.Int32(NUM_BYTES_A)
                                                + cutlass.Int32(q * NUM_BYTES_BOX1)
                                            ),
                                            tma_a1_desc.get_ptr(),
                                            (kx + cutlass.Int32(q * KT_BYTES), row, e),
                                            ab_full.subview(stage),
                                            l2_cache_hint=EVICT_FIRST,
                                        )
                                    if cutlass.const_expr(not SFA_CP):
                                        prims.cp_async_bulk_tensor_shared_cta_global(
                                            sSFAraw.subview(
                                                stage * cutlass.Int32(NUM_WORDS_SFA_RAW)
                                            ),
                                            tma_s1_desc.get_ptr(),
                                            (cutlass.Int32(0), sf_line),
                                            ab_full.subview(stage),
                                        )
                                if cutlass.const_expr(SFA_CP):
                                    _sf1_cp(lane, stage, sf_line, c, sSFA, s1w, ab_full)
                                stage, phase = _adv(stage, phase)
                            kx = kx + cutlass.Int32(C1 * KT_BYTES)
                            sf_line = sf_line + cutlass.Int32(NUM_BYTES_SFA_RAW // 128)
                if p >= cutlass.Int32(LAG):
                    g = p - cutlass.Int32(LAG)
                    n_here = g_n - g * cutlass.Int32(4)
                    n_here = cutlass.select_(n_here > cutlass.Int32(4), cutlass.Int32(4), n_here)
                    for bb in range(nb):
                        blk = blk0 + bb
                        for t2 in cutlass.range_constexpr(KT2):
                            _waitd(ab_empty.subview(stage), phase, tdbg, 2)
                            if prims.elect_sync():
                                prims.mbarrier_arrive_expect_tx(
                                    ab_full.subview(stage),
                                    n_here
                                    * cutlass.Int32(
                                        NUM_BYTES_BOX1
                                        if SFA_CP
                                        else NUM_BYTES_BOX1 + NUM_BYTES_SFA_RAW // 4
                                    ),
                                )
                                for jj in cutlass.range_constexpr(4):
                                    if cutlass.Int32(jj) < n_here:
                                        e = s_el.load(idx=g * cutlass.Int32(4) + cutlass.Int32(jj))
                                        prims.cp_async_bulk_tensor_shared_cta_global(
                                            sA.subview(
                                                stage * cutlass.Int32(NUM_BYTES_A)
                                                + cutlass.Int32(jj * NUM_BYTES_BOX1)
                                            ),
                                            tma_a2_desc.get_ptr(),
                                            (
                                                cutlass.Int32(t2 * KT_BYTES),
                                                blk * cutlass.Int32(R2),
                                                e,
                                            ),
                                            ab_full.subview(stage),
                                            l2_cache_hint=EVICT_FIRST,
                                        )
                                        if cutlass.const_expr(not SFA_CP):
                                            prims.cp_async_bulk_tensor_shared_cta_global(
                                                sSFAraw.subview(
                                                    stage * cutlass.Int32(NUM_WORDS_SFA_RAW)
                                                    + cutlass.Int32(jj * NUM_WORDS_SFA_RAW // 4)
                                                ),
                                                tma_s2_desc.get_ptr(),
                                                (
                                                    cutlass.Int32(0),
                                                    (
                                                        (
                                                            e * cutlass.Int32(SF2_RB)
                                                            + (blk >> cutlass.Int32(2))
                                                        )
                                                        * cutlass.Int32(SF2_KA)
                                                        + cutlass.Int32(4 * t2)
                                                    )
                                                    * cutlass.Int32(4),
                                                ),
                                                ab_full.subview(stage),
                                            )
                            if cutlass.const_expr(SFA_CP):
                                _sf2_cp(lane, stage, g, blk, t2, n_here, s_el, sSFA, s2w, ab_full)
                            stage, phase = _adv(stage, phase)
            if lane == 0:
                _stamp(tdbg, bx, 3)

        # ---- scale splicer (warp 6): every stage, atom kb word [r][q] = raw atom 4 q + kb word [r][m1]
        if (warp == S_WARP) & cutlass.Boolean(not SFA_CP):  # (SFA_CP: no splicer)
            s2t_shape, s2t_mc = prims.S2TCopyMode.S2T_32x128b_WARPX4
            stage = cutlass.Int32(0)
            phase = cutlass.Int32(0)
            for p in range(ng + cutlass.Int32(LAG)):
                i0 = cutlass.Int32(0)
                i1 = cutlass.Int32(0)
                for part in cutlass.range_constexpr(2):
                    cnt = cutlass.Int32(0)
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
                            cnt = j1 - j0
                        else:
                            m1 = (blk0 + k) & cutlass.Int32(3)
                            cnt = cutlass.Int32(KT2)
                        for _j in cutlass.range(cnt, unroll=1):
                            _waitd(ab_full.subview(stage), phase, tdbg, 3)
                            raw0 = (
                                stage * cutlass.Int32(NUM_WORDS_SFA_RAW)
                                + lane * cutlass.Int32(4)
                                + m1
                            )
                            dst0 = stage * cutlass.Int32(NUM_WORDS_SFA) + lane * cutlass.Int32(4)
                            for kb in cutlass.range_constexpr(NUM_KB):
                                w0 = sSFAraw.load(idx=raw0 + cutlass.Int32((0 * 4 + kb) * 128))
                                w1 = sSFAraw.load(idx=raw0 + cutlass.Int32((1 * 4 + kb) * 128))
                                w2 = sSFAraw.load(idx=raw0 + cutlass.Int32((2 * 4 + kb) * 128))
                                w3 = sSFAraw.load(idx=raw0 + cutlass.Int32((3 * 4 + kb) * 128))
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
                            prims.fence_proxy("async_shared", space=prims.SharedSpace.shared_cta)
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

        if cutlass.const_expr(B2IMG):
            # ---- (B2IMG) group releases (warp 6): this CTA's finished units of group p, one release add per group
            if warp == cutlass.Int32(S_WARP):
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
                        fc1_done.subview(p & cutlass.Int32(3)),
                        (p >> cutlass.Int32(2)) & cutlass.Int32(1),
                    )
                    if cutlass.const_expr(B2_DLY > 0):
                        t0d = _gtime()
                        dl = cutlass.Int64(
                            (
                                (bx * cutlass.Int32(37) + p * cutlass.Int32(11) + ep)
                                & cutlass.Int32(63)
                            )
                            * cutlass.Int32(B2_DLY // 64)
                        )
                        while _gtime() - t0d < dl:
                            pass
                    if (lane == 0) & (n_done > cutlass.Int32(0)):
                        _fence_proxy_async_global()
                        _red_release_add(
                            b2cnt.subview(p * cutlass.Int32(8)).data_ptr().toint(), n_done
                        )
                    cute.arch.sync_warp()
            # ---- (B2IMG) FC2 B loader (warp 5): group g's image once its counter is complete
            if warp == cutlass.Int32(X_WARP):
                s2t_shape, s2t_mc = prims.S2TCopyMode.S2T_32x128b_WARPX4
                cnt_target = (ep + cutlass.Int32(1)) * cutlass.Int32(CNT_PER_CALL)
                for g in range(ng):
                    buf = g & cutlass.Int32(1)
                    if g >= cutlass.Int32(2):
                        _waitd(
                            b2_free.subview(buf),
                            ((g >> cutlass.Int32(1)) - cutlass.Int32(1)) & cutlass.Int32(1),
                            tdbg,
                            101,
                        )
                    if lane == 0:
                        _stamp(tdbg, bx, 29)
                    caddr = b2cnt.subview(g * cutlass.Int32(8)).data_ptr().toint()
                    if lane == 0:
                        cv = _ld_acquire_u32(caddr)
                        while (cv - cnt_target) < cutlass.Int32(0):
                            _nanosleep(B2IMG_NS)
                            cv = _ld_acquire_u32(caddr)
                        _fence_proxy_async_global()
                        prims.mbarrier_arrive_expect_tx(img_bar.subview(buf), IMG_BYTES)
                        prims.cp_async_bulk_shared_cluster_global(
                            sB2.subview(buf * cutlass.Int32(IMG_B2)),
                            img32.subview(g * cutlass.Int32(IMG_BYTES // 4)),
                            img_bar.subview(buf),
                            IMG_B2,
                        )
                        prims.cp_async_bulk_shared_cluster_global(
                            sSFB2.subview(buf * cutlass.Int32(IMG_SF // 4)),
                            img32.subview(
                                g * cutlass.Int32(IMG_BYTES // 4) + cutlass.Int32(IMG_B2 // 4)
                            ),
                            img_bar.subview(buf),
                            IMG_SF,
                        )
                    cute.arch.sync_warp()
                    _waitd(
                        img_bar.subview(buf), (g >> cutlass.Int32(1)) & cutlass.Int32(1), tdbg, 108
                    )
                    if prims.elect_sync():
                        sd0 = prims.Tcgen05SmemDesc.build(
                            sSFB2.subview(0),
                            leading_byte_offset=16,
                            stride_byte_offset=128,
                            base_offset=0,
                            layout=0,
                        )
                        for a in cutlass.range_constexpr(KT2 * NUM_KB):
                            prims.tcgen05_cp(
                                s2t_shape,
                                _tptr(
                                    base_row,
                                    base_col
                                    + SFB2_COL
                                    + buf * cutlass.Int32(KT2 * NUM_KB * 4)
                                    + a * 4,
                                ),
                                sd0
                                + cutlass.Int64(
                                    (buf * cutlass.Int32(KT2 * NUM_KB) + cutlass.Int32(a))
                                    * cutlass.Int32(512 >> 4)
                                ),
                                multicast=s2t_mc,
                            )
                        prims.tcgen05_commit(b2_ready.subview(buf))
                    cute.arch.sync_warp()
                    if lane == 0:
                        _stamp(tdbg, bx, 25)
        # ---- FC2 B builder (warps 5, 8 ..): group g's records (all 48 per (slot, token), the call's tag on
        # every word) into B2 buffer g % 2 (row j M + t, k-tile t2 = record / 16) and its scale atoms, then
        # tcgen05.cp to TMEM.
        if ((warp == X_WARP) | (warp >= cutlass.Int32(8))) & cutlass.Boolean(not B2IMG):
            xi = cutlass.select_(
                warp == cutlass.Int32(X_WARP), cutlass.Int32(0), warp - cutlass.Int32(7)
            )
            if cutlass.const_expr(G_CAP > TOP_K):
                if (bx == 0) & (xi == cutlass.Int32(0)):
                    # rows of slots this call does not use: tag 0, never taken for a later call's
                    for i in range(
                        g_n * cutlass.Int32(M_MAX * UNITS_ROW) + lane,
                        cutlass.Int32(G_CAP * M_MAX * UNITS_ROW),
                        32,
                    ):
                        hbuf32.store(
                            (
                                cutlass.Int32(0),
                                cutlass.Int32(0),
                                cutlass.Int32(0),
                                cutlass.Int32(0),
                            ),
                            idx=i * cutlass.Int32(4),
                            alignment=16,
                        )
            s2t_shape, s2t_mc = prims.S2TCopyMode.S2T_32x128b_WARPX4
            sSFB2_8 = cutlass.Array(
                sSFB2.data_ptr(0), shape=(2 * KT2 * NUM_KB * 512,), dtype=cutlass.Int8, alignment=16
            )
            sB2_32 = cutlass.Array(
                sB2.data_ptr(0),
                shape=(2 * KT2 * NUM_BYTES_B2 // 4,),
                dtype=cutlass.Int32,
                alignment=16,
            )
            nrec_cap = 4 * M_MAX * UNITS_ROW  # records of one group
            NZ = (nrec_cap + 32 * XW - 1) // (32 * XW)  # records per lane: lane + 32 (XW z + xi)
            for g in range(ng):
                buf = g & cutlass.Int32(1)
                if g >= cutlass.Int32(2):
                    _waitd(
                        b2_free.subview(buf),
                        ((g >> cutlass.Int32(1)) - cutlass.Int32(1)) & cutlass.Int32(1),
                        tdbg,
                        101,
                    )
                _waitd(
                    fc1_done.subview(g & cutlass.Int32(3)),
                    (g >> cutlass.Int32(2)) & cutlass.Int32(1),
                    tdbg,
                    4,
                )  # this CTA's own share of group g is out
                if cutlass.const_expr(TIMING):
                    if (lane == 0) & (xi == cutlass.Int32(0)) & (g < cutlass.Int32(8)):
                        tdbg.store(
                            _gtime(), idx=bx * cutlass.Int32(TD_STRIDE) + cutlass.Int32(32) + g
                        )
                n_here = g_n - g * cutlass.Int32(4)
                n_here = cutlass.select_(n_here > cutlass.Int32(4), cutlass.Int32(4), n_here)
                nrec = n_here * cutlass.Int32(M_MAX * UNITS_ROW)
                rec0 = g * cutlass.Int32(
                    4 * M_MAX * UNITS_ROW
                )  # rows (4 g + j) M + t are contiguous
                done = cutlass.Int32(0)
                rspin = cutlass.Int32(0)
                if (lane == 0) & (xi == cutlass.Int32(0)):
                    _stamp(tdbg, bx, 29)
                npoll = cutlass.Int32(0)
                got = [[cutlass.Int32(0) for _ in range(3)] for _ in range(NZ)]
                while done == cutlass.Int32(0):
                    ok = cutlass.Boolean(True)
                    for z in cutlass.range_constexpr(NZ):
                        qi = lane + cutlass.Int32(32 * XW * z) + cutlass.Int32(32) * xi
                        if qi < nrec:
                            v4 = prims.load_ext(
                                hbuf32.subview((rec0 + qi) * cutlass.Int32(4)),
                                dtype=cutlass.Int32,
                                count=4,
                                order="relaxed",
                                scope="gpu",
                            )
                            for wq in cutlass.range_constexpr(3):
                                got[z][wq] = cutlass.Int32(v4[wq])
                            ok = ok & (
                                (
                                    ((got[z][0] ^ tagv) | (got[z][1] ^ tagv) | (got[z][2] ^ tagv))
                                    & cutlass.Int32(-16777216)
                                )
                                == cutlass.Int32(0)
                            )
                    npoll = npoll + cutlass.Int32(1)
                    if prims.vote_sync(0xFFFFFFFF, ok, prims.VoteSync.BALLOT) == cutlass.Int32(-1):
                        done = cutlass.Int32(1)
                    else:
                        _nanosleep(B2_POLL_NS)
                        if cutlass.const_expr(HANG_DBG):
                            rspin = rspin + cutlass.Int32(1)
                            if rspin > cutlass.Int32(1 << 14):
                                done = cutlass.Int32(1)
                                tdbg.store(
                                    cutlass.Int64(200),
                                    idx=bx * cutlass.Int32(TD_STRIDE) + cutlass.Int32(24) + warp,
                                )
                for z in cutlass.range_constexpr(NZ):
                    qi = lane + cutlass.Int32(32 * XW * z) + cutlass.Int32(32) * xi
                    if qi < nrec:
                        w0, w1, w2 = got[z][0], got[z][1], got[z][2]
                        rr = qi // cutlass.Int32(
                            UNITS_ROW
                        )  # row within the group: j M + t = B row n
                        c = qi - rr * cutlass.Int32(UNITS_ROW)
                        t2 = c >> cutlass.Int32(4)
                        cc = (c >> cutlass.Int32(1)) & cutlass.Int32(7)
                        lo = (w0 & cutlass.Int32(0xFFFFFF)) | (w1 << cutlass.Int32(24))
                        hi = ((w1 >> cutlass.Int32(8)) & cutlass.Int32(0xFFFF)) | (
                            (w2 & cutlass.Int32(0xFFFF)) << cutlass.Int32(16)
                        )
                        sB2_32.store(
                            (lo, hi),
                            idx=(
                                (buf * cutlass.Int32(KT2) + t2) * cutlass.Int32(NUM_BYTES_B2)
                                + rr * cutlass.Int32(KT_BYTES)
                                + (cc ^ (rr & cutlass.Int32(7))) * cutlass.Int32(16)
                                + (c & cutlass.Int32(1)) * cutlass.Int32(8)
                            )
                            // cutlass.Int32(4),
                            alignment=8,
                        )
                        # scale of block c: atom (t2, kb = (c % 16) / 4), row rr, byte c % 4
                        sSFB2_8.store(
                            cutlass.Int8((w2 >> cutlass.Int32(16)) & cutlass.Int32(0xFF)),
                            idx=(
                                (
                                    (buf * cutlass.Int32(KT2) + t2) * cutlass.Int32(NUM_KB)
                                    + ((c & cutlass.Int32(15)) >> cutlass.Int32(2))
                                )
                                * cutlass.Int32(128)
                                + rr * cutlass.Int32(4)
                            )
                            * cutlass.Int32(4)
                            + (c & cutlass.Int32(3)),
                        )
                prims.fence_proxy("async_shared", space=prims.SharedSpace.shared_cta)
                if cutlass.const_expr(XW > 1):
                    cute.arch.sync_warp()
                    cute.arch.barrier(barrier_id=B2_BAR, number_of_threads=32 * XW)
                cute.arch.sync_warp()
                if (xi == cutlass.Int32(0)) & prims.elect_sync():
                    sd0 = prims.Tcgen05SmemDesc.build(
                        sSFB2.subview(0),
                        leading_byte_offset=16,
                        stride_byte_offset=128,
                        base_offset=0,
                        layout=0,
                    )
                    for a in cutlass.range_constexpr(KT2 * NUM_KB):
                        prims.tcgen05_cp(
                            s2t_shape,
                            _tptr(
                                base_row,
                                base_col + SFB2_COL + buf * cutlass.Int32(KT2 * NUM_KB * 4) + a * 4,
                            ),
                            sd0
                            + cutlass.Int64(
                                (buf * cutlass.Int32(KT2 * NUM_KB) + cutlass.Int32(a))
                                * cutlass.Int32(512 >> 4)
                            ),
                            multicast=s2t_mc,
                        )
                    prims.tcgen05_commit(b2_ready.subview(buf))
                if (lane == 0) & (xi == cutlass.Int32(0)):
                    _stamp(tdbg, bx, 25)
                    if cutlass.const_expr(TIMING):
                        if g < cutlass.Int32(8):
                            tdbg.store(
                                _gtime(), idx=bx * cutlass.Int32(TD_STRIDE) + cutlass.Int32(11) + g
                            )
                    if cutlass.const_expr(TIMING):
                        tdbg.store(
                            cutlass.Int64(npoll),
                            idx=bx * cutlass.Int32(TD_STRIDE) + cutlass.Int32(30),
                        )

        # ---- MMA (warp 7)
        if warp == MMA_WARP:
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
            acc_b2 = cutlass.Int64(0)
            acc_sf = cutlass.Int64(0)
            for p in range(ng + cutlass.Int32(LAG)):
                if p < ng:
                    i0, i1 = _plan_range(s_plan, p)
                    for k in range(_nsegs(i0, i1)):
                        ug, j0, j1 = _seg(i0, i1, k)
                        ab = k1 & cutlass.Int32(1)
                        _waitd(
                            acc_empty.subview(ab),
                            ((k1 >> cutlass.Int32(1)) & cutlass.Int32(1)) ^ cutlass.Int32(1),
                            tdbg,
                            102,
                        )
                        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                        acc_ptr = cutlass.inttoptr(
                            tcol + ACC1_COL + ab * cutlass.Int32(N1), 6, cutlass.Float32
                        )
                        for j in cutlass.range(j0, j1, unroll=1):
                            tw0 = _gtime() if cutlass.const_expr(TIMING) else cutlass.Int64(0)
                            if cutlass.const_expr(SFA_CP):
                                _waitd(ab_full.subview(stage), phase, tdbg, 6)
                                prims.fence_proxy(
                                    "async_shared", space=prims.SharedSpace.shared_cta
                                )
                            else:
                                _waitd(sfa_ready.subview(stage), phase, tdbg, 6)
                            if cutlass.const_expr(TIMING):
                                acc_sf = acc_sf + (_gtime() - tw0)
                            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                            da = desc_a0 + cutlass.Int64(stage * cutlass.Int32(NUM_BYTES_A >> 4))
                            db = desc_b10 + cutlass.Int64(j * cutlass.Int32(NUM_BYTES_B1 >> 4))
                            sfa_c = tcol + SFA1_COL + stage * cutlass.Int32(16)
                            sfb_c = tcol + SFB1_COL + j * cutlass.Int32(NUM_KB * 4)
                            if prims.elect_sync():
                                if cutlass.const_expr(SFA_CP):
                                    _sfa_to_tmem(sSFA, stage, base_row, base_col)
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
                            stage, phase = _adv(stage, phase)
                        if prims.elect_sync():
                            prims.tcgen05_commit(acc_full.subview(ab))
                        k1 = k1 + cutlass.Int32(1)
                if p >= cutlass.Int32(LAG):
                    g = p - cutlass.Int32(LAG)
                    buf = g & cutlass.Int32(1)
                    tw2 = _gtime() if cutlass.const_expr(TIMING) else cutlass.Int64(0)
                    _waitd(
                        b2_ready.subview(buf), (g >> cutlass.Int32(1)) & cutlass.Int32(1), tdbg, 103
                    )
                    if cutlass.const_expr(TIMING):
                        acc_b2 = acc_b2 + (_gtime() - tw2)
                        if (lane == 0) & (g == ng - cutlass.Int32(1)):
                            _stamp(tdbg, bx, 4)
                    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                    for bb in range(nb):
                        a2 = k2 & cutlass.Int32(1)
                        _waitd(
                            acc2_empty.subview(a2),
                            ((k2 >> cutlass.Int32(1)) & cutlass.Int32(1)) ^ cutlass.Int32(1),
                            tdbg,
                            104,
                        )
                        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                        acc2_ptr = cutlass.inttoptr(
                            tcol + ACC2_COL + a2 * cutlass.Int32(N2), 6, cutlass.Float32
                        )
                        for t2 in cutlass.range_constexpr(KT2):
                            tw1 = _gtime() if cutlass.const_expr(TIMING) else cutlass.Int64(0)
                            if cutlass.const_expr(SFA_CP):
                                _waitd(ab_full.subview(stage), phase, tdbg, 7)
                                prims.fence_proxy(
                                    "async_shared", space=prims.SharedSpace.shared_cta
                                )
                            else:
                                _waitd(sfa_ready.subview(stage), phase, tdbg, 7)
                            if cutlass.const_expr(TIMING):
                                acc_sf = acc_sf + (_gtime() - tw1)
                            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                            da = desc_a0 + cutlass.Int64(stage * cutlass.Int32(NUM_BYTES_A >> 4))
                            db = desc_b20 + cutlass.Int64(
                                (buf * cutlass.Int32(KT2) + cutlass.Int32(t2))
                                * cutlass.Int32(NUM_BYTES_B2 >> 4)
                            )
                            sfa_c = tcol + SFA1_COL + stage * cutlass.Int32(16)
                            sfb_c = (
                                tcol
                                + SFB2_COL
                                + buf * cutlass.Int32(KT2 * NUM_KB * 4)
                                + cutlass.Int32(t2 * NUM_KB * 4)
                            )
                            if prims.elect_sync():
                                if cutlass.const_expr(SFA_CP):
                                    _sfa_to_tmem(sSFA, stage, base_row, base_col)
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
                            stage, phase = _adv(stage, phase)
                        if prims.elect_sync():
                            prims.tcgen05_commit(acc2_full.subview(a2))
                        k2 = k2 + cutlass.Int32(1)
            if cutlass.const_expr(TIMING):
                if lane == 0:
                    tdbg.store(acc_b2, idx=bx * cutlass.Int32(TD_STRIDE) + cutlass.Int32(26))
                    tdbg.store(acc_sf, idx=bx * cutlass.Int32(TD_STRIDE) + cutlass.Int32(27))

        # ---- epilogue warps (0-3)
        if warp < 4:
            row_w = base_row + warp * 32
            for r in cutlass.range_constexpr((6 * G_CAP + EPI_THREADS - 1) // EPI_THREADS):
                qi = tidx + cutlass.Int32(r * EPI_THREADS)
                sl_e = qi // cutlass.Int32(6)
                if (qi < cutlass.Int32(6 * G_CAP)) & (sl_e < g_n):
                    fld = qi - sl_e * cutlass.Int32(6)
                    s_esc.store(
                        cutlass.Float32(
                            escal.load(idx=fld * cutlass.Int32(E_LOCAL) + s_el.load(idx=sl_e))
                        ),
                        idx=sl_e * cutlass.Int32(8) + fld,
                    )
            cute.arch.sync_warp()
            cute.arch.barrier(barrier_id=EPI_BAR, number_of_threads=EPI_THREADS)
            k1 = cutlass.Int32(0)
            k2 = cutlass.Int32(0)
            SH_ADD = SHARED
            wys = [cutlass.Int32(0) for _ in range(NB2_CAP * M_MAX // 4 + 1)]
            pars = [cutlass.Int32(0) for _ in range(NB2_CAP * M_MAX // 4 + 1)]
            ard = (cutlass.Int64(0), cutlass.Int64(0), cutlass.Int64(0), cutlass.Int64(0))
            for p in range(ng + cutlass.Int32(LAG)):
                if cutlass.const_expr(AR_PRE):
                    if (p == ng + cutlass.Int32(LAG - 1)) & ar_on:
                        # one 8-byte load each (the vectorizer merges adjacent arp.load()s into 16-byte reads
                        # at 8-byte offsets: misaligned)
                        ard = tuple(
                            cutlass.Int64(_ld_relaxed_u64(arp.subview(i_).data_ptr().toint()))
                            for i_ in (1, 3, 4, 5)
                        )
                        for bt in cutlass.range_constexpr((NB2_CAP * M_MAX + 3) // 4):
                            f_pair = warp + cutlass.Int32(bt * 4)
                            f_bb = f_pair // cutlass.Int32(M_MAX)
                            f_t = f_pair - f_bb * cutlass.Int32(M_MAX)
                            f_cl = cutlass.select_(
                                f_bb < nb, (blk0 + f_bb) * cutlass.Int32(R2), cutlass.Int32(0)
                            )
                            pars[bt] = _ar_parity(arp, f_t, f_cl)
                if cutlass.const_expr(SH_ADD):
                    if p == ng + cutlass.Int32(LAG - 1):
                        # this CTA's y_sh words, loaded while the last group's FC2 runs (re-polled in finalize)
                        for bt in cutlass.range_constexpr((NB2_CAP * M_MAX + 3) // 4):
                            f_pair = warp + cutlass.Int32(bt * 4)
                            f_bb = f_pair // cutlass.Int32(M_MAX)
                            f_t = f_pair - f_bb * cutlass.Int32(M_MAX)
                            f_ch = cutlass.select_(
                                f_bb < nb,
                                f_t * cutlass.Int32(H)
                                + (blk0 + f_bb) * cutlass.Int32(R2)
                                + cutlass.Int32(4) * (lane & cutlass.Int32(7))
                                + (lane >> cutlass.Int32(3)),
                                cutlass.Int32(0),
                            )
                            wys[bt] = _ld_relaxed_u32(dnx.subview(f_ch).data_ptr().toint())
                if p < ng:
                    i0, i1 = _plan_range(s_plan, p)
                    for k in range(_nsegs(i0, i1)):
                        ug, j0, j1 = _seg(i0, i1, k)
                        ab = k1 & cutlass.Int32(1)
                        slot = p * cutlass.Int32(4) + ug // cutlass.Int32(U1)
                        cblk = ug - (ug // cutlass.Int32(U1)) * cutlass.Int32(U1)
                        unit = slot * cutlass.Int32(U1) + cblk
                        _waitd(
                            acc_full.subview(ab),
                            (k1 >> cutlass.Int32(1)) & cutlass.Int32(1),
                            tdbg,
                            105,
                        )
                        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                        acc = prims.tcgen05_ld(
                            "32x32b",
                            _tptr(
                                row_w, base_col + ACC1_COL + ab * cutlass.Int32(N1), cutlass.Float32
                            ),
                            num=N1,
                        )
                        prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                        prims.mbarrier_arrive(acc_empty.subview(ab))
                        pb = ab * cutlass.Int32(M_MAX * C1 * R1)
                        for t in cutlass.range_constexpr(M_MAX):
                            part = cutlass.Float32(0.0)
                            for w in cutlass.range_constexpr(4):
                                part = cutlass.select_(
                                    warp == cutlass.Int32(w), cutlass.Float32(acc[QS * w + t]), part
                                )
                            s_part.store(part, idx=pb + (t * C1 + warp) * R1 + lane)
                        cute.arch.sync_warp()
                        cute.arch.barrier(barrier_id=EPI_BAR, number_of_threads=EPI_THREADS)
                        for tr in cutlass.range_constexpr((M_MAX + 3) // 4):
                            t = warp + cutlass.Int32(4 * tr)
                            if t < cutlass.Int32(M_MAX):
                                xv = _add_rn(
                                    _add_rn(
                                        _add_rn(
                                            s_part.load(idx=pb + (t * C1 + 0) * R1 + lane),
                                            s_part.load(idx=pb + (t * C1 + 1) * R1 + lane),
                                        ),
                                        s_part.load(idx=pb + (t * C1 + 2) * R1 + lane),
                                    ),
                                    s_part.load(idx=pb + (t * C1 + 3) * R1 + lane),
                                )
                                finish = cutlass.Int32(1)
                                if (j0 != cutlass.Int32(0)) | (j1 != cutlass.Int32(S1)):
                                    paddr = (
                                        partials.subview(
                                            (unit * cutlass.Int32(M_MAX) + t) * cutlass.Int32(R1)
                                            + lane
                                        )
                                        .data_ptr()
                                        .toint()
                                    )
                                    if j0 == cutlass.Int32(0):
                                        _st_relaxed_u32(paddr, xv.bitcast(cutlass.Int32))
                                        finish = cutlass.Int32(0)
                                    else:
                                        pbits = _ld_relaxed_u32(paddr)
                                        spins = cutlass.Int32(0)
                                        while pbits == cutlass.Int32(PARTIAL_EMPTY):
                                            pbits = _ld_relaxed_u32(paddr)
                                            spins = spins + cutlass.Int32(1)
                                        _st_relaxed_u32(paddr, cutlass.Int32(PARTIAL_EMPTY))
                                        xv = _add_rn(pbits.bitcast(cutlass.Float32), xv)
                                        cute.arch.sync_warp()
                                if cutlass.const_expr(
                                    XVDUMP
                                ):  # debug: FC1 sums into the (unused) partial slots
                                    partials.store(
                                        xv.bitcast(cutlass.Int32),
                                        idx=(unit * cutlass.Int32(M_MAX) + t) * cutlass.Int32(R1)
                                        + lane,
                                    )
                                if finish == cutlass.Int32(1):
                                    gate_raw = cute.arch.shuffle_sync_bfly(xv, 8)
                                    is_up = (
                                        (lane >> cutlass.Int32(3)) & cutlass.Int32(1)
                                    ) == cutlass.Int32(0)
                                    es = slot * cutlass.Int32(8)
                                    qv, sfb = _swiglu_quant(
                                        xv,
                                        gate_raw,
                                        is_up,
                                        s_esc.load(idx=es),
                                        s_esc.load(idx=es + cutlass.Int32(1)),
                                        s_esc.load(idx=es + cutlass.Int32(2)),
                                        s_esc.load(idx=es + cutlass.Int32(3)),
                                        s_esc.load(idx=es + cutlass.Int32(4)),
                                    )
                                    byte = _e2m1x2(cute.arch.shuffle_sync_down(qv, 16), qv)
                                    bt = [
                                        cute.arch.shuffle_sync(byte, i) & cutlass.Int32(0xFF)
                                        for i in range(8)
                                    ]
                                    rw0 = (
                                        bt[0]
                                        | (bt[1] << cutlass.Int32(8))
                                        | (bt[2] << cutlass.Int32(16))
                                        | tagv
                                    )
                                    rw1 = (
                                        bt[3]
                                        | (bt[4] << cutlass.Int32(8))
                                        | (bt[5] << cutlass.Int32(16))
                                        | tagv
                                    )
                                    rw2 = (
                                        bt[6]
                                        | (bt[7] << cutlass.Int32(8))
                                        | ((sfb & cutlass.Int32(0xFF)) << cutlass.Int32(16))
                                        | tagv
                                    )
                                    if cutlass.const_expr(B2IMG):
                                        if lane == 0:
                                            # B2 row n = j M + t, k-tile c / 16, 16-byte chunk (c / 2) % 8
                                            # (swizzled), half c % 2; scale: atom (t2, kb = (c % 16) / 4),
                                            # row n at byte 16 n, byte c % 4
                                            n = (ug // cutlass.Int32(U1)) * cutlass.Int32(M_MAX) + t
                                            t2i = cblk >> cutlass.Int32(4)
                                            cci = (cblk >> cutlass.Int32(1)) & cutlass.Int32(7)
                                            gbase = p * cutlass.Int32(IMG_BYTES)
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
                                            img32.store(
                                                (lo, hi),
                                                idx=(
                                                    gbase
                                                    + t2i * cutlass.Int32(NUM_BYTES_B2)
                                                    + n * cutlass.Int32(KT_BYTES)
                                                    + (cci ^ (n & cutlass.Int32(7)))
                                                    * cutlass.Int32(16)
                                                    + (cblk & cutlass.Int32(1)) * cutlass.Int32(8)
                                                )
                                                // cutlass.Int32(4),
                                                alignment=8,
                                            )
                                            img8 = cutlass.Array(
                                                img32.data_ptr(0),
                                                shape=(NG_CAP * IMG_BYTES,),
                                                dtype=cutlass.Int8,
                                                alignment=16,
                                            )
                                            img8.store(
                                                cutlass.Int8(sfb & cutlass.Int32(0xFF)),
                                                idx=gbase
                                                + cutlass.Int32(IMG_B2)
                                                + (
                                                    (
                                                        t2i * cutlass.Int32(NUM_KB)
                                                        + (
                                                            (cblk & cutlass.Int32(15))
                                                            >> cutlass.Int32(2)
                                                        )
                                                    )
                                                    * cutlass.Int32(512)
                                                )
                                                + n * cutlass.Int32(16)
                                                + (cblk & cutlass.Int32(3)),
                                            )
                                    else:
                                        if lane == 0:
                                            hbuf32.store(
                                                (rw0, rw1, rw2, tagv),
                                                idx=(
                                                    (slot * cutlass.Int32(M_MAX) + t)
                                                    * cutlass.Int32(UNITS_ROW)
                                                    + cblk
                                                )
                                                * cutlass.Int32(4),
                                                alignment=16,
                                            )
                        k1 = k1 + cutlass.Int32(1)
                    prims.mbarrier_arrive(
                        fc1_done.subview(p & cutlass.Int32(3))
                    )  # warp 5 may poll group p
                    if tidx == 0:
                        _stamp(tdbg, bx, 10)
                        if cutlass.const_expr(TIMING):
                            if p < cutlass.Int32(8):
                                tdbg.store(
                                    _gtime(),
                                    idx=bx * cutlass.Int32(TD_STRIDE) + cutlass.Int32(40) + p,
                                )
                if p >= cutlass.Int32(LAG):
                    g = p - cutlass.Int32(LAG)
                    for bb in range(nb):
                        a2 = k2 & cutlass.Int32(1)
                        _waitd(
                            acc2_full.subview(a2),
                            (k2 >> cutlass.Int32(1)) & cutlass.Int32(1),
                            tdbg,
                            106,
                        )
                        if cutlass.const_expr(TIMING):
                            if (tidx == 0) & (g == ng - cutlass.Int32(1)):
                                tdbg.store(
                                    _gtime(),
                                    idx=bx * cutlass.Int32(TD_STRIDE) + cutlass.Int32(8) + bb,
                                )
                        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                        yv = prims.tcgen05_ld(
                            "32x32b",
                            _tptr(
                                row_w,
                                base_col
                                + ACC2_COL
                                + a2 * cutlass.Int32(N2)
                                + warp * cutlass.Int32(M_MAX),
                                cutlass.Float32,
                            ),
                            num=LD_POW2,
                        )
                        prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                        prims.mbarrier_arrive(acc2_empty.subview(a2))
                        sl = g * cutlass.Int32(4) + warp
                        if sl < g_n:
                            a2s = s_esc.load(idx=sl * cutlass.Int32(8) + cutlass.Int32(5))
                            for t in cutlass.range_constexpr(M_MAX):
                                kp = s_kpos.load(idx=sl * cutlass.Int32(M_MAX) + cutlass.Int32(t))
                                if kp >= cutlass.Int32(0):
                                    ys.store(
                                        _bf16_round(_mul_rn(cutlass.Float32(yv[t]), a2s)),
                                        idx=(
                                            (bb * cutlass.Int32(M_MAX) + cutlass.Int32(t))
                                            * cutlass.Int32(TOP_K)
                                            + kp
                                        )
                                        * cutlass.Int32(R2)
                                        + lane,
                                    )
                        k2 = k2 + cutlass.Int32(1)
                    prims.mbarrier_arrive(b2_free.subview(g & cutlass.Int32(1)))
            if tidx == 0:
                _stamp(tdbg, bx, 6)
            cute.arch.sync_warp()
            cute.arch.barrier(barrier_id=EPI_BAR, number_of_threads=EPI_THREADS)
            if tidx == 0:
                _stamp(tdbg, bx, 19)
            # finalize: (block bb, token t) per warp round: the fma chain over t's top-4 in order
            for bt in cutlass.range_constexpr((NB2_CAP * M_MAX + 3) // 4):
                pair = warp + cutlass.Int32(bt * 4)
                bb = pair // cutlass.Int32(M_MAX)
                t = pair - bb * cutlass.Int32(M_MAX)
                if bb < nb:
                    acc_o = cutlass.Float32(0.0)
                    for k in cutlass.range_constexpr(TOP_K):
                        acc_o = _fma_rn(
                            s_w.load(idx=t * cutlass.Int32(TOP_K) + cutlass.Int32(k)),
                            ys.load(
                                idx=(
                                    (bb * cutlass.Int32(M_MAX) + t) * cutlass.Int32(TOP_K)
                                    + cutlass.Int32(k)
                                )
                                * cutlass.Int32(R2)
                                + lane
                            ),
                            acc_o,
                        )
                        if cutlass.const_expr(
                            YDUMP
                        ):  # debug: y of (token, k) into the partials workspace
                            partials.store(
                                cutlass.Float32(
                                    ys.load(
                                        idx=(
                                            (bb * cutlass.Int32(M_MAX) + t) * cutlass.Int32(TOP_K)
                                            + cutlass.Int32(k)
                                        )
                                        * cutlass.Int32(R2)
                                        + lane
                                    )
                                ).bitcast(cutlass.Int32),
                                idx=(t * cutlass.Int32(TOP_K) + cutlass.Int32(k)) * cutlass.Int32(H)
                                + (blk0 + bb) * cutlass.Int32(R2)
                                + cutlass.Int32(4) * (lane & cutlass.Int32(7))
                                + (lane >> cutlass.Int32(3)),
                            )
                    ch = (
                        t * cutlass.Int32(H)
                        + (blk0 + bb) * cutlass.Int32(R2)
                        + cutlass.Int32(4) * (lane & cutlass.Int32(7))
                        + (lane >> cutlass.Int32(3))
                    )
                    if cutlass.const_expr(SH_ADD):
                        # + y_sh (this row's shared output, from its shared CTA), as shared_output.add_(routed_output)
                        wy = wys[bt]
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
                        if cutlass.const_expr(AR_PRE):
                            _ar_push_pre(
                                ard,
                                pars[bt],
                                _bf16_round(acc_o),
                                ((lane >> cutlass.Int32(3)) & cutlass.Int32(1)) == cutlass.Int32(0),
                                t,
                                cl,
                            )
                        else:
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
            # the all-reduce polls the pushed slots without a grid wait: release it after every push of this CTA
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
    logits: cute.Tensor,
    bias: cute.Tensor,
    x32: cute.Tensor,
    xsf32: cute.Tensor,
    escal: cute.Tensor,
    plan: cute.Tensor,
    partials: cute.Tensor,
    hbuf32: cute.Tensor,
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
    s1w: cute.Tensor,
    s2w: cute.Tensor,
    img32: cute.Tensor,
    b2cnt: cute.Tensor,
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
    if cutlass.const_expr(SHARED and not SH_PRETILE):
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
        logits,
        bias,
        x32,
        xsf32,
        escal,
        plan,
        partials,
        hbuf32,
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
        s1w,
        s2w,
        img32,
        b2cnt,
    ).launch(
        grid=(NCTA + NSH, 1, 1),
        block=(THREADS, 1, 1),
        cluster=(1, 1, 1),
        stream=stream,
        use_pdl=True,
    )
