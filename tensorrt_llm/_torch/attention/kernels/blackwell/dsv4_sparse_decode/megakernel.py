# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused DeepSeek-V4-Flash decode kernel (CuTe DSL, SM100): the cooperative launch, the indexer-K stream, the sampled
threshold, candidate publication and the teams' exact top-512 selection, plus the host entry points
``Dsv4SparseDecodeKernel`` and ``compile_mk`` used by ``op.py``.
"""

from __future__ import annotations

import dataclasses
import math

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cudax
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import Float32, Int32, Int64, Uint32, Uint64
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.experimental import primitives as prims
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from . import attention_roles as A
from . import cache_policy as CP
from . import ptx_helpers as B
from . import row_desc as D
from . import topk_select as S
from . import topk_select_lib as L

_RND_RN = "rn"
NEG_INF = -math.inf
K_TOP = 512
SPLIT_TOK = 256
TILE = 128
MAXS = D.MAXS
N_MAX = D.N_MAX
NTEAM = 8
CAPR = 2048
CAPL = 5632
BOARD_CAP = 4096
RMAX = 10
NSLOTS = 11
NTILES = 5
NB_SEL = S.NB_SEL
NB_INITIAL = 512
BUCKET_CAP = 512
assert NB_INITIAL * 8 == NB_SEL * 4 and CAPL < 65536
NB_DRS = 1024
DRS_DIG = 10
K50_BITS = 50
GPOS_MASK = (1 << 18) - 1
PLAN = D.PLAN
NGH = 8
NCSLOT = 3
SPIN_MS = 200
P_T1 = 1
P_T12 = 2
P_DRS = 3
P_ABORT = 9
K_DONE = 1
K_PUB = 2
K_HIST = 3
ST_GO = 0
ST_SKIP = 9
E_STALE = 1
E_TIMEOUT = 2
E_NLIMIT = 4
E_MISMATCH = 8
E_ABORTED = 16
W_ENTRY = 0
W_DONE = 1
W_DONE2 = 2
W_TSRV = 5
W_TLINE = 8
W_DIR = 16
W_DP = 17
W_INFO = 24
W_INFO2 = 25
W_INFO3 = 26
W_ERR = 32
W_ERRB = 33
W_ABORT = 34
W_FOL = 36
W_RC = 48
assert W_FOL + NTEAM <= W_RC
W_CNT = 64
WS_WORDS = 1024
NSTAMP = 128
(
    ST_ENTRY,
    ST_FIRST_TMA,
    ST_FIRST_EPI,
    ST_BOARD,
    ST_STREAM_END,
    ST_T_SEEN,
    ST_CAND_PUB,
    ST_VERDICT,
    ST_EXIT,
) = range(9)
(
    ST_M_SWA,
    ST_M_BOARD,
    ST_M_T,
    ST_M_FIN,
    ST_M_COPY,
    ST_M_SEL,
    ST_M_OUT,
    ST_M_EXIT,
    ST_M_WARMA,
    ST_M_WARMB,
) = range(10, 20)
ST_M_PV0 = 20
ST_A_LOAD, ST_B_CNT, ST_B_COPY2, ST_B_SELX, ST_S_SCAN, ST_S_WR = (52, 53, 54, 55, 56, 57)
S_OFF_KV0 = 0
S_OFF_KV1 = 49152
S_OFF_Q = 98304
S_OFF_W = 106496
S_OFF_SC0 = 106752
S_OFF_SC1 = 108288
S_OFF_LOG = 109824
S_OFF_BTC = S_OFF_LOG + MAXS * SPLIT_TOK * 4
S_OFF_BAR = S_OFF_BTC + MAXS * 8 * 4
S_OFF_MISC = S_OFF_BAR + 256
SUBCAP = 512
S_OFF_SUB = S_OFF_MISC + 512
S_BYTES = S_OFF_SUB + 8 * SUBCAP * 8
S_BAR_T = 25
# per-row logits buffers (up to 8 rows) and per-row directive state (8 x int32 [tot, ntok, t_lo, seq, done],
# then 8 x int64 gen) past the stream layout
S_OFF_LOGR = (S_BYTES + 1023) // 1024 * 1024
S_OFF_ROWST = S_OFF_LOGR + 8 * MAXS * SPLIT_TOK * 4
K_NONE = 15
# service-warpgroup scratch: K_HIST histogram, per-row C-pool block-table slice, per-row [T, bin scale],
# per-row handoff mbarriers, per-row entry words, service s_wc / s_dir
S_OFF_SVH = S_OFF_ROWST + 512
S_OFF_SVB = S_OFF_SVH + NB_DRS * 4
S_OFF_SVT = S_OFF_SVB + 8 * MAXS * 8 * 4
S_OFF_SVM = S_OFF_SVT + 64
S_OFF_SVG = S_OFF_SVM + 64
S_OFF_SVW = S_OFF_SVG + 8 * 3 * 8 + 8 * 8
S_OFF_SVTB = S_OFF_SVW + 64
S_OFF_LSTM = S_OFF_LOGR + 6 * MAXS * SPLIT_TOK * 4
S_SVC_END = S_OFF_SVTB + 64
# stream T-server scratch in the logit slots of rows 4-5 (unused for nrows <= 4)
S_TSRV_FA = S_OFF_LOGR + 4 * MAXS * SPLIT_TOK * 4
# nrows > 4: T-server scratch in row 7's logit slot; nrows > 6: the last row's list in the idle K ring 1
S_TSRV_FA8 = S_OFF_LOGR + 7 * MAXS * SPLIT_TOK * 4
S_OFF_LSTM8 = S_OFF_KV1
T_OFF_Q = 0
T_OFF_KV = 32768
T_OFF_LIST = T_OFF_KV + 8 * 16384
T_OFF_SCR = T_OFF_LIST + CAPL * 8
T_OFF_HIST = T_OFF_SCR
T_OFF_OUTS = T_OFF_HIST + 4096
T_OFF_CK = T_OFF_OUTS + 2048
T_OFF_CS = T_OFF_CK + 2304
T_OFF_COFF = T_OFF_CS + 1152
T_OFF_CU1 = T_OFF_COFF + 640
T_OFF_CU2 = T_OFF_CU1 + 640
T_OFF_RES = T_OFF_CU2 + 640
T_OFF_RED = T_OFF_RES + 64
T_OFF_WC = T_OFF_RED + 256
T_OFF_STAT = T_OFF_WC + 256
T_OFF_FIN = T_OFF_STAT + 1024
T_OFF_IDX = T_OFF_FIN + 1024
T_OFF_BAR = T_OFF_IDX + 2560
T_OFF_MISC = T_OFF_BAR + 512
T_OFF_FA = T_OFF_MISC + 256
T_OFF_CROW = T_OFF_FA + 1024 + 64 + 256 + 512
# entry words of up to MAXP passes (8 x int32 each) outside misc
MAXP = 8
T_OFF_PW = T_OFF_CROW + 288 * 4
T_BYTES = T_OFF_PW + MAXP * 8 * 4
# bucket s_map and the pre-select index buffer inside the list area, beyond the partner's merge push
SMAP2_OFF = T_OFF_LIST + 20480
SIDX2_OFF = T_OFF_LIST + 22528
assert SIDX2_OFF + 640 * 4 <= T_OFF_SCR
assert T_OFF_LIST + (NSLOTS - 8) * 16384 <= T_OFF_STAT
assert T_BYTES <= 232448, T_BYTES
T_OFF_F_RAW = T_OFF_KV + 4 * 16384
T_OFF_F_HIST = T_OFF_FA
T_OFF_F_RES = T_OFF_F_HIST + 1024
T_OFF_F_RED = T_OFF_F_RES + 64
T_OFF_F_MB = T_OFF_F_RED + 256
T_OFF_BTC = T_OFF_KV + 6 * 16384
T_OFF_NKEY = T_OFF_KV
assert CAPL * 8 <= 3 * 16384
BTC_CAP = 8192
assert T_OFF_F_RAW + BOARD_CAP * 8 <= T_OFF_BTC and T_OFF_BTC + BTC_CAP * 4 <= T_OFF_KV + 8 * 16384
ARENA = max(T_BYTES, S_BYTES)
assert S_OFF_ROWST + 8 * 8 * 4 + 8 * 8 <= ARENA
assert S_SVC_END <= ARENA, (S_SVC_END, ARENA)
B_QF = 0
B_KVF = 1
B_KVE = B_KVF + NSLOTS
B_SF = B_KVE + NSLOTS
B_SE = B_SF + 4
B_PF = B_SE + 4
B_OF = B_PF + 4
B_OC = B_OF + 1
B_FIN = B_OC + 1
B_SWA = B_FIN + 1
B_SEL = B_SWA + 1
B_T = B_SEL + 1
B_PVL = B_T + 1
B_BTC = B_PVL + 1
B_LST = B_BTC + 1
B_EAR = B_LST + 1
B_RX = B_EAR + 1
B_PR = B_EAR + 2
B_DRY = B_EAR + 3
assert B_EAR < 64
DUMMY_GPOS = 2147483647
BAR_CTL = 6
BAR_EPI = 7
BAR_SMX = 8
# pass transition of the 12 non-control warps / softmax -> control WG handshake
BAR_PASS = 13
BAR_HAND = 14


@dsl_user_op
def prefetch_l2_bulk(addr, nbytes, *, loc=None, ip=None):
    L._asm(
        None,
        [L._v(addr, loc, ip), L._v(nbytes, loc, ip)],
        "cp.async.bulk.prefetch.L2.global [$0], $1;",
        "l,r",
        loc,
        ip,
    )


@dsl_user_op
def st_rlx_u64(addr, val, *, loc=None, ip=None):
    L._asm(
        None,
        [L._v(addr, loc, ip), L._v(val, loc, ip)],
        "st.relaxed.gpu.global.u64 [$0], $1;",
        "l,l",
        loc,
        ip,
    )


@dsl_user_op
def ld_relaxed_v2u64(addr, *, loc=None, ip=None):
    from cutlass._mlir.dialects import llvm as _llvm
    from cutlass.cutlass_dsl import T as _T

    st = _llvm.StructType.get_literal([_T.i64(), _T.i64()])
    r = L._asm(
        st,
        [L._v(addr, loc, ip)],
        "ld.relaxed.gpu.global.v2.u64 {$0, $1}, [$2];",
        "=l,=l,l",
        loc,
        ip,
    )
    return (
        Uint64(_llvm.extractvalue(_T.i64(), r, [0], loc=loc, ip=ip)),
        Uint64(_llvm.extractvalue(_T.i64(), r, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def red_add_u32(addr, val, *, loc=None, ip=None):
    L._asm(
        None,
        [L._v(addr, loc, ip), L._v(val, loc, ip)],
        "red.relaxed.gpu.global.add.u32 [$0], $1;",
        "l,r",
        loc,
        ip,
    )


@dsl_user_op
def atom_add_u32(addr, val, *, loc=None, ip=None):
    from cutlass.cutlass_dsl import T as _T

    return Uint32(
        L._asm(
            _T.i32(),
            [L._v(addr, loc, ip), L._v(val, loc, ip)],
            "atom.relaxed.gpu.global.add.u32 $0, [$1], $2;",
            "=r,l,r",
            loc,
            ip,
        )
    )


@dsl_user_op
def atom_exch_u64(addr, val, *, loc=None, ip=None):
    # atomic swap of a 64-bit global word (relaxed, gpu scope); returns the old value
    from cutlass.cutlass_dsl import T as _T

    return Uint64(
        L._asm(
            _T.i64(),
            [L._v(addr, loc, ip), L._v(val, loc, ip)],
            "atom.relaxed.gpu.global.exch.b64 $0, [$1], $2;",
            "=l,l,l",
            loc,
            ip,
        )
    )


@dsl_user_op
def ldg_cg_u64(addr, *, loc=None, ip=None):
    # weak 64-bit global load cached in L2 only (loads of a thread overlap, unlike ld.relaxed.gpu)
    from cutlass.cutlass_dsl import T as _T

    return Uint64(
        L._asm(_T.i64(), [L._v(addr, loc, ip)], "ld.global.cg.u64 $0, [$1];", "=l,l", loc, ip)
    )


@dsl_user_op
def prefetch_l1(addr, *, loc=None, ip=None):
    # bring one global line into this SM's L1
    L._asm(None, [L._v(addr, loc, ip)], "prefetch.global.L1 [$0];", "l", loc, ip)


@dsl_user_op
def ldg_ca_u64(addr, *, loc=None, ip=None):
    # weak 64-bit global load, L1-allocating
    from cutlass.cutlass_dsl import T as _T

    return Uint64(
        L._asm(_T.i64(), [L._v(addr, loc, ip)], "ld.global.ca.u64 $0, [$1];", "=l,l", loc, ip)
    )


@dsl_user_op
def red_max_u64(addr, val, *, loc=None, ip=None):
    L._asm(
        None,
        [L._v(addr, loc, ip), L._v(val, loc, ip)],
        "red.relaxed.gpu.global.max.u64 [$0], $1;",
        "l,l",
        loc,
        ip,
    )


@dsl_user_op
def red_or_u64(addr, val, *, loc=None, ip=None):
    L._asm(
        None,
        [L._v(addr, loc, ip), L._v(val, loc, ip)],
        "red.relaxed.gpu.global.or.b64 [$0], $1;",
        "l,l",
        loc,
        ip,
    )


@dsl_user_op
def st_shared_v2u64(addr32, a, b, *, loc=None, ip=None):
    L._asm(
        None,
        [L._v(addr32, loc, ip), L._v(a, loc, ip), L._v(b, loc, ip)],
        "st.shared.v2.u64 [$0], {$1, $2};",
        "r,l,l",
        loc,
        ip,
    )


@dsl_user_op
def cp_async8(saddr, gaddr, *, loc=None, ip=None):
    L._asm(
        None,
        [L._v(saddr, loc, ip), L._v(gaddr, loc, ip)],
        "cp.async.ca.shared.global [$0], [$1], 8;",
        "r,l",
        loc,
        ip,
    )


@dsl_user_op
def lds_volatile_u32(saddr, *, loc=None, ip=None):
    return Int32(
        B._asm(
            B.T.i32(), [B._v(saddr, loc, ip)], "ld.volatile.shared.u32 $0, [$1];", "=r,r", loc, ip
        )
    )


@dsl_user_op
def fence_acq_rel_cta(*, loc=None, ip=None):
    B._asm(None, [], "fence.acq_rel.cta;", "", loc, ip)


@dsl_user_op
def sts_volatile_u32(saddr, val, *, loc=None, ip=None):
    B._asm(
        None,
        [B._v(saddr, loc, ip), B._v(val, loc, ip)],
        "st.volatile.shared.u32 [$0], $1;",
        "r,r",
        loc,
        ip,
    )


@dsl_user_op
def match_any_b32(v, *, loc=None, ip=None):
    from cutlass.cutlass_dsl import T as _T

    return Int32(
        L._asm(
            _T.i32(), [L._v(v, loc, ip)], "match.any.sync.b32 $0, $1, 0xffffffff;", "=r,r", loc, ip
        )
    )


@dsl_user_op
def clz_u64(x, *, loc=None, ip=None):
    from cutlass.cutlass_dsl import T as _T

    return Int32(L._asm(_T.i32(), [L._v(x, loc, ip)], "clz.b64 $0, $1;", "=r,l", loc, ip))


@dsl_user_op
def lds_v4f32(saddr32, *, loc=None, ip=None):
    # ld.shared.v4.f32 (16-byte aligned shared address)
    from cutlass._mlir.dialects import llvm as _llvm
    from cutlass.cutlass_dsl import T as _T

    st = _llvm.StructType.get_literal([_T.f32(), _T.f32(), _T.f32(), _T.f32()])
    r = L._asm(
        st,
        [L._v(saddr32, loc, ip)],
        "ld.shared.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,r",
        loc,
        ip,
    )
    return tuple(Float32(_llvm.extractvalue(_T.f32(), r, [i], loc=loc, ip=ip)) for i in range(4))


def _row2(t, r):
    # (1, C) view of row r of a 2-D tensor (row stride = stride[0])
    s0_ = t.layout.stride[0]
    return cute.make_tensor(
        t.iterator + r * s0_, cute.make_layout((1, t.shape[1]), stride=(s0_, t.layout.stride[1]))
    )


def _row1(t, r):
    return cute.make_tensor(t.iterator + r, cute.make_layout((1,)))


def _flat(t, r, size: int):
    return cute.make_tensor(t.iterator + r * size, cute.make_layout((size,)))


def _ptr(arena, off, dtype):
    return cutlass.Pointer(cute.recast_ptr(arena + off, dtype=dtype), dtype=dtype)


def _sm_tensor(arena, off, dtype, n, stride=1):
    return cute.make_tensor(
        cute.recast_ptr(arena + off, dtype=dtype), cute.make_layout(n, stride=stride)
    )


@cute.jit
def packed_hist_total(lo, hi):
    return Int32(
        (lo & Uint32(65535)) + (lo >> Uint32(16)) + (hi & Uint32(65535)) + (hi >> Uint32(16))
    )


@cute.jit
def find_cross_warp_counts(s_hist, target, tidx, s_res, s_wc, nb: cutlass.Constexpr):
    # Each 16-bit field counts one warp's candidates. No field can carry:
    # every prefix contains at most CAPL < 65536 entries. Native u32 shared
    # reductions build the low/high pairs; packed scans preserve all 4 counts.
    BPL = nb // 32
    if tidx < Int32(32):
        lane = tidx
        part0 = Uint32(0)
        part1 = Uint32(0)
        for j in cutlass.range_constexpr(BPL):
            idx = lane * Int32(BPL) + ((Int32(j) + lane) & Int32(BPL - 1))
            h = Uint64(s_hist[idx])
            part0 = part0 + Uint32(h)
            part1 = part1 + Uint32(h >> Uint64(32))
        v0 = L.warp_suffix_scan_add(part0, lane)
        v1 = L.warp_suffix_scan_add(part1, lane)
        total = packed_hist_total(v0, v1)
        if lane == Int32(0):
            s_res[L.RES_TOT] = total
        msk = Int32(cute.arch.vote_ballot_sync(total >= target))
        Lb = L.hi_bit_or_zero(msk)
        above0 = cute.arch.shuffle_sync(v0 - part0, Lb)
        above1 = cute.arch.shuffle_sync(v1 - part1, Lb)
        h = Uint64(0)
        if lane < Int32(BPL):
            h = Uint64(s_hist[Lb * Int32(BPL) + lane])
        h0 = Uint32(h)
        h1 = Uint32(h >> Uint64(32))
        w0 = L.warp_suffix_scan_add(h0, lane)
        w1 = L.warp_suffix_scan_add(h1, lane)
        msk2 = Int32(
            cute.arch.vote_ballot_sync(packed_hist_total(above0 + w0, above1 + w1) >= target)
        )
        J = L.hi_bit_or_zero(msk2)
        if lane == J:
            winner0 = above0 + w0 - h0
            winner1 = above1 + w1 - h1
            above = packed_hist_total(winner0, winner1)
            m = packed_hist_total(h0, h1)
            s_res[L.RES_B] = Lb * Int32(BPL) + J
            s_res[L.RES_M] = m
            s_res[L.RES_ABOVE] = above
            if target - above >= m:
                winner0 = winner0 + h0
                winner1 = winner1 + h1
                h0 = Uint32(0)
                h1 = Uint32(0)
            # Emit prefix offsets directly. The whole candidate counting pass
            # and its cross-warp barriers are unnecessary after this scan.
            pw1 = Int32(winner0 & Uint32(65535))
            pw2 = pw1 + Int32(winner0 >> Uint32(16))
            px1 = Int32(h0 & Uint32(65535))
            px2 = px1 + Int32(h0 >> Uint32(16))
            s_wc[0] = Int32(0)
            s_wc[1] = pw1
            s_wc[2] = pw2
            s_wc[3] = pw2 + Int32(winner1 & Uint32(65535))
            s_wc[4] = Int32(0)
            s_wc[5] = px1
            s_wc[6] = px2
            s_wc[7] = px2 + Int32(h1 & Uint32(65535))


class Dsv4SparseDecodeKernel:
    """Kernel configuration and device code: the constructor arguments are the compile-time knobs (``op.py`` holds
    the per-batch-size sets); ``row_sizes()`` gives the per-row workspace sizes.
    """

    def __init__(
        self,
        num_sms: int = 148,
        policy: int = CP.POL_EVICT_FIRST,
        debug: bool = False,
        stamps: bool = False,
        math_regs: int = 232,
        kv_stages: int = 2,
        phys_block_kv: int = 32,
        prewarm: bool = True,
        skew: int = 0,
        coop: bool = True,
        qk_ahead: bool = False,
        gjoin: bool = True,
        wdelay: int = 400,
        mma_block: bool = True,
        mma_idle_ns: int = 0,
        ctl_regs: bool = True,
        spin_ms: int = SPIN_MS,
        acqpoll: bool = False,
        relst: bool = False,
        wacq: int = 0,
        folpre: bool = True,
        heal: bool = True,
        nofence: bool = False,
        lean: bool = False,
        bulk: bool = True,
        fblka: bool = False,
        gen2: bool = True,
        hook: bool = True,
        late_done: int = 0,
        pref: bool = True,
        pchk: int = 16,
        full_attention_tiles: bool = False,
        swag: bool = False,
        nohoist: bool = False,
        rs24: bool = False,
        rs24_t4split: bool = False,
        rs24_dry: bool = True,
        rs24_f32: bool = False,
        rs24_exitbar: bool = False,
        rs24_asym: bool = False,
        rs24_lsu: int = 0,
        nrows: int = 1,
        nteams: int = 0,
        ppref: bool = True,
        presel: bool = False,
        pdec: bool = False,
        warm1: bool = False,
        pollns: int = 0,
        acqx: bool = False,
        swpre: bool = True,
        bpf: bool = False,
        entcg: bool = False,
        tsz01: bool = False,
        tsz2: bool = False,
        tsz: bool = False,
        drot: int = 0,
        awarm3: bool = False,
        awarm2a: bool = False,
        btcl: bool = False,
        pdwait: int = 0,
        awarm2: bool = False,
        t1a: bool = False,
        awarm: bool = False,
        pdirect: bool = False,
        tsrv1: bool = False,
        tl1r: bool = False,
        tszall: bool = False,
        qdb: bool = False,
        dpf: bool = False,
    ):
        assert coop, (
            "the protocol spins across CTAs: all 148 CTAs must be co-resident (cooperative launch)"
        )
        assert bulk, "warp-compacted acquire requires even-padded stream packets"
        assert kv_stages in (2, 3), kv_stages
        assert wdelay >= 0 and mma_idle_ns >= 0 and (skew >= 0)
        assert not (qk_ahead and gjoin), "qk_ahead belongs to the gjoin=False MMA role"
        assert 1 <= spin_ms <= 10000, spin_ms
        assert ladder_max_smp() * 8 <= min(BOARD_CAP, RMAX * 128 * 2), "board capacity"
        self.full_attention_tiles = bool(full_attention_tiles)
        self.swag = bool(swag)
        self.nohoist = bool(nohoist)
        self.rs24 = bool(rs24)
        self.rs24_t4split = bool(rs24_t4split)
        self.rs24_dry = bool(rs24_dry)
        self.rs24_f32 = bool(rs24_f32)
        self.rs24_exitbar = bool(rs24_exitbar)
        self.rs24_asym = bool(rs24_asym)
        self.rs24_lsu = int(rs24_lsu)
        assert self.rs24_lsu == 0 or (
            self.rs24 and self.rs24_lsu % 4 == 0 and 4 <= self.rs24_lsu <= 32
        ), "rs24_lsu: multiple of 4 in [4, 32], needs rs24"
        assert not (self.rs24_asym and self.rs24_f32), "rs24_asym pushes bf16"
        self.nrows = int(nrows)
        assert 1 <= self.nrows <= 8 and (self.nrows == 1 or self.rs24), "multi-row needs rs24"
        assert not self.rs24 or (
            gjoin and (num_sms - NTEAM * (int(nteams) if nteams else self.nrows)) % 2 == 0
        ), "rs24 pairs team CTAs 2k, 2k+1 of the gjoin layout"
        self.num_sms = num_sms
        # teams of 8 CTAs; nteams < nrows shares them (team t serves rows t, t + nteams, ...)
        self.nteams = min(int(nteams), self.nrows) if nteams else self.nrows
        assert 1 <= self.nteams <= self.nrows
        self.ns = num_sms - NTEAM * self.nteams
        self.npmax = (self.nrows + self.nteams - 1) // self.nteams
        assert self.npmax <= MAXP  # pass entry words in T_OFF_PW
        assert self.ns * CAPR >= NB_INITIAL * BUCKET_CAP
        assert self.ns * MAXS * SPLIT_TOK >= N_MAX and self.ns <= 160
        self.policy = policy
        self.debug = bool(debug)
        self.stamps = bool(stamps)
        self.prewarm = bool(prewarm)
        self.skew = int(skew)
        self.coop = bool(coop)
        self.qk_ahead = bool(qk_ahead)
        self.wdelay = int(wdelay)
        self.mma_block = bool(mma_block)
        self.mma_idle_ns = int(mma_idle_ns)
        self.ctl_regs = bool(ctl_regs)
        self.gjoin = bool(gjoin)
        self.spin_ns = int(spin_ms) * 1000000
        self.acqpoll = bool(acqpoll)
        self.relst = bool(relst)
        self.ppref = bool(ppref)
        self.presel = bool(presel)
        self.pdec = bool(pdec)
        self.warm1 = bool(warm1)
        self.pollns = int(pollns)
        self.acqx = bool(acqx)
        self.swpre = bool(swpre)
        self.bpf = bool(bpf)
        self.entcg = bool(entcg)
        self.tsz01 = bool(tsz01)
        # nrows > 4: CTA ns - 1 (empty in the last row) serves the last row; tsz01 scratch in row 7's slot
        assert not self.tsz01 or self.nrows <= 4 or (bool(tsz2) or bool(tsz))
        self.tsz2 = bool(tsz2)
        self.tsz = bool(tsz)
        self.drot = int(drot)
        self.awarm3 = bool(awarm3)
        self.awarm2a = bool(awarm2a)
        self.btcl = bool(btcl)
        self.pdwait = int(pdwait)
        self.awarm2 = bool(awarm2)
        self.t1a = bool(t1a)
        self.awarm = bool(awarm)
        self.pdirect = bool(pdirect)
        assert not self.pdirect or (self.presel and self.pdec)
        self.tsrv1 = bool(tsrv1)
        self.tl1r = bool(tl1r)
        # only with shared teams (nteams < nrows), like tsz01 / tsz2 (BS1 / BS2 of the same solution: off)
        self.tszall = bool(tszall) and self.nteams < self.nrows
        assert not self.tszall or (self.tsz01 and self.tsz2), "tszall extends tsz01 + tsz2"
        assert self.swpre or self.pdec
        assert not (self.npmax > 2 and self.presel and self.swpre), (
            "misc[60] is cleared only at pass 0"
        )
        assert not self.pdec or self.presel
        assert wacq in (0, 1, 2)
        self.wacq = int(wacq)
        self.folpre = bool(folpre)
        self.heal = bool(heal)
        self.nofence = bool(nofence)
        self.lean = bool(lean)
        self.bulk = bool(bulk)
        self.fblka = bool(fblka)
        self.gen2 = bool(gen2)
        self.hook = bool(hook)
        self.pref = bool(pref)
        assert pchk in (1, 2, 4, 8, 16, 32)
        self.pchk = int(pchk)
        assert late_done in (0, 1, 2, 3, 4)
        self.late_done = int(late_done)
        assert not (late_done and (not hook))
        assert self.heal or self.hook, (
            "without the exit hook only the directive-tag witness advances the generation"
        )
        self.math_regs = math_regs
        assert math_regs * 256 + 24 * 256 <= 65536 or (
            nrows > 1 and math_regs * 256 + 80 * 128 <= 65536
        )
        self.block_kv = TILE
        self.phys_block_kv = phys_block_kv
        self.num_blocks_per_mma = TILE // phys_block_kv
        self.num_heads = 64
        self.head_dim = 128
        self.N = 64
        self.num_kv_stages = kv_stages
        self.num_umma_stages = 1
        # Q / W double-buffered in the multi-row stream (knob qdb)
        self.qdb = bool(qdb) and self.nrows > 1
        # next row's descriptor / ctx loaded one row ahead in the stream roles (knob dpf)
        self.dpf = bool(dpf) and self.nrows > 1
        self.num_q_stages = 2 if self.qdb else 1
        if self.qdb:
            # Q 2 x 8 KB, W 2 x 256 B, then the scales, over the single-row logits / block-table buffers (unused for
            # nrows > 1); the last scale buffer ends at S_OFF_BAR
            self.s_off_q, self.s_off_w, self.s_off_sc0, self.s_off_sc1 = (
                98304,
                114688,
                115200,
                116736,
            )
            assert (
                self.s_off_sc1 + TILE * 3 * 4 <= S_OFF_BAR
                and self.s_off_q + 2 * 8192 <= self.s_off_w
            )
        else:
            self.s_off_q, self.s_off_w, self.s_off_sc0, self.s_off_sc1 = (
                S_OFF_Q,
                S_OFF_W,
                S_OFF_SC0,
                S_OFF_SC1,
            )
        self.acc_dtype = cutlass.Float32
        self.epi_dtype = cutlass.Float32
        self.output_dtype = cutlass.Float32
        self.epi_bytes = 4
        w_stage_bytes = self.N * self.epi_bytes
        self.w_stage_stride = (w_stage_bytes + 127) // 128 * 128 // self.epi_bytes
        self.cta_group = tcgen05.CtaGroup.ONE
        self.cluster_shape_mn = (1, 1)
        self.mma_tiler_mn = (TILE, self.N)
        self.num_w_in_reg = 64
        assert kv_stages * TILE * 128 <= 49152

    def _epi_value(self, tTR_rAcc, w_cache, sW, NUM_W_IN_REG, scale_val):
        s0x = cutlass.Float32(0.0)
        s0y = cutlass.Float32(0.0)
        s1x = cutlass.Float32(0.0)
        s1y = cutlass.Float32(0.0)
        acc_vec = tTR_rAcc.load()
        for h in range(0, self.num_heads, 4):
            x0 = acc_vec[h]
            x1 = acc_vec[h + 1]
            x2 = acc_vec[h + 2]
            x3 = acc_vec[h + 3]
            # Direct ReLU preserves the full FP32 range of weighted logits.
            # Doubling the activation can overflow before the final halving;
            # halving scale_val can also discard a subnormal input scale.
            a0 = cute.arch.fmax(x0, Float32(0.0), nan=True)
            a1 = cute.arch.fmax(x1, Float32(0.0), nan=True)
            a2 = cute.arch.fmax(x2, Float32(0.0), nan=True)
            a3 = cute.arch.fmax(x3, Float32(0.0), nan=True)
            w0 = w_cache[h]
            w1 = w_cache[h + 1]
            w2 = w_cache[h + 2]
            w3 = w_cache[h + 3]
            s0x, s0y = cute.arch.fma_packed_f32x2((a0, a1), (w0, w1), (s0x, s0y), rnd=_RND_RN)
            s1x, s1y = cute.arch.fma_packed_f32x2((a2, a3), (w2, w3), (s1x, s1y), rnd=_RND_RN)
        result_t = s0x + s0y + s1x + s1y
        return self.output_dtype(result_t * scale_val)

    def _setup_mma(self, a_dtype, b_dtype, a_major, b_major):
        self.a_dtype = a_dtype
        self.b_dtype = b_dtype
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            a_dtype, b_dtype, a_major, b_major, self.acc_dtype, self.cta_group, self.mma_tiler_mn
        )
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        full_k = mma_inst_shape_k * (self.head_dim // mma_inst_shape_k)
        self.mma_tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], full_k)
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)), (tiled_mma.thr_id.shape,)
        )
        self.epi_tile = self.cta_tile_shape_mnk[:2]
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, a_dtype, self.num_kv_stages
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, b_dtype, self.num_q_stages
        )
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)
        self.num_tmem_alloc_cols = cutlass.utils.get_num_tmem_alloc_cols(tCtAcc_fake)
        self.num_tmem_alloc_cols_total = self.num_tmem_alloc_cols * 2 * self.num_umma_stages
        return tiled_mma

    @cute.jit
    def __call__(
        self,
        kv_fused: cute.Tensor,
        b: cute.Tensor,
        weights: cute.Tensor,
        block_table: cute.Tensor,
        ctx: cute.Tensor,
        desc: cute.Tensor,
        q_att: cute.Tensor,
        swa_pool: cute.Tensor,
        c_pool: cute.Tensor,
        swa_local: cute.Tensor,
        bt_swa: cute.Tensor,
        bt_c: cute.Tensor,
        sinks: cute.Tensor,
        swa_lens: cute.Tensor,
        topk_out: cute.Tensor,
        gidx: cute.Tensor,
        out: cute.Tensor,
        ws64: cute.Tensor,
        hist: cute.Tensor,
        board: cute.Tensor,
        region: cute.Tensor,
        dbg_logits: cute.Tensor,
        stamps: cute.Tensor,
        ctl: cute.Tensor,
        num_phys_blocks: cutlass.Int32,
        softmax_scale: Float32,
        stream: cuda_driver.CUstream,
    ):
        phys_block_kv = self.phys_block_kv
        scale_offset_elems = phys_block_kv * self.head_dim
        kv_fp8 = cute.recast_tensor(kv_fused, cutlass.Float8E4M3FN)
        b = cute.recast_tensor(b, cutlass.Float8E4M3FN)
        kv_block_stride = kv_fused.layout.stride[0]
        a = cute.make_tensor(
            kv_fp8.iterator,
            cute.make_layout(
                (phys_block_kv, self.head_dim, num_phys_blocks),
                stride=(self.head_dim, 1, kv_block_stride),
            ),
        )
        scale_fp8 = cute.make_tensor(
            kv_fp8.iterator + scale_offset_elems,
            cute.make_layout((phys_block_kv * 4, num_phys_blocks), stride=(1, kv_block_stride)),
        )
        scales = cute.recast_tensor(scale_fp8, cutlass.Float32)
        a_major = cutlass.utils.LayoutEnum.from_tensor(a).mma_major_mode()
        b_major = cutlass.utils.LayoutEnum.ROW_MAJOR.mma_major_mode()
        tiled_mma = self._setup_mma(a.element_type, b.element_type, a_major, b_major)
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp()
        a_tma_view = sm100_utils.make_smem_layout(
            cute.nvgpu.OperandMajorMode.K,
            (self.block_kv, self.head_dim),
            a.element_type,
            self.num_kv_stages,
        )
        a_tma_view = cute.tiled_divide(a_tma_view, (self.phys_block_kv, self.head_dim))
        a_tma_view = cute.select(a_tma_view, mode=[0, 1, 3])
        self.a_tma_view_layout = a_tma_view
        tma_atom_a, tma_tensor_a = cpasync.make_tiled_tma_atom(
            tma_load_op, a, a_tma_view[0], (self.phys_block_kv, self.head_dim)
        )
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, b, b_smem_layout, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape
        )
        self.w_smem_layout_staged = cute.make_layout(
            (self.N, self.num_q_stages), stride=(1, self.w_stage_stride)
        )
        tma_atom_w, tma_tensor_w = cpasync.make_tiled_tma_atom(
            tma_load_op,
            weights,
            cute.select(self.w_smem_layout_staged, mode=[0]),
            self.w_smem_layout_staged.shape[:1],
        )
        self.s_smem_layout_staged = cute.make_layout((self.block_kv, self.num_kv_stages))
        tma_atom_s, tma_tensor_s = cpasync.make_tiled_tma_atom(
            tma_load_op, scales, cute.make_layout((phys_block_kv,)), (phys_block_kv,)
        )
        b_copy_size = cute.size_in_bytes(b.element_type, b_smem_layout)
        self.num_kv_scale_tma_bytes = self.num_blocks_per_mma * (
            phys_block_kv * self.head_dim + phys_block_kv * 4
        )
        self.num_q_tma_bytes = b_copy_size * atom_thr_size + self.N * self.epi_bytes
        tma_q = cudax.create_tensor_map_tiled(
            global_address=q_att.iterator.toint(),
            dtype=A.E4M3,
            global_dims=[A.LATENT, A.HEADS, q_att.shape[0]],
            global_strides=[A.LATENT // 16, A.HEADS * A.LATENT // 16],
            box_dims=[A.SLICE, A.HEADS, 1],
            swizzle=cudax.TensorMapSwizzle.s128b,
        )
        tma_swa = cudax.create_tensor_map_tiled(
            global_address=swa_pool.iterator.toint(),
            dtype=A.E4M3,
            global_dims=[A.LATENT, swa_pool.shape[0]],
            global_strides=[A.LATENT // 16],
            box_dims=[A.SLICE, 1],
            swizzle=cudax.TensorMapSwizzle.s128b,
        )
        tma_cmp = cudax.create_tensor_map_tiled(
            global_address=c_pool.iterator.toint(),
            dtype=A.E4M3,
            global_dims=[A.LATENT, c_pool.shape[0]],
            global_strides=[A.LATENT // 16],
            box_dims=[A.SLICE, 1],
            swizzle=cudax.TensorMapSwizzle.s128b,
        )
        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_w,
            tma_tensor_w,
            tma_atom_s,
            tma_tensor_s,
            tma_q,
            tma_swa,
            tma_cmp,
            out,
            block_table,
            ctx,
            desc,
            swa_local,
            bt_swa,
            bt_c,
            sinks,
            swa_lens,
            topk_out,
            gidx,
            ws64,
            hist,
            board,
            region,
            dbg_logits,
            stamps,
            ctl,
            c_pool,
            softmax_scale,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.w_smem_layout_staged,
            self.s_smem_layout_staged,
            self.a_tma_view_layout,
            self.epi_tile,
        ).launch(
            grid=(self.num_sms, 1, 1),
            block=[512, 1, 1],
            cluster=(2 if self.rs24 else 1, 1, 1),
            stream=stream,
            min_blocks_per_mp=1,
            cooperative=self.coop,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_w: cute.CopyAtom,
        mW_tma: cute.Tensor,
        tma_atom_s: cute.CopyAtom,
        mS_tma: cute.Tensor,
        tma_q: cutlass.GridConstant[cudax.TensorMap],
        tma_swa: cutlass.GridConstant[cudax.TensorMap],
        tma_cmp: cutlass.GridConstant[cudax.TensorMap],
        mOut: cute.Tensor,
        mBlockTable: cute.Tensor,
        mCtx: cute.Tensor,
        mDesc: cute.Tensor,
        mSwaLocal: cute.Tensor,
        mBtSwa: cute.Tensor,
        mBtC: cute.Tensor,
        mSinks: cute.Tensor,
        mSwaLens: cute.Tensor,
        mTopk: cute.Tensor,
        mGidx: cute.Tensor,
        mWs: cute.Tensor,
        mHist: cute.Tensor,
        mBoard: cute.Tensor,
        mRegion: cute.Tensor,
        mDbg: cute.Tensor,
        mStamps: cute.Tensor,
        mCtl: cute.Tensor,
        mCPool: cute.Tensor,
        softmax_scale: Float32,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        w_smem_layout_staged: cute.Layout,
        s_smem_layout_staged: cute.Layout,
        a_tma_view_layout: cute.ComposedLayout,
        epi_tile: cute.Tile,
    ):
        bidx, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        smem = cutlass.utils.SmemAllocator()
        arena = smem.allocate(ARENA, byte_alignment=1024)
        ws = mWs.iterator.toint()
        if cutlass.const_expr(self.stamps):
            if tidx == 0:
                (mStamps.iterator + bidx * NSTAMP + ST_ENTRY).store(Int64(L.gtimer()))
        if bidx < self.ns:
            self._stream_cta(
                arena,
                bidx,
                tidx,
                ws,
                mCtl,
                tiled_mma,
                tma_atom_a,
                mA_mkl,
                tma_atom_b,
                mB_nkl,
                tma_atom_w,
                mW_tma,
                tma_atom_s,
                mS_tma,
                mBlockTable,
                mCtx,
                mDesc,
                mBtC,
                mHist,
                mBoard,
                mRegion,
                mDbg,
                mStamps,
                mCPool,
                cluster_layout_vmnk,
                a_smem_layout_staged,
                b_smem_layout_staged,
                w_smem_layout_staged,
                s_smem_layout_staged,
                a_tma_view_layout,
                epi_tile,
            )
        else:
            if cutlass.const_expr(self.nrows == 1):
                self._team_cta(
                    arena,
                    bidx,
                    tidx,
                    ws,
                    tma_q,
                    tma_swa,
                    tma_cmp,
                    mOut,
                    mCtx,
                    mSwaLocal,
                    mBtSwa,
                    mBtC,
                    mSinks,
                    mSwaLens,
                    mTopk,
                    mGidx,
                    mHist,
                    mBoard,
                    mRegion,
                    mStamps,
                    mCtl,
                    softmax_scale,
                    mDesc,
                    mCPool,
                )
            elif cutlass.const_expr(self.nteams < self.nrows):
                # shared teams: team t = CTAs ns + 8t .. ns + 8t + 7 serves rows t, t + nteams, ...
                tm_ = (bidx - Int32(self.ns)) // Int32(NTEAM)
                cta_t = Int32(self.ns) + (bidx - Int32(self.ns)) % Int32(NTEAM)
                self._team_cta_mr(
                    arena,
                    cta_t,
                    tidx,
                    ws,
                    tma_q,
                    tma_swa,
                    tma_cmp,
                    mOut,
                    mCtx,
                    mSwaLocal,
                    mBtSwa,
                    mBtC,
                    mSinks,
                    mSwaLens,
                    mTopk,
                    mGidx,
                    mHist,
                    mBoard,
                    mRegion,
                    cute.make_tensor(
                        mStamps.iterator + (bidx - cta_t) * Int32(NSTAMP), mStamps.layout
                    ),
                    mCtl,
                    softmax_scale,
                    mDesc,
                    mCPool,
                    tm_,
                )
            else:
                # team r = CTAs ns + 8r .. ns + 8r + 7, unchanged team code on row views
                rs_ = self.row_sizes()
                tr_ = (bidx - Int32(self.ns)) // Int32(NTEAM)
                cta_t = Int32(self.ns) + (bidx - Int32(self.ns)) % Int32(NTEAM)
                self._team_cta(
                    arena,
                    cta_t,
                    tidx,
                    ws + Int64(tr_) * Int64(rs_["ws"] * 8),
                    tma_q,
                    tma_swa,
                    tma_cmp,
                    _row2(mOut, tr_),
                    _row1(mCtx, tr_),
                    _row2(mSwaLocal, tr_),
                    _row2(mBtSwa, tr_),
                    _row2(mBtC, tr_),
                    mSinks,
                    _row1(mSwaLens, tr_),
                    _row2(mTopk, tr_),
                    _row2(mGidx, tr_),
                    _flat(mHist, tr_, rs_["hist"]),
                    _flat(mBoard, tr_, rs_["board"]),
                    _flat(mRegion, tr_, rs_["region"]),
                    cute.make_tensor(
                        mStamps.iterator + (bidx - cta_t) * Int32(NSTAMP), mStamps.layout
                    ),
                    _flat(mCtl, tr_, rs_["ctl"]),
                    softmax_scale,
                    _flat(mDesc, tr_, rs_["desc"]),
                    mCPool,
                    tr_,
                )

    def row_sizes(self):
        # per-row element counts of the flat buffers (ws64, hist, board, region, ctl, desc)
        return dict(
            ws=WS_WORDS,
            hist=NGH * NB_DRS,
            board=BOARD_CAP,
            region=self.ns * CAPR * 2,
            ctl=16,
            desc=self.num_sms * D.DESC_WORDS,
        )

    @staticmethod
    def _w(ws, i):
        return ws + Int64(i) * Int64(8)

    @cute.jit
    def _skew(self, gen, cta, salt):
        if cutlass.const_expr(self.skew > 0):
            L.skew_sleep(
                Int32(Uint32(gen & Uint64(4294967295))),
                cta * Int32(64) + Int32(salt),
                Int32(self.skew),
            )

    @cute.jit
    def _stamp(self, mStamps, cta, slot):
        if cutlass.const_expr(self.stamps):
            (mStamps.iterator + cta * NSTAMP + slot).store(Int64(L.gtimer()))

    @cute.jit
    def _stream_cta(
        self,
        arena,
        cta,
        tidx,
        ws,
        mCtl,
        tiled_mma,
        tma_atom_a,
        mA_mkl,
        tma_atom_b,
        mB_nkl,
        tma_atom_w,
        mW_tma,
        tma_atom_s,
        mS_tma,
        mBlockTable,
        mCtx,
        mDesc,
        mBtC,
        mHist,
        mBoard,
        mRegion,
        mDbg,
        mStamps,
        mCPool,
        cluster_layout_vmnk,
        a_smem_layout_staged,
        b_smem_layout_staged,
        w_smem_layout_staged,
        s_smem_layout_staged,
        a_tma_view_layout,
        epi_tile,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = tidx % 32
        NUM_BLOCKS_PER_MMA = self.num_blocks_per_mma
        gw1 = Uint64(0)
        gw2 = Uint64(0)
        gw3 = Uint64(0)
        if tidx == 0:
            gw1, gw2, gw3 = self._entry_loads(ws)
        desc_lane = mDesc[cta * 32 + lane]
        s0 = cute.arch.shuffle_sync(desc_lane, 0)
        ns = cute.arch.shuffle_sync(desc_lane, 1)
        n_desc = cute.arch.shuffle_sync(desc_lane, 2)
        ord0 = cute.arch.shuffle_sync(desc_lane, 3)
        ord1 = cute.arch.shuffle_sync(desc_lane, 4)
        nwin = cute.arch.shuffle_sync(desc_lane, 5)
        n = n_desc
        SMP, SS2, TGT, TGT2 = L.gvr_ladder(
            n,
            Int32(K_TOP),
            Int32(PLAN["aim_base"]),
            Int32(PLAN["sfac"]),
            Int32(PLAN["amin"]),
            Int32(PLAN["sd_en"]),
            PLAN["r_const"],
            PLAN["scap"],
        )
        wst = Int32(8) * SS2
        t_lo = s0 * Int32(SPLIT_TOK)
        nwin0 = (nwin + Int32(1)) // Int32(2)
        nwin1 = nwin // Int32(2)
        storage_bar = cute.recast_ptr(arena + S_OFF_BAR, dtype=Int64)
        misc_i32 = cute.recast_ptr(arena + S_OFF_MISC, dtype=Int32)
        kv_mbar_0 = storage_bar
        kv_mbar_1 = storage_bar + 2 * self.num_kv_stages
        q_mbar = storage_bar + 4 * self.num_kv_stages
        umma_mbar_0 = q_mbar + 2 * self.num_q_stages
        umma_mbar_1 = umma_mbar_0 + 2
        tmem_hold = misc_i32
        _s_tl = cute.make_tensor(misc_i32 + 4, cute.make_layout(8))
        s_scan = cute.make_tensor(misc_i32 + 16, cute.make_layout(32))
        s_tv = cute.make_tensor(misc_i32 + 48, cute.make_layout(4))
        s_wc = cute.make_tensor(misc_i32 + 52, cute.make_layout(8))
        sSub = cute.make_tensor(
            cute.recast_ptr(arena + S_OFF_SUB, dtype=Int64), cute.make_layout(8 * SUBCAP)
        )
        tbar = storage_bar + S_BAR_T
        q_free = storage_bar + (S_BAR_T + 1)
        if tidx == 0:
            prims.mbarrier_init(tbar, 1)
            if cutlass.const_expr(self.nrows > 1):
                # the Q / W stage is free again once both MMA groups' last MMAs of a row completed (tcgen05.commit)
                # and the math warps read the row's weights
                prims.mbarrier_init(q_free, 3)
                if cutlass.const_expr(self.qdb):
                    prims.mbarrier_init(q_free + 1, 3)
                for rb_ in cutlass.range_constexpr(8):
                    prims.mbarrier_init(cute.recast_ptr(arena + S_OFF_SVM, dtype=Int64) + rb_, 1)
                    # row T armed by the T poller
                    prims.mbarrier_init(cute.recast_ptr(arena + S_OFF_SVTB, dtype=Int64) + rb_, 1)
            prims.fence_mbarrier_init()
        sLog = cute.make_tensor(
            cute.recast_ptr(arena + S_OFF_LOG, dtype=Float32), cute.make_layout(MAXS * SPLIT_TOK)
        )
        sBtc = cute.make_tensor(
            cute.recast_ptr(arena + S_OFF_BTC, dtype=Int32), cute.make_layout(MAXS * 8)
        )
        prod_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        kv_cons_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 4)
        q_cons_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 8)
        q_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=q_mbar,
            num_stages=self.num_q_stages,
            producer_group=prod_group,
            consumer_group=q_cons_group,
            tx_count=self.num_q_tma_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            tidx=tidx,
            defer_sync=True,
        )
        kv_pipeline_0 = pipeline.PipelineTmaAsync.create(
            barrier_storage=kv_mbar_0,
            num_stages=self.num_kv_stages,
            producer_group=prod_group,
            consumer_group=kv_cons_group,
            tx_count=self.num_kv_scale_tma_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            tidx=tidx,
            defer_sync=True,
        )
        kv_pipeline_1 = pipeline.PipelineTmaAsync.create(
            barrier_storage=kv_mbar_1,
            num_stages=self.num_kv_stages,
            producer_group=prod_group,
            consumer_group=kv_cons_group,
            tx_count=self.num_kv_scale_tma_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            tidx=tidx,
            defer_sync=True,
        )
        if cutlass.const_expr(self.rs24):
            lane0_ = tidx % Int32(32) == Int32(0)
            q_pipeline = dataclasses.replace(q_pipeline, is_signaling_thread=lane0_)
            kv_pipeline_0 = dataclasses.replace(kv_pipeline_0, is_signaling_thread=lane0_)
            kv_pipeline_1 = dataclasses.replace(kv_pipeline_1, is_signaling_thread=lane0_)
        umma_pipeline_0 = pipeline.PipelineUmmaAsync.create(
            barrier_storage=umma_mbar_0,
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 128),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        umma_pipeline_1 = pipeline.PipelineUmmaAsync.create(
            barrier_storage=umma_mbar_1,
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 128),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        q_prod_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_q_stages
        )
        q_cons_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_q_stages
        )
        kv_prod_state_0 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_kv_stages
        )
        kv_prod_state_1 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_kv_stages
        )
        kv_cons_state_umma_0 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_kv_stages
        )
        kv_cons_state_umma_1 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_kv_stages
        )
        kv_cons_state_math_0 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_kv_stages
        )
        kv_cons_state_math_1 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_kv_stages
        )
        umma_prod_state_0 = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
        umma_prod_state_1 = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
        umma_cons_state_0 = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
        umma_cons_state_1 = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
        tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=320)
        tmem = cutlass.utils.TmemAllocator(
            tmem_hold,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=0,
            is_two_cta=False,
        )
        gen_slot = cute.make_tensor(
            cute.recast_ptr(arena + S_OFF_MISC + 256, dtype=Int64), cute.make_layout(4)
        )
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)
        sKV_0 = cute.make_tensor(
            cute.recast_ptr(arena + S_OFF_KV0, a_smem_layout_staged.inner, dtype=self.a_dtype),
            a_smem_layout_staged.outer,
        )
        sKV_1 = cute.make_tensor(
            cute.recast_ptr(arena + S_OFF_KV1, a_smem_layout_staged.inner, dtype=self.a_dtype),
            a_smem_layout_staged.outer,
        )
        sQ = cute.make_tensor(
            cute.recast_ptr(arena + self.s_off_q, b_smem_layout_staged.inner, dtype=self.b_dtype),
            b_smem_layout_staged.outer,
        )
        sW = cute.make_tensor(
            cute.recast_ptr(arena + self.s_off_w, dtype=self.epi_dtype), w_smem_layout_staged
        )
        sScales_0 = cute.make_tensor(
            cute.recast_ptr(arena + self.s_off_sc0, dtype=cutlass.Float32), s_smem_layout_staged
        )
        sScales_1 = cute.make_tensor(
            cute.recast_ptr(arena + self.s_off_sc1, dtype=cutlass.Float32), s_smem_layout_staged
        )
        thr_mma = tiled_mma.get_slice(0)
        sKV_0_for_tma = cute.make_tensor(sKV_0.iterator, a_tma_view_layout.outer)
        sKV_1_for_tma = cute.make_tensor(sKV_1.iterator, a_tma_view_layout.outer)
        gA = cute.local_tile(mA_mkl, (self.phys_block_kv, self.head_dim), coord=(None, None, None))
        tAsA_0, tAgA_0 = cpasync.tma_partition(
            tma_atom_a, 0, cute.make_layout(1), sKV_0_for_tma, cute.group_modes(gA, 0, 2)
        )
        tAsA_1, tAgA_1 = cpasync.tma_partition(
            tma_atom_a, 0, cute.make_layout(1), sKV_1_for_tma, cute.group_modes(gA, 0, 2)
        )
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        tCgB = thr_mma.partition_B(gB_nkl)
        b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b, 0, b_cta_layout, cute.group_modes(sQ, 0, 3), cute.group_modes(tCgB, 0, 3)
        )
        tBgB = tBgB[None, 0, None, None]
        tWsW, tWgW = cpasync.tma_partition(
            tma_atom_w,
            0,
            cute.make_layout((1,)),
            cute.group_modes(sW, 0, 1),
            cute.group_modes(mW_tma, 0, 1),
        )
        s_tma_view_layout = cute.make_layout(
            (self.phys_block_kv, self.num_blocks_per_mma, self.num_kv_stages),
            stride=(1, self.phys_block_kv, self.block_kv),
        )
        sScales_0_for_tma = cute.make_tensor(sScales_0.iterator, s_tma_view_layout)
        sScales_1_for_tma = cute.make_tensor(sScales_1.iterator, s_tma_view_layout)
        gS = cute.local_tile(mS_tma, (self.phys_block_kv,), coord=(None, None))
        tSsS_0, tSgS_0 = cpasync.tma_partition(
            tma_atom_s, 0, cute.make_layout(1), sScales_0_for_tma, gS
        )
        tSsS_1, tSgS_1 = cpasync.tma_partition(
            tma_atom_s, 0, cute.make_layout(1), sScales_1_for_tma, gS
        )
        tCrA_0 = tiled_mma.make_fragment_A(sKV_0)
        tCrA_1 = tiled_mma.make_fragment_A(sKV_1)
        tCrB = tiled_mma.make_fragment_B(sQ)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        acc_shape_staged = cute.append(acc_shape, 1)
        tCtAcc_fake_staged = tiled_mma.make_fragment_C(acc_shape_staged)
        cols_per_group = self.num_tmem_alloc_cols * 1 * (32 // self.acc_dtype.width)
        c_layout = cutlass.utils.LayoutEnum.ROW_MAJOR
        epi_sub_mn = (epi_tile[0], self.num_heads)
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk, c_layout, self.acc_dtype, self.acc_dtype, epi_sub_mn, False
        )
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)
        if cutlass.const_expr(self.nrows == 1):
            if warp_idx == 8 or warp_idx == 9:
                cute.arch.warpgroup_reg_dealloc(24)
                grp = warp_idx - 8
                cpasync.prefetch_descriptor(tma_atom_a)
                cpasync.prefetch_descriptor(tma_atom_s)
                pol = None
                if cutlass.const_expr(self.policy != CP.POL_NONE):
                    pol = CP.make_l2_policy(self.policy)
                bt_cols = cute.size(mBlockTable, mode=[1])
                bta = mBlockTable[0, cutlass.min(Int32(8) * s0 + lane, bt_cols - 1)]
                btb = mBlockTable[0, cutlass.min(Int32(8) * s0 + Int32(32) + lane, bt_cols - 1)]
                ntile_row = (n + Int32(TILE - 1)) // Int32(TILE)
                t_ref = Int64(0)
                if cutlass.const_expr(self.wdelay > 0):
                    t_ref = Int64(L.gtimer())
                if grp == 0:
                    cpasync.prefetch_descriptor(tma_atom_b)
                    cpasync.prefetch_descriptor(tma_atom_w)
                    q_pipeline.producer_acquire(q_prod_state)
                    q_bar = q_pipeline.producer_get_barrier(q_prod_state)
                    cute.copy(
                        tma_atom_b,
                        tBgB[None, 0, 0],
                        tBsB[None, q_prod_state.index],
                        tma_bar_ptr=q_bar,
                    )
                    cute.copy(
                        tma_atom_w, tWgW[None, 0], tWsW[None, q_prod_state.index], tma_bar_ptr=q_bar
                    )
                    q_prod_state.advance()
                    for i in cutlass.range(ns, unroll=1):
                        if cutlass.const_expr(self.wdelay > 0):
                            if i == nwin0:
                                tnow = Int64(L.gtimer())
                                while tnow - t_ref < Int64(self.wdelay):
                                    tnow = Int64(L.gtimer())
                        tl = ord0 >> Int32(4) * i & Int32(15)
                        tile = Int32(2) * s0 + tl
                        tile_ok = tile < ntile_row
                        phys = [Int32(0)] * NUM_BLOCKS_PER_MMA
                        if i < 3:
                            for k in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                                phys[k] = cute.arch.shuffle_sync(
                                    desc_lane, Int32(8 + 4 * 0 + k) + Int32(8) * i
                                )
                        else:
                            for k in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                                rel = Int32(4) * tl + Int32(k)
                                pa = cute.arch.shuffle_sync(bta, rel & Int32(31))
                                pb = cute.arch.shuffle_sync(btb, rel & Int32(31))
                                pv = cutlass.select_(rel >= Int32(32), pb, pa)
                                phys[k] = cutlass.select_(tile_ok, pv, Int32(0))
                        kv_pipeline_0.producer_acquire(kv_prod_state_0)
                        if cutlass.const_expr(self.stamps):
                            if i == 0:
                                if lane == 0:
                                    (mStamps.iterator + cta * NSTAMP + ST_FIRST_TMA).store(
                                        Int64(L.gtimer())
                                    )
                        bar = kv_pipeline_0.producer_get_barrier(kv_prod_state_0)
                        stage = kv_prod_state_0.index
                        for k in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                            cute.copy(
                                tma_atom_a,
                                tAgA_0[None, 0, 0, phys[k]],
                                tAsA_0[None, k, stage],
                                tma_bar_ptr=bar,
                                cache_policy=pol,
                            )
                            cute.copy(
                                tma_atom_s,
                                tSgS_0[None, 0, phys[k]],
                                tSsS_0[None, k, stage],
                                tma_bar_ptr=bar,
                                cache_policy=pol,
                            )
                        kv_prod_state_0.advance()
                else:
                    for i in cutlass.range(ns, unroll=1):
                        if cutlass.const_expr(self.wdelay > 0):
                            if i == nwin1:
                                tnow = Int64(L.gtimer())
                                while tnow - t_ref < Int64(self.wdelay):
                                    tnow = Int64(L.gtimer())
                        tl = ord1 >> Int32(4) * i & Int32(15)
                        tile = Int32(2) * s0 + tl
                        tile_ok = tile < ntile_row
                        phys = [Int32(0)] * NUM_BLOCKS_PER_MMA
                        if i < 3:
                            for k in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                                phys[k] = cute.arch.shuffle_sync(
                                    desc_lane, Int32(8 + 4 * 1 + k) + Int32(8) * i
                                )
                        else:
                            for k in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                                rel = Int32(4) * tl + Int32(k)
                                pa = cute.arch.shuffle_sync(bta, rel & Int32(31))
                                pb = cute.arch.shuffle_sync(btb, rel & Int32(31))
                                pv = cutlass.select_(rel >= Int32(32), pb, pa)
                                phys[k] = cutlass.select_(tile_ok, pv, Int32(0))
                        kv_pipeline_1.producer_acquire(kv_prod_state_1)
                        bar = kv_pipeline_1.producer_get_barrier(kv_prod_state_1)
                        stage = kv_prod_state_1.index
                        for k in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                            cute.copy(
                                tma_atom_a,
                                tAgA_1[None, 0, 0, phys[k]],
                                tAsA_1[None, k, stage],
                                tma_bar_ptr=bar,
                                cache_policy=pol,
                            )
                            cute.copy(
                                tma_atom_s,
                                tSgS_1[None, 0, phys[k]],
                                tSsS_1[None, k, stage],
                                tma_bar_ptr=bar,
                                cache_policy=pol,
                            )
                        kv_prod_state_1.advance()
            elif warp_idx == 10 or warp_idx == 11:
                cute.arch.warpgroup_reg_dealloc(24)
                grp = warp_idx - 10
                L.bar(1, 320)
                tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
                tCtAcc_base_0 = cute.make_tensor(tmem_ptr, tCtAcc_fake_staged.layout)
                tCtAcc_base_1 = cute.make_tensor(
                    tmem_ptr + cols_per_group, tCtAcc_fake_staged.layout
                )
                num_k_blocks = cute.size(tCrA_0.shape[2])
                q_pipeline.consumer_wait(q_cons_state)
                q_stage = q_cons_state.index
                if grp == 0:
                    for i in cutlass.range(ns, unroll=1):
                        kv_pipeline_0.consumer_wait(kv_cons_state_umma_0)
                        umma_pipeline_0.producer_acquire(umma_prod_state_0)
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                        kv_stage = kv_cons_state_umma_0.index
                        tCtAcc_0 = tCtAcc_base_0[None, None, None, umma_prod_state_0.index]
                        for k_block in cutlass.range_constexpr(num_k_blocks):
                            cute.gemm(
                                tiled_mma,
                                tCtAcc_0,
                                tCrA_0[None, None, k_block, kv_stage],
                                tCrB[None, None, k_block, q_stage],
                                tCtAcc_0,
                            )
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        kv_cons_state_umma_0.advance()
                        umma_pipeline_0.producer_commit(umma_prod_state_0)
                        umma_prod_state_0.advance()
                else:
                    for i in cutlass.range(ns, unroll=1):
                        kv_pipeline_1.consumer_wait(kv_cons_state_umma_1)
                        umma_pipeline_1.producer_acquire(umma_prod_state_1)
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                        kv_stage_1 = kv_cons_state_umma_1.index
                        tCtAcc_1 = tCtAcc_base_1[None, None, None, umma_prod_state_1.index]
                        for k_block in cutlass.range_constexpr(num_k_blocks):
                            cute.gemm(
                                tiled_mma,
                                tCtAcc_1,
                                tCrA_1[None, None, k_block, kv_stage_1],
                                tCrB[None, None, k_block, q_stage],
                                tCtAcc_1,
                            )
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        kv_cons_state_umma_1.advance()
                        umma_pipeline_1.producer_commit(umma_prod_state_1)
                        umma_prod_state_1.advance()
            elif warp_idx < 8:
                cute.arch.warpgroup_reg_alloc(self.math_regs)
                wg = warp_idx // 4
                tmem.allocate(self.num_tmem_alloc_cols_total)
                L.bar(1, 320)
                tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
                tCtAcc_base_0 = cute.make_tensor(tmem_ptr, tCtAcc_fake_staged.layout)
                tCtAcc_base_1 = cute.make_tensor(
                    tmem_ptr + cols_per_group, tCtAcc_fake_staged.layout
                )
                local_tidx = tidx % 128
                cC = cute.make_identity_tensor(epi_sub_mn)
                if tidx == 0:
                    gen_slot[0] = Int64(gw1)
                    gen_slot[1] = Int64(gw2)
                    gen_slot[2] = Int64(gw3)
                q_pipeline.consumer_wait(q_cons_state)
                q_stage_local = q_cons_state.index
                w_cache = cute.make_rmem_tensor((self.num_w_in_reg,), self.epi_dtype)
                for w_j in cutlass.range_constexpr(self.num_w_in_reg):
                    w_cache[w_j] = sW[w_j, q_stage_local]
                L.bar(6, 288)
                gen = self._gen_from(Uint64(gen_slot[0]), Uint64(gen_slot[1]), Uint64(gen_slot[2]))
                if tidx == 0:
                    if n_desc != mCtx[0]:
                        self._flag(ws, gen, Int32(E_STALE))
                    if n_desc < Int32(0) or n_desc > Int32(N_MAX):
                        self._flag(ws, gen, Int32(E_NLIMIT))
                bboard = mBoard.iterator.toint()
                btag = Uint64(self._tag32(gen)) << Uint64(32)
                ci = Int32(0)
                cnt1 = Int32(0)
                tk = Int32(0)
                Tv = Float32(float("inf"))
                wsub = warp_idx * Int32(SUBCAP)
                if wg == 0:
                    tAcc_0_ref = tCtAcc_base_0[None, None, None, 0][(None, None), 0, 0]
                    tAcc_0_ref_epi = cute.flat_divide(tAcc_0_ref, epi_sub_mn)
                    tiled_copy_ref_0 = tcgen05.make_tmem_copy(
                        copy_atom_t2r, tAcc_0_ref_epi[None, None, 0, 0]
                    )
                    thr_copy_ref_0 = tiled_copy_ref_0.get_slice(local_tidx)
                    tTR_cC = thr_copy_ref_0.partition_D(cC)
                    m_coord = tTR_cC[0][0]
                    tTR_rAcc = cute.make_fragment_like(tTR_cC, self.acc_dtype)
                    for i in cutlass.range(ns + Int32(1), unroll=1):
                        if i < ns:
                            tl = ord0 >> Int32(4) * i & Int32(15)
                            pos = (Int32(2) * s0 + tl) * Int32(TILE) + m_coord
                            kv_ready = kv_pipeline_0.consumer_try_wait(kv_cons_state_math_0)
                            umma_ready = umma_pipeline_0.consumer_try_wait(umma_cons_state_0)
                            kv_pipeline_0.consumer_wait(kv_cons_state_math_0, kv_ready)
                            scale_val = sScales_0[m_coord, kv_cons_state_math_0.index]
                            umma_pipeline_0.consumer_wait(umma_cons_state_0, umma_ready)
                            tCtAcc_c0 = tCtAcc_base_0[None, None, None, umma_cons_state_0.index]
                            tAcc_c0 = tCtAcc_c0[(None, None), 0, 0]
                            tTR_0 = thr_copy_ref_0.partition_S(
                                cute.flat_divide(tAcc_c0, epi_sub_mn)
                            )
                            kv_pipeline_0.consumer_release(kv_cons_state_math_0)
                            kv_cons_state_math_0.advance()
                            cute.copy(tiled_copy_ref_0, tTR_0[None, None, None, 0, 0], tTR_rAcc)
                            cute.arch.fence_view_async_tmem_load()
                            umma_pipeline_0.consumer_release(umma_cons_state_0)
                            umma_cons_state_0.advance()
                            val = self._epi_value(
                                tTR_rAcc, w_cache, sW, self.num_w_in_reg, scale_val
                            )
                            sLog[tl * Int32(TILE) + m_coord] = val
                            if cutlass.const_expr(self.debug):
                                if pos < n:
                                    mDbg[pos] = val
                            jw = pos // wst
                            if pos - jw * wst < Int32(8):
                                if jw < SMP:
                                    st_rlx_u64(
                                        bboard + Int64(jw * Int32(8) + (pos - jw * wst)) * Int64(8),
                                        btag | Uint64(L.u32_of_f32(val)),
                                    )
                            if cutlass.const_expr(self.stamps):
                                if i == 0:
                                    if local_tidx == 0:
                                        (mStamps.iterator + cta * NSTAMP + ST_FIRST_EPI).store(
                                            Int64(L.gtimer())
                                        )
                            if cutlass.const_expr(self.stamps):
                                if i + Int32(1) == nwin0:
                                    if local_tidx == 0:
                                        (mStamps.iterator + cta * NSTAMP + ST_BOARD).store(
                                            Int64(L.gtimer())
                                        )
                        tk, Tv = self._t_known(tk, Tv, i, ns, tbar, s_tv)
                        if tk != Int32(0):
                            ci, cnt1 = self._classify_tasks(
                                ci,
                                cutlass.min(i, ns - Int32(1)),
                                ord0,
                                t_lo,
                                n,
                                m_coord,
                                sLog,
                                sSub,
                                wsub,
                                cnt1,
                                Tv,
                            )
                else:
                    tAcc_1_ref = tCtAcc_base_1[None, None, None, 0][(None, None), 0, 0]
                    tAcc_1_ref_epi = cute.flat_divide(tAcc_1_ref, epi_sub_mn)
                    tiled_copy_ref_1 = tcgen05.make_tmem_copy(
                        copy_atom_t2r, tAcc_1_ref_epi[None, None, 0, 0]
                    )
                    thr_copy_ref_1 = tiled_copy_ref_1.get_slice(local_tidx)
                    tTR_cC = thr_copy_ref_1.partition_D(cC)
                    m_coord = tTR_cC[0][0]
                    tTR_rAcc = cute.make_fragment_like(tTR_cC, self.acc_dtype)
                    for i in cutlass.range(ns + Int32(1), unroll=1):
                        if i < ns:
                            tl = ord1 >> Int32(4) * i & Int32(15)
                            pos = (Int32(2) * s0 + tl) * Int32(TILE) + m_coord
                            kv_ready = kv_pipeline_1.consumer_try_wait(kv_cons_state_math_1)
                            umma_ready = umma_pipeline_1.consumer_try_wait(umma_cons_state_1)
                            kv_pipeline_1.consumer_wait(kv_cons_state_math_1, kv_ready)
                            scale_val = sScales_1[m_coord, kv_cons_state_math_1.index]
                            umma_pipeline_1.consumer_wait(umma_cons_state_1, umma_ready)
                            tCtAcc_c1 = tCtAcc_base_1[None, None, None, umma_cons_state_1.index]
                            tAcc_c1 = tCtAcc_c1[(None, None), 0, 0]
                            tTR_1 = thr_copy_ref_1.partition_S(
                                cute.flat_divide(tAcc_c1, epi_sub_mn)
                            )
                            kv_pipeline_1.consumer_release(kv_cons_state_math_1)
                            kv_cons_state_math_1.advance()
                            cute.copy(tiled_copy_ref_1, tTR_1[None, None, None, 0, 0], tTR_rAcc)
                            cute.arch.fence_view_async_tmem_load()
                            umma_pipeline_1.consumer_release(umma_cons_state_1)
                            umma_cons_state_1.advance()
                            val = self._epi_value(
                                tTR_rAcc, w_cache, sW, self.num_w_in_reg, scale_val
                            )
                            sLog[tl * Int32(TILE) + m_coord] = val
                            if cutlass.const_expr(self.debug):
                                if pos < n:
                                    mDbg[pos] = val
                            jw = pos // wst
                            if pos - jw * wst < Int32(8):
                                if jw < SMP:
                                    st_rlx_u64(
                                        bboard + Int64(jw * Int32(8) + (pos - jw * wst)) * Int64(8),
                                        btag | Uint64(L.u32_of_f32(val)),
                                    )
                        tk, Tv = self._t_known(tk, Tv, i, ns, tbar, s_tv)
                        if tk != Int32(0):
                            ci, cnt1 = self._classify_tasks(
                                ci,
                                cutlass.min(i, ns - Int32(1)),
                                ord1,
                                t_lo,
                                n,
                                m_coord,
                                sLog,
                                sSub,
                                wsub,
                                cnt1,
                                Tv,
                            )
                prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                L.bar(3, 256)
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                if cutlass.const_expr(self.stamps):
                    if tidx == 0:
                        (mStamps.iterator + cta * NSTAMP + ST_STREAM_END).store(Int64(L.gtimer()))
                tmem.relinquish_alloc_permit()
                tmem.free(tmem_ptr)
                self._stream_publish(
                    cta,
                    tidx,
                    ws,
                    gen,
                    ns,
                    n,
                    t_lo,
                    cnt1,
                    wsub,
                    sSub,
                    sBtc,
                    s_wc,
                    s_scan,
                    sLog,
                    mRegion,
                    mHist,
                    mStamps,
                    mCPool,
                    arena,
                    mCtl,
                    Tv,
                    s_tv,
                )
            else:
                cute.arch.warpgroup_reg_dealloc(24)
                if warp_idx == 12:
                    btc_cols = cute.size(mBtC, mode=[1])
                    for e in cutlass.range_constexpr(2):
                        j = lane + Int32(32 * e)
                        if j < Int32(8) * ns:
                            sBtc[j] = mBtC[0, cutlass.min(Int32(8) * s0 + j, btc_cols - 1)]
                L.bar_arrive(5, 384)
                if warp_idx == 12:
                    L.bar(6, 288)
                    genh = self._gen_from(
                        Uint64(gen_slot[0]), Uint64(gen_slot[1]), Uint64(gen_slot[2])
                    )
                    if lane == 0:
                        tv, sv = self._poll_tline1(ws, genh)
                        s_tv[0] = Int32(tv)
                        s_tv[1] = Int32(sv)
                        prims.mbarrier_arrive(tbar)
                        self._stamp(mStamps, cta, ST_T_SEEN)
        else:
            if warp_idx == 8 or warp_idx == 9:
                cute.arch.warpgroup_reg_dealloc(24)
                grp = warp_idx - 8
                cpasync.prefetch_descriptor(tma_atom_a)
                cpasync.prefetch_descriptor(tma_atom_s)
                pol = None
                if cutlass.const_expr(self.policy != CP.POL_NONE):
                    pol = CP.make_l2_policy(self.policy)
                bt_cols = cute.size(mBlockTable, mode=[1])
                cpasync.prefetch_descriptor(tma_atom_b)
                cpasync.prefetch_descriptor(tma_atom_w)
                if cutlass.const_expr(self.dpf):
                    dnx_ = mDesc[cta * Int32(32) + lane]
                else:
                    dnx_ = Int32(0)
                for r_ in cutlass.range(self.nrows, unroll=1):
                    mStR_ = cute.make_tensor(mStamps.iterator + r_ * Int32(16), mStamps.layout)
                    if cutlass.const_expr(self.dpf):
                        desc_lane = dnx_
                        dnx_ = mDesc[
                            (
                                cutlass.min(r_ + Int32(1), Int32(self.nrows - 1))
                                * Int32(self.num_sms)
                                + cta
                            )
                            * Int32(32)
                            + lane
                        ]
                    else:
                        desc_lane = mDesc[(r_ * Int32(self.num_sms) + cta) * Int32(32) + lane]
                    s0 = cute.arch.shuffle_sync(desc_lane, 0)
                    ns = cute.arch.shuffle_sync(desc_lane, 1)
                    n_desc = cute.arch.shuffle_sync(desc_lane, 2)
                    ord0 = cute.arch.shuffle_sync(desc_lane, 3)
                    ord1 = cute.arch.shuffle_sync(desc_lane, 4)
                    nwin = cute.arch.shuffle_sync(desc_lane, 5)
                    n = n_desc
                    SMP, SS2, TGT, TGT2 = L.gvr_ladder(
                        n,
                        Int32(K_TOP),
                        Int32(PLAN["aim_base"]),
                        Int32(PLAN["sfac"]),
                        Int32(PLAN["amin"]),
                        Int32(PLAN["sd_en"]),
                        PLAN["r_const"],
                        PLAN["scap"],
                    )
                    wst = Int32(8) * SS2
                    t_lo = s0 * Int32(SPLIT_TOK)
                    nwin0 = (nwin + Int32(1)) // Int32(2)
                    nwin1 = nwin // Int32(2)
                    bt_r_ = _row2(mBlockTable, r_)
                    bta = bt_r_[0, cutlass.min(Int32(8) * s0 + lane, bt_cols - 1)]
                    btb = bt_r_[0, cutlass.min(Int32(8) * s0 + Int32(32) + lane, bt_cols - 1)]
                    ntile_row = (n + Int32(TILE - 1)) // Int32(TILE)
                    t_ref = Int64(0)
                    if cutlass.const_expr(self.wdelay > 0):
                        t_ref = Int64(L.gtimer())
                    if grp == 0:
                        if cutlass.const_expr(self.qdb):
                            if r_ >= Int32(2):
                                # Q / W double-buffered: only the row two back (same stage) must be done
                                while not prims.mbarrier_try_wait_parity(
                                    q_free + (r_ & Int32(1)),
                                    ((r_ >> Int32(1)) - Int32(1)) & Int32(1),
                                ):
                                    pass
                            if r_ > Int32(0):
                                if lane == 0:
                                    self._stamp(mStR_, cta, 11)
                        else:
                            if r_ > Int32(0):
                                # the previous row's MMAs completed and its weights were read
                                while not prims.mbarrier_try_wait_parity(
                                    q_free, (r_ - Int32(1)) & Int32(1)
                                ):
                                    pass
                                if lane == 0:
                                    self._stamp(mStR_, cta, 11)
                        q_bar = q_pipeline.producer_get_barrier(q_prod_state)
                        if prims.elect_sync():
                            prims.mbarrier_arrive_expect_tx(q_bar, self.num_q_tma_bytes)
                        cute.copy(
                            tma_atom_b,
                            tBgB[None, 0, r_],
                            tBsB[None, q_prod_state.index],
                            tma_bar_ptr=q_bar,
                        )
                        cute.copy(
                            tma_atom_w,
                            tWgW[None, r_],
                            tWsW[None, q_prod_state.index],
                            tma_bar_ptr=q_bar,
                        )
                        q_prod_state.advance()
                        for i in cutlass.range(ns, unroll=1):
                            if cutlass.const_expr(self.wdelay > 0):
                                if i == nwin0:
                                    tnow = Int64(L.gtimer())
                                    while tnow - t_ref < Int64(self.wdelay):
                                        tnow = Int64(L.gtimer())
                            tl = ord0 >> Int32(4) * i & Int32(15)
                            tile = Int32(2) * s0 + tl
                            tile_ok = tile < ntile_row
                            phys = [Int32(0)] * NUM_BLOCKS_PER_MMA
                            if i < 3:
                                for k in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                                    phys[k] = cute.arch.shuffle_sync(
                                        desc_lane, Int32(8 + 4 * 0 + k) + Int32(8) * i
                                    )
                            else:
                                for k in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                                    rel = Int32(4) * tl + Int32(k)
                                    pa = cute.arch.shuffle_sync(bta, rel & Int32(31))
                                    pb = cute.arch.shuffle_sync(btb, rel & Int32(31))
                                    pv = cutlass.select_(rel >= Int32(32), pb, pa)
                                    phys[k] = cutlass.select_(tile_ok, pv, Int32(0))
                            kv_pipeline_0.producer_acquire(kv_prod_state_0)
                            if cutlass.const_expr(self.stamps):
                                if i == 0:
                                    if lane == 0:
                                        (mStR_.iterator + cta * NSTAMP + ST_FIRST_TMA).store(
                                            Int64(L.gtimer())
                                        )
                            bar = kv_pipeline_0.producer_get_barrier(kv_prod_state_0)
                            stage = kv_prod_state_0.index
                            for k in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                                cute.copy(
                                    tma_atom_a,
                                    tAgA_0[None, 0, 0, phys[k]],
                                    tAsA_0[None, k, stage],
                                    tma_bar_ptr=bar,
                                    cache_policy=pol,
                                )
                                cute.copy(
                                    tma_atom_s,
                                    tSgS_0[None, 0, phys[k]],
                                    tSsS_0[None, k, stage],
                                    tma_bar_ptr=bar,
                                    cache_policy=pol,
                                )
                            kv_prod_state_0.advance()
                    else:
                        for i in cutlass.range(ns, unroll=1):
                            if cutlass.const_expr(self.wdelay > 0):
                                if i == nwin1:
                                    tnow = Int64(L.gtimer())
                                    while tnow - t_ref < Int64(self.wdelay):
                                        tnow = Int64(L.gtimer())
                            tl = ord1 >> Int32(4) * i & Int32(15)
                            tile = Int32(2) * s0 + tl
                            tile_ok = tile < ntile_row
                            phys = [Int32(0)] * NUM_BLOCKS_PER_MMA
                            if i < 3:
                                for k in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                                    phys[k] = cute.arch.shuffle_sync(
                                        desc_lane, Int32(8 + 4 * 1 + k) + Int32(8) * i
                                    )
                            else:
                                for k in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                                    rel = Int32(4) * tl + Int32(k)
                                    pa = cute.arch.shuffle_sync(bta, rel & Int32(31))
                                    pb = cute.arch.shuffle_sync(btb, rel & Int32(31))
                                    pv = cutlass.select_(rel >= Int32(32), pb, pa)
                                    phys[k] = cutlass.select_(tile_ok, pv, Int32(0))
                            kv_pipeline_1.producer_acquire(kv_prod_state_1)
                            bar = kv_pipeline_1.producer_get_barrier(kv_prod_state_1)
                            stage = kv_prod_state_1.index
                            for k in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                                cute.copy(
                                    tma_atom_a,
                                    tAgA_1[None, 0, 0, phys[k]],
                                    tAsA_1[None, k, stage],
                                    tma_bar_ptr=bar,
                                    cache_policy=pol,
                                )
                                cute.copy(
                                    tma_atom_s,
                                    tSgS_1[None, 0, phys[k]],
                                    tSsS_1[None, k, stage],
                                    tma_bar_ptr=bar,
                                    cache_policy=pol,
                                )
                            kv_prod_state_1.advance()
            elif warp_idx == 10 or warp_idx == 11:
                cute.arch.warpgroup_reg_dealloc(24)
                grp = warp_idx - 10
                L.bar(1, 320)
                tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
                tCtAcc_base_0 = cute.make_tensor(tmem_ptr, tCtAcc_fake_staged.layout)
                tCtAcc_base_1 = cute.make_tensor(
                    tmem_ptr + cols_per_group, tCtAcc_fake_staged.layout
                )
                num_k_blocks = cute.size(tCrA_0.shape[2])
                if cutlass.const_expr(self.dpf):
                    dnx_ = mDesc[cta * Int32(32) + lane]
                else:
                    dnx_ = Int32(0)
                for r_ in cutlass.range(self.nrows, unroll=1):
                    mStR_ = cute.make_tensor(mStamps.iterator + r_ * Int32(16), mStamps.layout)
                    if cutlass.const_expr(self.dpf):
                        desc_lane = dnx_
                        dnx_ = mDesc[
                            (
                                cutlass.min(r_ + Int32(1), Int32(self.nrows - 1))
                                * Int32(self.num_sms)
                                + cta
                            )
                            * Int32(32)
                            + lane
                        ]
                    else:
                        desc_lane = mDesc[(r_ * Int32(self.num_sms) + cta) * Int32(32) + lane]
                    s0 = cute.arch.shuffle_sync(desc_lane, 0)
                    ns = cute.arch.shuffle_sync(desc_lane, 1)
                    n_desc = cute.arch.shuffle_sync(desc_lane, 2)
                    ord0 = cute.arch.shuffle_sync(desc_lane, 3)
                    ord1 = cute.arch.shuffle_sync(desc_lane, 4)
                    nwin = cute.arch.shuffle_sync(desc_lane, 5)
                    n = n_desc
                    SMP, SS2, TGT, TGT2 = L.gvr_ladder(
                        n,
                        Int32(K_TOP),
                        Int32(PLAN["aim_base"]),
                        Int32(PLAN["sfac"]),
                        Int32(PLAN["amin"]),
                        Int32(PLAN["sd_en"]),
                        PLAN["r_const"],
                        PLAN["scap"],
                    )
                    wst = Int32(8) * SS2
                    t_lo = s0 * Int32(SPLIT_TOK)
                    nwin0 = (nwin + Int32(1)) // Int32(2)
                    nwin1 = nwin // Int32(2)
                    q_pipeline.consumer_wait(q_cons_state)
                    q_stage = q_cons_state.index
                    if grp == 0:
                        if lane == 0:
                            self._stamp(mStR_, cta, 10)
                        for i in cutlass.range(ns, unroll=1):
                            kv_pipeline_0.consumer_wait(kv_cons_state_umma_0)
                            umma_pipeline_0.producer_acquire(umma_prod_state_0)
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                            kv_stage = kv_cons_state_umma_0.index
                            tCtAcc_0 = tCtAcc_base_0[None, None, None, umma_prod_state_0.index]
                            for k_block in cutlass.range_constexpr(num_k_blocks):
                                cute.gemm(
                                    tiled_mma,
                                    tCtAcc_0,
                                    tCrA_0[None, None, k_block, kv_stage],
                                    tCrB[None, None, k_block, q_stage],
                                    tCtAcc_0,
                                )
                                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                            kv_cons_state_umma_0.advance()
                            umma_pipeline_0.producer_commit(umma_prod_state_0)
                            umma_prod_state_0.advance()
                    else:
                        for i in cutlass.range(ns, unroll=1):
                            kv_pipeline_1.consumer_wait(kv_cons_state_umma_1)
                            umma_pipeline_1.producer_acquire(umma_prod_state_1)
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                            kv_stage_1 = kv_cons_state_umma_1.index
                            tCtAcc_1 = tCtAcc_base_1[None, None, None, umma_prod_state_1.index]
                            for k_block in cutlass.range_constexpr(num_k_blocks):
                                cute.gemm(
                                    tiled_mma,
                                    tCtAcc_1,
                                    tCrA_1[None, None, k_block, kv_stage_1],
                                    tCrB[None, None, k_block, q_stage],
                                    tCtAcc_1,
                                )
                                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                            kv_cons_state_umma_1.advance()
                            umma_pipeline_1.producer_commit(umma_prod_state_1)
                            umma_prod_state_1.advance()
                    # this group's MMAs of the row (and their Q reads) complete -> one q_free arrival
                    if prims.elect_sync():
                        prims.tcgen05_commit(
                            (q_free + q_stage) if self.qdb else q_free, group=prims.CTAGroup.CTA_1
                        )
                    q_cons_state.advance()
            elif warp_idx < 8:
                cute.arch.warpgroup_reg_alloc(self.math_regs)
                wg = warp_idx // 4
                tmem.allocate(self.num_tmem_alloc_cols_total)
                L.bar(1, 320)
                tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
                tCtAcc_base_0 = cute.make_tensor(tmem_ptr, tCtAcc_fake_staged.layout)
                tCtAcc_base_1 = cute.make_tensor(
                    tmem_ptr + cols_per_group, tCtAcc_fake_staged.layout
                )
                local_tidx = tidx % 128
                cC = cute.make_identity_tensor(epi_sub_mn)
                rs_ = self.row_sizes()
                svg_ = cute.make_tensor(
                    cute.recast_ptr(arena + S_OFF_SVG, dtype=Int64), cute.make_layout(24)
                )
                svm_ = cute.recast_ptr(arena + S_OFF_SVM, dtype=Int64)
                rgen_ = cute.make_tensor(
                    cute.recast_ptr(arena + S_OFF_ROWST + 256, dtype=Int64), cute.make_layout(8)
                )
                if tidx == 0:
                    # every row's entry words at kernel start (the team writes a row's ws only after that row
                    # streamed)
                    for rr_ in cutlass.range_constexpr(self.nrows):
                        ge1_, ge2_, ge3_ = self._entry_loads(ws + Int64(rr_ * rs_["ws"] * 8))
                        svg_[rr_ * 3] = Int64(ge1_)
                        svg_[rr_ * 3 + 1] = Int64(ge2_)
                        svg_[rr_ * 3 + 2] = Int64(ge3_)
                # the math warps scan + publish the last row after their stream; warm that code now (memory idle,
                # first tile ~2 us away): same call site shape, scratch addresses in the debug buffer, T = -inf
                tbbm_ = cute.recast_ptr(arena + S_OFF_SVTB, dtype=Int64)
                s_tvm_ = cute.make_tensor(
                    cute.recast_ptr(arena + S_OFF_SVT, dtype=Int32), cute.make_layout(16)
                )
                sListM_ = cute.make_tensor(
                    cute.recast_ptr(
                        arena + (S_OFF_LSTM if self.nrows <= 6 else S_OFF_LSTM8), dtype=Int64
                    ),
                    cute.make_layout(MAXS * SPLIT_TOK),
                )
                cbasem_ = mCPool.iterator.toint()
                _dbgm = mDbg.iterator.toint()
                _sLog0 = cute.make_tensor(
                    cute.recast_ptr(arena + S_OFF_LOGR, dtype=Float32),
                    cute.make_layout(MAXS * SPLIT_TOK),
                )
                _sBtc0 = cute.make_tensor(
                    cute.recast_ptr(arena + S_OFF_SVB, dtype=Int32), cute.make_layout(MAXS * 8)
                )
                # (no warm-up pass: the last row's scan + publish code is fetched once, after the stream)
                if cutlass.const_expr(self.dpf):
                    dnx_ = mDesc[cta * Int32(32) + lane]
                    cnx_ = mCtx[0]
                else:
                    dnx_ = Int32(0)
                    cnx_ = Int32(0)
                for r_ in cutlass.range(self.nrows, unroll=1):
                    mStR_ = cute.make_tensor(mStamps.iterator + r_ * Int32(16), mStamps.layout)
                    sLogR_ = cute.make_tensor(
                        cute.recast_ptr(arena + S_OFF_LOGR, dtype=Float32)
                        + r_ * Int32(MAXS * SPLIT_TOK),
                        cute.make_layout(MAXS * SPLIT_TOK),
                    )
                    if cutlass.const_expr(self.dpf):
                        desc_lane = dnx_
                        dnx_ = mDesc[
                            (
                                cutlass.min(r_ + Int32(1), Int32(self.nrows - 1))
                                * Int32(self.num_sms)
                                + cta
                            )
                            * Int32(32)
                            + lane
                        ]
                        ctxr_ = cnx_
                        cnx_ = mCtx[cutlass.min(r_ + Int32(1), Int32(self.nrows - 1))]
                    else:
                        desc_lane = mDesc[(r_ * Int32(self.num_sms) + cta) * Int32(32) + lane]
                        ctxr_ = mCtx[r_]
                    s0 = cute.arch.shuffle_sync(desc_lane, 0)
                    ns = cute.arch.shuffle_sync(desc_lane, 1)
                    n_desc = cute.arch.shuffle_sync(desc_lane, 2)
                    ord0 = cute.arch.shuffle_sync(desc_lane, 3)
                    ord1 = cute.arch.shuffle_sync(desc_lane, 4)
                    nwin = cute.arch.shuffle_sync(desc_lane, 5)
                    n = n_desc
                    SMP, SS2, TGT, TGT2 = L.gvr_ladder(
                        n,
                        Int32(K_TOP),
                        Int32(PLAN["aim_base"]),
                        Int32(PLAN["sfac"]),
                        Int32(PLAN["amin"]),
                        Int32(PLAN["sd_en"]),
                        PLAN["r_const"],
                        PLAN["scap"],
                    )
                    wst = Int32(8) * SS2
                    t_lo = s0 * Int32(SPLIT_TOK)
                    nwin0 = (nwin + Int32(1)) // Int32(2)
                    nwin1 = nwin // Int32(2)
                    ws_r_ = ws + Int64(r_) * Int64(rs_["ws"] * 8)
                    q_pipeline.consumer_wait(q_cons_state)
                    q_stage_local = q_cons_state.index
                    w_cache = cute.make_rmem_tensor((self.num_w_in_reg,), self.epi_dtype)
                    for w_j in cutlass.range_constexpr(self.num_w_in_reg):
                        w_cache[w_j] = sW[w_j, q_stage_local]
                    q_cons_state.advance()
                    L.bar(3, 256)
                    if tidx == 0:
                        prims.mbarrier_arrive((q_free + q_stage_local) if self.qdb else q_free)
                    if tidx == 0:
                        self._stamp(mStR_, cta, 7)
                    gen = self._gen_from(
                        Uint64(svg_[r_ * Int32(3)]),
                        Uint64(svg_[r_ * Int32(3) + Int32(1)]),
                        Uint64(svg_[r_ * Int32(3) + Int32(2)]),
                    )
                    if tidx == 0:
                        rgen_[r_] = Int64(gen)
                    if tidx == 0:
                        if n_desc != ctxr_:
                            self._flag(ws_r_, gen, Int32(E_STALE))
                        if n_desc < Int32(0) or n_desc > Int32(N_MAX):
                            self._flag(ws_r_, gen, Int32(E_NLIMIT))
                    bboard = mBoard.iterator.toint() + Int64(r_) * Int64(rs_["board"] * 8)
                    btag = Uint64(self._tag32(gen)) << Uint64(32)
                    ci = Int32(0)
                    cnt1 = Int32(0)
                    tk = Int32(0)
                    Tv = Float32(float("inf"))
                    wsub = warp_idx * Int32(SUBCAP)
                    if wg == 0:
                        tAcc_0_ref = tCtAcc_base_0[None, None, None, 0][(None, None), 0, 0]
                        tAcc_0_ref_epi = cute.flat_divide(tAcc_0_ref, epi_sub_mn)
                        tiled_copy_ref_0 = tcgen05.make_tmem_copy(
                            copy_atom_t2r, tAcc_0_ref_epi[None, None, 0, 0]
                        )
                        thr_copy_ref_0 = tiled_copy_ref_0.get_slice(local_tidx)
                        tTR_cC = thr_copy_ref_0.partition_D(cC)
                        m_coord = tTR_cC[0][0]
                        tTR_rAcc = cute.make_fragment_like(tTR_cC, self.acc_dtype)
                        for i in cutlass.range(ns + Int32(1), unroll=1):
                            if i < ns:
                                tl = ord0 >> Int32(4) * i & Int32(15)
                                pos = (Int32(2) * s0 + tl) * Int32(TILE) + m_coord
                                kv_ready = kv_pipeline_0.consumer_try_wait(kv_cons_state_math_0)
                                umma_ready = umma_pipeline_0.consumer_try_wait(umma_cons_state_0)
                                kv_pipeline_0.consumer_wait(kv_cons_state_math_0, kv_ready)
                                scale_val = sScales_0[m_coord, kv_cons_state_math_0.index]
                                umma_pipeline_0.consumer_wait(umma_cons_state_0, umma_ready)
                                tCtAcc_c0 = tCtAcc_base_0[None, None, None, umma_cons_state_0.index]
                                tAcc_c0 = tCtAcc_c0[(None, None), 0, 0]
                                tTR_0 = thr_copy_ref_0.partition_S(
                                    cute.flat_divide(tAcc_c0, epi_sub_mn)
                                )
                                kv_pipeline_0.consumer_release(kv_cons_state_math_0)
                                kv_cons_state_math_0.advance()
                                cute.copy(tiled_copy_ref_0, tTR_0[None, None, None, 0, 0], tTR_rAcc)
                                cute.arch.fence_view_async_tmem_load()
                                umma_pipeline_0.consumer_release(umma_cons_state_0)
                                umma_cons_state_0.advance()
                                val = self._epi_value(
                                    tTR_rAcc, w_cache, sW, self.num_w_in_reg, scale_val
                                )
                                sLogR_[tl * Int32(TILE) + m_coord] = val
                                if cutlass.const_expr(self.debug):
                                    if pos < n:
                                        mDbg[pos] = val
                                jw = pos // wst
                                if pos - jw * wst < Int32(8):
                                    if jw < SMP:
                                        st_rlx_u64(
                                            bboard
                                            + Int64(jw * Int32(8) + (pos - jw * wst)) * Int64(8),
                                            btag | Uint64(L.u32_of_f32(val)),
                                        )
                                if cutlass.const_expr(self.stamps):
                                    if i == 0:
                                        if local_tidx == 0:
                                            (mStR_.iterator + cta * NSTAMP + ST_FIRST_EPI).store(
                                                Int64(L.gtimer())
                                            )
                                if cutlass.const_expr(self.stamps):
                                    if i + Int32(1) == nwin0:
                                        if local_tidx == 0:
                                            (mStR_.iterator + cta * NSTAMP + ST_BOARD).store(
                                                Int64(L.gtimer())
                                            )
                    else:
                        tAcc_1_ref = tCtAcc_base_1[None, None, None, 0][(None, None), 0, 0]
                        tAcc_1_ref_epi = cute.flat_divide(tAcc_1_ref, epi_sub_mn)
                        tiled_copy_ref_1 = tcgen05.make_tmem_copy(
                            copy_atom_t2r, tAcc_1_ref_epi[None, None, 0, 0]
                        )
                        thr_copy_ref_1 = tiled_copy_ref_1.get_slice(local_tidx)
                        tTR_cC = thr_copy_ref_1.partition_D(cC)
                        m_coord = tTR_cC[0][0]
                        tTR_rAcc = cute.make_fragment_like(tTR_cC, self.acc_dtype)
                        for i in cutlass.range(ns + Int32(1), unroll=1):
                            if i < ns:
                                tl = ord1 >> Int32(4) * i & Int32(15)
                                pos = (Int32(2) * s0 + tl) * Int32(TILE) + m_coord
                                kv_ready = kv_pipeline_1.consumer_try_wait(kv_cons_state_math_1)
                                umma_ready = umma_pipeline_1.consumer_try_wait(umma_cons_state_1)
                                kv_pipeline_1.consumer_wait(kv_cons_state_math_1, kv_ready)
                                scale_val = sScales_1[m_coord, kv_cons_state_math_1.index]
                                umma_pipeline_1.consumer_wait(umma_cons_state_1, umma_ready)
                                tCtAcc_c1 = tCtAcc_base_1[None, None, None, umma_cons_state_1.index]
                                tAcc_c1 = tCtAcc_c1[(None, None), 0, 0]
                                tTR_1 = thr_copy_ref_1.partition_S(
                                    cute.flat_divide(tAcc_c1, epi_sub_mn)
                                )
                                kv_pipeline_1.consumer_release(kv_cons_state_math_1)
                                kv_cons_state_math_1.advance()
                                cute.copy(tiled_copy_ref_1, tTR_1[None, None, None, 0, 0], tTR_rAcc)
                                cute.arch.fence_view_async_tmem_load()
                                umma_pipeline_1.consumer_release(umma_cons_state_1)
                                umma_cons_state_1.advance()
                                val = self._epi_value(
                                    tTR_rAcc, w_cache, sW, self.num_w_in_reg, scale_val
                                )
                                sLogR_[tl * Int32(TILE) + m_coord] = val
                                if cutlass.const_expr(self.debug):
                                    if pos < n:
                                        mDbg[pos] = val
                                jw = pos // wst
                                if pos - jw * wst < Int32(8):
                                    if jw < SMP:
                                        st_rlx_u64(
                                            bboard
                                            + Int64(jw * Int32(8) + (pos - jw * wst)) * Int64(8),
                                            btag | Uint64(L.u32_of_f32(val)),
                                        )
                    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                    L.bar(3, 256)
                    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                    if cutlass.const_expr(self.stamps):
                        if tidx == 0:
                            (mStR_.iterator + cta * NSTAMP + ST_STREAM_END).store(Int64(L.gtimer()))
                    if cutlass.const_expr(self.nteams < self.nrows):
                        srv_ = Int32(0)
                        if r_ == Int32(self.nrows - 1):
                            # T server of the last row (CTA 0, warps 0-3; scratch in the idle K stage buffer)
                            if cutlass.const_expr(self.tsrv1 and self.nrows <= 4):
                                # the first stream CTA to finish this stream serves the T line instead
                                fadr_ = Int32(
                                    cute.recast_ptr(
                                        arena + S_OFF_MISC + 62 * 4, dtype=Int32
                                    ).toint()
                                )
                                if tidx == Int32(0):
                                    w_ = Int32(0)
                                    if B.mbar_test_wait(
                                        Int32((tbbm_ + r_).toint()), Int32(0)
                                    ) == Int32(0):
                                        tk_ = gen + Uint64(5)
                                        if atom_exch_u64(self._w(ws_r_, W_TSRV), tk_) != tk_:
                                            w_ = Int32(1)
                                    sts_volatile_u32(fadr_, w_)
                                L.bar(3, 256)
                                srv_ = lds_volatile_u32(fadr_)
                            else:
                                # nrows > 4 with tsz2 / tsz: CTA ns - 1 (empty in the last row) serves it
                                if cta == Int32(
                                    (self.ns - 1)
                                    if (self.nrows > 4 and (self.tsz or self.tsz2))
                                    else 0
                                ):
                                    srv_ = Int32(1)
                        if cutlass.const_expr(self.tsz2):
                            # CTA ns - 1 (empty range in the last two rows) also serves row nrows - 2
                            if r_ == Int32(self.nrows - 2):
                                if cta == Int32(self.ns - 1):
                                    srv_ = Int32(1)
                        if cutlass.const_expr(self.tsz01):
                            # CTA ns - 2 (empty ranges in rows 0-1) serves rows 0 and 1
                            if r_ < Int32(2):
                                if cta == Int32(self.ns - 2):
                                    srv_ = Int32(1)
                        if cutlass.const_expr(self.tszall):
                            # rows 2 .. nrows - 3 in pairs: CTA ns - 2 - r // 2 (empty in its pair) serves them
                            if r_ >= Int32(2) and r_ < Int32(self.nrows - 2):
                                if cta == Int32(self.ns - 2) - r_ // Int32(2):
                                    srv_ = Int32(1)
                        if srv_ != Int32(0):
                            if tidx < Int32(128):
                                mStT_ = cute.make_tensor(
                                    mStamps.iterator
                                    + cutlass.select_(
                                        r_ == Int32(self.nrows - 1),
                                        Int32(3),
                                        cutlass.select_(r_ == Int32(1), Int32(1), Int32(2)),
                                    )
                                    * Int32(self.num_sms * NSTAMP),
                                    mStamps.layout,
                                )
                                self._block_a_lead(
                                    arena,
                                    cta,
                                    tidx,
                                    ws_r_,
                                    n,
                                    _flat(mBoard, r_, rs_["board"]),
                                    mStT_,
                                    Uint64(svg_[r_ * Int32(3)]),
                                    Uint64(svg_[r_ * Int32(3) + Int32(1)]),
                                    Uint64(svg_[r_ * Int32(3) + Int32(2)]),
                                    _flat(mCtl, r_, rs_["ctl"]),
                                    Int32(0),
                                    _flat(mHist, r_, rs_["hist"]),
                                    fa=(S_TSRV_FA if self.nrows <= 4 else S_TSRV_FA8)
                                    if self.tsz01
                                    else S_OFF_KV0,
                                    mi=((S_TSRV_FA if self.nrows <= 4 else S_TSRV_FA8) + 2048)
                                    if self.tsz01
                                    else S_OFF_KV0 + 2048,
                                )
                    if r_ == Int32(self.nrows - 1):
                        # the last row: wait its T (armed by the T poller), scan + publish with all 256 math threads
                        B.spin(Int32((tbbm_ + r_).toint()), Int32(0))
                        Tvl_ = L.f32_of_u32(Uint32(s_tvm_[r_ * Int32(2)]))
                        bsl_ = L.f32_of_u32(Uint32(s_tvm_[r_ * Int32(2) + Int32(1)]))
                        ntokl_ = cutlass.min(ns * Int32(SPLIT_TOK), n - t_lo)
                        if ntokl_ < Int32(0):
                            ntokl_ = Int32(0)
                        rgl_ = mRegion.iterator.toint() + Int64(r_) * Int64(rs_["region"] * 8)
                        sBtcL_ = cute.make_tensor(
                            cute.recast_ptr(arena + S_OFF_SVB, dtype=Int32) + r_ * Int32(MAXS * 8),
                            cute.make_layout(MAXS * 8),
                        )
                        totl_ = self._svc_scanpub(
                            cta,
                            tidx,
                            Int32(1),
                            self._cnt_addr(ws_r_, Int32(0), cta),
                            rgl_ + Int64(cta) * Int64(2 * CAPR * 8),
                            mHist.iterator.toint() + Int64(r_) * Int64(rs_["hist"] * 4),
                            rgl_,
                            gen,
                            ntokl_,
                            t_lo,
                            sLogR_,
                            Tvl_,
                            bsl_,
                            sBtcL_,
                            cbasem_,
                            s_wc,
                            sListM_,
                            mStR_,
                            nth=256,
                            bid=3,
                        )
                        if tidx == 0:
                            rstl_ = cute.make_tensor(
                                cute.recast_ptr(arena + S_OFF_ROWST, dtype=Int32) + r_ * Int32(8),
                                cute.make_layout(8),
                            )
                            rstl_[0] = totl_
                            rstl_[1] = ntokl_
                            rstl_[2] = t_lo
                            rstl_[3] = Int32(0)
                            rstl_[4] = Int32(0)
                        L.bar(3, 256)
                    # hand the row to the publishers (rows < last) / signal the drain (last row)
                    if tidx == 0:
                        prims.mbarrier_arrive(svm_ + r_)
                tmem.relinquish_alloc_permit()
                tmem.free(tmem_ptr)
            else:
                cute.arch.warpgroup_reg_dealloc(56)
                rs_ = self.row_sizes()
                stid = tidx - Int32(384)
                s_wc4 = cute.make_tensor(
                    cute.recast_ptr(arena + S_OFF_SVW, dtype=Int32), cute.make_layout(8)
                )
                s_dir4 = cute.make_tensor(
                    cute.recast_ptr(arena + S_OFF_SVW + 32, dtype=Int32), cute.make_layout(8)
                )
                s_tvr = cute.make_tensor(
                    cute.recast_ptr(arena + S_OFF_SVT, dtype=Int32), cute.make_layout(16)
                )
                svm_ = cute.recast_ptr(arena + S_OFF_SVM, dtype=Int64)
                tbb_ = cute.recast_ptr(arena + S_OFF_SVTB, dtype=Int64)
                rgen_ = cute.make_tensor(
                    cute.recast_ptr(arena + S_OFF_ROWST + 256, dtype=Int64), cute.make_layout(8)
                )
                cbase = mCPool.iterator.toint()
                svgs_ = cute.make_tensor(
                    cute.recast_ptr(arena + S_OFF_SVG + 8 * 3 * 8, dtype=Int64), cute.make_layout(8)
                )
                if warp_idx == 12:
                    # T poller: every row's C-pool block-table slice and T in row order; each T arms the row's
                    # mbarrier (publishers / math warps). Never waits for a publish.
                    if lane == Int32(0):
                        for rr_ in cutlass.range_constexpr(self.nrows):
                            ge1_, ge2_, ge3_ = self._entry_loads(ws + Int64(rr_ * rs_["ws"] * 8))
                            svgs_[rr_] = Int64(self._gen_from(ge1_, ge2_, ge3_))
                    for r_ in cutlass.range(self.nrows, unroll=1):
                        mStR_ = cute.make_tensor(mStamps.iterator + r_ * Int32(16), mStamps.layout)
                        desc_lane = mDesc[(r_ * Int32(self.num_sms) + cta) * Int32(32) + lane]
                        s0 = cute.arch.shuffle_sync(desc_lane, 0)
                        ns = cute.arch.shuffle_sync(desc_lane, 1)
                        ws_r_ = ws + Int64(r_) * Int64(rs_["ws"] * 8)
                        sBtcR_ = cute.make_tensor(
                            cute.recast_ptr(arena + S_OFF_SVB, dtype=Int32) + r_ * Int32(MAXS * 8),
                            cute.make_layout(MAXS * 8),
                        )
                        btc_r_ = _row2(mBtC, r_)
                        btc_cols = cute.size(mBtC, mode=[1])
                        for e in cutlass.range_constexpr(2):
                            j = lane + Int32(32 * e)
                            if j < Int32(8) * ns:
                                sBtcR_[j] = btc_r_[0, cutlass.min(Int32(8) * s0 + j, btc_cols - 1)]
                        cute.arch.sync_warp()
                        if lane == Int32(0):
                            tv, sv = self._poll_tline1(ws_r_, Uint64(svgs_[r_]))
                            s_tvr[r_ * Int32(2)] = Int32(tv)
                            s_tvr[r_ * Int32(2) + Int32(1)] = Int32(sv)
                            prims.mbarrier_arrive(tbb_ + r_)
                            self._stamp(mStR_, cta, ST_T_SEEN)
                        cute.arch.sync_warp()
                else:
                    # publishers (warps 13-15, 96 threads, named barrier 4): warm-up pass (it_ == 0), then rows
                    # 0 .. nrows - 2 (the math handoff and T_r)
                    pid = tidx - Int32(416)
                    dbgb_ = mDbg.iterator.toint()
                    sListP_ = cute.make_tensor(
                        cute.recast_ptr(arena + S_OFF_SUB, dtype=Int64),
                        cute.make_layout(8 * SUBCAP),
                    )
                    for it_ in cutlass.range(1, self.nrows, unroll=1):
                        live = cutlass.select_(it_ > Int32(0), Int32(1), Int32(0))
                        r_ = cutlass.max(it_ - Int32(1), Int32(0))
                        mStR_ = cute.make_tensor(mStamps.iterator + r_ * Int32(16), mStamps.layout)
                        desc_lane = mDesc[(r_ * Int32(self.num_sms) + cta) * Int32(32) + lane]
                        s0 = cute.arch.shuffle_sync(desc_lane, 0)
                        ns = cute.arch.shuffle_sync(desc_lane, 1)
                        n = cute.arch.shuffle_sync(desc_lane, 2)
                        t_lo = s0 * Int32(SPLIT_TOK)
                        ws_r_ = ws + Int64(r_) * Int64(rs_["ws"] * 8)
                        sBtcR_ = cute.make_tensor(
                            cute.recast_ptr(arena + S_OFF_SVB, dtype=Int32) + r_ * Int32(MAXS * 8),
                            cute.make_layout(MAXS * 8),
                        )
                        if live != Int32(0):
                            while not prims.mbarrier_try_wait_parity(svm_ + r_, Int32(0)):
                                pass
                            while not prims.mbarrier_try_wait_parity(tbb_ + r_, Int32(0)):
                                pass
                            if pid == Int32(0):
                                self._stamp(mStR_, cta, 13)
                        else:
                            if pid == Int32(0):
                                s_tvr[14] = Int32(-8388608)
                                s_tvr[15] = Int32(0)
                        L.bar(4, 96)
                        ti_ = cutlass.select_(live != Int32(0), r_ * Int32(2), Int32(14))
                        Tv = L.f32_of_u32(Uint32(s_tvr[ti_]))
                        bscale = L.f32_of_u32(Uint32(s_tvr[ti_ + Int32(1)]))
                        gen = Uint64(rgen_[r_])
                        sLogR_ = cute.make_tensor(
                            cute.recast_ptr(arena + S_OFF_LOGR, dtype=Float32)
                            + r_ * Int32(MAXS * SPLIT_TOK),
                            cute.make_layout(MAXS * SPLIT_TOK),
                        )
                        ntok_ = cutlass.min(ns * Int32(SPLIT_TOK), n - t_lo)
                        if ntok_ < Int32(0):
                            ntok_ = Int32(0)
                        rg_ = mRegion.iterator.toint() + Int64(r_) * Int64(rs_["region"] * 8)
                        cnta_ = self._cnt_addr(ws_r_, Int32(0), cta)
                        rbase_ = rg_ + Int64(cta) * Int64(2 * CAPR * 8)
                        ghist_ = mHist.iterator.toint() + Int64(r_) * Int64(rs_["hist"] * 4)
                        if live == Int32(0):
                            ntok_ = Int32(128)
                            cnta_ = dbgb_ + Int64(65536)
                            rbase_ = dbgb_
                            rg_ = dbgb_
                            ghist_ = dbgb_ + Int64(131072)
                        tot_ = self._svc_scanpub(
                            cta,
                            pid,
                            live,
                            cnta_,
                            rbase_,
                            ghist_,
                            rg_,
                            gen,
                            ntok_,
                            t_lo,
                            sLogR_,
                            Tv,
                            bscale,
                            sBtcR_,
                            cbase,
                            s_wc4,
                            sListP_,
                            mStR_,
                            nth=96,
                            bid=4,
                            pf=self.ppref,
                        )
                        if live != Int32(0):
                            if pid == Int32(0):
                                rst_ = cute.make_tensor(
                                    cute.recast_ptr(arena + S_OFF_ROWST, dtype=Int32)
                                    + r_ * Int32(8),
                                    cute.make_layout(8),
                                )
                                rst_[0] = tot_
                                rst_[1] = ntok_
                                rst_[2] = t_lo
                                rst_[3] = Int32(0)
                                rst_[4] = Int32(0)
                        L.bar(4, 96)
                        if live != Int32(0):
                            if pid == Int32(0):
                                self._stamp(mStR_, cta, 12)
                # final drain (all four warps, named barrier 2) once the math warps saved the last row's state
                while not prims.mbarrier_try_wait_parity(svm_ + Int32(self.nrows - 1), Int32(0)):
                    pass
                L.bar(2, 128)
                if stid == Int32(0):
                    self._stamp(mStamps, cta, 9)
                for q_ in cutlass.range(self.nrows, unroll=1):
                    self._serve_row(
                        q_,
                        cta,
                        stid,
                        ws,
                        arena,
                        s_wc4,
                        s_dir4,
                        mRegion,
                        mHist,
                        True,
                        nth=128,
                        bid=2,
                        hoff=S_OFF_SVH,
                        ebid=2,
                    )
                if stid == Int32(0):
                    self._stamp(mStamps, cta, ST_EXIT)

    @cute.jit
    def _entry_loads(self, ws):
        g1_ = L.ld_relaxed_u64(self._w(ws, W_DONE))
        g2_ = Uint64(0)
        if cutlass.const_expr(self.gen2):
            g2_ = L.ld_relaxed_u64(self._w(ws, W_DONE2))
        hd = Uint64(0)
        if cutlass.const_expr(self.heal):
            hd = L.ld_relaxed_u64(self._w(ws, W_DIR))
        return (g1_, g2_, hd)

    @cute.jit
    def _gen_from(self, g1_, g2_, hd):
        gen = g1_
        if g2_ > g1_:
            gen = g2_
        tdir = Int64(hd >> Uint64(24))
        if tdir != Int64(0):
            tcur = Int64(self._tag40(gen))
            ahead = tdir - tcur
            if ahead < Int64(0):
                ahead = ahead + Int64(1099511627775)
            if ahead < Int64(1 << 39):
                gen = gen + Uint64(ahead) + Uint64(1)
        return gen

    @cute.jit
    def _tag32(self, gen):
        return Uint32(gen % Uint64(4294967295) + Uint64(1))

    @cute.jit
    def _tag40(self, gen):
        return gen % Uint64(1099511627775) + Uint64(1)

    @cute.jit
    def _flag(self, ws, gen, bits):
        red_max_u64(self._w(ws, W_ERR), gen + Uint64(1) << Uint64(8) | Uint64(Uint32(bits)))
        red_or_u64(self._w(ws, W_ERRB), Uint64(Uint32(bits)))

    @cute.jit
    def _raise_abort(self, ws, gen, bits):
        self._flag(ws, gen, bits | Int32(E_ABORTED))
        st_rlx_u64(self._w(ws, W_ABORT), gen + Uint64(1))

    @cute.jit
    def _deadline(self):
        return Int64(L.gtimer()) + Int64(self.spin_ns)

    @cute.jit
    def _poll_word_eq(self, ws, gen, addr, want):
        t64 = gen + Uint64(1)
        tdl = self._deadline()
        st_ = Int32(0)
        while st_ == Int32(0):
            v = L.ld_relaxed_u64(addr)
            av = L.ld_relaxed_u64(self._w(ws, W_ABORT))
            if v == want:
                st_ = Int32(1)
            elif av == t64:
                st_ = Int32(2)
            elif Int64(L.gtimer()) > tdl:
                st_ = Int32(3)
        if st_ == Int32(3):
            self._raise_abort(ws, gen, Int32(E_TIMEOUT))
        ok = Int32(0)
        if st_ == Int32(1):
            ok = Int32(1)
        return ok

    @cute.jit
    def _followers_ready(self, ws, t64):
        ready = Int32(1)
        for fi in cutlass.range_constexpr(1, NTEAM):
            f = L.ld_relaxed_u64(self._w(ws, W_FOL + fi))
            if f != t64:
                ready = Int32(0)
        return ready

    @cute.jit
    def _wait_followers(self, ws, gen):
        t64 = gen + Uint64(1)
        tdl = self._deadline()
        st_ = Int32(0)
        while st_ == Int32(0):
            ready = self._followers_ready(ws, t64)
            av = L.ld_relaxed_u64(self._w(ws, W_ABORT))
            if ready != Int32(0):
                st_ = Int32(1)
            elif av == t64:
                st_ = Int32(2)
            elif Int64(L.gtimer()) > tdl:
                st_ = Int32(3)
        if st_ == Int32(3):
            self._raise_abort(ws, gen, Int32(E_TIMEOUT))
        ok = Int32(0)
        if st_ == Int32(1):
            ok = Int32(1)
        return ok

    @cute.jit
    def _publish_dir(self, ws, gen, seq, kind, path, p0, p1, p2):
        if kind != Int32(K_DONE):
            L.st_g_u64(self._w(ws, W_DP), p0)
            L.st_g_u64(self._w(ws, W_DP + 1), p1)
            L.st_g_u64(self._w(ws, W_DP + 2), p2)
            L.fence_acq_rel_gpu()
        self._skew(gen, Int32(1000) + seq, 6)
        st_rlx_u64(
            self._w(ws, W_DIR),
            self._tag40(gen) << Uint64(24)
            | Uint64(Uint32(seq)) << Uint64(8)
            | Uint64(Uint32(path)) << Uint64(4)
            | Uint64(Uint32(kind)),
        )

    @cute.jit
    def _poll_tline1(self, ws, gen):
        tg = self._tag32(gen)
        t64 = gen + Uint64(1)
        tdl = self._deadline()
        st_ = Int32(0)
        tv = Uint32(2139095040)
        it_ = Int32(0)
        hv0 = Uint64(0)
        mv0 = Uint64(0)
        while st_ == Int32(0):
            wt = L.ld_acquire_u64(self._w(ws, W_TLINE))
            if cutlass.const_expr(self.tl1r):
                # HIC / SMAX in the same round trip (relaxed, after the acquire of T in program order)
                hv0 = L.ld_relaxed_u64(self._w(ws, W_TLINE + 1))
                mv0 = L.ld_relaxed_u64(self._w(ws, W_TLINE + 3))
            av = Uint64(0)
            chk = Int32(1)
            if cutlass.const_expr(self.pchk > 1):
                chk = Int32(0)
                if it_ & Int32(self.pchk - 1) == Int32(self.pchk - 1):
                    chk = Int32(1)
            if chk != Int32(0):
                av = L.ld_relaxed_u64(self._w(ws, W_ABORT))
            it_ = it_ + Int32(1)
            if Uint32(wt >> Uint64(32)) == tg:
                st_ = Int32(1)
                tv = Uint32(wt & Uint64(4294967295))
            elif chk != Int32(0):
                if av == t64:
                    st_ = Int32(2)
                elif Int64(L.gtimer()) > tdl:
                    st_ = Int32(3)
        if st_ == Int32(3):
            self._raise_abort(ws, gen, Int32(E_TIMEOUT))
        elif st_ == Int32(2):
            self._flag(ws, gen, Int32(E_ABORTED))
        sv = Uint32(0)
        if st_ == Int32(1):
            # T is stored after a release fence that follows the per-call histogram clear, so this acquire also
            # orders all producer atomics after the clear.
            hv = hv0
            mv = mv0
            if cutlass.const_expr(not self.tl1r):
                hv = L.ld_relaxed_u64(self._w(ws, W_TLINE + 1))
                mv = L.ld_relaxed_u64(self._w(ws, W_TLINE + 3))
            # HIC / SMAX are tagged words of the same T line, stored just before T: wait for this launch's tags
            while Uint32(hv >> Uint64(32)) != tg or Uint32(mv >> Uint64(32)) != tg:
                hv = L.ld_relaxed_u64(self._w(ws, W_TLINE + 1))
                mv = L.ld_relaxed_u64(self._w(ws, W_TLINE + 3))
            tf = L.f32_of_u32(tv)
            hi = L.f32_of_u32(Uint32(hv))
            mx = L.f32_of_u32(Uint32(mv))
            lo, hi = self._band_range(tf, hi, mx)
            if tf > Float32(L.SENT_LO):
                sv = L.u32_of_f32(S.bin_scale(lo, hi) * Float32(0.5))
        return (tv, sv)

    @cute.jit
    def _t_known(self, tk, Tv, i, ns, tbar, s_tv, tpar=0):
        if tk == Int32(0):
            t_ = B.mbar_test_wait(Int32(tbar.toint()), Int32(tpar))
            if i == ns:
                while t_ == Int32(0):
                    t_ = B.mbar_try_wait(Int32(tbar.toint()), Int32(tpar))
            if cute.arch.vote_all_sync(t_ != Int32(0)):
                tk = Int32(1)
                Tv = L.f32_of_u32(Uint32(s_tv[0]))
        return (tk, Tv)

    @cute.jit
    def _classify_tasks(self, ci, last, ordw, t_lo, n, m, sLog, sSub, wsub, cnt1, Tv):
        lm = Int32(cute.arch.lanemask_lt())
        kT = L.fkey(Tv)
        while ci <= last:
            tl = ordw >> Int32(4) * ci & Int32(15)
            p = tl * Int32(TILE) + m
            x = sLog[p]
            gpos = t_lo + p
            ok = Int32(0)
            if gpos < n:
                if L.fkey(x) >= kT:
                    ok = Int32(1)
            bm = Int32(cute.arch.vote_ballot_sync(ok != Int32(0)))
            if ok != Int32(0):
                sSub[wsub + cnt1 + Int32(cute.arch.popc(bm & lm))] = Int64(
                    Uint64(Uint32(gpos)) << Uint64(32) | Uint64(L.u32_of_f32(x))
                )
            cnt1 = cnt1 + Int32(cute.arch.popc(bm))
            ci = ci + Int32(1)
        return (ci, cnt1)

    @cute.jit
    def _cnt_addr(self, ws, slot, c):
        return self._w(ws, W_CNT) + Int64(slot * Int32(160) + c) * Int64(8)

    @cute.jit
    def _stream_publish(
        self,
        cta,
        tidx,
        ws,
        gen,
        ns,
        n,
        t_lo,
        cnt1,
        wsub,
        sSub,
        sBtc,
        s_wc,
        s_dir,
        sLog,
        mRegion,
        mHist,
        mStamps,
        mCPool,
        arena,
        mCtl,
        Tv,
        s_tv,
        serve: cutlass.Constexpr[bool] = True,
    ):
        lane = tidx % 32
        wid = tidx // 32
        if lane == 0:
            s_wc[wid] = cnt1
        L.bar(5, 384)
        pre = Int32(0)
        tot = Int32(0)
        for w_ in cutlass.range_constexpr(8):
            v_ = s_wc[w_]
            if Int32(w_) < wid:
                pre = pre + v_
            tot = tot + v_
        t40 = self._tag40(gen)
        rbase = mRegion.iterator.toint() + Int64(cta) * Int64(2 * CAPR * 8)
        self._skew(gen, cta * Int32(8) + wid, 7)
        if tidx == 0:
            self._stamp(mStamps, cta, 40)
        bscale = L.f32_of_u32(Uint32(s_tv[1]))
        ghist = mHist.iterator.toint()
        bucket_base = mRegion.iterator.toint()
        k = lane
        while k < cnt1:
            e = Uint64(sSub[wsub + k])
            L.st_g_u64(rbase + Int64(pre + k) * Int64(8), e)
            if bscale > Float32(0.0):
                bq = L._vbin(e, Tv, bscale, NB_INITIAL)
                ordinal = atom_add_u32(ghist + Int64(bq) * Int64(4), Uint32(1))
                if ordinal < Uint32(BUCKET_CAP):
                    L.st_g_u64(self._bucket_addr(bucket_base, bq, Int32(ordinal)), e)
            k = k + Int32(32)
        if cutlass.const_expr(self.bulk):
            if tidx == 255:
                if tot & Int32(1) != Int32(0):
                    L.st_g_u64(
                        rbase + Int64(tot) * Int64(8),
                        Uint64(DUMMY_GPOS) << Uint64(32) | Uint64(4286578688),
                    )
            B.fence_proxy_async_global()
        L.bar(5, 256)
        if tidx == 0:
            self._stamp(mStamps, cta, 41)
            self._skew(gen, cta, 3)
            skipc = Int32(0)
            if cutlass.const_expr(self.debug):
                if mCtl[1] == Int32(2) and cta == Int32(77):
                    skipc = Int32(1)
            if cutlass.const_expr(self.relst):
                L.st_release_u64(
                    self._cnt_addr(ws, Int32(0), cta), t40 << Uint64(24) | Uint64(Uint32(tot))
                )
                self._stamp(mStamps, cta, 42)
            else:
                if cutlass.const_expr(not self.nofence):
                    L.fence_acq_rel_gpu()
                self._stamp(mStamps, cta, 42)
                if skipc == Int32(0):
                    st_rlx_u64(
                        self._cnt_addr(ws, Int32(0), cta), t40 << Uint64(24) | Uint64(Uint32(tot))
                    )
            self._stamp(mStamps, cta, ST_CAND_PUB)
        cbase = mCPool.iterator.toint()
        if cutlass.const_expr(self.pref):
            k = lane
            while k < cnt1:
                gpos = Int32(Uint32(Uint64(sSub[wsub + k]) >> Uint64(32)))
                kvr = sBtc[gpos - t_lo >> Int32(5)] * Int32(32) + (gpos & Int32(31))
                prefetch_l2_bulk(cbase + Int64(kvr) * Int64(512), Int32(512))
                k = k + Int32(32)
        if cutlass.const_expr(serve):
            ntok = cutlass.min(ns * Int32(SPLIT_TOK), n - t_lo)
            if ntok < Int32(0):
                ntok = Int32(0)
            seq = Int32(0)
            fin = Int32(0)
            while fin == Int32(0):
                if tidx == 0:
                    self._poll_dir(ws, gen, seq, s_dir)
                L.bar(5, 256)
                kind = s_dir[0]
                dseq = s_dir[1]
                if kind == Int32(K_PUB):
                    slot = s_dir[2] & Int32(255)
                    klo = Uint64(Uint32(s_dir[4])) << Uint64(32) | Uint64(Uint32(s_dir[3]))
                    khi = Uint64(Uint32(s_dir[6])) << Uint64(32) | Uint64(Uint32(s_dir[5]))
                    base = tot
                    if cutlass.const_expr(self.bulk):
                        base = tot + (tot & Int32(1))
                    if slot == Int32(2):
                        base = Int32(CAPR)
                    self._stream_pub(
                        cta, tidx, ws, gen, slot, base, klo, khi, ntok, t_lo, sLog, s_wc, rbase
                    )
                elif kind == Int32(K_HIST):
                    plen = s_dir[2] >> Int32(8) & Int32(255)
                    pref = Uint64(Uint32(s_dir[4])) << Uint64(32) | Uint64(Uint32(s_dir[3]))
                    self._stream_hist(
                        cta, tidx, ws, gen, dseq, pref, plen, ntok, t_lo, sLog, mHist, arena
                    )
                else:
                    fin = Int32(1)
                seq = dseq
                L.bar(5, 256)
            if tidx == 0:
                self._stamp(mStamps, cta, ST_EXIT)
        return tot

    @cute.jit
    def _svc_publish(
        self,
        cta,
        stid,
        live,
        cnt_addr,
        rbase,
        ghist,
        bucket_base,
        gen,
        ntok,
        t_lo,
        sLog,
        Tv,
        bscale,
        sBtcR,
        cbase,
        s_wc,
        sList,
        mStamps,
    ):
        # one row's first-round publish by the service warpgroup (128 threads, named barrier 4). One scan of the
        # row's logits (positions < n via ntok): each warp compacts its candidates (logit >= T) into its own SMEM list;
        # then the region packet, the DRS buckets, the even padding, the released count, the L2 prefetch of their KV
        # rows. live == 0: the kernel-start warm-up pass (same code, scratch addresses, no stamps / prefetch).
        lane = stid % Int32(32)
        wid = stid // Int32(32)
        lm = Int32(cute.arch.lanemask_lt())
        kT = L.fkey(Tv)
        lw = wid * Int32(MAXS * SPLIT_TOK // 4)
        cnt = Int32(0)
        sb_ = Int32(sLog.iterator.toint())
        vals_ = []
        for j_ in cutlass.range_constexpr(4):
            vals_.append(
                lds_v4f32(
                    sb_
                    + (wid * Int32(MAXS * SPLIT_TOK // 4) + Int32(128 * j_) + lane * Int32(4))
                    * Int32(4)
                )
            )
        for j_ in cutlass.range_constexpr(4):
            pb = wid * Int32(MAXS * SPLIT_TOK // 4) + Int32(128 * j_) + lane * Int32(4)
            oks_ = []
            anyv = Int32(0)
            for e_ in cutlass.range_constexpr(4):
                ok = Int32(0)
                if pb + Int32(e_) < ntok:
                    if L.fkey(vals_[j_][e_]) >= kT:
                        ok = Int32(1)
                oks_.append(ok)
                anyv = anyv | ok
            if cute.arch.vote_any_sync(anyv != Int32(0)):
                for e_ in cutlass.range_constexpr(4):
                    bm = Int32(cute.arch.vote_ballot_sync(oks_[e_] != Int32(0)))
                    if oks_[e_] != Int32(0):
                        sList[lw + cnt + Int32(cute.arch.popc(bm & lm))] = Int64(
                            Uint64(Uint32(t_lo + pb + Int32(e_))) << Uint64(32)
                            | Uint64(L.u32_of_f32(vals_[j_][e_]))
                        )
                    cnt = cnt + Int32(cute.arch.popc(bm))
        if lane == Int32(0):
            s_wc[wid] = cnt
        L.bar(4, 128)
        p1 = s_wc[0]
        p2 = p1 + s_wc[1]
        p3 = p2 + s_wc[2]
        tot = p3 + s_wc[3]
        t40 = self._tag40(gen)
        if live != Int32(0):
            if stid == Int32(0):
                self._stamp(mStamps, cta, 40)
        k = stid
        while k < tot:
            wk = (
                cutlass.select_(k >= p1, Int32(1), Int32(0))
                + cutlass.select_(k >= p2, Int32(1), Int32(0))
                + cutlass.select_(k >= p3, Int32(1), Int32(0))
            )
            bk = cutlass.select_(
                wk == Int32(1),
                p1,
                cutlass.select_(wk == Int32(2), p2, cutlass.select_(wk == Int32(3), p3, Int32(0))),
            )
            e = Uint64(sList[wk * Int32(MAXS * SPLIT_TOK // 4) + k - bk])
            L.st_g_u64(rbase + Int64(k) * Int64(8), e)
            if bscale > Float32(0.0):
                bq = L._vbin(e, Tv, bscale, NB_INITIAL)
                ordinal = atom_add_u32(ghist + Int64(bq) * Int64(4), Uint32(1))
                if ordinal < Uint32(BUCKET_CAP):
                    L.st_g_u64(self._bucket_addr(bucket_base, bq, Int32(ordinal)), e)
            k = k + Int32(128)
        if stid == Int32(127):
            if tot & Int32(1) != Int32(0):
                L.st_g_u64(
                    rbase + Int64(tot) * Int64(8),
                    Uint64(DUMMY_GPOS) << Uint64(32) | Uint64(4286578688),
                )
        B.fence_proxy_async_global()
        L.bar(4, 128)
        if stid == Int32(0):
            if live != Int32(0):
                self._stamp(mStamps, cta, 41)
            if cutlass.const_expr(self.relst):
                L.st_release_u64(cnt_addr, t40 << Uint64(24) | Uint64(Uint32(tot)))
            else:
                L.fence_acq_rel_gpu()
                st_rlx_u64(cnt_addr, t40 << Uint64(24) | Uint64(Uint32(tot)))
            if live != Int32(0):
                self._stamp(mStamps, cta, 42)
                self._stamp(mStamps, cta, ST_CAND_PUB)
        if cutlass.const_expr(self.pref):
            if live != Int32(0):
                k = stid
                while k < tot:
                    wk = (
                        cutlass.select_(k >= p1, Int32(1), Int32(0))
                        + cutlass.select_(k >= p2, Int32(1), Int32(0))
                        + cutlass.select_(k >= p3, Int32(1), Int32(0))
                    )
                    bk = cutlass.select_(
                        wk == Int32(1),
                        p1,
                        cutlass.select_(
                            wk == Int32(2), p2, cutlass.select_(wk == Int32(3), p3, Int32(0))
                        ),
                    )
                    gpos = Int32(
                        Uint32(
                            Uint64(sList[wk * Int32(MAXS * SPLIT_TOK // 4) + k - bk]) >> Uint64(32)
                        )
                    )
                    kvr = sBtcR[(gpos - t_lo) >> Int32(5)] * Int32(32) + (gpos & Int32(31))
                    prefetch_l2_bulk(cbase + Int64(kvr) * Int64(512), Int32(512))
                    k = k + Int32(128)
        return tot

    @cute.jit
    def _svc_scanpub(
        self,
        cta,
        tid,
        live,
        cnt_addr,
        rbase,
        ghist,
        bucket_base,
        gen,
        ntok,
        t_lo,
        sLog,
        Tv,
        bscale,
        sBtcR,
        cbase,
        s_wc,
        sList,
        mStamps,
        nth: cutlass.Constexpr[int],
        bid: cutlass.Constexpr[int],
        pf: cutlass.Constexpr[bool] = True,
    ):
        # one row's first-round publish by nth threads (named barrier bid): vectorized scan of the row's 2048
        # logits in 128-position chunks (round robin over the warps; positions < n via ntok), per-warp SMEM lists of the
        # candidates (logit >= T), then region packet, DRS buckets, even padding, released count, L2 prefetch of their
        # KV rows. live == 0: warm-up (scratch addresses, no stamps / prefetch).
        NW: cutlass.Constexpr[int] = nth // 32
        CPW: cutlass.Constexpr[int] = (16 + nth // 32 - 1) // (nth // 32)
        LCAP: cutlass.Constexpr[int] = CPW * 128
        lane = tid % Int32(32)
        wid = tid // Int32(32)
        lm = Int32(cute.arch.lanemask_lt())
        kT = L.fkey(Tv)
        lw = wid * Int32(LCAP)
        cnt = Int32(0)
        sb_ = Int32(sLog.iterator.toint())
        vals_ = []
        for c_ in cutlass.range_constexpr(CPW):
            pb = (wid + Int32(c_ * NW)) * Int32(128) + lane * Int32(4)
            vals_.append(lds_v4f32(sb_ + cutlass.min(pb, Int32(MAXS * SPLIT_TOK - 4)) * Int32(4)))
        for c_ in cutlass.range_constexpr(CPW):
            pb = (wid + Int32(c_ * NW)) * Int32(128) + lane * Int32(4)
            oks_ = []
            anyv = Int32(0)
            for e_ in cutlass.range_constexpr(4):
                ok = Int32(0)
                if pb + Int32(e_) < ntok:
                    if L.fkey(vals_[c_][e_]) >= kT:
                        ok = Int32(1)
                oks_.append(ok)
                anyv = anyv | ok
            if cute.arch.vote_any_sync(anyv != Int32(0)):
                for e_ in cutlass.range_constexpr(4):
                    bm = Int32(cute.arch.vote_ballot_sync(oks_[e_] != Int32(0)))
                    if oks_[e_] != Int32(0):
                        sList[lw + cnt + Int32(cute.arch.popc(bm & lm))] = Int64(
                            Uint64(Uint32(t_lo + pb + Int32(e_))) << Uint64(32)
                            | Uint64(L.u32_of_f32(vals_[c_][e_]))
                        )
                    cnt = cnt + Int32(cute.arch.popc(bm))
        if lane == Int32(0):
            s_wc[wid] = cnt
        L.bar(bid, nth)
        pre = [Int32(0)]
        for w_ in cutlass.range_constexpr(NW):
            pre.append(pre[w_] + s_wc[w_])
        tot = pre[NW]
        t40 = self._tag40(gen)
        if live != Int32(0):
            if tid == Int32(0):
                self._stamp(mStamps, cta, 40)
        k = tid
        while k < tot:
            wk = Int32(0)
            bk = Int32(0)
            for w_ in cutlass.range_constexpr(1, NW):
                if k >= pre[w_]:
                    wk = Int32(w_)
                    bk = pre[w_]
            e = Uint64(sList[wk * Int32(LCAP) + k - bk])
            L.st_g_u64(rbase + Int64(k) * Int64(8), e)
            if bscale > Float32(0.0):
                bq = L._vbin(e, Tv, bscale, NB_INITIAL)
                ordinal = atom_add_u32(ghist + Int64(bq) * Int64(4), Uint32(1))
                if ordinal < Uint32(BUCKET_CAP):
                    L.st_g_u64(self._bucket_addr(bucket_base, bq, Int32(ordinal)), e)
            k = k + Int32(nth)
        if tid == Int32(nth - 1):
            if tot & Int32(1) != Int32(0):
                L.st_g_u64(
                    rbase + Int64(tot) * Int64(8),
                    Uint64(DUMMY_GPOS) << Uint64(32) | Uint64(4286578688),
                )
        B.fence_proxy_async_global()
        L.bar(bid, nth)
        if tid == Int32(0):
            if live != Int32(0):
                self._stamp(mStamps, cta, 41)
            if cutlass.const_expr(self.relst):
                L.st_release_u64(cnt_addr, t40 << Uint64(24) | Uint64(Uint32(tot)))
            else:
                L.fence_acq_rel_gpu()
                st_rlx_u64(cnt_addr, t40 << Uint64(24) | Uint64(Uint32(tot)))
            if live != Int32(0):
                self._stamp(mStamps, cta, 42)
                self._stamp(mStamps, cta, ST_CAND_PUB)
        if cutlass.const_expr(self.pref and pf):
            if live != Int32(0):
                k = tid
                while k < tot:
                    wk = Int32(0)
                    bk = Int32(0)
                    for w_ in cutlass.range_constexpr(1, NW):
                        if k >= pre[w_]:
                            wk = Int32(w_)
                            bk = pre[w_]
                    gpos = Int32(Uint32(Uint64(sList[wk * Int32(LCAP) + k - bk]) >> Uint64(32)))
                    kvr = sBtcR[(gpos - t_lo) >> Int32(5)] * Int32(32) + (gpos & Int32(31))
                    prefetch_l2_bulk(cbase + Int64(kvr) * Int64(512), Int32(512))
                    k = k + Int32(nth)
        return tot

    @cute.jit
    def _poll_dir_nb(self, ws, gen, seq, s_dir):
        # one look at the row's directive word; kind K_NONE when nothing newer than seq is published
        t40 = self._tag40(gen)
        h = L.ld_relaxed_u64(self._w(ws, W_DIR))
        kind = Int32(K_NONE)
        dseq = seq
        if h >> Uint64(24) == t40 and Int32(Uint32(h >> Uint64(8) & Uint64(65535))) > seq:
            kind = Int32(Uint32(h & Uint64(15)))
            dseq = Int32(Uint32(h >> Uint64(8) & Uint64(65535)))
            if kind != Int32(K_DONE):
                L.fence_acq_rel_gpu()
                p0 = L.ld_relaxed_u64(self._w(ws, W_DP))
                p1 = L.ld_relaxed_u64(self._w(ws, W_DP + 1))
                p2 = L.ld_relaxed_u64(self._w(ws, W_DP + 2))
                s_dir[2] = Int32(Uint32(p2 & Uint64(4294967295)))
                s_dir[3] = Int32(Uint32(p0 & Uint64(4294967295)))
                s_dir[4] = Int32(Uint32(p0 >> Uint64(32)))
                s_dir[5] = Int32(Uint32(p1 & Uint64(4294967295)))
                s_dir[6] = Int32(Uint32(p1 >> Uint64(32)))
        s_dir[0] = kind
        s_dir[1] = dseq

    @cute.jit
    def _stream_serve(
        self,
        cta,
        tidx,
        ws,
        gen,
        seq,
        tot,
        ntok,
        t_lo,
        sLog,
        s_wc,
        s_dir,
        mRegion,
        mHist,
        arena,
        blocking: cutlass.Constexpr[bool],
        nth: cutlass.Constexpr[int] = 256,
        bid: cutlass.Constexpr[int] = 5,
        hoff: cutlass.Constexpr[int] = S_OFF_KV0,
    ):
        # serve one row's selection directives: until K_DONE (blocking) or one non-blocking check
        rbase = mRegion.iterator.toint() + Int64(cta) * Int64(2 * CAPR * 8)
        fin = Int32(0)
        go = Int32(1)
        while go != Int32(0):
            if tidx == 0:
                if cutlass.const_expr(blocking):
                    self._poll_dir(ws, gen, seq, s_dir)
                else:
                    self._poll_dir_nb(ws, gen, seq, s_dir)
            L.bar(bid, nth)
            kind = s_dir[0]
            dseq = s_dir[1]
            if kind == Int32(K_PUB):
                slot = s_dir[2] & Int32(255)
                klo = Uint64(Uint32(s_dir[4])) << Uint64(32) | Uint64(Uint32(s_dir[3]))
                khi = Uint64(Uint32(s_dir[6])) << Uint64(32) | Uint64(Uint32(s_dir[5]))
                base = tot
                if cutlass.const_expr(self.bulk):
                    base = tot + (tot & Int32(1))
                if slot == Int32(2):
                    base = Int32(CAPR)
                self._stream_pub(
                    cta,
                    tidx,
                    ws,
                    gen,
                    slot,
                    base,
                    klo,
                    khi,
                    ntok,
                    t_lo,
                    sLog,
                    s_wc,
                    rbase,
                    nth=nth,
                    bid=bid,
                )
            elif kind == Int32(K_HIST):
                plen = s_dir[2] >> Int32(8) & Int32(255)
                pref = Uint64(Uint32(s_dir[4])) << Uint64(32) | Uint64(Uint32(s_dir[3]))
                self._stream_hist(
                    cta,
                    tidx,
                    ws,
                    gen,
                    dseq,
                    pref,
                    plen,
                    ntok,
                    t_lo,
                    sLog,
                    mHist,
                    arena,
                    nth=nth,
                    bid=bid,
                    hoff=hoff,
                )
            elif kind != Int32(K_NONE):
                fin = Int32(1)
            if kind != Int32(K_NONE):
                seq = dseq
            L.bar(bid, nth)
            if cutlass.const_expr(not blocking):
                go = Int32(0)
            if fin != Int32(0):
                go = Int32(0)
        return (seq, fin)

    @cute.jit
    def _serve_row(
        self,
        q,
        cta,
        tidx,
        ws,
        arena,
        s_wc,
        s_dir,
        mRegion,
        mHist,
        blocking: cutlass.Constexpr[bool],
        nth: cutlass.Constexpr[int] = 256,
        bid: cutlass.Constexpr[int] = 5,
        hoff: cutlass.Constexpr[int] = S_OFF_KV0,
        ebid: cutlass.Constexpr[int] = 3,
    ):
        # serve row q from its saved state (all math threads; tidx 0 writes the state back)
        rs_ = self.row_sizes()
        rst = cute.make_tensor(
            cute.recast_ptr(arena + S_OFF_ROWST, dtype=Int32) + q * Int32(8), cute.make_layout(8)
        )
        rgen = cute.make_tensor(
            cute.recast_ptr(arena + S_OFF_ROWST + 256, dtype=Int64), cute.make_layout(8)
        )
        if rst[4] == Int32(0):
            sLogQ = cute.make_tensor(
                cute.recast_ptr(arena + S_OFF_LOGR, dtype=Float32) + q * Int32(MAXS * SPLIT_TOK),
                cute.make_layout(MAXS * SPLIT_TOK),
            )
            seq2, fin2 = self._stream_serve(
                cta,
                tidx,
                ws + Int64(q) * Int64(rs_["ws"] * 8),
                Uint64(rgen[q]),
                rst[3],
                rst[0],
                rst[1],
                rst[2],
                sLogQ,
                s_wc,
                s_dir,
                _flat(mRegion, q, rs_["region"]),
                _flat(mHist, q, rs_["hist"]),
                arena,
                blocking,
                nth=nth,
                bid=bid,
                hoff=hoff,
            )
            if tidx == 0:
                rst[3] = seq2
                rst[4] = fin2
        L.bar(ebid, nth)

    @cute.jit
    def _poll_dir(self, ws, gen, seq, s_dir):
        t40 = self._tag40(gen)
        t64 = gen + Uint64(1)
        tdl = self._deadline()
        st_ = Int32(0)
        h = Uint64(0)
        while st_ == Int32(0):
            h = L.ld_relaxed_u64(self._w(ws, W_DIR))
            av = L.ld_relaxed_u64(self._w(ws, W_ABORT))
            if h >> Uint64(24) == t40 and Int32(Uint32(h >> Uint64(8) & Uint64(65535))) > seq:
                st_ = Int32(1)
            elif av == t64:
                st_ = Int32(2)
            elif Int64(L.gtimer()) > tdl:
                st_ = Int32(3)
            else:
                L.nanosleep(Int32(64))
        if st_ == Int32(3):
            self._raise_abort(ws, gen, Int32(E_TIMEOUT))
        elif st_ == Int32(2):
            self._flag(ws, gen, Int32(E_ABORTED))
        kind = Int32(K_DONE)
        dseq = seq + Int32(1)
        if st_ == Int32(1):
            kind = Int32(Uint32(h & Uint64(15)))
            dseq = Int32(Uint32(h >> Uint64(8) & Uint64(65535)))
            if kind != Int32(K_DONE):
                L.fence_acq_rel_gpu()
                p0 = L.ld_relaxed_u64(self._w(ws, W_DP))
                p1 = L.ld_relaxed_u64(self._w(ws, W_DP + 1))
                p2 = L.ld_relaxed_u64(self._w(ws, W_DP + 2))
                s_dir[2] = Int32(Uint32(p2 & Uint64(4294967295)))
                s_dir[3] = Int32(Uint32(p0 & Uint64(4294967295)))
                s_dir[4] = Int32(Uint32(p0 >> Uint64(32)))
                s_dir[5] = Int32(Uint32(p1 & Uint64(4294967295)))
                s_dir[6] = Int32(Uint32(p1 >> Uint64(32)))
        s_dir[0] = kind
        s_dir[1] = dseq

    @cute.jit
    def _k50(self, x, gpos):
        return Uint64(L.fkey(x)) << Uint64(18) | Uint64(Uint32(Int32(GPOS_MASK) - gpos))

    @cute.jit
    def _stream_pub(
        self,
        cta,
        tidx,
        ws,
        gen,
        slot,
        base,
        klo,
        khi,
        ntok,
        t_lo,
        sLog,
        s_wc,
        rbase,
        nth: cutlass.Constexpr[int] = 256,
        bid: cutlass.Constexpr[int] = 5,
    ):
        lane = tidx % 32
        wid = tidx // 32
        lm = Int32(cute.arch.lanemask_lt())
        CW = (ntok + Int32(nth - 1)) // Int32(nth) * Int32(32)
        c0 = wid * CW
        c1 = cutlass.min(ntok, c0 + CW)
        cnt = Int32(0)
        kb = c0
        while kb < c1:
            p = kb + lane
            ok = Int32(0)
            if p < c1:
                kk = self._k50(sLog[p], t_lo + p)
                if kk >= klo and kk < khi:
                    ok = Int32(1)
            cnt = cnt + Int32(cute.arch.popc(cute.arch.vote_ballot_sync(ok != Int32(0))))
            kb = kb + Int32(32)
        if lane == 0:
            s_wc[wid] = cnt
        L.bar(bid, nth)
        o = base
        tot = Int32(0)
        for w_ in cutlass.range_constexpr(nth // 32):
            v_ = s_wc[w_]
            if Int32(w_) < wid:
                o = o + v_
            tot = tot + v_
        kb = c0
        while kb < c1:
            p = kb + lane
            ok = Int32(0)
            e = Uint64(0)
            if p < c1:
                x = sLog[p]
                gpos = t_lo + p
                kk = self._k50(x, gpos)
                if kk >= klo and kk < khi:
                    ok = Int32(1)
                    e = Uint64(Uint32(gpos)) << Uint64(32) | Uint64(L.u32_of_f32(x))
            bm = Int32(cute.arch.vote_ballot_sync(ok != Int32(0)))
            if ok != Int32(0):
                L.st_g_u64(rbase + Int64(o + Int32(cute.arch.popc(bm & lm))) * Int64(8), e)
            o = o + Int32(cute.arch.popc(bm))
            kb = kb + Int32(32)
        L.bar(bid, nth)
        if tidx == 0:
            L.fence_acq_rel_gpu()
            st_rlx_u64(
                self._cnt_addr(ws, slot, cta), self._tag40(gen) << Uint64(24) | Uint64(Uint32(tot))
            )

    @cute.jit
    def _stream_hist(
        self,
        cta,
        tidx,
        ws,
        gen,
        seq,
        pref,
        plen,
        ntok,
        t_lo,
        sLog,
        mHist,
        arena,
        nth: cutlass.Constexpr[int] = 256,
        bid: cutlass.Constexpr[int] = 5,
        hoff: cutlass.Constexpr[int] = S_OFF_KV0,
    ):
        sh = _sm_tensor(arena, hoff, Int32, NB_DRS)
        for z_ in cutlass.range_constexpr(NB_DRS // nth):
            sh[tidx + Int32(z_ * nth)] = Int32(0)
        L.bar(bid, nth)
        shp = Uint64(Int32(K50_BITS) - plen)
        shd = Uint64(Int32(K50_BITS - DRS_DIG) - plen)
        lmk = Int32(cute.arch.lanemask_lt())
        nit = (ntok + Int32(nth - 1)) // Int32(nth)
        jj = Int32(0)
        while jj < nit:
            p = tidx + jj * Int32(nth)
            dg = Int32(NB_DRS)
            if p < ntok:
                kk = self._k50(sLog[p], t_lo + p)
                if kk >> shp == pref:
                    dg = Int32(Uint32(kk >> shd & Uint64(NB_DRS - 1)))
            mm_ = match_any_b32(dg)
            if dg != Int32(NB_DRS):
                if mm_ & lmk == Int32(0):
                    cute.arch.red(
                        sh.iterator + dg,
                        Int32(cute.arch.popc(mm_)),
                        op="add",
                        dtype=Int32,
                        sem="relaxed",
                        scope="cta",
                    )
            jj = jj + Int32(1)
        L.bar(bid, nth)
        gh = mHist.iterator.toint() + Int64(seq) * Int64(NB_DRS * 4)
        for z_ in cutlass.range_constexpr(NB_DRS // nth):
            b_ = tidx + Int32(z_ * nth)
            v_ = sh[b_]
            if v_ != Int32(0):
                red_add_u32(gh + Int64(b_) * Int64(4), v_)
        L.bar(bid, nth)
        if tidx == 0:
            L.fence_acq_rel_gpu()
            L.red_add_u64(self._w(ws, W_RC + seq), Uint64(1))

    @cute.jit
    def _team_cta(
        self,
        arena,
        cta,
        tidx,
        ws,
        tma_q,
        tma_swa,
        tma_cmp,
        mOut,
        mCtx,
        mSwaLocal,
        mBtSwa,
        mBtC,
        mSinks,
        mSwaLens,
        mTopk,
        mGidx,
        mHist,
        mBoard,
        mRegion,
        mStamps,
        mCtl,
        softmax_scale,
        mDesc,
        mCPool=None,
        qrow=0,
    ):
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        d = cta - Int32(self.ns)
        bars = _ptr(arena, T_OFF_BAR, Int64)
        s_q = _ptr(arena, T_OFF_Q, A.E4M3)
        s_kv = _ptr(arena, T_OFF_KV, A.E4M3)
        s_ext = _ptr(arena, T_OFF_LIST, A.E4M3)
        s_stat = _ptr(arena, T_OFF_STAT, Float32)
        s_fin = _ptr(arena, T_OFF_FIN, Float32)
        s_idx = _ptr(arena, T_OFF_IDX, Int32)
        misc = _ptr(arena, T_OFF_MISC, Int32)
        gw1 = Uint64(0)
        gw2 = Uint64(0)
        gw3 = Uint64(0)
        if tidx == Int32(0):
            gw1, gw2, gw3 = self._entry_loads(ws)
            (misc + 12).store(Int32(0))
            (misc + 13).store(Int32(0))
            (misc + 14).store(Int32(0))
            (misc + 15).store(Int32(0))
            (misc + 16).store(Int32(0))
        if warp == Int32(0):
            if prims.elect_sync():
                prims.mbarrier_init(bars + B_QF, 1)
                for i in cutlass.range_constexpr(NSLOTS):
                    if cutlass.const_expr(self.rs24_lsu and 4 <= i < 8):
                        # g = 1: tile 4's slots also count the control WG's cp.async arrivals
                        prims.mbarrier_init(
                            bars + (B_KVF + i),
                            cutlass.select_(
                                (d & Int32(1)) != Int32(0), Int32(1 + A.LSU_ARR), Int32(1)
                            ),
                        )
                    else:
                        prims.mbarrier_init(bars + (B_KVF + i), 1)
                    prims.mbarrier_init(bars + (B_KVE + i), 1)
                for i in cutlass.range_constexpr(4):
                    prims.mbarrier_init(bars + (B_SF + i), 1)
                    prims.mbarrier_init(bars + (B_SE + i), 1)
                    prims.mbarrier_init(bars + (B_PF + i), 4)
                prims.mbarrier_init(bars + B_OF, 1)
                prims.mbarrier_init(bars + B_OC, 4)
                prims.mbarrier_init(bars + B_FIN, 4)
                prims.mbarrier_init(bars + B_SWA, 128)
                prims.mbarrier_init(bars + B_SEL, 128)
                prims.mbarrier_init(bars + B_T, 1)
                prims.mbarrier_init(bars + B_PVL, 1)
                prims.mbarrier_init(bars + B_BTC, 1)
                prims.mbarrier_init(bars + B_LST, 1)
                prims.mbarrier_init(bars + B_EAR, 128)
                if cutlass.const_expr(self.rs24):
                    prims.mbarrier_init(bars + B_RX, 1)
                    prims.mbarrier_init(bars + B_PR, 1)
                    prims.mbarrier_init(bars + B_DRY, 1)
                    prims.mbarrier_arrive_expect_tx(
                        bars + B_RX, A.rs24_rx_bytes(self.rs24_f32 or self.rs24_asym)
                    )
                    prims.mbarrier_arrive_expect_tx(
                        bars + B_DRY, A.rs24_rx_bytes(self.rs24_f32 or self.rs24_asym)
                    )
        if warp == Int32(A.W_G0):
            if prims.elect_sync():
                prims.prefetch_tensormap(tma_q.get_ptr())
                prims.prefetch_tensormap(tma_swa.get_ptr())
                prims.prefetch_tensormap(tma_cmp.get_ptr())
        prims.fence_mbarrier_init()
        if warp == Int32(A.W_MMA):
            prims.tcgen05_alloc(misc, A.TMEM_COLS, group=prims.CTAGroup.CTA_1)
            prims.tcgen05_relinquish_alloc_permit(group=prims.CTAGroup.CTA_1)
        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
        prims.barrier_cta_sync(0)
        if cutlass.const_expr(self.rs24):
            # phase 0 of the cluster barrier: arrive now (the mbarrier inits are fenced above); every thread waits only
            # before its first remote operation (softmax WG: real merge; control WG: B_PR arrive; others: at exit)
            cute.arch.cluster_arrive_relaxed()
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        tmem = misc.load()
        scale_log2 = softmax_scale * Float32(A.LOG2_E)
        tr = None
        if cutlass.const_expr(self.stamps):
            tr = mStamps.iterator.toint() + Int64(cta) * Int64(NSTAMP * 8)
        swa_len = mSwaLens[0]
        n = mDesc[2]
        if n < Int32(0) or n > Int32(N_MAX):
            n = Int32(0)
        n_sel = cutlass.min(n, Int32(K_TOP))
        if warp < Int32(4):
            prims.setmaxregister(A.REG_SOFTMAX, prims.SetMaxRegisterAction.INCREASE)
            if cutlass.const_expr(self.fblka):
                self._block_a_lead(
                    arena, cta, tidx, ws, n, mBoard, mStamps, gw1, gw2, gw3, mCtl, d, mHist
                )
            else:
                # One call site for the whole team: the followers run the leader's threshold code once, dry (one
                # stale board read, no wait, no publish), at kernel start, so the leader's real pass when the board
                # completes fetches its code from L2. Followers then read the leader's T line as before.
                self._block_a_lead(
                    arena, cta, tidx, ws, n, mBoard, mStamps, gw1, gw2, gw3, mCtl, d, mHist
                )
                if d != Int32(0):
                    self._follow_tline(arena, cta, d, tidx, ws, mStamps, gw1, gw2, gw3, mCtl)
            cute.arch.fence_proxy("async.shared", space="cta")
            L.bar(BAR_SMX, 128)
            if tidx == Int32(0):
                prims.mbarrier_arrive(bars + B_T)
            if cutlass.const_expr(self.gjoin):
                if cutlass.const_expr(self.rs24):
                    zsm = Int32(cute.recast_ptr(arena + T_OFF_MISC + 61 * 4, dtype=Int32).toint())
                    A.role_softmax_24(
                        tmem,
                        bars + B_SF,
                        bars + B_PF,
                        scale_log2,
                        swa_len,
                        n_sel,
                        d,
                        bars + B_OF,
                        mSinks.iterator.raw_ptr(),
                        mOut.iterator.toint(),
                        Int32(_ptr(arena, T_OFF_LIST, Int32).toint()),
                        Int32(_ptr(arena, T_OFF_STAT, Int32).toint()),
                        Int32((bars + B_PVL).toint()),
                        Int32((bars + B_RX).toint()),
                        Int32((bars + B_PR).toint()),
                        Int32((bars + B_DRY).toint()),
                        zsm,
                        tr,
                        full_tiles=self.full_attention_tiles,
                        dry_merge=self.rs24_dry,
                        f32x=self.rs24_f32,
                        asym=self.rs24_asym,
                    )
                else:
                    A.role_softmax_team(
                        tmem,
                        bars + B_SF,
                        bars + B_PF,
                        s_stat,
                        s_fin,
                        bars + B_FIN,
                        scale_log2,
                        swa_len,
                        n_sel,
                        Int32(0),
                        Int32(NTILES),
                        Float32(A.NEG_FLT_MAX),
                        Float32(0.0),
                        True,
                        tr,
                        bars + B_OF,
                        d,
                        full_tiles=self.full_attention_tiles,
                    )
                if cutlass.const_expr(self.late_done == 2):
                    if d == Int32(0):
                        if tidx == Int32(0):
                            mi_ = _sm_tensor(arena, T_OFF_MISC, Int32, 64)
                            if mi_[12] == Int32(P_T1) and mi_[14] != Int32(0):
                                gq_ = Uint64(Uint32(mi_[5])) << Uint64(32) | Uint64(Uint32(mi_[4]))
                                self._publish_dir(
                                    ws,
                                    gq_,
                                    Int32(1),
                                    Int32(K_DONE),
                                    Int32(P_T1),
                                    Uint64(0),
                                    Uint64(0),
                                    Uint64(0),
                                )
            else:
                A.role_softmax_team(
                    tmem,
                    bars + B_SF,
                    bars + B_PF,
                    s_stat,
                    s_fin,
                    bars + B_FIN,
                    scale_log2,
                    swa_len,
                    n_sel,
                    Int32(0),
                    Int32(NTILES),
                    Float32(A.NEG_FLT_MAX),
                    Float32(0.0),
                    True,
                    tr,
                    full_tiles=self.full_attention_tiles,
                )
        elif warp < Int32(8):
            if cutlass.const_expr(self.ctl_regs):
                prims.setmaxregister(A.REG_CTL, prims.SetMaxRegisterAction.INCREASE)
            if cutlass.const_expr(self.rs24):
                self._control_s1d(
                    arena,
                    cta,
                    d,
                    tidx - Int32(128),
                    ws,
                    n,
                    s_idx,
                    bars,
                    mSwaLocal,
                    mBtSwa,
                    mBtC,
                    mTopk,
                    mGidx,
                    mRegion,
                    mHist,
                    mStamps,
                    mCtl,
                    None,
                    None,
                )
                if cutlass.const_expr(self.rs24_lsu):
                    if (d & Int32(1)) != Int32(0):
                        # tile 4's rows were published by this warpgroup before its B_SEL arrive (named barrier); issue
                        # before the cluster wait / B_PR arrive (the partner pushes into slot 8 only at its merge)
                        A.lsu_tile4(
                            tidx - Int32(128),
                            d,
                            self.rs24_lsu,
                            Int32(s_kv.toint()),
                            Int32(s_idx.toint()),
                            Int32((bars + B_KVF).toint()),
                            bars + B_KVE,
                            mCPool.iterator.toint(),
                            tr,
                        )
                        if cutlass.const_expr(self.stamps):
                            A._stamp(
                                mStamps.iterator.toint() + Int64(cta) * Int64(NSTAMP * 8),
                                Int32(52),
                                tidx == Int32(128),
                            )
                cute.arch.cluster_wait()
                if tidx == Int32(160):
                    # (not ctid 0: on team CTA 0 it may still be in the late-done follower wait)
                    B.mbar_arrive_remote(
                        B.mapa_u32(Int32((bars + B_PR).toint()), B.cluster_ctarank() ^ Int32(1))
                    )
            elif cutlass.const_expr(self.gjoin):
                self._control_s1d(
                    arena,
                    cta,
                    d,
                    tidx - Int32(128),
                    ws,
                    n,
                    s_idx,
                    bars,
                    mSwaLocal,
                    mBtSwa,
                    mBtC,
                    mTopk,
                    mGidx,
                    mRegion,
                    mHist,
                    mStamps,
                    mCtl,
                    (
                        tma_swa,
                        tma_cmp,
                        s_kv,
                        s_ext,
                        s_idx,
                        bars + B_KVF,
                        bars + B_KVE,
                        bars + B_SWA,
                        bars + B_SEL,
                        Int32(A.NGW) + warp - Int32(4),
                        d,
                        NTILES,
                        NSLOTS,
                        None,
                        Int32(2),
                        A.NGW + 4,
                        bars + B_EAR,
                        2,
                    ),
                    (
                        tmem,
                        bars + B_PF,
                        bars + B_OF,
                        bars + B_OC,
                        s_stat,
                        s_fin,
                        bars + B_FIN,
                        mSinks.iterator.raw_ptr(),
                        mOut.iterator.toint(),
                        scale_log2,
                        d,
                        Int32(s_kv.toint()),
                        NTILES,
                        BAR_EPI,
                        tr,
                        bars + B_PVL,
                    ),
                )
            else:
                self._control_s1d(
                    arena,
                    cta,
                    d,
                    tidx - Int32(128),
                    ws,
                    n,
                    s_idx,
                    bars,
                    mSwaLocal,
                    mBtSwa,
                    mBtC,
                    mTopk,
                    mGidx,
                    mRegion,
                    mHist,
                    mStamps,
                    mCtl,
                )
                A.role_corr_epi_team(
                    tmem,
                    bars + B_PF,
                    bars + B_OF,
                    bars + B_OC,
                    s_stat,
                    s_fin,
                    bars + B_FIN,
                    mSinks.iterator.raw_ptr(),
                    mOut.iterator.toint(),
                    scale_log2,
                    d,
                    Int32(s_kv.toint()),
                    NTILES,
                    BAR_EPI,
                    tr,
                )
        elif warp == Int32(A.W_MMA):
            prims.setmaxregister(A.REG_OTHER, prims.SetMaxRegisterAction.DECREASE)
            if cutlass.const_expr(self.gjoin):
                if cutlass.const_expr(self.rs24):
                    A.role_mma_24(
                        tmem,
                        s_q,
                        s_kv,
                        s_ext,
                        bars + B_QF,
                        bars + B_KVF,
                        bars + B_KVE,
                        bars + B_SF,
                        bars + B_PF,
                        bars + B_OF,
                        d,
                        bars + B_PVL,
                        tr,
                        pfence=self.rs24_lsu > 0,
                    )
                else:
                    A.role_mma_ordered(
                        tmem,
                        s_q,
                        s_kv,
                        s_ext,
                        bars + B_QF,
                        bars + B_KVF,
                        bars + B_KVE,
                        bars + B_SF,
                        bars + B_SE,
                        bars + B_PF,
                        bars + B_OF,
                        d,
                        NTILES,
                        NSLOTS,
                        bars + B_PVL,
                        tr,
                        bars + B_EAR if self.mma_block else None,
                        self.mma_idle_ns,
                    )
            else:
                A.role_mma_team(
                    tmem,
                    s_q,
                    s_kv,
                    s_ext,
                    bars + B_QF,
                    bars + B_KVF,
                    bars + B_KVE,
                    bars + B_SF,
                    bars + B_SE,
                    bars + B_PF,
                    bars + B_OC,
                    bars + B_OF,
                    d,
                    NTILES,
                    NSLOTS,
                    tr,
                    self.qk_ahead,
                )
        else:
            prims.setmaxregister(A.REG_OTHER, prims.SetMaxRegisterAction.DECREASE)
            if warp == Int32(A.W_G0):
                A.role_q_load(tma_q, s_q, bars + B_QF, qrow)
            if warp == Int32(A.W_G0 + 1):
                if prims.elect_sync():
                    btc_cols = cute.size(mBtC, mode=[1])
                    npg = cutlass.min(
                        (n + Int32(31)) // Int32(32), cutlass.min(btc_cols, Int32(BTC_CAP))
                    )
                    nbt = npg * Int32(4) // Int32(16) * Int32(16)
                    if nbt > Int32(0):
                        B.mbar_arrive_expect_tx(Int32((bars + B_BTC).toint()), nbt)
                        B.bulk_g2s(
                            Int32(cute.recast_ptr(arena + T_OFF_BTC, dtype=Int32).toint()),
                            mBtC.iterator.toint(),
                            nbt,
                            Int32((bars + B_BTC).toint()),
                        )
                    else:
                        prims.mbarrier_arrive(bars + B_BTC)
            if warp == Int32(A.W_G0 + 2):
                rgb = mRegion.iterator.toint()
                for q in cutlass.range_constexpr(NB_INITIAL // 32):
                    bha = self._bucket_addr(rgb, cute.arch.lane_idx() + Int32(q * 32), Int32(0))
                    for h_ in cutlass.range_constexpr(4):
                        L.prefetch_l2(bha + Int64(128 * h_))
            if cutlass.const_expr(self.swag):
                if warp >= Int32(A.W_G0 + 3):
                    gt_ = (warp - Int32(A.W_G0 + 3)) * Int32(32) + cute.arch.lane_idx()
                    mbs_ = cute.size(mBtSwa, mode=[1])
                    loc_ = mSwaLocal[0, gt_]
                    rsw_ = L.l2g_swa_local(loc_, Int32(0), mBtSwa, mbs_, Int32(0))
                    (s_idx + gt_).store(rsw_)
                    # each writer releases its own index store to the gather warps
                    prims.mbarrier_arrive(bars + B_SWA)
            if cutlass.const_expr(self.rs24):
                A.role_gather_24(
                    tma_swa,
                    tma_cmp,
                    s_kv,
                    s_ext,
                    s_idx,
                    bars + B_KVF,
                    bars + B_KVE,
                    bars + B_SWA,
                    bars + B_EAR,
                    bars + B_SEL,
                    warp - Int32(A.W_G0),
                    d,
                    NTILES,
                    tr,
                    (misc + 16) if self.rs24_t4split else None,
                    lsu_g=self.rs24_lsu,
                )
            else:
                A.role_gather_team(
                    tma_swa,
                    tma_cmp,
                    s_kv,
                    s_ext,
                    s_idx,
                    bars + B_KVF,
                    bars + B_KVE,
                    bars + B_SWA,
                    bars + B_SEL,
                    warp - Int32(A.W_G0),
                    d,
                    NTILES,
                    NSLOTS,
                    tr,
                    Int32(0),
                    A.NGW + 4 if self.gjoin else A.NGW,
                    bars + B_EAR if self.gjoin else None,
                    2 if self.gjoin else 1,
                )
            if cutlass.const_expr(self.hook):
                if warp == Int32(A.W_G0 + A.NGW - 1):
                    if cute.arch.lane_idx() == Int32(0):
                        self._stamp(mStamps, cta, 106)
                        self._team_exit(arena, d, ws)
                        self._stamp(mStamps, cta, 107)
        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
        prims.barrier_cta_sync(0)
        if cutlass.const_expr(self.rs24 and self.rs24_exitbar):
            if warp >= Int32(8):
                cute.arch.cluster_wait()
            # neither CTA of the pair exits while the other may still push into it (not needed: every remote write or
            # arrive into a CTA is consumed by one of its own mbarrier waits before it finishes; a cluster wait that
            # actually has to wait costs ~1 us)
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
        if warp == Int32(A.W_MMA):
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            prims.tcgen05_dealloc(
                prims.make_tmem_ptr(tmem, Float32), A.TMEM_COLS, group=prims.CTAGroup.CTA_1
            )
        if cutlass.const_expr(self.stamps):
            if tidx == Int32(0):
                (mStamps.iterator + cta * NSTAMP + ST_M_EXIT).store(Int64(L.gtimer()))

    def _rsel(self, vals, r):
        # select vals[r] (Python list, dynamic r)
        v = vals[0]
        for k_ in range(1, len(vals)):
            v = cutlass.select_(r == Int32(k_), vals[k_], v)
        return v

    @cute.jit
    def _team_cta_mr(
        self,
        arena,
        cta,
        tidx,
        ws0,
        tma_q,
        tma_swa,
        tma_cmp,
        mOutA,
        mCtxA,
        mSwaLocalA,
        mBtSwaA,
        mBtCA,
        mSinks,
        mSwaLensA,
        mTopkA,
        mGidxA,
        mHistA,
        mBoardA,
        mRegionA,
        mStampsA,
        mCtlA,
        softmax_scale,
        mDescA,
        mCPool,
        tm,
    ):
        # one team serving rows tm, tm + nteams, ...
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        d = cta - Int32(self.ns)
        bars = _ptr(arena, T_OFF_BAR, Int64)
        s_q = _ptr(arena, T_OFF_Q, A.E4M3)
        s_kv = _ptr(arena, T_OFF_KV, A.E4M3)
        s_ext = _ptr(arena, T_OFF_LIST, A.E4M3)
        s_stat = _ptr(arena, T_OFF_STAT, Float32)
        s_fin = _ptr(arena, T_OFF_FIN, Float32)
        s_idx = _ptr(arena, T_OFF_IDX, Int32)
        misc = _ptr(arena, T_OFF_MISC, Int32)
        pw = _ptr(arena, T_OFF_PW, Int32)  # pass entry words
        rs_ = self.row_sizes()
        npass_ = (Int32(self.nrows) - tm + Int32(self.nteams - 1)) // Int32(self.nteams)
        # thread 0 loads every pass's entry words, n and swa_len at kernel start (memory idle) into
        # misc[32 + 8 k ..] (only warp 0 waits for them; the setmaxnreg.dec warps carry no pending loads)
        if tidx == Int32(0):
            for kk_ in cutlass.range_constexpr((self.nrows + self.nteams - 1) // self.nteams):
                rk_ = cutlass.min(tm + Int32(kk_ * self.nteams), Int32(self.nrows - 1))
                e1_, e2_, e3_ = self._entry_loads(ws0 + Int64(rk_) * Int64(rs_["ws"] * 8))
                for jj_, ev_ in enumerate((e1_, e2_, e3_)):
                    (pw + (8 * kk_ + 2 * jj_)).store(Int32(Uint32(ev_ & Uint64(4294967295))))
                    (pw + (1 + 8 * kk_ + 2 * jj_)).store(Int32(Uint32(ev_ >> Uint64(32))))
                (pw + (6 + 8 * kk_)).store(_flat(mDescA, rk_, rs_["desc"])[2])
                (pw + (7 + 8 * kk_)).store(_row1(mSwaLensA, rk_)[0])
        if warp == Int32(A.W_G0):
            if prims.elect_sync():
                prims.prefetch_tensormap(tma_q.get_ptr())
                prims.prefetch_tensormap(tma_swa.get_ptr())
                prims.prefetch_tensormap(tma_cmp.get_ptr())
        if warp == Int32(A.W_MMA):
            prims.tcgen05_alloc(misc, A.TMEM_COLS, group=prims.CTAGroup.CTA_1)
            prims.tcgen05_relinquish_alloc_permit(group=prims.CTAGroup.CTA_1)
        if warp < Int32(4):
            prims.setmaxregister(A.REG_SOFTMAX, prims.SetMaxRegisterAction.INCREASE)
            for k_ in cutlass.range(npass_, unroll=1):
                row_ = tm + k_ * Int32(self.nteams)
                mStamps = cute.make_tensor(
                    mStampsA.iterator + k_ * Int32(self.num_sms * NSTAMP), mStampsA.layout
                )
                ws = ws0 + Int64(row_) * Int64(rs_["ws"] * 8)
                mOut = _row2(mOutA, row_)
                _mCtx = _row1(mCtxA, row_)
                mSwaLocal = _row2(mSwaLocalA, row_)
                mBtSwa = _row2(mBtSwaA, row_)
                mBtC = _row2(mBtCA, row_)
                _mSwaLens = _row1(mSwaLensA, row_)
                mTopk = _row2(mTopkA, row_)
                mGidx = _row2(mGidxA, row_)
                mHist = _flat(mHistA, row_, rs_["hist"])
                mBoard = _flat(mBoardA, row_, rs_["board"])
                mRegion = _flat(mRegionA, row_, rs_["region"])
                mCtl = _flat(mCtlA, row_, rs_["ctl"])
                _mDesc = _flat(mDescA, row_, rs_["desc"])
                qrow = row_
                gw1 = Uint64(0)
                gw2 = Uint64(0)
                gw3 = Uint64(0)
                if tidx == Int32(0):
                    gw1 = Uint64(Uint32((pw + (Int32(0) + k_ * Int32(8))).load())) | Uint64(
                        Uint32((pw + (Int32(1) + k_ * Int32(8))).load())
                    ) << Uint64(32)
                    gw2 = Uint64(Uint32((pw + (Int32(2) + k_ * Int32(8))).load())) | Uint64(
                        Uint32((pw + (Int32(3) + k_ * Int32(8))).load())
                    ) << Uint64(32)
                    gw3 = Uint64(Uint32((pw + (Int32(4) + k_ * Int32(8))).load())) | Uint64(
                        Uint32((pw + (Int32(5) + k_ * Int32(8))).load())
                    ) << Uint64(32)
                    (misc + 12).store(Int32(0))
                    (misc + 13).store(Int32(0))
                    (misc + 14).store(Int32(0))
                    (misc + 15).store(Int32(0))
                    (misc + 16).store(Int32(0))
                    if cutlass.const_expr(self.pdec):
                        if k_ == Int32(0):
                            (misc + 60).store(Int32(0))
                            if cutlass.const_expr(self.pdirect):
                                (misc + 27).store(Int32(0))
                if warp == Int32(0):
                    if prims.elect_sync():
                        prims.mbarrier_init(bars + B_QF, 1)
                        for i in cutlass.range_constexpr(NSLOTS):
                            if cutlass.const_expr(self.rs24_lsu and 4 <= i < 8):
                                # g = 1: tile 4's slots also count the control WG's cp.async arrivals
                                prims.mbarrier_init(
                                    bars + (B_KVF + i),
                                    cutlass.select_(
                                        (d & Int32(1)) != Int32(0), Int32(1 + A.LSU_ARR), Int32(1)
                                    ),
                                )
                            else:
                                prims.mbarrier_init(bars + (B_KVF + i), 1)
                            prims.mbarrier_init(bars + (B_KVE + i), 1)
                        for i in cutlass.range_constexpr(4):
                            prims.mbarrier_init(bars + (B_SF + i), 1)
                            prims.mbarrier_init(bars + (B_SE + i), 1)
                            prims.mbarrier_init(bars + (B_PF + i), 4)
                        prims.mbarrier_init(bars + B_OF, 1)
                        prims.mbarrier_init(bars + B_OC, 4)
                        prims.mbarrier_init(bars + B_FIN, 4)
                        prims.mbarrier_init(bars + B_SWA, 128)
                        prims.mbarrier_init(bars + B_SEL, 128)
                        prims.mbarrier_init(bars + B_T, 1)
                        prims.mbarrier_init(bars + B_PVL, 1)
                        prims.mbarrier_init(bars + B_BTC, 1)
                        prims.mbarrier_init(bars + B_LST, 1)
                        prims.mbarrier_init(bars + B_EAR, 128)
                        if cutlass.const_expr(self.rs24):
                            prims.mbarrier_init(bars + B_RX, 1)
                            prims.mbarrier_init(bars + B_PR, 1)
                            prims.mbarrier_init(bars + B_DRY, 1)
                            prims.mbarrier_arrive_expect_tx(
                                bars + B_RX, A.rs24_rx_bytes(self.rs24_f32 or self.rs24_asym)
                            )
                            prims.mbarrier_arrive_expect_tx(
                                bars + B_DRY, A.rs24_rx_bytes(self.rs24_f32 or self.rs24_asym)
                            )
                prims.fence_mbarrier_init()
                prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                if cutlass.const_expr(self.pdec):
                    if k_ == Int32(0):
                        prims.barrier_cta_sync(0)
                    else:
                        L.bar(BAR_PASS, 384)
                else:
                    prims.barrier_cta_sync(0)
                if cutlass.const_expr(self.pdec):
                    if k_ > Int32(0):
                        if cutlass.const_expr(self.pdirect):
                            if tidx == Int32(0):
                                # this pass started (mbarriers re-initialized): a pre-select may join it
                                sts_volatile_u32(
                                    Int32(
                                        cute.recast_ptr(
                                            arena + T_OFF_MISC + 27 * 4, dtype=Int32
                                        ).toint()
                                    ),
                                    Int32(1),
                                )
                        L.bar_arrive(BAR_HAND, 256)
                if cutlass.const_expr(self.rs24):
                    # phase 0 of the cluster barrier: arrive now (the mbarrier inits are fenced above); every thread
                    # waits only
                    # before its first remote operation (softmax WG: real merge; control WG: B_PR arrive; others: at
                    # exit)
                    cute.arch.cluster_arrive_relaxed()
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                tmem = misc.load()
                scale_log2 = softmax_scale * Float32(A.LOG2_E)
                tr = None
                if cutlass.const_expr(self.stamps):
                    tr = mStamps.iterator.toint() + Int64(cta) * Int64(NSTAMP * 8)
                swa_len = (pw + (Int32(7) + k_ * Int32(8))).load()
                n = (pw + (Int32(6) + k_ * Int32(8))).load()
                if n < Int32(0) or n > Int32(N_MAX):
                    n = Int32(0)
                n_sel = cutlass.min(n, Int32(K_TOP))
                if k_ == Int32(0):
                    if cutlass.const_expr(self.fblka):
                        self._block_a_lead(
                            arena, cta, tidx, ws, n, mBoard, mStamps, gw1, gw2, gw3, mCtl, d, mHist
                        )
                    else:
                        # One call site for the whole team: the followers run the leader's threshold code once, dry (one
                        # stale board read, no wait, no publish), at kernel start, so the leader's real pass when the
                        # board
                        # completes fetches its code from L2. Followers then read the leader's T line as before.
                        if cutlass.const_expr(self.tsz01 and self.nteams < self.nrows):
                            # rows 0-1 come from the stream T server: every team CTA follows its row's T line
                            self._follow_tline(
                                arena, cta, d, tidx, ws, mStamps, gw1, gw2, gw3, mCtl
                            )
                        elif cutlass.const_expr(self.t1a and self.nteams < self.nrows):
                            # team A's lead computes rows 0 and 1 here (row 1's pass runs warm code); team B
                            # computes nothing; every CTA then follows its row's T line (team A's lead: row 0 again)
                            npa_ = Int32(1)
                            if tm != Int32(0):
                                npa_ = Int32(0)
                            elif d == Int32(0):
                                npa_ = Int32(2)
                            for pa_ in cutlass.range(npa_, unroll=1):
                                rra_ = row_ + pa_
                                wsa_ = ws0 + Int64(rra_) * Int64(rs_["ws"] * 8)
                                na_ = n
                                g1a_ = gw1
                                g2a_ = gw2
                                g3a_ = gw3
                                if pa_ != Int32(0):
                                    na_ = Int32(_flat(mDescA, rra_, rs_["desc"])[2])
                                    if na_ < Int32(0) or na_ > Int32(N_MAX):
                                        na_ = Int32(0)
                                    if tidx == Int32(0):
                                        g1a_, g2a_, g3a_ = self._entry_loads(wsa_)
                                msa_ = cute.make_tensor(
                                    mStampsA.iterator
                                    + cutlass.select_(pa_ != Int32(0), Int32(2), k_)
                                    * Int32(self.num_sms * NSTAMP),
                                    mStampsA.layout,
                                )
                                self._block_a_lead(
                                    arena,
                                    cta,
                                    tidx,
                                    wsa_,
                                    na_,
                                    _flat(mBoardA, rra_, rs_["board"]),
                                    msa_,
                                    g1a_,
                                    g2a_,
                                    g3a_,
                                    _flat(mCtlA, rra_, rs_["ctl"]),
                                    d,
                                    _flat(mHistA, rra_, rs_["hist"]),
                                )
                            self._follow_tline(
                                arena, cta, d, tidx, ws, mStamps, gw1, gw2, gw3, mCtl
                            )
                        else:
                            npa_ = Int32(1)
                            if cutlass.const_expr(self.awarm):
                                if d == Int32(0):
                                    npa_ = Int32(2)
                            if cutlass.const_expr(self.awarm2):
                                if tm != Int32(0) and d == Int32(0):
                                    npa_ = Int32(2)
                            if cutlass.const_expr(self.awarm2a):
                                if d == Int32(0):
                                    npa_ = Int32(2)
                            for pa_ in cutlass.range(npa_, unroll=1):
                                # the lead runs this code once dry first (as a follower: no wait, no publish)
                                de_ = d
                                if pa_ + Int32(1) < npa_:
                                    de_ = Int32(1)
                                if cutlass.const_expr(self.awarm2 or self.awarm2a):
                                    # ... and without board loads
                                    dr_ = Int32(0)
                                    if pa_ + Int32(1) < npa_:
                                        dr_ = Int32(1)
                                    self._block_a_lead(
                                        arena,
                                        cta,
                                        tidx,
                                        ws,
                                        n,
                                        mBoard,
                                        mStamps,
                                        gw1,
                                        gw2,
                                        gw3,
                                        mCtl,
                                        de_,
                                        mHist,
                                        dry_=dr_,
                                    )
                                else:
                                    self._block_a_lead(
                                        arena,
                                        cta,
                                        tidx,
                                        ws,
                                        n,
                                        mBoard,
                                        mStamps,
                                        gw1,
                                        gw2,
                                        gw3,
                                        mCtl,
                                        de_,
                                        mHist,
                                    )
                            if d != Int32(0):
                                self._follow_tline(
                                    arena, cta, d, tidx, ws, mStamps, gw1, gw2, gw3, mCtl
                                )
                else:
                    # this row's T line was computed in the previous pass (misc slots + 13)
                    if tidx == Int32(0):
                        (misc + 4).store((misc + 17).load())
                        (misc + 5).store((misc + 18).load())
                        (misc + 8).store((misc + 21).load())
                        (misc + 9).store((misc + 22).load())
                        (misc + 10).store((misc + 23).load())
                        (misc + 11).store((misc + 24).load())
                        (misc + 13).store((misc + 26).load())
                cute.arch.fence_proxy("async.shared", space="cta")
                L.bar(BAR_SMX, 128)
                if tidx == Int32(0):
                    prims.mbarrier_arrive(bars + B_T)
                nx2_ = Int32(-1)
                if cutlass.const_expr(self.tsz2):
                    # row nrows - 2's T line comes from the stream T server
                    nx2_ = Int32(self.nrows - 2)
                if (
                    k_ + Int32(1) < npass_
                    and row_ + Int32(self.nteams) != Int32(self.nrows - 1)
                    and row_ + Int32(self.nteams) != nx2_
                    and Int32(1 if self.tszall else 0) == Int32(0)
                ):
                    # the next pass's T line now (its board completes while this row is selected); the lead
                    # also clears the next row's count histogram here, before its T word is released
                    kn_ = k_ + Int32(1)
                    rown_ = row_ + Int32(self.nteams)
                    wsn_ = ws0 + Int64(rown_) * Int64(rs_["ws"] * 8)
                    mStampsN_ = cute.make_tensor(
                        mStampsA.iterator + kn_ * Int32(self.num_sms * NSTAMP), mStampsA.layout
                    )
                    mCtlN_ = _flat(mCtlA, rown_, rs_["ctl"])
                    nn_ = (pw + (Int32(6) + kn_ * Int32(8))).load()
                    if nn_ < Int32(0) or nn_ > Int32(N_MAX):
                        nn_ = Int32(0)
                    gn1_ = Uint64(0)
                    gn2_ = Uint64(0)
                    gn3_ = Uint64(0)
                    if tidx == Int32(0):
                        gn1_ = Uint64(Uint32((pw + (Int32(0) + kn_ * Int32(8))).load())) | Uint64(
                            Uint32((pw + (Int32(1) + kn_ * Int32(8))).load())
                        ) << Uint64(32)
                        gn2_ = Uint64(Uint32((pw + (Int32(2) + kn_ * Int32(8))).load())) | Uint64(
                            Uint32((pw + (Int32(3) + kn_ * Int32(8))).load())
                        ) << Uint64(32)
                        gn3_ = Uint64(Uint32((pw + (Int32(4) + kn_ * Int32(8))).load())) | Uint64(
                            Uint32((pw + (Int32(5) + kn_ * Int32(8))).load())
                        ) << Uint64(32)
                        (misc + 26).store(Int32(0))
                    npb_ = Int32(1)
                    if cutlass.const_expr(self.awarm or self.awarm3):
                        if d == Int32(0):
                            npb_ = Int32(2)
                    for pb_ in cutlass.range(npb_, unroll=1):
                        # dry first on the lead (as a follower: no wait, no publish)
                        db_ = d
                        if pb_ + Int32(1) < npb_:
                            db_ = Int32(1)
                        if cutlass.const_expr(self.awarm3):
                            # ... and without board loads
                            dr_ = Int32(0)
                            if pb_ + Int32(1) < npb_:
                                dr_ = Int32(1)
                            self._block_a_lead(
                                arena,
                                cta,
                                tidx,
                                wsn_,
                                nn_,
                                _flat(mBoardA, rown_, rs_["board"]),
                                mStampsN_,
                                gn1_,
                                gn2_,
                                gn3_,
                                mCtlN_,
                                db_,
                                _flat(mHistA, rown_, rs_["hist"]),
                                mo=13,
                                dry_=dr_,
                            )
                        else:
                            self._block_a_lead(
                                arena,
                                cta,
                                tidx,
                                wsn_,
                                nn_,
                                _flat(mBoardA, rown_, rs_["board"]),
                                mStampsN_,
                                gn1_,
                                gn2_,
                                gn3_,
                                mCtlN_,
                                db_,
                                _flat(mHistA, rown_, rs_["hist"]),
                                mo=13,
                            )
                    if d != Int32(0):
                        self._follow_tline(
                            arena, cta, d, tidx, wsn_, mStampsN_, gn1_, gn2_, gn3_, mCtlN_, mo=13
                        )
                if cutlass.const_expr(self.gjoin):
                    if cutlass.const_expr(self.rs24):
                        zsm = Int32(
                            cute.recast_ptr(arena + T_OFF_MISC + 61 * 4, dtype=Int32).toint()
                        )
                        A.role_softmax_24(
                            tmem,
                            bars + B_SF,
                            bars + B_PF,
                            scale_log2,
                            swa_len,
                            n_sel,
                            d,
                            bars + B_OF,
                            mSinks.iterator.raw_ptr(),
                            mOut.iterator.toint(),
                            Int32(_ptr(arena, T_OFF_LIST, Int32).toint()),
                            Int32(_ptr(arena, T_OFF_STAT, Int32).toint()),
                            Int32((bars + B_PVL).toint()),
                            Int32((bars + B_RX).toint()),
                            Int32((bars + B_PR).toint()),
                            Int32((bars + B_DRY).toint()),
                            zsm,
                            tr,
                            full_tiles=self.full_attention_tiles,
                            dry_merge=self.rs24_dry,
                            f32x=self.rs24_f32,
                            asym=self.rs24_asym,
                        )
                    else:
                        A.role_softmax_team(
                            tmem,
                            bars + B_SF,
                            bars + B_PF,
                            s_stat,
                            s_fin,
                            bars + B_FIN,
                            scale_log2,
                            swa_len,
                            n_sel,
                            Int32(0),
                            Int32(NTILES),
                            Float32(A.NEG_FLT_MAX),
                            Float32(0.0),
                            True,
                            tr,
                            bars + B_OF,
                            d,
                            full_tiles=self.full_attention_tiles,
                        )
                    if cutlass.const_expr(self.late_done == 2):
                        if d == Int32(0):
                            if tidx == Int32(0):
                                mi_ = _sm_tensor(arena, T_OFF_MISC, Int32, 64)
                                if mi_[12] == Int32(P_T1) and mi_[14] != Int32(0):
                                    gq_ = Uint64(Uint32(mi_[5])) << Uint64(32) | Uint64(
                                        Uint32(mi_[4])
                                    )
                                    self._publish_dir(
                                        ws,
                                        gq_,
                                        Int32(1),
                                        Int32(K_DONE),
                                        Int32(P_T1),
                                        Uint64(0),
                                        Uint64(0),
                                        Uint64(0),
                                    )
                else:
                    A.role_softmax_team(
                        tmem,
                        bars + B_SF,
                        bars + B_PF,
                        s_stat,
                        s_fin,
                        bars + B_FIN,
                        scale_log2,
                        swa_len,
                        n_sel,
                        Int32(0),
                        Int32(NTILES),
                        Float32(A.NEG_FLT_MAX),
                        Float32(0.0),
                        True,
                        tr,
                        full_tiles=self.full_attention_tiles,
                    )
                prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                if cutlass.const_expr(self.pdec):
                    if k_ + Int32(1) < npass_:
                        L.bar(BAR_PASS, 384)
                    else:
                        prims.barrier_cta_sync(0)
                else:
                    prims.barrier_cta_sync(0)
        elif warp < Int32(8):
            if cutlass.const_expr(self.ctl_regs):
                prims.setmaxregister(A.REG_CTL, prims.SetMaxRegisterAction.INCREASE)
            for k_ in cutlass.range(npass_, unroll=1):
                row_ = tm + k_ * Int32(self.nteams)
                mStamps = cute.make_tensor(
                    mStampsA.iterator + k_ * Int32(self.num_sms * NSTAMP), mStampsA.layout
                )
                ws = ws0 + Int64(row_) * Int64(rs_["ws"] * 8)
                mOut = _row2(mOutA, row_)
                _mCtx = _row1(mCtxA, row_)
                mSwaLocal = _row2(mSwaLocalA, row_)
                mBtSwa = _row2(mBtSwaA, row_)
                mBtC = _row2(mBtCA, row_)
                _mSwaLens = _row1(mSwaLensA, row_)
                mTopk = _row2(mTopkA, row_)
                mGidx = _row2(mGidxA, row_)
                mHist = _flat(mHistA, row_, rs_["hist"])
                mBoard = _flat(mBoardA, row_, rs_["board"])
                mRegion = _flat(mRegionA, row_, rs_["region"])
                mCtl = _flat(mCtlA, row_, rs_["ctl"])
                _mDesc = _flat(mDescA, row_, rs_["desc"])
                qrow = row_
                prims.fence_mbarrier_init()
                prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                if cutlass.const_expr(self.pdec):
                    if k_ == Int32(0):
                        prims.barrier_cta_sync(0)
                    else:
                        # joins after the softmax WG's start barrier (mbarriers re-initialized)
                        if cutlass.const_expr(self.pdirect):
                            # a direct pre-select joined this pass already (misc[58] = 2)
                            if (misc + 58).load() != Int32(2):
                                L.bar(BAR_HAND, 256)
                        else:
                            L.bar(BAR_HAND, 256)
                else:
                    prims.barrier_cta_sync(0)
                if cutlass.const_expr(self.rs24):
                    # phase 0 of the cluster barrier: arrive now (the mbarrier inits are fenced above); every thread
                    # waits only
                    # before its first remote operation (softmax WG: real merge; control WG: B_PR arrive; others: at
                    # exit)
                    cute.arch.cluster_arrive_relaxed()
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                tmem = misc.load()
                scale_log2 = softmax_scale * Float32(A.LOG2_E)
                tr = None
                if cutlass.const_expr(self.stamps):
                    tr = mStamps.iterator.toint() + Int64(cta) * Int64(NSTAMP * 8)
                swa_len = (pw + (Int32(7) + k_ * Int32(8))).load()
                n = (pw + (Int32(6) + k_ * Int32(8))).load()
                if n < Int32(0) or n > Int32(N_MAX):
                    n = Int32(0)
                n_sel = cutlass.min(n, Int32(K_TOP))
                if cutlass.const_expr(self.rs24):
                    # job 0 = this pass's row (skipped when pre-selected), job 1 = the next pass's row (pre-select)
                    pok_ = Int32(0)
                    njb_ = Int32(1)
                    if cutlass.const_expr(self.presel):
                        if k_ > Int32(0):
                            pok_ = (misc + 58).load()
                        if k_ + Int32(1) < npass_:
                            njb_ = Int32(2)
                    for jb_ in cutlass.range(njb_, unroll=1):
                        if jb_ != Int32(0) or pok_ == Int32(0):
                            kj_ = k_ + jb_
                            rowj_ = row_ + jb_ * Int32(self.nteams)
                            nj_ = (pw + (Int32(6) + kj_ * Int32(8))).load()
                            if nj_ < Int32(0) or nj_ > Int32(N_MAX):
                                nj_ = Int32(0)
                            gj1_ = Uint64(
                                Uint32((pw + (Int32(0) + kj_ * Int32(8))).load())
                            ) | Uint64(Uint32((pw + (Int32(1) + kj_ * Int32(8))).load())) << Uint64(
                                32
                            )
                            gj2_ = Uint64(
                                Uint32((pw + (Int32(2) + kj_ * Int32(8))).load())
                            ) | Uint64(Uint32((pw + (Int32(3) + kj_ * Int32(8))).load())) << Uint64(
                                32
                            )
                            gj3_ = Uint64(
                                Uint32((pw + (Int32(4) + kj_ * Int32(8))).load())
                            ) | Uint64(Uint32((pw + (Int32(5) + kj_ * Int32(8))).load())) << Uint64(
                                32
                            )
                            mStampsJ_ = cute.make_tensor(
                                mStampsA.iterator + kj_ * Int32(self.num_sms * NSTAMP),
                                mStampsA.layout,
                            )
                            s_idxj_ = _ptr(
                                arena,
                                cutlass.select_(
                                    jb_ == Int32(0), Int32(T_OFF_IDX), Int32(SIDX2_OFF)
                                ),
                                Int32,
                            )
                            p0j_ = Int32(1)
                            if k_ == Int32(0) and jb_ == Int32(0):
                                p0j_ = Int32(0 if self.prewarm else 1)
                                if cutlass.const_expr(self.warm1):
                                    if tm != Int32(0):
                                        p0j_ = Int32(1)
                            self._control_s1d(
                                arena,
                                cta,
                                d,
                                tidx - Int32(128),
                                ws0 + Int64(rowj_) * Int64(rs_["ws"] * 8),
                                nj_,
                                s_idxj_,
                                bars,
                                _row2(mSwaLocalA, rowj_),
                                _row2(mBtSwaA, rowj_),
                                _row2(mBtCA, rowj_),
                                _row2(mTopkA, rowj_),
                                _row2(mGidxA, rowj_),
                                _flat(mRegionA, rowj_, rs_["region"]),
                                _flat(mHistA, rowj_, rs_["hist"]),
                                mStampsJ_,
                                _flat(mCtlA, rowj_, rs_["ctl"]),
                                None,
                                None,
                                p0=p0j_,
                                pre_=jb_,
                                gwn=(gj1_, gj2_, gj3_),
                            )
                        if jb_ == Int32(0):
                            if pok_ == Int32(1):
                                self._shipped_sel(
                                    arena,
                                    cta,
                                    d,
                                    tidx - Int32(128),
                                    ws,
                                    s_idx,
                                    bars,
                                    mStamps,
                                    n,
                                    mBtC,
                                )
                            if cutlass.const_expr(self.rs24_lsu):
                                if (d & Int32(1)) != Int32(0):
                                    # tile 4's rows were published by this warpgroup before its B_SEL arrive (named
                                    # barrier); issue
                                    # before the cluster wait / B_PR arrive (the partner pushes into slot 8 only at its
                                    # merge)
                                    A.lsu_tile4(
                                        tidx - Int32(128),
                                        d,
                                        self.rs24_lsu,
                                        Int32(s_kv.toint()),
                                        Int32(s_idx.toint()),
                                        Int32((bars + B_KVF).toint()),
                                        bars + B_KVE,
                                        mCPool.iterator.toint(),
                                        tr,
                                    )
                                    if cutlass.const_expr(self.stamps):
                                        A._stamp(
                                            mStamps.iterator.toint()
                                            + Int64(cta) * Int64(NSTAMP * 8),
                                            Int32(52),
                                            tidx == Int32(128),
                                        )
                            cute.arch.cluster_wait()
                            if tidx == Int32(160):
                                # (not ctid 0: on team CTA 0 it may still be in the late-done follower wait)
                                B.mbar_arrive_remote(
                                    B.mapa_u32(
                                        Int32((bars + B_PR).toint()), B.cluster_ctarank() ^ Int32(1)
                                    )
                                )
                elif cutlass.const_expr(self.gjoin):
                    self._control_s1d(
                        arena,
                        cta,
                        d,
                        tidx - Int32(128),
                        ws,
                        n,
                        s_idx,
                        bars,
                        mSwaLocal,
                        mBtSwa,
                        mBtC,
                        mTopk,
                        mGidx,
                        mRegion,
                        mHist,
                        mStamps,
                        mCtl,
                        (
                            tma_swa,
                            tma_cmp,
                            s_kv,
                            s_ext,
                            s_idx,
                            bars + B_KVF,
                            bars + B_KVE,
                            bars + B_SWA,
                            bars + B_SEL,
                            Int32(A.NGW) + warp - Int32(4),
                            d,
                            NTILES,
                            NSLOTS,
                            None,
                            Int32(2),
                            A.NGW + 4,
                            bars + B_EAR,
                            2,
                        ),
                        (
                            tmem,
                            bars + B_PF,
                            bars + B_OF,
                            bars + B_OC,
                            s_stat,
                            s_fin,
                            bars + B_FIN,
                            mSinks.iterator.raw_ptr(),
                            mOut.iterator.toint(),
                            scale_log2,
                            d,
                            Int32(s_kv.toint()),
                            NTILES,
                            BAR_EPI,
                            tr,
                            bars + B_PVL,
                        ),
                    )
                else:
                    self._control_s1d(
                        arena,
                        cta,
                        d,
                        tidx - Int32(128),
                        ws,
                        n,
                        s_idx,
                        bars,
                        mSwaLocal,
                        mBtSwa,
                        mBtC,
                        mTopk,
                        mGidx,
                        mRegion,
                        mHist,
                        mStamps,
                        mCtl,
                    )
                    A.role_corr_epi_team(
                        tmem,
                        bars + B_PF,
                        bars + B_OF,
                        bars + B_OC,
                        s_stat,
                        s_fin,
                        bars + B_FIN,
                        mSinks.iterator.raw_ptr(),
                        mOut.iterator.toint(),
                        scale_log2,
                        d,
                        Int32(s_kv.toint()),
                        NTILES,
                        BAR_EPI,
                        tr,
                    )
                # B_BTC is re-initialized next pass: its bulk copy must have landed (abort paths skip the wait)
                if tidx == Int32(128):
                    A._wait(bars + B_BTC, Int32(0))
                prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                if cutlass.const_expr(self.pdec):
                    if k_ + Int32(1) == npass_:
                        prims.barrier_cta_sync(0)
                else:
                    prims.barrier_cta_sync(0)
        elif warp == Int32(A.W_MMA):
            prims.setmaxregister(A.REG_OTHER, prims.SetMaxRegisterAction.DECREASE)
            for k_ in cutlass.range(npass_, unroll=1):
                row_ = tm + k_ * Int32(self.nteams)
                mStamps = cute.make_tensor(
                    mStampsA.iterator + k_ * Int32(self.num_sms * NSTAMP), mStampsA.layout
                )
                ws = ws0 + Int64(row_) * Int64(rs_["ws"] * 8)
                mOut = _row2(mOutA, row_)
                _mCtx = _row1(mCtxA, row_)
                mSwaLocal = _row2(mSwaLocalA, row_)
                mBtSwa = _row2(mBtSwaA, row_)
                mBtC = _row2(mBtCA, row_)
                _mSwaLens = _row1(mSwaLensA, row_)
                mTopk = _row2(mTopkA, row_)
                mGidx = _row2(mGidxA, row_)
                mHist = _flat(mHistA, row_, rs_["hist"])
                mBoard = _flat(mBoardA, row_, rs_["board"])
                mRegion = _flat(mRegionA, row_, rs_["region"])
                mCtl = _flat(mCtlA, row_, rs_["ctl"])
                _mDesc = _flat(mDescA, row_, rs_["desc"])
                qrow = row_
                prims.fence_mbarrier_init()
                prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                if cutlass.const_expr(self.pdec):
                    if k_ == Int32(0):
                        prims.barrier_cta_sync(0)
                    else:
                        L.bar(BAR_PASS, 384)
                else:
                    prims.barrier_cta_sync(0)
                if cutlass.const_expr(self.rs24):
                    # phase 0 of the cluster barrier: arrive now (the mbarrier inits are fenced above); every thread
                    # waits only
                    # before its first remote operation (softmax WG: real merge; control WG: B_PR arrive; others: at
                    # exit)
                    cute.arch.cluster_arrive_relaxed()
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                tmem = misc.load()
                scale_log2 = softmax_scale * Float32(A.LOG2_E)
                tr = None
                if cutlass.const_expr(self.stamps):
                    tr = mStamps.iterator.toint() + Int64(cta) * Int64(NSTAMP * 8)
                swa_len = (pw + (Int32(7) + k_ * Int32(8))).load()
                n = (pw + (Int32(6) + k_ * Int32(8))).load()
                if n < Int32(0) or n > Int32(N_MAX):
                    n = Int32(0)
                n_sel = cutlass.min(n, Int32(K_TOP))
                if cutlass.const_expr(self.gjoin):
                    if cutlass.const_expr(self.rs24):
                        A.role_mma_24(
                            tmem,
                            s_q,
                            s_kv,
                            s_ext,
                            bars + B_QF,
                            bars + B_KVF,
                            bars + B_KVE,
                            bars + B_SF,
                            bars + B_PF,
                            bars + B_OF,
                            d,
                            bars + B_PVL,
                            tr,
                            pfence=self.rs24_lsu > 0,
                        )
                    else:
                        A.role_mma_ordered(
                            tmem,
                            s_q,
                            s_kv,
                            s_ext,
                            bars + B_QF,
                            bars + B_KVF,
                            bars + B_KVE,
                            bars + B_SF,
                            bars + B_SE,
                            bars + B_PF,
                            bars + B_OF,
                            d,
                            NTILES,
                            NSLOTS,
                            bars + B_PVL,
                            tr,
                            bars + B_EAR if self.mma_block else None,
                            self.mma_idle_ns,
                        )
                else:
                    A.role_mma_team(
                        tmem,
                        s_q,
                        s_kv,
                        s_ext,
                        bars + B_QF,
                        bars + B_KVF,
                        bars + B_KVE,
                        bars + B_SF,
                        bars + B_SE,
                        bars + B_PF,
                        bars + B_OC,
                        bars + B_OF,
                        d,
                        NTILES,
                        NSLOTS,
                        tr,
                        self.qk_ahead,
                    )
                if k_ + Int32(1) < npass_:
                    nx2m_ = Int32(-1)
                    if cutlass.const_expr(self.tsz2):
                        nx2m_ = Int32(self.nrows - 2)
                    if (
                        row_ + Int32(self.nteams) == Int32(self.nrows - 1)
                        or row_ + Int32(self.nteams) == nx2m_
                        or Int32(1 if self.tszall else 0) != Int32(0)
                    ):
                        # the last row's T line (stream T server), read now by lane 0 into misc + 13
                        if cute.arch.lane_idx() == Int32(0):
                            kn_ = k_ + Int32(1)
                            rown_ = row_ + Int32(self.nteams)
                            mStampsN_ = cute.make_tensor(
                                mStampsA.iterator + kn_ * Int32(self.num_sms * NSTAMP),
                                mStampsA.layout,
                            )
                            gn1_ = Uint64(
                                Uint32((pw + (Int32(0) + kn_ * Int32(8))).load())
                            ) | Uint64(Uint32((pw + (Int32(1) + kn_ * Int32(8))).load())) << Uint64(
                                32
                            )
                            gn2_ = Uint64(
                                Uint32((pw + (Int32(2) + kn_ * Int32(8))).load())
                            ) | Uint64(Uint32((pw + (Int32(3) + kn_ * Int32(8))).load())) << Uint64(
                                32
                            )
                            gn3_ = Uint64(
                                Uint32((pw + (Int32(4) + kn_ * Int32(8))).load())
                            ) | Uint64(Uint32((pw + (Int32(5) + kn_ * Int32(8))).load())) << Uint64(
                                32
                            )
                            (misc + 26).store(Int32(0))
                            self._follow_tline1(
                                arena,
                                cta,
                                d,
                                Int32(0),
                                ws0 + Int64(rown_) * Int64(rs_["ws"] * 8),
                                mStampsN_,
                                gn1_,
                                gn2_,
                                gn3_,
                                _flat(mCtlA, rown_, rs_["ctl"]),
                                mo=13,
                            )
                prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                if cutlass.const_expr(self.pdec):
                    if k_ + Int32(1) < npass_:
                        L.bar(BAR_PASS, 384)
                    else:
                        prims.barrier_cta_sync(0)
                else:
                    prims.barrier_cta_sync(0)
                if cutlass.const_expr(self.rs24):
                    if k_ + Int32(1) < npass_:
                        # this warp never waited on the cluster barrier in the pass: wait before the next arrive
                        cute.arch.cluster_wait()
        else:
            prims.setmaxregister(A.REG_OTHER, prims.SetMaxRegisterAction.DECREASE)
            for k_ in cutlass.range(npass_, unroll=1):
                row_ = tm + k_ * Int32(self.nteams)
                mStamps = cute.make_tensor(
                    mStampsA.iterator + k_ * Int32(self.num_sms * NSTAMP), mStampsA.layout
                )
                ws = ws0 + Int64(row_) * Int64(rs_["ws"] * 8)
                mOut = _row2(mOutA, row_)
                _mCtx = _row1(mCtxA, row_)
                mSwaLocal = _row2(mSwaLocalA, row_)
                mBtSwa = _row2(mBtSwaA, row_)
                mBtC = _row2(mBtCA, row_)
                _mSwaLens = _row1(mSwaLensA, row_)
                mTopk = _row2(mTopkA, row_)
                mGidx = _row2(mGidxA, row_)
                mHist = _flat(mHistA, row_, rs_["hist"])
                mBoard = _flat(mBoardA, row_, rs_["board"])
                mRegion = _flat(mRegionA, row_, rs_["region"])
                mCtl = _flat(mCtlA, row_, rs_["ctl"])
                _mDesc = _flat(mDescA, row_, rs_["desc"])
                qrow = row_
                prims.fence_mbarrier_init()
                prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                if cutlass.const_expr(self.pdec):
                    if k_ == Int32(0):
                        prims.barrier_cta_sync(0)
                    else:
                        L.bar(BAR_PASS, 384)
                else:
                    prims.barrier_cta_sync(0)
                if cutlass.const_expr(self.rs24):
                    # phase 0 of the cluster barrier: arrive now (the mbarrier inits are fenced above); every thread
                    # waits only
                    # before its first remote operation (softmax WG: real merge; control WG: B_PR arrive; others: at
                    # exit)
                    cute.arch.cluster_arrive_relaxed()
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                tmem = misc.load()
                scale_log2 = softmax_scale * Float32(A.LOG2_E)
                tr = None
                if cutlass.const_expr(self.stamps):
                    tr = mStamps.iterator.toint() + Int64(cta) * Int64(NSTAMP * 8)
                swa_len = (pw + (Int32(7) + k_ * Int32(8))).load()
                n = (pw + (Int32(6) + k_ * Int32(8))).load()
                if n < Int32(0) or n > Int32(N_MAX):
                    n = Int32(0)
                n_sel = cutlass.min(n, Int32(K_TOP))
                if warp == Int32(A.W_G0):
                    A.role_q_load(tma_q, s_q, bars + B_QF, qrow)
                if warp == Int32(A.W_G0 + 1):
                    if prims.elect_sync():
                        btc_cols = cute.size(mBtC, mode=[1])
                        npg = cutlass.min(
                            (n + Int32(31)) // Int32(32), cutlass.min(btc_cols, Int32(BTC_CAP))
                        )
                        nbt = npg * Int32(4) // Int32(16) * Int32(16)
                        if nbt > Int32(0):
                            B.mbar_arrive_expect_tx(Int32((bars + B_BTC).toint()), nbt)
                            B.bulk_g2s(
                                Int32(cute.recast_ptr(arena + T_OFF_BTC, dtype=Int32).toint()),
                                mBtC.iterator.toint(),
                                nbt,
                                Int32((bars + B_BTC).toint()),
                            )
                        else:
                            prims.mbarrier_arrive(bars + B_BTC)
                pfb_ = Int32(1)
                if cutlass.const_expr(self.presel):
                    if k_ > Int32(0):
                        pfb_ = Int32(0)
                if warp == Int32(A.W_G0 + 2) and pfb_ != Int32(0):
                    rgb = mRegion.iterator.toint()
                    for q in cutlass.range_constexpr(NB_INITIAL // 32):
                        bha = self._bucket_addr(rgb, cute.arch.lane_idx() + Int32(q * 32), Int32(0))
                        for h_ in cutlass.range_constexpr(4):
                            L.prefetch_l2(bha + Int64(128 * h_))
                if cutlass.const_expr(self.swag):
                    if warp >= Int32(A.W_G0 + 3):
                        gt_ = (warp - Int32(A.W_G0 + 3)) * Int32(32) + cute.arch.lane_idx()
                        mbs_ = cute.size(mBtSwa, mode=[1])
                        pks_ = Int32(0)
                        if cutlass.const_expr(self.presel):
                            if k_ > Int32(0):
                                if cutlass.const_expr(self.pdec):
                                    pks_ = lds_volatile_u32(
                                        Int32(
                                            cute.recast_ptr(
                                                arena + T_OFF_MISC + 60 * 4, dtype=Int32
                                            ).toint()
                                        )
                                    )
                                else:
                                    pks_ = (misc + 58).load()
                        rsw_ = Int32(0)
                        if pks_ != Int32(0):
                            # SWA rows prepared by the previous pass's pre-select
                            if cutlass.const_expr(self.pdec):
                                fence_acq_rel_cta()
                            rsw_ = (_ptr(arena, SIDX2_OFF, Int32) + gt_).load()
                        else:
                            loc_ = mSwaLocal[0, gt_]
                            rsw_ = L.l2g_swa_local(loc_, Int32(0), mBtSwa, mbs_, Int32(0))
                        (s_idx + gt_).store(rsw_)
                        # each writer releases its own index store to the gather warps
                        prims.mbarrier_arrive(bars + B_SWA)
                if cutlass.const_expr(self.rs24):
                    A.role_gather_24(
                        tma_swa,
                        tma_cmp,
                        s_kv,
                        s_ext,
                        s_idx,
                        bars + B_KVF,
                        bars + B_KVE,
                        bars + B_SWA,
                        bars + B_EAR,
                        bars + B_SEL,
                        warp - Int32(A.W_G0),
                        d,
                        NTILES,
                        tr,
                        (misc + 16) if self.rs24_t4split else None,
                        lsu_g=self.rs24_lsu,
                    )
                else:
                    A.role_gather_team(
                        tma_swa,
                        tma_cmp,
                        s_kv,
                        s_ext,
                        s_idx,
                        bars + B_KVF,
                        bars + B_KVE,
                        bars + B_SWA,
                        bars + B_SEL,
                        warp - Int32(A.W_G0),
                        d,
                        NTILES,
                        NSLOTS,
                        tr,
                        Int32(0),
                        A.NGW + 4 if self.gjoin else A.NGW,
                        bars + B_EAR if self.gjoin else None,
                        2 if self.gjoin else 1,
                    )
                if cutlass.const_expr(self.hook):
                    if warp == Int32(A.W_G0 + A.NGW - 1):
                        if cute.arch.lane_idx() == Int32(0):
                            self._stamp(mStamps, cta, 106)
                            self._team_exit(arena, d, ws)
                            self._stamp(mStamps, cta, 107)
                prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                if cutlass.const_expr(self.pdec):
                    if k_ + Int32(1) < npass_:
                        L.bar(BAR_PASS, 384)
                    else:
                        prims.barrier_cta_sync(0)
                else:
                    prims.barrier_cta_sync(0)
                if cutlass.const_expr(self.rs24):
                    if k_ + Int32(1) < npass_:
                        # this warp never waited on the cluster barrier in the pass: wait before the next arrive
                        cute.arch.cluster_wait()
        if warp == Int32(A.W_MMA):
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            prims.tcgen05_dealloc(
                prims.make_tmem_ptr(misc.load(), Float32), A.TMEM_COLS, group=prims.CTAGroup.CTA_1
            )
        if cutlass.const_expr(self.stamps):
            if tidx == Int32(0):
                (mStampsA.iterator + cta * NSTAMP + ST_M_EXIT).store(Int64(L.gtimer()))

    @cute.jit
    def _team_exit(self, arena, d, ws):
        misc = _sm_tensor(arena, T_OFF_MISC, Int32, 64)
        gen = Uint64(Uint32(misc[5])) << Uint64(32) | Uint64(Uint32(misc[4]))
        t64 = gen + Uint64(1)
        ab = misc[13]
        if d == Int32(0):
            tdl = self._deadline()
            st_ = Int32(0)
            while st_ == Int32(0):
                ready = self._followers_ready(ws, t64)
                av = L.ld_relaxed_u64(self._w(ws, W_ABORT))
                if ready != Int32(0):
                    st_ = Int32(1)
                elif av == t64:
                    st_ = Int32(2)
                elif Int64(L.gtimer()) > tdl:
                    st_ = Int32(3)
            if st_ == Int32(3):
                self._raise_abort(ws, gen, Int32(E_TIMEOUT))
            elif st_ == Int32(2):
                self._flag(ws, gen, Int32(E_ABORTED))
            if st_ != Int32(1):
                ab = Int32(1)
            if cutlass.const_expr(self.late_done == 1):
                if ab == Int32(0):
                    if misc[12] == Int32(P_T1):
                        self._publish_dir(
                            ws,
                            gen,
                            Int32(1),
                            Int32(K_DONE),
                            Int32(P_T1),
                            Uint64(0),
                            Uint64(0),
                            Uint64(0),
                        )
            nxt = gen + Uint64(1)
            if ab != Int32(0):
                nxt = gen + Uint64(2)
            st_rlx_u64(self._w(ws, W_DONE), nxt)
            st_rlx_u64(self._w(ws, W_DONE2), nxt)
        else:
            t40 = self._tag40(gen)
            tdl = self._deadline()
            st_ = Int32(0)
            h = Uint64(0)
            while st_ == Int32(0):
                h = L.ld_relaxed_u64(self._w(ws, W_DIR))
                av = L.ld_relaxed_u64(self._w(ws, W_ABORT))
                if h >> Uint64(24) == t40 and Int32(Uint32(h & Uint64(15))) == Int32(K_DONE):
                    st_ = Int32(1)
                elif av == t64:
                    st_ = Int32(2)
                elif Int64(L.gtimer()) > tdl:
                    st_ = Int32(3)
            if st_ == Int32(3):
                self._raise_abort(ws, gen, Int32(E_TIMEOUT))
            elif st_ == Int32(2):
                self._flag(ws, gen, Int32(E_ABORTED))
            if st_ == Int32(1):
                if Int32(Uint32(h >> Uint64(4) & Uint64(15))) != misc[12]:
                    self._flag(ws, gen, Int32(E_MISMATCH))

    @cute.jit
    def _scan4(self, v0, v1, v2, v3, ctid, arena):
        NT: cutlass.Constexpr[int] = 128
        s_wc = _sm_tensor(arena, T_OFF_WC, Int32, 64)
        lane = ctid % Int32(32)
        wid = ctid // Int32(32)
        i0 = L.warp_incl_scan_add(v0, lane)
        i1 = L.warp_incl_scan_add(v1, lane)
        i2 = L.warp_incl_scan_add(v2, lane)
        i3 = L.warp_incl_scan_add(v3, lane)
        if lane == Int32(31):
            s_wc[wid * Int32(4)] = i0
            s_wc[wid * Int32(4) + Int32(1)] = i1
            s_wc[wid * Int32(4) + Int32(2)] = i2
            s_wc[wid * Int32(4) + Int32(3)] = i3
        L.bar(BAR_CTL, NT)
        b0 = Int32(0)
        b1 = Int32(0)
        b2 = Int32(0)
        b3 = Int32(0)
        t0 = Int32(0)
        t1 = Int32(0)
        t2 = Int32(0)
        t3 = Int32(0)
        for w_ in cutlass.range_constexpr(NT // 32):
            c0 = s_wc[4 * w_]
            c1 = s_wc[4 * w_ + 1]
            c2 = s_wc[4 * w_ + 2]
            c3 = s_wc[4 * w_ + 3]
            if Int32(w_) < wid:
                b0 = b0 + c0
                b1 = b1 + c1
                b2 = b2 + c2
                b3 = b3 + c3
            t0 = t0 + c0
            t1 = t1 + c1
            t2 = t2 + c2
            t3 = t3 + c3
        L.bar(BAR_CTL, NT)
        return (b0 + i0 - v0, b1 + i1 - v1, b2 + i2 - v2, b3 + i3 - v3, t0, t1, t2, t3)

    @cute.jit
    def _band_range(self, T, HIC, SMAX):
        lo = T
        hi = HIC
        if not hi > lo:
            hi = SMAX
        if not hi > lo:
            hi = lo + Float32(1.0)
        return (lo, hi)

    @cute.jit
    def _wg_any(self, v, ctid, s_red, bar_id: cutlass.Constexpr[int]):
        lane = ctid % Int32(32)
        wid = ctid // Int32(32)
        w = Int32(0)
        bm = Int32(cute.arch.vote_ballot_sync(v != Int32(0)))
        if bm != Int32(0):
            w = Int32(1)
        if lane == Int32(0):
            s_red[Int32(40) + wid] = Uint32(w)
        L.bar(bar_id, 128)
        r = Int32(s_red[40] | s_red[41] | s_red[42] | s_red[43])
        L.bar(bar_id, 128)
        return r

    @cute.jit
    def _warm_abort(self, real, status, ctid, bars, misc):
        # the warm-up pass runs to completion: the producers' counts arrive well after T is known, so finishing
        # the synthetic emission / rank / epilogue code first keeps all of it cache-resident for the real pass
        return status

    @cute.jit
    def _follow_tline(
        self, arena, cta, d, ctid, ws, mStamps, gw1, gw2, gw3, mCtl, mo: cutlass.Constexpr[int] = 0
    ):
        misc = _sm_tensor(arena, T_OFF_MISC, Int32, 64)
        if ctid == Int32(0):
            gen = self._gen_from(gw1, gw2, gw3)
            misc[4 + mo] = Int32(Uint32(gen & Uint64(4294967295)))
            misc[5 + mo] = Int32(Uint32(gen >> Uint64(32)))
            tg = self._tag32(gen)
            t64 = gen + Uint64(1)
            skipf = Int32(0)
            if cutlass.const_expr(self.debug):
                if mCtl[1] == Int32(4) and d == Int32(2):
                    skipf = Int32(1)
            if skipf == Int32(0):
                st_rlx_u64(self._w(ws, W_FOL) + Int64(d) * Int64(8), t64)
            tdl = self._deadline()
            st_ = Int32(0)
            l0 = Uint64(0)
            l1 = Uint64(0)
            l2 = Uint64(0)
            l3 = Uint64(0)
            while st_ == Int32(0):
                l0 = L.ld_relaxed_u64(self._w(ws, W_TLINE))
                l1 = L.ld_relaxed_u64(self._w(ws, W_TLINE + 1))
                l2 = L.ld_relaxed_u64(self._w(ws, W_TLINE + 2))
                l3 = L.ld_relaxed_u64(self._w(ws, W_TLINE + 3))
                av = L.ld_relaxed_u64(self._w(ws, W_ABORT))
                if (
                    Uint32(l0 >> Uint64(32)) == tg
                    and Uint32(l1 >> Uint64(32)) == tg
                    and (Uint32(l2 >> Uint64(32)) == tg)
                    and (Uint32(l3 >> Uint64(32)) == tg)
                ):
                    st_ = Int32(1)
                elif av == t64:
                    st_ = Int32(2)
                elif Int64(L.gtimer()) > tdl:
                    st_ = Int32(3)
            if st_ == Int32(3):
                self._raise_abort(ws, gen, Int32(E_TIMEOUT))
            if st_ == Int32(1):
                misc[8 + mo] = Int32(Uint32(l0 & Uint64(4294967295)))
                misc[9 + mo] = Int32(Uint32(l1 & Uint64(4294967295)))
                misc[10 + mo] = Int32(Uint32(l2 & Uint64(4294967295)))
                misc[11 + mo] = Int32(Uint32(l3 & Uint64(4294967295)))
            else:
                misc[8 + mo] = Int32(2139095040)
                misc[9 + mo] = Int32(-8388608)
                misc[10 + mo] = Int32(-8388608)
                misc[11 + mo] = Int32(-8388608)
                misc[13 + mo] = Int32(1)
            self._stamp(mStamps, cta, ST_M_T)
        L.bar(BAR_SMX, 128)

    @cute.jit
    def _follow_tline1(
        self, arena, cta, d, ctid, ws, mStamps, gw1, gw2, gw3, mCtl, mo: cutlass.Constexpr[int] = 0
    ):
        misc = _sm_tensor(arena, T_OFF_MISC, Int32, 64)
        if True:
            gen = self._gen_from(gw1, gw2, gw3)
            misc[4 + mo] = Int32(Uint32(gen & Uint64(4294967295)))
            misc[5 + mo] = Int32(Uint32(gen >> Uint64(32)))
            tg = self._tag32(gen)
            t64 = gen + Uint64(1)
            skipf = Int32(0)
            if cutlass.const_expr(self.debug):
                if mCtl[1] == Int32(4) and d == Int32(2):
                    skipf = Int32(1)
            if skipf == Int32(0):
                st_rlx_u64(self._w(ws, W_FOL) + Int64(d) * Int64(8), t64)
            tdl = self._deadline()
            st_ = Int32(0)
            l0 = Uint64(0)
            l1 = Uint64(0)
            l2 = Uint64(0)
            l3 = Uint64(0)
            while st_ == Int32(0):
                l0 = L.ld_relaxed_u64(self._w(ws, W_TLINE))
                l1 = L.ld_relaxed_u64(self._w(ws, W_TLINE + 1))
                l2 = L.ld_relaxed_u64(self._w(ws, W_TLINE + 2))
                l3 = L.ld_relaxed_u64(self._w(ws, W_TLINE + 3))
                av = L.ld_relaxed_u64(self._w(ws, W_ABORT))
                if (
                    Uint32(l0 >> Uint64(32)) == tg
                    and Uint32(l1 >> Uint64(32)) == tg
                    and (Uint32(l2 >> Uint64(32)) == tg)
                    and (Uint32(l3 >> Uint64(32)) == tg)
                ):
                    st_ = Int32(1)
                elif av == t64:
                    st_ = Int32(2)
                elif Int64(L.gtimer()) > tdl:
                    st_ = Int32(3)
            if st_ == Int32(3):
                self._raise_abort(ws, gen, Int32(E_TIMEOUT))
            if st_ == Int32(1):
                misc[8 + mo] = Int32(Uint32(l0 & Uint64(4294967295)))
                misc[9 + mo] = Int32(Uint32(l1 & Uint64(4294967295)))
                misc[10 + mo] = Int32(Uint32(l2 & Uint64(4294967295)))
                misc[11 + mo] = Int32(Uint32(l3 & Uint64(4294967295)))
            else:
                misc[8 + mo] = Int32(2139095040)
                misc[9 + mo] = Int32(-8388608)
                misc[10 + mo] = Int32(-8388608)
                misc[11 + mo] = Int32(-8388608)
                misc[13 + mo] = Int32(1)
            self._stamp(mStamps, cta, ST_M_T)

    @cute.jit
    def _block_a_lead(
        self,
        arena,
        cta,
        ctid,
        ws,
        n,
        mBoard,
        mStamps,
        gw1,
        gw2,
        gw3,
        mCtl,
        d,
        mHist,
        mo: cutlass.Constexpr[int] = 0,
        fa: cutlass.Constexpr[int] = T_OFF_FA,
        mi: cutlass.Constexpr[int] = T_OFF_MISC,
        dry_=None,
    ):
        NT: cutlass.Constexpr[int] = 128
        SMP, SS2, TGT, TGT2 = L.gvr_ladder(
            n,
            Int32(K_TOP),
            Int32(PLAN["aim_base"]),
            Int32(PLAN["sfac"]),
            Int32(PLAN["amin"]),
            Int32(PLAN["sd_en"]),
            PLAN["r_const"],
            PLAN["scap"],
        )
        s_keys = _sm_tensor(arena, fa + (T_OFF_F_HIST - T_OFF_FA), Uint32, 256)
        s_mins = _sm_tensor(arena, fa + (T_OFF_F_MB - T_OFF_FA), Uint32, NT)
        s_res = _sm_tensor(arena, fa + (T_OFF_F_RES - T_OFF_FA), Uint32, 16)
        s_red = _sm_tensor(arena, fa + (T_OFF_F_RED - T_OFF_FA), Uint32, 64)
        misc = _sm_tensor(arena, mi, Int32, 64)
        lane = ctid % Int32(32)
        wid = ctid // Int32(32)
        bbase = mBoard.iterator.toint()
        if d == Int32(0):
            # DRS starts at histogram row 1, leaving row 0 for these counts.
            for j in cutlass.range_constexpr(NB_INITIAL // NT):
                L.st_g_u32(
                    mHist.iterator.toint() + Int64(ctid + Int32(j * NT)) * Int64(4), Uint32(0)
                )
            L.fence_acq_rel_gpu()
        if ctid == Int32(0):
            self._stamp(mStamps, cta, 68)
            gg_ = self._gen_from(gw1, gw2, gw3)
            misc[4 + mo] = Int32(Uint32(gg_ & Uint64(4294967295)))
            misc[5 + mo] = Int32(Uint32(gg_ >> Uint64(32)))
        L.bar(BAR_SMX, NT)
        if d == Int32(0):
            if ctid == Int32(0):
                # Release of the histogram clear (cumulative over the clearing threads, ordered by the barrier
                # above): every later store of this thread, the T word included, is ordered after it.
                L.fence_acq_rel_gpu()
        gen = Uint64(Uint32(misc[5 + mo])) << Uint64(32) | Uint64(Uint32(misc[4 + mo]))
        tg = self._tag32(gen)
        t64 = gen + Uint64(1)
        if cutlass.const_expr(self.fblka):
            if d != Int32(0):
                if ctid == Int32(0):
                    st_rlx_u64(self._w(ws, W_FOL) + Int64(d) * Int64(8), t64)
        self._skew(gen, cta * Int32(4) + Int32(2), ctid)
        nq = SMP * Int32(4)
        nmine = Int32(0)
        if ctid < nq:
            nmine = cutlass.min((nq - ctid + Int32(NT - 1)) // Int32(NT), Int32(RMAX))
        # Keep the tagged board's values in registers until the local top-two
        # reduction. No raw-sample shared staging or histogram rescan remains.
        samples = cute.make_rmem_tensor((2 * RMAX,), Float32)
        okp = Int32(0)
        ab = Int32(0)
        if nmine == Int32(0):
            okp = Int32(1)
        if cutlass.const_expr(dry_ is not None):
            # dry pass without board loads (warms this code in the SM's instruction cache)
            if dry_ != Int32(0):
                okp = Int32(1)
        tdl = self._deadline()
        while okp == Int32(0):
            wsr = []
            for r_ in cutlass.range_constexpr(RMAX):
                q2 = ctid + cutlass.min(Int32(r_), nmine - Int32(1)) * Int32(NT)
                wsr.append(ld_relaxed_v2u64(bbase + Int64(q2) * Int64(16)))
            av = L.ld_relaxed_u64(self._w(ws, W_ABORT))
            okp = Int32(1)
            for r_ in cutlass.range_constexpr(RMAX):
                if Int32(r_) < nmine:
                    samples[2 * r_] = L.f32_of_u32(Uint32(wsr[r_][0]))
                    samples[2 * r_ + 1] = L.f32_of_u32(Uint32(wsr[r_][1]))
                    if (
                        Uint32(wsr[r_][0] >> Uint64(32)) != tg
                        or Uint32(wsr[r_][1] >> Uint64(32)) != tg
                    ):
                        okp = Int32(0)
            if d != Int32(0):
                okp = Int32(1)
            if okp == Int32(0):
                if av == t64:
                    ab = Int32(1)
                    okp = Int32(1)
                elif Int64(L.gtimer()) > tdl:
                    ab = Int32(2)
                    okp = Int32(1)
        if ab == Int32(2):
            self._raise_abort(ws, gen, Int32(E_TIMEOUT))
        smn = Float32(float("inf"))
        smx = Float32(float("-inf"))
        smx2 = Float32(float("-inf"))
        # Consume a thread's registers only after its tagged snapshot passed.
        if ab == Int32(0):
            for r_ in cutlass.range_constexpr(RMAX):
                if Int32(r_) < nmine:
                    va = samples[2 * r_]
                    vb = samples[2 * r_ + 1]
                    smn = cute.arch.fmin(smn, va)
                    smn = cute.arch.fmin(smn, vb)
                    pair_hi = cute.arch.fmax(va, vb)
                    pair_lo = cute.arch.fmin(va, vb)
                    smx2 = cute.arch.fmax(smx2, cute.arch.fmin(smx, pair_hi))
                    smx2 = cute.arch.fmax(smx2, pair_lo)
                    smx = cute.arch.fmax(smx, pair_hi)
        s_keys[ctid] = L.fkey(smx)
        s_keys[ctid + Int32(NT)] = L.fkey(smx2)
        s_mins[ctid] = L.fkey(smn)
        wab = cute.arch.warp_redux_sync(Uint32(ab), "or")
        if lane == Int32(0):
            s_red[Int32(40) + wid] = wab
        L.bar(BAR_SMX, NT)
        ab = Int32(s_red[40] | s_red[41] | s_red[42] | s_red[43])
        if ctid == Int32(0):
            self._stamp(mStamps, cta, ST_M_BOARD)
            self._stamp(mStamps, cta, 21)
        nkeys = Int32(2) * cutlass.min(nq, Int32(NT))
        if wid < Int32(3):
            rank = Int32(TGT)
            if wid == Int32(1):
                rank = Int32(TGT2)
            elif wid == Int32(2):
                rank = Int32(TGT * Int32(2))
            selected = Uint32(L.fkey(Float32(float("-inf"))))
            if ab == Int32(0) and rank > Int32(0) and rank <= nkeys:
                keys = cute.make_rmem_tensor((8,), Uint32)
                lo = Uint32(4294967295)
                hi = Uint32(0)
                for j in cutlass.range_constexpr(8):
                    shared_key = Uint32(s_keys[lane + Int32(j * 32)])
                    keys[j] = shared_key
                    lo = Uint32(cutlass.min(lo, shared_key))
                    hi = Uint32(cutlass.max(hi, shared_key))
                lo = Uint32(L.warp_min_u32(lo))
                hi = Uint32(L.warp_max_u32(hi))
                steps = Int32(0)
                # Each pivot removes a differing key bit. Count the whole
                # array at every pass, so rank remains a global rank.
                while lo < hi and steps < Int32(32):
                    bit = Uint32(Uint32(31) - cute.arch.clz(lo ^ hi))
                    pivot = Uint32((hi >> bit) << bit)
                    nabove = Int32(0)
                    above_min = Uint32(4294967295)
                    below_max = Uint32(0)
                    for j in cutlass.range_constexpr(8):
                        radix_key = Uint32(keys[j])
                        if radix_key >= pivot:
                            nabove = nabove + Int32(1)
                            above_min = Uint32(cutlass.min(above_min, radix_key))
                        else:
                            below_max = Uint32(cutlass.max(below_max, radix_key))
                    nabove = Int32(cute.arch.warp_redux_sync(nabove, "add"))
                    above_min = Uint32(L.warp_min_u32(above_min))
                    below_max = Uint32(L.warp_max_u32(below_max))
                    if nabove >= rank:
                        lo = Uint32(above_min)
                    else:
                        hi = Uint32(below_max)
                    steps = steps + Int32(1)
                selected = Uint32(lo)
            if lane == Int32(0):
                s_res[wid] = selected
        else:
            vmin = Uint32(4294967295)
            vmax = Uint32(0)
            for j in cutlass.range_constexpr(4):
                vmin = Uint32(cutlass.min(vmin, Uint32(s_mins[lane + Int32(j * 32)])))
                vmax = Uint32(cutlass.max(vmax, Uint32(s_keys[lane + Int32(j * 32)])))
            vmin = Uint32(L.warp_min_u32(vmin))
            vmax = Uint32(L.warp_max_u32(vmax))
            if lane == Int32(0):
                s_res[3] = vmin
                s_res[4] = vmax
        L.bar(BAR_SMX, NT)
        if ctid == Int32(0):
            self._stamp(mStamps, cta, 22)
            self._stamp(mStamps, cta, 23)
        SMIN = L.invkey(Uint32(s_res[3]))
        SMAX = L.invkey(Uint32(s_res[4]))
        GMIN = Float32(L.SENT_LO)
        T = Float32(NEG_INF)
        HIC = Float32(NEG_INF)
        T3 = Float32(NEG_INF)
        Tk = Float32(NEG_INF)
        wv = Float32(0.0)
        sc_s = Float32(0.0)
        sok = Int32(0)
        if SMP > Int32(0):
            if SMAX > SMIN:
                sok = Int32(1)
        tot0 = Int32(0)
        b1v = Int32(-1)
        b2v = Int32(-1)
        b3v = Int32(-1)
        if sok != Int32(0):
            wv = (SMAX - SMIN) * Float32(1.0 / 256.0)
            sc_s = cute.arch.rcp_approx(wv)
            tot0 = nkeys
            # Monotone binning commutes with order statistics. Reconstruct
            # the same 256-bin proposal; all-logit certification is unchanged.
            if tot0 >= TGT:
                b1v = L.f2s_rz((L.invkey(Uint32(s_res[0])) - SMIN) * sc_s)
                b1v = cutlass.max(Int32(0), cutlass.min(Int32(255), b1v))
            if tot0 >= TGT2:
                b2v = L.f2s_rz((L.invkey(Uint32(s_res[1])) - SMIN) * sc_s)
                b2v = cutlass.max(Int32(0), cutlass.min(Int32(255), b2v))
            if tot0 >= TGT * Int32(2):
                b3v = L.f2s_rz((L.invkey(Uint32(s_res[2])) - SMIN) * sc_s)
                b3v = cutlass.max(Int32(0), cutlass.min(Int32(255), b3v))
        if sok != Int32(0):
            if tot0 >= TGT:
                T = L._fmaf(Float32(b1v), wv, SMIN)
        tinf = Int32(1)
        if T > Float32(NEG_INF):
            tinf = Int32(0)
        if tinf != Int32(0):
            T = GMIN
        if sok != Int32(0):
            if tot0 >= TGT:
                if b2v >= Int32(0):
                    Tk = L._fmaf(Float32(b2v), wv, SMIN)
                    d_ = cute.arch.fmax(Tk - T, Float32(0.0))
                    HIC = cute.arch.fmax(L._fmaf(Float32(4.0), d_, T), L._fmaf(Float32(8.0), wv, T))
        if sok != Int32(0):
            if tot0 >= TGT * Int32(2):
                if b3v >= Int32(0):
                    if T > GMIN:
                        t3c = L._fmaf(Float32(b3v), wv, SMIN)
                        if t3c < T:
                            T3 = t3c
        L.bar(BAR_SMX, NT)
        if ctid == Int32(0):
            if ab != Int32(0):
                misc[8 + mo] = Int32(2139095040)
                misc[9 + mo] = Int32(-8388608)
                misc[10 + mo] = Int32(-8388608)
                misc[11 + mo] = Int32(-8388608)
                misc[13 + mo] = Int32(1)
            else:
                misc[8 + mo] = Int32(L.u32_of_f32(T))
                misc[9 + mo] = Int32(L.u32_of_f32(HIC))
                misc[10 + mo] = Int32(L.u32_of_f32(T3))
                misc[11 + mo] = Int32(L.u32_of_f32(SMAX))
                self._skew(gen, cta, 5)
                skip_t = Int32(0)
                if d != Int32(0):
                    skip_t = Int32(1)
                if cutlass.const_expr(self.debug):
                    if mCtl[1] == Int32(1):
                        skip_t = Int32(1)
                if skip_t == Int32(0):
                    tgl = Uint64(tg) << Uint64(32)
                    st_rlx_u64(self._w(ws, W_TLINE + 1), tgl | Uint64(L.u32_of_f32(HIC)))
                    st_rlx_u64(self._w(ws, W_TLINE + 2), tgl | Uint64(L.u32_of_f32(T3)))
                    st_rlx_u64(self._w(ws, W_TLINE + 3), tgl | Uint64(L.u32_of_f32(SMAX)))
                    st_rlx_u64(self._w(ws, W_TLINE), tgl | Uint64(L.u32_of_f32(T)))
            self._stamp(mStamps, cta, ST_M_T)
        L.bar(BAR_SMX, NT)

    @cute.jit
    def _poll_counts(self, ws, gen, slot, ctid):
        NT: cutlass.Constexpr[int] = 128
        NS = self.ns
        t40 = self._tag40(gen)
        t64 = gen + Uint64(1)
        hasa = ctid < Int32(NS)
        hasb = ctid + Int32(NT) < Int32(NS)
        oka = Int32(0)
        okb = Int32(0)
        if not hasa:
            oka = Int32(1)
        if not hasb:
            okb = Int32(1)
        na = Int32(0)
        nb = Int32(0)
        ab = Int32(0)
        go = Int32(1)
        if oka != Int32(0) and okb != Int32(0):
            go = Int32(0)
        tdl = self._deadline()
        ca = self._cnt_addr(ws, slot, cutlass.min(ctid, Int32(NS - 1)))
        cb = self._cnt_addr(ws, slot, cutlass.min(ctid + Int32(NT), Int32(NS - 1)))
        it_ = Int32(0)
        while go != Int32(0):
            wa = Uint64(0)
            wb = Uint64(0)
            if cutlass.const_expr(self.acqpoll):
                wa = L.ld_acquire_u64(ca)
                wb = L.ld_acquire_u64(cb)
            else:
                wa = L.ld_relaxed_u64(ca)
                wb = L.ld_relaxed_u64(cb)
            av = Uint64(0)
            chk = Int32(1)
            if cutlass.const_expr(self.pchk > 1):
                chk = Int32(0)
                if it_ & Int32(self.pchk - 1) == Int32(self.pchk - 1):
                    chk = Int32(1)
            if chk != Int32(0):
                av = L.ld_relaxed_u64(self._w(ws, W_ABORT))
            it_ = it_ + Int32(1)
            if oka == Int32(0):
                if wa >> Uint64(24) == t40:
                    oka = Int32(1)
                    na = Int32(Uint32(wa & Uint64(16777215)))
            if okb == Int32(0):
                if wb >> Uint64(24) == t40:
                    okb = Int32(1)
                    nb = Int32(Uint32(wb & Uint64(16777215)))
            if oka != Int32(0) and okb != Int32(0):
                go = Int32(0)
            elif chk != Int32(0):
                if av == t64:
                    ab = Int32(1)
                    go = Int32(0)
                elif Int64(L.gtimer()) > tdl:
                    ab = Int32(2)
                    go = Int32(0)
        if ab == Int32(2):
            self._raise_abort(ws, gen, Int32(E_TIMEOUT))
        if cutlass.const_expr(not self.acqpoll and (not self.nofence)):
            L.fence_acq_rel_gpu()
        return (na, nb, ab)

    @cute.jit
    def _poll_counts_hist(self, ws, gen, ctid, hptr, s_wc):
        # CTA-uniform count poll that also re-reads this thread's four histogram bins on every pass, so the
        # histogram snapshot is as fresh as the last count word seen; the caller validates it against C1.
        NT: cutlass.Constexpr[int] = 128
        NS = self.ns
        lane = ctid % Int32(32)
        wid = ctid // Int32(32)
        t40 = self._tag40(gen)
        t64 = gen + Uint64(1)
        oka = Int32(0)
        okb = Int32(0)
        if not ctid < Int32(NS):
            oka = Int32(1)
        if not ctid + Int32(NT) < Int32(NS):
            okb = Int32(1)
        na = Int32(0)
        nb = Int32(0)
        ab = Int32(0)
        h0 = Int32(0)
        h1 = Int32(0)
        h2 = Int32(0)
        h3 = Int32(0)
        tdl = self._deadline()
        ca = self._cnt_addr(ws, Int32(0), cutlass.min(ctid, Int32(NS - 1)))
        cb = self._cnt_addr(ws, Int32(0), cutlass.min(ctid + Int32(NT), Int32(NS - 1)))
        it_ = Int32(0)
        go = Int32(1)
        while go != Int32(0):
            wa = Uint64(0)
            wb = Uint64(0)
            if cutlass.const_expr(self.acqx):
                wa = L.ld_relaxed_u64(ca)
                wb = L.ld_relaxed_u64(cb)
            else:
                hv = B.ldg_cg_v4(hptr)
                h0 = hv[0]
                h1 = hv[1]
                h2 = hv[2]
                h3 = hv[3]
                wa = L.ld_acquire_u64(ca)
                wb = L.ld_acquire_u64(cb)
            if oka == Int32(0):
                if wa >> Uint64(24) == t40:
                    oka = Int32(1)
                    na = Int32(Uint32(wa & Uint64(16777215)))
            if okb == Int32(0):
                if wb >> Uint64(24) == t40:
                    okb = Int32(1)
                    nb = Int32(Uint32(wb & Uint64(16777215)))
            if it_ & Int32(15) == Int32(15):
                if ab == Int32(0):
                    av = L.ld_relaxed_u64(self._w(ws, W_ABORT))
                    if av == t64:
                        ab = Int32(1)
                    elif Int64(L.gtimer()) > tdl:
                        ab = Int32(2)
            mine = (oka != Int32(0) and okb != Int32(0)) or ab != Int32(0)
            wd = Int32(cute.arch.vote_ballot_sync(mine))
            if lane == Int32(0):
                s_wc[Int32(40) + (it_ & Int32(1)) * Int32(4) + wid] = wd
            L.bar(BAR_CTL, NT)
            base = Int32(40) + (it_ & Int32(1)) * Int32(4)
            alld = (
                s_wc[base] & s_wc[base + Int32(1)] & s_wc[base + Int32(2)] & s_wc[base + Int32(3)]
            )
            it_ = it_ + Int32(1)
            if alld == Int32(-1):
                go = Int32(0)
            else:
                if cutlass.const_expr(self.pollns > 0):
                    L.nanosleep(Int32(self.pollns))
        if cutlass.const_expr(self.acqx):
            # acquire for the count releases this thread observed, passed on by the barrier, then one
            # histogram read that includes every published contribution
            L.fence_acq_rel_gpu()
            L.bar(BAR_CTL, NT)
            hx = B.ldg_cg_v4(hptr)
            h0 = hx[0]
            h1 = hx[1]
            h2 = hx[2]
            h3 = hx[3]
        if ab == Int32(2):
            self._raise_abort(ws, gen, Int32(E_TIMEOUT))
        return (na, nb, ab, h0, h1, h2, h3)

    @cute.jit
    def _k3row(self, gp, npg_s, s_btc, mBtC, mbc, real):
        pg = gp >> Int32(5)
        bt = Int32(0)
        # Prewarm uses synthetic candidates and does not wait for the BTC TMA.
        # Its rows never reach attention, so only real selection reads pages.
        if real:
            if pg < npg_s:
                bt = s_btc[pg]
            else:
                bt = mBtC[0, cutlass.min(pg, mbc - Int32(1))]
        return bt * Int32(32) + (gp & Int32(31))

    @cute.jit
    def _bucket_addr(self, rg, b, ordinal):
        # The second half of each producer region is idle until DRS starts.
        # Four 512-entry buckets fit in each such half; no extra allocation.
        flat = b * Int32(BUCKET_CAP) + ordinal
        off = flat // Int32(CAPR) * Int32(2 * CAPR) + Int32(CAPR) + flat % Int32(CAPR)
        return rg + Int64(off) * Int64(8)

    @cute.jit
    def _bucket_acquire(
        self, arena, cta, d, ctid, ws, gen, Kq, mHist, mStamps, mCtl, real=True, mo_=None, rgp_=None
    ):
        NT: cutlass.Constexpr[int] = 128
        lane = ctid % Int32(32)
        wid = ctid // Int32(32)
        misc = _sm_tensor(arena, T_OFF_MISC, Int32, 64)
        s_red = _sm_tensor(arena, T_OFF_RED, Uint32, 64)
        s_wc = _sm_tensor(arena, T_OFF_WC, Int32, 64)
        s_hist = _sm_tensor(arena, T_OFF_HIST, Int32, NB_SEL)
        s_res = _sm_tensor(arena, T_OFF_RES, Int32, 16)
        s_map = _sm_tensor(arena, SMAP2_OFF, Int32, K_TOP)
        if cutlass.const_expr(mo_ is None):
            mo_ = Int32(0)
        # warm-up pass (real == 0): the same code without the count poll; the histogram values it reads are
        # discarded and it never takes the fast path, publishes or writes misc
        na = Int32(0)
        nb = Int32(0)
        ab = Int32(0)
        sh0 = Int32(0)
        sh1 = Int32(0)
        sh2 = Int32(0)
        sh3 = Int32(0)
        hptr = mHist.iterator.toint() + Int64(ctid) * Int64(16)
        if real:
            na, nb, ab, sh0, sh1, sh2, sh3 = self._poll_counts_hist(ws, gen, ctid, hptr, s_wc)
        else:
            hw_ = B.ldg_cg_v4(hptr)
            sh0 = hw_[0]
            sh1 = hw_[1]
            sh2 = hw_[2]
            sh3 = hw_[3]
        wc = Int32(cute.arch.warp_redux_sync(na + nb, "add"))
        wa = Uint32(cute.arch.warp_redux_sync(Uint32(ab), "or"))
        if lane == Int32(0):
            s_red[wid * Int32(2)] = Uint32(wc)
            s_red[wid * Int32(2) + Int32(1)] = wa
        # This barrier also carries every count-acquire to all histogram readers.
        L.bar(BAR_CTL, NT)
        C1 = Int32(s_red[0] + s_red[2] + s_red[4] + s_red[6])
        bad = Int32(s_red[1] | s_red[3] | s_red[5] | s_red[7])
        T = L.f32_of_u32(Uint32(misc[Int32(8) + mo_]))
        HIC = L.f32_of_u32(Uint32(misc[Int32(9) + mo_]))
        SMAX = L.f32_of_u32(Uint32(misc[Int32(11) + mo_]))
        lo, hi = self._band_range(T, HIC, SMAX)
        sc = S.bin_scale(lo, hi) * Float32(0.5)
        eligible = (
            bad == Int32(0)
            and misc[Int32(13) + mo_] == Int32(0)
            and C1 >= Kq
            and Kq > Int32(0)
            and T > Float32(L.SENT_LO)
            and sc > Float32(0.0)
        )
        if cutlass.const_expr(self.debug):
            if mCtl[0] != Int32(0):
                eligible = False
        fast = Int32(0)
        Bb = Int32(-1)
        above = Int32(0)
        m = Int32(0)
        elig_ = Int32(0)
        if eligible:
            elig_ = Int32(1)
        if not real:
            elig_ = Int32(1)
        if elig_ != Int32(0):
            if ctid == Int32(0):
                s_res[L.RES_B] = Int32(-1)
                s_res[L.RES_ABOVE] = Int32(0)
                s_res[L.RES_M] = Int32(0)
            # Adjacent bins are one aligned vector load, avoiding four
            # dependent scalar CG load/consume sequences per thread.
            for q in cutlass.range_constexpr(4):
                s_map[ctid * Int32(4) + Int32(q)] = Int32(2147483647)
            subtotal = (sh0 + sh1) + (sh2 + sh3)
            inc = L.warp_incl_scan_add(subtotal, lane)
            if lane == Int32(31):
                s_wc[wid] = inc
            L.bar(BAR_CTL, NT)
            before = inc - subtotal
            total = Int32(0)
            for w in cutlass.range_constexpr(4):
                v = s_wc[w]
                if Int32(w) < wid:
                    before = before + v
                total = total + v
            if real:
                if total != C1:
                    # stale snapshot: one re-read, ordered after every count acquire by the barrier above
                    L.bar(BAR_CTL, NT)
                    hw2 = B.ldg_cg_v4(hptr)
                    sh0 = hw2[0]
                    sh1 = hw2[1]
                    sh2 = hw2[2]
                    sh3 = hw2[3]
                    subtotal = (sh0 + sh1) + (sh2 + sh3)
                    inc = L.warp_incl_scan_add(subtotal, lane)
                    if lane == Int32(31):
                        s_wc[wid] = inc
                    L.bar(BAR_CTL, NT)
                    before = inc - subtotal
                    total = Int32(0)
                    for w in cutlass.range_constexpr(4):
                        v = s_wc[w]
                        if Int32(w) < wid:
                            before = before + v
                        total = total + v
            hv = [sh0, sh1, sh2, sh3]
            suffix = total - before
            for q in cutlass.range_constexpr(4):
                bi = ctid * Int32(4) + Int32(q)
                s_hist[bi] = suffix
                next_suffix = suffix - hv[q]
                # Mark each nonempty bucket's output interval. Its physical
                # source index is a constant delta plus the destination slot.
                # The deltas decrease with slot, so one prefix-min fills gaps.
                if hv[q] > Int32(0) and next_suffix < Kq:
                    flat = bi * Int32(BUCKET_CAP)
                    off = flat // Int32(CAPR) * Int32(2 * CAPR) + Int32(CAPR) + flat % Int32(CAPR)
                    s_map[next_suffix] = off - next_suffix
                    if cutlass.const_expr(self.bpf and rgp_ is not None):
                        # this bin feeds the top Kq rows: its entry lines into L1 now
                        if real:
                            nl_ = (cutlass.min(hv[q], Int32(BUCKET_CAP)) + Int32(15)) // Int32(16)
                            jl_ = Int32(0)
                            while jl_ < nl_:
                                prefetch_l1(rgp_ + Int64(off + jl_ * Int32(16)) * Int64(8))
                                jl_ = jl_ + Int32(1)
                if suffix >= Kq and next_suffix < Kq:
                    s_res[L.RES_B] = bi
                    s_res[L.RES_ABOVE] = next_suffix
                    s_res[L.RES_M] = hv[q]
                suffix = next_suffix
            if ctid == Int32(0):
                s_hist[NB_INITIAL] = Int32(0)
            L.bar(BAR_CTL, NT)
            Bb = s_res[L.RES_B]
            above = s_res[L.RES_ABOVE]
            m = s_res[L.RES_M]
            # Every bin strictly above the crossing contains at most Kq rows,
            # so it fits in BUCKET_CAP. Only the crossing needs a capacity test.
            if total == C1 and Bb >= Int32(0) and m <= Int32(S.QUADC):
                if real:
                    fast = Int32(1)
        if fast != Int32(0):
            if ctid == Int32(0):
                misc[Int32(12) + mo_] = Int32(P_T1)
                if d == Int32(0 if self.late_done != 4 else -1):
                    fol_ok = self._wait_followers(ws, gen)
                    misc[Int32(14) + mo_] = fol_ok
                    if cutlass.const_expr(self.late_done == 0):
                        if fol_ok != Int32(0):
                            self._publish_dir(
                                ws,
                                gen,
                                Int32(1),
                                Int32(K_DONE),
                                Int32(P_T1),
                                Uint64(0),
                                Uint64(0),
                                Uint64(0),
                            )
                self._stamp(mStamps, cta, ST_M_FIN)
        return (fast, C1, Bb, above, m)

    @cute.jit
    def _bucket_select(
        self,
        arena,
        cta,
        ctid,
        Kw,
        Bb,
        above,
        m,
        real,
        st,
        bars,
        n,
        s_idx,
        mBtC,
        mbc,
        mRegion,
        mStamps,
        pre_=None,
    ):
        NT: cutlass.Constexpr[int] = 128
        lane = ctid % Int32(32)
        wid = ctid // Int32(32)
        s_map = _sm_tensor(arena, SMAP2_OFF, Int32, K_TOP)
        if cutlass.const_expr(pre_ is None):
            pre_ = Int32(0)
        s_wc = _sm_tensor(arena, T_OFF_WC, Int32, 64)
        s_outs = _sm_tensor(arena, T_OFF_OUTS, Int32, K_TOP)
        s_ck = _sm_tensor(arena, T_OFF_CK, Uint64, S.QUADC)
        s_cs = _sm_tensor(arena, T_OFF_CS, Int32, S.QUADC)
        s_crow = _sm_tensor(arena, T_OFF_CROW, Int32, S.QUADC)
        s_btc = _sm_tensor(arena, T_OFF_BTC, Int32, BTC_CAP)
        misc = _sm_tensor(arena, T_OFF_MISC, Int32, 64)
        rg = mRegion.iterator.toint()
        st = self._warm_abort(real, st, ctid, bars, misc)
        if st == Int32(ST_GO):
            if cutlass.const_expr(not self.btcl):
                if real and pre_ == Int32(0):
                    A._wait(bars + B_BTC, Int32(0))
            npg_s = (
                cutlass.min((n + Int32(31)) // Int32(32), cutlass.min(mbc, Int32(BTC_CAP)))
                // Int32(4)
                * Int32(4)
            )
            if pre_ != Int32(0):
                # pre-select: this pass's BTC copy (KV slot 6) holds another row, every page is read from global
                npg_s = Int32(0)
            # Expand the interval markers with one CTA prefix-min, replacing
            # four serial nine-step shared-memory searches per thread.
            offsets = []
            scan = Int32(2147483647)
            for q in cutlass.range_constexpr(K_TOP // NT):
                scan = cutlass.min(scan, s_map[ctid * Int32(4) + Int32(q)])
                offsets.append(scan)
            for step in cutlass.range_constexpr(5):
                other = cute.arch.shuffle_sync_up(scan, 1 << step, mask_and_clamp=0)
                if lane >= Int32(1 << step):
                    scan = cutlass.min(scan, other)
            if lane == Int32(31):
                s_wc[wid] = scan
            carry = cute.arch.shuffle_sync_up(scan, 1, mask_and_clamp=0)
            if lane == Int32(0):
                carry = Int32(2147483647)
            L.bar(BAR_CTL, NT)
            for w in cutlass.range_constexpr(4):
                if Int32(w) < wid:
                    carry = cutlass.min(carry, s_wc[w])
            # Stop synthetic work promptly when the real threshold arrives.
            st = self._warm_abort(real, st, ctid, bars, misc)
            if st == Int32(ST_GO):
                # Issue independent bucket reads before consuming any result.
                # Crossing entries are included to overlap their load latency too.
                entries = [Uint64(0)] * (K_TOP // NT)
                cross = [Uint64(0)] * ((S.QUADC + NT - 1) // NT)
                for q in cutlass.range_constexpr(K_TOP // NT):
                    po = ctid * Int32(4) + Int32(q)
                    if po < above:
                        e = Uint64(Uint32(po)) << Uint64(32) | Uint64(L.u32_of_f32(Float32(po)))
                        if real:
                            off = cutlass.min(offsets[q], carry) + po
                            if cutlass.const_expr(self.bpf):
                                e = ldg_ca_u64(rg + Int64(off) * Int64(8))
                            elif cutlass.const_expr(self.entcg):
                                e = ldg_cg_u64(rg + Int64(off) * Int64(8))
                            else:
                                e = L.ld_relaxed_u64(rg + Int64(off) * Int64(8))
                        entries[q] = e
                for q in cutlass.range_constexpr((S.QUADC + NT - 1) // NT):
                    i = ctid + Int32(q * NT)
                    if i < m:
                        e = Uint64(Uint32(above + i)) << Uint64(32) | Uint64(
                            L.u32_of_f32(Float32(above + i))
                        )
                        if real:
                            if cutlass.const_expr(self.bpf):
                                e = ldg_ca_u64(self._bucket_addr(rg, Bb, i))
                            elif cutlass.const_expr(self.entcg):
                                e = ldg_cg_u64(self._bucket_addr(rg, Bb, i))
                            else:
                                e = L.ld_relaxed_u64(self._bucket_addr(rg, Bb, i))
                        cross[q] = e
                if cutlass.const_expr(self.btcl):
                    # the block table is needed only from here (translation); the loads above are in flight
                    if real and pre_ == Int32(0):
                        A._wait(bars + B_BTC, Int32(0))
                for q in cutlass.range_constexpr(K_TOP // NT):
                    po = ctid * Int32(4) + Int32(q)
                    if po < above:
                        e = entries[q]
                        gp = Int32(Uint32(e >> Uint64(32)))
                        s_outs[po] = gp
                        rv_ = gp
                        if pre_ == Int32(0):
                            rv_ = self._k3row(gp, npg_s, s_btc, mBtC, mbc, real)
                        (s_idx + Int32(128) + po).store(rv_)
                for q in cutlass.range_constexpr((S.QUADC + NT - 1) // NT):
                    i = ctid + Int32(q * NT)
                    if i < m:
                        e = cross[q]
                        gp = Int32(Uint32(e >> Uint64(32)))
                        s_ck[i] = L.entry_key(e)
                        s_cs[i] = gp
                        rc_ = gp
                        if pre_ == Int32(0):
                            rc_ = self._k3row(gp, npg_s, s_btc, mBtC, mbc, real)
                        s_crow[i] = rc_
                L.bar(BAR_CTL, NT)
                if cutlass.const_expr(self.gjoin):
                    if real and above >= Int32(384) and pre_ == Int32(0):
                        cute.arch.fence_proxy("async.shared", space="cta")
                        L.bar(BAR_CTL, NT)
                        if ctid == Int32(0):
                            misc[15] = Int32(1)
                            misc[16] = above
                        prims.mbarrier_arrive(bars + B_EAR)
                if ctid == Int32(0) and real:
                    self._stamp(mStamps, cta, 70)
        st = self._warm_abort(real, st, ctid, bars, misc)
        if st == Int32(ST_GO):
            need = Kw - above
            if m <= Int32(32):
                if ctid < Int32(32):
                    key = Uint64(0)
                    if ctid < m:
                        key = Uint64(s_ck[ctid])
                    kh = Uint32(key >> Uint64(32))
                    kl = Uint32(key)
                    # Retire the smaller side of the boundary partition.
                    # Two native unsigned minima select an exact 64-bit key:
                    # first its high word, then its low word among high ties.
                    discard = m - need <= need
                    flip = Uint32(0)
                    if not discard:
                        flip = Uint32(4294967295)
                    xh = kh ^ flip
                    xl = kl ^ flip
                    live = ctid < m
                    nr = cutlass.min(need, m - need)
                    for j in cutlass.range(nr, unroll=1):
                        vh = Uint32(4294967295)
                        if live:
                            vh = xh
                        mh = Uint32(cute.arch.warp_redux_sync(vh, "min"))
                        vl = Uint32(4294967295)
                        if live and xh == mh:
                            vl = xl
                        ml = Uint32(cute.arch.warp_redux_sync(vl, "min"))
                        if live and xh == mh and xl == ml:
                            live = False
                    keep = live
                    if not discard:
                        keep = ctid < m and not live
                    mask = Uint32(cute.arch.vote_ballot_sync(keep))
                    prior = (Uint32(1) << Uint32(ctid)) - Uint32(1)
                    rank = Int32(cute.arch.popc(mask & prior))
                    if keep:
                        s_outs[above + rank] = s_cs[ctid]
                        (s_idx + Int32(128) + above + rank).store(s_crow[ctid])
            else:
                i = ctid
                while i < m:
                    key = Uint64(s_ck[i])
                    rank = Int32(0)
                    for j in cutlass.range(m, unroll=1):
                        if Uint64(s_ck[j]) > key:
                            rank = rank + Int32(1)
                    if rank < need:
                        s_outs[above + rank] = s_cs[i]
                        (s_idx + Int32(128) + above + rank).store(s_crow[i])
                    i = i + Int32(NT)
            L.bar(BAR_CTL, NT)
        return st

    @cute.jit
    def _shipped_sel(self, arena, cta, d, ctid, ws, s_idx, bars, mStamps, n, mBtC):
        # this row was pre-selected in the previous pass: copy its KV rows, release the gathers, then the
        # same directive tail as _control_s1d's real pass
        NT: cutlass.Constexpr[int] = 128
        misc = _sm_tensor(arena, T_OFF_MISC, Int32, 64)
        s2_ = _ptr(arena, SIDX2_OFF, Int32)
        A._wait(bars + B_T, Int32(0))
        gen = Uint64(Uint32(misc[5])) << Uint64(32) | Uint64(Uint32(misc[4]))
        if ctid == Int32(0):
            self._stamp(mStamps, cta, 71)
        # SIDX2 holds positions (-1 = padding): translate with this pass's block-table copy
        A._wait(bars + B_BTC, Int32(0))
        if ctid == Int32(0):
            self._stamp(mStamps, cta, 74)
        mbc = cute.size(mBtC, mode=[1])
        s_btc = _sm_tensor(arena, T_OFF_BTC, Int32, BTC_CAP)
        npg_s = (
            cutlass.min((n + Int32(31)) // Int32(32), cutlass.min(mbc, Int32(BTC_CAP)))
            // Int32(4)
            * Int32(4)
        )
        for q in cutlass.range_constexpr(K_TOP // NT):
            ii = Int32(128) + ctid + Int32(q * NT)
            gp = (s2_ + ii).load()
            rv_ = Int32(-1)
            if gp >= Int32(0):
                rv_ = self._k3row(gp, npg_s, s_btc, mBtC, mbc, True)
            (s_idx + ii).store(rv_)
        if ctid == Int32(0):
            misc[12] = Int32(P_T1)
            misc[15] = Int32(0)
            misc[16] = misc[59]
            self._stamp(mStamps, cta, ST_M_SEL)
        cute.arch.fence_proxy("async.shared", space="cta")
        L.bar(BAR_CTL, NT)
        prims.mbarrier_arrive(bars + B_EAR)
        prims.mbarrier_arrive(bars + B_SEL)
        if ctid == Int32(0):
            self._stamp(mStamps, cta, ST_M_OUT)
            if cutlass.const_expr(self.late_done == 4):
                if d == Int32(0):
                    if self._wait_followers(ws, gen) != Int32(0):
                        self._publish_dir(
                            ws,
                            gen,
                            Int32(1),
                            Int32(K_DONE),
                            Int32(P_T1),
                            Uint64(0),
                            Uint64(0),
                            Uint64(0),
                        )

    @cute.jit
    def _control_s1d(
        self,
        arena,
        cta,
        d,
        ctid,
        ws,
        n,
        s_idx,
        bars,
        mSwaLocal,
        mBtSwa,
        mBtC,
        mTopk,
        mGidx,
        mRegion,
        mHist,
        mStamps,
        mCtl,
        gargs=None,
        eargs=None,
        p0=None,
        pre_=None,
        gwn=None,
    ):
        NT: cutlass.Constexpr[int] = 128
        is_t0 = d == Int32(0)
        misc = _sm_tensor(arena, T_OFF_MISC, Int32, 64)
        s_outs = _sm_tensor(arena, T_OFF_OUTS, Int32, K_TOP)
        _s_lst = _sm_tensor(arena, T_OFF_LIST, Uint64, CAPL)
        s_hist = _sm_tensor(arena, T_OFF_HIST, Int32, NB_SEL)
        s_map = _sm_tensor(arena, SMAP2_OFF, Int32, K_TOP)
        pre_on = pre_ is not None
        if cutlass.const_expr(pre_ is None):
            pre_ = Int32(0)
        # misc offset of this job's T line (pre-select: polled into misc + 44)
        mo_ = cutlass.select_(pre_ != Int32(0), Int32(44), Int32(0))
        mbc = cute.size(mBtC, mode=[1])
        mbs = cute.size(mBtSwa, mode=[1])
        if cutlass.const_expr(not self.swag):
            loc = mSwaLocal[0, ctid]
            rsw = L.l2g_swa_local(loc, Int32(0), mBtSwa, mbs, Int32(0))
            (s_idx + ctid).store(rsw)
            L.bar(BAR_CTL, NT)
            # Every writer releases its own index stores to the gather warps.
            prims.mbarrier_arrive(bars + B_SWA)
        if ctid == Int32(0):
            self._stamp(mStamps, cta, ST_M_SWA)
        Kq = cutlass.min(n, Int32(K_TOP))
        P0: cutlass.Constexpr[int] = 0 if self.prewarm else 1
        gen = Uint64(0)
        _path = Int32(P_T1)
        C1 = Int32(0)
        C = Int32(0)
        rounds = Int32(0)
        mret = Int32(0)
        info = Int32(0)
        Bb = Int32(-1)
        nout = Int32(0)
        zaddr = Int32(cute.recast_ptr(arena + T_OFF_MISC + 62 * 4, dtype=Int32).toint())
        if cutlass.const_expr(self.nohoist):
            sts_volatile_u32(zaddr, Int32(0))
        ctid0 = ctid
        for ps in cutlass.range(P0 if p0 is None else p0, 2, unroll=1):
            ctid = ctid0
            if cutlass.const_expr(self.nohoist):
                ctid = ctid0 + lds_volatile_u32(zaddr)
            real = ps == Int32(1)
            if ctid == Int32(0):
                self._stamp(mStamps, cta, 92 + ps)
            st = Int32(ST_GO)
            lo = Float32(1.0)
            hi = Float32(1.0) + Float32(0.001 * NT)
            mm = Int32(0)
            Kw = Int32(K_TOP)
            C = Int32(K_TOP)
            fast = Int32(0)
            above = Int32(0)
            bm = Int32(0)
            if real:
                if cutlass.const_expr(pre_on):
                    if pre_ != Int32(0):
                        # the next row's SWA KV rows into SIDX2[0..127] (copied by the next pass's swag warps)
                        if cutlass.const_expr(self.swpre):
                            locn_ = mSwaLocal[0, ctid]
                            (s_idx + ctid).store(
                                L.l2g_swa_local(locn_, Int32(0), mBtSwa, mbs, Int32(0))
                            )
                            if cutlass.const_expr(self.pdec):
                                # the SWA rows are final now: flag them for the next pass's swag warps
                                L.bar(BAR_CTL, NT)
                                if ctid == Int32(0):
                                    fence_acq_rel_cta()
                                    sts_volatile_u32(
                                        Int32(
                                            cute.recast_ptr(
                                                arena + T_OFF_MISC + 60 * 4, dtype=Int32
                                            ).toint()
                                        ),
                                        Int32(1),
                                    )
                        if ctid == Int32(0):
                            misc[57] = Int32(0)
                            misc[58] = Int32(0)
                            if cutlass.const_expr(self.pdirect and self.npmax > 2):
                                # misc[27] = "the pass this pre-select prepares has started": clear the
                                # previous pass's start (set before this pre-select began)
                                sts_volatile_u32(
                                    Int32(
                                        cute.recast_ptr(
                                            arena + T_OFF_MISC + 27 * 4, dtype=Int32
                                        ).toint()
                                    ),
                                    Int32(0),
                                )
                            self._follow_tline1(
                                arena,
                                cta,
                                d,
                                Int32(0),
                                ws,
                                mStamps,
                                gwn[0],
                                gwn[1],
                                gwn[2],
                                mCtl,
                                mo=44,
                            )
                        L.bar(BAR_CTL, NT)
                if pre_ == Int32(0):
                    A._wait(bars + B_T, Int32(0))
                gen = Uint64(Uint32(misc[Int32(5) + mo_])) << Uint64(32) | Uint64(
                    Uint32(misc[Int32(4) + mo_])
                )
                Kw = Kq
            # The warm-up pass is split over the team: CTAs >= 2 run the bucket-acquire code, CTAs 0-1 the bucket
            # emission / rank, so both are L2-resident for every team CTA when the stream ends.
            fa_ = Int32(0)
            c1a_ = Int32(0)
            bba_ = Int32(-1)
            aba_ = Int32(0)
            bma_ = Int32(0)
            if real or d >= Int32(2):
                fa_, c1a_, bba_, aba_, bma_ = self._bucket_acquire(
                    arena,
                    cta,
                    d,
                    ctid,
                    ws,
                    gen,
                    Kq,
                    mHist,
                    mStamps,
                    mCtl,
                    real,
                    mo_=mo_,
                    rgp_=mRegion.iterator.toint(),
                )
            if real:
                fast = fa_
                C1 = c1a_
                Bb = bba_
                above = aba_
                bm = bma_
                if fast == Int32(0):
                    if pre_ == Int32(0):
                        st, _path, C, lo, hi, mm, C1, rounds = self._acquire(
                            arena, cta, d, ctid, ws, gen, n, Kq, bars, mRegion, mHist, mStamps, mCtl
                        )
                else:
                    _path = Int32(P_T1)
                    C = C1
            elif d < Int32(2):
                # Warm the bucket-emission and boundary-rank code with purely
                # synthetic shared state. No unpublished global packets are read.
                fast = Int32(1)
                Bb = Int32(0)
                above = Int32(K_TOP - 1)
                bm = Int32(1)
                for q in cutlass.range_constexpr(NB_INITIAL // NT):
                    bi = ctid + Int32(q * NT)
                    s_hist[bi] = Int32(NB_INITIAL) - bi
                    flat = (Int32(NB_INITIAL - 1) - bi) * Int32(BUCKET_CAP)
                    off = flat // Int32(CAPR) * Int32(2 * CAPR) + Int32(CAPR) + flat % Int32(CAPR)
                    s_map[bi] = off - bi
                if ctid == Int32(0):
                    s_hist[NB_INITIAL] = Int32(0)
                L.bar(BAR_CTL, NT)
            # this pre-select's acquire completed after the pass it prepares started: join that pass now
            # (BAR_HAND) and emit physical rows into its s_idx with the normal release tail
            pre_e_ = pre_
            s_idxe_ = s_idx
            if cutlass.const_expr(self.pdirect and pre_on):
                if real:
                    if pre_ != Int32(0) and fast != Int32(0):
                        if ctid == Int32(0):
                            fpa_ = Int32(
                                cute.recast_ptr(arena + T_OFF_MISC + 27 * 4, dtype=Int32).toint()
                            )
                            fv_ = lds_volatile_u32(fpa_)
                            if cutlass.const_expr(self.pdwait > 0):
                                # wait a little for the pass start before taking the positions path
                                tw_ = Int64(L.gtimer()) + Int64(self.pdwait)
                                while fv_ == Int32(0) and Int64(L.gtimer()) < tw_:
                                    fv_ = lds_volatile_u32(fpa_)
                            misc[28] = fv_
                        L.bar(BAR_CTL, NT)
                        if misc[28] != Int32(0):
                            L.bar(BAR_HAND, 256)
                            pre_e_ = Int32(0)
                            if ctid == Int32(0):
                                misc[12] = Int32(P_T1)
                                misc[58] = Int32(2)
                                self._stamp(mStamps, cta, 78)
                s_idxe_ = _ptr(
                    arena,
                    cutlass.select_(pre_e_ != Int32(0), Int32(SIDX2_OFF), Int32(T_OFF_IDX)),
                    Int32,
                )
            fnar = Int32(0)
            if cutlass.const_expr(self.debug):
                if real:
                    if mCtl[0] == Int32(3):
                        fnar = Int32(1)
            if fast != Int32(0):
                st = self._bucket_select(
                    arena,
                    cta,
                    ctid,
                    Kw,
                    Bb,
                    above,
                    bm,
                    real,
                    st,
                    bars,
                    n,
                    s_idxe_,
                    mBtC,
                    mbc,
                    mRegion,
                    mStamps,
                    pre_=pre_e_,
                )
                nout = Kw
                mret = bm
                info = Int32(1)
            else:
                if real and pre_ == Int32(0):
                    st, mret, info, Bb, nout = self._sel_list(
                        arena,
                        cta,
                        ctid,
                        C,
                        Kw,
                        lo,
                        hi,
                        mm,
                        fnar,
                        real,
                        st,
                        bars,
                        n,
                        s_idx,
                        mBtC,
                        mbc,
                        mStamps,
                        d,
                    )
            if cutlass.const_expr(self.wacq == 2):
                if not real:
                    if st == Int32(ST_GO):
                        self._dry_acquire(arena, ctid, mRegion)
            if real:
                for u_ in cutlass.range_constexpr(K_TOP // NT):
                    ii = ctid + Int32(u_ * NT)
                    if ii >= nout:
                        s_outs[ii] = Int32(-1)
                        (s_idxe_ + Int32(128) + ii).store(Int32(-1))
                L.bar(BAR_CTL, NT)
                if is_t0 and (pre_ == Int32(0) or fast != Int32(0)):
                    for u_ in cutlass.range_constexpr(K_TOP // NT):
                        ii = ctid + Int32(u_ * NT)
                        mTopk[0, ii] = s_outs[ii]
            L.bar(BAR_CTL, NT)
            if real and pre_e_ != Int32(0):
                # pre-select result for the next pass (misc[58] = 1: _shipped_sel copies SIDX2)
                if ctid == Int32(0):
                    if fast != Int32(0) and st == Int32(ST_GO):
                        misc[58] = Int32(1)
                    misc[59] = above
                    self._stamp(mStamps, cta, 72)
            if real and pre_e_ == Int32(0):
                if ctid == Int32(0):
                    self._stamp(mStamps, cta, ST_M_SEL)
                cute.arch.fence_proxy("async.shared", space="cta")
                L.bar(BAR_CTL, NT)
                if cutlass.const_expr(self.gjoin):
                    # The flag is stable across this whole warpgroup. Do not update
                    # it here: all 128 writers must take the same fallback branch.
                    if misc[15] == Int32(0):
                        prims.mbarrier_arrive(bars + B_EAR)
                prims.mbarrier_arrive(bars + B_SEL)
                if ctid == Int32(0):
                    self._stamp(mStamps, cta, ST_M_OUT)
                    if cutlass.const_expr(self.late_done == 4):
                        if is_t0:
                            if misc[12] == Int32(P_T1):
                                if self._wait_followers(ws, gen) != Int32(0):
                                    self._publish_dir(
                                        ws,
                                        gen,
                                        Int32(1),
                                        Int32(K_DONE),
                                        Int32(P_T1),
                                        Uint64(0),
                                        Uint64(0),
                                        Uint64(0),
                                    )
                    if cutlass.const_expr(self.late_done == 3):
                        if is_t0:
                            if misc[12] == Int32(P_T1) and misc[14] != Int32(0):
                                self._publish_dir(
                                    ws,
                                    gen,
                                    Int32(1),
                                    Int32(K_DONE),
                                    Int32(P_T1),
                                    Uint64(0),
                                    Uint64(0),
                                    Uint64(0),
                                )
                if cutlass.const_expr(gargs is not None):
                    A.role_gather_team(*gargs)
            if cutlass.const_expr(eargs is not None):
                # The epilogue runs at this one call site in both passes: dry at the end of the warm-up pass on team
                # CTAs >= 2 (its single real execution then fetches L2-resident code), for real after the tail.
                if real or d >= Int32(2):
                    dry_ = Int32(1)
                    if real:
                        dry_ = Int32(0)
                    A.role_corr_epi_team(*eargs, dry=dry_)

    @cute.jit
    def _dry_acquire(self, arena, ctid, mRegion):
        NT: cutlass.Constexpr[int] = 128
        NS = self.ns
        s_coff = _sm_tensor(arena, T_OFF_COFF, Int32, 160)
        s_cu1 = _sm_tensor(arena, T_OFF_CU1, Int32, 160)
        lbase = Int32(cute.recast_ptr(arena + T_OFF_LIST, dtype=Int32).toint())
        da = Int32(0)
        db = Int32(0)
        if ctid < Int32(NS):
            da = Int32(1)
        if ctid + Int32(NT) < Int32(NS):
            db = Int32(1)
        xa, xb, xc, xd, Sa, Sb, Sc, Sd = self._scan4(da, db, Int32(0), Int32(0), ctid, arena)
        if ctid < Int32(NS):
            s_coff[ctid] = xa
            s_cu1[ctid] = da
        if ctid + Int32(NT) < Int32(NS):
            s_coff[ctid + Int32(NT)] = Sa + xb
            s_cu1[ctid + Int32(NT)] = db
        L.bar(BAR_CTL, NT)
        self._copy_runs(ctid, mRegion.iterator.toint(), Int32(0), lbase, s_cu1, s_coff)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        L.bar(BAR_CTL, NT)

    @cute.jit
    def _copy_runs(self, ctid, rg, rb, lbase, s_cu1, s_coff):
        NS = self.ns
        sub = ctid % Int32(4)
        cc = ctid // Int32(4)
        while cc < Int32(NS):
            uu = s_cu1[cc]
            oo = s_coff[cc]
            src = rg + Int64(cc) * Int64(2 * CAPR * 8) + Int64(rb) * Int64(8)
            e = sub
            while e < uu:
                cp_async8(lbase + (oo + e) * Int32(8), src + Int64(e) * Int64(8))
                e = e + Int32(4)
            cc = cc + Int32(32)

    @cute.jit
    def _acquire(self, arena, cta, d, ctid, ws, gen, n, Kq, bars, mRegion, mHist, mStamps, mCtl):
        NT: cutlass.Constexpr[int] = 128
        NS = self.ns
        is_t0 = d == Int32(0)
        lane = ctid % Int32(32)
        misc = _sm_tensor(arena, T_OFF_MISC, Int32, 64)
        # COFF is private to acquire until the rare path rebuilds its offsets.
        s_acq = _sm_tensor(arena, T_OFF_COFF, Int32, 160)
        s_cu2 = _sm_tensor(arena, T_OFF_CU2, Int32, 160)
        lbase = Int32(cute.recast_ptr(arena + T_OFF_LIST, dtype=Int32).toint())
        rg = mRegion.iterator.toint()
        T = L.f32_of_u32(Uint32(misc[8]))
        HIC = L.f32_of_u32(Uint32(misc[9]))
        T3 = L.f32_of_u32(Uint32(misc[10]))
        SMAX = L.f32_of_u32(Uint32(misc[11]))
        lo, hi = self._band_range(T, HIC, SMAX)
        mm = Int32(0)
        if not T > Float32(L.SENT_LO):
            mm = Int32(1)
        st = Int32(ST_GO)
        path = Int32(P_T1)
        rounds = Int32(0)
        if ctid < Int32(3):
            s_acq[ctid] = Int32(0)
        # Retire every warm generic LIST access before any warp can overwrite
        # LIST through the async proxy. This barrier also initializes counters.
        B.fence_proxy_async_smem()
        L.bar(BAR_CTL, NT)
        self._skew(gen, cta * Int32(4) + Int32(3), ctid)
        t64f = gen + Uint64(1)
        fol_ready = Int32(0)
        if cutlass.const_expr(self.folpre):
            if is_t0 and ctid == Int32(0):
                fol_ready = self._followers_ready(ws, t64f)
        na, nb, ab = self._poll_counts(ws, gen, Int32(0), ctid)
        fol_ok = fol_ready
        if ctid == Int32(0):
            self._stamp(mStamps, cta, 100)
        abv = Int32(0)
        if ab != Int32(0) or misc[13] != Int32(0):
            abv = Int32(1)
        pa = na + (na & Int32(1))
        pb = nb + (nb & Int32(1))
        if ctid < Int32(NS):
            s_cu2[ctid] = pa
        if ctid + Int32(NT) < Int32(NS):
            s_cu2[ctid + Int32(NT)] = pb
        # A warp may reserve and issue before the other three warps finish
        # their count polls. Each lane keeps its packet offsets in registers.
        ncopy = pa + pb
        incl = L.warp_incl_scan_add(ncopy, lane)
        wcopy = cute.arch.shuffle_sync(incl, Int32(31))
        wactual = cute.arch.warp_redux_sync(na + nb, "add")
        wab = Int32(cute.arch.warp_redux_sync(Uint32(abv), "or"))
        wbase = Int32(0)
        if lane == Int32(0):
            wbase = Int32(cute.arch.atomic_add(s_acq.iterator, wcopy, sem="relaxed", scope="cta"))
            cute.arch.red(
                s_acq.iterator + Int32(1),
                wactual,
                op="add",
                dtype=Int32,
                sem="relaxed",
                scope="cta",
            )
            cute.arch.red(
                s_acq.iterator + Int32(2), wab, op="add", dtype=Int32, sem="relaxed", scope="cta"
            )
        wbase = cute.arch.shuffle_sync(wbase, Int32(0))
        own_base = wbase + incl - ncopy
        if wbase + wcopy <= Int32(CAPL):
            if wcopy > Int32(0):
                if lane == Int32(0):
                    B.mbar_expect_tx(Int32((bars + B_LST).toint()), wcopy * Int32(8))
                cute.arch.sync_warp()
                B.fence_proxy_async_global()
                if ctid < Int32(NS) and pa > Int32(0):
                    B.bulk_g2s(
                        lbase + own_base * Int32(8),
                        rg + Int64(ctid) * Int64(2 * CAPR * 8),
                        pa * Int32(8),
                        Int32((bars + B_LST).toint()),
                    )
                if ctid + Int32(NT) < Int32(NS) and pb > Int32(0):
                    B.bulk_g2s(
                        lbase + (own_base + pa) * Int32(8),
                        rg + Int64(ctid + Int32(NT)) * Int64(2 * CAPR * 8),
                        pb * Int32(8),
                        Int32((bars + B_LST).toint()),
                    )
        # The arrival is delayed until all warps have registered their bytes.
        # Overflow copies were skipped in complete warp spans and use fallback.
        L.bar(BAR_CTL, NT)
        C = s_acq[0]
        C1 = s_acq[1]
        ab = s_acq[2]
        if ctid == Int32(0):
            self._stamp(mStamps, cta, ST_M_FIN)
            prims.mbarrier_arrive(bars + B_LST)
            self._stamp(mStamps, cta, 84)
        rare = Int32(1)
        if ab == Int32(0) and C1 >= Kq and C <= Int32(CAPL):
            rare = Int32(0)
        if cutlass.const_expr(self.rs24):
            # rs24: the T1 list path (this bulk-copied list read by _sel_list) gave wrong attention while its top-512
            # was right; rows without a bucket selection go through the rare path (T12 / DRS, list rebuilt by cp.async)
            rare = Int32(1)
        if cutlass.const_expr(self.debug):
            if mCtl[0] == Int32(1) or mCtl[0] == Int32(2):
                rare = Int32(1)
        if rare == Int32(0):
            if ctid == Int32(0):
                misc[12] = Int32(P_T1)
                if is_t0:
                    skip_d = Int32(0)
                    if cutlass.const_expr(self.debug):
                        if mCtl[1] == Int32(3):
                            skip_d = Int32(1)
                    if cutlass.const_expr(self.late_done > 0):
                        skip_d = Int32(1)
                        if fol_ok == Int32(0):
                            fol_ok = self._wait_followers(ws, gen)
                        misc[14] = fol_ok
                    if skip_d == Int32(0):
                        if fol_ok == Int32(0):
                            fol_ok = self._wait_followers(ws, gen)
                        if fol_ok != Int32(0):
                            self._publish_dir(
                                ws,
                                gen,
                                Int32(1),
                                Int32(K_DONE),
                                Int32(P_T1),
                                Uint64(0),
                                Uint64(0),
                                Uint64(0),
                            )
        A._wait(bars + B_LST, Int32(0))
        if ctid == Int32(0):
            self._stamp(mStamps, cta, 85)
        # Make every reader finish C/C1 before the rare path reuses COFF.
        L.bar(BAR_CTL, NT)
        if rare != Int32(0):
            if cutlass.const_expr(self.lean):
                st = Int32(ST_SKIP)
                C = Int32(0)
            else:
                st, path, C, lo, hi, mm, rounds = self._acquire_rare(
                    arena,
                    cta,
                    d,
                    ctid,
                    ws,
                    gen,
                    n,
                    Kq,
                    bars,
                    mRegion,
                    mHist,
                    mStamps,
                    mCtl,
                    ab,
                    C1,
                    T,
                    HIC,
                    T3,
                    SMAX,
                    lo,
                    hi,
                    mm,
                    fol_ok,
                )
        L.bar(BAR_CTL, NT)
        if ctid == Int32(0):
            self._stamp(mStamps, cta, ST_M_COPY)
        return (st, path, C, lo, hi, mm, C1, rounds)

    @cute.jit
    def _acquire_rare(
        self,
        arena,
        cta,
        d,
        ctid,
        ws,
        gen,
        n,
        Kq,
        bars,
        mRegion,
        mHist,
        mStamps,
        mCtl,
        ab,
        C1,
        T,
        HIC,
        T3,
        SMAX,
        lo,
        hi,
        mm,
        fol_ok,
    ):
        NT: cutlass.Constexpr[int] = 128
        NS = self.ns
        is_t0 = d == Int32(0)
        misc = _sm_tensor(arena, T_OFF_MISC, Int32, 64)
        s_red = _sm_tensor(arena, T_OFF_RED, Uint32, 64)
        s_coff = _sm_tensor(arena, T_OFF_COFF, Int32, 160)
        s_cu1 = _sm_tensor(arena, T_OFF_CU1, Int32, 160)
        s_cu2 = _sm_tensor(arena, T_OFF_CU2, Int32, 160)
        lbase = Int32(cute.recast_ptr(arena + T_OFF_LIST, dtype=Int32).toint())
        rg = mRegion.iterator.toint()
        st = Int32(ST_GO)
        path = Int32(P_T1)
        rounds = Int32(0)
        seq = Int32(0)
        rb = Int32(0)
        C = C1
        want_t12 = Int32(0)
        want_drs = Int32(0)
        fm = Int32(0)
        if cutlass.const_expr(self.debug):
            fm = mCtl[0]
        t3ok = Int32(0)
        if T3 > Float32(NEG_INF):
            if T3 < T:
                t3ok = Int32(1)
        if is_t0:
            if ab == Int32(0):
                abf = Int32(0)
                if ctid == Int32(0):
                    if fol_ok == Int32(0):
                        if self._wait_followers(ws, gen) == Int32(0):
                            abf = Int32(1)
                ab = self._wg_any(abf, ctid, s_red, BAR_CTL)
        if ab != Int32(0):
            st = Int32(ST_SKIP)
            path = Int32(P_ABORT)
        else:
            if C1 < Kq and t3ok != Int32(0):
                want_t12 = Int32(1)
            else:
                want_drs = Int32(1)
            if fm == Int32(1):
                if t3ok != Int32(0):
                    want_t12 = Int32(1)
                    want_drs = Int32(0)
                else:
                    want_drs = Int32(1)
            if fm == Int32(2):
                want_t12 = Int32(0)
                want_drs = Int32(1)
        if want_t12 != Int32(0):
            seq = seq + Int32(1)
            rounds = rounds + Int32(1)
            if is_t0 and ctid == Int32(0):
                klo = Uint64(L.fkey(T3)) << Uint64(18)
                khi = Uint64(L.fkey(T)) << Uint64(18)
                self._publish_dir(ws, gen, seq, Int32(K_PUB), Int32(P_T12), klo, khi, Uint64(1))
            ma, mb2, ab2 = self._poll_counts(ws, gen, Int32(1), ctid)
            ab2 = self._wg_any(ab2, ctid, s_red, BAR_CTL)
            ua = Int32(0)
            ub = Int32(0)
            if ctid < Int32(NS):
                ua = s_cu2[ctid] + ma
            if ctid + Int32(NT) < Int32(NS):
                ub = s_cu2[ctid + Int32(NT)] + mb2
            ya, yb, yc, yd, Ua, Ub, Uc, Ud = self._scan4(ua, ub, ma + mb2, Int32(0), ctid, arena)
            Ct = C1 + Uc
            if ab2 != Int32(0):
                st = Int32(ST_SKIP)
                path = Int32(P_ABORT)
            elif Ct >= Kq and Ua + Ub <= Int32(CAPL):
                path = Int32(P_T12)
                C = Ua + Ub
                lo, hi = self._band_range(T3, HIC, SMAX)
                if ctid < Int32(NS):
                    s_coff[ctid] = ya
                    s_cu1[ctid] = ua
                if ctid + Int32(NT) < Int32(NS):
                    s_coff[ctid + Int32(NT)] = Ua + yb
                    s_cu1[ctid + Int32(NT)] = ub
                seq = seq + Int32(1)
                if is_t0 and ctid == Int32(0):
                    self._publish_dir(
                        ws, gen, seq, Int32(K_DONE), Int32(P_T12), Uint64(0), Uint64(0), Uint64(0)
                    )
            else:
                want_drs = Int32(1)
            L.bar(BAR_CTL, NT)
        if want_drs != Int32(0):
            if st == Int32(ST_GO):
                ab_drs = Int32(0)
                if is_t0:
                    seq, rnd, ab_drs = self._drs_lead(arena, ctid, ws, gen, seq, Kq, mHist)
                    rounds = rounds + rnd
                ma, mb2, ab3 = self._poll_counts(ws, gen, Int32(2), ctid)
                ab3 = self._wg_any(ab3 | ab_drs, ctid, s_red, BAR_CTL)
                ya, yb, yc, yd, Ua, Ub, Uc, Ud = self._scan4(
                    ma, mb2, Int32(0), Int32(0), ctid, arena
                )
                if ab3 != Int32(0):
                    st = Int32(ST_SKIP)
                    path = Int32(P_ABORT)
                else:
                    path = Int32(P_DRS)
                    C = Ua + Ub
                    rb = Int32(CAPR)
                    mm = Int32(1)
                    if ctid < Int32(NS):
                        s_coff[ctid] = ya
                        s_cu1[ctid] = ma
                    if ctid + Int32(NT) < Int32(NS):
                        s_coff[ctid + Int32(NT)] = Ua + yb
                        s_cu1[ctid + Int32(NT)] = mb2
                    seq = seq + Int32(1)
                    if is_t0 and ctid == Int32(0):
                        self._publish_dir(
                            ws,
                            gen,
                            seq,
                            Int32(K_DONE),
                            Int32(P_DRS),
                            Uint64(0),
                            Uint64(0),
                            Uint64(0),
                        )
                L.bar(BAR_CTL, NT)
        if st != Int32(ST_GO):
            C = Int32(0)
            if is_t0 and ctid == Int32(0):
                self._publish_dir(
                    ws,
                    gen,
                    seq + Int32(1),
                    Int32(K_DONE),
                    Int32(P_ABORT),
                    Uint64(0),
                    Uint64(0),
                    Uint64(0),
                )
        if ctid == Int32(0):
            misc[12] = path
            if st != Int32(ST_GO):
                misc[13] = Int32(1)
                self._flag(ws, gen, Int32(E_ABORTED))
        L.bar(BAR_CTL, NT)
        if st == Int32(ST_GO):
            self._copy_runs(ctid, rg, rb, lbase, s_cu1, s_coff)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
        return (st, path, C, lo, hi, mm, rounds)

    @cute.jit
    def _drs_lead(self, arena, ctid, ws, gen, seq, Kq, mHist):
        NT: cutlass.Constexpr[int] = 128
        NS = self.ns
        s_hist = _sm_tensor(arena, T_OFF_HIST, Int32, NB_DRS)
        s_res = _sm_tensor(arena, T_OFF_RES, Int32, 16)
        pref = Uint64(0)
        plen = Int32(0)
        acc = Int32(0)
        need = Kq
        fin = Int32(0)
        ab = Int32(0)
        rounds = Int32(0)
        while fin == Int32(0):
            seq = seq + Int32(1)
            rounds = rounds + Int32(1)
            gh = mHist.iterator.toint() + Int64(seq) * Int64(NB_DRS * 4)
            for z_ in cutlass.range_constexpr(NB_DRS // NT):
                L.st_g_u32(gh + Int64(ctid + Int32(z_ * NT)) * Int64(4), Uint32(0))
            if ctid == Int32(0):
                L.st_g_u64(self._w(ws, W_RC + seq), Uint64(0))
            L.bar(BAR_CTL, NT)
            if ctid == Int32(0):
                self._publish_dir(
                    ws,
                    gen,
                    seq,
                    Int32(K_HIST),
                    Int32(P_DRS),
                    pref,
                    Uint64(0),
                    Uint64(Uint32(plen)) << Uint64(8),
                )
                okr = self._poll_word_eq(ws, gen, self._w(ws, W_RC + seq), Uint64(NS))
                s_res[7] = okr
                L.fence_acq_rel_gpu()
            L.bar(BAR_CTL, NT)
            if s_res[7] == Int32(0):
                ab = Int32(1)
                fin = Int32(1)
            else:
                L.fence_acq_rel_gpu()
                hv = []
                for z_ in cutlass.range_constexpr(NB_DRS // NT):
                    hv.append(L.ld_cg_i32(gh + Int64(ctid + Int32(z_ * NT)) * Int64(4)))
                for z_ in cutlass.range_constexpr(NB_DRS // NT):
                    s_hist[ctid + Int32(z_ * NT)] = hv[z_]
                L.bar(BAR_CTL, NT)
                L.find_cross(s_hist, need, ctid, s_res, NB_DRS)
                L.bar(BAR_CTL, NT)
                dg = s_res[L.RES_B]
                ad = s_res[L.RES_ABOVE]
                md = s_res[L.RES_M]
                L.bar(BAR_CTL, NT)
                acc = acc + ad
                need = need - ad
                pref = pref << Uint64(DRS_DIG) | Uint64(Uint32(dg))
                plen = plen + Int32(DRS_DIG)
                if acc + md <= Int32(CAPL) or plen >= Int32(K50_BITS):
                    fin = Int32(1)
        if ab == Int32(0):
            seq = seq + Int32(1)
            if ctid == Int32(0):
                klo = pref << Uint64(Int32(K50_BITS) - plen)
                self._publish_dir(
                    ws,
                    gen,
                    seq,
                    Int32(K_PUB),
                    Int32(P_DRS),
                    klo,
                    Uint64(1) << Uint64(K50_BITS),
                    Uint64(2),
                )
        return (seq, rounds, ab)

    @cute.jit
    def _wg_minmax_u64(self, kmn, kmx, s_red, ctid):
        lane = ctid % Int32(32)
        wid = ctid // Int32(32)
        mh = Uint32(kmn >> Uint64(32))
        ml = Uint32(kmn & Uint64(4294967295))
        wmh = L.warp_min_u32(mh)
        c1_ = Uint32(4294967295)
        if mh == wmh:
            c1_ = ml
        wml = L.warp_min_u32(c1_)
        xh = Uint32(kmx >> Uint64(32))
        xl = Uint32(kmx & Uint64(4294967295))
        wxh = L.warp_max_u32(xh)
        c2_ = Uint32(0)
        if xh == wxh:
            c2_ = xl
        wxl = L.warp_max_u32(c2_)
        if lane == Int32(0):
            s_red[Int32(16) + wid * Int32(4)] = wmh
            s_red[Int32(17) + wid * Int32(4)] = wml
            s_red[Int32(18) + wid * Int32(4)] = wxh
            s_red[Int32(19) + wid * Int32(4)] = wxl
        L.bar(BAR_CTL, 128)
        gmh = Uint32(4294967295)
        gml = Uint32(4294967295)
        gxh = Uint32(0)
        gxl = Uint32(0)
        for w_ in cutlass.range_constexpr(4):
            a_ = Uint32(s_red[16 + 4 * w_])
            b_ = Uint32(s_red[17 + 4 * w_])
            c_ = Uint32(s_red[18 + 4 * w_])
            e_ = Uint32(s_red[19 + 4 * w_])
            if a_ < gmh:
                gmh = a_
                gml = b_
            elif a_ == gmh:
                if b_ < gml:
                    gml = b_
            if c_ > gxh:
                gxh = c_
                gxl = e_
            elif c_ == gxh:
                if e_ > gxl:
                    gxl = e_
        L.bar(BAR_CTL, 128)
        return (Uint64(gmh) << Uint64(32) | Uint64(gml), Uint64(gxh) << Uint64(32) | Uint64(gxl))

    @cute.jit
    def _sel_list(
        self,
        arena,
        cta,
        ctid,
        C,
        Kw,
        lo,
        hi,
        mm,
        fnar,
        real,
        st,
        bars,
        n,
        s_idx,
        mBtC,
        mbc,
        mStamps,
        d,
    ):
        NT: cutlass.Constexpr[int] = 128
        NWC: cutlass.Constexpr[int] = NT // 32
        lane = ctid % Int32(32)
        wid = ctid // Int32(32)
        misc = _sm_tensor(arena, T_OFF_MISC, Int32, 64)
        s_res = _sm_tensor(arena, T_OFF_RES, Int32, 16)
        s_red = _sm_tensor(arena, T_OFF_RED, Uint32, 64)
        s_wc = _sm_tensor(arena, T_OFF_WC, Int32, 64)
        s_hist = _sm_tensor(arena, T_OFF_HIST, Int32, NB_SEL)
        s_phist = _sm_tensor(arena, T_OFF_HIST, Uint64, NB_INITIAL)
        s_outs = _sm_tensor(arena, T_OFF_OUTS, Int32, K_TOP)
        s_ck = _sm_tensor(arena, T_OFF_CK, Uint64, S.QUADC)
        s_cs = _sm_tensor(arena, T_OFF_CS, Int32, S.QUADC)
        s_btc = _sm_tensor(arena, T_OFF_BTC, Int32, BTC_CAP)
        s_crow = _sm_tensor(arena, T_OFF_CROW, Int32, S.QUADC)
        s_lst = _sm_tensor(arena, T_OFF_LIST, Uint64, CAPL)
        mret = Int32(0)
        info = Int32(0)
        Bb = Int32(-1)
        Bw = Int32(-1)
        nout = Int32(0)
        Sw = Int32(0)
        Sx = Int32(0)
        pw = Int32(0)
        px = Int32(0)
        need = Int32(0)
        sc = Float32(0.0)
        CW = (C + Int32(NT - 1)) // Int32(NT) * Int32(32)
        c0 = wid * CW
        c1 = cutlass.min(C, c0 + CW)
        if st == Int32(ST_GO):
            nout = cutlass.min(Kw, C)
            if mm != Int32(0):
                kmn = Uint32(4294967295)
                kmx = Uint32(0)
                kq = ctid
                while kq < C:
                    e_ = Uint64(s_lst[kq])
                    kb_ = L.fkey_bits(Uint32(e_ & Uint64(4294967295)))
                    if Uint32(e_ >> Uint64(32)) != Uint32(DUMMY_GPOS):
                        if kb_ < kmn:
                            kmn = kb_
                        if kb_ > kmx:
                            kmx = kb_
                    kq = kq + Int32(NT)
                gmn, gmx = L._block_min_max_u32(kmn, kmx, s_red, ctid, NT, BAR_CTL)
                lo = L.invkey(gmn)
                hi = L.invkey(gmx)
            sc = S.bin_scale(lo, hi) * Float32(0.5)
            for z_ in cutlass.range_constexpr(NB_INITIAL // NT):
                s_phist[ctid + Int32(z_ * NT)] = Uint64(0)
            L.bar(BAR_CTL, NT)
            hpair = wid >> Int32(1)
            hinc = Int32(1) << ((wid & Int32(1)) * Int32(16))
            i4 = c0 + lane
            while i4 < c1:
                bq_ = [Int32(0)] * 4
                for q in cutlass.range_constexpr(4):
                    icl = cutlass.min(i4 + Int32(32 * q), c1 - Int32(1))
                    bq_[q] = L._vbin(Uint64(s_lst[icl]), lo, sc, NB_INITIAL)
                for q in cutlass.range_constexpr(4):
                    if i4 + Int32(32 * q) < c1:
                        cute.arch.red(
                            s_hist.iterator + bq_[q] * Int32(2) + hpair,
                            hinc,
                            op="add",
                            dtype=Int32,
                            sem="relaxed",
                            scope="cta",
                        )
                i4 = i4 + Int32(128)
            L.bar(BAR_CTL, NT)
            if ctid == Int32(0) and real:
                self._stamp(mStamps, cta, 87)
        st = self._warm_abort(real, st, ctid, bars, misc)
        if st == Int32(ST_GO):
            find_cross_warp_counts(s_phist, Kw, ctid, s_res, s_wc, NB_INITIAL)
            L.bar(BAR_CTL, NT)
            Bb = s_res[L.RES_B]
            above = s_res[L.RES_ABOVE]
            mret = s_res[L.RES_M]
            need = Kw - above
            Bw = Bb
            Sw = above
            Sx = mret
            if need >= mret:
                Bw = Bb - Int32(1)
                Sw = above + mret
                Sx = Int32(0)
            else:
                info = Int32(1)
            pw = s_wc[wid]
            px = s_wc[Int32(NWC) + wid]
            if ctid == Int32(0) and real:
                self._stamp(mStamps, cta, 88)
        st = self._warm_abort(real, st, ctid, bars, misc)
        if st == Int32(ST_GO):
            if real:
                A._wait(bars + B_BTC, Int32(0))
            npg_s = (
                cutlass.min((n + Int32(31)) // Int32(32), cutlass.min(mbc, Int32(BTC_CAP)))
                // Int32(4)
                * Int32(4)
            )
            lm = Int32(cute.arch.lanemask_lt())
            nkey = Int32(0)
            if Sw < Kw:
                if Sx > Int32(S.QUADC) or fnar != Int32(0):
                    nkey = Int32(1)
                    # s_nkey aliases KV ring slots 0-2: wait for their first release by this CTA's QK. With rs24, row
                    # group 1 (odd d: tiles 3 and 4) fills the ring only after this selection's B_EAR / B_SEL arrive,
                    # so its slots are still unused here and are never released before then (waiting deadlocks).
                    rg0_ = Int32(1)
                    if cutlass.const_expr(self.rs24):
                        rg0_ = Int32(1) - (d & Int32(1))
                    if rg0_ != Int32(0):
                        for sl_ in cutlass.range_constexpr(3):
                            A._wait(bars + (B_KVE + sl_), Int32(0))
            s_nkey = _sm_tensor(arena, T_OFF_NKEY, Uint64, CAPL)
            kb = c0
            while kb < c1:
                sl = kb + lane
                e = Uint64(s_lst[cutlass.min(sl, c1 - Int32(1))])
                fw = Int32(0)
                fc = Int32(0)
                gp = Int32(Uint32(e >> Uint64(32)))
                if sl < c1:
                    bq = L._vbin(e, lo, sc, NB_INITIAL)
                    if bq > Bw:
                        fw = Int32(1)
                    elif bq == Bb:
                        fc = Int32(1)
                mw = Int32(cute.arch.vote_ballot_sync(fw != Int32(0)))
                mc = Int32(cute.arch.vote_ballot_sync(fc != Int32(0)))
                if fw != Int32(0) or fc != Int32(0):
                    row = self._k3row(gp, npg_s, s_btc, mBtC, mbc, real)
                    if fw != Int32(0):
                        po = pw + Int32(cute.arch.popc(mw & lm))
                        if po < Kw:
                            s_outs[po] = gp
                            (s_idx + Int32(128) + po).store(row)
                    else:
                        pc_ = px + Int32(cute.arch.popc(mc & lm))
                        if pc_ < Int32(S.QUADC):
                            s_ck[pc_] = L.entry_key(e)
                            s_cs[pc_] = gp
                            s_crow[pc_] = row
                        if nkey != Int32(0):
                            s_nkey[pc_] = L.entry_key(e)
                pw = pw + Int32(cute.arch.popc(mw))
                px = px + Int32(cute.arch.popc(mc))
                kb = kb + Int32(32)
            L.bar(BAR_CTL, NT)
            if cutlass.const_expr(self.gjoin):
                if real and Sw >= Int32(384) and nkey == Int32(0):
                    # The first three compressed tiles are exact winners. All
                    # ring-aliased selection state is dead on this path.
                    cute.arch.fence_proxy("async.shared", space="cta")
                    L.bar(BAR_CTL, NT)
                    if ctid == Int32(0):
                        misc[15] = Int32(1)
                    prims.mbarrier_arrive(bars + B_EAR)
            if ctid == Int32(0) and real:
                self._stamp(mStamps, cta, 70)
            if Sw < Kw:
                if Sx <= Int32(32) and fnar == Int32(0):
                    if ctid < Int32(32):
                        kq_ = Uint64(0)
                        if ctid < Sx:
                            kq_ = Uint64(s_ck[ctid])
                        khi = Uint32(kq_ >> Uint64(32))
                        klo = Uint32(kq_ & Uint64(4294967295))
                        r_ = Int32(0)
                        for j_ in cutlass.range(Sx, unroll=1):
                            ohi = cute.arch.shuffle_sync(khi, Int32(j_))
                            olo = cute.arch.shuffle_sync(klo, Int32(j_))
                            if Int32(j_) < Sx:
                                if ohi > khi:
                                    r_ = r_ + Int32(1)
                                elif ohi == khi:
                                    if olo > klo:
                                        r_ = r_ + Int32(1)
                        if ctid < Sx:
                            if r_ < Kw - Sw:
                                s_outs[Sw + r_] = s_cs[ctid]
                                (s_idx + Int32(128) + Sw + r_).store(s_crow[ctid])
                elif Sx <= Int32(S.QUADC) and fnar == Int32(0):
                    i = ctid
                    while i < Sx:
                        ki = Uint64(s_ck[i])
                        r_ = Int32(0)
                        jq = Int32(0)
                        while jq < Sx:
                            if Uint64(s_ck[jq]) > ki:
                                r_ = r_ + Int32(1)
                            jq = jq + Int32(1)
                        if r_ < Kw - Sw:
                            s_outs[Sw + r_] = s_cs[i]
                            (s_idx + Int32(128) + Sw + r_).store(s_crow[i])
                        i = i + Int32(NT)
                else:
                    info = Int32(2)
                    if cutlass.const_expr(not self.lean):
                        self._narrow(
                            arena,
                            ctid,
                            Sx,
                            Sw,
                            Kw - Sw,
                            npg_s,
                            mBtC,
                            mbc,
                            real,
                            s_idx,
                            mStamps,
                            cta,
                        )
            L.bar(BAR_CTL, NT)
        return (st, mret, info, Bb, nout)

    @cute.jit
    def _pmask(self, pref, plen):
        pm = Uint64(0)
        pv = Uint64(0)
        if plen > Int32(0):
            pm = Uint64(18446744073709551615) << Uint64(Int32(64) - plen)
            pv = pref << Uint64(Int32(64) - plen)
        return (pm, pv)

    @cute.jit
    def _narrow(self, arena, ctid, m, Sw, need, npg_s, mBtC, mbc, real, s_idx, mStamps, cta):
        NT: cutlass.Constexpr[int] = 128
        lane = ctid % Int32(32)
        wid = ctid // Int32(32)
        s_res = _sm_tensor(arena, T_OFF_RES, Int32, 16)
        s_red = _sm_tensor(arena, T_OFF_RED, Uint32, 64)
        s_wc = _sm_tensor(arena, T_OFF_WC, Int32, 64)
        s_hist = _sm_tensor(arena, T_OFF_HIST, Int32, NB_SEL)
        s_outs = _sm_tensor(arena, T_OFF_OUTS, Int32, K_TOP)
        s_btc = _sm_tensor(arena, T_OFF_BTC, Int32, BTC_CAP)
        s_key = _sm_tensor(arena, T_OFF_NKEY, Uint64, CAPL)
        L.bar(BAR_CTL, NT)
        if ctid == Int32(0):
            self._stamp(mStamps, cta, 110)
        plen = Int32(0)
        pref = Uint64(0)
        needr = need
        done = Int32(0)
        lmk = Int32(cute.arch.lanemask_lt())
        nit = (m + Int32(NT * 4 - 1)) // Int32(NT * 4)
        while done == Int32(0):
            pm0, pv0 = self._pmask(pref, plen)
            kmn = Uint64(18446744073709551615)
            kmx = Uint64(0)
            jj = Int32(0)
            while jj < nit:
                i0 = jj * Int32(NT * 4) + ctid
                kq_ = []
                for q in cutlass.range_constexpr(4):
                    kq_.append(Uint64(s_key[cutlass.min(i0 + Int32(NT * q), m - Int32(1))]))
                for q in cutlass.range_constexpr(4):
                    if i0 + Int32(NT * q) < m:
                        if kq_[q] & pm0 == pv0:
                            if kq_[q] < kmn:
                                kmn = kq_[q]
                            if kq_[q] > kmx:
                                kmx = kq_[q]
                jj = jj + Int32(1)
            gmn, gmx = self._wg_minmax_u64(kmn, kmx, s_red, ctid)
            if ctid == Int32(0):
                self._stamp(mStamps, cta, 111)
            xr = gmn ^ gmx
            cpl = Int32(64)
            if xr != Uint64(0):
                cpl = clz_u64(xr)
            if cpl > plen:
                plen = cpl
                pref = gmn >> Uint64(Int32(64) - plen)
            if plen >= Int32(64):
                done = Int32(1)
            if done == Int32(0):
                nbd = cutlass.min(Int32(10), Int32(64) - plen)
                shift = Uint64(Int32(64) - plen - nbd)
                pm1, pv1 = self._pmask(pref, plen)
                dmask = Uint64((Int32(1) << nbd) - Int32(1))
                for z_ in cutlass.range_constexpr(NB_SEL // NT):
                    s_hist[ctid + Int32(z_ * NT)] = Int32(0)
                L.bar(BAR_CTL, NT)
                jj = Int32(0)
                while jj < nit:
                    i0 = jj * Int32(NT * 4) + ctid
                    dgs = []
                    for q in cutlass.range_constexpr(4):
                        kq = i0 + Int32(NT * q)
                        k_ = Uint64(s_key[cutlass.min(kq, m - Int32(1))])
                        dg = Int32(NB_SEL)
                        if kq < m:
                            if k_ & pm1 == pv1:
                                dg = Int32(Uint32(k_ >> shift & dmask))
                        dgs.append(dg)
                    for q in cutlass.range_constexpr(4):
                        mm_ = match_any_b32(dgs[q])
                        if dgs[q] != Int32(NB_SEL):
                            if mm_ & lmk == Int32(0):
                                cute.arch.red(
                                    s_hist.iterator + dgs[q],
                                    Int32(cute.arch.popc(mm_)),
                                    op="add",
                                    dtype=Int32,
                                    sem="relaxed",
                                    scope="cta",
                                )
                    jj = jj + Int32(1)
                L.bar(BAR_CTL, NT)
                L.find_cross(s_hist, needr, ctid, s_res, NB_SEL)
                L.bar(BAR_CTL, NT)
                dB = s_res[L.RES_B]
                dab = s_res[L.RES_ABOVE]
                dm = s_res[L.RES_M]
                L.bar(BAR_CTL, NT)
                pref = pref << Uint64(nbd) | Uint64(Uint32(dB))
                plen = plen + nbd
                needr = needr - dab
                if ctid == Int32(0):
                    if cutlass.const_expr(self.stamps):
                        rr_ = cutlass.min(plen // Int32(8), Int32(7))
                        (mStamps.iterator + cta * NSTAMP + Int32(112) + rr_).store(
                            Int64(L.gtimer())
                        )
                if needr >= dm or plen >= Int32(64):
                    done = Int32(1)
        if ctid == Int32(0):
            self._stamp(mStamps, cta, 120)
        thr = pref
        if plen < Int32(64):
            thr = pref << Uint64(Int32(64) - plen)
        lm = Int32(cute.arch.lanemask_lt())
        CW = (m + Int32(NT * 4 - 1)) // Int32(NT * 4) * Int32(128)
        e0 = wid * CW
        e1 = cutlass.min(m, e0 + CW)
        cnt = Int32(0)
        kb = e0
        while kb < e1:
            fs = []
            for q in cutlass.range_constexpr(4):
                sl = kb + Int32(32 * q) + lane
                k_ = Uint64(s_key[cutlass.min(sl, e1 - Int32(1))])
                f2 = Int32(0)
                if sl < e1:
                    if k_ >= thr:
                        f2 = Int32(1)
                fs.append(f2)
            for q in cutlass.range_constexpr(4):
                cnt = cnt + Int32(cute.arch.popc(cute.arch.vote_ballot_sync(fs[q] != Int32(0))))
            kb = kb + Int32(128)
        ex2, tot2 = L._warp_prefix(cnt, s_wc, ctid, NT, BAR_CTL)
        p2 = Sw + ex2
        kb = e0
        while kb < e1:
            ks = []
            for q in cutlass.range_constexpr(4):
                ks.append(Uint64(s_key[cutlass.min(kb + Int32(32 * q) + lane, e1 - Int32(1))]))
            for q in cutlass.range_constexpr(4):
                sl = kb + Int32(32 * q) + lane
                f2 = Int32(0)
                if sl < e1:
                    if ks[q] >= thr:
                        f2 = Int32(1)
                m2 = Int32(cute.arch.vote_ballot_sync(f2 != Int32(0)))
                if f2 != Int32(0):
                    so = p2 + Int32(cute.arch.popc(m2 & lm))
                    gp = Int32(Uint32(4294967295) - Uint32(ks[q] & Uint64(4294967295)))
                    if so < Int32(K_TOP):
                        s_outs[so] = gp
                        (s_idx + Int32(128) + so).store(
                            self._k3row(gp, npg_s, s_btc, mBtC, mbc, real)
                        )
                p2 = p2 + Int32(cute.arch.popc(m2))
            kb = kb + Int32(128)
        L.bar(BAR_CTL, NT)


_COMPILED = {}


def compile_mk(mk: Dsv4SparseDecodeKernel):
    """Compile ``mk`` with TVM-FFI over symbolic shapes; returns the callable that ``op.run`` launches."""
    key = tuple(
        sorted(((k, v) for k, v in vars(mk).items() if isinstance(v, (int, float, bool, str))))
    )
    if key in _COMPILED:
        return _COMPILED[key]
    rt = cute.runtime
    s = cute.sym_int
    block_bytes = mk.phys_block_kv * (mk.head_dim + 4)
    fk = [
        rt.make_fake_tensor(cutlass.Uint8, (s(), block_bytes), stride=(cute.sym_int64(), 1)),
        rt.make_fake_compact_tensor(cutlass.Uint8, (64, 128, mk.nrows), stride_order=(1, 0, 2)),
        rt.make_fake_compact_tensor(cutlass.Float32, (64, mk.nrows), stride_order=(0, 1)),
        rt.make_fake_compact_tensor(Int32, (mk.nrows, s()), stride_order=(1, 0)),
        rt.make_fake_compact_tensor(Int32, (mk.nrows,), stride_order=(0,)),
        rt.make_fake_compact_tensor(Int32, (s(),), stride_order=(0,)),
        rt.make_fake_compact_tensor(
            cutlass.Uint8, (mk.nrows, 64, 512), stride_order=(2, 1, 0), assumed_align=16
        ),
        rt.make_fake_compact_tensor(
            cutlass.Uint8, (cute.sym_int64(), 512), stride_order=(1, 0), assumed_align=16
        ),
        rt.make_fake_compact_tensor(
            cutlass.Uint8, (cute.sym_int64(), 512), stride_order=(1, 0), assumed_align=16
        ),
        rt.make_fake_compact_tensor(Int32, (mk.nrows, 128), stride_order=(1, 0)),
        rt.make_fake_compact_tensor(Int32, (mk.nrows, s()), stride_order=(1, 0)),
        rt.make_fake_compact_tensor(Int32, (mk.nrows, s()), stride_order=(1, 0)),
        rt.make_fake_compact_tensor(Float32, (64,), stride_order=(0,), assumed_align=16),
        rt.make_fake_compact_tensor(Int32, (mk.nrows,), stride_order=(0,)),
        rt.make_fake_compact_tensor(Int32, (mk.nrows, 512), stride_order=(1, 0)),
        rt.make_fake_compact_tensor(Int32, (mk.nrows, 640), stride_order=(1, 0)),
        rt.make_fake_compact_tensor(
            cutlass.BFloat16, (mk.nrows, 64 * 512), stride_order=(1, 0), assumed_align=128
        ),
        rt.make_fake_compact_tensor(Int64, (s(),), stride_order=(0,), assumed_align=16),
        rt.make_fake_compact_tensor(Int32, (s(),), stride_order=(0,), assumed_align=16),
        rt.make_fake_compact_tensor(Int64, (s(),), stride_order=(0,), assumed_align=16),
        rt.make_fake_compact_tensor(Int64, (s(),), stride_order=(0,), assumed_align=16),
        rt.make_fake_compact_tensor(Float32, (s(),), stride_order=(0,), assumed_align=16),
        rt.make_fake_compact_tensor(Int64, (s(),), stride_order=(0,), assumed_align=16),
        rt.make_fake_compact_tensor(Int32, (16 * mk.nrows,), stride_order=(0,)),
    ]
    stream = rt.make_fake_stream(use_tvm_ffi_env_stream=True)
    c = cute.compile(
        mk, *fk, Int32(1), Float32(1.0), stream, options="--enable-tvm-ffi --generate-line-info"
    )
    _COMPILED[key] = c
    return c


def ladder_host(n: int):
    import numpy as np

    k = K_TOP
    if n <= k:
        return (0, 1)
    x6 = 6 * n
    ri = int(math.isqrt(x6))
    r6 = ri + 1 if x6 - ri * ri > ri else ri
    aim = max(PLAN["aim_base"], r6)
    aim = max(aim, PLAN["amin"])
    aim = min(aim, PLAN["scap"] >> 1)
    aim = max(aim, k)
    n4w = n >> 2
    gate = 1 if n > PLAN["scap"] else 0
    if n4w < 4:
        gate = 0
    if not gate:
        return (0, 1)
    f32 = np.float32
    sel = int(f32(PLAN["sfac"] * n) * (f32(1.0) / f32(aim)))
    sel = max(sel, 256)
    sel = min(sel, n >> 1)
    pairs = max(sel >> 3, 1)
    half = max(n4w >> 1, 1)
    pairs = min(pairs, half)
    ss2 = max(int(f32(half) * (f32(1.0) / f32(pairs))), 1)
    smp = int(f32(half) * (f32(1.0) / f32(ss2)))
    if smp * ss2 > half:
        smp -= 1
    return (max(smp, 1), ss2)


_SMP_MAX = None


def ladder_max_smp() -> int:
    global _SMP_MAX
    if _SMP_MAX is None:
        _SMP_MAX = max((ladder_host(n)[0] for n in range(N_MAX - 4096, N_MAX + 1)))
    return _SMP_MAX
