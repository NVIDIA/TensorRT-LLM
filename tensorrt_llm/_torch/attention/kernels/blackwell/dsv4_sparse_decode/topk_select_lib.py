# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Selection primitives: sortable float keys, histograms, crossing-bin search, warp and block reductions and memory
helpers for the threshold and the exact top-512.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.cute.math as cmath
from cutlass import Float32, Int32, Int64, Uint32, Uint64
from cutlass._mlir.dialects import arith as mlir_arith
from cutlass._mlir.dialects import llvm
from cutlass._mlir.dialects import math as mlir_math
from cutlass.cutlass_dsl import T, dsl_user_op

K_TOP = 512
NBS = 256
QUADC = 288
NB_SEL = 1024
SENT_LO = -3e38
SENT_HI = 3e38
NEG_INF = float("-inf")
RES_B, RES_M, RES_ABOVE, RES_TOT, RES_B2, RES_B3 = (0, 1, 2, 3, 4, 5)
GEN_SHIFT = 16
GEN_ONE = 1 << GEN_SHIFT
GRAIN = 32
SWA = 128
TPB_SWA = 128
TPB_C = 32
V_OK = 1
V_FB_RADIX = 2
V_FB_RECOMP = 3
PATH_SHORT, PATH_T1, PATH_T12, PATH_FB_RADIX, PATH_FB_RECOMP, PATH_FLOOR = (1, 2, 3, 4, 5, 6)


def _asm(res_t, args, s, cons, loc, ip, side=True):
    return llvm.inline_asm(
        res_t,
        args,
        s,
        cons,
        has_side_effects=side,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _v(x, loc, ip):
    return x.ir_value(loc=loc, ip=ip)


@dsl_user_op
def gtimer(*, loc=None, ip=None):
    return Int64(_asm(T.i64(), [], "mov.u64 $0, %globaltimer;", "=l", loc, ip))


@dsl_user_op
def clk64(*, loc=None, ip=None):
    return Int64(_asm(T.i64(), [], "mov.u64 $0, %clock64;", "=l", loc, ip))


@dsl_user_op
def nanosleep(ns, *, loc=None, ip=None):
    _asm(None, [_v(ns, loc, ip)], "nanosleep.u32 $0;", "r", loc, ip)


@dsl_user_op
def prefetch_l2(addr, *, loc=None, ip=None):
    _asm(None, [_v(addr, loc, ip)], "prefetch.global.L2 [$0];", "l", loc, ip)


@dsl_user_op
def fence_acq_rel_gpu(*, loc=None, ip=None):
    _asm(None, [], "fence.acq_rel.gpu;", "", loc, ip)


@dsl_user_op
def atom_add_u64(addr, val, *, loc=None, ip=None):
    return Uint64(
        _asm(
            T.i64(),
            [_v(addr, loc, ip), _v(val, loc, ip)],
            "atom.relaxed.gpu.global.add.u64 $0, [$1], $2;",
            "=l,l,l",
            loc,
            ip,
        )
    )


@dsl_user_op
def red_add_u64(addr, val, *, loc=None, ip=None):
    _asm(
        None,
        [_v(addr, loc, ip), _v(val, loc, ip)],
        "red.relaxed.gpu.global.add.u64 [$0], $1;",
        "l,l",
        loc,
        ip,
    )


@dsl_user_op
def red_add_u32(addr, val, *, loc=None, ip=None):
    _asm(
        None,
        [_v(addr, loc, ip), _v(val, loc, ip)],
        "red.relaxed.gpu.global.add.u32 [$0], $1;",
        "l,r",
        loc,
        ip,
    )


@dsl_user_op
def ld_relaxed_u64(addr, *, loc=None, ip=None):
    return Uint64(
        _asm(T.i64(), [_v(addr, loc, ip)], "ld.relaxed.gpu.global.u64 $0, [$1];", "=l,l", loc, ip)
    )


@dsl_user_op
def ld_acquire_u64(addr, *, loc=None, ip=None):
    return Uint64(
        _asm(T.i64(), [_v(addr, loc, ip)], "ld.acquire.gpu.global.u64 $0, [$1];", "=l,l", loc, ip)
    )


@dsl_user_op
def st_release_u64(addr, val, *, loc=None, ip=None):
    _asm(
        None,
        [_v(addr, loc, ip), _v(val, loc, ip)],
        "st.release.gpu.global.u64 [$0], $1;",
        "l,l",
        loc,
        ip,
    )


@dsl_user_op
def ld_cg_u32(addr, *, loc=None, ip=None):
    return Uint32(_asm(T.i32(), [_v(addr, loc, ip)], "ld.global.cg.u32 $0, [$1];", "=r,l", loc, ip))


@dsl_user_op
def ld_cg_i32(addr, *, loc=None, ip=None):
    return Int32(_asm(T.i32(), [_v(addr, loc, ip)], "ld.global.cg.u32 $0, [$1];", "=r,l", loc, ip))


@dsl_user_op
def ld_cg_u64(addr, *, loc=None, ip=None):
    return Uint64(_asm(T.i64(), [_v(addr, loc, ip)], "ld.global.cg.u64 $0, [$1];", "=l,l", loc, ip))


@dsl_user_op
def ld_cg_f32(addr, *, loc=None, ip=None):
    return Float32(
        _asm(T.f32(), [_v(addr, loc, ip)], "ld.global.cg.f32 $0, [$1];", "=f,l", loc, ip)
    )


@dsl_user_op
def st_g_u64(addr, val, *, loc=None, ip=None):
    _asm(None, [_v(addr, loc, ip), _v(val, loc, ip)], "st.global.u64 [$0], $1;", "l,l", loc, ip)


@dsl_user_op
def st_g_u32(addr, val, *, loc=None, ip=None):
    _asm(None, [_v(addr, loc, ip), _v(val, loc, ip)], "st.global.u32 [$0], $1;", "l,r", loc, ip)


@dsl_user_op
def st_g_f32(addr, val, *, loc=None, ip=None):
    _asm(None, [_v(addr, loc, ip), _v(val, loc, ip)], "st.global.f32 [$0], $1;", "l,f", loc, ip)


@dsl_user_op
def ld_nc_f32x4(addr, *, loc=None, ip=None):
    st = llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()])
    r = _asm(
        st,
        [_v(addr, loc, ip)],
        "ld.global.nc.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,l",
        loc,
        ip,
        side=False,
    )
    return tuple((Float32(llvm.extractvalue(T.f32(), r, [i], loc=loc, ip=ip)) for i in range(4)))


@dsl_user_op
def ld_nc_f32(addr, *, loc=None, ip=None):
    return Float32(
        _asm(
            T.f32(), [_v(addr, loc, ip)], "ld.global.nc.f32 $0, [$1];", "=f,l", loc, ip, side=False
        )
    )


@dsl_user_op
def ld_cg_f32x4(addr, *, loc=None, ip=None):
    st = llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()])
    r = _asm(
        st,
        [_v(addr, loc, ip)],
        "ld.global.cg.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,l",
        loc,
        ip,
    )
    return tuple((Float32(llvm.extractvalue(T.f32(), r, [i], loc=loc, ip=ip)) for i in range(4)))


@dsl_user_op
def bar(bar_id, nthr, *, loc=None, ip=None):
    # Warp-specialized participants may reach this named barrier at distinct PCs.
    _asm(
        None,
        [_v(Int32(bar_id), loc, ip), _v(Int32(nthr), loc, ip)],
        "barrier.cta.sync $0, $1;",
        "r,r,~{memory}",
        loc,
        ip,
    )


@dsl_user_op
def bar_arrive(bar_id, nthr, *, loc=None, ip=None):
    _asm(
        None,
        [_v(Int32(bar_id), loc, ip), _v(Int32(nthr), loc, ip)],
        "barrier.cta.arrive $0, $1;",
        "r,r,~{memory}",
        loc,
        ip,
    )


def u32_of_f32(v):
    return Uint32(llvm.bitcast(Uint32.mlir_type, v.ir_value()))


def f32_of_u32(u):
    return Float32(llvm.bitcast(Float32.mlir_type, u.ir_value()))


def fkey_bits(u):
    neg = Uint32(0) - (u >> Uint32(31))
    return u ^ (neg | Uint32(2147483648))


def fkey(x):
    return fkey_bits(u32_of_f32(x))


def invkey_bits(K):
    s = K >> Uint32(31)
    m = s - Uint32(1) | Uint32(2147483648)
    return K ^ m


def invkey(K):
    return f32_of_u32(invkey_bits(K))


def warp_min_u32(v):
    return cute.arch.warp_redux_sync(v, "min")


def warp_max_u32(v):
    return cute.arch.warp_redux_sync(v, "max")


def f2u_rz(v):
    return Uint32(v)


def f2s_rz(v):
    return Int32(v)


@dsl_user_op
def _fmaf(a, b, c, *, loc=None, ip=None):
    return Float32(
        mlir_math.fma(
            a.ir_value(loc=loc, ip=ip),
            b.ir_value(loc=loc, ip=ip),
            c.ir_value(loc=loc, ip=ip),
            fastmath=mlir_arith.FastMathFlags.none,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def _shfl_up_add(val, lane, offset: cutlass.Constexpr):
    other = cute.arch.shuffle_sync_up(val, offset, mask_and_clamp=0)
    if lane >= Int32(offset):
        val = val + other
    return val


@cute.jit
def warp_incl_scan_add(val, lane):
    for o in [1, 2, 4, 8, 16]:
        val = _shfl_up_add(val, lane, o)
    return val


def smem_atom_i32_128():
    return cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Int32, num_bits_per_copy=128)


def _smem_v4_tensor(base_addr, byte_off):
    p = cute.make_ptr(Int32, base_addr + byte_off, cute.AddressSpace.smem, assumed_align=16)
    return cute.make_tensor(p, cute.make_layout((4,)))


def lds128_i32(copy_atom, base_addr, byte_off, frag):
    cute.copy(copy_atom, _smem_v4_tensor(base_addr, byte_off), frag)


def sts128_i32(copy_atom, frag, base_addr, byte_off):
    cute.copy(copy_atom, frag, _smem_v4_tensor(base_addr, byte_off))


@cute.jit
def scan_cross0(
    s_hist,
    target,
    tidx,
    s_res,
    target2,
    target3,
    s_addv,
    nb: cutlass.Constexpr,
    zero: cutlass.Constexpr,
    two: cutlass.Constexpr = False,
    three: cutlass.Constexpr = False,
    addf: cutlass.Constexpr = False,
):
    BPT = nb // 32
    NV = BPT // 4
    HOLD = NV <= 2
    if tidx < Int32(32):
        lane = tidx
        atom = smem_atom_i32_128()
        hbase = s_hist.iterator.toint()
        frags = [cute.make_rmem_tensor((4,), Int32) for _ in range(NV)]
        sm = Int32(0)
        for q in cutlass.range_constexpr(NV):
            boff = (lane * Int32(NV) + Int32(q)) * Int32(16)
            lds128_i32(atom, hbase, boff, frags[q])
            sm = sm + frags[q][0] + frags[q][1] + frags[q][2] + frags[q][3]
        w = warp_incl_scan_add(sm, lane)
        tot = cute.arch.shuffle_sync(w, Int32(31))
        after = tot - w
        if lane == Int32(0):
            s_res[RES_TOT] = tot
        base = lane * Int32(BPT)
        for q in cutlass.range_constexpr(NV - 1, -1, -1):
            if cutlass.const_expr(HOLD):
                vv = frags[q]
            else:
                vv = cute.make_rmem_tensor((4,), Int32)
                boff = (lane * Int32(NV) + Int32(q)) * Int32(16)
                lds128_i32(atom, hbase, boff, vv)
            o4 = cute.make_rmem_tensor((4,), Int32)
            for j in cutlass.range_constexpr(3, -1, -1):
                cq = vv[j]
                if cutlass.const_expr(zero):
                    o4[j] = Int32(0)
                else:
                    o4[j] = after
                gb = base + Int32(4 * q + j)
                cross = Int32(0)
                if after < target:
                    if after + cq >= target:
                        cross = Int32(1)
                    if gb == Int32(0):
                        cross = Int32(1)
                if cross != Int32(0):
                    s_res[RES_B] = gb
                    s_res[RES_ABOVE] = after
                    s_res[RES_M] = cq
                if cutlass.const_expr(two):
                    cross2 = Int32(0)
                    if after < target2:
                        if after + cq >= target2:
                            cross2 = Int32(1)
                        if gb == Int32(0):
                            cross2 = Int32(1)
                    if cross2 != Int32(0):
                        s_res[RES_B2] = gb
                if cutlass.const_expr(three):
                    cross3 = Int32(0)
                    if after < target3:
                        if after + cq >= target3:
                            cross3 = Int32(1)
                        if gb == Int32(0):
                            cross3 = Int32(1)
                    if cross3 != Int32(0):
                        s_res[RES_B3] = gb
                after = after + cq
            if cutlass.const_expr(addf):
                av = cute.make_rmem_tensor((4,), Int32)
                aoff = (lane * Int32(NV) + Int32(q)) * Int32(16)
                lds128_i32(atom, s_addv.iterator.toint(), aoff, av)
                for j in cutlass.range_constexpr(4):
                    o4[j] = o4[j] + av[j]
            boff = (lane * Int32(NV) + Int32(q)) * Int32(16)
            sts128_i32(atom, o4, hbase, boff)


@cute.jit
def hi_bit_or_zero(msk):
    r = Int32(0)
    if msk != Int32(0):
        r = Int32(31) - Int32(cute.arch.clz(msk))
    return r


@cute.jit
def _shfl_down_add(val, lane, offset: cutlass.Constexpr):
    other = cute.arch.shuffle_sync_down(val, offset, mask_and_clamp=31)
    if lane + Int32(offset) < Int32(32):
        val = val + other
    return val


@cute.jit
def warp_suffix_scan_add(val, lane):
    for o in [1, 2, 4, 8, 16]:
        val = _shfl_down_add(val, lane, o)
    return val


@cute.jit
def find_cross(s_hist, target, tidx, s_res, nb: cutlass.Constexpr):
    BPL = nb // 32
    if tidx < Int32(32):
        lane = tidx
        part = Int32(0)
        for j in cutlass.range_constexpr(BPL):
            idx = lane * Int32(BPL) + (Int32(j) + lane & Int32(BPL - 1))
            part = part + s_hist[idx]
        v = warp_suffix_scan_add(part, lane)
        if lane == Int32(0):
            s_res[RES_TOT] = v
        msk = Int32(cute.arch.vote_ballot_sync(v >= target))
        Lb = hi_bit_or_zero(msk)
        aboveL = cute.arch.shuffle_sync(v - part, Lb)
        h = Int32(0)
        if lane < Int32(BPL):
            h = s_hist[Lb * Int32(BPL) + lane]
        w = warp_suffix_scan_add(h, lane)
        msk2 = Int32(cute.arch.vote_ballot_sync(aboveL + w >= target))
        J = hi_bit_or_zero(msk2)
        if lane == J:
            s_res[RES_B] = Lb * Int32(BPL) + J
            s_res[RES_M] = h
            s_res[RES_ABOVE] = aboveL + (w - h)


def pack_entry(xbits_u32, idx_i32):
    return Uint64(Uint32(idx_i32)) << Uint64(32) | Uint64(xbits_u32)


def entry_bits(e):
    return Uint32(e & Uint64(4294967295))


def entry_idx(e):
    return Int32(Uint32(e >> Uint64(32)))


def entry_key(e):
    kb = fkey_bits(entry_bits(e))
    return Uint64(kb) << Uint64(32) | Uint64(Uint32(4294967295) - Uint32(e >> Uint64(32)))


def key_to_entry(k):
    idx = Uint32(4294967295) - Uint32(k & Uint64(4294967295))
    bits = invkey_bits(Uint32(k >> Uint64(32)))
    return Uint64(idx) << Uint64(32) | Uint64(bits)


@cute.jit
def ld_entry(lst, i, src_gmem: cutlass.Constexpr):
    r = Uint64(0)
    if cutlass.const_expr(src_gmem):
        r = ld_cg_u64(lst.iterator.toint() + Int64(i) * Int64(8))
    else:
        r = Uint64(lst[i])
    return r


@cute.jit
def gvr_ladder(
    n, k, aim_base, sfac, amin, sd_en, r_const: cutlass.Constexpr, scap: cutlass.Constexpr
):
    SMP = Int32(0)
    SS2 = Int32(1)
    TGT = Int32(0)
    TGT2 = Int32(0)
    if n <= k:
        TGT = Int32(1073741823)
        TGT2 = Int32(1073741823)
    else:
        x6 = Int32(6) * n
        ri = Int32(cmath.sqrt(Float32(x6)))
        if ri * ri > x6:
            ri = ri - Int32(1)
        if (ri + Int32(1)) * (ri + Int32(1)) <= x6:
            ri = ri + Int32(1)
        r6 = ri
        if x6 - ri * ri > ri:
            r6 = ri + Int32(1)
        aim = aim_base
        if r6 > aim:
            aim = r6
        if cutlass.const_expr(r_const > 1):
            if aim < amin:
                aim = amin
        scap_c = Int32(scap)
        if aim > scap_c >> Int32(1):
            aim = scap_c >> Int32(1)
        if aim < k:
            aim = k
        n4w = n >> Int32(2)
        gate = Int32(0)
        if n > scap_c:
            gate = Int32(1)
        if sd_en != Int32(0):
            if n <= scap_c:
                if n > k << Int32(1):
                    gate = Int32(1)
        if n4w < Int32(4):
            gate = Int32(0)
        if gate != Int32(0):
            sel = Int32(Float32(sfac * n) * cute.arch.rcp_approx(Float32(aim)))
            if sel < Int32(256):
                sel = Int32(256)
            nh = n >> Int32(1)
            if sel > nh:
                sel = nh
            pairs = sel >> Int32(3)
            if pairs < Int32(1):
                pairs = Int32(1)
            half = n4w >> Int32(1)
            if half < Int32(1):
                half = Int32(1)
            if pairs > half:
                pairs = half
            SS2 = Int32(Float32(half) * cute.arch.rcp_approx(Float32(pairs)))
            if SS2 < Int32(1):
                SS2 = Int32(1)
            SMP = Int32(Float32(half) * cute.arch.rcp_approx(Float32(SS2)))
            if SMP * SS2 > half:
                SMP = SMP - Int32(1)
            if SMP < Int32(1):
                SMP = Int32(1)
            rn_ = cute.arch.rcp_approx(Float32(n))
            smp8f = Float32(SMP) * Float32(8.0)
            TGT = Int32(Float32(aim) * smp8f * rn_)
            if TGT < Int32(1):
                TGT = Int32(1)
            TGT2 = Int32(Float32(k) * smp8f * rn_)
            if TGT2 < Int32(1):
                TGT2 = Int32(1)
    return (SMP, SS2, TGT, TGT2)


@cute.jit
def _pst(prof, slot, tid):
    if cutlass.const_expr(prof is not None):
        if tid == Int32(0):
            cutlass.inttoptr(prof + Int64(slot) * Int64(8), mem_space=1, dtype=Int64).store(
                Int64(gtimer())
            )


@cute.jit
def gvr_sample_thresholds(
    s_smp,
    SMP,
    TGT,
    TGT2,
    s_hist,
    s_res,
    s_wred,
    tid,
    split: cutlass.Constexpr,
    nthr: cutlass.Constexpr,
    bar_id: cutlass.Constexpr,
    prof=None,
):
    NW = nthr // 32
    lane = tid & Int32(31)
    wid = tid >> Int32(5)
    if tid < Int32(NBS):
        s_hist[tid] = Int32(0)
    if cutlass.const_expr(nthr < NBS):
        for z_ in cutlass.range_constexpr(1, NBS // nthr):
            s_hist[tid + Int32(z_ * nthr)] = Int32(0)
    if tid == Int32(0):
        s_res[RES_B2] = Int32(-1)
        s_res[RES_B3] = Int32(-1)
    smn = Float32(float("inf"))
    smx = Float32(float("-inf"))
    j = tid
    while j < SMP:
        for t in cutlass.range_constexpr(8):
            v = Float32(s_smp[j * Int32(8) + Int32(t)])
            smn = cute.arch.fmin(smn, v)
            smx = cute.arch.fmax(smx, v)
        j = j + Int32(nthr)
    a0 = warp_min_u32(fkey(smn))
    c0m = warp_max_u32(fkey(smx))
    if lane == Int32(0):
        s_wred[wid] = a0
        s_wred[Int32(NW) + wid] = c0m
    bar(bar_id, nthr)
    _pst(prof, 21, tid)
    av = Uint32(4294967295)
    cv = Uint32(0)
    if lane < Int32(NW):
        av = Uint32(s_wred[lane])
        cv = Uint32(s_wred[Int32(NW) + lane])
    SMIN = invkey(warp_min_u32(av))
    SMAX = invkey(warp_max_u32(cv))
    GMIN = Float32(SENT_LO)
    T = Float32(NEG_INF)
    HIC = Float32(NEG_INF)
    T3 = Float32(NEG_INF)
    Tk = Float32(NEG_INF)
    w = Float32(0.0)
    sok = Int32(0)
    if SMP > Int32(0):
        if SMAX > SMIN:
            sok = Int32(1)
    if sok != Int32(0):
        w = (SMAX - SMIN) * Float32(1.0 / 256.0)
        sc_s = cute.arch.rcp_approx(w)
        j2 = tid
        while j2 < SMP:
            for t in cutlass.range_constexpr(8):
                v2 = Float32(s_smp[j2 * Int32(8) + Int32(t)])
                bq = f2s_rz((v2 - SMIN) * sc_s)
                if bq > Int32(NBS - 1):
                    bq = Int32(NBS - 1)
                cute.arch.red(
                    s_hist.iterator + bq,
                    Int32(1),
                    op="add",
                    dtype=Int32,
                    sem="relaxed",
                    scope="cta",
                )
            j2 = j2 + Int32(nthr)
    bar(bar_id, nthr)
    _pst(prof, 22, tid)
    scan_cross0(
        s_hist,
        TGT,
        tid,
        s_res,
        TGT2,
        TGT * Int32(2),
        s_hist,
        nb=NBS,
        zero=True,
        two=True,
        three=True,
    )
    bar(bar_id, nthr)
    _pst(prof, 23, tid)
    tot0 = s_res[RES_TOT]
    b1v = s_res[RES_B]
    b2v = s_res[RES_B2]
    b3v = s_res[RES_B3]
    if sok != Int32(0):
        if tot0 >= TGT:
            T = _fmaf(Float32(b1v), w, SMIN)
    Trung = T
    tinf = Int32(1)
    if T > Float32(NEG_INF):
        tinf = Int32(0)
    if tinf != Int32(0):
        T = GMIN
    if sok != Int32(0):
        if tot0 >= TGT:
            if b2v >= Int32(0):
                Tk = _fmaf(Float32(b2v), w, SMIN)
                anch = T
                if cutlass.const_expr(not split):
                    anch = cute.arch.fmin(T, Trung)
                d_ = cute.arch.fmax(Tk - anch, Float32(0.0))
                HIC = cute.arch.fmax(_fmaf(Float32(4.0), d_, T), _fmaf(Float32(8.0), w, T))
    if sok != Int32(0):
        if tot0 >= TGT * Int32(2):
            if b3v >= Int32(0):
                if T > GMIN:
                    t3c = _fmaf(Float32(b3v), w, SMIN)
                    if t3c < T:
                        T3 = t3c
    bar(bar_id, nthr)
    return (T, HIC, T3, Tk, SMIN, SMAX, w, sok, tot0, b1v, b2v, b3v)


@cute.jit
def tier_stage_warp(
    xs, pos0, nlim, T, T3, s_st1, s_st2, s_cnt, cap1, cap2, lane, tier2: cutlass.Constexpr
):
    lm = Int32(cute.arch.lanemask_lt())
    t2on = Int32(0)
    if cutlass.const_expr(tier2):
        if T3 > Float32(NEG_INF):
            t2on = Int32(1)
    off1 = Int32(0)
    off2 = Int32(0)
    m1s = [Int32(0)] * 4
    m2s = [Int32(0)] * 4
    vmax = Float32(NEG_INF)
    for q in cutlass.range_constexpr(4):
        p = pos0 + Int32(4) * lane + Int32(q)
        x = xs[q]
        ok = p < nlim
        a1 = Int32(0)
        a2 = Int32(0)
        if ok:
            vmax = cute.arch.fmax(vmax, x)
            if x >= T:
                a1 = Int32(1)
            elif t2on != Int32(0):
                if x >= T3:
                    a2 = Int32(1)
        m1s[q] = Int32(cute.arch.vote_ballot_sync(a1 != Int32(0)))
        m2s[q] = Int32(cute.arch.vote_ballot_sync(a2 != Int32(0)))
    tot1 = Int32(0)
    tot2 = Int32(0)
    for q in cutlass.range_constexpr(4):
        tot1 = tot1 + Int32(cute.arch.popc(m1s[q]))
        tot2 = tot2 + Int32(cute.arch.popc(m2s[q]))
    b1 = Int32(0)
    b2 = Int32(0)
    if lane == Int32(0):
        if tot1 > Int32(0):
            b1 = Int32(
                cute.arch.atomic_add(s_cnt.iterator + Int32(0), tot1, sem="relaxed", scope="cta")
            )
        if tot2 > Int32(0):
            b2 = Int32(
                cute.arch.atomic_add(s_cnt.iterator + Int32(1), tot2, sem="relaxed", scope="cta")
            )
    b1 = cute.arch.shuffle_sync(b1, Int32(0))
    b2 = cute.arch.shuffle_sync(b2, Int32(0))
    for q in cutlass.range_constexpr(4):
        p = pos0 + Int32(4) * lane + Int32(q)
        if m1s[q] >> lane & Int32(1) != Int32(0):
            slot = b1 + off1 + Int32(cute.arch.popc(m1s[q] & lm))
            if slot < cap1:
                s_st1[slot] = pack_entry(u32_of_f32(xs[q]), p)
        if m2s[q] >> lane & Int32(1) != Int32(0):
            slot2 = b2 + off2 + Int32(cute.arch.popc(m2s[q] & lm))
            if slot2 < cap2:
                s_st2[slot2] = pack_entry(u32_of_f32(xs[q]), p)
        off1 = off1 + Int32(cute.arch.popc(m1s[q]))
        off2 = off2 + Int32(cute.arch.popc(m2s[q]))
    return vmax


@cute.jit
def grain_max_8lanes(vmax_lane):
    v = vmax_lane
    for o in [1, 2, 4]:
        v = cute.arch.fmax(v, cute.arch.shuffle_sync_bfly(v, o))
    return v


@cute.jit
def floor_admit_warp(xs, pos0, nlim, t_run, s_buf, s_cnt, cap, lane):
    lm = Int32(cute.arch.lanemask_lt())
    ms = [Int32(0)] * 4
    for q in cutlass.range_constexpr(4):
        p = pos0 + Int32(4) * lane + Int32(q)
        a = Int32(0)
        if p < nlim:
            if xs[q] >= t_run:
                a = Int32(1)
        ms[q] = Int32(cute.arch.vote_ballot_sync(a != Int32(0)))
    tot = Int32(0)
    for q in cutlass.range_constexpr(4):
        tot = tot + Int32(cute.arch.popc(ms[q]))
    b = Int32(0)
    if lane == Int32(0):
        if tot > Int32(0):
            b = Int32(
                cute.arch.atomic_add(s_cnt.iterator + Int32(0), tot, sem="relaxed", scope="cta")
            )
    b = cute.arch.shuffle_sync(b, Int32(0))
    off = Int32(0)
    for q in cutlass.range_constexpr(4):
        p = pos0 + Int32(4) * lane + Int32(q)
        if ms[q] >> lane & Int32(1) != Int32(0):
            slot = b + off + Int32(cute.arch.popc(ms[q] & lm))
            if slot < cap:
                s_buf[slot] = pack_entry(u32_of_f32(xs[q]), p)
        off = off + Int32(cute.arch.popc(ms[q]))
    return tot


@cute.jit
def gc_arrive(addr):
    fence_acq_rel_gpu()
    return atom_add_u64(addr, Uint64(1))


@cute.jit
def gc_pad(addr, P):
    red_add_u64(addr, Uint64(GEN_ONE) - Uint64(P))


@cute.jit
def gc_wait(addr, target, skew_ns):
    v = ld_relaxed_u64(addr)
    while v < target:
        nanosleep(Int32(64))
        v = ld_relaxed_u64(addr)
    fence_acq_rel_gpu()
    return v


@cute.jit
def verdict_publish(addr, gen, status):
    st_release_u64(addr, gen + Uint64(1) << Uint64(8) | Uint64(status))


@cute.jit
def verdict_wait(addr, gen):
    want = gen + Uint64(1)
    v = ld_acquire_u64(addr)
    while v >> Uint64(8) != want:
        nanosleep(Int32(128))
        v = ld_acquire_u64(addr)
    return Int32(Uint32(v & Uint64(255)))


@cute.jit
def skew_sleep(seed, salt, skew_ns):
    if skew_ns > Int32(0):
        h = Uint32(seed) * Uint32(2654435761) + Uint32(salt) * Uint32(2246822519)
        h = h ^ h >> Uint32(15)
        h = h * Uint32(739982445)
        h = h ^ h >> Uint32(13)
        ns = Int32(h % Uint32(skew_ns))
        nanosleep(ns)


@cute.jit
def _vbin(e, lo, sc, nb: cutlass.Constexpr):
    x = f32_of_u32(entry_bits(e))
    b = f2u_rz((x - lo) * sc)
    if b > Uint32(nb - 1):
        b = Uint32(nb - 1)
    return Int32(b)


@cute.jit
def _block_min_max_u32(
    a_min, a_max, s_red, tid, nthr: cutlass.Constexpr, bar_id: cutlass.Constexpr
):
    NW = nthr // 32
    lane = tid & Int32(31)
    wid = tid >> Int32(5)
    wmn = warp_min_u32(a_min)
    wmx = warp_max_u32(a_max)
    if lane == Int32(0):
        s_red[wid] = wmn
        s_red[Int32(NW) + wid] = wmx
    bar(bar_id, nthr)
    av = Uint32(4294967295)
    cv = Uint32(0)
    if lane < Int32(NW):
        av = Uint32(s_red[lane])
        cv = Uint32(s_red[Int32(NW) + lane])
    rmn = warp_min_u32(av)
    rmx = warp_max_u32(cv)
    bar(bar_id, nthr)
    return (rmn, rmx)


@cute.jit
def _warp_prefix(cnt_w, s_wc, tid, nthr: cutlass.Constexpr, bar_id: cutlass.Constexpr):
    NW = nthr // 32
    lane = tid & Int32(31)
    wid = tid >> Int32(5)
    if lane == Int32(0):
        s_wc[wid] = cnt_w
    bar(bar_id, nthr)
    ex = Int32(0)
    tot = Int32(0)
    for w_ in cutlass.range_constexpr(NW):
        c = Int32(s_wc[w_])
        if Int32(w_) < wid:
            ex = ex + c
        tot = tot + c
    bar(bar_id, nthr)
    return (ex, tot)


@cute.jit
def select_topk_list(
    lst,
    C,
    K,
    out_e,
    s_hist,
    s_ck,
    s_res,
    s_red,
    s_wc,
    tid,
    src_gmem: cutlass.Constexpr,
    nthr: cutlass.Constexpr,
    bar_id: cutlass.Constexpr,
    want_kth: cutlass.Constexpr = False,
    prof=None,
    rng_lo=None,
    rng_hi=None,
    hist_ready: cutlass.Constexpr = False,
):
    NW = nthr // 32
    lane = tid & Int32(31)
    wid = tid >> Int32(5)
    lm = Int32(cute.arch.lanemask_lt())
    nout = C
    if nout > K:
        nout = K
    kth = Uint64(18446744073709551615)
    mret = Int32(0)
    info = Int32(0)
    if C <= K:
        i = tid
        while i < C:
            out_e[i] = ld_entry(lst, i, src_gmem)
            i = i + Int32(nthr)
        bar(bar_id, nthr)
    else:
        if cutlass.const_expr(prof is not None):
            if tid == Int32(0):
                prof[0] = clk64()
        lo = Float32(0.0)
        hi = Float32(0.0)
        if cutlass.const_expr(rng_lo is None):
            for z_ in cutlass.range_constexpr(NB_SEL // nthr if NB_SEL >= nthr else 1):
                if tid + Int32(z_ * nthr) < Int32(NB_SEL):
                    s_hist[tid + Int32(z_ * nthr)] = Int32(0)
            kmn = Uint32(4294967295)
            kmx = Uint32(0)
            i4 = (tid >> Int32(5)) * Int32(128) + lane
            while i4 < C:
                for q in cutlass.range_constexpr(4):
                    iq = i4 + Int32(32 * q)
                    icl = iq
                    if icl > C - Int32(1):
                        icl = C - Int32(1)
                    kbq = fkey_bits(entry_bits(ld_entry(lst, icl, src_gmem)))
                    if kbq < kmn:
                        kmn = kbq
                    if kbq > kmx:
                        kmx = kbq
                i4 = i4 + Int32(nthr * 4)
            gmn, gmx = _block_min_max_u32(kmn, kmx, s_red, tid, nthr, bar_id)
            lo = invkey(gmn)
            hi = invkey(gmx)
        else:
            lo = rng_lo
            hi = rng_hi
        if cutlass.const_expr(prof is not None):
            if tid == Int32(0):
                prof[1] = clk64()
        wv = (hi - lo) * Float32(1.0 / NB_SEL)
        sc = Float32(0.0)
        if wv > Float32(0.0):
            if wv < Float32(3e38):
                sc = cute.arch.rcp_approx(wv)
        if cutlass.const_expr(not hist_ready):
            if cutlass.const_expr(rng_lo is not None):
                for z_ in cutlass.range_constexpr(NB_SEL // nthr if NB_SEL >= nthr else 1):
                    if tid + Int32(z_ * nthr) < Int32(NB_SEL):
                        s_hist[tid + Int32(z_ * nthr)] = Int32(0)
                bar(bar_id, nthr)
            i4 = (tid >> Int32(5)) * Int32(128) + lane
            while i4 < C:
                bq_ = [Int32(0)] * 4
                for q in cutlass.range_constexpr(4):
                    icl = i4 + Int32(32 * q)
                    if icl > C - Int32(1):
                        icl = C - Int32(1)
                    bq_[q] = _vbin(ld_entry(lst, icl, src_gmem), lo, sc, NB_SEL)
                for q in cutlass.range_constexpr(4):
                    if i4 + Int32(32 * q) < C:
                        cute.arch.red(
                            s_hist.iterator + bq_[q],
                            Int32(1),
                            op="add",
                            dtype=Int32,
                            sem="relaxed",
                            scope="cta",
                        )
                i4 = i4 + Int32(nthr * 4)
        bar(bar_id, nthr)
        if cutlass.const_expr(prof is not None):
            if tid == Int32(0):
                prof[2] = clk64()
        find_cross(s_hist, K, tid, s_res, NB_SEL)
        bar(bar_id, nthr)
        if cutlass.const_expr(prof is not None):
            if tid == Int32(0):
                prof[3] = clk64()
        B = s_res[RES_B]
        above = s_res[RES_ABOVE]
        m = s_res[RES_M]
        need = K - above
        mret = m
        whole = Int32(0)
        if need >= m:
            whole = Int32(1)
        keepc = Int32(0)
        if whole == Int32(0):
            if m <= Int32(QUADC):
                keepc = Int32(1)
        Bw = B
        if whole != Int32(0):
            Bw = B - Int32(1)
        for z_ in cutlass.range_constexpr(NB_SEL // nthr if NB_SEL >= nthr else 1):
            if tid + Int32(z_ * nthr) < Int32(NB_SEL):
                s_hist[tid + Int32(z_ * nthr)] = Int32(0)
        chunk = (C + Int32(nthr * 4) - Int32(1)) // Int32(nthr * 4) * Int32(128)
        c0 = wid * chunk
        c1 = c0 + chunk
        if c1 > C:
            c1 = C
        cw = Int32(0)
        cc = Int32(0)
        i4 = c0 + lane
        while i4 < c1:
            for q in cutlass.range_constexpr(4):
                icl = i4 + Int32(32 * q)
                if icl > C - Int32(1):
                    icl = C - Int32(1)
                bb = _vbin(ld_entry(lst, icl, src_gmem), lo, sc, NB_SEL)
                if i4 + Int32(32 * q) < c1:
                    if bb > Bw:
                        cw = cw + Int32(1)
                    elif bb == B:
                        cc = cc + Int32(1)
            i4 = i4 + Int32(128)
        wcw = Int32(cute.arch.warp_redux_sync(cw, "add"))
        wcc = Int32(cute.arch.warp_redux_sync(cc, "add"))
        if lane == Int32(0):
            s_wc[wid] = wcw
            s_wc[Int32(NW) + wid] = wcc
        bar(bar_id, nthr)
        pw = Int32(0)
        pc = Int32(0)
        for w_ in cutlass.range_constexpr(NW):
            if Int32(w_) < wid:
                pw = pw + Int32(s_wc[w_])
                pc = pc + Int32(s_wc[NW + w_])
        if cutlass.const_expr(prof is not None):
            if tid == Int32(0):
                prof[4] = clk64()
        base0 = c0
        while base0 < c1:
            ed = []
            for q in cutlass.range_constexpr(4):
                icl = base0 + Int32(32 * q) + lane
                if icl > C - Int32(1):
                    icl = C - Int32(1)
                ed.append(ld_entry(lst, icl, src_gmem))
            for q in cutlass.range_constexpr(4):
                fw = Int32(0)
                fc = Int32(0)
                if base0 + Int32(32 * q) + lane < c1:
                    bd_ = _vbin(ed[q], lo, sc, NB_SEL)
                    if bd_ > Bw:
                        fw = Int32(1)
                    elif bd_ == B:
                        fc = Int32(1)
                mw = Int32(cute.arch.vote_ballot_sync(fw != Int32(0)))
                mc = Int32(cute.arch.vote_ballot_sync(fc != Int32(0)))
                if fw != Int32(0):
                    out_e[pw + Int32(cute.arch.popc(mw & lm))] = ed[q]
                if fc != Int32(0):
                    if keepc != Int32(0):
                        s_ck[pc + Int32(cute.arch.popc(mc & lm))] = entry_key(ed[q])
                pw = pw + Int32(cute.arch.popc(mw))
                pc = pc + Int32(cute.arch.popc(mc))
            base0 = base0 + Int32(128)
        bar(bar_id, nthr)
        if cutlass.const_expr(prof is not None):
            if tid == Int32(0):
                prof[5] = clk64()
        if whole == Int32(0):
            if m <= Int32(32):
                info = Int32(1)
                if wid == Int32(0):
                    kq = Uint64(0)
                    if lane < m:
                        kq = Uint64(s_ck[lane])
                    khi = Uint32(kq >> Uint64(32))
                    klo = Uint32(kq & Uint64(4294967295))
                    r_ = Int32(0)
                    for j in cutlass.range_constexpr(32):
                        ohi = cute.arch.shuffle_sync(khi, Int32(j))
                        olo = cute.arch.shuffle_sync(klo, Int32(j))
                        if Int32(j) < m:
                            if ohi > khi:
                                r_ = r_ + Int32(1)
                            elif ohi == khi:
                                if olo > klo:
                                    r_ = r_ + Int32(1)
                    if lane < m:
                        if r_ < need:
                            out_e[above + r_] = key_to_entry(kq)
                bar(bar_id, nthr)
            elif m <= Int32(QUADC):
                info = Int32(1)
                i = tid
                while i < m:
                    ki = Uint64(s_ck[i])
                    r_ = Int32(0)
                    jq = Int32(0)
                    while jq < m:
                        if Uint64(s_ck[jq]) > ki:
                            r_ = r_ + Int32(1)
                        jq = jq + Int32(1)
                    if r_ < need:
                        out_e[above + r_] = key_to_entry(ki)
                    i = i + Int32(nthr)
                bar(bar_id, nthr)
            else:
                info = Int32(2)
                pref = Uint64(0)
                plen = Int32(0)
                needr = need
                done = Int32(0)
                while done == Int32(0):
                    shift = Int32(56) - plen
                    if tid < Int32(NBS):
                        s_hist[tid] = Int32(0)
                    if cutlass.const_expr(nthr < NBS):
                        for z_ in cutlass.range_constexpr(1, NBS // nthr):
                            s_hist[tid + Int32(z_ * nthr)] = Int32(0)
                    bar(bar_id, nthr)
                    i = tid
                    while i < C:
                        e2_ = ld_entry(lst, i, src_gmem)
                        if _vbin(e2_, lo, sc, NB_SEL) == B:
                            k2 = entry_key(e2_)
                            okp = Int32(1)
                            if plen > Int32(0):
                                if k2 >> Uint64(Int32(64) - plen) != pref:
                                    okp = Int32(0)
                            if okp != Int32(0):
                                dg = Int32(Uint32(k2 >> Uint64(shift) & Uint64(255)))
                                cute.arch.red(
                                    s_hist.iterator + dg,
                                    Int32(1),
                                    op="add",
                                    dtype=Int32,
                                    sem="relaxed",
                                    scope="cta",
                                )
                        i = i + Int32(nthr)
                    bar(bar_id, nthr)
                    scan_cross0(s_hist, needr, tid, s_res, needr, needr, s_hist, nb=NBS, zero=True)
                    bar(bar_id, nthr)
                    dB = s_res[RES_B]
                    dab = s_res[RES_ABOVE]
                    dm = s_res[RES_M]
                    bar(bar_id, nthr)
                    pref = pref << Uint64(8) | Uint64(dB)
                    plen = plen + Int32(8)
                    needr = needr - dab
                    if needr >= dm:
                        done = Int32(1)
                    if plen >= Int32(64):
                        done = Int32(1)
                thr = pref << Uint64(Int32(64) - plen)
                if plen >= Int32(64):
                    thr = pref
                nw2 = Int32(0)
                r0 = c0
                while r0 < c1:
                    ii = r0 + lane
                    f2_ = Int32(0)
                    if ii < c1:
                        e3_ = ld_entry(lst, ii, src_gmem)
                        if _vbin(e3_, lo, sc, NB_SEL) == B:
                            if entry_key(e3_) >= thr:
                                f2_ = Int32(1)
                    nw2 = nw2 + Int32(cute.arch.popc(cute.arch.vote_ballot_sync(f2_ != Int32(0))))
                    r0 = r0 + Int32(32)
                ex2, tot2 = _warp_prefix(nw2, s_wc, tid, nthr, bar_id)
                p2 = above + ex2
                r0 = c0
                while r0 < c1:
                    ii = r0 + lane
                    f2_ = Int32(0)
                    e4 = Uint64(0)
                    if ii < c1:
                        e4 = ld_entry(lst, ii, src_gmem)
                        if _vbin(e4, lo, sc, NB_SEL) == B:
                            if entry_key(e4) >= thr:
                                f2_ = Int32(1)
                    m2 = Int32(cute.arch.vote_ballot_sync(f2_ != Int32(0)))
                    if f2_ != Int32(0):
                        out_e[p2 + Int32(cute.arch.popc(m2 & lm))] = e4
                    p2 = p2 + Int32(cute.arch.popc(m2))
                    r0 = r0 + Int32(32)
                bar(bar_id, nthr)
        if cutlass.const_expr(prof is not None):
            if tid == Int32(0):
                prof[6] = clk64()
    if cutlass.const_expr(want_kth):
        kmn64 = Uint64(18446744073709551615)
        i = tid
        while i < nout:
            kk = entry_key(Uint64(out_e[i]))
            if kk < kmn64:
                kmn64 = kk
            i = i + Int32(nthr)
        hi32 = Uint32(kmn64 >> Uint64(32))
        lo32 = Uint32(kmn64 & Uint64(4294967295))
        wmh = warp_min_u32(hi32)
        cand = Uint32(4294967295)
        if hi32 == wmh:
            cand = lo32
        wml = warp_min_u32(cand)
        if lane == Int32(0):
            s_red[wid] = wmh
            s_red[Int32(NW) + wid] = wml
        bar(bar_id, nthr)
        bh = Uint32(4294967295)
        bl = Uint32(4294967295)
        for w_ in cutlass.range_constexpr(NW):
            h_ = Uint32(s_red[w_])
            l_ = Uint32(s_red[Int32(NW) + Int32(w_)])
            if h_ < bh:
                bh = h_
                bl = l_
            elif h_ == bh:
                if l_ < bl:
                    bl = l_
        kth = Uint64(bh) << Uint64(32) | Uint64(bl)
        bar(bar_id, nthr)
    if cutlass.const_expr(prof is not None):
        if tid == Int32(0):
            prof[7] = clk64()
    return (nout, kth, mret, info)


@cute.jit
def l2g_swa(i, pos, req, bt_swa, max_blocks_swa, swa_off_tokens):
    st = pos - Int32(SWA - 1)
    if st < Int32(0):
        st = Int32(0)
    loc = st + i
    blk = loc // Int32(TPB_SWA)
    bc = blk
    if bc > max_blocks_swa - Int32(1):
        bc = max_blocks_swa - Int32(1)
    page = Int32(bt_swa[req, bc])
    g = Int32(-1)
    if loc <= pos:
        if blk < max_blocks_swa:
            g = swa_off_tokens + page * Int32(TPB_SWA) + (loc - blk * Int32(TPB_SWA))
    return g


@cute.jit
def l2g_swa_local(loc, req, bt_swa, max_blocks_swa, swa_off_tokens):
    blk = loc // Int32(TPB_SWA)
    bc = blk
    if bc > max_blocks_swa - Int32(1):
        bc = max_blocks_swa - Int32(1)
    if bc < Int32(0):
        bc = Int32(0)
    page = Int32(bt_swa[req, bc])
    g = Int32(-1)
    if loc >= Int32(0):
        if blk < max_blocks_swa:
            g = swa_off_tokens + page * Int32(TPB_SWA) + (loc - blk * Int32(TPB_SWA))
    return g


@cute.jit
def l2g_compressed(c, req, bt_c, max_blocks_c, c_off_tokens):
    blk = c // Int32(TPB_C)
    bc = blk
    if bc > max_blocks_c - Int32(1):
        bc = max_blocks_c - Int32(1)
    if bc < Int32(0):
        bc = Int32(0)
    page = Int32(bt_c[req, bc])
    g = Int32(-1)
    if c >= Int32(0):
        if blk < max_blocks_c:
            g = c_off_tokens + page * Int32(TPB_C) + (c - blk * Int32(TPB_C))
    return g


@cute.jit
def block_excl_scan(v, s_wc, tid, nthr: cutlass.Constexpr, bar_id: cutlass.Constexpr):
    NW = nthr // 32
    lane = tid & Int32(31)
    wid = tid >> Int32(5)
    inc = warp_incl_scan_add(v, lane)
    if lane == Int32(31):
        s_wc[wid] = inc
    bar(bar_id, nthr)
    base = Int32(0)
    tot = Int32(0)
    for w_ in cutlass.range_constexpr(NW):
        c = Int32(s_wc[w_])
        if Int32(w_) < wid:
            base = base + c
        tot = tot + c
    bar(bar_id, nthr)
    return (base + inc - v, tot)


@cute.jit
def sample_window_range(t0, t1, SMP, SS2):
    st = Int32(8) * SS2
    j0 = (t0 + st - Int32(1)) // st
    j1 = (t1 + st - Int32(1)) // st
    if j1 > SMP:
        j1 = SMP
    if j0 > j1:
        j0 = j1
    return (j0, j1)


@cute.jit
def flatten_regions(
    dst, dst_gmem: cutlass.Constexpr, reg_base, cap, s_pc, s_po, P, tid, nthr: cutlass.Constexpr
):
    NW = nthr // 32
    lane = tid & Int32(31)
    wid = tid >> Int32(5)
    p = wid
    while p < P:
        cnt = Int32(s_pc[p])
        off = Int32(s_po[p])
        src = reg_base + Int64(p) * Int64(cap) * Int64(8)
        i = lane
        while i < cnt:
            e = ld_cg_u64(src + Int64(i) * Int64(8))
            if cutlass.const_expr(dst_gmem):
                st_g_u64(dst.iterator.toint() + Int64(off + i) * Int64(8), e)
            else:
                dst[off + i] = e
            i = i + Int32(32)
        p = p + Int32(NW)


@cute.jit
def parts_table_warp0(desc_base, P, cap1, cap2, s_pt, s_tot, tid):
    if tid < Int32(32):
        lane = tid
        ppl = P + Int32(31) >> Int32(5)
        p0 = lane * ppl
        a1 = [Int32(0)] * 8
        a2 = [Int32(0)] * 8
        for q in cutlass.range_constexpr(8):
            pq = p0 + Int32(q)
            pc = pq
            if pc > P - Int32(1):
                pc = P - Int32(1)
            v1 = ld_cg_i32(desc_base + Int64(pc) * Int64(16))
            v2 = ld_cg_i32(desc_base + Int64(pc) * Int64(16) + Int64(4))
            ok = Int32(0)
            if Int32(q) < ppl:
                if pq < P:
                    ok = Int32(1)
            a1[q] = v1 * ok
            a2[q] = v2 * ok
        s1 = Int32(0)
        s2 = Int32(0)
        n1 = Int32(0)
        n2 = Int32(0)
        o1 = Int32(0)
        o2 = Int32(0)
        for q in cutlass.range_constexpr(8):
            c1_ = a1[q]
            if c1_ > cap1:
                c1_ = cap1
                o1 = o1 + Int32(1)
            c2_ = a2[q]
            if c2_ > cap2:
                c2_ = cap2
                o2 = o2 + Int32(1)
            s1 = s1 + c1_
            s2 = s2 + c2_
            n1 = n1 + a1[q]
            n2 = n2 + a2[q]
        i1 = warp_incl_scan_add(s1, lane)
        i2 = warp_incl_scan_add(s2, lane)
        t1s = cute.arch.shuffle_sync(i1, Int32(31))
        t2s = cute.arch.shuffle_sync(i2, Int32(31))
        b1 = i1 - s1
        b2 = i2 - s2 + t1s
        for q in cutlass.range_constexpr(8):
            pq = p0 + Int32(q)
            c1_ = a1[q]
            if c1_ > cap1:
                c1_ = cap1
            c2_ = a2[q]
            if c2_ > cap2:
                c2_ = cap2
            if Int32(q) < ppl:
                if pq < P:
                    s_pt[pq] = b1
                    s_pt[P + pq] = b2
            b1 = b1 + c1_
            b2 = b2 + c2_
        tn1 = Int32(cute.arch.warp_redux_sync(n1, "add"))
        tn2 = Int32(cute.arch.warp_redux_sync(n2, "add"))
        to1 = Int32(cute.arch.warp_redux_sync(o1, "add"))
        to2 = Int32(cute.arch.warp_redux_sync(o2, "add"))
        if lane == Int32(0):
            s_pt[Int32(2) * P] = t1s + t2s
            s_tot[0] = t1s
            s_tot[1] = t2s
            s_tot[2] = tn1
            s_tot[3] = tn2
            s_tot[4] = to1
            s_tot[5] = to2


@cute.jit
def build_owner_map(
    s_own, s_pt, np_, C, s_wc, tid, nthr: cutlass.Constexpr, bar_id: cutlass.Constexpr
):
    NW = nthr // 32
    lane = tid & Int32(31)
    wid = tid >> Int32(5)
    i = tid
    while i < C:
        s_own[i] = Int32(0)
        i = i + Int32(nthr)
    bar(bar_id, nthr)
    q = tid
    while q < np_:
        a = Int32(s_pt[q])
        if Int32(s_pt[q + Int32(1)]) > a:
            s_own[a] = q
        q = q + Int32(nthr)
    bar(bar_id, nthr)
    E = (C + Int32(nthr) - Int32(1)) // Int32(nthr)
    b0 = tid * E
    b1 = b0 + E
    if b1 > C:
        b1 = C
    mx = Int32(0)
    i = b0
    while i < b1:
        v = Int32(s_own[i])
        if v > mx:
            mx = v
        i = i + Int32(1)
    inc = mx
    for o in [1, 2, 4, 8, 16]:
        other = cute.arch.shuffle_sync_up(inc, o, mask_and_clamp=0)
        if lane >= Int32(o):
            if other > inc:
                inc = other
    if lane == Int32(31):
        s_wc[wid] = inc
    ex = cute.arch.shuffle_sync_up(inc, 1, mask_and_clamp=0)
    if lane == Int32(0):
        ex = Int32(0)
    bar(bar_id, nthr)
    for w_ in cutlass.range_constexpr(NW):
        if Int32(w_) < wid:
            wv = Int32(s_wc[w_])
            if wv > ex:
                ex = wv
    run = ex
    i = b0
    while i < b1:
        v = Int32(s_own[i])
        if v > run:
            run = v
        s_own[i] = run
        i = i + Int32(1)
    bar(bar_id, nthr)


@cute.jit
def gather_parts(
    dst,
    dst_gmem: cutlass.Constexpr,
    np_,
    C,
    s_pt,
    P,
    reg1_base,
    reg2_base,
    cap1,
    cap2,
    s_hist,
    lo,
    sc,
    with_hist,
    tid,
    nthr: cutlass.Constexpr,
    s_own=None,
):
    UNR = 8
    base = Int32(0)
    while base < C:
        es = []
        fs = []
        for u in cutlass.range_constexpr(UNR):
            f = base + Int32(u * nthr) + tid
            fl = f
            if fl > C - Int32(1):
                fl = C - Int32(1)
            p_ = Int32(0)
            if cutlass.const_expr(s_own is not None):
                p_ = Int32(s_own[fl])
            else:
                lo_ = Int32(0)
                hi_ = np_
                for it in cutlass.range_constexpr(9):
                    mid = lo_ + hi_ >> Int32(1)
                    if Int32(s_pt[mid]) <= fl:
                        lo_ = mid
                    else:
                        hi_ = mid
                p_ = lo_
            off = fl - Int32(s_pt[p_])
            src = reg1_base + (Int64(p_) * Int64(cap1) + Int64(off)) * Int64(8)
            if p_ >= P:
                src = reg2_base + (Int64(p_ - P) * Int64(cap2) + Int64(off)) * Int64(8)
            es.append(ld_cg_u64(src))
            fs.append(f)
        for u in cutlass.range_constexpr(UNR):
            if fs[u] < C:
                if cutlass.const_expr(dst_gmem):
                    st_g_u64(dst.iterator.toint() + Int64(fs[u]) * Int64(8), es[u])
                else:
                    dst[fs[u]] = es[u]
                if with_hist != Int32(0):
                    b = _vbin(es[u], lo, sc, NB_SEL)
                    cute.arch.red(
                        s_hist.iterator + b,
                        Int32(1),
                        op="add",
                        dtype=Int32,
                        sem="relaxed",
                        scope="cta",
                    )
        base = base + Int32(UNR * nthr)


@cute.jit
def finalize_tiers(
    n,
    P,
    T,
    T3,
    HIC,
    desc_base,
    reg1_base,
    reg2_base,
    cap1,
    cap2,
    flat,
    s_list,
    sel_cap: cutlass.Constexpr,
    s_out,
    s_hist,
    s_ck,
    s_res,
    s_red,
    s_wc,
    s_pt,
    s_tot,
    s_own,
    tid,
    nthr: cutlass.Constexpr,
    bar_id: cutlass.Constexpr,
    prof=None,
):
    K = Int32(K_TOP)
    if cutlass.const_expr(prof is not None):
        if tid == Int32(0):
            prof[8] = gtimer()
    parts_table_warp0(desc_base, P, cap1, cap2, s_pt, s_tot, tid)
    for z_ in cutlass.range_constexpr(NB_SEL // nthr if NB_SEL >= nthr else 1):
        if tid + Int32(z_ * nthr) < Int32(NB_SEL):
            s_hist[tid + Int32(z_ * nthr)] = Int32(0)
    bar(bar_id, nthr)
    if cutlass.const_expr(prof is not None):
        if tid == Int32(0):
            prof[9] = gtimer()
    tot1s = Int32(s_tot[0])
    tot2s = Int32(s_tot[1])
    tot1 = Int32(s_tot[2])
    tot2 = Int32(s_tot[3])
    ovf1 = Int32(s_tot[4])
    ovf2 = Int32(s_tot[5])
    use = Int32(0)
    if tot1 >= K:
        if ovf1 == Int32(0):
            use = Int32(1)
    if use == Int32(0):
        if T3 > Float32(NEG_INF):
            if tot1 + tot2 >= K:
                if ovf1 + ovf2 == Int32(0):
                    use = Int32(2)
    C = Int32(0)
    m = Int32(0)
    nout = Int32(0)
    info = Int32(0)
    if use != Int32(0):
        C = tot1s
        np_ = P
        lo = T
        if use == Int32(2):
            C = tot1s + tot2s
            np_ = Int32(2) * P
            lo = T3
        kth = Uint64(0)
        rng = Int32(0)
        if HIC > lo:
            if HIC < Float32(3e38):
                rng = Int32(1)
        wv = (HIC - lo) * Float32(1.0 / NB_SEL)
        sc = Float32(0.0)
        if rng != Int32(0):
            sc = cute.arch.rcp_approx(wv)
        if C <= Int32(sel_cap):
            build_owner_map(s_own, s_pt, np_, C, s_wc, tid, nthr, bar_id)
            gather_parts(
                s_list,
                False,
                np_,
                C,
                s_pt,
                P,
                reg1_base,
                reg2_base,
                cap1,
                cap2,
                s_hist,
                lo,
                sc,
                rng,
                tid,
                nthr,
                s_own,
            )
            bar(bar_id, nthr)
            if cutlass.const_expr(prof is not None):
                if tid == Int32(0):
                    prof[10] = gtimer()
            if rng != Int32(0):
                nout, kth, m, info = select_topk_list(
                    s_list,
                    C,
                    K,
                    s_out,
                    s_hist,
                    s_ck,
                    s_res,
                    s_red,
                    s_wc,
                    tid,
                    False,
                    nthr,
                    bar_id,
                    False,
                    None,
                    lo,
                    HIC,
                    True,
                )
            else:
                nout, kth, m, info = select_topk_list(
                    s_list, C, K, s_out, s_hist, s_ck, s_res, s_red, s_wc, tid, False, nthr, bar_id
                )
        else:
            gather_parts(
                flat,
                True,
                np_,
                C,
                s_pt,
                P,
                reg1_base,
                reg2_base,
                cap1,
                cap2,
                s_hist,
                lo,
                sc,
                Int32(0),
                tid,
                nthr,
            )
            fence_acq_rel_gpu()
            bar(bar_id, nthr)
            nout, kth, m, info = select_topk_list(
                flat, C, K, s_out, s_hist, s_ck, s_res, s_red, s_wc, tid, True, nthr, bar_id
            )
    if cutlass.const_expr(prof is not None):
        if tid == Int32(0):
            prof[11] = gtimer()
    return (use, C, m, tot1, tot2, nout, info)


@cute.jit
def write_row_outputs(
    row,
    n,
    s_out,
    nout,
    short,
    topk_out,
    gidx,
    bt_swa,
    bt_c,
    pos,
    req,
    mbs,
    mbc,
    swa_off,
    c_off,
    tid,
    nthr: cutlass.Constexpr,
):
    NQ = (640 + nthr - 1) // nthr
    gv = []
    cols = []
    for q in cutlass.range_constexpr(NQ):
        col = tid + Int32(q * nthr)
        g = Int32(-1)
        if col < Int32(SWA):
            g = l2g_swa(col, pos, req, bt_swa, mbs, swa_off)
        elif col < Int32(SWA + K_TOP):
            i = col - Int32(SWA)
            idx = Int32(-1)
            if short != Int32(0):
                if i < n:
                    idx = i
            elif i < nout:
                idx = entry_idx(Uint64(s_out[i]))
            topk_out[row, i] = idx
            g = l2g_compressed(idx, req, bt_c, mbc, c_off)
        gv.append(g)
        cols.append(col)
    for q in cutlass.range_constexpr(NQ):
        if cols[q] < Int32(SWA + K_TOP):
            gidx[row, cols[q]] = gv[q]


@cute.jit
def grain_threshold(
    gmax_addr,
    ng,
    s_list,
    s_out,
    s_hist,
    s_ck,
    s_res,
    s_red,
    s_wc,
    tid,
    nthr: cutlass.Constexpr,
    bar_id: cutlass.Constexpr,
):
    i = tid
    while i < ng:
        gm = ld_cg_f32(gmax_addr + Int64(i) * Int64(4))
        s_list[i] = pack_entry(u32_of_f32(gm), i)
        i = i + Int32(nthr)
    bar(bar_id, nthr)
    nout, kth, m, info = select_topk_list(
        s_list,
        ng,
        Int32(K_TOP),
        s_out,
        s_hist,
        s_ck,
        s_res,
        s_red,
        s_wc,
        tid,
        False,
        nthr,
        bar_id,
        True,
    )
    return invkey(Uint32(kth >> Uint64(32)))


@cute.jit
def recompute_grains(
    n,
    t0,
    t1,
    Tstar,
    x_row,
    gmax_addr,
    fbr_addr,
    s_cnt,
    tid,
    nthr: cutlass.Constexpr,
    bar_id: cutlass.Constexpr,
):
    NW = nthr // 32
    lane = tid & Int32(31)
    wid = tid >> Int32(5)
    lm = Int32(cute.arch.lanemask_lt())
    if tid == Int32(0):
        s_cnt[0] = Int32(0)
    bar(bar_id, nthr)
    g = (t0 >> Int32(5)) + wid
    g1 = t1 + Int32(31) >> Int32(5)
    while g < g1:
        gm = ld_cg_f32(gmax_addr + Int64(g) * Int64(4))
        if gm >= Tstar:
            p = g * Int32(32) + lane
            a = Int32(0)
            x = Float32(NEG_INF)
            if p < t1:
                if p < n:
                    x = ld_nc_f32(x_row + Int64(p) * Int64(4))
                    if x >= Tstar:
                        a = Int32(1)
            msk = Int32(cute.arch.vote_ballot_sync(a != Int32(0)))
            tot = Int32(cute.arch.popc(msk))
            b = Int32(0)
            if lane == Int32(0):
                if tot > Int32(0):
                    b = Int32(
                        cute.arch.atomic_add(
                            s_cnt.iterator + Int32(0), tot, sem="relaxed", scope="cta"
                        )
                    )
            b = cute.arch.shuffle_sync(b, Int32(0))
            if a != Int32(0):
                slot = b + Int32(cute.arch.popc(msk & lm))
                st_g_u64(fbr_addr + Int64(slot) * Int64(8), pack_entry(u32_of_f32(x), p))
        g = g + Int32(NW)
    bar(bar_id, nthr)
    c = Int32(s_cnt[0])
    bar(bar_id, nthr)
    return c


@cute.jit
def radix_round_local(
    s_chunk,
    cnt,
    prefix,
    plen,
    shift,
    mask,
    s_rh,
    nbins: cutlass.Constexpr,
    tid,
    nthr: cutlass.Constexpr,
    bar_id: cutlass.Constexpr,
):
    i = tid
    while i < Int32(nbins):
        s_rh[i] = Int32(0)
        i = i + Int32(nthr)
    bar(bar_id, nthr)
    i = tid
    while i < cnt:
        kb = fkey(Float32(s_chunk[i]))
        okp = Int32(1)
        if plen > Int32(0):
            if kb >> Uint32(Int32(32) - plen) != prefix:
                okp = Int32(0)
        if okp != Int32(0):
            d = Int32(kb >> Uint32(shift) & Uint32(mask))
            cute.arch.atomic_add(s_rh.iterator + d, Int32(1), sem="relaxed", scope="cta")
        i = i + Int32(nthr)
    bar(bar_id, nthr)


@cute.jit
def radix_participant(
    row,
    part,
    P,
    n,
    t0,
    t1,
    s_chunk,
    ctr_fb,
    fbh_row,
    fbt_row,
    s_rh,
    s_res,
    s_wc,
    s_scal,
    topk_out,
    gidx,
    bt_c,
    req,
    mbc,
    c_off,
    tid,
    nthr: cutlass.Constexpr,
    bar_id: cutlass.Constexpr,
):
    lane = tid & Int32(31)
    wid = tid >> Int32(5)
    lm = Int32(cute.arch.lanemask_lt())
    cnt = t1 - t0
    if t1 > n:
        cnt = n - t0
    need = Int32(K_TOP)
    prefix = Uint32(0)
    plen = Int32(0)
    gF = Uint64(0)
    for r in cutlass.range_constexpr(3):
        bits = 11 if r < 2 else 10
        shift = 21 if r == 0 else 10 if r == 1 else 0
        nb_ = 1 << bits
        radix_round_local(
            s_chunk, cnt, prefix, plen, Int32(shift), Int32(nb_ - 1), s_rh, 2048, tid, nthr, bar_id
        )
        i = tid
        while i < Int32(nb_):
            c = Int32(s_rh[i])
            if c != Int32(0):
                red_add_u32(fbh_row + Int64(r * 2048 + i) * Int64(4), c)
            i = i + Int32(nthr)
        bar(bar_id, nthr)
        if tid == Int32(0):
            old = gc_arrive(ctr_fb)
            if cutlass.const_expr(r == 0):
                gF = old >> Uint64(GEN_SHIFT)
                s_scal[6] = Int32(Uint32(gF & Uint64(4294967295)))
                s_scal[7] = Int32(Uint32(gF >> Uint64(32)))
            gF2 = Uint64(Uint32(s_scal[7])) << Uint64(32) | Uint64(Uint32(s_scal[6]))
            gc_wait(ctr_fb, (gF2 << Uint64(GEN_SHIFT)) + Uint64(r + 1) * Uint64(P), Int32(0))
        bar(bar_id, nthr)
        i = tid
        while i < Int32(nb_):
            s_rh[i] = ld_cg_i32(fbh_row + Int64(r * 2048 + i) * Int64(4))
            i = i + Int32(nthr)
        if cutlass.const_expr(nb_ < 2048):
            i = tid + Int32(nb_)
            while i < Int32(2048):
                s_rh[i] = Int32(0)
                i = i + Int32(nthr)
        bar(bar_id, nthr)
        scan_cross0(s_rh, need, tid, s_res, need, need, s_rh, nb=2048, zero=True)
        bar(bar_id, nthr)
        dB = s_res[RES_B]
        dab = s_res[RES_ABOVE]
        bar(bar_id, nthr)
        prefix = prefix << Uint32(bits) | Uint32(dB)
        plen = plen + Int32(bits)
        need = need - dab
    kth = prefix
    my_ab = Int32(0)
    my_eq = Int32(0)
    i = tid
    while i < cnt:
        kb = fkey(Float32(s_chunk[i]))
        if kb > kth:
            my_ab = my_ab + Int32(1)
        elif kb == kth:
            my_eq = my_eq + Int32(1)
        i = i + Int32(nthr)
    e_, tab = block_excl_scan(my_ab, s_wc, tid, nthr, bar_id)
    e2_, teq = block_excl_scan(my_eq, s_wc, tid, nthr, bar_id)
    if tid == Int32(0):
        st_g_u64(
            fbt_row + Int64(part) * Int64(8),
            Uint64(Uint32(teq)) << Uint64(32) | Uint64(Uint32(tab)),
        )
        gF3 = Uint64(Uint32(s_scal[7])) << Uint64(32) | Uint64(Uint32(s_scal[6]))
        old = gc_arrive(ctr_fb)
        if Int32(Uint32(old & Uint64(65535))) == Int32(4) * P - Int32(1):
            gc_pad(ctr_fb, Int32(4) * P)
        gc_wait(ctr_fb, (gF3 << Uint64(GEN_SHIFT)) + Uint64(4) * Uint64(P), Int32(0))
    bar(bar_id, nthr)
    ab_t = Int32(0)
    eq_t = Int32(0)
    if tid < P:
        w2 = ld_cg_u64(fbt_row + Int64(tid) * Int64(8))
        ab_t = Int32(Uint32(w2 & Uint64(4294967295)))
        eq_t = Int32(Uint32(w2 >> Uint64(32)))
    tie_ex, tie_tot_ = block_excl_scan(eq_t, s_wc, tid, nthr, bar_id)
    q_t = need - tie_ex
    if q_t < Int32(0):
        q_t = Int32(0)
    if q_t > eq_t:
        q_t = eq_t
    win_ex, win_tot_ = block_excl_scan(ab_t + q_t, s_wc, tid, nthr, bar_id)
    if tid == part:
        s_scal[0] = win_ex
        s_scal[1] = q_t
    bar(bar_id, nthr)
    slot0 = Int32(s_scal[0])
    myq = Int32(s_scal[1])
    NW = nthr // 32
    chunk = (cnt + Int32(nthr) - Int32(1)) // Int32(nthr) * Int32(32)
    c0 = wid * chunk
    c1 = c0 + chunk
    if c1 > cnt:
        c1 = cnt
    nwin = Int32(0)
    ntie = Int32(0)
    r0 = c0
    while r0 < c1:
        ii = r0 + lane
        fa = Int32(0)
        fe = Int32(0)
        if ii < c1:
            kb = fkey(Float32(s_chunk[ii]))
            if kb > kth:
                fa = Int32(1)
            elif kb == kth:
                fe = Int32(1)
        nwin = nwin + Int32(cute.arch.popc(cute.arch.vote_ballot_sync(fa != Int32(0))))
        ntie = ntie + Int32(cute.arch.popc(cute.arch.vote_ballot_sync(fe != Int32(0))))
        r0 = r0 + Int32(32)
    if lane == Int32(0):
        s_wc[wid] = ntie
    bar(bar_id, nthr)
    tex = Int32(0)
    for w_ in cutlass.range_constexpr(NW):
        if Int32(w_) < wid:
            tex = tex + Int32(s_wc[w_])
    bar(bar_id, nthr)
    tq = myq - tex
    if tq < Int32(0):
        tq = Int32(0)
    if tq > ntie:
        tq = ntie
    if lane == Int32(0):
        s_wc[wid] = nwin + tq
    bar(bar_id, nthr)
    wex = Int32(0)
    for w_ in cutlass.range_constexpr(NW):
        if Int32(w_) < wid:
            wex = wex + Int32(s_wc[w_])
    bar(bar_id, nthr)
    pw = slot0 + wex
    tr = tex
    r0 = c0
    while r0 < c1:
        ii = r0 + lane
        fa = Int32(0)
        fe = Int32(0)
        if ii < c1:
            kb = fkey(Float32(s_chunk[ii]))
            if kb > kth:
                fa = Int32(1)
            elif kb == kth:
                fe = Int32(1)
        me = Int32(cute.arch.vote_ballot_sync(fe != Int32(0)))
        trank = tr + Int32(cute.arch.popc(me & lm))
        if fe != Int32(0):
            if trank < myq:
                fa = Int32(1)
        mw = Int32(cute.arch.vote_ballot_sync(fa != Int32(0)))
        if fa != Int32(0):
            slot = pw + Int32(cute.arch.popc(mw & lm))
            idx = t0 + ii
            topk_out[row, slot] = idx
            gidx[row, Int32(SWA) + slot] = l2g_compressed(idx, req, bt_c, mbc, c_off)
        pw = pw + Int32(cute.arch.popc(mw))
        tr = tr + Int32(cute.arch.popc(me))
        r0 = r0 + Int32(32)
    per = (Int32(3 * 2048) + P - Int32(1)) // P
    z0 = part * per
    z1 = z0 + per
    if z1 > Int32(3 * 2048):
        z1 = Int32(3 * 2048)
    i = z0 + tid
    while i < z1:
        st_g_u32(fbh_row + Int64(i) * Int64(4), Int32(0))
        i = i + Int32(nthr)
    bar(bar_id, nthr)
    return (kth, need)
