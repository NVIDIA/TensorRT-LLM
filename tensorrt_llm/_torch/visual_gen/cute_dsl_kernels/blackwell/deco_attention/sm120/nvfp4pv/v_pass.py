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
"""V pass of the sm_120 NVFP4-PV kernel: V' = V minus its BF16 means of every 16 keys, NVFP4 along the keys in V's
own 128 channels (E2M1 codes with one UE4M3 scale per channel and 16 keys, under the power-of-two tensor scale
v4_scale of max|v - vbar| / 2688), and v_mean = vbar / v4_scale.

The pass shares the basis launch (prep/basis.py): the launch's spare CTAs scan V for max|v - vbar| (the tensor scale
is needed before any code is formed) and, when the launch has room for it, run the pass on the tiles a counter hands
out (_vpass); v_kernel is the pass as its own launch otherwise."""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint8, Uint16, Uint32
from cutlass.cute.math import RoundingMode
from cutlass.cute.typing import Float8E4M3FN
from cutlass.utils import SmemAllocator

from ...common.dsl import FPRoundingMode, SaturationMode, cvt_f32x2_to_f8x2
from ...common.expmath import bf16x2, pow2_ceil, u2f
from ...common.mem import gptr, load_vector, sptr, store_vector
from ...common.ptx import f32x8_to_e2m1x8
from ..prep.basis import (
    EIG_THREADS,
    SM_RED,
    SM_VSTG,
    basis_cta,
    bf16_f32,
    group16,
    stage_rows,
    warp_max_f32,
)
from ..ptx.nvfp4pv import e4m3_to_f32
from ..ptx.shared import ld_global_nc_b32_hint_pred

V_THREADS = 128


@cute.jit
def tensor_scale(
    vmax_p: Int64,
    nspare: Int32,
    tid: Int32,
    red: Int32,
    cop: cutlass.Constexpr,
    NT: cutlass.Constexpr,
    SYNC: cutlass.Constexpr,
):
    """The power-of-two v4_scale from the max|v - vbar| of all nspare scan slots: the NT threads load the slots by
    stride (cop: the cache operator of the loads, "cg" for slots written by CTAs of this launch) and reduce by warp,
    then across the warps through red (SYNC: wait for the CTA's earlier readers of red first)."""
    c = Float32(0.0)
    for i in cutlass.range(tid, nspare, NT, unroll=1):
        c = cute.arch.fmax(
            c, u2f(cute.arch.load(gptr(Uint32, vmax_p + Int64(i) * 4), Uint32, cop=cop))
        )
    c = warp_max_f32(c)
    if cutlass.const_expr(SYNC):
        cute.arch.sync_threads()
    if (tid & 31) == 0:
        cute.arch.store(sptr(Float32, red + (tid >> 5) * 4), c)
    cute.arch.sync_threads()
    c = Float32(0.0)
    for wi in cutlass.range_constexpr(NT // 32):
        c = cute.arch.fmax(c, cute.arch.load(sptr(Float32, red + wi * 4), Float32))
    return pow2_ceil(cute.math.div(c, Float32(2688.0), rounding=RoundingMode.NEAREST_EVEN))


def group_mean(sums, cnt):
    """vbar of the thread's channels, BF16 values as f32 (sums: their f32 sums over the cnt valid keys of its 16-key
    group): the sums divided by the key count, as the scan that sets v4_scale divides them."""
    fc = Float32(cutlass.max(cnt, 1))
    mbw = [
        bf16x2(
            cute.math.div(sums[2 * e], fc, rounding=RoundingMode.NEAREST_EVEN),
            cute.math.div(sums[2 * e + 1], fc, rounding=RoundingMode.NEAREST_EVEN),
        )
        for e in range(len(sums) // 2)
    ]
    return [bf16_f32(w, j) for w in mbw for j in range(2)]


def channel_codes(val, m, cnt, i4):
    """V' of one channel of a 16-key group: the UE4M3 block scale of max|v - vbar| / v4_scale over the cnt valid
    keys (val(kk): key kk's value, m: vbar) in the low byte of a word, and the two words of E2M1 codes of
    (v - vbar) / (v4_scale x block scale), zero past cnt."""
    am = Float32(0.0)
    for kk in range(16):
        vv = val(kk)
        am = cute.arch.fmax(
            am, Float32(cutlass.select_(kk < cnt, cute.math.abs(vv - m), Float32(0.0)))
        )
    sf = cute.math.div(am * i4, Float32(6.0), rounding=RoundingMode.NEAREST_EVEN)
    sfw = Uint32(
        Uint16(
            cvt_f32x2_to_f8x2(
                Float32(0.0),
                sf,
                Float8E4M3FN,
                rnd=FPRoundingMode.RN,
                sat=SaturationMode.SATFINITE,
            )
        )
    )
    sff = e4m3_to_f32(sfw)
    rs = Float32(cutlass.select_(sff > Float32(0.0), cute.math.rcp(sff), Float32(1.0))) * i4
    ys = []
    for kk in range(16):
        vv = val(kk)
        ys.append(Float32(cutlass.select_(kk < cnt, (vv - m) * rs, Float32(0.0))))
    return sfw, f32x8_to_e2m1x8(ys[0:8]), f32x8_to_e2m1x8(ys[8:16])


def _fold(sums, vmx, vmn, fc, any_, mx):
    """mx updated with max|v - vbar| over the 8 channels, vbar = BF16(sums / fc); any_: the lane holds valid keys."""
    for c in range(8):
        mb = u2f(
            bf16x2(cute.math.div(sums[c], fc, rounding=RoundingMode.NEAREST_EVEN), Float32(0.0))
            << Uint32(16)
        )
        mx = cute.arch.fmax(
            mx,
            Float32(
                cutlass.select_(any_ != 0, cute.arch.fmax(vmx[c] - mb, mb - vmn[c]), Float32(0.0))
            ),
        )
    return mx


@cute.jit
def _store_slot(mx: Float32, tid: Int32, sb: Int32, vmax_p: Int64, slot: Int32):
    """The block's max|v - vbar| into V-scan slot `slot`."""
    mx = warp_max_f32(mx)
    red = sb + SM_RED
    if (tid & 31) == 0:
        cute.arch.store(sptr(Float32, red + (tid >> 5) * 4), mx)
    cute.arch.sync_threads()
    if tid == 0:
        c2 = Float32(0.0)
        for wi in cutlass.range_constexpr(16):
            c2 = cute.arch.fmax(c2, cute.arch.load(sptr(Float32, red + wi * 4), Float32))
        cute.arch.store(gptr(Float32, vmax_p + Int64(slot) * 4), c2)


@cute.jit
def _vscan(
    v_p: Int64,
    vmax_p: Int64,
    S: Int32,
    H: Int32,
    T: Int32,
    lo: Int32,
    hi: Int32,
    sp: Int32,
    tid: Int32,
    sb: Int32,
):
    """max|v - vbar| (vbar = BF16 16-key means) over the 128-key tiles [lo, hi) (flat over (batch, head, tile)); four
    tiles at a time, thread = (16-key group, 8-channel chunk). Writes slot sp."""
    u = tid & 127
    grp = u >> 4
    cc = u & 15
    sub4 = tid >> 7
    rstride = Int64(H) * 256
    mx = Float32(0.0)
    for t4 in cutlass.range(lo, hi, 4, unroll=1):
        tile = t4 + sub4
        tv = Int32(tile < hi)
        tile = cutlass.min(tile, hi - 1)
        bh = tile // T
        tt = tile - bh * T
        bb = bh // H
        hh = bh - bb * H
        k0 = tt * 128 + 16 * grp
        cnt = cutlass.max(cutlass.min(S - k0, 16), 0) * tv
        kb = cutlass.min(k0, S - 1)
        base = (
            v_p + ((Int64(bb) * Int64(S) + Int64(kb)) * Int64(H) + Int64(hh)) * 256 + Int64(cc * 16)
        )
        sums, vmx, vmn = group16(base, rstride, cnt)
        mx = _fold(sums, vmx, vmn, Float32(cutlass.max(cnt, 1)), Int32(cnt > 0), mx)
    _store_slot(mx, tid, sb, vmax_p, sp)


@cute.jit
def _vscan_gs(
    v_p: Int64,
    vmax_p: Int64,
    B: Int32,
    S: Int32,
    H: Int32,
    cta: Int32,
    ncta: Int32,
    tid: Int32,
    sb: Int32,
):
    """max|v - vbar| over all of V, grid-stride: item = (batch, 16-key group, head, 8-channel chunk). Writes slot
    cta."""
    NG = (S + 15) >> 4
    nitems = B * NG * H * 16
    mx = Float32(0.0)
    for it in cutlass.range(cta * EIG_THREADS + tid, nitems, ncta * EIG_THREADS, unroll=1):
        cc = it & 15
        rest = it >> 4
        hh = rest % H
        rest = rest // H
        gg = rest % NG
        bb = rest // NG
        k0 = gg * 16
        cnt = cutlass.min(S - k0, 16)
        base = (
            v_p + ((Int64(bb) * Int64(S) + Int64(k0)) * Int64(H) + Int64(hh)) * 256 + Int64(cc * 16)
        )
        rstride = Int64(H) * 256
        sums = [Float32(0.0)] * 8
        vmx = [Float32(-3.0e38)] * 8
        vmn = [Float32(3.0e38)] * 8
        for kk in cutlass.range_constexpr(16):
            w = ld_global_nc_b32_hint_pred(base + rstride * kk, Int32(kk < cnt), "evict_last", 4)
            ok = Int32(kk < cnt)
            for e in cutlass.range_constexpr(4):
                lo_ = u2f(w[e] << Uint32(16))
                hi_ = u2f(w[e] & Uint32(0xFFFF0000))
                sums[2 * e] = sums[2 * e] + lo_
                sums[2 * e + 1] = sums[2 * e + 1] + hi_
                vmx[2 * e] = cute.arch.fmax(
                    vmx[2 * e], Float32(cutlass.select_(ok != 0, lo_, Float32(-3.0e38)))
                )
                vmx[2 * e + 1] = cute.arch.fmax(
                    vmx[2 * e + 1], Float32(cutlass.select_(ok != 0, hi_, Float32(-3.0e38)))
                )
                vmn[2 * e] = cute.arch.fmin(
                    vmn[2 * e], Float32(cutlass.select_(ok != 0, lo_, Float32(3.0e38)))
                )
                vmn[2 * e + 1] = cute.arch.fmin(
                    vmn[2 * e + 1], Float32(cutlass.select_(ok != 0, hi_, Float32(3.0e38)))
                )
        fc = Float32(cnt)
        for c in cutlass.range_constexpr(8):
            mb = u2f(
                bf16x2(cute.math.div(sums[c], fc, rounding=RoundingMode.NEAREST_EVEN), Float32(0.0))
                << Uint32(16)
            )
            mx = cute.arch.fmax(mx, cute.arch.fmax(vmx[c] - mb, mb - vmn[c]))
    _store_slot(mx, tid, sb, vmax_p, cta)


@cute.jit
def _vpass(
    v_p: Int64,
    vmax_p: Int64,
    v4_p: Int64,
    v4sf_p: Int64,
    vmean_p: Int64,
    v4s_p: Int64,
    done_p: Int64,
    S: Int32,
    H: Int32,
    T: Int32,
    ntiles: Int32,
    pid: Int32,
    nspare: Int32,
    tid: Int32,
    sb: Int32,
):
    """The V pass in the basis launch, tiles handed out by a counter in descending order to two 256-thread halves per
    CTA, thread = (16-key group, 4-channel quad); V' codes / scales transposed through smem ([c&3][quad][64 B] /
    [c][8 B])."""
    red = sb + SM_RED
    # the tensor scale from all scan slots (written by this launch's scan CTAs: cop "cg"; red was the scan's)
    v4s = tensor_scale(vmax_p, nspare, tid, red, "cg", EIG_THREADS, True)
    if pid == 0 and tid == 0:
        cute.arch.store(gptr(Float32, v4s_p), v4s)
    i4 = cute.math.rcp(v4s)
    half = tid >> 8
    st = tid & 255
    grp = st & 7
    cq = st >> 3
    vin = sb + half * 32768  # next-tile input, rows of 256 B, 16-byte part ^ (row >> 4)
    stg = sb + SM_VSTG + half * 9216
    sfs = stg + 8192
    SP2 = T * 64
    SP16 = T * 8
    slot = red + 192 + half * 4
    ctr = done_p + 4
    # tile t: issue its 128 x 256 B rows into vin (zero past S)
    if st == 0:
        cute.arch.store(
            sptr(Uint32, slot),
            cute.arch.atomic_add(gptr(Uint32, ctr), Uint32(1), sem="relaxed", scope="gpu"),
        )
    cute.arch.barrier(barrier_id=1 + half, number_of_threads=256)
    tix = Int32(cute.arch.load(sptr(Uint32, slot), Uint32))
    if tix < ntiles:
        stage_rows(v_p, vin, st, ntiles - 1 - tix, S, H, T)
    cute.arch.cp_async_commit_group()
    cute.arch.barrier(barrier_id=1 + half, number_of_threads=256)
    if st == 0:
        cute.arch.store(
            sptr(Uint32, slot),
            cute.arch.atomic_add(gptr(Uint32, ctr), Uint32(1), sem="relaxed", scope="gpu"),
        )
    cute.arch.barrier(barrier_id=1 + half, number_of_threads=256)
    nix = Int32(cute.arch.load(sptr(Uint32, slot), Uint32))
    while tix < ntiles:
        tile = ntiles - 1 - tix
        bh = tile // T
        tt = tile - bh * T
        k0 = tt * 128 + 16 * grp
        cnt = cutlass.max(cutlass.min(S - k0, 16), 0)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier(barrier_id=1 + half, number_of_threads=256)
        wv = cute.make_rmem_tensor((32,), Uint32)
        sums = [Float32(0.0)] * 4
        for kk in cutlass.range_constexpr(16):
            w = load_vector(
                sptr(
                    Uint32, vin + (16 * grp + kk) * 256 + (((cq >> 1) ^ grp) * 16) + (cq & 1) * 8, 8
                ),
                2,
            )
            for e in cutlass.range_constexpr(2):
                wv[kk * 2 + e] = w[e]
                sums[2 * e] = sums[2 * e] + u2f(w[e] << Uint32(16))
                sums[2 * e + 1] = sums[2 * e + 1] + u2f(w[e] & Uint32(0xFFFF0000))
        cute.arch.barrier(barrier_id=1 + half, number_of_threads=256)
        # prefetch the next tile while this one is processed
        if nix < ntiles:
            stage_rows(v_p, vin, st, ntiles - 1 - nix, S, H, T)
        cute.arch.cp_async_commit_group()
        # the mean of the 16-key group; v_mean = vbar / v4_scale (exact)
        mf = group_mean(sums, cnt)
        mean_dst = (
            vmean_p + ((Int64(bh) * Int64(SP16) + Int64(tt * 8 + grp)) * 128 + Int64(cq * 4)) * 2
        )
        mean_words = [bf16x2(mf[2 * e] * i4, mf[2 * e + 1] * i4) for e in range(2)]
        store_vector(gptr(Uint32, mean_dst, 8), mean_words, cop="cs")
        for c in cutlass.range_constexpr(4):
            sfw, cw0, cw1 = channel_codes(
                lambda kk: bf16_f32(wv[kk * 2 + c // 2], c % 2), mf[c], cnt, i4
            )
            store_vector(sptr(Uint32, stg + (c * 32 + cq) * 64 + grp * 8, 8), (cw0, cw1))
            cute.arch.store(sptr(Uint8, sfs + (cq * 4 + c) * 8 + grp), sfw.to(Uint8))
        cute.arch.barrier(barrier_id=1 + half, number_of_threads=256)
        for m in cutlass.range_constexpr(2):
            ci = st + 256 * m
            ch = ci >> 2
            part = ci & 3
            w4 = load_vector(
                sptr(Uint32, stg + ((ch & 3) * 32 + (ch >> 2)) * 64 + part * 16, 16), 4
            )
            code_dst = (
                v4_p + (Int64(bh) * 128 + Int64(ch)) * Int64(SP2) + Int64(tt * 64 + part * 16)
            )
            store_vector(gptr(Uint32, code_dst, 16), w4, cop="cs")
        if st < 128:
            s2 = load_vector(sptr(Uint32, sfs + st * 8, 8), 2)
            scale_dst = v4sf_p + (Int64(bh) * 128 + Int64(st)) * Int64(SP16) + Int64(tt * 8)
            store_vector(gptr(Uint32, scale_dst, 8), s2, cop="cs")
        if st == 0:
            cute.arch.store(
                sptr(Uint32, slot),
                cute.arch.atomic_add(gptr(Uint32, ctr), Uint32(1), sem="relaxed", scope="gpu"),
            )
        cute.arch.barrier(barrier_id=1 + half, number_of_threads=256)
        tix = nix
        nix = Int32(cute.arch.load(sptr(Uint32, slot), Uint32))
    cute.arch.cp_async_commit_group()
    cute.arch.cp_async_wait_group(0)


@cute.kernel
def basis_kernel(
    q_p: Int64,
    k_p: Int64,
    v_p: Int64,
    rows_p: Int64,
    had_p: Int64,
    ut_p: Int64,
    vmax_p: Int64,
    done_p: Int64,
    v4_p: Int64,
    v4sf_p: Int64,
    vmean_p: Int64,
    v4s_p: Int64,
    B: Int32,
    S: Int32,
    H: Int32,
    T: Int32,
    NQK: Int32,
    BH: Int32,
    nspare: Int32,
    eigwait: Int32,
    FUSED: cutlass.Constexpr,
):
    """The basis launch of the NVFP4-PV kernel: basis_cta with the scan and the pass of V'."""
    basis_cta(
        q_p,
        k_p,
        rows_p,
        had_p,
        ut_p,
        done_p,
        S,
        H,
        T,
        NQK,
        BH,
        nspare,
        eigwait,
        FUSED,
        lambda cta, tid, sb: _vscan_gs(v_p, vmax_p, B, S, H, cta, nspare, tid, sb),
        lambda lo, hi, sp, tid, sb: _vscan(v_p, vmax_p, S, H, T, lo, hi, sp, tid, sb),
        lambda ntiles, pid, tid, sb: _vpass(
            v_p, vmax_p, v4_p, v4sf_p, vmean_p, v4s_p, done_p, S, H, T, ntiles, pid, nspare, tid, sb
        ),
    )


@cute.kernel
def v_kernel(
    v_p: Int64,
    vmax_p: Int64,
    v4_p: Int64,
    v4sf_p: Int64,
    vmean_p: Int64,
    v4s_p: Int64,
    S: Int32,
    H: Int32,
    T: Int32,
    nspare: Int32,
    nitems: Int32,
    ncta: Int32,
):
    """One 128-key tile of one (batch, head) per iteration: thread (group of 16 keys, 8-channel chunk) loads
    16 x 16 bytes; V' codes and scales are transposed through smem ([e][chunk][64 B] / [channel][8 B])."""
    tid, _, _ = cute.arch.thread_idx()
    cta, _, _ = cute.arch.block_idx()
    smem = SmemAllocator()
    buf = smem.allocate_array(Uint32, (64 + 2048 + 256), byte_alignment=128)
    red = Int32(buf.toint())
    stg = red + 256
    sfs = stg + 8192
    v4s = tensor_scale(vmax_p, nspare, tid, red, None, V_THREADS, False)
    if cta == 0 and tid == 0:
        cute.arch.store(gptr(Float32, v4s_p), v4s)
    i4 = cute.math.rcp(v4s)
    lane = tid & 31
    warp = tid >> 5
    grp = lane & 7
    cc = (lane >> 3) + 4 * warp
    SP2 = T * 64
    SP16 = T * 8
    rstride = Int64(H) * 256
    item_lo = (cta * nitems) // ncta
    item_hi = ((cta + 1) * nitems) // ncta
    for item in cutlass.range(item_lo, item_hi, unroll=1):
        bh = item // T
        tile = item - bh * T
        bb = bh // H
        hh = bh - bb * H
        k0 = tile * 128 + 16 * grp
        cnt = cutlass.max(cutlass.min(S - k0, 16), 0)
        kb = cutlass.min(k0, S - 1)
        base = (
            v_p + ((Int64(bb) * Int64(S) + Int64(kb)) * Int64(H) + Int64(hh)) * 256 + Int64(cc * 16)
        )
        wv = cute.make_rmem_tensor((64,), Uint32)
        sums = [Float32(0.0)] * 8
        for kk in cutlass.range_constexpr(16):
            w = ld_global_nc_b32_hint_pred(base + rstride * kk, Int32(kk < cnt), "evict_first", 4)
            for e in cutlass.range_constexpr(4):
                wv[kk * 4 + e] = w[e]
                sums[2 * e] = sums[2 * e] + u2f(w[e] << Uint32(16))
                sums[2 * e + 1] = sums[2 * e + 1] + u2f(w[e] & Uint32(0xFFFF0000))
        mf = group_mean(sums, cnt)
        # v_mean = vbar / v4_scale (exact), 8 channels = 16 bytes
        mean_dst = (
            vmean_p + ((Int64(bh) * Int64(SP16) + Int64(tile * 8 + grp)) * 128 + Int64(cc * 8)) * 2
        )
        mean_words = [bf16x2(mf[2 * e] * i4, mf[2 * e + 1] * i4) for e in range(4)]
        store_vector(gptr(Uint32, mean_dst, 16), mean_words, cop="cs")
        # per channel: amax of y = (v - vbar) / v4_scale over the valid keys, UE4M3 block scale, codes
        for c in cutlass.range_constexpr(8):
            sfw, cw0, cw1 = channel_codes(
                lambda kk: bf16_f32(wv[kk * 4 + c // 2], c % 2), mf[c], cnt, i4
            )
            store_vector(sptr(Uint32, stg + ((c * 16 + cc) * 64 + grp * 8), 8), (cw0, cw1))
            cute.arch.store(sptr(Uint8, sfs + (cc * 8 + c) * 8 + grp), sfw.to(Uint8))
        cute.arch.sync_threads()
        # V' codes: 128 channel rows x 64 bytes; scales: 128 rows x 8 bytes
        for m in cutlass.range_constexpr(4):
            ci = tid + 128 * m
            ch = ci >> 2
            part = ci & 3
            w4 = load_vector(
                sptr(Uint32, stg + ((ch & 7) * 16 + (ch >> 3)) * 64 + part * 16, 16), 4
            )
            code_dst = (
                v4_p + (Int64(bh) * 128 + Int64(ch)) * Int64(SP2) + Int64(tile * 64 + part * 16)
            )
            store_vector(gptr(Uint32, code_dst, 16), w4, cop="cs")
        s2 = load_vector(sptr(Uint32, sfs + tid * 8, 8), 2)
        scale_dst = v4sf_p + (Int64(bh) * 128 + Int64(tid)) * Int64(SP16) + Int64(tile * 8)
        store_vector(gptr(Uint32, scale_dst, 8), s2, cop="cs")
        cute.arch.sync_threads()
