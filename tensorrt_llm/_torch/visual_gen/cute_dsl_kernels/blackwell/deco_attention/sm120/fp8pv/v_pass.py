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
"""V pass of the sm_120 FP8-PV kernel: E4M3 V with smoothing over the 128 keys of a tile. Every key tile of a (batch,
head) loses its mean over its keys below S (rounded to BF16) before the E4M3 cast, so the cast spends its range on
the variation inside a tile: v8 = E4M3((V - vbar) / v8_scale) and v_mean = vbar / v8_scale, with v8_scale the power
of two at or above max|V - vbar| / 448 of the call. v_mean holds one row per 16 keys, a tile's mean in all 8 of its
rows: the rows the attention's means MMA weights by the 16-key block sums of P.

The pass shares the basis launch (prep/basis.py): the launch's spare CTAs scan V for max|V - vbar| (the tensor scale
needs it before any byte is cast) and, when the launch has room for it, run the pass on the tiles a counter hands
out (_vpass); v_kernel is the pass as its own launch otherwise."""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32
from cutlass.cute.math import RoundingMode
from cutlass.utils import SmemAllocator

from ...common.expmath import bf16x2, e4m3x4, pow2_ceil_448, u2f, warp_max
from ...common.mem import gptr, load_vector, sptr, store_vector
from ..prep.basis import EIG_THREADS, SM_RED, basis_cta, bf16_f32, group16, stage_rows
from ..ptx.fp8pv import st_global_cs_b32_pred, st_global_cs_b32_v2_pred
from ..ptx.shared import ld_global_nc_b32_hint_pred
from ..valid_len import valid_len

V_THREADS = 128


def sum8(xs, n):
    """xs and the count n summed over the 8 lanes 8 j .. 8 j + 7 (butterfly 1, 2, 4)."""
    nf = Float32(n)
    for m in (1, 2, 4):
        xs = [x + cute.arch.shuffle_sync_bfly(x, offset=m) for x in xs]
        nf = nf + cute.arch.shuffle_sync_bfly(nf, offset=m)
    return xs, nf


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
    """The power-of-two v8_scale from the max|v - vbar| of all nspare scan slots, the range of the smoothed V: the NT
    threads load the slots by stride (cop: the cache operator of the loads, "cg" for slots written by CTAs of this
    launch) and reduce by warp, then across the warps through red (SYNC: wait for the CTA's earlier readers of red
    first)."""
    c = Float32(0.0)
    for i in cutlass.range(tid, nspare, NT, unroll=1):
        c = cute.arch.fmax(
            c, u2f(cute.arch.load(gptr(Uint32, vmax_p + Int64(i) * 4), Uint32, cop=cop))
        )
    c = warp_max(c)
    if cutlass.const_expr(SYNC):
        cute.arch.sync_threads()
    if (tid & 31) == 0:
        cute.arch.store(sptr(Float32, red + (tid >> 5) * 4), c)
    cute.arch.sync_threads()
    c = Float32(0.0)
    for wi in cutlass.range_constexpr(NT // 32):
        c = cute.arch.fmax(c, cute.arch.load(sptr(Float32, red + wi * 4), Float32))
    return pow2_ceil_448(c)


def tile_mean(sums, cnt):
    """vbar of the thread's channels over its tile, BF16 values as f32 (sums: their f32 sums over the cnt valid keys
    of the thread's 16-key group; the tile's 8 groups sit in lanes 8 j .. 8 j + 7): the tile sums divided by the key
    count, as the scan that sets v8_scale divides them."""
    sums, nf = sum8(sums, cnt)
    fc = cute.arch.fmax(nf, Float32(1.0))
    mbw = [
        bf16x2(
            cute.math.div(sums[2 * e], fc, rounding=RoundingMode.NEAREST_EVEN),
            cute.math.div(sums[2 * e + 1], fc, rounding=RoundingMode.NEAREST_EVEN),
        )
        for e in range(len(sums) // 2)
    ]
    return [bf16_f32(w, j) for w in mbw for j in range(2)]


def _tile_fold(sums, vmx, vmn, cnt, mx):
    """mx updated with max|v - vbar| over the thread's 8 channels, vbar the BF16 mean of the valid keys of the lane's
    tile (sums, vmx, vmn: the sums, maxima and minima of its 16-key group; cnt: its valid keys). Only the sums and
    counts are added across the tile's lanes, the maxima fold in the block's maximum."""
    sums, nf = sum8(sums, cnt)
    fc = cute.arch.fmax(nf, Float32(1.0))
    any_ = Int32(cnt > 0)
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
    mx = warp_max(mx)
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
    lens_p,
):
    """max|v - vbar| (vbar = BF16 tile means) over the 128-key tiles [lo, hi) (flat over (batch, head, tile)); four
    tiles at a time, thread = (8-channel chunk, 16-key group), the groups of a chunk in lanes 8 j .. 8 j + 7. Writes
    slot sp. lens_p: None, or the global address of the valid prefix lengths: the keys of a batch element then end
    at its length."""
    u = tid & 127
    grp = u & 7
    cc = u >> 3
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
        nk = S if cutlass.const_expr(lens_p is None) else valid_len(lens_p, bb, S)
        cnt = cutlass.max(cutlass.min(nk - k0, 16), 0) * tv
        kb = cutlass.min(k0, S - 1)
        base = (
            v_p + ((Int64(bb) * Int64(S) + Int64(kb)) * Int64(H) + Int64(hh)) * 256 + Int64(cc * 16)
        )
        # with lengths: a tile without a valid key is skipped
        live = cutlass.const_expr(lens_p is None) or tt * 128 < nk
        if live:
            sums, vmx, vmn = group16(base, rstride, cnt)
            mx = _tile_fold(sums, vmx, vmn, cnt, mx)
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
    lens_p,
):
    """max|v - vbar| over all of V, grid-stride: item = (batch, tile, head, 8-channel chunk, 16-key group), the 8
    groups of a tile in lanes 8 j .. 8 j + 7 (a warp takes 32 consecutive items, so its lanes run the same
    iterations). Writes slot cta. lens_p: as in _vscan."""
    T = (S + 127) >> 7
    nitems = B * T * H * 128
    mx = Float32(0.0)
    for it in cutlass.range(cta * EIG_THREADS + tid, nitems, ncta * EIG_THREADS, unroll=1):
        grp = it & 7
        cc = (it >> 3) & 15
        rest = it >> 7
        hh = rest % H
        rest = rest // H
        tt = rest % T
        bb = rest // T
        k0 = tt * 128 + 16 * grp
        nk = S if cutlass.const_expr(lens_p is None) else valid_len(lens_p, bb, S)
        cnt = cutlass.max(cutlass.min(nk - k0, 16), 0)
        kb = cutlass.min(k0, S - 1)
        base = (
            v_p + ((Int64(bb) * Int64(S) + Int64(kb)) * Int64(H) + Int64(hh)) * 256 + Int64(cc * 16)
        )
        # with lengths: a tile without a valid key is skipped
        live = cutlass.const_expr(lens_p is None) or tt * 128 < nk
        if live:
            sums, vmx, vmn = group16(base, Int64(H) * 256, cnt)
            mx = _tile_fold(sums, vmx, vmn, cnt, mx)
    _store_slot(mx, tid, sb, vmax_p, cta)


@cute.jit
def _vpass(
    v_p: Int64,
    vmax_p: Int64,
    v8_p: Int64,
    vmean_p: Int64,
    v8s_p: Int64,
    done_p: Int64,
    S: Int32,
    H: Int32,
    T: Int32,
    ntiles: Int32,
    pid: Int32,
    nspare: Int32,
    tid: Int32,
    sb: Int32,
    lens_p,
):
    """The V pass in the basis launch, tiles handed out by a counter in descending order to two 256-thread halves per
    CTA, thread = (16-key group, 4-channel quad), the 8 groups of a quad in lanes 8 j .. 8 j + 7: the E4M3 V - vbar of
    the keys below S and v_mean = vbar / v8_scale in the tile's 8 rows. lens_p: None, or the global address of the
    valid prefix lengths: the mean of a tile is then over the keys below the length of its batch element, the keys
    from there on get zero codes, and a tile without a valid key is left as it is."""
    red = sb + SM_RED
    # the tensor scale from all scan slots (written by this launch's scan CTAs: cop "cg"; red was the scan's)
    v8s = tensor_scale(vmax_p, nspare, tid, red, "cg", EIG_THREADS, True)
    if pid == 0 and tid == 0:
        cute.arch.store(gptr(Float32, v8s_p), v8s)
    i8 = cute.math.rcp(v8s)
    half = tid >> 8
    st = tid & 255
    grp = st & 7
    cq = st >> 3
    vin = sb + half * 32768  # next-tile input, rows of 256 B, 16-byte part ^ (row >> 4)
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
        stage_rows(v_p, vin, st, ntiles - 1 - tix, S, H, T, lens_p)
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
        bb = bh // H
        hh = bh - bb * H
        k0 = tt * 128 + 16 * grp
        cnt_s = cutlass.max(cutlass.min(S - k0, 16), 0)
        nk = S if cutlass.const_expr(lens_p is None) else valid_len(lens_p, bb, S)
        cnt = (
            cnt_s
            if cutlass.const_expr(lens_p is None)
            else cutlass.max(cutlass.min(nk - k0, 16), 0)
        )
        kb = cutlass.min(k0, S - 1)
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
            stage_rows(v_p, vin, st, ntiles - 1 - nix, S, H, T, lens_p)
        cute.arch.cp_async_commit_group()
        # with lengths: a tile without a valid key is skipped
        live = cutlass.const_expr(lens_p is None) or tt * 128 < nk
        if live:
            # the mean of the tile; v_mean = vbar / v8_scale (exact), in the row of the thread's group
            mf = tile_mean(sums, cnt)
            mean_dst = (
                vmean_p
                + ((Int64(bh) * Int64(SP16) + Int64(tt * 8 + grp)) * 128 + Int64(cq * 4)) * 2
            )
            mean_words = [bf16x2(mf[2 * e] * i8, mf[2 * e + 1] * i8) for e in range(2)]
            store_vector(gptr(Uint32, mean_dst, 8), mean_words, cop="cs")
            # E4M3 (V - vbar) / v8_scale: 4 channels per key = 4 bytes, running address
            v8a = (
                v8_p
                + ((Int64(bb) * Int64(S) + Int64(kb)) * Int64(H) + Int64(hh)) * 128
                + Int64(cq * 4)
            )
            v8st = Int64(H) * 128
            for kk in cutlass.range_constexpr(16):
                w0 = wv[kk * 2]
                w1 = wv[kk * 2 + 1]
                pk = e4m3x4(
                    (u2f(w0 << Uint32(16)) - mf[0]) * i8,
                    (u2f(w0 & Uint32(0xFFFF0000)) - mf[1]) * i8,
                    (u2f(w1 << Uint32(16)) - mf[2]) * i8,
                    (u2f(w1 & Uint32(0xFFFF0000)) - mf[3]) * i8,
                )
                if cutlass.const_expr(lens_p is not None):
                    pk = Uint32(cutlass.select_(kk < cnt, pk, Uint32(0)))
                st_global_cs_b32_pred(v8a, pk, Int32(kk < cnt_s))
                v8a = v8a + v8st
        if st == 0:
            cute.arch.store(
                sptr(Uint32, slot),
                cute.arch.atomic_add(gptr(Uint32, ctr), Uint32(1), sem="relaxed", scope="gpu"),
            )
        cute.arch.barrier(barrier_id=1 + half, number_of_threads=256)
        tix = nix
        nix = Int32(cute.arch.load(sptr(Uint32, slot), Uint32))


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
    v8_p: Int64,
    vmean_p: Int64,
    v8s_p: Int64,
    B: Int32,
    S: Int32,
    H: Int32,
    T: Int32,
    NQK: Int32,
    BH: Int32,
    nspare: Int32,
    FUSED: cutlass.Constexpr,
    lens_p,
    rows_mode,
):
    """The basis launch of the FP8-PV kernel: basis_cta with the scan and the pass of the smoothed V. lens_p,
    rows_mode: None, or the global address of the valid prefix lengths and the rows mode of the eigensolver."""
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
        FUSED,
        lambda cta, tid, sb: _vscan_gs(v_p, vmax_p, B, S, H, cta, nspare, tid, sb, lens_p),
        lambda lo, hi, sp, tid, sb: _vscan(v_p, vmax_p, S, H, T, lo, hi, sp, tid, sb, lens_p),
        lambda ntiles, pid, tid, sb: _vpass(
            v_p, vmax_p, v8_p, vmean_p, v8s_p, done_p, S, H, T, ntiles, pid, nspare, tid, sb, lens_p
        ),
        None if lens_p is None else (lens_p, rows_mode),
    )


@cute.kernel
def v_kernel(
    v_p: Int64,
    vmax_p: Int64,
    v8_p: Int64,
    vmean_p: Int64,
    v8s_p: Int64,
    S: Int32,
    H: Int32,
    T: Int32,
    nspare: Int32,
    nitems: Int32,
    ncta: Int32,
    lens_p,
):
    """One 128-key tile of one (batch, head) per iteration: thread (16-key group, 8-channel chunk), the tile's groups
    in lanes 8 j .. 8 j + 7, loads 16 x 16 bytes and stores the E4M3 V - vbar of the keys below S and its row of
    v_mean. lens_p: as in _vpass."""
    tid, _, _ = cute.arch.thread_idx()
    cta, _, _ = cute.arch.block_idx()
    smem = SmemAllocator()
    buf = smem.allocate_array(Uint32, 64, byte_alignment=128)
    red = Int32(buf.toint())
    v8s = tensor_scale(vmax_p, nspare, tid, red, None, V_THREADS, False)
    if cta == 0 and tid == 0:
        cute.arch.store(gptr(Float32, v8s_p), v8s)
    i8 = cute.math.rcp(v8s)
    lane = tid & 31
    warp = tid >> 5
    grp = lane & 7
    cc = (lane >> 3) + 4 * warp
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
        cnt_s = cutlass.max(cutlass.min(S - k0, 16), 0)
        nk = S if cutlass.const_expr(lens_p is None) else valid_len(lens_p, bb, S)
        cnt = (
            cnt_s
            if cutlass.const_expr(lens_p is None)
            else cutlass.max(cutlass.min(nk - k0, 16), 0)
        )
        kb = cutlass.min(k0, S - 1)
        base = (
            v_p + ((Int64(bb) * Int64(S) + Int64(kb)) * Int64(H) + Int64(hh)) * 256 + Int64(cc * 16)
        )
        # with lengths: a tile without a valid key is skipped
        live = cutlass.const_expr(lens_p is None) or tile * 128 < nk
        if live:
            wv = cute.make_rmem_tensor((64,), Uint32)
            sums = [Float32(0.0)] * 8
            for kk in cutlass.range_constexpr(16):
                w = ld_global_nc_b32_hint_pred(
                    base + rstride * kk, Int32(kk < cnt), "evict_first", 4
                )
                for e in cutlass.range_constexpr(4):
                    wv[kk * 4 + e] = w[e]
                    sums[2 * e] = sums[2 * e] + u2f(w[e] << Uint32(16))
                    sums[2 * e + 1] = sums[2 * e + 1] + u2f(w[e] & Uint32(0xFFFF0000))
            mf = tile_mean(sums, cnt)
            # v_mean = vbar / v8_scale (exact), 8 channels = 16 bytes
            mean_dst = (
                vmean_p
                + ((Int64(bh) * Int64(SP16) + Int64(tile * 8 + grp)) * 128 + Int64(cc * 8)) * 2
            )
            mean_words = [bf16x2(mf[2 * e] * i8, mf[2 * e + 1] * i8) for e in range(4)]
            store_vector(gptr(Uint32, mean_dst, 16), mean_words, cop="cs")
            # E4M3 (V - vbar) / v8_scale: 8 channels per key = 8 bytes
            v8b = (
                v8_p
                + ((Int64(bb) * Int64(S) + Int64(kb)) * Int64(H) + Int64(hh)) * 128
                + Int64(cc * 8)
            )
            for kk in cutlass.range_constexpr(16):
                w4 = [wv[kk * 4 + e] for e in range(4)]
                lo = e4m3x4(
                    (u2f(w4[0] << Uint32(16)) - mf[0]) * i8,
                    (u2f(w4[0] & Uint32(0xFFFF0000)) - mf[1]) * i8,
                    (u2f(w4[1] << Uint32(16)) - mf[2]) * i8,
                    (u2f(w4[1] & Uint32(0xFFFF0000)) - mf[3]) * i8,
                )
                hi = e4m3x4(
                    (u2f(w4[2] << Uint32(16)) - mf[4]) * i8,
                    (u2f(w4[2] & Uint32(0xFFFF0000)) - mf[5]) * i8,
                    (u2f(w4[3] << Uint32(16)) - mf[6]) * i8,
                    (u2f(w4[3] & Uint32(0xFFFF0000)) - mf[7]) * i8,
                )
                if cutlass.const_expr(lens_p is not None):
                    lo = Uint32(cutlass.select_(kk < cnt, lo, Uint32(0)))
                    hi = Uint32(cutlass.select_(kk < cnt, hi, Uint32(0)))
                st_global_cs_b32_v2_pred(v8b + Int64(H) * 128 * kk, lo, hi, Int32(kk < cnt_s))
        cute.arch.sync_threads()
        cute.arch.sync_threads()
