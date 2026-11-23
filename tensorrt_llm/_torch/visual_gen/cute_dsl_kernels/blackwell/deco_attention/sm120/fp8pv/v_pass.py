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
"""V pass of the sm_120 FP8-PV kernel: E4M3 V with smoothing over the 128 keys of a tile.

* Every key tile of a (batch, head) loses its mean over its keys below S (rounded to BF16) before the E4M3 cast,
  so the cast spends its range on the variation inside a tile: v8 = E4M3((V - vbar) / v8_scale) and
  v_mean = vbar / v8_scale, with v8_scale the power of two at or above max|V - vbar| / 448 of the call.
* v_mean holds one row per 16 keys, a tile's mean in all 8 of its rows: the rows that the means MMA of the
  attention weights by the 16-key block sums of P.
* The pass shares the basis launch (prep/basis.py): its spare CTAs scan V for max|V - vbar| (the tensor scale
  needs it before any byte is cast) and, when the launch has room for it, run the pass on the tiles a counter
  hands out (_vpass); v_kernel is the pass as its own launch otherwise.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32
from cutlass.utils import SmemAllocator

from ...common.expmath import RN, bf16x2, e4m3x4, pow2_ceil_448, u2f
from ...common.mem import gptr, load_vector, sptr, store_vector
from ..prep.basis import (
    DONE_TILES,
    EIG_THREADS,
    RED_TILE,
    SM_RED,
    VIN_BYTES,
    basis_cta,
    bf16_f32,
    group16,
    key_count,
    next_tile,
    row_addr,
    scan_max,
    stage_rows,
    store_slot,
)
from ..ptx.fp8pv import st_global_cs_b32_pred, st_global_cs_b32_v2_pred
from ..ptx.shared import ld_global_nc_b32_hint_pred
from ..valid_len import valid_len

V_THREADS = 128


def sum8(xs, n):
    """Return xs and the count n summed over the 8 lanes 8 j .. 8 j + 7 (butterfly 1, 2, 4)."""
    nf = Float32(n)
    for m in (1, 2, 4):
        xs = [x + cute.arch.shuffle_sync_bfly(x, offset=m) for x in xs]
        nf = nf + cute.arch.shuffle_sync_bfly(nf, offset=m)
    return xs, nf


def tile_mean(sums, cnt):
    """Return vbar of the thread's channels over its tile, BF16 values as f32: the tile sums over the key count."""
    sums, nf = sum8(sums, cnt)
    fc = cute.arch.fmax(nf, Float32(1.0))
    mbw = [
        bf16x2(
            cute.math.div(sums[2 * e], fc, rounding=RN),
            cute.math.div(sums[2 * e + 1], fc, rounding=RN),
        )
        for e in range(len(sums) // 2)
    ]
    return [bf16_f32(w, j) for w in mbw for j in range(2)]


def _tile_fold(sums, vmx, vmn, cnt, mx):
    """Return mx updated with max|v - vbar| of the thread's 8 channels; only sums and counts cross the tile's lanes."""
    sums, nf = sum8(sums, cnt)
    fc = cute.arch.fmax(nf, Float32(1.0))
    any_ = Int32(cnt > 0)
    for c in range(8):
        mb = u2f(bf16x2(cute.math.div(sums[c], fc, rounding=RN), Float32(0.0)) << Uint32(16))
        mx = cute.arch.fmax(
            mx,
            Float32(
                cutlass.select_(any_ != 0, cute.arch.fmax(vmx[c] - mb, mb - vmn[c]), Float32(0.0))
            ),
        )
    return mx


@cute.jit
def _vscan(
    v_p: Int64,
    vmax_p: Int64,
    S: Int32,
    H: Int32,
    T: Int32,
    lo: Int32,
    hi: Int32,
    slot: Int32,
    tid: Int32,
    sb: Int32,
    lens_p,
):
    """Scan the 128-key tiles [lo, hi) (flat over (batch, head, tile)) for max|v - vbar| and write slot `slot`.
    Four tiles at a time; thread = (8-channel chunk, 16-key group), the groups of a chunk in lanes 8 j .. 8 j + 7."""
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
        nk = valid_len(lens_p, bb, S)
        cnt = key_count(k0, nk) * tv
        base = row_addr(v_p, bb, cutlass.min(k0, S - 1), hh, S, H, 256) + Int64(cc * 16)
        # with lengths: a tile without a valid key is skipped
        live = cutlass.const_expr(lens_p is None) or tt * 128 < nk
        if live:
            sums, vmx, vmn = group16(base, rstride, cnt)
            mx = _tile_fold(sums, vmx, vmn, cnt, mx)
    store_slot(mx, tid, sb, vmax_p, slot)


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
    """Scan all of V for max|v - vbar|, grid-stride, and write slot cta.
    Item = (batch, tile, head, 8-channel chunk, 16-key group), the 8 groups of a tile in lanes 8 j .. 8 j + 7: a warp
    takes 32 consecutive items, so its lanes run the same iterations."""
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
        nk = valid_len(lens_p, bb, S)
        cnt = key_count(k0, nk)
        base = row_addr(v_p, bb, cutlass.min(k0, S - 1), hh, S, H, 256) + Int64(cc * 16)
        # with lengths: a tile without a valid key is skipped
        live = cutlass.const_expr(lens_p is None) or tt * 128 < nk
        if live:
            sums, vmx, vmn = group16(base, Int64(H) * 256, cnt)
            mx = _tile_fold(sums, vmx, vmn, cnt, mx)
    store_slot(mx, tid, sb, vmax_p, cta)


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
    """Run the V pass in the basis launch: the E4M3 V - vbar of the keys below S and v_mean in the 8 rows of the tile.

    * A counter hands out the tiles in descending order to two 256-thread halves per CTA; thread = (16-key group,
      4-channel quad), the 8 groups of a quad in lanes 8 j .. 8 j + 7.
    * With lengths the mean of a tile is over the keys below the length of its batch element, the keys from there on
      get zero codes, and a tile without a valid key is left as it is.
    """
    # the tensor scale from all scan slots (written by this launch's scan CTAs: cop "cg"; red was the scan's)
    red = sb + SM_RED
    v8s = pow2_ceil_448(scan_max(vmax_p, nspare, tid, red, "cg", EIG_THREADS, True))
    if pid == 0 and tid == 0:
        cute.arch.store(gptr(Float32, v8s_p), v8s)
    i8 = cute.math.rcp(v8s)
    half = tid >> 8
    tid_h = tid & 255
    grp = tid_h & 7
    cq = tid_h >> 3
    vin = sb + half * VIN_BYTES  # next-tile input, rows of 256 B, 16-byte part ^ (row >> 4)
    SP16 = T * 8
    word = red + RED_TILE + half * 4
    ctr = done_p + DONE_TILES
    # tile t: issue its 128 x 256 B rows into vin (zero past S)
    tix = next_tile(word, ctr, tid_h, half)
    if tix < ntiles:
        stage_rows(v_p, vin, tid_h, ntiles - 1 - tix, S, H, T, lens_p)
    cute.arch.cp_async_commit_group()
    cute.arch.barrier(barrier_id=1 + half, number_of_threads=256)
    nix = next_tile(word, ctr, tid_h, half)
    while tix < ntiles:
        tile = ntiles - 1 - tix
        bh = tile // T
        tt = tile - bh * T
        bb = bh // H
        hh = bh - bb * H
        k0 = tt * 128 + 16 * grp
        cnt_s = key_count(k0, S)
        nk = valid_len(lens_p, bb, S)
        cnt = cnt_s if cutlass.const_expr(lens_p is None) else key_count(k0, nk)
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
            stage_rows(v_p, vin, tid_h, ntiles - 1 - nix, S, H, T, lens_p)
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
            v8a = row_addr(v8_p, bb, kb, hh, S, H, 128) + Int64(cq * 4)
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
        tix = nix
        nix = next_tile(word, ctr, tid_h, half)


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
    """Run the basis launch of the FP8-PV kernel: basis_cta with the scan and the pass of the smoothed V."""
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
        lambda lo, hi, slot, tid, sb: _vscan(v_p, vmax_p, S, H, T, lo, hi, slot, tid, sb, lens_p),
        lambda ntiles, pid, tid, sb: _vpass(
            v_p, vmax_p, v8_p, vmean_p, v8s_p, done_p, S, H, T, ntiles, pid, nspare, tid, sb, lens_p
        ),
        lens_p,
        rows_mode,
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
    """Run the V pass as its own launch, one 128-key tile of one (batch, head) per iteration; lens_p as in _vpass.
    Thread = (16-key group, 8-channel chunk), the groups of the tile in lanes 8 j .. 8 j + 7: 16 x 16 bytes
    per thread."""
    tid, _, _ = cute.arch.thread_idx()
    cta, _, _ = cute.arch.block_idx()
    smem = SmemAllocator()
    buf = smem.allocate_array(Uint32, 64, byte_alignment=128)
    red = Int32(buf.toint())
    v8s = pow2_ceil_448(scan_max(vmax_p, nspare, tid, red, None, V_THREADS, False))
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
        cnt_s = key_count(k0, S)
        nk = valid_len(lens_p, bb, S)
        cnt = cnt_s if cutlass.const_expr(lens_p is None) else key_count(k0, nk)
        kb = cutlass.min(k0, S - 1)
        base = row_addr(v_p, bb, kb, hh, S, H, 256) + Int64(cc * 16)
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
            v8b = row_addr(v8_p, bb, kb, hh, S, H, 128) + Int64(cc * 8)
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
        # the threads of the CTA start the next item together
        cute.arch.sync_threads()
