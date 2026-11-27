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
"""V pass of the sm_120 NVFP4-PV kernel: V' = V minus its BF16 means of every 16 keys, in NVFP4 along the keys.

* V' is in V's own 128 channels: E2M1 codes with one UE4M3 scale per channel and 16 keys, under the power-of-two
  tensor scale v4_scale of max|v - vbar| / 2688; v_mean = vbar / v4_scale. All three go to the tile images (layout.py).
* The pass shares the basis launch (prep/basis.py): the launch's spare CTAs scan V for max|v - vbar| (the tensor
  scale is needed before any code is formed) and, when the launch has room for it, run the pass on the tiles a
  counter hands out (_vpass); v_kernel is the pass as its own launch otherwise.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint8, Uint16, Uint32
from cutlass.cute.typing import Float8E4M3FN
from cutlass.utils import SmemAllocator

from ...common.dsl import FPRoundingMode, SaturationMode, cvt_f32x2_to_f8x2
from ...common.expmath import RN, bf16x2, pow2_ceil, u2f
from ...common.mem import gptr, load_vector, sptr, store_vector
from ...common.ptx import f32x8_to_e2m1x8
from ...constants import E2M1_MAX, E4M3_MAX, V_MEAN_HEADROOM
from ..prep.basis import (
    EIG_THREADS,
    SM_VSTG,
    basis_cta,
    bf16_f32,
    group16,
    key_count,
    scan_max,
    scan_max_word,
    store_slot,
    tok_addr,
    vpass_tiles,
    vscan_tiles,
)
from ..ptx.nvfp4pv import e4m3_to_f32
from ..ptx.shared import ld_global_nc_b32_hint_pred
from ..valid_len import valid_len
from .layout import STAGE, SV_OFF, key_tiles, mean_off, store_means, v_scale_off

V_THREADS = 128


def tensor_scale(c):
    """Return the power-of-two v4_scale for max|v - vbar| = c: at or above c / (6 x 448), the E2M1 range times the
    UE4M3 range."""
    return pow2_ceil(cute.math.div(c, Float32(E2M1_MAX * E4M3_MAX), rounding=RN))


def group_mean(sums, cnt):
    """vbar of the thread's channels, BF16 values as f32 (sums: their f32 sums over the cnt valid keys of its 16-key
    group): the sums divided by the key count, as the scan that sets v4_scale divides them."""
    fc = Float32(cutlass.max(cnt, 1))
    mbw = [
        bf16x2(
            cute.math.div(sums[2 * e], fc, rounding=RN),
            cute.math.div(sums[2 * e + 1], fc, rounding=RN),
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
    sf = cute.math.div(am * i4, Float32(E2M1_MAX), rounding=RN)
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


def stage_codes(stg, sfs, wv, mf, cnt, i4, chunk, grp, NC):
    """Stage V' of the thread's NC channels (chunk `chunk`) of a 16-key group, by channel_codes: the codes at stg,
    [channel in chunk][chunk][64 B], the block scales at sfs, [channel][8 B]. wv: its BF16 words, NC / 2 per key."""
    for c in range(NC):
        sfw, cw0, cw1 = channel_codes(
            lambda kk: bf16_f32(wv[kk * (NC // 2) + c // 2], c % 2), mf[c], cnt, i4
        )
        store_vector(sptr(Uint32, stg + (c * (128 // NC) + chunk) * 64 + grp * 8, 8), (cw0, cw1))
        cute.arch.store(sptr(Uint8, sfs + (chunk * NC + c) * 8 + grp), sfw.to(Uint8))


def store_codes(stg, img, thread, NT, NC):
    """Store the staged codes of a tile into its image img, 128 channel rows per 16-byte part: 512 / NT per thread."""
    for m in range(512 // NT):
        ci = thread + NT * m
        ch = ci >> 2
        part = ci & 3
        row = (ch % NC) * (128 // NC) + ch // NC
        w4 = load_vector(sptr(Uint32, stg + row * 64 + part * 16, 16), 4)
        store_vector(gptr(Uint32, img + (SV_OFF + part * 2048 + ch * 16), 16), w4, cop="cs")


def _fold(sums, vmx, vmn, fc, any_, mx):
    """mx raised to max|v - vbar| over the 8 channels, vbar = BF16(sums / fc); any_: the lane has keys (None: all)."""
    for c in range(8):
        mb = u2f(bf16x2(cute.math.div(sums[c], fc, rounding=RN), Float32(0.0)) << Uint32(16))
        # the deviation, or |mean| times the headroom that keeps the scaled mean finite
        d = cute.arch.fmax(
            cute.arch.fmax(vmx[c] - mb, mb - vmn[c]),
            cute.arch.fmax(mb, -mb) * Float32(V_MEAN_HEADROOM),
        )
        mx = cute.arch.fmax(
            mx, d if any_ is None else Float32(cutlass.select_(any_ != 0, d, Float32(0.0)))
        )
    return mx


@cute.jit
def _vscan_gs(
    v_p: Int64,
    vmax_p: Int64,
    B: Int32,
    S: Int32,
    H: Int32,
    VP: Int32,
    cta: Int32,
    ncta: Int32,
    tid: Int32,
    sb: Int32,
    lens_p,
):
    """max|v - vbar| over all of V, grid-stride: item = (batch, 16-key group, head, 8-channel chunk). Writes slot
    cta. lens_p: as in _vscan."""
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
        nk = valid_len(lens_p, bb, S)
        cnt = cutlass.min(nk - k0, 16)
        base = tok_addr(v_p, bb, k0, hh, S, VP) + Int64(cc * 16)
        # with lengths: a 16-key group without a valid key is skipped
        live = cutlass.const_expr(lens_p is None) or k0 < nk
        if live:
            sums, vmx, vmn = group16(base, Int64(VP), cnt)
            mx = _fold(sums, vmx, vmn, Float32(cnt), None, mx)
    store_slot(mx, tid, sb, vmax_p, cta)


@cute.jit
def _encode_tile(img_p, i4, stg, sfs, half, wv, sv, tile, tt, cnt, nk, tid_h):
    """Encode one tile of the V pass in the basis launch (vpass_tiles) for the thread (group tid_h % 8, quad tid_h / 8):
    the means of its 16-key group, then V' codes and scales transposed through smem ([c&3][quad][64 B] / [c][8 B])
    into the image of the tile. The keys from cnt on, those past the valid keys nk of the batch element, get zero
    codes."""
    grp = tid_h & 7
    cq = tid_h >> 3
    img = img_p + Int64(tile) * STAGE
    # the mean of the 16-key group; v_mean = vbar / v4_scale (exact)
    mf = group_mean([sv[e] for e in range(4)], cnt)
    mean_words = [bf16x2(mf[2 * e] * i4, mf[2 * e + 1] * i4) for e in range(2)]
    mean_at = mean_off(grp, cq >> 1) + (cq & 1) * 8
    store_means(img, tt, key_tiles(nk) - 1, mean_at, mean_words)
    stage_codes(stg, sfs, wv, mf, cnt, i4, cq, grp, NC=4)
    cute.arch.barrier(barrier_id=1 + half, number_of_threads=256)
    store_codes(stg, img, tid_h, NT=256, NC=4)
    if tid_h < 128:
        s2 = load_vector(sptr(Uint32, sfs + tid_h * 8, 8), 2)
        store_vector(gptr(Uint32, img + v_scale_off(tid_h), 8), s2, cop="cs")


@cute.jit
def _vpass(
    v_p: Int64,
    img_p: Int64,
    v4s_p: Int64,
    done_p: Int64,
    S: Int32,
    H: Int32,
    VP: Int32,
    T: Int32,
    ntiles: Int32,
    pid: Int32,
    tid: Int32,
    sb: Int32,
    lens_p,
):
    """The V pass in the basis launch: the tensor scale from the maximum of this launch's scan CTAs, then the tiles
    of vpass_tiles with _encode_tile. A tile without a valid key is left as it is."""
    v4s = tensor_scale(scan_max_word(done_p))
    if pid == 0 and tid == 0:
        cute.arch.store(gptr(Float32, v4s_p), v4s)
    i4 = cute.math.rcp(v4s)
    half = tid >> 8
    stg = sb + SM_VSTG + half * 9216
    sfs = stg + 8192
    vpass_tiles(
        v_p,
        done_p,
        S,
        H,
        VP,
        T,
        ntiles,
        tid,
        sb,
        lens_p,
        lambda wv, sv, tile, bh, tt, k0, cnt, nk, tid_h: _encode_tile(
            img_p, i4, stg, sfs, half, wv, sv, tile, tt, cnt, nk, tid_h
        ),
    )


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
    img_p: Int64,
    v4s_p: Int64,
    B: Int32,
    S: Int32,
    H: Int32,
    QP: Int32,
    KP: Int32,
    VP: Int32,
    T: Int32,
    NQK: Int32,
    BH: Int32,
    nspare: Int32,
    FUSED: cutlass.Constexpr,
    lens_p,
    rows_mode,
):
    """The basis launch of the NVFP4-PV kernel: basis_cta with the scan and the pass of V'. lens_p, rows_mode: None,
    or the global address of the valid prefix lengths and the rows mode of the eigensolver."""
    basis_cta(
        q_p,
        k_p,
        rows_p,
        had_p,
        ut_p,
        done_p,
        S,
        H,
        QP,
        KP,
        T,
        NQK,
        BH,
        nspare,
        FUSED,
        lambda cta, tid, sb: _vscan_gs(v_p, vmax_p, B, S, H, VP, cta, nspare, tid, sb, lens_p),
        # thread = (16-key group, 8-channel chunk); the mean of a lane's group is over its own keys
        lambda ntiles, tid, sb: vscan_tiles(
            v_p,
            done_p,
            S,
            H,
            VP,
            T,
            ntiles,
            tid,
            sb,
            lens_p,
            lambda u: (u >> 4, u & 15),
            lambda sums, vmx, vmn, cnt, mx: _fold(
                sums, vmx, vmn, Float32(cutlass.max(cnt, 1)), Int32(cnt > 0), mx
            ),
        ),
        lambda ntiles, pid, tid, sb: _vpass(
            v_p, img_p, v4s_p, done_p, S, H, VP, T, ntiles, pid, tid, sb, lens_p
        ),
        lens_p,
        rows_mode,
    )


@cute.kernel
def v_kernel(
    v_p: Int64,
    vmax_p: Int64,
    img_p: Int64,
    v4s_p: Int64,
    S: Int32,
    H: Int32,
    VP: Int32,
    T: Int32,
    nspare: Int32,
    nitems: Int32,
    ncta: Int32,
    lens_p,
):
    """One 128-key tile of one (batch, head) per iteration: thread (group of 16 keys, 8-channel chunk) loads
    16 x 16 bytes; V' codes and scales are transposed through smem ([e][chunk][64 B] / [channel][8 B]). lens_p: as in
    _vpass."""
    tid, _, _ = cute.arch.thread_idx()
    cta, _, _ = cute.arch.block_idx()
    smem = SmemAllocator()
    buf = smem.allocate_array(Uint32, (64 + 2048 + 256), byte_alignment=128)
    red = Int32(buf.toint())
    stg = red + 256
    sfs = stg + 8192
    v4s = tensor_scale(scan_max(vmax_p, nspare, tid, red, None, V_THREADS, False))
    if cta == 0 and tid == 0:
        cute.arch.store(gptr(Float32, v4s_p), v4s)
    i4 = cute.math.rcp(v4s)
    lane = tid & 31
    warp = tid >> 5
    grp = lane & 7
    cc = (lane >> 3) + 4 * warp
    rstride = Int64(VP)
    item_lo = (cta * nitems) // ncta
    item_hi = ((cta + 1) * nitems) // ncta
    for item in cutlass.range(item_lo, item_hi, unroll=1):
        bh = item // T
        tile = item - bh * T
        bb = bh // H
        hh = bh - bb * H
        k0 = tile * 128 + 16 * grp
        nk = valid_len(lens_p, bb, S)
        cnt = key_count(k0, nk)
        base = tok_addr(v_p, bb, cutlass.min(k0, S - 1), hh, S, VP) + Int64(cc * 16)
        img = img_p + Int64(item) * STAGE
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
                    sums[2 * e] = sums[2 * e] + bf16_f32(w[e], 0)
                    sums[2 * e + 1] = sums[2 * e + 1] + bf16_f32(w[e], 1)
            mf = group_mean(sums, cnt)
            # v_mean = vbar / v4_scale (exact), 8 channels = 16 bytes
            mean_words = [bf16x2(mf[2 * e] * i4, mf[2 * e + 1] * i4) for e in range(4)]
            store_means(img, tile, key_tiles(nk) - 1, mean_off(grp, cc), mean_words)
            # per channel: amax of y = (v - vbar) / v4_scale over the valid keys, UE4M3 block scale, codes
            stage_codes(stg, sfs, wv, mf, cnt, i4, cc, grp, NC=8)
        cute.arch.sync_threads()
        if live:
            store_codes(stg, img, tid, NT=V_THREADS, NC=8)
            s2 = load_vector(sptr(Uint32, sfs + tid * 8, 8), 2)
            store_vector(gptr(Uint32, img + v_scale_off(tid), 8), s2, cop="cs")
        cute.arch.sync_threads()
