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
"""Encode launch of the sm_120 preparation: the Q/K group scales from the group maxima of the Q'/K' pass, the
score-scale floor (2^msl, by growing the Q group scales), score_scale, and the UE4M3 Q'/K' block scales from the
provisional BF16 scales."""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32
from cutlass.utils import SmemAllocator

from ...common.expmath import e4m3x4, f2u, pow2_ceil_448, u2f, warp_max
from ...common.mem import gptr, load_vector, sptr, store_vector

ENCODE_THREADS = 256


@cute.kernel
def encode_kernel(
    prov_p: Int64,
    gmax_p: Int64,
    qsf_p: Int64,
    ksf_p: Int64,
    score_scale_p: Int64,
    B: Int32,
    H: Int32,
    T: Int32,
    nqk_cta: Int32,
    msl: Int32,
    ncta: Int32,
):
    tid, _, _ = cute.arch.thread_idx()
    cta, _, _ = cute.arch.block_idx()
    lane = tid & 31
    warp = tid >> 5
    smem = SmemAllocator()
    buf = smem.allocate_array(Uint32, 64, byte_alignment=16)
    red = Int32(buf.toint())
    # group maxima over the Q/K pass CTAs: 6 values
    gmv = cute.make_rmem_tensor((6,), Float32)
    gmv.fill(0.0)
    for i in cutlass.range(tid, nqk_cta, ENCODE_THREADS, unroll=1):
        for j in cutlass.range_constexpr(6):
            gmv[j] = cute.arch.fmax(
                gmv[j], u2f(cute.arch.load(gptr(Uint32, gmax_p + (Int64(i) * 8 + j) * 4), Uint32))
            )
    for j in cutlass.range_constexpr(6):
        mv = warp_max(gmv[j])
        if lane == 0:
            cute.arch.store(sptr(Float32, red + (warp * 6 + j) * 4), mv)
    cute.arch.sync_threads()
    for j in cutlass.range_constexpr(6):
        mv = Float32(0.0)
        for wi in cutlass.range_constexpr(ENCODE_THREADS // 32):
            mv = cute.arch.fmax(mv, cute.arch.load(sptr(Float32, red + (wi * 6 + j) * 4), Float32))
        gmv[j] = mv
    gq0 = pow2_ceil_448(gmv[0])
    gq1 = pow2_ceil_448(gmv[1])
    gq2 = pow2_ceil_448(gmv[2])
    gk0 = pow2_ceil_448(gmv[3])
    gk1 = pow2_ceil_448(gmv[4])
    gk2 = pow2_ceil_448(gmv[5])
    prod = cute.arch.fmax(cute.arch.fmax(gq0 * gk0, gq1 * gk1), gq2 * gk2)
    minp = u2f(Uint32((127 + 7 + msl) << 23))
    ep = Int32(f2u(prod) >> Uint32(23))
    em = Int32(f2u(minp) >> Uint32(23))
    raise_ = Int32(prod < minp)
    e = em - ep
    up = u2f(Uint32((127 + ((e + 1) >> 1)) << 23))
    upf = Float32(cutlass.select_(raise_ != 0, up, Float32(1.0)))
    gq0 = gq0 * upf
    gq1 = gq1 * upf
    gq2 = gq2 * upf
    prod = Float32(cutlass.select_(raise_ != 0, minp, prod))
    if cta == 0 and tid == 0:
        score_scale = prod * Float32(1.0 / 128.0)
        cute.arch.store(gptr(Float32, score_scale_p), score_scale)
    iq0 = cute.math.rcp(gq0)
    iq1 = cute.math.rcp(gq1)
    iq2 = cute.math.rcp(gq2)
    ik0 = gq0 * cute.math.rcp(prod)
    ik1 = gq1 * cute.math.rcp(prod)
    ik2 = gq2 * cute.math.rcp(prod)
    # provisional BF16 scales -> UE4M3 bytes, 8 per thread-iteration
    per_side = B * H * T * 192  # 8-element chunks per side
    nch = 2 * per_side
    for i in cutlass.range(cta * ENCODE_THREADS + tid, nch, ncta * ENCODE_THREADS, unroll=1):
        side = Int32(i >= per_side)
        li = i - side * per_side
        atom = (li % 192) >> 6
        w = load_vector(gptr(Uint32, prov_p + Int64(i) * 16, 16), 4)
        f = [Float32(0.0)] * 8
        for e2 in cutlass.range_constexpr(4):
            f[2 * e2] = u2f(w[e2] << Uint32(16))
            f[2 * e2 + 1] = u2f(w[e2] & Uint32(0xFFFF0000))
        d0 = Float32(cutlass.select_(side != 0, ik0, iq0))
        d1 = Float32(cutlass.select_(side != 0, ik1, iq1))
        d2 = Float32(cutlass.select_(side != 0, ik2, iq2))
        db = Float32(cutlass.select_(atom == 2, d1, d0))
        dc = Float32(cutlass.select_(atom == 2, d2, d0))
        lo = e4m3x4(f[0] * db, f[1] * db, f[2] * dc, f[3] * dc)
        hi = e4m3x4(f[4] * db, f[5] * db, f[6] * dc, f[7] * dc)
        if side == 0:
            store_vector(gptr(Uint32, qsf_p + Int64(li) * 8, 8), (lo, hi))
        else:
            store_vector(gptr(Uint32, ksf_p + Int64(li) * 8, 8), (lo, hi))
