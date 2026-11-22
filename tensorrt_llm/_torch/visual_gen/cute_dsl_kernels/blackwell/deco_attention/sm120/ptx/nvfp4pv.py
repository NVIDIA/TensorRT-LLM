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
"""Inline PTX that the NVFP4-PV kernel of the sm_120 family alone issues (sm120/nvfp4pv/): one of the three modules
of the package, see its docstring. The sections:

  barriers and waits                  the turn barrier of a warp pair
  conversions and packed arithmetic   the E4M3 decoding, and the blocks of the softmax (nvfp4pv/softmax.py): the
                                      anchor update, the rescale of O, the block scales, the landing with the
                                      sums of P
  memory access                       predicated global loads, the atomic of the release count, TMA bulk copies"""

import math

from cutlass.cute.typing import Float32, Int32
from cutlass.cutlass_dsl import dsl_user_op

from ...common.ptx import asm, f32, i32, i64, tied, u32
from ...constants import E4M3_MAX, RESCALE_THRESHOLD
from .shared import f32_literal


# -------------------------------------------------------------------------------- Barriers and waits
@dsl_user_op
def barrier_turn(bid, pred, op: str, *, loc=None, ip=None):
    """i32 barrier id bid, i32 pred: bar.<op> bid, 64 when pred != 0 (pred warp-uniform; the 64 threads are the two
    warps that take turns). op: "sync", the barrier, or "arrive", its arrival without the wait."""
    asm(
        "{\n.reg .pred p;\nsetp.ne.u32 p, $1, 0;\n@p bar." + op + " $0, 64;\n}",
        [i32(bid), i32(pred)],
        side_effects=True,
        loc=loc,
        ip=ip,
    )


# -------------------------------------------------------------------------------- Conversions and packed arithmetic
@dsl_user_op
def e4m3_to_f32(byte, *, loc=None, ip=None):
    """u32 byte -> f32 value of the E4M3 code in its low byte (cvt.rn.f16x2.e4m3x2 of the low 16 bits, cvt.f32.f16 of
    the low half); no side effects."""
    return asm(
        "{ .reg .b16 s; .reg .b32 h; .reg .f16 l, u;\n cvt.u16.u32 s, $1;\n"
        " cvt.rn.f16x2.e4m3x2 h, s;\n mov.b32 {l, u}, h;\n cvt.f32.f16 $0, l; }",
        [u32(byte)],
        Float32,
        side_effects=False,
        loc=loc,
        ip=ip,
    )


# the blocks of the softmax (nvfp4pv/softmax.py), down to land32_sum
@dsl_user_op
def need_vote(bm, thr0, thr1, *, loc=None, ip=None):
    """Eight f32 bm (the block maxima), f32 thr0, thr1 -> i32 warp-uniform 1 if on any lane a bm[2 i] exceeds thr0 or a
    bm[2 i + 1] exceeds thr1 (setp.gt.f32 with .or, vote.sync.any.pred), else 0: whether an anchor may have to move.
    Each raw threshold is div.rm(RN(anchor + T), c), T = RESCALE_THRESHOLD, so a block max at or below it cannot meet
    the move condition of anchor_update, RN(c max) > RN(anchor + T)."""
    bmi, thr0i, thr1i = 1, 9, 10
    body = ["{", ".reg .pred p;", f"setp.gt.f32 p, ${bmi}, ${thr0i};"]
    body += [f"setp.gt.or.f32 p, ${bmi + 2 * i}, ${thr0i}, p;" for i in range(1, 4)]
    body += [f"setp.gt.or.f32 p, ${bmi + 2 * i + 1}, ${thr1i}, p;" for i in range(4)]
    body += ["vote.sync.any.pred p, p, 0xffffffff;", "selp.u32 $0, 1, 0, p;", "}"]
    return asm(
        "\n".join(body),
        [f32(x) for x in bm] + [f32(thr0), f32(thr1)],
        Int32,
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def anchor_update(need, bm, a0, a1, th0, th1, ls0, ls1, safe, c, *, loc=None, ip=None):
    """i32 need, eight f32 bm (the block maxima), f32 a0, a1 (the anchors), th0, th1 (their thresholds), ls0, ls1 (the
    row sums), i32 safe (the last seven tied to their results), f32 c -> eight f32 (the anchors, the two correction
    factors of O, the thresholds, the sums) and the i32 fold-safety flag. When need != 0 (rare): the full row maxima,
    the skip-correction update of the anchors (a move when RN(c max) > RN(anchor + T), T = RESCALE_THRESHOLD) with
    their thresholds, the sums times the correction factors, and the flag; when need is 0: the tied values and factors
    of 1.0. The update runs behind a warp-uniform branch, written as a loop that runs at most once."""
    # $0 .. $8: the results: the anchors (ai), the factors (fi), the thresholds (thresh_i), the sums (lsi) and the flag;
    # then, behind the seven tied operands, need, bm and c
    ai, fi, thresh_i, lsi, safei, needi, bmi, ci = 0, 2, 4, 6, 8, 16, 17, 25
    # the skip threshold T, as a PTX literal
    thr = f32_literal(RESCALE_THRESHOLD)
    body = [
        "{",
        ".reg .pred p, cpow2;",
        ".reg .b32 n, cu, iu, mant;",
        ".reg .f32 m0, m1, x, a, ic;",
        f"mov.f32 ${fi}, 0f3F800000;",
        f"mov.f32 ${fi + 1}, 0f3F800000;",
        f"mov.b32 n, ${needi};",
        "L_${:uid}:",
        "setp.eq.u32 p, n, 0;",
        "@p bra.uni D_${:uid};",
        f"max.f32 m0, ${bmi}, ${bmi + 2};",
        f"max.f32 x, ${bmi + 4}, ${bmi + 6};",
        "max.f32 m0, m0, x;",
        f"max.f32 m1, ${bmi + 1}, ${bmi + 3};",
        f"max.f32 x, ${bmi + 5}, ${bmi + 7};",
        "max.f32 m1, m1, x;",
        "shfl.sync.bfly.b32 x, m0, 2, 0x1f, 0xffffffff;",
        "max.f32 m0, m0, x;",
        "shfl.sync.bfly.b32 x, m1, 2, 0x1f, 0xffffffff;",
        "max.f32 m1, m1, x;",
    ]
    # cpow2: c is a power of two whose reciprocal is a normal f32, ic
    body += [
        f"mov.b32 cu, ${ci};",
        "and.b32 mant, cu, 0x007fffff;",
        "setp.eq.u32 cpow2, mant, 0;",
        "setp.ge.and.u32 cpow2, cu, 0x00800000, cpow2;",
        "setp.le.and.u32 cpow2, cu, 0x7e800000, cpow2;",
        "sub.u32 iu, 0x7f000000, cu;",
        "mov.b32 ic, iu;",
    ]
    # The thresholds are rounded down, so they are conservative for every positive finite c: a raw max at or below its
    # threshold cannot exceed RN(anchor + T) once scaled and rounded, and a false positive rechecks that comparison
    # here. A sum and the correction factor of its rows change together; an anchor that stays has the factor one.
    for r in range(2):
        aa, ff, tt, ss = (f"${i + r}" for i in (ai, fi, thresh_i, lsi))
        body += [
            f"mul.rn.f32 x, m{r}, ${ci};",
            f"add.rn.f32 a, {aa}, {thr};",
            "setp.gt.f32 p, x, a;",
            f"selp.f32 a, x, {aa}, p;",
            f"sub.rn.f32 x, {aa}, a;",
            f"ex2.approx.ftz.f32 {ff}, x;",
            f"mov.f32 {aa}, a;",
            f"mul.rn.f32 {ss}, {ss}, {ff};",
            f"add.rn.f32 x, a, {thr};",
            f"@cpow2 mul.rm.f32 {tt}, x, ic;",
            f"@!cpow2 div.rm.f32 {tt}, x, ${ci};",
        ]
    # The fold guard, kept until the next update: every anchor of the warp within [-7900, 7900] and c <= 1/16. Until a
    # move is requested the tile maxima are at most anchor + T; the blocks with nonzero F16 scales are within 30
    # binades below the anchor, so their folded offsets stay on the fine subnormal grid; the bound on c also keeps the
    # landing of the other blocks finite.
    body += [
        f"abs.f32 x, ${ai};",
        f"abs.f32 a, ${ai + 1};",
        "max.f32 x, x, a;",
        "setp.le.f32 p, x, 0f45F6E000;",
        f"setp.le.and.f32 p, ${ci}, 0f3D800000, p;",
        "vote.sync.all.pred p, p, 0xffffffff;",
        f"selp.u32 ${safei}, 1, 0, p;",
        "mov.b32 n, 0;",
        "bra.uni L_${:uid};",
        "D_${:uid}:",
        "}",
    ]
    return asm(
        "\n".join(body),
        [
            tied(k, f32(x))
            for k, x in zip(
                (ai, ai + 1, thresh_i, thresh_i + 1, lsi, lsi + 1), (a0, a1, th0, th1, ls0, ls1)
            )
        ]
        + [tied(safei, i32(safe)), i32(need)]
        + [f32(x) for x in bm]
        + [f32(c)],
        [Float32] * 8 + [Int32],
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def rescale64_loop(o, f0, f1, need, *, loc=None, ip=None):
    """64 f32 o (the O words of the 16 channel tiles; tied to the results), f32 f0, f1, i32 need -> 64 f32: when need !=
    0, o[i] times f0 for i % 4 < 2 and times f1 otherwise (mul.f32); o unchanged when need is 0. Written as a loop that
    runs at most once behind warp-uniform branches: with need 0 the 64 multiplies are skipped."""
    # $0 .. $63: the O words; then, behind the 64 tied operands, f0, f1 and need
    f0i, f1i, needi = 128, 129, 130
    body = [
        "{",
        ".reg .pred p;",
        ".reg .b32 n;",
        f"mov.b32 n, ${needi};",
        "L_${:uid}:",
        "setp.eq.u32 p, n, 0;",
        "@p bra.uni D_${:uid};",
    ]
    for i in range(64):
        body.append(f"mul.f32 ${i}, ${i}, ${f0i if i % 4 < 2 else f1i};")
    body += ["mov.b32 n, 0;", "bra.uni L_${:uid};", "D_${:uid}:", "}"]
    return asm(
        "\n".join(body),
        [tied(i, f32(x)) for i, x in enumerate(o)] + [f32(f0), f32(f1), i32(need)],
        [Float32] * 64,
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def block_scales2(bm0, bm1, c, anchor0, anchor1, *, loc=None, ip=None):
    """f32 bm0, bm1 (the maxima of two 16-key blocks, of rows g and g + 8), f32 c, anchor0, anchor1 -> i32 word of two
    f16, i32 of two UE4M3 bytes, and two f32; no side effects. The P scales of the two blocks, 448 * 2^(c bm - anchor -
    T) (fma.rn.f32x2, ex2.approx.ftz.f32; T = RESCALE_THRESHOLD), rounded to UE4M3 and read back
    (cvt.rn.satfinite.e4m3x2.f16x2, cvt.rn.f16x2.e4m3x2): the decoded scales as f16x2, floored at 2^-10 so that a block
    whose scale rounds to zero keeps finite elements (the sum factor); the UE4M3 bytes; and per block the log2 of the
    unrounded scale over the decoded one (lg2.approx.ftz.f32), the shift of its element offset."""
    # the results: the decoded scales, the UE4M3 bytes and the two shifts; then the operands: the maxima, c, the anchors
    schi, sfi, shifti, bmi, ci, anchori = 0, 1, 2, 4, 6, 7
    # the headroom of the P block scales, UE4M3(448 * 2^(bm - anchor - T)), as a PTX literal
    k = f32_literal(math.log2(E4M3_MAX) - RESCALE_THRESHOLD)
    body = [
        "{",
        ".reg .b64 bb, cc, ll, tt;",
        ".reg .f32 t0, t1, u0, u1;",
        ".reg .b16 e, d0, d1;",
        ".reg .b32 ah, dh, fl;",
        f"mov.b64 bb, {{${bmi}, ${bmi + 1}}};",
        f"mov.b64 cc, {{${ci}, ${ci}}};",
        f"neg.f32 t0, ${anchori};",
        f"neg.f32 t1, ${anchori + 1};",
        "mov.b64 ll, {t0, t1};",
        "fma.rn.f32x2 tt, bb, cc, ll;",
        f"mov.b64 ll, {{{k}, {k}}};",
        "add.rn.f32x2 tt, tt, ll;",
        "mov.b64 {t0, t1}, tt;",
        "ex2.approx.ftz.f32 u0, t0;",
        "ex2.approx.ftz.f32 u1, t1;",
        "cvt.rn.f16x2.f32 ah, u1, u0;",
        "cvt.rn.satfinite.e4m3x2.f16x2 e, ah;",
        f"cvt.u32.u16 ${sfi}, e;",
        "cvt.rn.f16x2.e4m3x2 dh, e;",
        "mov.b32 fl, 0x14001400;",
        "max.f16x2 dh, dh, fl;",
        f"mov.b32 ${schi}, dh;",
        "mov.b32 {d0, d1}, dh;",
        "cvt.f32.f16 u0, d0;",
        "cvt.f32.f16 u1, d1;",
        "lg2.approx.ftz.f32 u0, u0;",
        "lg2.approx.ftz.f32 u1, u1;",
        f"sub.rn.f32 ${shifti}, t0, u0;",
        f"sub.rn.f32 ${shifti + 1}, t1, u1;",
        "}",
    ]
    return asm(
        "\n".join(body),
        [f32(x) for x in (bm0, bm1, c, anchor0, anchor1)],
        [Int32, Int32, Float32, Float32],
        side_effects=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def prepare_land64_flag(scores, maxima, m, safe, *, loc=None, ip=None):
    """64 f32 scores and 8 f32 maxima (the block maxima of the eight groups of eight scores; both tied to the results),
    f32 m, i32 safe (warp-uniform) -> the 64 scores and the 8 block offsets of the folded landing. With safe != 0 the
    scores stay and the offset of a block is (17.5 - c bm) 2^-139 = K0 - bm m (fma.rn.f32x2 of the maximum, -m and K0 =
    0f00004600). With safe 0 every score has the maximum of its group subtracted first (sub.rn.f32x2, behind a
    warp-uniform branch) and every offset is K0."""
    # $0 .. $71: the scores and, from maxi, the maxima; then, behind the 72 tied operands, m and safe
    maxi, mi, safei = 64, 144, 145
    body = [
        "{",
        ".reg .pred p;",
        ".reg .b32 n;",
        ".reg .f32 nm;",
        ".reg .b64 s2, b2, m2, k2;",
        f"setp.eq.u32 p, ${safei}, 0;",
        "selp.u32 n, 1, 0, p;",
        "L_${:uid}:",
        "setp.eq.u32 p, n, 0;",
        "@p bra.uni D_${:uid};",
    ]
    for g in range(8):
        body.append(f"mov.b64 b2, {{${maxi + g}, ${maxi + g}}};")
        for i in range(4):
            j = 8 * g + 2 * i
            body += [
                f"mov.b64 s2, {{${j}, ${j + 1}}};",
                "sub.rn.f32x2 s2, s2, b2;",
                f"mov.b64 {{${j}, ${j + 1}}}, s2;",
            ]
        body.append(f"mov.f32 ${maxi + g}, 0f00000000;")
    body += [
        "mov.u32 n, 0;",
        "bra.uni L_${:uid};",
        "D_${:uid}:",
        f"neg.f32 nm, ${mi};",
        "mov.b64 m2, {nm, nm};",
        "mov.b64 k2, {0f00004600, 0f00004600};",
    ]
    for g in range(4):
        j = maxi + 2 * g
        body += [
            f"mov.b64 b2, {{${j}, ${j + 1}}};",
            "fma.rn.f32x2 b2, b2, m2, k2;",
            f"mov.b64 {{${j}, ${j + 1}}}, b2;",
        ]
    body.append("}")
    return asm(
        "\n".join(body),
        [tied(i, f32(x)) for i, x in enumerate([*scores, *maxima])] + [f32(m), i32(safe)],
        [Float32] * 72,
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def land32_sum(scores, offsets, m, sch, par, kofs=None, *, loc=None, ip=None):
    """32 f32 scores (the four groups of eight of one h-block: (s0, row 0), (s0, row 1), (s1, row 0), (s1, row 1)),
    four f32 offsets, f32 m, i32 sch (the two decoded block scales as f16), i32 par, optionally four f32 kofs -> four
    i32 words of eight E2M1 codes (nibble i = score i) and the two f32 16-key sums this lane owns. Each score lands as
    s m + offset on the f32 subnormal grid (fma.rn.f32x2), whose bits are the f16 pattern of its element; the
    saturating pack (cvt.pack.sat.u16.s32) flushes everything below 2^-15, and -inf, to +0, and
    cvt.rn.satfinite.e2m1x2.f16x2 codes the elements. The f16x2 sums of the landed elements of a row meet those of the
    XOR-1 lane (shfl.sync.bfly.b32; each lane keeps the half of its parity par) and enter f32 times the decoded block
    scale of sch (fma.rn.f32.f16). With kofs (the two-step landing) the offset is subtracted first, the FMA adds the
    constant kofs of the group (17.5 plus the shift of its decoded scale) and a second multiply scales the result onto
    the grid."""
    two_step = kofs is not None
    # $0 .. $3: the P words; then the results sumi (the sums); then the operands: the scores (si), m, the offsets, sch,
    # par and kofs
    sumi, si, mi, offi, schi, pari, kofsi = 4, 6, 38, 39, 43, 44, 45
    body = [
        "{",
        ".reg .pred odd;",
        ".reg .b64 m2, o2, x0, x1, x2, x3;",
        ".reg .b32 y0, y1, y2, y3, y4, y5, y6, y7;",
        ".reg .b32 h0, h1, h2, h3, t0, t1, sm0, sm1, keep, send;",
        ".reg .b8 q0, q1, q2, q3;",
        ".reg .b16 v0, v1, scale0, scale1;",
        ".reg .f32 sum0;",
        f"setp.ne.u32 odd, ${pari}, 0;",
        f"mov.b32 {{scale0, scale1}}, ${schi};",
    ]
    for rr in range(2):
        for s_ in range(2):
            g = 2 * s_ + rr
            first = si + 8 * g
            body += [
                f"mov.b64 m2, {{${mi}, ${mi}}};",
                f"mov.b64 o2, {{${offi + g}, ${offi + g}}};",
                f"mov.b64 x0, {{${first}, ${first + 1}}};",
                f"mov.b64 x1, {{${first + 2}, ${first + 3}}};",
                f"mov.b64 x2, {{${first + 4}, ${first + 5}}};",
                f"mov.b64 x3, {{${first + 6}, ${first + 7}}};",
            ]
            if two_step:
                body += [
                    "sub.rn.f32x2 x0, x0, o2;",
                    "sub.rn.f32x2 x1, x1, o2;",
                    "sub.rn.f32x2 x2, x2, o2;",
                    "sub.rn.f32x2 x3, x3, o2;",
                    f"mov.b64 o2, {{${kofsi + g}, ${kofsi + g}}};",
                ]
            body += [
                "fma.rn.f32x2 x0, x0, m2, o2;",
                "fma.rn.f32x2 x1, x1, m2, o2;",
                "fma.rn.f32x2 x2, x2, m2, o2;",
                "fma.rn.f32x2 x3, x3, m2, o2;",
            ]
            if two_step:
                body += [
                    "mov.b64 m2, {0f00000400, 0f00000400};",
                    "mul.rn.f32x2 x0, x0, m2;",
                    "mul.rn.f32x2 x1, x1, m2;",
                    "mul.rn.f32x2 x2, x2, m2;",
                    "mul.rn.f32x2 x3, x3, m2;",
                ]
            body += [
                "mov.b64 {y0, y1}, x0;",
                "mov.b64 {y2, y3}, x1;",
                "mov.b64 {y4, y5}, x2;",
                "mov.b64 {y6, y7}, x3;",
                "cvt.pack.sat.u16.s32 h0, y1, y0;",
                "cvt.pack.sat.u16.s32 h1, y3, y2;",
                "cvt.pack.sat.u16.s32 h2, y5, y4;",
                "cvt.pack.sat.u16.s32 h3, y7, y6;",
                "cvt.rn.satfinite.e2m1x2.f16x2 q0, h0;",
                "cvt.rn.satfinite.e2m1x2.f16x2 q1, h1;",
                "cvt.rn.satfinite.e2m1x2.f16x2 q2, h2;",
                "cvt.rn.satfinite.e2m1x2.f16x2 q3, h3;",
                f"mov.b32 ${g}, {{q0, q1, q2, q3}};",
                "add.rn.f16x2 t0, h0, h1;",
                "add.rn.f16x2 t1, h2, h3;",
                f"add.rn.f16x2 sm{s_}, t0, t1;",
            ]
        body += [
            "selp.b32 keep, sm1, sm0, odd;",
            "selp.b32 send, sm0, sm1, odd;",
            "shfl.sync.bfly.b32 send, send, 1, 0x1f, 0xffffffff;",
            "add.rn.f16x2 keep, keep, send;",
            "mov.b32 {v0, v1}, keep;",
            f"fma.rn.f32.f16 sum0, v1, scale{rr}, 0f00000000;",
            f"fma.rn.f32.f16 ${sumi + rr}, v0, scale{rr}, sum0;",
        ]
    body.append("}")
    r = asm(
        "\n".join(body),
        [f32(x) for x in (*scores, m, *offsets)]
        + [i32(sch), i32(par)]
        + [f32(x) for x in (kofs or [])],
        [Int32] * 4 + [Float32] * 2,
        side_effects=True,
        loc=loc,
        ip=ip,
    )
    return r[:4], r[4:]


# -------------------------------------------------------------------------------- Memory access
@dsl_user_op
def ld_global_nc_b32_pred(addr, pred, *, loc=None, ip=None):
    """i64 global address, i32 pred -> i32 word by ld.global.nc.b32 when pred != 0 (the load is predicated), 0
    otherwise."""
    return asm(
        "{\n.reg .pred p;\nsetp.ne.s32 p, $2, 0;\nmov.b32 $0, 0;\n@p ld.global.nc.b32 $0, [$1];\n}",
        [i64(addr), i32(pred)],
        Int32,
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ld_global_nc_b32_v4_pred(addr, pred, *, loc=None, ip=None):
    """i64 global address, i32 pred -> four i32 words by ld.global.nc.v4.b32 (16 bytes) when pred != 0 (the load is
    predicated), zeros otherwise."""
    return asm(
        "{\n.reg .pred p;\nsetp.ne.s32 p, $5, 0;\nmov.b32 $0, 0;\nmov.b32 $1, 0;\nmov.b32 $2, 0;\n"
        "mov.b32 $3, 0;\n@p ld.global.nc.v4.b32 {$0, $1, $2, $3}, [$4];\n}",
        [i64(addr), i32(pred)],
        [Int32] * 4,
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def atom_add_shared_u32_elect(addr, *, loc=None, ip=None):
    """i32 shared address -> i32: the lane elect.sync picks adds 1 by atom.relaxed.cta.shared::cta.add.u32 and returns
    the old count, the other lanes return 0: one increment of the shared counter per warp."""
    return asm(
        "{ .reg .pred leader; .reg .b32 old; elect.sync _|leader, 0xffffffff; "
        "mov.u32 old, 0; @leader atom.relaxed.cta.shared::cta.add.u32 old, [$1], 1; "
        "mov.u32 $0, old; }",
        [i32(addr)],
        Int32,
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def bulk_complete(dst, src, nbytes: int, mbar, *, loc=None, ip=None):
    """i32 shared address dst, i64 global address src, i32 shared address mbar of an mbarrier:
    cp.async.bulk.shared::cta.global of nbytes that completes on the mbarrier (mbarrier::complete_tx::bytes); the
    caller arms the phase with the bytes of its copies (cute.arch.mbarrier_arrive_and_expect_tx)."""
    asm(
        f"cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [$0], [$1], {nbytes}, [$2];",
        [i32(dst), i64(src), i32(mbar)],
        side_effects=True,
        loc=loc,
        ip=ip,
    )
