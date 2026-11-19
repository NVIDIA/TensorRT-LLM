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
"""Inline PTX that the FP8-PV kernel of the sm_120 family alone issues (sm120/fp8pv/): one of the three modules of
the package, see its docstring. The sections:

  MMA and descriptors                 mma.sync of E4M3 (the PV of the 32-key quarters) and of BF16 (the means
                                      term), each with its ldmatrix loads behind an indexed branch
  barriers and waits                  the mbarrier wait
  conversions and packed arithmetic   the comparison, a per-lane zero, the step of a ring stage, the warp vote, and
                                      the blocks of the softmax (fp8pv/softmax.py, fp8pv/small.py): the I-ExpCast
                                      landing, the E4M3 words, row sums, block sums and flags of P
  memory access                       the transposed ldmatrix of bytes, predicated streaming stores, the predicated
                                      atomic of the release count"""

from cutlass.cute.typing import Float32, Int32
from cutlass.cutlass_dsl import dsl_user_op

from ...common.ptx import asm, early, f32, i32, i64, operand, tied, u32
from ...constants import RESCALE_THRESHOLD, SUBNORMAL_ULPS
from .shared import f32_literal, ldsm


# -------------------------------------------------------------------------------- MMA and descriptors
@dsl_user_op
def mma_pv_quarters_branch(c, p, vaddrs, mask, *, loc=None, ip=None):
    """64 f32 c (the O words; tied to the results), 16 i32 words p (the packed P), 8 i32 shared addresses vaddrs (the
    V row of the lane per channel group), i32 mask -> 64 f32: the PV of the 32-key quarters whose bit is set in mask,
    each behind a warp-uniform indexed branch (brx.idx.uni) that skips the quarter's V loads and MMAs. A quarter is
    eight ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8, each four channel tiles ahead of its MMAs, and sixteen
    mma.sync.aligned.kind::f8f6f4.m16n8k32 (E4M3 x E4M3, f32 accumulation)."""
    # $0 .. $63: the O words; then the operands p, vaddrs and mask
    pi, vaddri, maski = 64, 80, 88
    body = ["{", ".reg .b32 v<32>, index;"]

    def load(ct, quarter):
        vv = ", ".join(f"v{4 * ct + j}" for j in range(4))
        return f"ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {{{vv}}}, [${vaddri + ct}+{quarter * 4096}];"

    for quarter in range(4):
        label = f"QPV{quarter}"
        body += [
            f"bfe.u32 index, ${maski}, {quarter}, 1;",
            f"{label}_TARGETS: .branchtargets {label}_SKIP, {label}_DO;",
            f"brx.idx.uni index, {label}_TARGETS;",
            f"{label}_DO:",
        ]
        for ct in range(4):
            body.append(load(ct, quarter))
        for ct in range(8):
            if ct + 4 < 8:
                body.append(load(ct + 4, quarter))
            a = ", ".join(f"v{4 * ct + j}" for j in range(4))
            for h in range(2):
                o = ", ".join(f"${8 * ct + 4 * h + j}" for j in range(4))
                b = ", ".join(f"${pi + 4 * quarter + 2 * h + j}" for j in range(2))
                body.append(
                    "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                    f"{{{o}}}, {{{a}}}, {{{b}}}, {{{o}}};"
                )
        body.append(f"{label}_SKIP:")
    body.append("}")
    return asm(
        "\n".join(body),
        [i32(x) for x in p]
        + [i32(x) for x in vaddrs]
        + [i32(mask)]
        + [tied(i, f32(x)) for i, x in enumerate(c)],
        [early(Float32)] * 64,
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def mma_means_branch(o, b, addr, idx, *, loc=None, ip=None):
    """64 f32 o (the O words; tied to the results), four i32 words b (the BF16 block sums of the first and of the second
    step, of query n-tile 0, then of n-tile 1), i32 shared address addr, i32 idx -> 64 f32: the means term of a step
    pair when idx is 1, behind one warp-uniform indexed branch (brx.idx.uni; idx 0 skips the loads and the MMAs of the
    pair). Eight ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 of the v_mean rows of the pair (channel tile ct at
    (addr ^ 32 (ct % 4)) + 1024 (ct / 4)), then sixteen mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32."""
    # $0 .. $63: the O words; then the operands b, addr and idx
    bi, addri, idxi = 64, 68, 69
    body = [
        "{",
        ".reg .b32 f<32>, a<4>;",
        "MF_T: .branchtargets MF_SKIP, MF_DO;",
        f"brx.idx.uni ${idxi}, MF_T;",
        "MF_DO:",
    ]
    for j in range(4):
        body.append(f"xor.b32 a{j}, ${addri}, {32 * j};")
    for ct in range(8):
        j, hf = ct & 3, ct >> 2
        body.append(
            f"ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {{f{4 * ct}, f{4 * ct + 1}, f{4 * ct + 2}, "
            f"f{4 * ct + 3}}}, [a{j}+{hf * 1024}];"
        )
    for ct in range(8):
        for h in range(2):
            cc = 2 * ct + h
            d = ", ".join(f"${4 * cc + i}" for i in range(4))
            body.append(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                f"{{{d}}}, {{f{4 * ct}, f{4 * ct + 1}, f{4 * ct + 2}, f{4 * ct + 3}}}, "
                f"{{${bi + 2 * h}, ${bi + 2 * h + 1}}}, {{{d}}};"
            )
    body += ["MF_SKIP:", "}"]
    return asm(
        "\n".join(body),
        [i32(x) for x in b] + [i32(addr), i32(idx)] + [tied(i, f32(x)) for i, x in enumerate(o)],
        [early(Float32)] * 64,
        side_effects=True,
        loc=loc,
        ip=ip,
    )


# -------------------------------------------------------------------------------- Barriers and waits
@dsl_user_op
def mbar_wait(addr, parity, *, loc=None, ip=None):
    """i32 shared address of an mbarrier, i32 parity: loops on mbarrier.try_wait.parity.shared::cta.b64 until the phase
    of that parity has completed, without a limit on the tries; the loop label is unique per use (${:uid})."""
    asm(
        "{\n.reg .pred p;\nWAIT_${:uid}:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [$0], $1;\n"
        "@!p bra WAIT_${:uid};\n}",
        [i32(addr), i32(parity)],
        side_effects=True,
        loc=loc,
        ip=ip,
    )


# -------------------------------------------------------------------------------- Conversions and packed arithmetic
@dsl_user_op
def compare(a, b, op: str, dtype, *, loc=None, ip=None):
    """a, b of class dtype -> i32 1 when a <op> b, else 0, for an op of setp ("gt", "lt"): setp.<op>.f32 for Float32,
    setp.<op>.s32 for Int32, then selp.s32; no side effects."""
    return asm(
        "{\n.reg .pred p;\nsetp."
        + f"{op}.{'f32' if dtype is Float32 else 's32'}"
        + " p, $1, $2;\nselp.s32 $0, 1, 0, p;\n}",
        [operand(dtype, a), operand(dtype, b)],
        Int32,
        side_effects=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def lane_zero(sc, lane, *, loc=None, ip=None):
    """f32 sc, i32 lane -> i32 (the sign bit of sc) & lane (mov.b32, shr.u32 by 31, and.b32); no side effects. The
    value is 0 on every lane, since score_scale is positive; it is computed per lane at run time."""
    return asm(
        "{\n.reg .b32 t;\nmov.b32 t, $1;\nshr.u32 t, t, 31;\nand.b32 $0, t, $2;\n}",
        [f32(sc), i32(lane)],
        Int32,
        side_effects=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def stage_next(sk, ph, stages: int, *, loc=None, ip=None):
    """i32 sk, ph -> two i32: sk + 1, or 0 when that equals `stages` (add.s32, setp.eq.s32, selp.b32), and ph, flipped
    (xor.b32 with 1) on that wrap; no side effects. The ring stage after sk (0 .. stages - 1) and the full-barrier
    parity of its next use."""
    return asm(
        "{\n.reg .pred w;\n.reg .b32 t, f;\nadd.s32 t, $2, 1;\nsetp.eq.s32 w, t, %d;\n"
        "selp.b32 $0, 0, t, w;\nxor.b32 f, $3, 1;\nselp.b32 $1, f, $3, w;\n}" % stages,
        [i32(sk), i32(ph)],
        [Int32] * 2,
        side_effects=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def vote_any(x, *, loc=None, ip=None):
    """i32 x -> i32 1 if x != 0 on any lane of the warp, else 0 (setp.ne.s32, vote.sync.any.pred over all 32 lanes)."""
    return asm(
        "{\n.reg .pred pi, po;\nsetp.ne.s32 pi, $1, 0;\nvote.sync.any.pred po, pi, 0xffffffff;\n"
        "selp.s32 $0, 1, 0, po;\n}",
        [i32(x)],
        Int32,
        side_effects=True,
        loc=loc,
        ip=ip,
    )


def _landed_bits(t):
    """The f16 bit pattern of 2^t, which a score that lands on 2^t has as its f32 bits: SUBNORMAL_ULPS per log2 unit
    above the f16 exponent bias."""
    return int((15 + t) * SUBNORMAL_ULPS)


# the blocks of the softmax (fp8pv/softmax.py, fp8pv/small.py), down to expcast_transpose4
@dsl_user_op
def expcast_guard(sc, anchor0, anchor1, *, loc=None, ip=None):
    """f32 sc, anchor0, anchor1 -> i32 warp-uniform landing path of land_pack (vote.sync.any.pred), valid until either
    anchor changes: 1, the exact two-FMA path, if on any lane sc * 2^-139 (mul.rn.f32) is not exact on the f32
    subnormal grid or an anchor is infinite or NaN; else 0, the path of one FMA per pair, which needs that multiplier
    exact (sc a multiple of 2^-10, as the power-of-two scales of the preparation are) and finite anchors."""
    sci, m0i, m1i = 1, 2, 3
    return asm(
        "{\n.reg .f32 a, r, a0, a1;\n.reg .pred bad, b0, b1, wb;\n"
        f"mul.rn.f32 a, ${sci}, 0f00000400;\n"
        "mul.rn.f32 r, a, 0f62800000;\n"
        "mul.rn.f32 r, r, 0f62000000;\n"
        f"setp.neu.f32 bad, r, ${sci};\n"
        f"abs.f32 a0, ${m0i};\nabs.f32 a1, ${m1i};\n"
        "setp.eq.f32 b0, a0, 0f7F800000;\n"
        "setp.eq.f32 b1, a1, 0f7F800000;\n"
        "or.pred bad, bad, b0;\nor.pred bad, bad, b1;\n"
        "setp.nan.f32 b0, a0, a1;\n"
        "or.pred bad, bad, b0;\n"
        "vote.sync.any.pred wb, bad, 0xffffffff;\n"
        "selp.u32 $0, 1, 0, wb;\n}",
        [f32(sc), f32(anchor0), f32(anchor1)],
        Int32,
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def land_pack(acc, sc, anchor0, anchor1, path, *, loc=None, ip=None):
    """64 f32 scores acc[n][e], f32 sc, anchor0, anchor1, i32 path (warp-uniform, taken by brx.idx.uni) -> 32 i32 words
    of two u16; no side effects. The scores of the tile of the warp landed on the f32 subnormal grid (fma.rn.f32x2: the
    f16 pattern of 2^t, t = s - anchor + T, T = RESCALE_THRESHOLD) and packed by pairs of n-tiles
    (cvt.pack.sat.u16.s32): x[8 i + 2 e + j] = f16x2 of (n-tile 4 i + 2 j, 4 i + 2 j + 1) at element e, saturating
    (t < -15, -inf -> +0; overflow and NaN -> 0xFFFF). Path 0: one FMA per score (exact multiplier); path 1: d = x sc -
    anchor, then a second FMA."""
    inputs = [acc[n][e] for n in range(16) for e in range(4)] + [sc, anchor0, anchor1]
    # $0 .. $31: the packed words; then the operands: the scores, sc, the anchors and path
    acci, sci, m0i, m1i, pathi = 32, 96, 97, 98, 99
    # the offset of the landing: a score at its anchor lands on 2^T
    bias = "0f%08X" % _landed_bits(RESCALE_THRESHOLD)
    body = [
        "{",
        ".reg .f32 z<64>, a, nm0, nm1, b0, b1;",
        ".reg .b32 t0, t1;",
        ".reg .b64 xx, zz, aa, bb0, bb1, cc, dd;",
        f"neg.f32 nm0, ${m0i};",
        f"neg.f32 nm1, ${m1i};",
        "LP_T: .branchtargets LP_FAST, LP_SLOW;",
        f"brx.idx.uni ${pathi}, LP_T;",
        "LP_FAST:",
        f"mul.rn.f32 a, ${sci}, 0f00000400;",
        f"fma.rn.f32 b0, nm0, 0f00000400, {bias};",
        f"fma.rn.f32 b1, nm1, 0f00000400, {bias};",
        "mov.b64 aa, {a, a};",
        "mov.b64 bb0, {b0, b0};",
        "mov.b64 bb1, {b1, b1};",
    ]
    for n in range(16):
        for h in range(2):
            k = 4 * n + 2 * h
            body += [
                f"mov.b64 xx, {{${acci + k}, ${acci + k + 1}}};",
                f"fma.rn.f32x2 zz, xx, aa, bb{h};",
                f"mov.b64 {{z{k}, z{k + 1}}}, zz;",
            ]
    body += [
        "bra.uni LP_PACK;",
        "LP_SLOW:",
        f"mov.b64 aa, {{${sci}, ${sci}}};",
        "mov.b64 bb0, {nm0, nm0};",
        "mov.b64 bb1, {nm1, nm1};",
        "mov.b64 cc, {0f00000400, 0f00000400};",
        f"mov.b64 dd, {{{bias}, {bias}}};",
    ]
    for n in range(16):
        for h in range(2):
            k = 4 * n + 2 * h
            body += [
                f"mov.b64 xx, {{${acci + k}, ${acci + k + 1}}};",
                f"fma.rn.f32x2 zz, xx, aa, bb{h};",
                "fma.rn.f32x2 zz, zz, cc, dd;",
                f"mov.b64 {{z{k}, z{k + 1}}}, zz;",
            ]
    body.append("LP_PACK:")
    for i in range(4):
        for e in range(4):
            for j in range(2):
                n = 4 * i + 2 * j
                body += [
                    f"mov.b32 t0, z{4 * n + e};",
                    f"mov.b32 t1, z{4 * (n + 1) + e};",
                    f"cvt.pack.sat.u16.s32 ${8 * i + 2 * e + j}, t1, t0;",
                ]
    body.append("}")
    return asm(
        "\n".join(body),
        [f32(x) for x in inputs] + [i32(path)],
        [early(Int32)] * 32,
        side_effects=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def landed_flags(xs, *, loc=None, ip=None):
    """32 i32 words xs of two u16 (the landed tile of land_pack) -> i32 warp-uniform flags from their maximum (a
    max.u16x2 tree, vote.sync.any.pred and vote.sync.all.pred). Bit 0: on some lane a landed score exceeds the f16
    pattern of 2^(2 T) (t > 2 T, that is s > anchor + T, T = RESCALE_THRESHOLD), is saturated or is NaN: the anchor may
    have to move, so the exact tile max decides. Bit 1, exclusive with bit 0: every landed value is at most 2^-10 (f16
    0x1400), which E4M3 rounds to zero: the P of the tile, its block sums and its row-sum terms are all zero."""
    xi = 1
    body = ["{", ".reg .b32 m<31>, t, u;", ".reg .pred p, wp, z, wz;"]
    vals = [f"${xi + k}" for k in range(32)]
    c = 0
    while len(vals) > 1:
        nxt = []
        for k in range(0, len(vals), 2):
            body.append(f"max.u16x2 m{c}, {vals[k]}, {vals[k + 1]};")
            nxt.append(f"m{c}")
            c += 1
        vals = nxt
    body += [
        f"shr.u32 t, {vals[0]}, 16;",
        f"and.b32 u, {vals[0]}, 65535;",
        "max.u32 t, t, u;",
        f"setp.gt.u32 p, t, {_landed_bits(2 * RESCALE_THRESHOLD)};",
        "vote.sync.any.pred wp, p, 0xffffffff;",
        "setp.le.u32 z, t, 5120;",
        "vote.sync.all.pred wz, z, 0xffffffff;",
        "selp.s32 t, 1, 0, wp;",
        "selp.s32 u, 2, 0, wz;",
        "or.b32 $0, t, u;",
        "}",
    ]
    return asm("\n".join(body), [i32(x) for x in xs], Int32, side_effects=True, loc=loc, ip=ip)


def _block_sums_decl():
    """PTX declarations of the registers of _block_sums_ptx, for the block that holds its lines."""
    return [
        ".reg .b32 bs_u, bs_k, bs_s, bs_r, bs_t, bs_z<4>;",
        ".reg .b16 bs_l, bs_h;",
        ".reg .f32 bs_fl, bs_fh;",
        ".reg .pred bs_p0, bs_p1;",
    ]


def _block_sums_ptx(y, outs, qsel):
    """PTX lines of the quad reduce-scatter of the 16-key block sums: y[4 r + i] holds this lane's partial sums (f16x2)
    of blocks 2 i (low) and 2 i + 1 (high) of row r; lane q of the quad ends with the full sums of blocks 2 q, 2 q + 1
    of row r as a BF16 pair in outs[r]: the B fragment (k = 2 q, 2 q + 1) of the means MMA for query n-tile r."""
    body = [
        f"and.b32 bs_u, {qsel}, 2;",
        "setp.ne.u32 bs_p1, bs_u, 0;",
        f"and.b32 bs_u, {qsel}, 1;",
        "setp.ne.u32 bs_p0, bs_u, 0;",
    ]
    for r in range(2):
        for k in range(2):
            body += [
                f"selp.b32 bs_k, {y[4 * r + 2 + k]}, {y[4 * r + k]}, bs_p1;",
                f"selp.b32 bs_s, {y[4 * r + k]}, {y[4 * r + 2 + k]}, bs_p1;",
                "shfl.sync.bfly.b32 bs_r, bs_s, 2, 0x1f, 0xffffffff;",
                f"add.rn.f16x2 bs_z{2 * r + k}, bs_k, bs_r;",
            ]
    for r in range(2):
        body += [
            f"selp.b32 bs_k, bs_z{2 * r + 1}, bs_z{2 * r}, bs_p0;",
            f"selp.b32 bs_s, bs_z{2 * r}, bs_z{2 * r + 1}, bs_p0;",
            "shfl.sync.bfly.b32 bs_r, bs_s, 1, 0x1f, 0xffffffff;",
            "add.rn.f16x2 bs_t, bs_k, bs_r;",
            "mov.b32 {bs_l, bs_h}, bs_t;",
            "cvt.f32.f16 bs_fl, bs_l;",
            "cvt.f32.f16 bs_fh, bs_h;",
            f"cvt.rn.bf16x2.f32 {outs[r]}, bs_fh, bs_fl;",
        ]
    return body


@dsl_user_op
def land_finish(xs, l0, l1, qsel, *, loc=None, ip=None):
    """32 i32 words xs of two f16 (the landed tile of land_pack), f32 l0, l1 (the row sums; tied to their results), i32
    qsel -> 16 i32 words of four E4M3 bytes, the two f32 sums, two i32 words of two BF16 and an i32 mask. The B words of
    the PV, P[4 i + e] (bytes: n-tiles 4 i .. 4 i + 3 at element e), by cvt.rn.satfinite.e4m3x2.f16x2; the row sums
    with those of that quantized P added (E4M3 pairs back to f16x2, exact; f16x2 trees per row grouped by 16-key
    block, added to the f32 sums); the block sums of the means MMA (the per-block nodes of the trees through the quad
    reduce-scatter of _block_sums_ptx, BF16 pairs); and the warp OR (redux.sync.or.b32) of the 4-bit mask of the
    quarters with a nonzero byte."""
    # $0 .. $15: the P words; then the results: the sums (li), the block sums (bsi) and the mask; then the operands:
    # xs (xi) and, behind the two tied sums, qsel
    li, bsi, maski, xi, qseli = 16, 18, 20, 21, 55
    body = [
        "{",
        ".reg .b16 hl, hh, ul, uh;",
        ".reg .b32 q<32>, y<8>, c<3>, w, mn, mm, mb;",
        ".reg .pred mp;",
    ] + _block_sums_decl()
    body.append("mov.u32 mm, 0;")
    for i in range(4):
        for e in range(4):
            # the unpack reads the two 16-bit halves of the packed word
            xa, xb = 8 * i + 2 * e, 8 * i + 2 * e + 1
            body += [
                f"cvt.rn.satfinite.e4m3x2.f16x2 hl, ${xi + xa};",
                f"cvt.rn.satfinite.e4m3x2.f16x2 hh, ${xi + xb};",
                "mov.b32 w, {hl, hh};",
                f"mov.b32 ${4 * i + e}, w;",
                "mov.b32 {ul, uh}, w;",
                f"cvt.rn.f16x2.e4m3x2 q{xa}, ul;",
                f"cvt.rn.f16x2.e4m3x2 q{xb}, uh;",
            ]
        # Start each quarter's exact byte mask when its four packed words are ready.
        body += [
            f"lop3.b32 mn, ${4 * i}, ${4 * i + 1}, ${4 * i + 2}, 0xfe;",
            f"or.b32 mn, mn, ${4 * i + 3};",
            "setp.ne.u32 mp, mn, 0;",
            f"selp.u32 mb, {1 << i}, 0, mp;",
            "or.b32 mm, mm, mb;",
        ]
    # The warp reduction is issued before the row and block-sum trees, which do not depend on it.
    body.append(f"redux.sync.or.b32 ${maski}, mm, 0xffffffff;")
    for row in range(2):
        for i in range(4):
            b = 8 * i + 4 * row
            body += [
                f"add.rn.f16x2 c0, q{b}, q{b + 1};",
                f"add.rn.f16x2 c1, q{b + 2}, q{b + 3};",
                f"add.rn.f16x2 y{4 * row + i}, c0, c1;",
            ]
        y = 4 * row
        body += [
            f"add.rn.f16x2 c0, y{y}, y{y + 1};",
            f"add.rn.f16x2 c1, y{y + 2}, y{y + 3};",
            "add.rn.f16x2 c2, c0, c1;",
            "mov.b32 {hl, hh}, c2;",
            f"add.rn.f32.f16 ${li + row}, hl, ${li + row};",
            f"add.rn.f32.f16 ${li + row}, hh, ${li + row};",
        ]
    body += _block_sums_ptx([f"y{k}" for k in range(8)], [f"${bsi}", f"${bsi + 1}"], f"${qseli}")
    body.append("}")
    r = asm(
        "\n".join(body),
        [i32(x) for x in xs] + [tied(li, f32(l0)), tied(li + 1, f32(l1)), i32(qsel)],
        [early(Int32)] * 16 + [Float32] * 2 + [early(Int32)] * 3,
        side_effects=True,
        loc=loc,
        ip=ip,
    )
    return r[:16], r[16], r[17], r[18:20], r[20]


@dsl_user_op
def block_sums_of_p(P, qsel, *, loc=None, ip=None):
    """16 i32 words P of four E4M3 bytes, i32 qsel -> two i32 words of two BF16: the B fragments of the means MMA, the
    sums of P over the 16-key blocks 2 q and 2 q + 1 of each of the two rows (cvt.rn.f16x2.e4m3x2, add.rn.f16x2, the
    quad reduce-scatter of _block_sums_ptx)."""
    # $0, $1: the block sums; then the operands P and qsel
    pi, qseli = 2, 18
    body = ["{", ".reg .b16 hl, hh;", ".reg .b32 a, b, y<8>;"] + _block_sums_decl()
    for r in range(2):
        for i in range(4):
            for j, e in enumerate((2 * r, 2 * r + 1)):
                body += [
                    f"mov.b32 {{hl, hh}}, ${pi + 4 * i + e};",
                    "cvt.rn.f16x2.e4m3x2 a, hl;",
                    "cvt.rn.f16x2.e4m3x2 b, hh;",
                ]
                if j == 0:
                    body.append(f"add.rn.f16x2 y{4 * r + i}, a, b;")
                else:
                    body += [
                        "add.rn.f16x2 a, a, b;",
                        f"add.rn.f16x2 y{4 * r + i}, y{4 * r + i}, a;",
                    ]
    body += _block_sums_ptx([f"y{k}" for k in range(8)], ["$0", "$1"], f"${qseli}")
    body.append("}")
    return asm(
        "\n".join(body),
        [i32(P[i]) for i in range(16)] + [i32(qsel)],
        [early(Int32)] * 2,
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def packed_pv_needed(P, v_nan, *, loc=None, ip=None):
    """16 i32 words P, i32 v_nan -> i32 warp-uniform 1 if any of the 17 operands is nonzero on any lane (a lop3.b32 OR
    tree, vote.sync.any.pred), else 0: whether the tile needs its PV MMAs, for a nonzero packed P byte or for v_nan,
    the flag of a non-finite value in the V or the v_mean of the head (rebuilt from V every call). An E4M3 NaN of
    either sign in P counts: skipping 0 * NaN would change the propagation."""
    pi, vnani = 1, 17
    body = ["{", ".reg .b32 u<8>;", ".reg .pred p, wp;"]
    for i in range(5):
        body.append(f"lop3.b32 u{i}, ${pi + 3 * i}, ${pi + 3 * i + 1}, ${pi + 3 * i + 2}, 0xfe;")
    body += [
        "lop3.b32 u5, u0, u1, u2, 0xfe;",
        f"lop3.b32 u6, u3, u4, ${pi + 15}, 0xfe;",
        f"lop3.b32 u7, u5, u6, ${vnani}, 0xfe;",
        "setp.ne.u32 p, u7, 0;",
        "vote.sync.any.pred wp, p, 0xffffffff;",
        "selp.s32 $0, 1, 0, wp;",
        "}",
    ]
    return asm(
        "\n".join(body),
        [i32(P[i]) for i in range(16)] + [i32(v_nan)],
        Int32,
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def expcast_transpose4(acc, base: int, sc, anchor0, anchor1, sum0, sum1, *, loc=None, ip=None):
    """16 f32 scores acc[base .. base + 3], f32 sc, anchor0, anchor1, f32 sum0, sum1 (the row sums of rows g and g + 8;
    tied to their results) -> four i32 words of four E4M3 bytes and the two f32 sums; no side effects. Four n-tiles of
    I-ExpCast on the bit grid of FP32, packed directly as P^T: d = fma(s, sc, -anchor) in one rounding, so that one
    path serves every positive score scale; (d + 127 + T) 2^23 converted to u32 (cvt.rzi.u32.f32) is the bit pattern
    of the FP32 2^(d + T) with a linear mantissa (T = RESCALE_THRESHOLD), which cvt.rn.satfinite.e4m3x2.f32 takes
    directly; the sums get the values of the quantized P (E4M3 back to f32, exact). The conversion to u32 clamps
    negative values, -inf (the masked keys) and NaN to zero; the anchor is the tile max, so a finite d is at most 0 and
    P at most 2^T: this path has no positive overflow."""
    inputs = [acc[base + n][e] for n in range(4) for e in range(4)] + [sc, anchor0, anchor1]
    # $0 .. $3: the P words; then the results sumi (the sums); then the operands: the scores, sc and the anchors
    sumi, acci, sci, m0i, m1i = 4, 6, 22, 23, 24
    # the offset of the conversion: d = 0 becomes the bit pattern of 2^T
    bias = f32_literal((127 + RESCALE_THRESHOLD) * 2.0**23)
    body = [
        "{",
        ".reg .f32 p<16>, z, d, nm0, nm1, f0, f1;",
        ".reg .b32 u, w;",
        ".reg .b16 lo, hi, h0, h1;",
        f"neg.f32 nm0, ${m0i};",
        f"neg.f32 nm1, ${m1i};",
    ]
    for n in range(4):
        for e in range(4):
            k = 4 * n + e
            body += [
                f"fma.rn.f32 d, ${acci + k}, ${sci}, nm{e // 2};",
                f"fma.rn.f32 z, d, 0f4B000000, {bias};",
                "cvt.rzi.u32.f32 u, z;",
                f"mov.b32 p{k}, u;",
            ]
    for e in range(4):
        body += [
            f"cvt.rn.satfinite.e4m3x2.f32 lo, p{4 + e}, p{e};",
            f"cvt.rn.satfinite.e4m3x2.f32 hi, p{12 + e}, p{8 + e};",
            f"mov.b32 ${e}, {{lo, hi}};",
        ]
        # the four values of word e belong to row e // 2: add their E4M3 values to its sum
        for half in ("lo", "hi"):
            body += [
                f"cvt.rn.f16x2.e4m3x2 w, {half};",
                "mov.b32 {h0, h1}, w;",
                "cvt.f32.f16 f0, h0;",
                "cvt.f32.f16 f1, h1;",
                f"add.f32 ${sumi + e // 2}, ${sumi + e // 2}, f0;",
                f"add.f32 ${sumi + e // 2}, ${sumi + e // 2}, f1;",
            ]
    body.append("}")
    r = asm(
        "\n".join(body),
        [f32(x) for x in inputs] + [tied(sumi, f32(sum0)), tied(sumi + 1, f32(sum1))],
        [early(Int32)] * 4 + [early(Float32)] * 2,
        side_effects=False,
        loc=loc,
        ip=ip,
    )
    return r[:4], r[4], r[5]


# -------------------------------------------------------------------------------- Memory access
@dsl_user_op
def ldsm_b8_x2_trans(addr, x=None, off=0, *, loc=None, ip=None):
    """i32 shared address addr (the row address of the lane), optionally Python ints x and off -> four i32 words by
    ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 at addr, or at (addr ^ x) + off with x (see ldsm): two 16x16 8-bit
    matrices, transposed, the rows of matrix 0 from lanes 0-15 and of matrix 1 from lanes 16-31."""
    return ldsm("m16n16.x2.trans.shared.b8", addr, x, off, loc, ip)


@dsl_user_op
def st_global_cs_b32_pred(addr, x, pred, *, loc=None, ip=None):
    """i64 global address, u32 x, i32 pred: st.global.cs.b32 (streaming) when pred != 0."""
    asm(
        "{ .reg .pred q; setp.ne.s32 q, $2, 0; @q st.global.cs.b32 [$0], $1; }",
        [i64(addr), u32(x), i32(pred)],
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_global_cs_b32_v2_pred(addr, x, y, pred, *, loc=None, ip=None):
    """i64 global address, u32 x, y, i32 pred: st.global.cs.v2.b32 (8 bytes, streaming) when pred != 0."""
    asm(
        "{ .reg .pred q; setp.ne.s32 q, $3, 0; @q st.global.cs.v2.b32 [$0], {$1, $2}; }",
        [i64(addr), u32(x), u32(y), i32(pred)],
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def atom_add_shared_u32_lane0(addr, lane, *, loc=None, ip=None):
    """i32 shared address, i32 lane -> i32: on lane 0, atom.acq_rel.cta.shared::cta.add.u32 of 1 (the atomic is
    predicated), returning the old value; 0 on the other lanes. The caller passes addr with the per-lane value of
    lane_zero added (0 on every lane)."""
    return asm(
        "{\n.reg .pred p;\nsetp.eq.s32 p, $2, 0;\nmov.u32 $0, 0;\n"
        "@p atom.acq_rel.cta.shared::cta.add.u32 $0, [$1], 1;\n}",
        [i32(addr), i32(lane)],
        early(Int32),
        side_effects=True,
        loc=loc,
        ip=ip,
    )
