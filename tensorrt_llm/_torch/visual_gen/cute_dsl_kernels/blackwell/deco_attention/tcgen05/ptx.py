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
"""Inline PTX that the tcgen05 family alone issues (B200, GB300, Rubin), through asm of common/ptx.py, whose module
docstring states the contract and the names of the wrappers.

Here: the encodings of the shared-memory matrix descriptors and of the instruction descriptors; tcgen05.mma of
kind::f8f6f4 with A in tensor memory and the means MMA of the attention; the bounded mbarrier wait of the Q/K pass,
tcgen05.commit; E2M1 decoding, an opaque identity; the load of the score scale by the epilogue warpgroup of the
attention."""

from cutlass.cute.typing import Float32, Int32, Int64
from cutlass.cutlass_dsl import dsl_user_op

from ..common.ptx import asm, i32, i64, u32


# -------------------------------------------------------------------------------- MMA and descriptors
def smem_desc_hi(sbo_bytes: int, layout: int) -> int:
    """High word of a tcgen05 shared-memory matrix descriptor: SBO (16B units), version 1, layout."""
    return ((sbo_bytes >> 4) & 0x3FFF) | (1 << 14) | (layout << 29)


def smem_desc_sw128(addr):
    """K-major SWIZZLE_128B UMMA shared-memory descriptor (8-row groups 1024 B apart)."""
    return (
        Int64((addr >> 4) & 0x3FFF)
        | Int64(1 << 16)
        | Int64(64 << 32)
        | Int64(1 << 46)
        | Int64(2 << 61)
    )


def smem_desc_mn_sw128(addr):
    """MN-major SW128: 64 channels by 8 tokens per 1024-byte atom."""
    return (
        Int64((addr >> 4) & 0x3FFF)
        | Int64(64 << 16)
        | Int64(128 << 32)
        | Int64(1 << 46)
        | Int64(2 << 61)
    )


def idesc_bf16(m, n):
    """kind::f16 instruction descriptor: BF16 x BF16 -> F32, both K-major."""
    return (1 << 4) | (1 << 7) | (1 << 10) | ((n >> 3) << 17) | ((m >> 4) << 24)


def idesc_tf32(m, n):
    """kind::tf32 instruction descriptor: TF32 x TF32 -> F32, both K-major."""
    return (1 << 4) | (2 << 7) | (2 << 10) | ((n >> 3) << 17) | ((m >> 4) << 24)


@dsl_user_op
def tc_mma_f8f6f4(d_addr, a_addr, b_desc, idesc, acc, collector: str, *, loc=None, ip=None):
    """i32 TMEM addresses d_addr (D) and a_addr (A), i64 shared-memory descriptor b_desc (B), i32 instruction descriptor
    idesc, i32 acc: tcgen05.mma.cta_group::1.kind::f8f6f4 with the A-collector qualifier `collector`, D += A B when acc
    != 0 and D = A B otherwise. The caller issues it from one elected thread."""
    asm(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, $4, 0;\n"
        "  tcgen05.mma.cta_group::1.kind::f8f6f4.collector::a::"
        + collector
        + " [$0], [$1], $2, $3, p;\n"
        "}",
        [i32(d_addr), i32(a_addr), i64(b_desc), i32(idesc), i32(acc)],
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def tc_mma_means(
    d_addr,
    smem,
    stage,
    a_off: int,
    b_off: int,
    a_desc: int,
    b_desc: int,
    idesc: int,
    *,
    loc=None,
    ip=None,
) -> None:
    """i32 TMEM address d_addr (D), i32 shared address smem, i32 stage: the means MMA of the attention, D += A B with A
    the K16 BF16 block sums at the shared address smem + a_off and B the v_mean ring stage `stage` at smem + b_off +
    4096 stage (4 KB stages), issued by the lane elect.sync picks as tcgen05.mma.cta_group::1.kind::f16. a_desc and
    b_desc are the 64-bit shared-memory descriptors of A and B without the address field (the low 14 bits take the
    address >> 4), idesc is the instruction descriptor. The descriptors are built inside the block from the base
    address and the constants."""

    def desc(name, addr, const):
        return (
            f"  shr.u32 {name}lo, {addr}, 4;\n  and.b32 {name}lo, {name}lo, 16383;\n"
            f"  or.b32 {name}lo, {name}lo, {const & 0xFFFFFFFF};\n  mov.b32 {name}hi, {const >> 32};\n"
            f"  mov.b64 {name}, {{{name}lo, {name}hi}};\n"
        )

    asm(
        "{\n"
        "  .reg .pred e, acc;\n"
        "  .reg .b32 aaddr, baddr, " + ", ".join(f"{n}lo, {n}hi" for n in "ab") + ", id;\n"
        "  .reg .b64 a, b;\n"
        f"  add.u32 aaddr, $1, {a_off};\n"
        "  shl.b32 baddr, $2, 12;\n"
        "  add.u32 baddr, baddr, $1;\n"
        f"  add.u32 baddr, baddr, {b_off};\n"
        + desc("a", "aaddr", a_desc)
        + desc("b", "baddr", b_desc)
        + f"  mov.b32 id, {idesc};\n"
        "  mov.pred acc, -1;\n"
        "  elect.sync _|e, -1;\n"
        "  @e tcgen05.mma.cta_group::1.kind::f16 [$0], a, b, id, acc;\n"
        "}",
        [i32(d_addr), i32(smem), i32(stage)],
        side_effects=True,
        loc=loc,
        ip=ip,
    )


# -------------------------------------------------------------------------------- Barriers and waits
@dsl_user_op
def mbar_wait_bounded(addr, parity, *, loc=None, ip=None):
    """i32 shared address of an mbarrier, i32 parity: loops on mbarrier.try_wait.parity.shared::cta.b64 until the phase
    of that parity has completed; trap after 2^25 failed tries. The Q/K pass waits with it."""
    asm(
        "{\n .reg .pred p;\n .reg .u32 c;\n mov.u32 c, 0;\n LAB_WAIT:\n"
        " mbarrier.try_wait.parity.shared::cta.b64 p, [$0], $1;\n @p bra LAB_DONE;\n add.u32 c, c, 1;\n"
        " setp.lt.u32 p, c, 0x2000000;\n @p bra LAB_WAIT;\n trap;\n LAB_DONE:\n}",
        [i32(addr), i32(parity)],
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def tc_commit_means(smem, stage, bar_off: int, *, loc=None, ip=None) -> None:
    """i32 shared address smem, i32 stage: the lane elect.sync picks issues tcgen05.commit (cta_group::1,
    mbarrier::arrive::one) on the mbarrier at smem + bar_off + 8 stage, the empty barrier of the v_mean ring stage."""
    asm(
        "{\n"
        "  .reg .pred e;\n"
        "  .reg .b32 bar;\n"
        "  shl.b32 bar, $1, 3;\n"
        "  add.u32 bar, bar, $0;\n"
        f"  add.u32 bar, bar, {bar_off};\n"
        "  elect.sync _|e, -1;\n"
        "  @e tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [bar];\n"
        "}",
        [i32(smem), i32(stage)],
        side_effects=True,
        loc=loc,
        ip=ip,
    )


# -------------------------------------------------------------------------------- Conversions and packed arithmetic
@dsl_user_op
def e2m1x8_to_f32x8(w, *, loc=None, ip=None):
    """u32 word of four E2M1 code bytes -> eight f32, the code values unscaled (cvt.rn.f16x2.e2m1x2 per byte,
    cvt.f32.f16); no side effects."""
    return asm(
        "{ .reg .b8 b0, b1, b2, b3; .reg .b32 h0, h1, h2, h3;\n"
        " .reg .f16 l0, u0, l1, u1, l2, u2, l3, u3;\n"
        " mov.b32 {b0, b1, b2, b3}, $8;\n"
        " cvt.rn.f16x2.e2m1x2 h0, b0;\n cvt.rn.f16x2.e2m1x2 h1, b1;\n"
        " cvt.rn.f16x2.e2m1x2 h2, b2;\n cvt.rn.f16x2.e2m1x2 h3, b3;\n"
        " mov.b32 {l0, u0}, h0;\n mov.b32 {l1, u1}, h1;\n mov.b32 {l2, u2}, h2;\n mov.b32 {l3, u3}, h3;\n"
        " cvt.f32.f16 $0, l0;\n cvt.f32.f16 $1, u0;\n cvt.f32.f16 $2, l1;\n cvt.f32.f16 $3, u1;\n"
        " cvt.f32.f16 $4, l2;\n cvt.f32.f16 $5, u2;\n cvt.f32.f16 $6, l3;\n cvt.f32.f16 $7, u3; }",
        [u32(w)],
        [Float32] * 8,
        side_effects=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def opaque_i32(x, *, loc=None, ip=None) -> Int32:
    """i32 x -> i32 x by mov.b32, declared with side effects. Callers use it to keep an address computation at the
    statement that uses it."""
    return asm("mov.b32 $0, $1;", [i32(x)], Int32, side_effects=True, loc=loc, ip=ip)


# -------------------------------------------------------------------------------- Memory access
@dsl_user_op
def ld_global_nc_f32(addr, *, loc=None, ip=None):
    """i64 global address -> f32 by ld.global.nc.f32: a per-call scalar written by an earlier kernel of the stream
    (read-only for this kernel). The epilogue warpgroup of the attention kernel reads its score scale with it
    (fp8pv/kernel.py)."""
    return asm(
        "ld.global.nc.f32 $0, [$1];", [i64(addr)], Float32, side_effects=True, loc=loc, ip=ip
    )
