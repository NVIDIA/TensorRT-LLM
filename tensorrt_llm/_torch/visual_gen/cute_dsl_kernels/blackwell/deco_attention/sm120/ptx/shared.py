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
"""Inline PTX of the sm_120 family that both of its kernel packages issue, or the preparation of Q and K
(sm120/prep/): one of the three modules of the package, see its docstring. The sections:

  MMA and descriptors                 mma.sync of BF16 and of block-scaled NVFP4 (kind::mxf4nvf4)
  barriers and waits                  mbarrier initialization
  conversions and packed arithmetic   the f32 exp2, the multiply and the FMA that keep subnormals, E2M1
                                      conversions
  memory access                       the predicated vector load of shared memory, ldmatrix, and the global loads
                                      and the global store under an L2 cache policy"""

import struct

from cutlass.cute.typing import Float32, Int32, Uint32
from cutlass.cutlass_dsl import dsl_user_op

from ...common.ptx import asm, f32, i32, i64, u32


# -------------------------------------------------------------------------------- MMA and descriptors
@dsl_user_op
def mma_bf16(c, a, b, *, loc=None, ip=None):
    """Four u32 words a (the A fragment, 16x16 BF16), two u32 words b (the B fragment, 16x8 BF16), four f32 c -> four
    f32 D = A B + C by mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32; no side effects."""
    return asm(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0, $1, $2, $3}, {$4, $5, $6, $7}, "
        "{$8, $9}, {$10, $11, $12, $13};",
        [u32(x) for x in a] + [u32(x) for x in b] + [f32(x) for x in c],
        [Float32] * 4,
        side_effects=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def mma_nvf4(c, a, b, sfa, sfb, tidb: int, *, loc=None, ip=None):
    """Four f32 c, four i32 words a, two i32 words b, i32 scale words sfa and sfb -> four f32 D = (A sfa) (B sfb) + C by
    mma.sync.aligned.m16n8k64 kind::mxf4nvf4: E2M1 x E2M1 with UE4M3 scales per 16 k (scale_vec::4X); no side effects.
    The A scales come from lanes 4g (row g) and 4g + 1 (row g + 8), the B scales from lane 4g + tidb (column g)."""
    return asm(
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
        "{$0, $1, $2, $3}, {$8, $9, $10, $11}, {$12, $13}, {$4, $5, $6, $7}, $14, {0, 0}, $15, {0, %d};"
        % tidb,
        [f32(x) for x in c] + [i32(x) for x in a] + [i32(x) for x in b] + [i32(sfa), i32(sfb)],
        [Float32] * 4,
        side_effects=False,
        loc=loc,
        ip=ip,
    )


# -------------------------------------------------------------------------------- Barriers and waits
@dsl_user_op
def mbar_init_imm(addr, count: int, *, loc=None, ip=None):
    """i32 shared address of an mbarrier: mbarrier.init.shared::cta.b64 with the immediate count."""
    asm(
        f"mbarrier.init.shared::cta.b64 [$0], {count};",
        [i32(addr)],
        side_effects=True,
        loc=loc,
        ip=ip,
    )


# -------------------------------------------------------------------------------- Conversions and packed arithmetic
@dsl_user_op
def ex2_approx_f32(x, *, loc=None, ip=None):
    """f32 x -> f32 2^x by ex2.approx.ftz.f32; no side effects."""
    return asm("ex2.approx.ftz.f32 $0, $1;", [f32(x)], Float32, side_effects=False, loc=loc, ip=ip)


@dsl_user_op
def mul_rn_f32(a, b, *, loc=None, ip=None):
    """f32 a, b -> f32 a * b by mul.rn.f32, which keeps subnormal results (no flush to zero); no side effects."""
    return asm(
        "mul.rn.f32 $0, $1, $2;", [f32(a), f32(b)], Float32, side_effects=False, loc=loc, ip=ip
    )


@dsl_user_op
def fma_rn_f32(a, b, c, *, loc=None, ip=None):
    """f32 a, b, c -> f32 a * b + c by fma.rn.f32, which keeps subnormal results (no flush to zero); no side effects."""
    return asm(
        "fma.rn.f32 $0, $1, $2, $3;",
        [f32(a), f32(b), f32(c)],
        Float32,
        side_effects=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def f32x4_to_e2m1x4(a, b, c, d, *, loc=None, ip=None):
    """f32 a, b, c, d -> u32 with four E2M1 codes in its low 16 bits, byte 0 = (b << 4) | a and byte 1 = (d << 4) | c
    (cvt.rn.satfinite.e2m1x2.f32 per pair: round to nearest even, saturating); no side effects."""
    return asm(
        "{ .reg .b8 x0, x1; .reg .b32 w0, w1;\n cvt.rn.satfinite.e2m1x2.f32 x0, $2, $1;\n"
        " cvt.rn.satfinite.e2m1x2.f32 x1, $4, $3;\n cvt.u32.u8 w0, x0;\n cvt.u32.u8 w1, x1;\n"
        " shl.b32 w1, w1, 8;\n or.b32 $0, w0, w1; }",
        [f32(a), f32(b), f32(c), f32(d)],
        Uint32,
        side_effects=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def e2m1x2_to_f32x2(byte, *, loc=None, ip=None):
    """u32 byte -> two f32, the values of the E2M1 codes in the low and in the high nibble of its low byte
    (cvt.rn.f16x2.e2m1x2, cvt.f32.f16); no side effects."""
    return asm(
        "{ .reg .b8 x; .reg .b32 h; .reg .f16 l, u; .reg .b16 s;\n cvt.u16.u32 s, $2;\n"
        " cvt.u8.u16 x, s;\n cvt.rn.f16x2.e2m1x2 h, x;\n mov.b32 {l, u}, h;\n"
        " cvt.f32.f16 $0, l;\n cvt.f32.f16 $1, u; }",
        [u32(byte)],
        [Float32] * 2,
        side_effects=False,
        loc=loc,
        ip=ip,
    )


def f32_literal(x):
    """x rounded to f32, as a PTX literal (0f and the eight hex digits of its bit pattern): the constants of the
    softmax blocks of fp8pv.py and nvfp4pv.py."""
    return "0f%08X" % struct.unpack("<I", struct.pack("<f", x))[0]


# -------------------------------------------------------------------------------- Memory access
@dsl_user_op
def ld_shared_f32_v4_pred(addr, pred, *, loc=None, ip=None):
    """i32 shared address, i32 pred -> four f32 by ld.shared.v4.f32 when pred != 0 (the load is predicated), zeros
    otherwise."""
    return asm(
        "{ .reg .pred q; setp.ne.s32 q, $5, 0;\n mov.f32 $0, 0f00000000;\n mov.f32 $1, 0f00000000;\n"
        " mov.f32 $2, 0f00000000;\n mov.f32 $3, 0f00000000;\n @q ld.shared.v4.f32 {$0, $1, $2, $3}, [$4]; }",
        [i32(addr), i32(pred)],
        [Float32] * 4,
        side_effects=True,
        loc=loc,
        ip=ip,
    )


def ldsm(form, addr, x, off, loc, ip):
    """The statement of the ldsm_* wrappers (this module and fp8pv.py): i32 shared address addr -> four i32 words by
    ldmatrix.sync.aligned.<form> at addr or, with the Python int x, at (addr ^ x) + off (a swizzled row address formed
    inside the block from the lane's base address)."""
    text = f"ldmatrix.sync.aligned.{form} {{$0, $1, $2, $3}}, [$4];"
    if x is not None:
        text = (
            f"{{\n.reg .b32 a;\nxor.b32 a, $4, {x};\n"
            f"ldmatrix.sync.aligned.{form} {{$0, $1, $2, $3}}, [a+{off}];\n}}"
        )
    return asm(text, [i32(addr)], [Int32] * 4, side_effects=True, loc=loc, ip=ip)


@dsl_user_op
def ldsm_b16_x4(addr, *, loc=None, ip=None):
    """i32 shared address addr (the row address of the lane) -> four i32 words by
    ldmatrix.sync.aligned.m8n8.x4.shared.b16: four 8x8 16-bit matrices, matrix m from the row addresses of lanes 8 m ..
    8 m + 7."""
    return ldsm("m8n8.x4.shared.b16", addr, None, 0, loc, ip)


@dsl_user_op
def ldsm_b16_x4_trans(addr, x=None, off=0, *, loc=None, ip=None):
    """i32 shared address addr (the row address of the lane), optionally Python ints x and off -> four i32 words by
    ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 at addr, or at (addr ^ x) + off with x (see ldsm): four 8x8 16-bit
    matrices, transposed, matrix m from the row addresses of lanes 8 m .. 8 m + 7."""
    return ldsm("m8n8.x4.trans.shared.b16", addr, x, off, loc, ip)


@dsl_user_op
def ld_global_nc_b32_hint_pred(addr, pred, policy, n: int, *, loc=None, ip=None):
    """i64 global address, i32 pred -> n u32 words (n 2 or 4) by ld.global.nc.L1::no_allocate.L2::cache_hint.v<n>.b32
    (4 n bytes) under the L2 cache policy `policy` (createpolicy.fractional) when pred != 0, zeros otherwise."""
    words = ", ".join(f"${i}" for i in range(n))
    return asm(
        f"{{ .reg .pred q; .reg .b64 pol; setp.ne.s32 q, ${n + 1}, 0;\n"
        f" createpolicy.fractional.L2::{policy}.b64 pol, 1.0;\n"
        + "".join(f" mov.b32 ${i}, 0;\n" for i in range(n))
        + f" @q ld.global.nc.L1::no_allocate.L2::cache_hint.v{n}.b32 {{{words}}}, [${n}], pol; }}",
        [i64(addr), i32(pred)],
        [Uint32] * n,
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_global_b32_v2_hint(addr, x, y, policy, *, loc=None, ip=None):
    """i64 global address, u32 x, y: st.global.L2::cache_hint.v2.b32 (8 bytes) under the L2 cache policy `policy`
    (createpolicy.fractional)."""
    asm(
        f"{{ .reg .b64 pol; createpolicy.fractional.L2::{policy}.b64 pol, 1.0;\n"
        " st.global.L2::cache_hint.v2.b32 [$0], {$1, $2}, pol; }",
        [i64(addr), u32(x), u32(y)],
        side_effects=True,
        loc=loc,
        ip=ip,
    )
