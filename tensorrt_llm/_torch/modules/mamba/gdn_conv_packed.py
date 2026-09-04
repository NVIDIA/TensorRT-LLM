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
"""Packed Triton implementation of the GDN causal-conv1d update.

Moves the three contiguous per-feature window values as 32-bit words instead of
three stride-3 2-byte accesses, which is what the layout of ``conv_state`` and
``intermediate_conv_window`` (``state_len`` innermost) otherwise forces. Entry
point is :func:`run`; ``causal_conv1d_update`` dispatches here only when its
host-side guard admits the exact contract these kernels address against.
"""

import triton
import triton.language as tl


@triton.jit
def _unpack_bf16x2(word):
    # bf16 -> f32 is an exact bit-extension: one shift / one mask, no cvt.
    w = word.to(tl.int32, bitcast=True)
    lo = (w << 16).to(tl.float32, bitcast=True)
    hi = (w & -65536).to(tl.float32, bitcast=True)
    return lo, hi


@triton.jit
def _unpack_bf16x4(qword):
    lo0, lo1 = _unpack_bf16x2(qword.to(tl.uint32))
    hi0, hi1 = _unpack_bf16x2((qword >> 32).to(tl.uint32))
    return lo0, lo1, hi0, hi1


@triton.jit
def _pack_two_records(a0, a1, a2, b0, b1, b2):
    first = tl.cat(tl.join(a0, a1), tl.join(a2, b0), dim=1)
    second = tl.cat(tl.join(b1, b2), tl.join(b2, b2), dim=1)
    return tl.cat(first, second, dim=1)


@triton.jit
def _pack_bf16x2(a, b):
    # one instruction: two f32 -> packed bf16x2 (first source lands in the
    # high half, so `b` is passed first).
    return tl.inline_asm_elementwise(
        asm="cvt.rn.bf16x2.f32 $0, $2, $1;",
        constraints="=r,f,f",
        args=[a, b],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _fma_bf16x2(a, b, c):
    return tl.inline_asm_elementwise(
        asm="fma.rn.bf16x2 $0, $1, $2, $3;",
        constraints="=r,r,r,r",
        args=[a, b, c],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _store_out2(base_ptr, d0, y00, y10, y01, y11, y02, y12):
    ptr = tl.cast(base_ptr + d0, tl.pointer_type(tl.uint32), bitcast=True)
    p0 = _pack_bf16x2(y00, y10)
    p1 = _pack_bf16x2(y01, y11)
    p2 = _pack_bf16x2(y02, y12)
    tl.inline_asm_elementwise(
        asm=(
            "{ st.global.b32 [$1], $4; st.global.b32 [$2], $5; "
            "st.global.b32 [$3], $6; mov.u32 $0, 0; }"
        ),
        constraints="=r,l,l,l,r,r,r",
        args=[ptr, ptr + 2048, ptr + 4096, p0, p1, p2],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _store_u32x2(ptr, lo, hi):
    return tl.inline_asm_elementwise(
        asm="{ st.global.v2.b32 [$1], {$2, $3}; mov.u32 $0, 0; }",
        constraints="=r,l,r,r",
        args=[ptr, lo, hi],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _store_out4(base_ptr, d0, y00, y10, y20, y30, y01, y11, y21, y31, y02, y12, y22, y32):
    ptr = tl.cast(base_ptr + d0, tl.pointer_type(tl.uint32), bitcast=True)
    p00, p01 = _pack_bf16x2(y00, y10), _pack_bf16x2(y20, y30)
    p10, p11 = _pack_bf16x2(y01, y11), _pack_bf16x2(y21, y31)
    p20, p21 = _pack_bf16x2(y02, y12), _pack_bf16x2(y22, y32)
    tl.inline_asm_elementwise(
        asm=(
            "{ st.global.v2.b32 [$1], {$4, $5}; "
            "st.global.v2.b32 [$2], {$6, $7}; "
            "st.global.v2.b32 [$3], {$8, $9}; mov.u32 $0, 0; }"
        ),
        constraints="=r,l,l,l,r,r,r,r,r,r",
        args=[ptr, ptr + 2048, ptr + 4096, p00, p01, p10, p11, p20, p21],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _silu_fast(x):
    # silu(x) = t + t*tanh(t), t=x/2. Reuse the destination register for t.
    return tl.inline_asm_elementwise(
        asm=(
            "{ .reg .f32 h;"
            " mul.f32 $0, $1, 0f3F000000;"
            " tanh.approx.f32 h, $0;"
            " fma.rn.f32 $0, $0, h, $0; }"
        ),
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _silu_narrow(x):
    return tl.inline_asm_elementwise(
        asm=(
            "{ .reg .f32 t;"
            " mul.f32 t, $1, 0f3F000000;"
            " tanh.approx.f32 t, t;"
            " fma.rn.f32 t, t, 0f3F000000, 0f3F000000;"
            " mul.f32 $0, $1, t; }"
        ),
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _shfl_down1_u32(value):
    return tl.inline_asm_elementwise(
        asm="shfl.sync.down.b32 $0, $1, 1, 0x1f, 0xffffffff;",
        constraints="=r,r",
        args=[value],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _shfl_up1_u32(value):
    return tl.inline_asm_elementwise(
        asm="shfl.sync.up.b32 $0, $1, 1, 0, 0xffffffff;",
        constraints="=r,r",
        args=[value],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _prmt_ll(a, b):
    return tl.inline_asm_elementwise(
        asm="prmt.b32 $0, $1, $2, 0x5410;",
        constraints="=r,r,r",
        args=[a, b],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _prmt_lh(a, b):
    return tl.inline_asm_elementwise(
        asm="prmt.b32 $0, $1, $2, 0x7610;",
        constraints="=r,r,r",
        args=[a, b],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _prmt_hl(a, b):
    return tl.inline_asm_elementwise(
        asm="prmt.b32 $0, $1, $2, 0x5432;",
        constraints="=r,r,r",
        args=[a, b],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _prmt_hh(a, b):
    return tl.inline_asm_elementwise(
        asm="prmt.b32 $0, $1, $2, 0x7632;",
        constraints="=r,r,r",
        args=[a, b],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _load_reblocked_state(base_ptr, d0, lane):
    sublane = lane & 3
    group_d0 = d0 - sublane * 2
    word_ptr = tl.cast(base_ptr + group_d0 * 3, tl.pointer_type(tl.uint32), bitcast=True)
    load_ptr = word_ptr + sublane * 4
    r0, r1, r2, r3 = tl.inline_asm_elementwise(
        asm=("{ .reg .pred p; setp.ne.u32 p, $4, 3; @p ld.global.v4.b32 {$0, $1, $2, $3}, [$5]; }"),
        constraints="=r,=r,=r,=r,r,l",
        args=[sublane, load_ptr],
        dtype=(tl.uint32, tl.uint32, tl.uint32, tl.uint32),
        is_pure=False,
        pack=1,
    )
    u1 = _shfl_up1_u32(r1)
    u2 = _shfl_up1_u32(r2)
    u3 = _shfl_up1_u32(r3)
    state_a = tl.where(sublane == 0, r0, tl.where(sublane == 1, u3, tl.where(sublane == 2, u2, u1)))
    state_b = tl.where(sublane == 0, r1, tl.where(sublane == 1, r0, tl.where(sublane == 2, u3, u2)))
    state_c = tl.where(sublane == 0, r2, tl.where(sublane == 1, r1, tl.where(sublane == 2, r0, u3)))
    return state_a, state_b, state_c


@triton.jit
def _load_reblocked_state_four(base_ptr, d0, lane):
    sublane = lane & 3
    group_d0 = d0 - sublane * 4
    word_ptr = tl.cast(base_ptr + group_d0 * 3, tl.pointer_type(tl.uint32), bitcast=True)
    load_ptr = word_ptr + sublane * 8
    r0, r1, r2, r3, r4, r5, r6, r7 = tl.inline_asm_elementwise(
        asm=(
            "{ .reg .pred p; setp.ne.u32 p, $8, 3; "
            "@p ld.global.L2::evict_first.v8.b32 {$0, $1, $2, $3, $4, $5, $6, $7}, [$9]; }"
        ),
        constraints="=r,=r,=r,=r,=r,=r,=r,=r,r,l",
        args=[sublane, load_ptr],
        dtype=(
            tl.uint32,
            tl.uint32,
            tl.uint32,
            tl.uint32,
            tl.uint32,
            tl.uint32,
            tl.uint32,
            tl.uint32,
        ),
        is_pure=False,
        pack=1,
    )
    u2 = _shfl_up1_u32(r2)
    u3 = _shfl_up1_u32(r3)
    u4 = _shfl_up1_u32(r4)
    u5 = _shfl_up1_u32(r5)
    u6 = _shfl_up1_u32(r6)
    u7 = _shfl_up1_u32(r7)
    z0 = tl.where(sublane == 0, r0, tl.where(sublane == 1, u6, tl.where(sublane == 2, u4, u2)))
    z1 = tl.where(sublane == 0, r1, tl.where(sublane == 1, u7, tl.where(sublane == 2, u5, u3)))
    z2 = tl.where(sublane == 0, r2, tl.where(sublane == 1, r0, tl.where(sublane == 2, u6, u4)))
    z3 = tl.where(sublane == 0, r3, tl.where(sublane == 1, r1, tl.where(sublane == 2, u7, u5)))
    z4 = tl.where(sublane == 0, r4, tl.where(sublane == 1, r2, tl.where(sublane == 2, r0, u6)))
    z5 = tl.where(sublane == 0, r5, tl.where(sublane == 1, r3, tl.where(sublane == 2, r1, u7)))
    return z0, z1, z2, z3, z4, z5


@triton.jit
def _reblock_two_records(a0, a1, a2, b0, b1, b2, lane):
    # A down-one shuffle of each packed word exposes the next 12-byte record.
    # Sublanes 0..2 then select consecutive 16-byte slices of the 48-byte group.
    p0 = _pack_bf16x2(a0, a1)
    p1 = _pack_bf16x2(a2, b0)
    p2 = _pack_bf16x2(b1, b2)
    q0 = _shfl_down1_u32(p0)
    q1 = _shfl_down1_u32(p1)
    q2 = _shfl_down1_u32(p2)
    sublane = lane & 3

    v0 = tl.where(sublane == 0, p0, tl.where(sublane == 1, p1, p2))
    v1 = tl.where(sublane == 0, p1, tl.where(sublane == 1, p2, q0))
    v2 = tl.where(sublane == 0, p2, tl.where(sublane == 1, q0, q1))
    v3 = tl.where(sublane == 0, q0, tl.where(sublane == 1, q1, q2))
    return v0, v1, v2, v3


@triton.jit
def _reblock_two_words(p0, p1, p2, lane):
    q0 = _shfl_down1_u32(p0)
    q1 = _shfl_down1_u32(p1)
    q2 = _shfl_down1_u32(p2)
    sublane = lane & 3
    v0 = tl.where(sublane == 0, p0, tl.where(sublane == 1, p1, p2))
    v1 = tl.where(sublane == 0, p1, tl.where(sublane == 1, p2, q0))
    v2 = tl.where(sublane == 0, p2, tl.where(sublane == 1, q0, q1))
    v3 = tl.where(sublane == 0, q0, tl.where(sublane == 1, q1, q2))
    return v0, v1, v2, v3


@triton.jit
def _reblock_four_records(a0, a1, a2, b0, b1, b2, c0, c1, c2, d0v, d1v, d2v, lane):
    p0 = _pack_bf16x2(a0, a1)
    p1 = _pack_bf16x2(a2, b0)
    p2 = _pack_bf16x2(b1, b2)
    p3 = _pack_bf16x2(c0, c1)
    p4 = _pack_bf16x2(c2, d0v)
    p5 = _pack_bf16x2(d1v, d2v)
    q0 = _shfl_down1_u32(p0)
    q1 = _shfl_down1_u32(p1)
    q2 = _shfl_down1_u32(p2)
    q3 = _shfl_down1_u32(p3)
    q4 = _shfl_down1_u32(p4)
    q5 = _shfl_down1_u32(p5)
    sublane = lane & 3
    v0 = tl.where(sublane == 0, p0, tl.where(sublane == 1, p2, p4))
    v1 = tl.where(sublane == 0, p1, tl.where(sublane == 1, p3, p5))
    v2 = tl.where(sublane == 0, p2, tl.where(sublane == 1, p4, q0))
    v3 = tl.where(sublane == 0, p3, tl.where(sublane == 1, p5, q1))
    v4 = tl.where(sublane == 0, p4, tl.where(sublane == 1, q0, q2))
    v5 = tl.where(sublane == 0, p5, tl.where(sublane == 1, q1, q3))
    v6 = tl.where(sublane == 0, q0, tl.where(sublane == 1, q2, q4))
    v7 = tl.where(sublane == 0, q1, tl.where(sublane == 1, q3, q5))
    return v0, v1, v2, v3, v4, v5, v6, v7


@triton.jit
def _reblock_words(p0, p1, p2, p3, p4, p5, lane):
    q0 = _shfl_down1_u32(p0)
    q1 = _shfl_down1_u32(p1)
    q2 = _shfl_down1_u32(p2)
    q3 = _shfl_down1_u32(p3)
    q4 = _shfl_down1_u32(p4)
    q5 = _shfl_down1_u32(p5)
    sublane = lane & 3
    v0 = tl.where(sublane == 0, p0, tl.where(sublane == 1, p2, p4))
    v1 = tl.where(sublane == 0, p1, tl.where(sublane == 1, p3, p5))
    v2 = tl.where(sublane == 0, p2, tl.where(sublane == 1, p4, q0))
    v3 = tl.where(sublane == 0, p3, tl.where(sublane == 1, p5, q1))
    v4 = tl.where(sublane == 0, p4, tl.where(sublane == 1, q0, q2))
    v5 = tl.where(sublane == 0, p5, tl.where(sublane == 1, q1, q3))
    v6 = tl.where(sublane == 0, q0, tl.where(sublane == 1, q2, q4))
    v7 = tl.where(sublane == 0, q1, tl.where(sublane == 1, q3, q5))
    return v0, v1, v2, v3, v4, v5, v6, v7


@triton.jit
def _store_reblocked(base_ptr, d0, lane, v0, v1, v2, v3):
    sublane = lane & 3
    group_d0 = d0 - sublane * 2
    word_ptr = tl.cast(base_ptr + group_d0 * 3, tl.pointer_type(tl.uint32), bitcast=True)
    store_ptr = word_ptr + sublane * 4
    predicate = (sublane < 3).to(tl.uint32)
    tl.inline_asm_elementwise(
        asm=(
            "{ .reg .pred p; setp.ne.u32 p, $1, 0; "
            "@p st.global.v4.b32 [$2], {$3, $4, $5, $6}; "
            "mov.u32 $0, 0; }"
        ),
        constraints="=r,r,l,r,r,r,r",
        args=[predicate, store_ptr, v0, v1, v2, v3],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _store_reblocked_pair(base0_ptr, base1_ptr, d0, lane, v0, v1, v2, v3):
    sublane = lane & 3
    group_d0 = d0 - sublane * 2
    word0_ptr = tl.cast(base0_ptr + group_d0 * 3, tl.pointer_type(tl.uint32), bitcast=True)
    word1_ptr = tl.cast(base1_ptr + group_d0 * 3, tl.pointer_type(tl.uint32), bitcast=True)
    store0_ptr = word0_ptr + sublane * 4
    store1_ptr = word1_ptr + sublane * 4
    tl.inline_asm_elementwise(
        asm=(
            "{ .reg .pred p; setp.ne.u32 p, $1, 3; "
            "@p st.global.v4.b32 [$2], {$4, $5, $6, $7}; "
            "@p st.global.v4.b32 [$3], {$4, $5, $6, $7}; "
            "mov.u32 $0, 0; }"
        ),
        constraints="=r,r,l,l,r,r,r,r",
        args=[sublane, store0_ptr, store1_ptr, v0, v1, v2, v3],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _store_reblocked_linear(base_ptr, d0, lane, v0, v1, v2, v3):
    sublane = lane & 3
    word_ptr = tl.cast(base_ptr, tl.pointer_type(tl.uint32), bitcast=True)
    store_ptr = word_ptr + ((d0 * 3) >> 1) + sublane
    tl.inline_asm_elementwise(
        asm=(
            "{ .reg .pred p; setp.ne.u32 p, $1, 3; "
            "@p st.global.v4.b32 [$2], {$3, $4, $5, $6}; "
            "mov.u32 $0, 0; }"
        ),
        constraints="=r,r,l,r,r,r,r",
        args=[sublane, store_ptr, v0, v1, v2, v3],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _store_reblocked_pair_linear(base0_ptr, base1_ptr, d0, lane, v0, v1, v2, v3):
    sublane = lane & 3
    word0_ptr = tl.cast(base0_ptr, tl.pointer_type(tl.uint32), bitcast=True)
    word1_ptr = tl.cast(base1_ptr, tl.pointer_type(tl.uint32), bitcast=True)
    record_word = ((d0 * 3) >> 1) + sublane
    store0_ptr = word0_ptr + record_word
    store1_ptr = word1_ptr + record_word
    tl.inline_asm_elementwise(
        asm=(
            "{ .reg .pred p; setp.ne.u32 p, $1, 3; "
            "@p st.global.v4.b32 [$2], {$4, $5, $6, $7}; "
            "@p st.global.v4.b32 [$3], {$4, $5, $6, $7}; "
            "mov.u32 $0, 0; }"
        ),
        constraints="=r,r,l,l,r,r,r,r",
        args=[sublane, store0_ptr, store1_ptr, v0, v1, v2, v3],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _store_reblocked_four(base_ptr, d0, lane, v0, v1, v2, v3, v4, v5, v6, v7):
    sublane = lane & 3
    group_d0 = d0 - sublane * 4
    word_ptr = tl.cast(base_ptr + group_d0 * 3, tl.pointer_type(tl.uint32), bitcast=True)
    store_ptr = word_ptr + sublane * 8
    predicate = (sublane < 3).to(tl.uint32)
    tl.inline_asm_elementwise(
        asm=(
            "{ .reg .pred p; setp.ne.u32 p, $1, 0; "
            "@p st.global.L2::evict_last.v8.b32 [$2], "
            "{$3, $4, $5, $6, $7, $8, $9, $10}; "
            "mov.u32 $0, 0; }"
        ),
        constraints="=r,r,l,r,r,r,r,r,r,r,r",
        args=[predicate, store_ptr, v0, v1, v2, v3, v4, v5, v6, v7],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _store_reblocked_four_pair(base0_ptr, base1_ptr, d0, lane, v0, v1, v2, v3, v4, v5, v6, v7):
    sublane = lane & 3
    group_d0 = d0 - sublane * 4
    word0_ptr = tl.cast(base0_ptr + group_d0 * 3, tl.pointer_type(tl.uint32), bitcast=True)
    word1_ptr = tl.cast(base1_ptr + group_d0 * 3, tl.pointer_type(tl.uint32), bitcast=True)
    store0_ptr = word0_ptr + sublane * 8
    store1_ptr = word1_ptr + sublane * 8
    tl.inline_asm_elementwise(
        asm=(
            "{ .reg .pred p; setp.ne.u32 p, $1, 3; "
            "@p st.global.L2::evict_last.v8.b32 [$2], {$4, $5, $6, $7, $8, $9, $10, $11}; "
            "@p st.global.L2::evict_last.v8.b32 [$3], {$4, $5, $6, $7, $8, $9, $10, $11}; "
            "mov.u32 $0, 0; }"
        ),
        constraints="=r,r,l,l,r,r,r,r,r,r,r,r",
        args=[sublane, store0_ptr, store1_ptr, v0, v1, v2, v3, v4, v5, v6, v7],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _gdn_conv_update_paired_kernel(
    x_ptr,
    conv_state_ptr,
    weight_ptr,
    bias_ptr,
    conv_state_indices_ptr,
    intermediate_ptr,
    intermediate_state_indices_ptr,
    out_ptr,
):
    tile = tl.program_id(0)
    batch_idx = tl.program_id(1)
    thread = tl.arange(0, 128)
    lane = thread & 31
    warp = thread >> 5
    warp_channel_base = tile * 256 + warp * 64
    d0 = warp_channel_base + lane * 2

    cache_line = tl.load(conv_state_indices_ptr + batch_idx)
    intermediate_line = tl.load(intermediate_state_indices_ptr + batch_idx)

    state_half_ptr = conv_state_ptr + cache_line * 12288 + d0 * 3
    state_word_ptr = tl.cast(state_half_ptr, tl.pointer_type(tl.uint32), bitcast=True)
    state_a = tl.load(state_word_ptr + 0)
    state_b = tl.load(state_word_ptr + 1)
    state_c = tl.load(state_word_ptr + 2)
    s00, s01 = _unpack_bf16x2(state_a)
    s02, s10 = _unpack_bf16x2(state_b)
    s11, s12 = _unpack_bf16x2(state_c)

    weight0, weight1, weight2, weight3 = tl.inline_asm_elementwise(
        asm="ld.global.v4.b32 {$0, $1, $2, $3}, [$4];",
        constraints="=r,=r,=r,=r,l",
        args=[weight_ptr + d0 * 4],
        dtype=(tl.uint32, tl.uint32, tl.uint32, tl.uint32),
        is_pure=False,
        pack=1,
    )
    w00, w01 = _unpack_bf16x2(weight0)
    w02, w03 = _unpack_bf16x2(weight1)
    w10, w11 = _unpack_bf16x2(weight2)
    w12, w13 = _unpack_bf16x2(weight3)

    bias_word_ptr = tl.cast(bias_ptr + d0, tl.pointer_type(tl.uint32), bitcast=True)
    bias0_bf16, bias1_bf16 = _unpack_bf16x2(tl.load(bias_word_ptr))
    bias0 = bias0_bf16.to(tl.float32)
    bias1 = bias1_bf16.to(tl.float32)

    x_word_ptr = tl.cast(
        x_ptr + batch_idx * 12288 + d0,
        tl.pointer_type(tl.uint32),
        bitcast=True,
    )
    x00, x10 = _unpack_bf16x2(tl.load(x_word_ptr + 0 * 2048))
    x01, x11 = _unpack_bf16x2(tl.load(x_word_ptr + 1 * 2048))
    x02, x12 = _unpack_bf16x2(tl.load(x_word_ptr + 2 * 2048))

    acc00 = bias0 + s00 * w00
    acc00 += s01 * w01
    acc00 += s02 * w02
    acc00 += x00 * w03
    acc10 = bias1 + s10 * w10
    acc10 += s11 * w11
    acc10 += s12 * w12
    acc10 += x10 * w13

    acc01 = bias0 + s01 * w00
    acc01 += s02 * w01
    acc01 += x00 * w02
    acc01 += x01 * w03
    acc11 = bias1 + s11 * w10
    acc11 += s12 * w11
    acc11 += x10 * w12
    acc11 += x11 * w13

    acc02 = bias0 + s02 * w00
    acc02 += x00 * w01
    acc02 += x01 * w02
    acc02 += x02 * w03
    acc12 = bias1 + s12 * w10
    acc12 += x10 * w11
    acc12 += x11 * w12
    acc12 += x12 * w13

    y00 = _silu_fast(acc00)
    y10 = _silu_fast(acc10)
    y01 = _silu_fast(acc01)
    y11 = _silu_fast(acc11)
    y02 = _silu_fast(acc02)
    y12 = _silu_fast(acc12)

    intermediate_base = intermediate_ptr + intermediate_line * 36864
    w20s, w21s, w22s, w23s = _reblock_two_records(x00, x01, x02, x10, x11, x12, lane)
    _store_reblocked_pair_linear(
        conv_state_ptr + cache_line * 12288,
        intermediate_base + 2 * 12288,
        d0,
        lane,
        w20s,
        w21s,
        w22s,
        w23s,
    )
    _store_out2(
        out_ptr + batch_idx * 12288,
        d0,
        y00,
        y10,
        y01,
        y11,
        y02,
        y12,
    )

    w00s, w01s, w02s, w03s = _reblock_two_records(s01, s02, x00, s11, s12, x10, lane)
    w10s, w11s, w12s, w13s = _reblock_two_records(s02, x00, x01, s12, x10, x11, lane)
    _store_reblocked_linear(intermediate_base + 0 * 12288, d0, lane, w00s, w01s, w02s, w03s)
    _store_reblocked_linear(intermediate_base + 1 * 12288, d0, lane, w10s, w11s, w12s, w13s)


@triton.jit
def _gdn_conv_update_paired_bf16_kernel(
    x_ptr,
    conv_state_ptr,
    weight_ptr,
    bias_ptr,
    conv_state_indices_ptr,
    intermediate_ptr,
    intermediate_state_indices_ptr,
    out_ptr,
    INTERLEAVE: tl.constexpr,
):
    tile = tl.program_id(0)
    batch_idx = tl.program_id(1)
    thread = tl.arange(0, 128)
    lane = thread & 31
    warp = thread >> 5
    d0 = tile * 256 + warp * 64 + lane * 2

    cache_line = tl.load(conv_state_indices_ptr + batch_idx)
    intermediate_line = tl.load(intermediate_state_indices_ptr + batch_idx)
    state_base = conv_state_ptr + cache_line * 12288
    state_word_ptr = tl.cast(state_base + d0 * 3, tl.pointer_type(tl.uint32), bitcast=True)
    s0 = tl.load(state_word_ptr + 0)
    s1 = tl.load(state_word_ptr + 1)
    s2 = tl.load(state_word_ptr + 2)

    w0, w1, w2, w3 = tl.inline_asm_elementwise(
        asm="ld.global.v4.b32 {$0, $1, $2, $3}, [$4];",
        constraints="=r,=r,=r,=r,l",
        args=[weight_ptr + d0 * 4],
        dtype=(tl.uint32, tl.uint32, tl.uint32, tl.uint32),
        is_pure=False,
        pack=1,
    )
    bias_word = tl.load(tl.cast(bias_ptr + d0, tl.pointer_type(tl.uint32), bitcast=True))
    x_word_ptr = tl.cast(x_ptr + batch_idx * 12288 + d0, tl.pointer_type(tl.uint32), bitcast=True)
    x0 = tl.load(x_word_ptr + 0 * 2048)
    x1 = tl.load(x_word_ptr + 1 * 2048)
    x2 = tl.load(x_word_ptr + 2 * 2048)

    st0 = _prmt_lh(s0, s1)
    st1 = _prmt_hl(s0, s2)
    st2 = _prmt_lh(s1, s2)
    wt0 = _prmt_ll(w0, w2)
    wt1 = _prmt_hh(w0, w2)
    wt2 = _prmt_ll(w1, w3)
    wt3 = _prmt_hh(w1, w3)

    if INTERLEAVE:
        acc0 = _fma_bf16x2(st0, wt0, bias_word)
        acc1 = _fma_bf16x2(st1, wt0, bias_word)
        acc2 = _fma_bf16x2(st2, wt0, bias_word)
        acc0 = _fma_bf16x2(st1, wt1, acc0)
        acc1 = _fma_bf16x2(st2, wt1, acc1)
        acc2 = _fma_bf16x2(x0, wt1, acc2)
        acc0 = _fma_bf16x2(st2, wt2, acc0)
        acc1 = _fma_bf16x2(x0, wt2, acc1)
        acc2 = _fma_bf16x2(x1, wt2, acc2)
        acc0 = _fma_bf16x2(x0, wt3, acc0)
        acc1 = _fma_bf16x2(x1, wt3, acc1)
        acc2 = _fma_bf16x2(x2, wt3, acc2)
    else:
        acc0 = _fma_bf16x2(st0, wt0, bias_word)
        acc0 = _fma_bf16x2(st1, wt1, acc0)
        acc0 = _fma_bf16x2(st2, wt2, acc0)
        acc0 = _fma_bf16x2(x0, wt3, acc0)
        acc1 = _fma_bf16x2(st1, wt0, bias_word)
        acc1 = _fma_bf16x2(st2, wt1, acc1)
        acc1 = _fma_bf16x2(x0, wt2, acc1)
        acc1 = _fma_bf16x2(x1, wt3, acc1)
        acc2 = _fma_bf16x2(st2, wt0, bias_word)
        acc2 = _fma_bf16x2(x0, wt1, acc2)
        acc2 = _fma_bf16x2(x1, wt2, acc2)
        acc2 = _fma_bf16x2(x2, wt3, acc2)
    y00, y10 = _unpack_bf16x2(acc0)
    y01, y11 = _unpack_bf16x2(acc1)
    y02, y12 = _unpack_bf16x2(acc2)

    intermediate_base = intermediate_ptr + intermediate_line * 36864
    p20 = _prmt_ll(x0, x1)
    p21 = _prmt_lh(x2, x0)
    p22 = _prmt_hh(x1, x2)
    v20, v21, v22, v23 = _reblock_two_words(p20, p21, p22, lane)
    _store_reblocked_pair_linear(
        state_base,
        intermediate_base + 24576,
        d0,
        lane,
        v20,
        v21,
        v22,
        v23,
    )
    _store_out2(
        out_ptr + batch_idx * 12288,
        d0,
        _silu_fast(y00),
        _silu_fast(y10),
        _silu_fast(y01),
        _silu_fast(y11),
        _silu_fast(y02),
        _silu_fast(y12),
    )

    p00 = _prmt_hl(s0, s1)
    p01 = _prmt_ll(x0, s2)
    p02 = _prmt_hh(s2, x0)
    v00, v01, v02, v03 = _reblock_two_words(p00, p01, p02, lane)
    p10 = _prmt_ll(s1, x0)
    p11 = _prmt_lh(x1, s2)
    p12 = _prmt_hh(x0, x1)
    v10, v11, v12, v13 = _reblock_two_words(p10, p11, p12, lane)
    _store_reblocked_linear(intermediate_base, d0, lane, v00, v01, v02, v03)
    _store_reblocked_linear(intermediate_base + 12288, d0, lane, v10, v11, v12, v13)


@triton.jit
def _gdn_conv_update_quad_kernel(
    x_ptr,
    conv_state_ptr,
    weight_ptr,
    bias_ptr,
    conv_state_indices_ptr,
    intermediate_ptr,
    intermediate_state_indices_ptr,
    out_ptr,
):
    tile = tl.program_id(0)
    batch_idx = tl.program_id(1)
    thread = tl.arange(0, 128)
    lane = thread & 31
    warp = thread >> 5
    d0 = tile * 512 + warp * 128 + lane * 4

    cache_line = tl.load(conv_state_indices_ptr + batch_idx)
    intermediate_line = tl.load(intermediate_state_indices_ptr + batch_idx)

    state_qword_ptr = tl.cast(
        conv_state_ptr + cache_line * 12288 + d0 * 3,
        tl.pointer_type(tl.uint64),
        bitcast=True,
    )
    state_a = tl.load(state_qword_ptr + 0)
    state_b = tl.load(state_qword_ptr + 1)
    state_c = tl.load(state_qword_ptr + 2)
    s00, s01, s02, s10 = _unpack_bf16x4(state_a)
    s11, s12, s20, s21 = _unpack_bf16x4(state_b)
    s22, s30, s31, s32 = _unpack_bf16x4(state_c)

    weight0, weight1, weight2, weight3 = tl.inline_asm_elementwise(
        asm="ld.global.v4.b64 {$0, $1, $2, $3}, [$4];",
        constraints="=l,=l,=l,=l,l",
        args=[weight_ptr + d0 * 4],
        dtype=(tl.uint64, tl.uint64, tl.uint64, tl.uint64),
        is_pure=False,
        pack=1,
    )
    w00, w01, w02, w03 = _unpack_bf16x4(weight0)
    w10, w11, w12, w13 = _unpack_bf16x4(weight1)
    w20, w21, w22, w23 = _unpack_bf16x4(weight2)
    w30, w31, w32, w33 = _unpack_bf16x4(weight3)

    bias_qword_ptr = tl.cast(bias_ptr + d0, tl.pointer_type(tl.uint64), bitcast=True)
    bias0_bf16, bias1_bf16, bias2_bf16, bias3_bf16 = _unpack_bf16x4(tl.load(bias_qword_ptr))
    bias0 = bias0_bf16.to(tl.float32)
    bias1 = bias1_bf16.to(tl.float32)
    bias2 = bias2_bf16.to(tl.float32)
    bias3 = bias3_bf16.to(tl.float32)

    x_qword_ptr = tl.cast(
        x_ptr + batch_idx * 12288 + d0,
        tl.pointer_type(tl.uint64),
        bitcast=True,
    )
    x00, x10, x20, x30 = _unpack_bf16x4(tl.load(x_qword_ptr + 0 * 1024))
    x01, x11, x21, x31 = _unpack_bf16x4(tl.load(x_qword_ptr + 1 * 1024))
    x02, x12, x22, x32 = _unpack_bf16x4(tl.load(x_qword_ptr + 2 * 1024))

    acc00 = bias0 + s00 * w00
    acc00 += s01 * w01
    acc00 += s02 * w02
    acc00 += x00 * w03
    acc10 = bias1 + s10 * w10
    acc10 += s11 * w11
    acc10 += s12 * w12
    acc10 += x10 * w13
    acc20 = bias2 + s20 * w20
    acc20 += s21 * w21
    acc20 += s22 * w22
    acc20 += x20 * w23
    acc30 = bias3 + s30 * w30
    acc30 += s31 * w31
    acc30 += s32 * w32
    acc30 += x30 * w33

    acc01 = bias0 + s01 * w00
    acc01 += s02 * w01
    acc01 += x00 * w02
    acc01 += x01 * w03
    acc11 = bias1 + s11 * w10
    acc11 += s12 * w11
    acc11 += x10 * w12
    acc11 += x11 * w13
    acc21 = bias2 + s21 * w20
    acc21 += s22 * w21
    acc21 += x20 * w22
    acc21 += x21 * w23
    acc31 = bias3 + s31 * w30
    acc31 += s32 * w31
    acc31 += x30 * w32
    acc31 += x31 * w33

    acc02 = bias0 + s02 * w00
    acc02 += x00 * w01
    acc02 += x01 * w02
    acc02 += x02 * w03
    acc12 = bias1 + s12 * w10
    acc12 += x10 * w11
    acc12 += x11 * w12
    acc12 += x12 * w13
    acc22 = bias2 + s22 * w20
    acc22 += x20 * w21
    acc22 += x21 * w22
    acc22 += x22 * w23
    acc32 = bias3 + s32 * w30
    acc32 += x30 * w31
    acc32 += x31 * w32
    acc32 += x32 * w33

    y00 = _silu_fast(acc00)
    y10 = _silu_fast(acc10)
    y20 = _silu_fast(acc20)
    y30 = _silu_fast(acc30)
    y01 = _silu_fast(acc01)
    y11 = _silu_fast(acc11)
    y21 = _silu_fast(acc21)
    y31 = _silu_fast(acc31)
    y02 = _silu_fast(acc02)
    y12 = _silu_fast(acc12)
    y22 = _silu_fast(acc22)
    y32 = _silu_fast(acc32)

    o0, o1, o2, o3, ot0, ot1, ot2, ot3 = _reblock_four_records(
        y00,
        y01,
        y02,
        y10,
        y11,
        y12,
        y20,
        y21,
        y22,
        y30,
        y31,
        y32,
        lane,
    )
    _store_reblocked_four(
        out_ptr + batch_idx * 12288,
        d0,
        lane,
        o0,
        o1,
        o2,
        o3,
        ot0,
        ot1,
        ot2,
        ot3,
    )

    v00, v01, v02, v03, t00, t01, t02, t03 = _reblock_four_records(
        s01,
        s02,
        x00,
        s11,
        s12,
        x10,
        s21,
        s22,
        x20,
        s31,
        s32,
        x30,
        lane,
    )
    v10, v11, v12, v13, t10, t11, t12, t13 = _reblock_four_records(
        s02,
        x00,
        x01,
        s12,
        x10,
        x11,
        s22,
        x20,
        x21,
        s32,
        x30,
        x31,
        lane,
    )
    v20, v21, v22, v23, t20, t21, t22, t23 = _reblock_four_records(
        x00,
        x01,
        x02,
        x10,
        x11,
        x12,
        x20,
        x21,
        x22,
        x30,
        x31,
        x32,
        lane,
    )
    intermediate_base = intermediate_ptr + intermediate_line * 36864
    _store_reblocked_four(
        intermediate_base,
        d0,
        lane,
        v00,
        v01,
        v02,
        v03,
        t00,
        t01,
        t02,
        t03,
    )
    _store_reblocked_four(
        intermediate_base + 12288,
        d0,
        lane,
        v10,
        v11,
        v12,
        v13,
        t10,
        t11,
        t12,
        t13,
    )
    _store_reblocked_four_pair(
        intermediate_base + 24576,
        conv_state_ptr + cache_line * 12288,
        d0,
        lane,
        v20,
        v21,
        v22,
        v23,
        t20,
        t21,
        t22,
        t23,
    )


@triton.jit
def _gdn_conv_update_paired_early_kernel(
    x_ptr,
    conv_state_ptr,
    weight_ptr,
    bias_ptr,
    conv_state_indices_ptr,
    intermediate_ptr,
    intermediate_state_indices_ptr,
    out_ptr,
    B_START: tl.constexpr,
):
    tile = tl.program_id(0)
    batch_idx = tl.program_id(1) + B_START
    thread = tl.arange(0, 128)
    lane = thread & 31
    warp = thread >> 5
    warp_channel_base = tile * 256 + warp * 64
    d0 = warp_channel_base + lane * 2

    cache_line = tl.load(conv_state_indices_ptr + batch_idx)
    intermediate_line = tl.load(intermediate_state_indices_ptr + batch_idx)

    state_half_ptr = conv_state_ptr + cache_line * 12288 + d0 * 3
    state_word_ptr = tl.cast(state_half_ptr, tl.pointer_type(tl.uint32), bitcast=True)
    state_a = tl.load(state_word_ptr + 0)
    state_b = tl.load(state_word_ptr + 1)
    state_c = tl.load(state_word_ptr + 2)
    s00, s01 = _unpack_bf16x2(state_a)
    s02, s10 = _unpack_bf16x2(state_b)
    s11, s12 = _unpack_bf16x2(state_c)

    weight0, weight1, weight2, weight3 = tl.inline_asm_elementwise(
        asm="ld.global.v4.b32 {$0, $1, $2, $3}, [$4];",
        constraints="=r,=r,=r,=r,l",
        args=[weight_ptr + d0 * 4],
        dtype=(tl.uint32, tl.uint32, tl.uint32, tl.uint32),
        is_pure=False,
        pack=1,
    )
    w00, w01 = _unpack_bf16x2(weight0)
    w02, w03 = _unpack_bf16x2(weight1)
    w10, w11 = _unpack_bf16x2(weight2)
    w12, w13 = _unpack_bf16x2(weight3)

    bias_word_ptr = tl.cast(bias_ptr + d0, tl.pointer_type(tl.uint32), bitcast=True)
    bias0_bf16, bias1_bf16 = _unpack_bf16x2(tl.load(bias_word_ptr))
    bias0 = bias0_bf16.to(tl.float32)
    bias1 = bias1_bf16.to(tl.float32)

    x_word_ptr = tl.cast(
        x_ptr + batch_idx * 12288 + d0,
        tl.pointer_type(tl.uint32),
        bitcast=True,
    )
    x00, x10 = _unpack_bf16x2(tl.load(x_word_ptr + 0 * 2048))
    x01, x11 = _unpack_bf16x2(tl.load(x_word_ptr + 1 * 2048))
    x02, x12 = _unpack_bf16x2(tl.load(x_word_ptr + 2 * 2048))

    intermediate_base = intermediate_ptr + intermediate_line * 36864
    w00s, w01s, w02s, w03s = _reblock_two_records(s01, s02, x00, s11, s12, x10, lane)
    w10s, w11s, w12s, w13s = _reblock_two_records(s02, x00, x01, s12, x10, x11, lane)
    w20s, w21s, w22s, w23s = _reblock_two_records(x00, x01, x02, x10, x11, x12, lane)
    _store_reblocked_linear(intermediate_base + 0 * 12288, d0, lane, w00s, w01s, w02s, w03s)
    _store_reblocked_linear(intermediate_base + 1 * 12288, d0, lane, w10s, w11s, w12s, w13s)
    _store_reblocked_linear(intermediate_base + 2 * 12288, d0, lane, w20s, w21s, w22s, w23s)
    _store_reblocked_linear(
        conv_state_ptr + cache_line * 12288,
        d0,
        lane,
        w20s,
        w21s,
        w22s,
        w23s,
    )

    acc00 = bias0 + s00 * w00
    acc00 += s01 * w01
    acc00 += s02 * w02
    acc00 += x00 * w03
    acc10 = bias1 + s10 * w10
    acc10 += s11 * w11
    acc10 += s12 * w12
    acc10 += x10 * w13

    acc01 = bias0 + s01 * w00
    acc01 += s02 * w01
    acc01 += x00 * w02
    acc01 += x01 * w03
    acc11 = bias1 + s11 * w10
    acc11 += s12 * w11
    acc11 += x10 * w12
    acc11 += x11 * w13

    acc02 = bias0 + s02 * w00
    acc02 += x00 * w01
    acc02 += x01 * w02
    acc02 += x02 * w03
    acc12 = bias1 + s12 * w10
    acc12 += x10 * w11
    acc12 += x11 * w12
    acc12 += x12 * w13

    y00 = _silu_fast(acc00)
    y10 = _silu_fast(acc10)
    y01 = _silu_fast(acc01)
    y11 = _silu_fast(acc11)
    y02 = _silu_fast(acc02)
    y12 = _silu_fast(acc12)

    _store_out2(
        out_ptr + batch_idx * 12288,
        d0,
        y00,
        y10,
        y01,
        y11,
        y02,
        y12,
    )


@triton.jit
def _gdn_conv_update_quad_body(
    x_ptr,
    conv_state_ptr,
    intermediate_ptr,
    out_ptr,
    batch_idx,
    cache_line,
    intermediate_line,
    d0,
    lane,
    weight0,
    weight1,
    weight2,
    weight3,
    bias_word,
):
    state_qword_ptr = tl.cast(
        conv_state_ptr + cache_line * 12288 + d0 * 3,
        tl.pointer_type(tl.uint64),
        bitcast=True,
    )
    state_a = tl.load(state_qword_ptr + 0)
    state_b = tl.load(state_qword_ptr + 1)
    state_c = tl.load(state_qword_ptr + 2)

    x_qword_ptr = tl.cast(
        x_ptr + batch_idx * 12288 + d0,
        tl.pointer_type(tl.uint64),
        bitcast=True,
    )
    x0q = tl.load(x_qword_ptr + 0 * 1024)
    x1q = tl.load(x_qword_ptr + 1 * 1024)
    x2q = tl.load(x_qword_ptr + 2 * 1024)

    s0w = state_a.to(tl.uint32)
    s1w = (state_a >> 32).to(tl.uint32)
    s2w = state_b.to(tl.uint32)
    s3w = (state_b >> 32).to(tl.uint32)
    s4w = state_c.to(tl.uint32)
    s5w = (state_c >> 32).to(tl.uint32)
    p0 = x0q.to(tl.uint32)
    p1 = (x0q >> 32).to(tl.uint32)
    q0 = x1q.to(tl.uint32)
    q1 = (x1q >> 32).to(tl.uint32)
    r0 = x2q.to(tl.uint32)
    r1 = (x2q >> 32).to(tl.uint32)

    intermediate_base = intermediate_ptr + intermediate_line * 36864
    # Commit the read/write cache line first, then drain the exact packed
    # rollback records while the independent FP32 convolution becomes ready.
    e0, e1, e2, e3, e4, e5, e6, e7 = _reblock_words(
        _prmt_ll(p0, q0),
        _prmt_lh(r0, p0),
        _prmt_hh(q0, r0),
        _prmt_ll(p1, q1),
        _prmt_lh(r1, p1),
        _prmt_hh(q1, r1),
        lane,
    )
    _store_reblocked_four_pair(
        intermediate_base + 24576,
        conv_state_ptr + cache_line * 12288,
        d0,
        lane,
        e0,
        e1,
        e2,
        e3,
        e4,
        e5,
        e6,
        e7,
    )
    a0, a1, a2, a3, a4, a5, a6, a7 = _reblock_words(
        _prmt_hl(s0w, s1w),
        _prmt_ll(p0, s2w),
        _prmt_hh(s2w, p0),
        _prmt_hl(s3w, s4w),
        _prmt_ll(p1, s5w),
        _prmt_hh(s5w, p1),
        lane,
    )
    _store_reblocked_four(
        intermediate_base,
        d0,
        lane,
        a0,
        a1,
        a2,
        a3,
        a4,
        a5,
        a6,
        a7,
    )
    c0, c1, c2, c3, c4, c5, c6, c7 = _reblock_words(
        _prmt_ll(s1w, p0),
        _prmt_lh(q0, s2w),
        _prmt_hh(p0, q0),
        _prmt_ll(s4w, p1),
        _prmt_lh(q1, s5w),
        _prmt_hh(p1, q1),
        lane,
    )
    _store_reblocked_four(
        intermediate_base + 12288,
        d0,
        lane,
        c0,
        c1,
        c2,
        c3,
        c4,
        c5,
        c6,
        c7,
    )
    s00, s01, s02, s10 = _unpack_bf16x4(state_a)
    s11, s12, s20, s21 = _unpack_bf16x4(state_b)
    s22, s30, s31, s32 = _unpack_bf16x4(state_c)

    w00, w01, w02, w03 = _unpack_bf16x4(weight0)
    w10, w11, w12, w13 = _unpack_bf16x4(weight1)
    w20, w21, w22, w23 = _unpack_bf16x4(weight2)
    w30, w31, w32, w33 = _unpack_bf16x4(weight3)

    bias0_bf16, bias1_bf16, bias2_bf16, bias3_bf16 = _unpack_bf16x4(bias_word)
    bias0 = bias0_bf16.to(tl.float32)
    bias1 = bias1_bf16.to(tl.float32)
    bias2 = bias2_bf16.to(tl.float32)
    bias3 = bias3_bf16.to(tl.float32)

    x00, x10, x20, x30 = _unpack_bf16x4(x0q)
    x01, x11, x21, x31 = _unpack_bf16x4(x1q)
    x02, x12, x22, x32 = _unpack_bf16x4(x2q)

    acc00 = bias0 + s00 * w00
    acc00 += s01 * w01
    acc00 += s02 * w02
    acc00 += x00 * w03
    acc10 = bias1 + s10 * w10
    acc10 += s11 * w11
    acc10 += s12 * w12
    acc10 += x10 * w13
    acc20 = bias2 + s20 * w20
    acc20 += s21 * w21
    acc20 += s22 * w22
    acc20 += x20 * w23
    acc30 = bias3 + s30 * w30
    acc30 += s31 * w31
    acc30 += s32 * w32
    acc30 += x30 * w33

    acc01 = bias0 + s01 * w00
    acc01 += s02 * w01
    acc01 += x00 * w02
    acc01 += x01 * w03
    acc11 = bias1 + s11 * w10
    acc11 += s12 * w11
    acc11 += x10 * w12
    acc11 += x11 * w13
    acc21 = bias2 + s21 * w20
    acc21 += s22 * w21
    acc21 += x20 * w22
    acc21 += x21 * w23
    acc31 = bias3 + s31 * w30
    acc31 += s32 * w31
    acc31 += x30 * w32
    acc31 += x31 * w33

    acc02 = bias0 + s02 * w00
    acc02 += x00 * w01
    acc02 += x01 * w02
    acc02 += x02 * w03
    acc12 = bias1 + s12 * w10
    acc12 += x10 * w11
    acc12 += x11 * w12
    acc12 += x12 * w13
    acc22 = bias2 + s22 * w20
    acc22 += x20 * w21
    acc22 += x21 * w22
    acc22 += x22 * w23
    acc32 = bias3 + s32 * w30
    acc32 += x30 * w31
    acc32 += x31 * w32
    acc32 += x32 * w33

    y00 = _silu_fast(acc00)
    y10 = _silu_fast(acc10)
    y20 = _silu_fast(acc20)
    y30 = _silu_fast(acc30)
    y01 = _silu_fast(acc01)
    y11 = _silu_fast(acc11)
    y21 = _silu_fast(acc21)
    y31 = _silu_fast(acc31)
    y02 = _silu_fast(acc02)
    y12 = _silu_fast(acc12)
    y22 = _silu_fast(acc22)
    y32 = _silu_fast(acc32)

    _store_out4(
        out_ptr + batch_idx * 12288,
        d0,
        y00,
        y10,
        y20,
        y30,
        y01,
        y11,
        y21,
        y31,
        y02,
        y12,
        y22,
        y32,
    )


@triton.jit
def _gdn_conv_update_quad_two_batch_kernel(
    x_ptr,
    conv_state_ptr,
    weight_ptr,
    bias_ptr,
    conv_state_indices_ptr,
    intermediate_ptr,
    intermediate_state_indices_ptr,
    out_ptr,
):
    tile = tl.program_id(0)
    batch_pair = tl.program_id(1)
    thread = tl.arange(0, 128)
    lane = thread & 31
    warp = thread >> 5
    d0 = tile * 512 + warp * 128 + lane * 4

    # These packed records are common to both sequence rows owned by the CTA.
    # Keeping them packed across the first call reduces their live range and
    # avoids holding all 20 unpacked weight/bias values through that call.
    weight0, weight1, weight2, weight3 = tl.inline_asm_elementwise(
        asm="ld.global.v4.b64 {$0, $1, $2, $3}, [$4];",
        constraints="=l,=l,=l,=l,l",
        args=[weight_ptr + d0 * 4],
        dtype=(tl.uint64, tl.uint64, tl.uint64, tl.uint64),
        is_pure=False,
        pack=1,
    )
    bias_word = tl.load(tl.cast(bias_ptr + d0, tl.pointer_type(tl.uint64), bitcast=True))

    batch_idx = batch_pair * 2
    # Resolve both rows' gathers before row 0 can commit conv_state. Otherwise
    # alias analysis must serialize row 1's index reads behind that store.
    line0 = tl.load(conv_state_indices_ptr + batch_idx)
    line1 = tl.load(conv_state_indices_ptr + batch_idx + 1)
    il0 = tl.load(intermediate_state_indices_ptr + batch_idx)
    il1 = tl.load(intermediate_state_indices_ptr + batch_idx + 1)
    _gdn_conv_update_quad_body(
        x_ptr,
        conv_state_ptr,
        intermediate_ptr,
        out_ptr,
        batch_idx,
        line0,
        il0,
        d0,
        lane,
        weight0,
        weight1,
        weight2,
        weight3,
        bias_word,
    )
    _gdn_conv_update_quad_body(
        x_ptr,
        conv_state_ptr,
        intermediate_ptr,
        out_ptr,
        batch_idx + 1,
        line1,
        il1,
        d0,
        lane,
        weight0,
        weight1,
        weight2,
        weight3,
        bias_word,
    )


@triton.jit
def _gdn_conv_update_quad_multi_kernel(
    x_ptr,
    conv_state_ptr,
    weight_ptr,
    bias_ptr,
    conv_state_indices_ptr,
    intermediate_ptr,
    intermediate_state_indices_ptr,
    out_ptr,
    NB: tl.constexpr,
):
    tile = tl.program_id(0)
    batch_group = tl.program_id(1)
    thread = tl.arange(0, 128)
    lane = thread & 31
    warp = thread >> 5
    d0 = tile * 512 + warp * 128 + lane * 4

    # weight/bias records are shared by every sequence row this CTA owns; keeping
    # them packed across the bodies costs 10 registers and removes NB-1 copies of
    # the 32-sector weight load and 8-sector bias load from the L1 request stream.
    weight0, weight1, weight2, weight3 = tl.inline_asm_elementwise(
        asm="ld.global.v4.b64 {$0, $1, $2, $3}, [$4];",
        constraints="=l,=l,=l,=l,l",
        args=[weight_ptr + d0 * 4],
        dtype=(tl.uint64, tl.uint64, tl.uint64, tl.uint64),
        is_pure=False,
        pack=1,
    )
    bias_word = tl.load(tl.cast(bias_ptr + d0, tl.pointer_type(tl.uint64), bitcast=True))

    for i in tl.static_range(NB):
        batch_idx = batch_group * NB + i
        cache_line = tl.load(conv_state_indices_ptr + batch_idx)
        intermediate_line = tl.load(intermediate_state_indices_ptr + batch_idx)
        _gdn_conv_update_quad_body(
            x_ptr,
            conv_state_ptr,
            intermediate_ptr,
            out_ptr,
            batch_idx,
            cache_line,
            intermediate_line,
            d0,
            lane,
            weight0,
            weight1,
            weight2,
            weight3,
            bias_word,
        )


@triton.jit
def _quad_compute_store(
    z0,
    z1,
    z2,
    z3,
    z4,
    z5,
    xq0,
    xq1,
    xq2,
    weight0,
    weight1,
    weight2,
    weight3,
    bias_word,
    out_base,
    intermediate_base,
    state_base,
    d0,
    lane,
):
    s00, s01 = _unpack_bf16x2(z0)
    s02, s10 = _unpack_bf16x2(z1)
    s11, s12 = _unpack_bf16x2(z2)
    s20, s21 = _unpack_bf16x2(z3)
    s22, s30 = _unpack_bf16x2(z4)
    s31, s32 = _unpack_bf16x2(z5)

    w00, w01, w02, w03 = _unpack_bf16x4(weight0)
    w10, w11, w12, w13 = _unpack_bf16x4(weight1)
    w20, w21, w22, w23 = _unpack_bf16x4(weight2)
    w30, w31, w32, w33 = _unpack_bf16x4(weight3)

    bias0, bias1, bias2, bias3 = _unpack_bf16x4(bias_word)

    x00, x10, x20, x30 = _unpack_bf16x4(xq0)
    x01, x11, x21, x31 = _unpack_bf16x4(xq1)
    x02, x12, x22, x32 = _unpack_bf16x4(xq2)

    acc00 = bias0 + s00 * w00
    acc00 += s01 * w01
    acc00 += s02 * w02
    acc00 += x00 * w03
    acc10 = bias1 + s10 * w10
    acc10 += s11 * w11
    acc10 += s12 * w12
    acc10 += x10 * w13
    acc20 = bias2 + s20 * w20
    acc20 += s21 * w21
    acc20 += s22 * w22
    acc20 += x20 * w23
    acc30 = bias3 + s30 * w30
    acc30 += s31 * w31
    acc30 += s32 * w32
    acc30 += x30 * w33

    acc01 = bias0 + s01 * w00
    acc01 += s02 * w01
    acc01 += x00 * w02
    acc01 += x01 * w03
    acc11 = bias1 + s11 * w10
    acc11 += s12 * w11
    acc11 += x10 * w12
    acc11 += x11 * w13
    acc21 = bias2 + s21 * w20
    acc21 += s22 * w21
    acc21 += x20 * w22
    acc21 += x21 * w23
    acc31 = bias3 + s31 * w30
    acc31 += s32 * w31
    acc31 += x30 * w32
    acc31 += x31 * w33

    acc02 = bias0 + s02 * w00
    acc02 += x00 * w01
    acc02 += x01 * w02
    acc02 += x02 * w03
    acc12 = bias1 + s12 * w10
    acc12 += x10 * w11
    acc12 += x11 * w12
    acc12 += x12 * w13
    acc22 = bias2 + s22 * w20
    acc22 += x20 * w21
    acc22 += x21 * w22
    acc22 += x22 * w23
    acc32 = bias3 + s32 * w30
    acc32 += x30 * w31
    acc32 += x31 * w32
    acc32 += x32 * w33

    o0, o1, o2, o3, ot0, ot1, ot2, ot3 = _reblock_four_records(
        _silu_fast(acc00),
        _silu_fast(acc01),
        _silu_fast(acc02),
        _silu_fast(acc10),
        _silu_fast(acc11),
        _silu_fast(acc12),
        _silu_fast(acc20),
        _silu_fast(acc21),
        _silu_fast(acc22),
        _silu_fast(acc30),
        _silu_fast(acc31),
        _silu_fast(acc32),
        lane,
    )
    _store_reblocked_four(out_base, d0, lane, o0, o1, o2, o3, ot0, ot1, ot2, ot3)

    v00, v01, v02, v03, t00, t01, t02, t03 = _reblock_four_records(
        s01,
        s02,
        x00,
        s11,
        s12,
        x10,
        s21,
        s22,
        x20,
        s31,
        s32,
        x30,
        lane,
    )
    _store_reblocked_four(
        intermediate_base,
        d0,
        lane,
        v00,
        v01,
        v02,
        v03,
        t00,
        t01,
        t02,
        t03,
    )
    v10, v11, v12, v13, t10, t11, t12, t13 = _reblock_four_records(
        s02,
        x00,
        x01,
        s12,
        x10,
        x11,
        s22,
        x20,
        x21,
        s32,
        x30,
        x31,
        lane,
    )
    _store_reblocked_four(
        intermediate_base + 12288,
        d0,
        lane,
        v10,
        v11,
        v12,
        v13,
        t10,
        t11,
        t12,
        t13,
    )
    v20, v21, v22, v23, t20, t21, t22, t23 = _reblock_four_records(
        x00,
        x01,
        x02,
        x10,
        x11,
        x12,
        x20,
        x21,
        x22,
        x30,
        x31,
        x32,
        lane,
    )
    _store_reblocked_four_pair(
        intermediate_base + 24576,
        state_base,
        d0,
        lane,
        v20,
        v21,
        v22,
        v23,
        t20,
        t21,
        t22,
        t23,
    )


@triton.jit
def _gdn_conv_update_quad_pipe2_kernel(
    x_ptr,
    conv_state_ptr,
    weight_ptr,
    bias_ptr,
    conv_state_indices_ptr,
    intermediate_ptr,
    intermediate_state_indices_ptr,
    out_ptr,
):
    tile = tl.program_id(0)
    bp = tl.program_id(1)
    thread = tl.arange(0, 128)
    lane = thread & 31
    warp = thread >> 5
    d0 = tile * 512 + warp * 128 + lane * 4

    b0 = bp * 2
    line0 = tl.load(conv_state_indices_ptr + b0)
    line1 = tl.load(conv_state_indices_ptr + b0 + 1)
    il0 = tl.load(intermediate_state_indices_ptr + b0)
    il1 = tl.load(intermediate_state_indices_ptr + b0 + 1)

    state0_base = conv_state_ptr + line0 * 12288
    state1_base = conv_state_ptr + line1 * 12288

    # Every load for BOTH sequence rows is issued before the first store, so the
    # two gathered-state round trips overlap. Interleaving loads with stores lets
    # the compiler keep no more than one conv_state miss in flight, because the
    # store to conv_state may alias the next row's load.
    a0, a1, a2, a3, a4, a5 = _load_reblocked_state_four(state0_base, d0, lane)
    b_0, b_1, b_2, b_3, b_4, b_5 = _load_reblocked_state_four(state1_base, d0, lane)

    xa_ptr = tl.cast(x_ptr + b0 * 12288 + d0, tl.pointer_type(tl.uint64), bitcast=True)
    xa0 = tl.load(xa_ptr + 0 * 1024)
    xa1 = tl.load(xa_ptr + 1 * 1024)
    xa2 = tl.load(xa_ptr + 2 * 1024)
    xb_ptr = tl.cast(x_ptr + (b0 + 1) * 12288 + d0, tl.pointer_type(tl.uint64), bitcast=True)
    xb0 = tl.load(xb_ptr + 0 * 1024)
    xb1 = tl.load(xb_ptr + 1 * 1024)
    xb2 = tl.load(xb_ptr + 2 * 1024)

    weight0, weight1, weight2, weight3 = tl.inline_asm_elementwise(
        asm="ld.global.v4.b64 {$0, $1, $2, $3}, [$4];",
        constraints="=l,=l,=l,=l,l",
        args=[weight_ptr + d0 * 4],
        dtype=(tl.uint64, tl.uint64, tl.uint64, tl.uint64),
        is_pure=False,
        pack=1,
    )
    bias_word = tl.load(tl.cast(bias_ptr + d0, tl.pointer_type(tl.uint64), bitcast=True))

    _quad_compute_store(
        a0,
        a1,
        a2,
        a3,
        a4,
        a5,
        xa0,
        xa1,
        xa2,
        weight0,
        weight1,
        weight2,
        weight3,
        bias_word,
        out_ptr + b0 * 12288,
        intermediate_ptr + il0 * 36864,
        state0_base,
        d0,
        lane,
    )
    _quad_compute_store(
        b_0,
        b_1,
        b_2,
        b_3,
        b_4,
        b_5,
        xb0,
        xb1,
        xb2,
        weight0,
        weight1,
        weight2,
        weight3,
        bias_word,
        out_ptr + (b0 + 1) * 12288,
        intermediate_ptr + il1 * 36864,
        state1_base,
        d0,
        lane,
    )


@triton.jit
def _load_state6_plain(base_ptr, d0):
    p = tl.cast(base_ptr + d0 * 3, tl.pointer_type(tl.uint64), bitcast=True)
    qa = tl.load(p + 0)
    qb = tl.load(p + 1)
    qc = tl.load(p + 2)
    return (
        qa.to(tl.uint32),
        (qa >> 32).to(tl.uint32),
        qb.to(tl.uint32),
        (qb >> 32).to(tl.uint32),
        qc.to(tl.uint32),
        (qc >> 32).to(tl.uint32),
    )


@triton.jit
def _gdn_conv_update_narrow_body(
    z0,
    z1,
    z2,
    z3,
    z4,
    z5,
    xq0,
    xq1,
    xq2,
    weight0,
    weight1,
    weight2,
    weight3,
    weight4,
    weight5,
    weight6,
    weight7,
    bias_word0,
    bias_word1,
    out_base,
    intermediate_base,
    state_base,
    d0,
    lane,
):
    s00, s01 = _unpack_bf16x2(z0)
    s02, s10 = _unpack_bf16x2(z1)
    s11, s12 = _unpack_bf16x2(z2)
    s20, s21 = _unpack_bf16x2(z3)
    s22, s30 = _unpack_bf16x2(z4)
    s31, s32 = _unpack_bf16x2(z5)

    p00, p01 = xq0.to(tl.uint32), (xq0 >> 32).to(tl.uint32)
    p10, p11 = xq1.to(tl.uint32), (xq1 >> 32).to(tl.uint32)
    p20, p21 = xq2.to(tl.uint32), (xq2 >> 32).to(tl.uint32)

    v20, v21, v22, v23, t20, t21, t22, t23 = _reblock_words(
        _prmt_ll(p00, p10),
        _prmt_lh(p20, p00),
        _prmt_hh(p10, p20),
        _prmt_ll(p01, p11),
        _prmt_lh(p21, p01),
        _prmt_hh(p11, p21),
        lane,
    )
    _store_reblocked_four_pair(
        intermediate_base + 24576,
        state_base,
        d0,
        lane,
        v20,
        v21,
        v22,
        v23,
        t20,
        t21,
        t22,
        t23,
    )

    v00, v01, v02, v03, t00, t01, t02, t03 = _reblock_words(
        _prmt_hl(z0, z1),
        _prmt_ll(p00, z2),
        _prmt_hh(z2, p00),
        _prmt_hl(z3, z4),
        _prmt_ll(p01, z5),
        _prmt_hh(z5, p01),
        lane,
    )
    _store_reblocked_four(
        intermediate_base,
        d0,
        lane,
        v00,
        v01,
        v02,
        v03,
        t00,
        t01,
        t02,
        t03,
    )
    v10, v11, v12, v13, t10, t11, t12, t13 = _reblock_words(
        _prmt_ll(z1, p00),
        _prmt_lh(p10, z2),
        _prmt_hh(p00, p10),
        _prmt_ll(z4, p01),
        _prmt_lh(p11, z5),
        _prmt_hh(p01, p11),
        lane,
    )
    _store_reblocked_four(
        intermediate_base + 12288,
        d0,
        lane,
        v10,
        v11,
        v12,
        v13,
        t10,
        t11,
        t12,
        t13,
    )

    w00, w01 = _unpack_bf16x2(weight0)
    w02, w03 = _unpack_bf16x2(weight1)
    w10, w11 = _unpack_bf16x2(weight2)
    w12, w13 = _unpack_bf16x2(weight3)
    w20, w21 = _unpack_bf16x2(weight4)
    w22, w23 = _unpack_bf16x2(weight5)
    w30, w31 = _unpack_bf16x2(weight6)
    w32, w33 = _unpack_bf16x2(weight7)
    bias0, bias1 = _unpack_bf16x2(bias_word0)
    bias2, bias3 = _unpack_bf16x2(bias_word1)
    x00, x10, x20, x30 = _unpack_bf16x4(xq0)
    x01, x11, x21, x31 = _unpack_bf16x4(xq1)
    x02, x12, x22, x32 = _unpack_bf16x4(xq2)

    acc00 = bias0 + s00 * w00 + s01 * w01 + s02 * w02 + x00 * w03
    acc01 = bias0 + s01 * w00 + s02 * w01 + x00 * w02 + x01 * w03
    acc02 = bias0 + s02 * w00 + x00 * w01 + x01 * w02 + x02 * w03
    acc10 = bias1 + s10 * w10 + s11 * w11 + s12 * w12 + x10 * w13
    acc11 = bias1 + s11 * w10 + s12 * w11 + x10 * w12 + x11 * w13
    acc12 = bias1 + s12 * w10 + x10 * w11 + x11 * w12 + x12 * w13
    acc20 = bias2 + s20 * w20 + s21 * w21 + s22 * w22 + x20 * w23
    acc21 = bias2 + s21 * w20 + s22 * w21 + x20 * w22 + x21 * w23
    acc22 = bias2 + s22 * w20 + x20 * w21 + x21 * w22 + x22 * w23
    acc30 = bias3 + s30 * w30 + s31 * w31 + s32 * w32 + x30 * w33
    acc31 = bias3 + s31 * w30 + s32 * w31 + x30 * w32 + x31 * w33
    acc32 = bias3 + s32 * w30 + x30 * w31 + x31 * w32 + x32 * w33

    _store_out4(
        out_base,
        d0,
        _silu_narrow(acc00),
        _silu_narrow(acc10),
        _silu_narrow(acc20),
        _silu_narrow(acc30),
        _silu_narrow(acc01),
        _silu_narrow(acc11),
        _silu_narrow(acc21),
        _silu_narrow(acc31),
        _silu_narrow(acc02),
        _silu_narrow(acc12),
        _silu_narrow(acc22),
        _silu_narrow(acc32),
    )


@triton.jit
def _gdn_conv_update_narrow_kernel(
    x_ptr,
    conv_state_ptr,
    weight_ptr,
    bias_ptr,
    conv_state_indices_ptr,
    intermediate_ptr,
    intermediate_state_indices_ptr,
    out_ptr,
    TS: tl.constexpr,
):
    tile = tl.program_id(0)
    batch_pair = tl.program_id(1)
    thread = tl.arange(0, TS)
    lane = thread & 31
    d0 = tile * (TS * 4) + thread * 4

    b0 = batch_pair * 2
    line0 = tl.load(conv_state_indices_ptr + b0)
    line1 = tl.load(conv_state_indices_ptr + b0 + 1)
    il0 = tl.load(intermediate_state_indices_ptr + b0)
    il1 = tl.load(intermediate_state_indices_ptr + b0 + 1)
    state0_base = conv_state_ptr + line0 * 12288
    state1_base = conv_state_ptr + line1 * 12288

    a0, a1, a2, a3, a4, a5 = _load_state6_plain(state0_base, d0)
    xa_ptr = tl.cast(x_ptr + b0 * 12288 + d0, tl.pointer_type(tl.uint64), bitcast=True)
    xa0 = tl.load(xa_ptr + 0 * 1024)
    xa1 = tl.load(xa_ptr + 1 * 1024)
    xa2 = tl.load(xa_ptr + 2 * 1024)
    c0, c1, c2, c3, c4, c5 = _load_state6_plain(state1_base, d0)
    xb_ptr = tl.cast(x_ptr + (b0 + 1) * 12288 + d0, tl.pointer_type(tl.uint64), bitcast=True)
    xb0 = tl.load(xb_ptr + 0 * 1024)
    xb1 = tl.load(xb_ptr + 1 * 1024)
    xb2 = tl.load(xb_ptr + 2 * 1024)

    weight0, weight1, weight2, weight3, weight4, weight5, weight6, weight7 = (
        tl.inline_asm_elementwise(
            asm="ld.global.v8.b32 {$0, $1, $2, $3, $4, $5, $6, $7}, [$8];",
            constraints="=r,=r,=r,=r,=r,=r,=r,=r,l",
            args=[weight_ptr + d0 * 4],
            dtype=(
                tl.uint32,
                tl.uint32,
                tl.uint32,
                tl.uint32,
                tl.uint32,
                tl.uint32,
                tl.uint32,
                tl.uint32,
            ),
            is_pure=False,
            pack=1,
        )
    )
    bias_word0, bias_word1 = tl.inline_asm_elementwise(
        asm="ld.global.v2.b32 {$0, $1}, [$2];",
        constraints="=r,=r,l",
        args=[bias_ptr + d0],
        dtype=(tl.uint32, tl.uint32),
        is_pure=False,
        pack=1,
    )

    _gdn_conv_update_narrow_body(
        a0,
        a1,
        a2,
        a3,
        a4,
        a5,
        xa0,
        xa1,
        xa2,
        weight0,
        weight1,
        weight2,
        weight3,
        weight4,
        weight5,
        weight6,
        weight7,
        bias_word0,
        bias_word1,
        out_ptr + b0 * 12288,
        intermediate_ptr + il0 * 36864,
        state0_base,
        d0,
        lane,
    )
    _gdn_conv_update_narrow_body(
        c0,
        c1,
        c2,
        c3,
        c4,
        c5,
        xb0,
        xb1,
        xb2,
        weight0,
        weight1,
        weight2,
        weight3,
        weight4,
        weight5,
        weight6,
        weight7,
        bias_word0,
        bias_word1,
        out_ptr + (b0 + 1) * 12288,
        intermediate_ptr + il1 * 36864,
        state1_base,
        d0,
        lane,
    )


def run(
    x,
    conv_state,
    weight,
    bias,
    conv_state_indices,
    intermediate_conv_window,
    intermediate_state_indices,
    out,
):
    batch = x.shape[0]
    args = (
        x,
        conv_state,
        weight,
        bias,
        conv_state_indices,
        intermediate_conv_window,
        intermediate_state_indices,
        out,
    )
    if batch >= 512:
        _gdn_conv_update_narrow_kernel[(16, batch // 2)](*args, TS=64, num_warps=2, num_stages=1)
    elif batch >= 256:
        _gdn_conv_update_quad_two_batch_kernel[(8, batch // 2)](*args, num_warps=4, num_stages=1)
    elif batch >= 128:
        _gdn_conv_update_narrow_kernel[(32, batch // 2)](*args, TS=32, num_warps=1, num_stages=1)
    elif batch >= 64:
        _gdn_conv_update_paired_bf16_kernel[(16, batch)](
            *args, INTERLEAVE=True, num_warps=4, num_stages=1
        )
    else:
        _gdn_conv_update_paired_bf16_kernel[(16, batch)](
            *args, INTERLEAVE=False, num_warps=4, num_stages=1
        )
        return
    if batch & 1:
        _gdn_conv_update_paired_early_kernel[(16, 1)](
            *args, B_START=batch - 1, num_warps=4, num_stages=1
        )
