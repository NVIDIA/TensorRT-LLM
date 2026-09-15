# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# Vendored from https://github.com/NVlabs/Sana (Apache-2.0); see
# THIRD_PARTY_NOTICES.md in this directory for the pin and scope.
"""CTA-local routing-mask helpers shared by the CuTe architecture backends."""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op


@dsl_user_op
def sol_attn_bfind_b32(
    value: Int32,
    *,
    loc=None,
    ip=None,
) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int32(value).ir_value(loc=loc, ip=ip)],
            "bfind.u32 $0, $1;",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
        )
    )


@dsl_user_op
def sol_attn_popc_b32(
    value: Int32,
    *,
    loc=None,
    ip=None,
) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int32(value).ir_value(loc=loc, ip=ip)],
            "popc.b32 $0, $1;",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
        )
    )


@cute.jit
def _mask_word(
    mask0: Int32,
    mask1: Int32,
    mask2: Int32,
    mask3: Int32,
    word: Int32,
) -> Int32:
    result = mask0
    if word == Int32(1):
        result = mask1
    if word == Int32(2):
        result = mask2
    if word == Int32(3):
        result = mask3
    return result


@cute.jit
def _test_exact_bit(
    mask0: Int32,
    mask1: Int32,
    mask2: Int32,
    mask3: Int32,
    offset: Int32,
) -> cutlass.Boolean:
    word = offset // Int32(32)
    bit = offset - word * Int32(32)
    return (_mask_word(mask0, mask1, mask2, mask3, word) & (Int32(1) << bit)) != Int32(0)


@cute.jit
def sol_attn_test_exact_bit_limited_words(
    mask0: Int32,
    mask1: Int32,
    mask2: Int32,
    mask3: Int32,
    offset: Int32,
    group_words: cutlass.Constexpr[int],
) -> cutlass.Boolean:
    bit = offset & Int32(31)
    if const_expr(group_words == 1):
        return (mask0 & (Int32(1) << bit)) != Int32(0)
    if const_expr(group_words == 2):
        word = mask0
        if offset >= Int32(32):
            word = mask1
        return (word & (Int32(1) << bit)) != Int32(0)
    if const_expr(group_words == 3):
        index = offset // Int32(32)
        word = mask0
        if index == Int32(1):
            word = mask1
        if index == Int32(2):
            word = mask2
        return (word & (Int32(1) << bit)) != Int32(0)
    return _test_exact_bit(mask0, mask1, mask2, mask3, offset)


@cute.jit
def sol_attn_set_exact_bit(
    mask0: Int32,
    mask1: Int32,
    mask2: Int32,
    mask3: Int32,
    offset: Int32,
):
    word = offset // Int32(32)
    bit_value = Int32(1) << (offset - word * Int32(32))
    if word == Int32(0):
        mask0 = mask0 | bit_value
    if word == Int32(1):
        mask1 = mask1 | bit_value
    if word == Int32(2):
        mask2 = mask2 | bit_value
    if word == Int32(3):
        mask3 = mask3 | bit_value
    return mask0, mask1, mask2, mask3


@cute.jit
def sol_attn_route_is_exact(
    q_block: Int32,
    kv_block: Int32,
    column_mean: Float32,
    threshold: Float32,
    valid: cutlass.Boolean,
) -> cutlass.Boolean:
    distance = q_block - kv_block
    if distance < Int32(0):
        distance = Int32(0) - distance
    return ((column_mean > threshold) or distance <= Int32(1)) and valid


@cute.jit
def sol_attn_mask_word_constexpr(
    mask0: Int32,
    mask1: Int32,
    mask2: Int32,
    mask3: Int32,
    word: cutlass.Constexpr[int],
) -> Int32:
    if const_expr(word == 0):
        return mask0
    if const_expr(word == 1):
        return mask1
    if const_expr(word == 2):
        return mask2
    return mask3


__all__ = [
    "sol_attn_bfind_b32",
    "sol_attn_mask_word_constexpr",
    "sol_attn_popc_b32",
    "sol_attn_route_is_exact",
    "sol_attn_set_exact_bit",
    "sol_attn_test_exact_bit_limited_words",
]
