# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared pieces of the WinoVae CuTe DSL kernels: packed bf16 pair conversions, flat 16-byte
vector views and the compile-once launcher (cached on disk across processes)."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import cutlass.cute as cute
from cutlass import Float32, Int32
from cutlass._mlir.dialects import llvm
from cutlass._mlir_helpers.op import dsl_user_op

VEC = 8  # 16-byte vectors of 16-bit elements
EXTENT = 1 << 40  # extent of the flat vector views (indexing is unchecked)


@dsl_user_op
def pack2(lo: Float32, hi: Float32, fp16: bool = False, *, loc=None, ip=None) -> Int32:
    """Two floats rounded to nearest into one packed bf16x2 (or f16x2) word, ``lo`` in the low
    half: one ``F2FP`` (the per-element conversion takes the slow ``F2F`` path)."""
    from cutlass._mlir import ir

    kind = "f16x2" if fp16 else "bf16x2"
    res = llvm.inline_asm(
        ir.IntegerType.get_signless(32),
        [Float32(hi).ir_value(loc=loc, ip=ip), Float32(lo).ir_value(loc=loc, ip=ip)],
        f"cvt.rn.{kind}.f32 $0, $1, $2;",
        "=r,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Int32(res)


@dsl_user_op
def as_float(bits: Int32, *, loc=None, ip=None) -> Float32:
    from cutlass._mlir import ir

    value = Int32(bits).ir_value(loc=loc, ip=ip)
    return Float32(llvm.bitcast(ir.F32Type.get(), value, loc=loc, ip=ip))


def bf16_lo(word: Int32) -> Float32:
    return as_float(word << 16)


def bf16_hi(word: Int32) -> Float32:
    return as_float(word & Int32(-65536))


def unpack_bf16(words: list) -> list:
    """Four packed bf16 pairs as eight floats, in channel order."""
    return [f for w in words for f in (bf16_lo(w), bf16_hi(w))]


def vec_view(ptr: cute.Pointer) -> cute.Tensor:
    """A flat 16-bit buffer as ``[VEC, n]`` 16-byte vectors (column i is elements 8i .. 8i+7)."""
    return cute.make_tensor(ptr, cute.make_layout((VEC, EXTENT), stride=(1, VEC)))


def u32_view(ptr: cute.Pointer) -> cute.Tensor:
    """A flat 16-bit buffer as ``[4, n]`` packed pairs (column i is the 16-byte vector i)."""
    return cute.make_tensor(
        cute.recast_ptr(ptr, dtype=Int32), cute.make_layout((4, EXTENT), stride=(1, 4))
    )


def load_words(view: cute.Tensor, idx) -> list:
    w = view[(None, idx)].load()
    return [w[i] for i in range(4)]


def store_words(view: cute.Tensor, idx, words: list) -> None:
    t = cute.make_rmem_tensor((4,), Int32)
    for i in range(4):
        t[i] = words[i]
    view[(None, idx)].store(t.load())


def compile_cached(
    family: str, name: str, compile_fn: Callable[[], Any], sources: Sequence[str]
) -> Any:
    """``compile_fn`` (a ``cute.compile(..., options="--enable-tvm-ffi")`` call) through the
    on-disk CuTe DSL kernel cache when flashinfer provides it: a later process loads the
    exported object instead of compiling. ``sources``: files whose content keys the cache."""
    try:
        from flashinfer.jit.cute_dsl_core import build_and_load_cute_dsl_kernel
    except ImportError:
        return compile_fn()
    return build_and_load_cute_dsl_kernel(
        family, name, compile_fn, extra_key_files=[__file__, *sources]
    )
