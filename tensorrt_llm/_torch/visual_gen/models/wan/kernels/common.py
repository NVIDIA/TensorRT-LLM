# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared pieces of the WinoVae CuTe DSL kernels: packed bf16 pair conversions, flat 16-byte
vector views, the 1-D launch shape, and compilation through the on-disk kernel cache."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import cuda.bindings.driver as cuda
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass._mlir_helpers.op import dsl_user_op

# bf16 / fp16 channels in one 16-byte load or store; the kernels are channels-last, so a vector
# is 8 consecutive channels of one pixel.
CHANNELS_PER_VECTOR = 8
# 32-bit words (packed channel pairs) in one 16-byte vector.
WORDS_PER_VECTOR = CHANNELS_PER_VECTOR // 2
# Threads per block of the element-wise kernels (one thread per work item).
THREADS = 256
_UNBOUNDED = 1 << 40  # extent of the flat views: indexing is unchecked, callers bound it


@dsl_user_op
def pack2(lo: Float32, hi: Float32, fp16: bool = False, *, loc=None, ip=None) -> Int32:
    """Two floats rounded to nearest into one packed bf16x2 (or f16x2) word, ``lo`` in the low
    half: one ``F2FP`` (the per-element conversion takes the slow ``F2F`` path)."""
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
    """The 32 bits of ``bits`` reinterpreted as a float."""
    value = Int32(bits).ir_value(loc=loc, ip=ip)
    return Float32(llvm.bitcast(ir.F32Type.get(), value, loc=loc, ip=ip))


def bf16_lo(word: Int32) -> Float32:
    """The bf16 in the low half of a packed pair, as a float."""
    return as_float(word << 16)


def bf16_hi(word: Int32) -> Float32:
    """The bf16 in the high half of a packed pair, as a float."""
    return as_float(word & Int32(-65536))


def unpack_bf16(words: list) -> list:
    """Packed bf16 pairs as floats, low half first, in channel order."""
    return [f for w in words for f in (bf16_lo(w), bf16_hi(w))]


def round_pair_bf16(lo: Float32, hi: Float32) -> tuple[Int32, Float32, Float32]:
    """Two floats rounded to bf16: the packed word and the two rounded values as floats."""
    w = pack2(lo, hi)
    return w, bf16_lo(w), bf16_hi(w)


def pack_words(vals: list, fp16: bool = False) -> list:
    """An even number of floats as packed bf16 (or fp16) pairs, in channel order."""
    return [pack2(vals[2 * i], vals[2 * i + 1], fp16) for i in range(len(vals) // 2)]


def flat_view(ptr: cute.Pointer, width: int) -> cute.Tensor:
    """A flat buffer as ``[width, n]``: column ``i`` is elements ``width * i .. width * i +
    width - 1``, ``n`` unbounded."""
    return cute.make_tensor(ptr, cute.make_layout((width, _UNBOUNDED), stride=(1, width)))


def flat_1d(ptr: cute.Pointer) -> cute.Tensor:
    """A flat buffer as a 1-D tensor of unbounded extent."""
    return cute.make_tensor(ptr, cute.make_layout((_UNBOUNDED,)))


def vec_view(ptr: cute.Pointer) -> cute.Tensor:
    """A flat 16-bit buffer as 16-byte vectors (column i is elements 8i .. 8i+7)."""
    return flat_view(ptr, CHANNELS_PER_VECTOR)


def u32_view(ptr: cute.Pointer) -> cute.Tensor:
    """A flat 16-bit buffer as ``[WORDS_PER_VECTOR, n]`` packed pairs (column i is the 16-byte
    vector i)."""
    return flat_view(cute.recast_ptr(ptr, dtype=Int32), WORDS_PER_VECTOR)


def load_words(view: cute.Tensor, idx) -> list:
    """The packed pairs of vector ``idx`` of a :func:`u32_view`."""
    w = view[(None, idx)].load()
    return [w[i] for i in range(WORDS_PER_VECTOR)]


def store_words(view: cute.Tensor, idx, words: list) -> None:
    """Store packed pairs as vector ``idx`` of a :func:`u32_view`, one 16-byte store."""
    t = cute.make_rmem_tensor((WORDS_PER_VECTOR,), Int32)
    for i in range(WORDS_PER_VECTOR):
        t[i] = words[i]
    view[(None, idx)].store(t.load())


def grid_1d(items) -> list:
    """The grid of a launch with one thread per item, :data:`THREADS` per block."""
    return [(items + THREADS - 1) // THREADS, 1, 1]


def linear_thread_index():
    """This thread's global index in a :func:`grid_1d` launch."""
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    return Int32(bid) * THREADS + Int32(tid)


def current_stream(device: int) -> cuda.CUstream:
    """The current torch stream of ``device`` as a driver stream handle."""
    return cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)


def check_aligned(align: int, **tensors: torch.Tensor | None) -> None:
    """Reject tensors whose data does not start on an ``align``-byte boundary (the kernels
    load and store whole vectors)."""
    for name, t in tensors.items():
        if t is not None and t.data_ptr() % align:
            raise ValueError(f"{name} must be {align}-byte aligned")


def check_bias(bias: torch.Tensor | None, values: int, dtype: torch.dtype, what: str) -> None:
    """A bias is a contiguous 1-D tensor of ``values`` elements of ``dtype`` (``what`` names
    its extent in the error)."""
    if bias is not None and (
        bias.dtype != dtype or tuple(bias.shape) != (values,) or not bias.is_contiguous()
    ):
        raise ValueError(f"bias must be a contiguous {dtype} [{what}] tensor")


def compile_cached(
    family: str, name: str, compile_fn: Callable[[], Any], sources: Sequence[str]
) -> Any:
    """Run ``compile_fn`` (a ``cute.compile(..., options="--enable-tvm-ffi")`` call) through
    flashinfer's on-disk CuTe DSL kernel cache when it is installed, so a later process loads
    the exported object instead of compiling. ``sources``: files whose content keys the cache."""
    try:
        from flashinfer.jit.cute_dsl_core import build_and_load_cute_dsl_kernel
    except ImportError:
        return compile_fn()
    return build_and_load_cute_dsl_kernel(
        family, name, compile_fn, extra_key_files=[__file__, *sources]
    )
