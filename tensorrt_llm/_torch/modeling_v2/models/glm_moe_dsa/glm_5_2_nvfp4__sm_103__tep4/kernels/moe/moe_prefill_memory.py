# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Initialization and local BF16 merge for GLM prefill buffers.

Zero consumes a flat uint8 view, a positive byte count divisible by 16, and a
positive CTA cap. Merge consumes flat contiguous BF16 tensors whose equal
length is divisible by 8. All pointers are 16-byte aligned. Zero writes exactly
the requested extent; merge may alias either input exactly, but partial overlap
is unsupported. Callers retain the current buffers through graph replay.

These operations are shape-generic low-level kernels; serving selection must
remain restricted to T=65..8192. Initialization on the native auxiliary stream
retains its main-ready and zero-completion event generations. A same-stream PDL
wait cannot replace an event that joins an independent shared/routed producer.
The merge must execute after both branches have joined. PDL-enabled launches
wait before any load/store and signal only after each CTA has issued its stores.
Dependent readers still WAIT for producer completion before accessing output.
"""

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver as cuda
from cutlass.cute import runtime
from cutlass.experimental import primitives as prims

_THREADS = 256
_BYTES_PER_VECTOR = 16
_BF16_PER_VECTOR = 8


@cute.kernel
def zero_kernel(output: cute.Tensor, byte_count: cutlass.Int32, use_pdl: cutlass.Constexpr[bool]):
    tid, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    grid, _, _ = cute.arch.grid_dim()
    if cutlass.const_expr(use_pdl):
        prims.griddepcontrol(prims.GridDepAction.WAIT)
    zero = cutlass.Vector.from_elements((cutlass.Int64(0), cutlass.Int64(0)), dtype=cutlass.Int64)
    pointer = output.iterator.raw_ptr()
    for vector in cutlass.range(
        block * _THREADS + tid, byte_count // _BYTES_PER_VECTOR, grid * _THREADS
    ):
        prims.store_ext(zero, pointer + vector * _BYTES_PER_VECTOR)
    if cutlass.const_expr(use_pdl):
        prims.barrier_cta_sync(0)
        if tid == 0:
            prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)


zero_kernel.set_name_prefix("glm52_prefill_zero", remove_cutlass_symbol=True)


@cute.jit
def launch_zero(
    output: cute.Tensor,
    byte_count: cutlass.Int32,
    maximum_ctas: cutlass.Int32,
    use_pdl: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    grid = cute.min(cute.ceil_div(byte_count, _THREADS * _BYTES_PER_VECTOR), maximum_ctas)
    zero_kernel(output, byte_count, use_pdl).launch(
        grid=(grid, 1, 1), block=(_THREADS, 1, 1), stream=stream, use_pdl=use_pdl
    )


@cute.kernel
def merge_kernel(
    shared: cute.Tensor,
    routed: cute.Tensor,
    output: cute.Tensor,
    elements: cutlass.Int32,
    use_pdl: cutlass.Constexpr[bool],
):
    tid, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    grid, _, _ = cute.arch.grid_dim()
    if cutlass.const_expr(use_pdl):
        prims.griddepcontrol(prims.GridDepAction.WAIT)
    shared_ptr, routed_ptr, output_ptr = (
        shared.iterator.raw_ptr(),
        routed.iterator.raw_ptr(),
        output.iterator.raw_ptr(),
    )
    for vector in cutlass.range(
        block * _THREADS + tid, elements // _BF16_PER_VECTOR, grid * _THREADS
    ):
        offset = vector * _BF16_PER_VECTOR
        lhs = (
            prims.load_ext(shared_ptr + offset, dtype=cutlass.Int16, count=_BF16_PER_VECTOR)
            .bitcast(cutlass.BFloat16)
            .to(cutlass.Float32)
        )
        rhs = (
            prims.load_ext(routed_ptr + offset, dtype=cutlass.Int16, count=_BF16_PER_VECTOR)
            .bitcast(cutlass.BFloat16)
            .to(cutlass.Float32)
        )
        result = (lhs + rhs).to(cutlass.BFloat16).bitcast(cutlass.Int16)
        prims.store_ext(result, output_ptr + offset)
    if cutlass.const_expr(use_pdl):
        prims.barrier_cta_sync(0)
        if tid == 0:
            prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)


merge_kernel.set_name_prefix("glm52_prefill_local_merge", remove_cutlass_symbol=True)


@cute.jit
def launch_merge(
    shared: cute.Tensor,
    routed: cute.Tensor,
    output: cute.Tensor,
    elements: cutlass.Int32,
    maximum_ctas: cutlass.Int32,
    use_pdl: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    grid = cute.min(cute.ceil_div(elements, _THREADS * _BF16_PER_VECTOR), maximum_ctas)
    merge_kernel(shared, routed, output, elements, use_pdl).launch(
        grid=(grid, 1, 1), block=(_THREADS, 1, 1), stream=stream, use_pdl=use_pdl
    )


def compile_zero(*, use_pdl: bool):
    """Compile runtime-size zero; uint8 view and byte count stay caller-owned."""
    if type(use_pdl) is not bool:
        raise ValueError("use_pdl must be bool")
    output = runtime.make_fake_compact_tensor(
        cutlass.Uint8, (cute.sym_int64(divisibility=16),), assumed_align=16
    )
    return cute.compile(
        launch_zero,
        output,
        cutlass.Int32(16),
        cutlass.Int32(148),
        use_pdl,
        runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi --generate-line-info",
    )


def compile_merge(*, use_pdl: bool):
    """Compile rounded BF16 shared+routed addition after the caller's join."""
    if type(use_pdl) is not bool:
        raise ValueError("use_pdl must be bool")
    tensor = runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (cute.sym_int64(divisibility=8),), assumed_align=16
    )
    return cute.compile(
        launch_merge,
        tensor,
        tensor,
        tensor,
        cutlass.Int32(8),
        cutlass.Int32(148),
        use_pdl,
        runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi --generate-line-info",
    )
