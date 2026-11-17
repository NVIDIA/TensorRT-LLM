# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CuTe DSL GLM-5.2 TP4 dense-MLP BF16 SwiGLU.

The first three GLM layers use an unquantized local gate/up projection whose
contiguous output is ``source[M,6144]``.  The first 3072 columns are the gate
and the second 3072 columns are the up projection.  This module computes

``output = silu(source[:, :3072]) * source[:, 3072:]``

into caller-owned contiguous BF16 ``output[M,3072]`` storage.  ``M`` is a
runtime shape so the same dispatcher covers decode, mixed, context, and larger
future batches.  The dispatcher selects a separately compiled column schedule
for the active ``M`` range; model dimensions and each schedule remain compile-
time specializations.

Each thread reads 16 consecutive BF16 values from both gate and up and writes
16 BF16 results.  The resulting 32-byte accesses express the B300 256-bit GMEM
vector contract directly.  Every production schedule assigns exactly six full
warps to a row: ``cols512`` uses six one-warp CTAs, ``cols1024`` uses three
two-warp CTAs, ``cols1536`` uses two three-warp CTAs, and ``cols3072`` uses one
six-warp CTA.  The runtime dispatcher chooses among these precompiled layouts
at the B300 wave boundaries measured by the balanced M sweep.  ``cols256`` is
deliberately absent because its 16-thread CTA leaves half a warp idle.

Shared memory is intentionally absent: every input is consumed exactly once,
so staging would add traffic and synchronization without enabling reuse.  A
future staged variant must use 16-byte (128-bit) SMEM accesses.  The approximate
``exp2`` and reciprocal explicitly lower the sigmoid to ``ex2.approx`` and
``rcp.approx`` rather than a software exponential/divide sequence.

The model dimensions, BF16 dtype, 256-bit vector width, column schedule, and
PDL mode are compile-time specializations.  ``M``, tensor addresses, and the
CUDA stream remain runtime values.  ``run`` validates the tensors and launches
into caller-owned output storage.  Callers route only unquantized GLM dense-layer
SwiGLU with no positive limit to this kernel; quantized or limited calls stay on
TRT-LLM's ``swiglu``.

This kernel is written in the CUTLASS Python DSL.  It
uses ``@cute.kernel`` for its entry point and permits ``cute.arch`` indexing,
scalar math, and raw-pointer vector access; hardware synchronization uses
``cutlass.experimental.primitives``.  The implementation does not use CuTe
tensor tiling, ``cute.copy``, tiled MMA, or high-level collective algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_compact_tensor
from cutlass.experimental import primitives as prims

INPUT_FEATURES = 6_144
OUTPUT_FEATURES = 3_072
MAX_M = 2_147_483_647

_VECTOR_ELEMENTS = 16
_GMEM_VECTOR_BYTES = 32
_LOG2E = 1.4426950408889634

CompiledKernel = Callable[[torch.Tensor, torch.Tensor], object]


@dataclass(frozen=True)
class Schedule:
    """One compile-time column decomposition for a runtime row."""

    name: str
    block_columns: int

    @property
    def ctas_per_row(self) -> int:
        """Return the number of CTAs used by one output row."""
        return OUTPUT_FEATURES // self.block_columns

    @property
    def threads(self) -> int:
        """Return the number of 256-bit vector threads in one CTA."""
        return self.block_columns // _VECTOR_ELEMENTS


SCHEDULES = (
    Schedule("cols512", 512),
    Schedule("cols1024", 1_024),
    Schedule("cols1536", 1_536),
    Schedule("cols3072", 3_072),
)
_SCHEDULE_BY_COLUMNS = {schedule.block_columns: schedule for schedule in SCHEDULES}

# The non-monotonic ranges reflect B300's 148-SM launch waves rather than a
# change in the SwiGLU math.  They come from a forward/reverse-balanced sweep
# of every M in [1,512] plus production and large-M points through 32768.
_DISPATCH_BLOCK_COLUMNS = (512, 1_024, 1_536, 3_072)
_COMPILED_VARIANTS: dict[tuple[int, bool], CompiledKernel] = {}
_COMPILED_DISPATCHERS: dict[bool, CompiledDispatcher] = {}


def schedule_for_m(active_m: int) -> Schedule:
    """Select the compile-time schedule for any valid runtime ``M``."""
    if not 1 <= active_m <= MAX_M:
        raise ValueError(f"M must be in [1,{MAX_M}], got {active_m}")
    if active_m <= 24:
        return _SCHEDULE_BY_COLUMNS[512]
    if active_m <= 74:
        return _SCHEDULE_BY_COLUMNS[1_536]
    if active_m <= 97:
        return _SCHEDULE_BY_COLUMNS[1_024]
    if active_m <= 148:
        return _SCHEDULE_BY_COLUMNS[3_072]
    if active_m <= 184:
        return _SCHEDULE_BY_COLUMNS[1_536]
    return _SCHEDULE_BY_COLUMNS[3_072]


@dataclass(frozen=True)
class CompiledDispatcher:
    """Runtime-M dispatcher over independently compiled column schedules."""

    kernels: dict[int, CompiledKernel]
    use_pdl: bool

    def __call__(self, source: torch.Tensor, output: torch.Tensor) -> object:
        """Launch the schedule selected for the runtime row count."""
        schedule = schedule_for_m(int(source.shape[0]))
        return self.kernels[schedule.block_columns](source, output)


@cute.kernel
def _kernel(
    source: cute.Tensor,
    output: cute.Tensor,
    BLOCK_COLUMNS: cutlass.Constexpr[int],
    USE_PDL: cutlass.Constexpr[bool],
) -> None:
    """Apply BF16 SwiGLU to one compile-time column tile of one runtime row."""
    row, tile, _ = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()

    if cutlass.const_expr(USE_PDL):
        cute.arch.griddepcontrol_wait()

    column = tile * BLOCK_COLUMNS + tid * _VECTOR_ELEMENTS
    source_row = source.iterator.raw_ptr() + row * INPUT_FEATURES
    output_row = output.iterator.raw_ptr() + row * OUTPUT_FEATURES

    gate = (source_row + column).load(
        count=_VECTOR_ELEMENTS,
        alignment=_GMEM_VECTOR_BYTES,
    )
    up = (source_row + OUTPUT_FEATURES + column).load(
        count=_VECTOR_ELEMENTS,
        alignment=_GMEM_VECTOR_BYTES,
    )

    gate_f32 = gate.to(cutlass.Float32)
    up_f32 = up.to(cutlass.Float32)
    log2e = cutlass.vector.full_like(gate_f32, _LOG2E)
    one = cutlass.vector.full_like(gate_f32, 1.0)
    exp_neg = cute.math.exp2(-gate_f32 * log2e, fastmath=True)
    sigmoid = cute.math.rcp(one + exp_neg, approx=True, ftz=True)
    result = up_f32 * gate_f32 * sigmoid

    (output_row + column).store(
        result.to(cutlass.BFloat16),
        alignment=_GMEM_VECTOR_BYTES,
    )

    if cutlass.const_expr(USE_PDL):
        prims.barrier_cta_sync(0)
        if tid == 0:
            cute.arch.griddepcontrol_launch_dependents()


@cute.jit
def _launch(
    source: cute.Tensor,
    output: cute.Tensor,
    stream: cuda.CUstream,
    BLOCK_COLUMNS: cutlass.Constexpr[int],
    USE_PDL: cutlass.Constexpr[bool],
) -> None:
    """Launch one specialized column schedule over a runtime number of rows."""
    _kernel(source, output, BLOCK_COLUMNS, USE_PDL).launch(
        grid=(source.shape[0], OUTPUT_FEATURES // BLOCK_COLUMNS, 1),
        block=(BLOCK_COLUMNS // _VECTOR_ELEMENTS, 1, 1),
        stream=stream,
        use_pdl=USE_PDL,
    )


def compile_variant(
    block_columns: int,
    use_pdl: bool = True,
) -> CompiledKernel:
    """Compile one column schedule with runtime ``M``.

    ``source`` and ``output`` use symbolic row counts but fixed contiguous
    feature dimensions.  The 32-byte alignment assumption is part of the
    replacement ABI and enables 256-bit global loads and stores.

    :param block_columns: Output columns computed by one CTA.
    :type block_columns: int
    :param use_pdl: Enable programmatic dependent launch, defaults to True.
    :type use_pdl: bool, optional
    :return: Runtime callable accepting ``(source, output)`` tensors.
    :rtype: CompiledKernel
    """
    if block_columns not in _SCHEDULE_BY_COLUMNS:
        choices = tuple(schedule.block_columns for schedule in SCHEDULES)
        raise ValueError(f"block_columns must be one of {choices}, got {block_columns}")
    cache_key = (block_columns, bool(use_pdl))
    compiled = _COMPILED_VARIANTS.get(cache_key)
    if compiled is not None:
        return compiled
    rows = cute.sym_int64()
    source = make_fake_compact_tensor(
        cutlass.BFloat16,
        (rows, INPUT_FEATURES),
        stride_order=(1, 0),
        assumed_align=_GMEM_VECTOR_BYTES,
    )
    output = make_fake_compact_tensor(
        cutlass.BFloat16,
        (rows, OUTPUT_FEATURES),
        stride_order=(1, 0),
        assumed_align=_GMEM_VECTOR_BYTES,
    )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    compiled = cute.compile(
        _launch,
        source,
        output,
        stream,
        block_columns,
        bool(use_pdl),
        options="--enable-tvm-ffi",
    )
    _COMPILED_VARIANTS[cache_key] = compiled
    return compiled


def compile(use_pdl: bool = True) -> CompiledDispatcher:  # noqa: A001
    """Compile all schedules and return the continuous runtime-M dispatcher."""
    pdl = bool(use_pdl)
    cached = _COMPILED_DISPATCHERS.get(pdl)
    if cached is not None:
        return cached
    kernels = {
        block_columns: compile_variant(block_columns, pdl)
        for block_columns in _DISPATCH_BLOCK_COLUMNS
    }
    dispatcher = CompiledDispatcher(kernels=kernels, use_pdl=pdl)
    _COMPILED_DISPATCHERS[pdl] = dispatcher
    return dispatcher


def _validate_tensor_contract(
    source: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Validate the specialized TRT-LLM tensor boundary."""
    if source.ndim != 2 or tuple(source.shape[1:]) != (INPUT_FEATURES,):
        raise ValueError(f"source must have shape [M,{INPUT_FEATURES}], got {tuple(source.shape)}")
    active_m = int(source.shape[0])
    if not 1 <= active_m <= MAX_M:
        raise ValueError(f"source M must be in [1,{MAX_M}], got {active_m}")
    expected_output = (active_m, OUTPUT_FEATURES)
    if tuple(output.shape) != expected_output:
        raise ValueError(f"output must have shape {expected_output}, got {tuple(output.shape)}")
    for name, tensor in (("source", source), ("output", output)):
        if tensor.dtype != torch.bfloat16 or not tensor.is_cuda:
            raise ValueError(f"{name} must be a CUDA BF16 tensor")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous, got stride={tensor.stride()}")
        if tensor.data_ptr() % _GMEM_VECTOR_BYTES:
            raise ValueError(
                f"{name} must be {_GMEM_VECTOR_BYTES}-byte aligned for 256-bit GMEM access"
            )
    if source.device != output.device:
        raise ValueError("source and output must share one CUDA device")


def run(
    compiled_fn: CompiledKernel,
    source: torch.Tensor,
    output: torch.Tensor,
) -> object:
    """Validate and enqueue one call into caller-owned output storage."""
    _validate_tensor_contract(source, output)
    return compiled_fn(source, output)
