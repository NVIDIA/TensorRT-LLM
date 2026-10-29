# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Low-resource CuTe primitive output initialization for GLM-5.2 MoE."""

from collections.abc import Callable

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import torch
from cutlass.cute.runtime import from_dlpack
from cutlass.experimental import primitives as prims

_THREADS = 256
_DEFAULT_ROWS_PER_CTA = 1
_HIDDEN = 6144
_BF16_PER_STORE = 8

ZeroFn = Callable[..., None]
_CACHE: dict[tuple[int, int, bool, int], ZeroFn] = {}


@cute.kernel
def zero_rows_kernel(
    output: cute.Tensor,
    token_count: cutlass.Constexpr,
    signal_pdl: cutlass.Constexpr,
    rows_per_cta: cutlass.Constexpr,
) -> None:
    """Clear independent BF16 rows using one 256-thread subgroup per row."""
    tidx, _, _ = cute.arch.thread_idx()
    row_group, _, _ = cute.arch.block_idx()
    row_in_cta = tidx // cutlass.Int32(_THREADS)
    row_thread = tidx % cutlass.Int32(_THREADS)
    row = row_group * cutlass.Int32(rows_per_cta) + row_in_cta
    output_ptr = output.iterator.raw_ptr()
    vectors_per_row = _HIDDEN // _BF16_PER_STORE
    zero_128b = cutlass.Vector.from_elements(
        (cutlass.Int64(0), cutlass.Int64(0)), dtype=cutlass.Int64
    )
    if row < cutlass.Int32(token_count):
        vector = row_thread
        while vector < cutlass.Int32(vectors_per_row):
            offset = row * cutlass.Int32(_HIDDEN) + vector * cutlass.Int32(_BF16_PER_STORE)
            prims.store_ext(zero_128b, output_ptr + offset)
            vector += cutlass.Int32(_THREADS)

    if cutlass.const_expr(signal_pdl):
        prims.barrier_cta_sync(0)
        if tidx == cutlass.Int32(0):
            prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)


zero_rows_kernel.set_name_prefix("glm52_moe_zero_output", remove_cutlass_symbol=True)


@cute.jit
def launch_zero_rows(
    output: cute.Tensor,
    token_count: cutlass.Constexpr,
    signal_pdl: cutlass.Constexpr,
    rows_per_cta: cutlass.Constexpr,
    stream: cuda_driver.CUstream,
) -> None:
    grid_rows = token_count
    if cutlass.const_expr(rows_per_cta > 1):
        grid_rows = (token_count + rows_per_cta - 1) // rows_per_cta
    zero_rows_kernel(output, token_count, signal_pdl, rows_per_cta).launch(
        grid=(grid_rows, 1, 1),
        block=(_THREADS * rows_per_cta, 1, 1),
        stream=stream,
    )


def _as_tensor(tensor: torch.Tensor) -> cute.Tensor:
    return from_dlpack(tensor, assumed_align=16, enable_tvm_ffi=True).mark_layout_dynamic(
        leading_dim=1
    )


def compile_zero_rows(
    output: torch.Tensor,
    *,
    signal_pdl: bool = False,
    rows_per_cta: int | None = None,
) -> ZeroFn:
    """Compile or retrieve an exact-T output-zero variant."""
    if output.ndim != 2 or output.shape[1] != _HIDDEN:
        raise ValueError(f"output must have shape [T,{_HIDDEN}], got {tuple(output.shape)}")
    if not 1 <= output.shape[0] <= 64:
        raise ValueError(f"output T must be in [1,64], got {output.shape[0]}")
    if output.dtype != torch.bfloat16:
        raise TypeError("output must be BF16")
    if not output.is_contiguous():
        raise ValueError("output must be contiguous")
    device = output.device.index
    if device is None:
        device = torch.cuda.current_device()
    token_count = int(output.shape[0])
    rows = _DEFAULT_ROWS_PER_CTA if rows_per_cta is None else rows_per_cta
    if rows not in (1, 2, 4):
        raise ValueError(f"rows_per_cta must be 1, 2, or 4, got {rows}")
    rows = min(rows, token_count)
    key = (device, token_count, signal_pdl, rows)
    if key not in _CACHE:
        cutlass.cuda.initialize_cuda_context(device)
        _CACHE[key] = cute.compile(
            launch_zero_rows,
            _as_tensor(output),
            token_count,
            signal_pdl,
            rows,
            cutlass_torch.current_stream(),
            options="--opt-level 2 --enable-tvm-ffi",
        )
    return _CACHE[key]


def run_zero_rows(
    output: torch.Tensor,
    *,
    signal_pdl: bool = False,
    rows_per_cta: int | None = None,
) -> None:
    """Clear output on the current PyTorch stream without allocating."""
    compiled = compile_zero_rows(output, signal_pdl=signal_pdl, rows_per_cta=rows_per_cta)
    compiled(_as_tensor(output), cutlass_torch.current_stream())
