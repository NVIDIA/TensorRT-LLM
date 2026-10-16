# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Paged NVFP4 conversion with an optional native-precision KV pool."""

import cutlass
import cutlass.cute as cute
import torch
from cutlass.base_dsl.dsl import BaseDSL
from cutlass.cute.runtime import from_dlpack, make_fake_stream
from flashinfer.cute_dsl.fp4_common import get_ptr_as_int64, st_global_u64
from flashinfer.quantization.quantization_cute_dsl_utils import (
    process_nvfp4_block_bfloat,
    process_nvfp4_block_fp8,
    process_nvfp4_block_half,
)


@cute.jit
def _copy_native(source: cute.Tensor, destination: cute.Tensor) -> None:
    """Copy native KV bits, including FP8 NaNs and signed zero."""
    bits = cutlass.Uint8 if cutlass.const_expr(source.element_type.width == 8) else cutlass.Uint16
    cute.autovec_copy(cute.recast_tensor(source, bits), cute.recast_tensor(destination, bits))


class _PagedMixedPrecisionKV:
    def __init__(self, write: bool, capacity: int):
        self.write = write
        self.capacity = capacity

    @cute.jit
    def __call__(
        self,
        pool: cute.Tensor,
        history: cute.Tensor,
        scales: cute.Tensor,
        high: cute.Tensor,
        tables: cute.Tensor,
        lengths: cute.Tensor,
        stream,
    ):
        groups = self.capacity * pool.shape[1] * pool.shape[3] * (pool.shape[4] // 16)
        self.kernel(pool, history, scales, high, tables, lengths).launch(
            grid=((groups + 127) // 128, lengths.shape[1], 1), block=(128, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        pool: cute.Tensor,
        history: cute.Tensor,
        scales: cute.Tensor,
        high: cute.Tensor,
        tables: cute.Tensor,
        lengths: cute.Tensor,
    ):
        thread, _, _ = cute.arch.thread_idx()
        block, request, _ = cute.arch.block_idx()
        index = cutlass.Int64(block) * 128 + thread
        request = cutlass.Int64(request)
        groups = pool.shape[4] // 16
        group = index % groups
        row = index // groups
        head = row % pool.shape[3]
        kv = row // pool.shape[3] % pool.shape[1]
        token = row // (pool.shape[1] * pool.shape[3])
        pages_per_request = pool.shape[0] // lengths.shape[1]
        position = token
        if cutlass.const_expr(self.write):
            position += lengths[0, request]
        if token < self.capacity and position < pages_per_request * pool.shape[2]:
            view_page = request * pages_per_request + position // pool.shape[2]
            view_row = pool[view_page, kv, position % pool.shape[2], head, None]
            view_tile = cute.local_tile(view_row, (16,), (group,))
            page = position // history.shape[2]
            offset = position % history.shape[2]
            valid = position < lengths[0, request]
            if cutlass.const_expr(self.write):
                valid = position < lengths[1, request]
            if valid and page < tables.shape[2]:
                low_page = cutlass.Int64(tables[0, request, page])
                scale_page = cutlass.Int64(tables[1, request, page])
                high_page = cutlass.Int64(tables[2, request, page])
                if cutlass.const_expr(self.write):
                    if (
                        low_page >= 0
                        and low_page < history.shape[0]
                        and scale_page >= 0
                        and scale_page < scales.shape[0]
                    ):
                        if cutlass.const_expr(pool.element_type == cutlass.Float16):
                            scale, packed = process_nvfp4_block_half(
                                view_row, cutlass.Int32(group * 16), cutlass.Float32(1.0)
                            )
                        elif cutlass.const_expr(pool.element_type == cutlass.BFloat16):
                            scale, packed = process_nvfp4_block_bfloat(
                                view_row, cutlass.Int32(group * 16), cutlass.Float32(1.0)
                            )
                        else:
                            scale, packed = process_nvfp4_block_fp8(
                                view_row, cutlass.Int32(group * 16), cutlass.Float32(6.0)
                            )
                        low_row = history[low_page, kv, offset, head, None]
                        st_global_u64(get_ptr_as_int64(low_row, cutlass.Int32(group * 8)), packed)
                        scales[scale_page, kv, offset, head, group] = scale
                    if high_page >= 0 and high_page < high.shape[0]:
                        high_row = high[high_page, kv, offset, head, None]
                        _copy_native(view_tile, cute.local_tile(high_row, (16,), (group,)))
                else:
                    if high_page >= 0 and high_page < high.shape[0]:
                        high_row = high[high_page, kv, offset, head, None]
                        _copy_native(cute.local_tile(high_row, (16,), (group,)), view_tile)
                    elif (
                        low_page >= 0
                        and low_page < history.shape[0]
                        and scale_page >= 0
                        and scale_page < scales.shape[0]
                    ):
                        low_row = history[low_page, kv, offset, head, None]
                        packed = cute.make_rmem_tensor((8,), cutlass.Uint8)
                        cute.autovec_copy(cute.local_tile(low_row, (8,), (group,)), packed)
                        values = (
                            cute.recast_tensor(packed, cutlass.Float4E2M1FN)
                            .load()
                            .to(cutlass.Float32)
                        )
                        scale_bits = cute.make_rmem_tensor((1,), cutlass.Uint8)
                        scale_bits[0] = scales[scale_page, kv, offset, head, group]
                        scale = cute.recast_tensor(scale_bits, cutlass.Float8E4M3FN)[0].to(
                            cutlass.Float32
                        )
                        values = values * scale
                        if cutlass.const_expr(pool.element_type == cutlass.Float8E4M3FN):
                            values = (values * (1.0 / 6.0)).to(cutlass.BFloat16).to(cutlass.Float32)
                            values = cute.where(values > 448, cute.full_like(values, 448), values)
                            values = cute.where(values < -448, cute.full_like(values, -448), values)
                        restored = cute.make_rmem_tensor((16,), pool.element_type)
                        restored.store(values.to(pool.element_type))
                        cute.autovec_copy(restored, view_tile)
                    else:
                        cute.recast_tensor(view_tile, cutlass.Uint8).fill(0)
            elif cutlass.const_expr(not self.write):
                cute.recast_tensor(view_tile, cutlass.Uint8).fill(0)


_compiled_kernels = {}


def _run(
    write: bool,
    capacity: int,
    pool: torch.Tensor,
    history: torch.Tensor,
    scales: torch.Tensor,
    high: torch.Tensor,
    page_tables: torch.Tensor,
    lengths: torch.Tensor,
    layout: str,
) -> None:
    if capacity == 0 or lengths.shape[1] == 0:
        return
    pools = (pool, history.view(torch.uint8), scales.view(torch.uint8), high)
    if layout == "HND":
        pools = tuple(t.transpose(2, 3) for t in pools)
    tensors = (*pools, page_tables, lengths)
    key = (write, capacity, tuple((t.dtype, t.shape, t.stride(), t.device.index) for t in tensors))
    if key not in _compiled_kernels:
        with BaseDSL.enable_pyir():
            _compiled_kernels[key] = cute.compile(
                _PagedMixedPrecisionKV(write, capacity),
                *(from_dlpack(t) for t in tensors),
                make_fake_stream(use_tvm_ffi_env_stream=True),
                options="--enable-tvm-ffi",
            )
    _compiled_kernels[key](*tensors)


def materialize_mixed_precision_kv(
    pool: torch.Tensor,
    history: torch.Tensor,
    scales: torch.Tensor,
    high: torch.Tensor,
    page_tables: torch.Tensor,
    lengths: torch.Tensor,
    layout: str,
) -> None:
    """Restore cached KV from native high pages or NVFP4 low pages.

    Pools are 5D HND/NHD views; the native view may use a different page size.
    CUDA int32 page_tables[3, B, P] contains independent low/scale/high page IDs
    (missing high pages are -1); lengths[2, B] contains cached/total lengths.
    NVFP4 global scale is 6 for FP8 native KV and 1 for FP16/BF16.
    """
    block_size = pool.shape[3 if layout == "HND" else 2]
    capacity = pool.shape[0] // lengths.shape[1] * block_size if lengths.shape[1] else 0
    _run(False, capacity, pool, history, scales, high, page_tables, lengths, layout)


def writeback_mixed_precision_kv(
    pool: torch.Tensor,
    history: torch.Tensor,
    scales: torch.Tensor,
    high: torch.Tensor,
    page_tables: torch.Tensor,
    lengths: torch.Tensor,
    max_new_tokens_per_request: int,
    layout: str,
) -> None:
    """Write every new token to low/scales and any retained high page.

    Tensor layouts and scales match materialize_mixed_precision_kv. The launch
    bound max_new_tokens_per_request must cover each request throughout replay.
    """
    _run(
        True, max_new_tokens_per_request, pool, history, scales, high, page_tables, lengths, layout
    )
