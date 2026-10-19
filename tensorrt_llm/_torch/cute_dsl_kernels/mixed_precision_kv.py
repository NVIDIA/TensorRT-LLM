# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Paged conversion between native and mixed-precision KV pools."""

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

from tensorrt_llm.bindings import DataType


@cute.jit
def _copy_native(source: cute.Tensor, destination: cute.Tensor) -> None:
    """Copy native KV bits, including FP8 NaNs and signed zero."""
    cute.autovec_copy(
        cute.recast_tensor(source, cutlass.Uint8), cute.recast_tensor(destination, cutlass.Uint8)
    )


class _NVFP4Codec:
    group_size = 16
    packed_group_size = 8

    def __init__(self, dtype: torch.dtype) -> None:
        self.quantize = {
            torch.float16: process_nvfp4_block_half,
            torch.bfloat16: process_nvfp4_block_bfloat,
            torch.float8_e4m3fn: process_nvfp4_block_fp8,
        }[dtype]
        self.fp8 = dtype == torch.float8_e4m3fn
        self.scale = 6.0 if self.fp8 else 1.0

    @cute.jit
    def encode(
        self,
        source: cute.Tensor,
        destination: cute.Tensor,
        scales: cute.Tensor,
        group: cutlass.Int64,
    ) -> None:
        scale, packed = self.quantize(
            source, cutlass.Int32(group * self.group_size), cutlass.Float32(self.scale)
        )
        st_global_u64(
            get_ptr_as_int64(destination, cutlass.Int32(group * self.packed_group_size)), packed
        )
        scales[group] = scale

    @cute.jit
    def decode(
        self,
        source: cute.Tensor,
        scales: cute.Tensor,
        group: cutlass.Int64,
        destination: cute.Tensor,
    ) -> None:
        packed = cute.make_rmem_tensor((self.packed_group_size,), cutlass.Uint8)
        cute.autovec_copy(cute.local_tile(source, (self.packed_group_size,), (group,)), packed)
        values = cute.recast_tensor(packed, cutlass.Float4E2M1FN).load().to(cutlass.Float32)
        scale_bits = cute.make_rmem_tensor((1,), cutlass.Uint8)
        scale_bits[0] = scales[group]
        scale = cute.recast_tensor(scale_bits, cutlass.Float8E4M3FN)[0].to(cutlass.Float32)
        values = values * scale
        if cutlass.const_expr(self.fp8):
            values = (values * (1.0 / 6.0)).to(cutlass.BFloat16).to(cutlass.Float32)
            values = cute.where(values > 448, cute.full_like(values, 448), values)
            values = cute.where(values < -448, cute.full_like(values, -448), values)
        restored = cute.make_rmem_tensor((self.group_size,), destination.element_type)
        restored.store(values.to(destination.element_type))
        cute.autovec_copy(restored, destination)


class _PagedMixedPrecisionKV:
    def __init__(self, write: bool, capacity: int, codec: object) -> None:
        self.write = write
        self.capacity = capacity
        self.codec = codec

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
        groups = (
            self.capacity * pool.shape[1] * pool.shape[3] * (pool.shape[4] // self.codec.group_size)
        )
        self.convert_paged_kv(pool, history, scales, high, tables, lengths).launch(
            grid=((groups + 127) // 128, lengths.shape[1], 1), block=(128, 1, 1), stream=stream
        )

    @cute.kernel
    def convert_paged_kv(
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
        groups = pool.shape[4] // self.codec.group_size
        group = index % groups
        row = index // groups
        head = row % pool.shape[3]
        kv = row // pool.shape[3] % pool.shape[1]
        token = row // (pool.shape[1] * pool.shape[3])
        pages_per_request = pool.shape[0] // lengths.shape[1]
        position = token
        if cutlass.const_expr(self.write):
            position += lengths[0, request]
        if token < self.capacity:
            view_page = request * pages_per_request + position // pool.shape[2]
            view_row = pool[view_page, kv, position % pool.shape[2], head, None]
            view_tile = cute.local_tile(view_row, (self.codec.group_size,), (group,))
            page = position // history.shape[2]
            offset = position % history.shape[2]
            valid = position < lengths[0, request]
            if cutlass.const_expr(self.write):
                valid = position < lengths[1, request]
            if valid:
                low_page = cutlass.Int64(tables[0, request, page])
                scale_page = cutlass.Int64(tables[1, request, page])
                high_page = cutlass.Int64(tables[2, request, page])
                if cutlass.const_expr(self.write):
                    self.codec.encode(
                        view_row,
                        history[low_page, kv, offset, head, None],
                        scales[scale_page, kv, offset, head, None],
                        group,
                    )
                    if high_page >= 0:
                        high_row = high[high_page, kv, offset, head, None]
                        _copy_native(
                            view_tile, cute.local_tile(high_row, (self.codec.group_size,), (group,))
                        )
                else:
                    if high_page >= 0:
                        high_row = high[high_page, kv, offset, head, None]
                        _copy_native(
                            cute.local_tile(high_row, (self.codec.group_size,), (group,)), view_tile
                        )
                    else:
                        self.codec.decode(
                            history[low_page, kv, offset, head, None],
                            scales[scale_page, kv, offset, head, None],
                            group,
                            view_tile,
                        )
            elif cutlass.const_expr(not self.write):
                cute.recast_tensor(view_tile, cutlass.Uint8).fill(0)


_LOW_PRECISION_CODECS = {DataType.NVFP4: _NVFP4Codec}
_compiled_kernels = {}


def convert_mixed_precision_kv(
    write: bool,
    capacity: int,
    pool: torch.Tensor,
    history: torch.Tensor,
    scales: torch.Tensor,
    high: torch.Tensor,
    page_tables: torch.Tensor,
    lengths: torch.Tensor,
    layout: str,
    *,
    low_dtype: DataType,
) -> None:
    """Convert up to capacity tokens per request between native and mixed KV pools."""
    if capacity == 0 or lengths.shape[1] == 0:
        return
    pools = (pool, history.view(torch.uint8), scales.view(torch.uint8), high)
    if layout == "HND":
        pools = tuple(t.transpose(2, 3) for t in pools)
    tensors = (*pools, page_tables, lengths)
    key = (
        low_dtype,
        write,
        capacity,
        tuple((t.dtype, t.shape, t.stride(), t.device.index) for t in tensors),
    )
    if key not in _compiled_kernels:
        try:
            codec = _LOW_PRECISION_CODECS[low_dtype](pool.dtype)
        except KeyError:
            raise NotImplementedError(
                f"Mixed KV conversion between {low_dtype} and {pool.dtype} is not implemented"
            ) from None
        with BaseDSL.enable_pyir():
            _compiled_kernels[key] = cute.compile(
                _PagedMixedPrecisionKV(write, capacity, codec),
                *(from_dlpack(t) for t in tensors),
                make_fake_stream(use_tvm_ffi_env_stream=True),
                options="--enable-tvm-ffi",
            )
    _compiled_kernels[key](*tensors)
