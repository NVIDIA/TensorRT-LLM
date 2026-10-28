# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Paged conversion between native and mixed-precision KV pools."""

import cutlass
import cutlass.cute as cute
import torch
from cutlass.base_dsl.dsl import BaseDSL
from cutlass.cute.runtime import from_dlpack, make_fake_stream

# Internal FlashInfer symbols; requirements.txt pins flashinfer-python==0.6.18.
# A bump must keep these names or update this module. The CPU-only import test
# in test_mixed_precision_backends.py fails fast if they disappear.
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
    """NVFP4 (E2M1 values, FP8 scale per 16 elements) record codec for one head."""

    group_size = 16
    packed_group_size = 8

    @classmethod
    def row_bytes(cls, head_dim: int) -> int:
        """Pack one head's data and scales, aligned for vector stores."""
        if head_dim % cls.group_size:
            raise ValueError("NVFP4 mixed-precision KV requires head_dim divisible by 16")
        size = head_dim // cls.group_size * (cls.packed_group_size + 1)
        return (size + 7) // 8 * 8

    def __init__(self, dtype: torch.dtype, head_dim: int) -> None:
        self.quantize = {
            torch.float16: process_nvfp4_block_half,
            torch.bfloat16: process_nvfp4_block_bfloat,
            torch.float8_e4m3fn: process_nvfp4_block_fp8,
        }[dtype]
        self.fp8 = dtype == torch.float8_e4m3fn
        self.scale = 6.0 if self.fp8 else 1.0
        self.scale_offset = head_dim // self.group_size * self.packed_group_size

    @cute.jit
    def encode(
        self,
        source: cute.Tensor,
        destination: cute.Tensor,
        group: cutlass.Int64,
    ) -> None:
        """Quantize one 16-element group of ``source`` into ``destination``."""
        scale, packed = self.quantize(
            source, cutlass.Int32(group * self.group_size), cutlass.Float32(self.scale)
        )
        st_global_u64(
            get_ptr_as_int64(destination, cutlass.Int32(group * self.packed_group_size)), packed
        )
        destination[self.scale_offset + group] = scale

    @cute.jit
    def decode(
        self,
        source: cute.Tensor,
        group: cutlass.Int64,
        destination: cute.Tensor,
    ) -> None:
        """Dequantize one 16-element group of ``source`` into ``destination``."""
        packed = cute.make_rmem_tensor((self.packed_group_size,), cutlass.Uint8)
        cute.autovec_copy(cute.local_tile(source, (self.packed_group_size,), (group,)), packed)
        values = cute.recast_tensor(packed, cutlass.Float4E2M1FN).load().to(cutlass.Float32)
        scale_bits = cute.make_rmem_tensor((1,), cutlass.Uint8)
        scale_bits[0] = source[self.scale_offset + group]
        scale = cute.recast_tensor(scale_bits, cutlass.Float8E4M3FN)[0].to(cutlass.Float32)
        values = values * scale
        if cutlass.const_expr(self.fp8):
            values = (values * (1.0 / 6.0)).to(cutlass.BFloat16).to(cutlass.Float32)
            values = cute.where(values > 448, cute.full_like(values, 448), values)
            values = cute.where(values < -448, cute.full_like(values, -448), values)
        restored = cute.make_rmem_tensor((self.group_size,), destination.element_type)
        restored.store(values.to(destination.element_type))
        cute.autovec_copy(restored, destination)


class _NativeCodec:
    """History pool at the active dtype: 16-element groups are copied bit for bit.

    Cold-page mixed precision keeps the GPU history pool at the runtime KV type and
    quantizes only its cold pages, so the staging conversion is a plain copy.
    """

    group_size = 16
    element_bytes = 1

    @classmethod
    def row_bytes(cls, head_dim: int) -> int:
        """One head's record is its native row."""
        if head_dim % cls.group_size:
            raise ValueError("Mixed-precision KV requires head_dim divisible by 16")
        return head_dim * cls.element_bytes

    def __init__(self, dtype: torch.dtype, head_dim: int) -> None:
        if torch.empty((), dtype=dtype).element_size() != self.element_bytes:
            raise ValueError(f"{dtype} is not a {self.element_bytes}-byte KV element")
        self.group_bytes = self.group_size * self.element_bytes

    @cute.jit
    def encode(
        self,
        source: cute.Tensor,
        destination: cute.Tensor,
        group: cutlass.Int64,
    ) -> None:
        """Copy one 16-element group of ``source`` into ``destination``."""
        _copy_native(
            cute.local_tile(source, (self.group_size,), (group,)),
            cute.local_tile(destination, (self.group_bytes,), (group,)),
        )

    @cute.jit
    def decode(
        self,
        source: cute.Tensor,
        group: cutlass.Int64,
        destination: cute.Tensor,
    ) -> None:
        """Copy one 16-element group of ``source`` into ``destination``."""
        _copy_native(cute.local_tile(source, (self.group_bytes,), (group,)), destination)


class _NativeCodec1B(_NativeCodec):
    element_bytes = 1


class _NativeCodec2B(_NativeCodec):
    element_bytes = 2


class _PagedMixedPrecisionKV:
    """Paged gather (read) or scatter (write) between staging and the two pools."""

    def __init__(self, write: bool, codec: object) -> None:
        self.write = write
        self.codec = codec

    # Tokens covered by one pass of a request's threads; longer requests loop. The grid
    # therefore does not grow with the capacity, which a CUDA graph freezes at the longest
    # sequence: a 128K capacity would otherwise launch 131072 blocks per request and layer.
    tokens_per_pass = 4096

    @cute.jit
    def __call__(
        self,
        pool: cute.Tensor,
        low: cute.Tensor,
        high: cute.Tensor,
        tables: cute.Tensor,
        lengths: cute.Tensor,
        capacity: cutlass.Int32,
        span: cutlass.Int32,
        stream,
    ):
        """Launch one thread per (token, kv, head, 16-element group) of a ``span``-token pass."""
        groups = (
            cutlass.Int64(span)
            * pool.shape[1]
            * pool.shape[3]
            * (pool.shape[4] // self.codec.group_size)
        )
        self.convert_paged_kv(pool, low, high, tables, lengths, capacity).launch(
            grid=((groups + 127) // 128, lengths.shape[1], 1), block=(128, 1, 1), stream=stream
        )

    @cute.kernel
    def convert_paged_kv(
        self,
        pool: cute.Tensor,
        low: cute.Tensor,
        high: cute.Tensor,
        tables: cute.Tensor,
        lengths: cute.Tensor,
        capacity: cutlass.Int32,
    ):
        """Move this request's 16-element groups between staging rows and pool pages."""
        thread, _, _ = cute.arch.thread_idx()
        block, request, _ = cute.arch.block_idx()
        blocks, _, _ = cute.arch.grid_dim()
        request = cutlass.Int64(request)
        groups = pool.shape[4] // self.codec.group_size
        rows_per_token = pool.shape[1] * pool.shape[3]
        pages_per_request = pool.shape[0] // lengths.shape[1]
        start = lengths[0, request]
        # Only the tokens this request really has are visited, read from device memory so
        # a replayed graph follows the current lengths.
        count = start
        if cutlass.const_expr(self.write):
            count = lengths[1, request] - start
        count = cutlass.min(count, capacity)
        total = count * (rows_per_token * groups)
        stride = cutlass.Int32(blocks) * 128
        for index in cutlass.range(cutlass.Int32(block) * 128 + thread, total, stride):
            group = index % groups
            row = index // groups
            head = row % pool.shape[3]
            kv = row // pool.shape[3] % pool.shape[1]
            position = row // rows_per_token
            if cutlass.const_expr(self.write):
                position += start
            view_page = request * pages_per_request + position // pool.shape[2]
            view_row = pool[view_page, kv, position % pool.shape[2], head, None]
            view_tile = cute.local_tile(view_row, (self.codec.group_size,), (group,))
            page = position // low.shape[2]
            offset = position % low.shape[2]
            low_page = cutlass.Int64(tables[0, request, page])
            high_page = cutlass.Int64(tables[1, request, page])
            if cutlass.const_expr(self.write):
                self.codec.encode(
                    view_row,
                    low[low_page, kv, offset, head, None],
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
                        low[low_page, kv, offset, head, None],
                        group,
                        view_tile,
                    )


_LOW_PRECISION_CODECS = {
    DataType.NVFP4: _NVFP4Codec,
    DataType.FP8: _NativeCodec1B,
    DataType.HALF: _NativeCodec2B,
    DataType.BF16: _NativeCodec2B,
}
_compiled_kernels = {}


def get_mixed_precision_kv_row_bytes(low_dtype: DataType, head_dim: int) -> int:
    """Return the complete encoded size of one token/head record."""
    return _LOW_PRECISION_CODECS[low_dtype].row_bytes(head_dim)


def convert_mixed_precision_kv(
    write: bool,
    capacity: int,
    pool: torch.Tensor,
    low: torch.Tensor,
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
    pools = (pool, low.view(torch.uint8), high)
    if layout == "HND":
        pools = tuple(t.transpose(2, 3) for t in pools)
    tensors = (*pools, page_tables, lengths)
    key = (
        low_dtype,
        write,
        tuple((t.dtype, t.shape[1:], t.stride(), t.device.index) for t in pools),
        tuple((t.dtype, t.shape[0], t.ndim, t.device.index) for t in (page_tables, lengths)),
    )
    if key not in _compiled_kernels:
        try:
            codec = _LOW_PRECISION_CODECS[low_dtype](pool.dtype, pool.shape[-1])
        except KeyError:
            raise NotImplementedError(
                f"Mixed KV conversion between {low_dtype} and {pool.dtype} is not implemented"
            ) from None
        fake_tensors = [
            cute.runtime.make_fake_tensor(
                from_dlpack(t).element_type,
                (cute.sym_int64(), *t.shape[1:]),
                t.stride(),
                assumed_align=16,
            )
            for t in pools
        ]
        # Metadata slices retain their reserved-capacity strides as batches change.
        for t in (page_tables, lengths):
            fake_tensors.append(
                cute.runtime.make_fake_tensor(
                    from_dlpack(t).element_type,
                    (t.shape[0], *(cute.sym_int64() for _ in t.shape[1:])),
                    tuple(cute.sym_int64() for _ in t.stride()),
                    assumed_align=t.element_size(),
                )
            )
        with BaseDSL.enable_pyir():
            _compiled_kernels[key] = cute.compile(
                _PagedMixedPrecisionKV(write, codec),
                *fake_tensors,
                cutlass.Int32(capacity),
                cutlass.Int32(capacity),
                make_fake_stream(use_tvm_ffi_env_stream=True),
                options="--enable-tvm-ffi",
            )
    span = min(capacity, _PagedMixedPrecisionKV.tokens_per_pass)
    _compiled_kernels[key](*tensors, capacity, span)
