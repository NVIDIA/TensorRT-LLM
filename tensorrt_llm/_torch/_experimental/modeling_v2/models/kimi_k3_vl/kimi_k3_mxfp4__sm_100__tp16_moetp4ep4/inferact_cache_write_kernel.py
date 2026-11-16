# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Isolated B1/K7 Inferact MLA RMSNorm/RoPE producer with direct paged stores.

The arithmetic follows DSparkRMSNormRoPEKernel at TensorRT-LLM revision
adab54f2b6d0bd1d98d5fa73d535ca5d0773ea65: RMSNorm and its weight cover only
the first 512 columns; adjacent-pair RoPE consumes the remaining 64 raw
columns and the caller's existing FP32 YaRN frequencies. Only the output
address changes. This prototype does not modify the stock kernel or model.

The caller owns the device-value contract: every page is in [0, cache.shape[0]),
every offset is in [0, 64), and the seven (page, offset) pairs are distinct.
Structural admission does not read these device values or synchronize CUDA.
All seven rows are written, including speculative rows not ultimately accepted.
"""

import functools
import math

import cutlass
import cutlass.cute as cute
import torch

try:
    from cuda.bindings import driver as cuda
except ImportError:
    from cuda import cuda

__all__ = ["is_compiled", "is_supported", "paged_cache_write"]

_ROWS = 7
_LATENT = 512
_ROPE = 64
_WIDTH = _LATENT + _ROPE
_PAGE_SIZE = 64
_ALIGNMENT = 16
_COMPILED_KEYS: set[tuple[float, int]] = set()


class _PagedCacheWriteKernel:
    """Use the stock one-warp reduction and BF16 rounding for each of seven rows."""

    num_threads = cute.arch.WARP_SIZE

    def __init__(self, eps: float) -> None:
        self.eps = eps
        self.norm_dim = _LATENT
        self.nope_dim = _LATENT
        self.norm_elements_per_thread = _LATENT // self.num_threads
        self.nope_elements_per_thread = _LATENT // self.num_threads
        self.pairs_per_thread = (_ROPE // 2) // self.num_threads

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        freqs: cute.Tensor,
        cache: cute.Tensor,
        pages: cute.Tensor,
        offsets: cute.Tensor,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(x, weight, freqs, cache, pages, offsets).launch(
            grid=[x.shape[0], 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        freqs: cute.Tensor,
        cache: cute.Tensor,
        pages: cute.Tensor,
        offsets: cute.Tensor,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        page = pages[row]
        offset = offsets[row]

        # Keep the stock accumulation, reduction, division, and rsqrt order.
        sum_sq = cutlass.Float32(0.0)
        for item in cutlass.range_constexpr(self.norm_elements_per_thread):
            dim = tidx + item * self.num_threads
            value = cutlass.Float32(x[row, dim])
            sum_sq += value * value
        sum_sq = cute.arch.warp_reduction_sum(sum_sq)
        inverse_rms = cute.math.rsqrt(sum_sq / self.norm_dim + self.eps)

        for item in cutlass.range_constexpr(self.nope_elements_per_thread):
            dim = tidx + item * self.num_threads
            value = cutlass.Float32(x[row, dim]) * inverse_rms
            value *= cutlass.Float32(weight[dim])
            cache[page, offset, dim] = value.to(cache.element_type)

        for item in cutlass.range_constexpr(self.pairs_per_thread):
            pair = tidx + item * self.num_threads
            real_dim = self.nope_dim + pair * 2
            imag_dim = real_dim + 1
            real = cutlass.Float32(x[row, real_dim])
            imag = cutlass.Float32(x[row, imag_dim])
            cos = cutlass.Float32(freqs[row, pair, 0])
            sin = cutlass.Float32(freqs[row, pair, 1])
            cache[page, offset, real_dim] = (real * cos - imag * sin).to(cache.element_type)
            cache[page, offset, imag_dim] = (imag * cos + real * sin).to(cache.element_type)


def _byte_span(tensor: torch.Tensor) -> tuple[int, int] | None:
    """Return a conservative positive-stride span, including gaps, inside backing storage."""
    strides = tensor.stride()
    if any(size <= 0 for size in tensor.shape) or any(stride <= 0 for stride in strides):
        return None
    last_element = sum((size - 1) * stride for size, stride in zip(tensor.shape, strides))
    if last_element >= 2**63 - 1:
        return None
    start = tensor.data_ptr()
    stop = start + (last_element + 1) * tensor.element_size()
    storage = tensor.untyped_storage()
    storage_start = storage.data_ptr()
    if start < storage_start or stop > storage_start + storage.nbytes():
        return None
    return start, stop


@functools.cache
def _is_sm100(device_index: int) -> bool:
    return torch.cuda.get_device_capability(device_index) == (10, 0)


def is_supported(
    x: torch.Tensor,
    weight: torch.Tensor,
    freqs: torch.Tensor,
    cache: torch.Tensor,
    pages: torch.Tensor,
    offsets: torch.Tensor,
    eps: float,
) -> bool:
    """Check the fixed structural contract without reading device tensor values.

    Args:
        x: Contiguous BF16 CUDA tensor [7, 576].
        weight: Contiguous BF16 CUDA tensor [512].
        freqs: Contiguous FP32 CUDA tensor [7, 32, 2], with existing YaRN scaling.
        cache: BF16 CUDA tensor [P, 64, 576] with contiguous rows and a positive,
            nonoverlapping page stride. Nonzero aligned storage offsets are allowed.
        pages: Contiguous int64 CUDA tensor [7]; caller validates page values.
        offsets: Contiguous int64 CUDA tensor [7]; caller validates offset values
            and uniqueness of the seven destination pairs.
        eps: Finite positive RMSNorm epsilon, specialized at compilation.

    All tensors must share an SM100 device and have 16-byte-aligned base pointers.
    The entire cache address span must be disjoint from every input address span.
    No allocation, page-table mutation, CUDA synchronization, or value check occurs.
    """
    tensors = (x, weight, freqs, cache, pages, offsets)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        return False
    if not isinstance(eps, (float, int)) or isinstance(eps, bool):
        return False
    if not math.isfinite(eps) or eps <= 0:
        return False
    if not all(tensor.layout == torch.strided and tensor.is_cuda for tensor in tensors):
        return False
    if not all(tensor.device == x.device for tensor in tensors):
        return False
    if x.device.index is None or not _is_sm100(x.device.index):
        return False
    if any(tensor.is_conj() or tensor.is_neg() for tensor in tensors):
        return False
    if x.dtype != torch.bfloat16 or tuple(x.shape) != (_ROWS, _WIDTH):
        return False
    if weight.dtype != torch.bfloat16 or tuple(weight.shape) != (_LATENT,):
        return False
    if freqs.dtype != torch.float32 or tuple(freqs.shape) != (_ROWS, _ROPE // 2, 2):
        return False
    if cache.dtype != torch.bfloat16 or cache.ndim != 3:
        return False
    if cache.shape[0] <= 0 or tuple(cache.shape[1:]) != (_PAGE_SIZE, _WIDTH):
        return False
    if cache.stride(1) != _WIDTH or cache.stride(2) != 1 or cache.stride(0) <= 0:
        return False
    if cache.shape[0] > 1 and cache.stride(0) < _PAGE_SIZE * _WIDTH:
        return False
    if any(
        tensor.dtype != torch.int64 or tuple(tensor.shape) != (_ROWS,)
        for tensor in (pages, offsets)
    ):
        return False
    if not all(tensor.is_contiguous() for tensor in (x, weight, freqs, pages, offsets)):
        return False
    if any(tensor.data_ptr() % _ALIGNMENT != 0 for tensor in tensors):
        return False
    spans = [_byte_span(tensor) for tensor in tensors]
    if any(span is None for span in spans):
        return False
    cache_start, cache_stop = spans[3]
    for index in (0, 1, 2, 4, 5):
        start, stop = spans[index]
        if start < cache_stop and cache_start < stop:
            return False
    return True


def is_compiled(eps: float, device_index: int) -> bool:
    """Read readiness for this exact epsilon/device key; never query CUDA or compile."""
    return (float(eps), device_index) in _COMPILED_KEYS


@functools.cache
def _compile(eps: float, device_index: int):
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "Paged RMSNorm/RoPE cache-write must be warmed before CUDA graph capture"
        )
    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (_ROWS, _WIDTH), stride_order=(1, 0)
    )
    weight_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (_LATENT,), stride_order=(0,)
    )
    freqs_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (_ROWS, _ROPE // 2, 2), stride_order=(2, 1, 0)
    )
    cache_fake = cute.runtime.make_fake_tensor(
        cutlass.BFloat16,
        (cute.sym_int(), _PAGE_SIZE, _WIDTH),
        stride=(cute.sym_int64(), _WIDTH, 1),
    )
    pages_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int64, (_ROWS,), stride_order=(0,)
    )
    offsets_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int64, (_ROWS,), stride_order=(0,)
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    compiled = cute.compile(
        _PagedCacheWriteKernel(eps),
        x_fake,
        weight_fake,
        freqs_fake,
        cache_fake,
        pages_fake,
        offsets_fake,
        stream_fake,
        options="--opt-level 2 --enable-tvm-ffi",
    )
    _COMPILED_KEYS.add((eps, device_index))
    return compiled


@torch.library.custom_op(
    "trtllm::inferact_cachewrite1_paged_cache_write",
    mutates_args=("cache",),
    device_types="cuda",
)
def paged_cache_write(
    x: torch.Tensor,
    weight: torch.Tensor,
    freqs: torch.Tensor,
    cache: torch.Tensor,
    pages: torch.Tensor,
    offsets: torch.Tensor,
    eps: float,
) -> None:
    """Normalize/rotate seven rows and write their distinct, valid paged cache destinations.

    Structural mismatches raise ValueError; the caller chooses stock fallback before
    calling. Warm one invocation outside capture for each epsilon/device key. There
    are no temporary output tensors, hidden index conversions, or accepted-row masks.
    """
    if not is_supported(x, weight, freqs, cache, pages, offsets, eps):
        raise ValueError("Unsupported fixed B1/K7 SM100 paged RMSNorm/RoPE cache-write operands")
    with torch.cuda.device(x.device):
        compiled = _compile(float(eps), x.device.index)
        compiled(x, weight, freqs, cache, pages, offsets)


@torch.library.register_fake("trtllm::inferact_cachewrite1_paged_cache_write")
def _paged_cache_write_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    freqs: torch.Tensor,
    cache: torch.Tensor,
    pages: torch.Tensor,
    offsets: torch.Tensor,
    eps: float,
) -> None:
    return None
