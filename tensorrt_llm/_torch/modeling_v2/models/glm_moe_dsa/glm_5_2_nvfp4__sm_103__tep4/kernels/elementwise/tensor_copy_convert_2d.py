# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strided 2D tensor copy with dtype conversion: ``dst.copy_(src)`` for the
GLM-5.2 serving glue.

The serving forward moves data between layouts and dtypes with
``Tensor.copy_`` -- fp32 staging of a BF16 activation for a TF32 GEMM, BF16
keys out of a strided FP32 projection, int32 / int64 index casts, slices of
token tables -- each one a separate PyTorch elementwise launch.  This kernel
takes any pair of same-shape operands that are 2D row blocks after collapsing
their leading dimensions (inner dimension contiguous, one row pitch each) and
writes ``dst[r, c] = convert(src[r, c])``:

* same dtype: bitwise (every 1-, 2-, 4- and 8-byte type, as raw integers);
* BF16 / FP16 -> FP32: exact;  FP32 -> BF16 / FP16: round-to-nearest-even,
  as PyTorch converts;
* int32 -> int64: sign extension;  int64 -> int32: truncation to the low 32
  bits, as PyTorch converts.

One CTA of 256 threads covers ``_COLS_PER_CTA`` columns of one row, each
thread four elements 256 apart, so every warp access is coalesced whatever
the element size.  Rows and column tiles are flattened into a 1D grid, which
leaves no limit on the row count.

"""

from collections.abc import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

_THREADS = 256
_PER_THREAD = 4
_COLS_PER_CTA = _THREADS * _PER_THREAD

# torch dtype -> (DSL type, size); same-dtype copies of the 1-byte types run as
# raw uint8 so fp8 / bool / int8 need no type of their own.
_DSL = {
    torch.bfloat16: cutlass.BFloat16,
    torch.float16: cutlass.Float16,
    torch.float32: cutlass.Float32,
    torch.int32: cutlass.Int32,
    torch.int64: cutlass.Int64,
    torch.uint8: cutlass.Uint8,
    torch.int16: cutlass.Int16,
}
_RAW_BY_SIZE = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}
_CONVERSIONS = {
    (torch.bfloat16, torch.float32),
    (torch.float16, torch.float32),
    (torch.float32, torch.bfloat16),
    (torch.float32, torch.float16),
    (torch.int32, torch.int64),
    (torch.int64, torch.int32),
}

CompiledKernel = Callable[..., object]
_COMPILED: dict[tuple[torch.dtype, torch.dtype], CompiledKernel] = {}


@cute.kernel
def _copy_kernel(
    src: cute.Tensor,
    dst: cute.Tensor,
    col_tiles: cutlass.Int32,
    DST_T: cutlass.Constexpr,
) -> None:
    bidx, _, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    row = bidx // col_tiles
    tile = bidx - row * col_tiles
    cols = cute.size(dst, mode=[1])
    base = tile * _COLS_PER_CTA + tidx
    for k in cutlass.range_constexpr(_PER_THREAD):
        col = base + k * _THREADS
        if col < cols:
            dst[row, col] = src[row, col].to(DST_T)


@cute.jit
def _launch(
    src: cute.Tensor,
    dst: cute.Tensor,
    col_tiles: cutlass.Int32,
    num_ctas: cutlass.Int32,
    stream: cuda.CUstream,
    DST_T: cutlass.Constexpr,
) -> None:
    _copy_kernel(src, dst, col_tiles, DST_T).launch(
        grid=(num_ctas, 1, 1), block=(_THREADS, 1, 1), stream=stream
    )


def _fake(dtype) -> object:
    return make_fake_tensor(
        dtype, (cute.sym_int32(), cute.sym_int32()), stride=(cute.sym_int64(), 1), assumed_align=1
    )


def _compiled(src_dtype: torch.dtype, dst_dtype: torch.dtype) -> CompiledKernel:
    key = (src_dtype, dst_dtype)
    if key not in _COMPILED:
        z = cutlass.Int32(0)
        _COMPILED[key] = cute.compile(
            _launch,
            _fake(_DSL[src_dtype]),
            _fake(_DSL[dst_dtype]),
            z,
            z,
            make_fake_stream(),
            _DSL[dst_dtype],
            options="--enable-tvm-ffi",
        )
    return _COMPILED[key]


def _as_rows(t: torch.Tensor) -> torch.Tensor | None:
    """``t`` as a 2D ``[R, C]`` view with a unit inner stride, or None."""

    if t.ndim == 0 or t.numel() == 0:
        return None
    if t.ndim == 1:
        return t.view(1, -1) if t.stride(0) == 1 else None
    if t.stride(-1) != 1:
        return None
    for i in range(t.ndim - 2):
        if t.shape[i] > 1 and t.stride(i) != t.stride(i + 1) * t.shape[i + 1]:
            return None
    rows = 1
    for n in t.shape[:-1]:
        rows *= n
    return t.as_strided((rows, t.shape[-1]), (t.stride(-2), 1))


def _overlap(a: torch.Tensor, b: torch.Tensor) -> bool:
    def span(t):
        lo = t.data_ptr()
        hi = lo + (sum((n - 1) * s for n, s in zip(t.shape, t.stride())) + 1) * t.element_size()
        return lo, hi

    a_lo, a_hi = span(a)
    b_lo, b_hi = span(b)
    return a_lo < b_hi and b_lo < a_hi


def plan(dst: torch.Tensor, src: torch.Tensor):
    """``(dst2d, src2d, src_dtype, dst_dtype)`` when this kernel takes the
    pair, else None."""

    if not (dst.is_cuda and src.is_cuda) or dst.device != src.device:
        return None
    if dst.shape != src.shape or dst.numel() == 0:
        return None
    # Conjugate / negative-bit views carry a lazy transform a bit copy would
    # drop; Tensor.copy_ resolves it.
    if src.is_conj() or dst.is_conj() or src.is_neg() or dst.is_neg():
        return None
    sd, dd = src.dtype, dst.dtype
    raw = None
    if sd == dd:
        size = src.element_size()
        if size not in _RAW_BY_SIZE:
            return None
        raw = _RAW_BY_SIZE[size]
    elif (sd, dd) not in _CONVERSIONS:
        return None
    # Decline on layout before any dtype view, so an unsupported pair returns
    # None (the caller falls back) instead of raising.
    s2, d2 = _as_rows(src), _as_rows(dst)
    if s2 is None or d2 is None:
        return None
    if raw is not None:
        s2, d2, sd, dd = s2.view(raw), d2.view(raw), raw, raw
    # A row pitch below the row width (e.g. an expanded destination) makes
    # threads write one address twice; Tensor.copy_ rejects that internal
    # overlap, so leave such pairs to it.
    if d2.shape[0] > 1 and d2.stride(0) < d2.shape[1]:
        return None
    if _overlap(s2, d2):
        return None
    return d2, s2, sd, dd


def copy_(dst: torch.Tensor, src: torch.Tensor) -> bool:
    """``dst.copy_(src)`` on this kernel; False (nothing done) when the pair
    is outside its domain."""

    p = plan(dst, src)
    if p is None:
        return False
    d2, s2, sd, dd = p
    rows, cols = d2.shape
    col_tiles = (cols + _COLS_PER_CTA - 1) // _COLS_PER_CTA
    _compiled(sd, dd)(
        s2, d2, col_tiles, rows * col_tiles, torch.cuda.current_stream(d2.device).cuda_stream
    )
    return True


def warm(device: torch.device | str = "cuda") -> None:
    """Compile and launch every dtype pair once (eagerly, before any capture)."""

    pairs = [(t, t) for t in _RAW_BY_SIZE.values()] + sorted(_CONVERSIONS, key=str)
    for sd, dd in pairs:
        src = torch.zeros((2, 3), dtype=sd, device=device)
        dst = torch.empty((2, 3), dtype=dd, device=device)
        if not copy_(dst, src):
            raise RuntimeError(f"warm: {sd} -> {dd} not taken")
    torch.cuda.synchronize()
