# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hidden-width (6144) BF16 row ops of GLM-5.2 outside the fused all-reduce:
RMSNorm and residual add.

The trunk's per-layer norms ride in the all-reduce kernel; a few hidden-width
row ops are left over -- layer 0's input norm, and in the MTP module its input
norms (``enorm`` / ``hnorm``), ``norm1``, the output head's norm and the two
residual adds of the module's output.  Two kernels cover them:

* ``rmsnorm``: ``y = bf16((x * rsqrt(mean(x^2) + eps)) * w)``, FP32
  accumulation, ``rsqrt.approx`` -- the operation order of the FlashInfer
  RMSNorm it replaces; only the summation order of ``x^2`` differs, which can
  move a result by one BF16 ulp.  The ``rmsnorm_rows`` form reads row
  ``rows[i]`` of ``x`` for output row ``i`` (int64 ``rows``): the MTP head's
  norm of the gathered last-accepted-token rows, without the gather.
* ``add``: ``y = bf16(float(a) + float(b))`` -- bitwise ``torch.add`` on BF16.
* ``add_rmsnorm``: ``rmsnorm(add(a, b))`` in one launch, bit for bit the two
  kernels in sequence (the sum rounded to BF16 as ``add`` stores it, then
  ``rmsnorm``'s operations in its order); the sum is not stored.  The MTP
  module's residual add and head norm on an MTP3 / MTP5 decode step.
* ``gather_rows``: ``y[i] = x[rows[i]]`` (int32 or int64 ``rows``) -- the
  shell's hidden-state lookup for the logits rows, bitwise.

All take ``[M, 6144]`` rows with a runtime row pitch (inner dimension
contiguous, 16-byte-aligned rows) and write compact ``[M, 6144]`` outputs.
One 256-thread CTA per row; each thread owns three 8-element (16-byte)
vectors, so a row is read once with full-width loads.

"""

from collections.abc import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream, make_fake_tensor

HIDDEN = 6144
_THREADS = 256
_WARP_SIZE = 32
_WARPS = _THREADS // _WARP_SIZE
_VEC = 8
_VECS_PER_THREAD = HIDDEN // (_THREADS * _VEC)
_ALIGNMENT_BYTES = 16
assert _VECS_PER_THREAD * _THREADS * _VEC == HIDDEN

CompiledKernel = Callable[..., object]
_COMPILED: dict[tuple, CompiledKernel] = {}


@cute.jit
def _load8(ptr, offset):
    values = cute.make_rmem_tensor((_VEC,), cutlass.BFloat16)
    src = cute.make_tensor(ptr + cute.assume(offset, _VEC), cute.make_layout((_VEC,)))
    cute.autovec_copy(src, values)
    return values


@cute.jit
def _store8(ptr, offset, values) -> None:
    dst = cute.make_tensor(ptr + cute.assume(offset, _VEC), cute.make_layout((_VEC,)))
    cute.autovec_copy(values, dst)


@cute.jit
def _warp_sum(value):
    for offset in (16, 8, 4, 2, 1):
        value = value + cute.arch.shuffle_sync_bfly(value, offset)
    return value


@cute.kernel
def _rmsnorm_kernel(
    x: cute.Tensor,
    weight: cute.Tensor,
    y: cute.Tensor,
    rows: cute.Tensor,
    x_row_stride: cutlass.Int32,
    eps: cutlass.Float32,
    use_pdl: cutlass.Constexpr[bool],
    gathered: cutlass.Constexpr[bool],
) -> None:
    row, _, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    lane = tidx % _WARP_SIZE
    warp = tidx // _WARP_SIZE
    warp_sums = cutlass.Array(
        cutlass.Float32, _WARPS, space=cutlass.AddressSpace.smem, alignment=_ALIGNMENT_BYTES
    )

    # Weights are immutable: fetch them before waiting on the producer.
    weights = []
    for v in cutlass.range_constexpr(_VECS_PER_THREAD):
        weights.append(_load8(weight.iterator, (v * _THREADS + tidx) * _VEC))
    if cutlass.const_expr(use_pdl):
        cute.arch.griddepcontrol_wait()

    src_row = cutlass.Int64(row)
    if cutlass.const_expr(gathered):
        src_row = rows[row]
    src_base = src_row * cutlass.Int64(x_row_stride)
    values = []
    sum_sq = cutlass.Float32(0.0)
    for v in cutlass.range_constexpr(_VECS_PER_THREAD):
        loaded = _load8(x.iterator, src_base + (v * _THREADS + tidx) * _VEC)
        values.append(loaded)
        for k in cutlass.range_constexpr(_VEC):
            f = loaded[k].to(cutlass.Float32)
            sum_sq = sum_sq + f * f
    sum_sq = _warp_sum(sum_sq)
    if lane == 0:
        warp_sums[warp] = sum_sq
    cute.arch.barrier(barrier_id=0, number_of_threads=_THREADS)
    total = cutlass.Float32(0.0)
    for w in cutlass.range_constexpr(_WARPS):
        total = total + warp_sums[w]
    inv_rms = cute.math.rsqrt(total / cutlass.Float32(HIDDEN) + eps, fastmath=True)

    dst_base = cutlass.Int64(row) * HIDDEN
    for v in cutlass.range_constexpr(_VECS_PER_THREAD):
        out = cute.make_rmem_tensor((_VEC,), cutlass.BFloat16)
        for k in cutlass.range_constexpr(_VEC):
            f = values[v][k].to(cutlass.Float32) * inv_rms
            out[k] = (f * weights[v][k].to(cutlass.Float32)).to(cutlass.BFloat16)
        _store8(y.iterator, dst_base + (v * _THREADS + tidx) * _VEC, out)
    if cutlass.const_expr(use_pdl):
        if tidx == 0:
            cute.arch.griddepcontrol_launch_dependents()


@cute.kernel
def _add_kernel(
    a: cute.Tensor,
    b: cute.Tensor,
    y: cute.Tensor,
    a_row_stride: cutlass.Int32,
    b_row_stride: cutlass.Int32,
    use_pdl: cutlass.Constexpr[bool],
) -> None:
    row, _, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    if cutlass.const_expr(use_pdl):
        cute.arch.griddepcontrol_wait()
    for v in cutlass.range_constexpr(_VECS_PER_THREAD):
        col = (v * _THREADS + tidx) * _VEC
        lhs = _load8(a.iterator, row * a_row_stride + col)
        rhs = _load8(b.iterator, row * b_row_stride + col)
        out = cute.make_rmem_tensor((_VEC,), cutlass.BFloat16)
        for k in cutlass.range_constexpr(_VEC):
            out[k] = (lhs[k].to(cutlass.Float32) + rhs[k].to(cutlass.Float32)).to(cutlass.BFloat16)
        _store8(y.iterator, row * HIDDEN + col, out)
    if cutlass.const_expr(use_pdl):
        if tidx == 0:
            cute.arch.griddepcontrol_launch_dependents()


@cute.kernel
def _add_rmsnorm_kernel(
    a: cute.Tensor,
    b: cute.Tensor,
    weight: cute.Tensor,
    y: cute.Tensor,
    a_row_stride: cutlass.Int32,
    b_row_stride: cutlass.Int32,
    eps: cutlass.Float32,
) -> None:
    """``_add_kernel`` then ``_rmsnorm_kernel`` on its output, fused: each thread forms the
    BF16 sum of its own three vectors exactly as ``_add_kernel`` stores it, keeps it in
    registers, and runs ``_rmsnorm_kernel``'s body on it unchanged (same vectors per thread,
    same per-thread, warp-butterfly and warp-sum order, same ``rsqrt.approx``), so every
    output bit is the two kernels'. No PDL, as the served ``compile_add`` /
    ``compile_rmsnorm`` instances it replaces."""
    row, _, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    lane = tidx % _WARP_SIZE
    warp = tidx // _WARP_SIZE
    warp_sums = cutlass.Array(
        cutlass.Float32, _WARPS, space=cutlass.AddressSpace.smem, alignment=_ALIGNMENT_BYTES
    )

    weights = []
    for v in cutlass.range_constexpr(_VECS_PER_THREAD):
        weights.append(_load8(weight.iterator, (v * _THREADS + tidx) * _VEC))

    values = []
    sum_sq = cutlass.Float32(0.0)
    for v in cutlass.range_constexpr(_VECS_PER_THREAD):
        col = (v * _THREADS + tidx) * _VEC
        lhs = _load8(a.iterator, row * a_row_stride + col)
        rhs = _load8(b.iterator, row * b_row_stride + col)
        summed = cute.make_rmem_tensor((_VEC,), cutlass.BFloat16)
        for k in cutlass.range_constexpr(_VEC):
            summed[k] = (lhs[k].to(cutlass.Float32) + rhs[k].to(cutlass.Float32)).to(
                cutlass.BFloat16
            )
        values.append(summed)
        for k in cutlass.range_constexpr(_VEC):
            f = summed[k].to(cutlass.Float32)
            sum_sq = sum_sq + f * f
    sum_sq = _warp_sum(sum_sq)
    if lane == 0:
        warp_sums[warp] = sum_sq
    cute.arch.barrier(barrier_id=0, number_of_threads=_THREADS)
    total = cutlass.Float32(0.0)
    for w in cutlass.range_constexpr(_WARPS):
        total = total + warp_sums[w]
    inv_rms = cute.math.rsqrt(total / cutlass.Float32(HIDDEN) + eps, fastmath=True)

    dst_base = cutlass.Int64(row) * HIDDEN
    for v in cutlass.range_constexpr(_VECS_PER_THREAD):
        out = cute.make_rmem_tensor((_VEC,), cutlass.BFloat16)
        for k in cutlass.range_constexpr(_VEC):
            f = values[v][k].to(cutlass.Float32) * inv_rms
            out[k] = (f * weights[v][k].to(cutlass.Float32)).to(cutlass.BFloat16)
        _store8(y.iterator, dst_base + (v * _THREADS + tidx) * _VEC, out)


@cute.kernel
def _gather_kernel(
    x: cute.Tensor,
    rows: cute.Tensor,
    y: cute.Tensor,
    x_row_stride: cutlass.Int32,
) -> None:
    row, _, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    src_base = cutlass.Int64(rows[row]) * cutlass.Int64(x_row_stride)
    dst_base = cutlass.Int64(row) * HIDDEN
    for v in cutlass.range_constexpr(_VECS_PER_THREAD):
        col = (v * _THREADS + tidx) * _VEC
        _store8(y.iterator, dst_base + col, _load8(x.iterator, src_base + col))


@cute.jit
def _launch_gather(
    x: cute.Tensor,
    rows: cute.Tensor,
    y: cute.Tensor,
    x_row_stride: cutlass.Int32,
    stream: cuda.CUstream,
) -> None:
    _gather_kernel(x, rows, y, x_row_stride).launch(
        grid=(cute.size(y, mode=[0]), 1, 1), block=(_THREADS, 1, 1), stream=stream
    )


@cute.jit
def _launch_rmsnorm(
    x: cute.Tensor,
    weight: cute.Tensor,
    y: cute.Tensor,
    rows: cute.Tensor,
    x_row_stride: cutlass.Int32,
    eps: cutlass.Float32,
    stream: cuda.CUstream,
    use_pdl: cutlass.Constexpr[bool],
    gathered: cutlass.Constexpr[bool],
) -> None:
    _rmsnorm_kernel(x, weight, y, rows, x_row_stride, eps, use_pdl, gathered).launch(
        grid=(cute.size(y, mode=[0]), 1, 1),
        block=(_THREADS, 1, 1),
        stream=stream,
        use_pdl=use_pdl,
    )


@cute.jit
def _launch_add(
    a: cute.Tensor,
    b: cute.Tensor,
    y: cute.Tensor,
    a_row_stride: cutlass.Int32,
    b_row_stride: cutlass.Int32,
    stream: cuda.CUstream,
    use_pdl: cutlass.Constexpr[bool],
) -> None:
    _add_kernel(a, b, y, a_row_stride, b_row_stride, use_pdl).launch(
        grid=(cute.size(y, mode=[0]), 1, 1),
        block=(_THREADS, 1, 1),
        stream=stream,
        use_pdl=use_pdl,
    )


@cute.jit
def _launch_add_rmsnorm(
    a: cute.Tensor,
    b: cute.Tensor,
    weight: cute.Tensor,
    y: cute.Tensor,
    a_row_stride: cutlass.Int32,
    b_row_stride: cutlass.Int32,
    eps: cutlass.Float32,
    stream: cuda.CUstream,
) -> None:
    _add_rmsnorm_kernel(a, b, weight, y, a_row_stride, b_row_stride, eps).launch(
        grid=(cute.size(y, mode=[0]), 1, 1), block=(_THREADS, 1, 1), stream=stream
    )


def _fake_rows():
    """[M, 6144] BF16 with a runtime row pitch, 16-byte-aligned."""

    rows = cute.sym_int32()
    pitch = cute.sym_int32(divisibility=_VEC)
    return make_fake_tensor(
        cutlass.BFloat16, (rows, HIDDEN), stride=(pitch, 1), assumed_align=_ALIGNMENT_BYTES
    )


def _fake_out():
    return make_fake_compact_tensor(
        cutlass.BFloat16,
        (cute.sym_int32(), HIDDEN),
        stride_order=(1, 0),
        assumed_align=_ALIGNMENT_BYTES,
    )


def compile_rmsnorm(use_pdl: bool = False, gathered: bool = False) -> CompiledKernel:
    """``fn(x[N,6144], weight[6144], y[M,6144], rows[M] i64, x_row_stride, eps,
    stream)``; ``rows`` is read only when ``gathered`` (else ``M == N``)."""

    key = ("rmsnorm", use_pdl, gathered)
    if key not in _COMPILED:
        weight = make_fake_compact_tensor(
            cutlass.BFloat16, (HIDDEN,), assumed_align=_ALIGNMENT_BYTES
        )
        rows = make_fake_compact_tensor(cutlass.Int64, (cute.sym_int32(),), assumed_align=8)
        _COMPILED[key] = cute.compile(
            _launch_rmsnorm,
            _fake_rows(),
            weight,
            _fake_out(),
            rows,
            cutlass.Int32(HIDDEN),
            cutlass.Float32(1e-5),
            make_fake_stream(),
            use_pdl,
            gathered,
            options="--enable-tvm-ffi",
        )
    return _COMPILED[key]


def compile_add(use_pdl: bool = False) -> CompiledKernel:
    """``fn(a[M,6144], b[M,6144], y[M,6144], a_row_stride, b_row_stride, stream)``."""

    key = ("add", use_pdl)
    if key not in _COMPILED:
        _COMPILED[key] = cute.compile(
            _launch_add,
            _fake_rows(),
            _fake_rows(),
            _fake_out(),
            cutlass.Int32(HIDDEN),
            cutlass.Int32(HIDDEN),
            make_fake_stream(),
            use_pdl,
            options="--enable-tvm-ffi",
        )
    return _COMPILED[key]


def compile_add_rmsnorm() -> CompiledKernel:
    """``fn(a[M,6144], b[M,6144], weight[6144], y[M,6144], a_row_stride, b_row_stride,
    eps, stream)``."""

    key = ("add_rmsnorm",)
    if key not in _COMPILED:
        weight = make_fake_compact_tensor(
            cutlass.BFloat16, (HIDDEN,), assumed_align=_ALIGNMENT_BYTES
        )
        _COMPILED[key] = cute.compile(
            _launch_add_rmsnorm,
            _fake_rows(),
            _fake_rows(),
            weight,
            _fake_out(),
            cutlass.Int32(HIDDEN),
            cutlass.Int32(HIDDEN),
            cutlass.Float32(1e-5),
            make_fake_stream(),
            options="--enable-tvm-ffi",
        )
    return _COMPILED[key]


def compile_gather(index_dtype: torch.dtype = torch.int32) -> CompiledKernel:
    """``fn(x[N,6144], rows[M] int32|int64, y[M,6144], x_row_stride, stream)``."""

    key = ("gather", index_dtype)
    if key not in _COMPILED:
        idx_t, align = (cutlass.Int64, 8) if index_dtype == torch.int64 else (cutlass.Int32, 4)
        _COMPILED[key] = cute.compile(
            _launch_gather,
            _fake_rows(),
            make_fake_compact_tensor(idx_t, (cute.sym_int32(),), assumed_align=align),
            _fake_out(),
            cutlass.Int32(HIDDEN),
            make_fake_stream(),
            options="--enable-tvm-ffi",
        )
    return _COMPILED[key]


def _on_device_of(x: torch.Tensor, *aux: torch.Tensor) -> bool:
    """Whether every auxiliary operand lives on ``x``'s CUDA device: the kernel
    launches on ``x.device``, so a host or other-device operand would only fail
    inside the launch."""

    return all(t.is_cuda and t.device == x.device for t in aux)


def gather_rows(fn: CompiledKernel, x: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """New compact ``x[rows]`` (``fn`` from ``compile_gather(rows.dtype)``).

    Every ``rows`` value must lie in ``[0, x.shape[0])``; it is not checked here
    because the indices live on the device: a check would cost a host
    synchronization per call and cannot run under CUDA-graph capture, where the
    serving path calls this. Callers pass in-range gather plans.
    """

    if (
        not supports(x)
        or not _on_device_of(x, rows)
        or rows.dtype not in (torch.int32, torch.int64)
        or rows.ndim != 1
        or not rows.is_contiguous()
    ):
        raise ValueError("gather_rows: unsupported operands")
    y = torch.empty((rows.numel(), HIDDEN), dtype=torch.bfloat16, device=x.device)
    if rows.numel():
        fn(x, rows, y, x.stride(0), torch.cuda.current_stream(x.device).cuda_stream)
    return y


def supports(*tensors: torch.Tensor) -> bool:
    """Whether every tensor is a CUDA BF16 ``[M, 6144]`` row block this kernel
    takes: inner dimension contiguous, rows 16-byte aligned, ``M >= 1``."""

    for t in tensors:
        if (
            not t.is_cuda
            or t.dtype != torch.bfloat16
            or t.ndim != 2
            or t.shape[1] != HIDDEN
            or t.shape[0] < 1
            or t.stride(1) != 1
            or t.stride(0) % _VEC != 0
            or t.data_ptr() % _ALIGNMENT_BYTES != 0
        ):
            return False
    return True


def rmsnorm(fn: CompiledKernel, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """New compact ``[M, 6144]`` BF16 RMSNorm of ``x``."""

    if (
        not supports(x)
        or not _on_device_of(x, weight)
        or weight.shape != (HIDDEN,)
        or weight.dtype != torch.bfloat16
        or not weight.is_contiguous()
    ):
        raise ValueError("rmsnorm: unsupported operands")
    y = torch.empty((x.shape[0], HIDDEN), dtype=torch.bfloat16, device=x.device)
    fn(
        x,
        weight,
        y,
        _no_rows(x.device),
        x.stride(0),
        float(eps),
        torch.cuda.current_stream(x.device).cuda_stream,
    )
    return y


def rmsnorm_rows(
    fn: CompiledKernel, x: torch.Tensor, rows: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """New compact ``[len(rows), 6144]`` BF16 RMSNorm of ``x[rows]`` (``fn``
    from ``compile_rmsnorm(gathered=True)``; ``rows`` int64, in range)."""

    if (
        not supports(x)
        or not _on_device_of(x, weight, rows)
        or weight.shape != (HIDDEN,)
        or weight.dtype != torch.bfloat16
        or not weight.is_contiguous()
        or rows.dtype != torch.int64
        or rows.ndim != 1
        or not rows.is_contiguous()
        or rows.numel() < 1
    ):
        raise ValueError("rmsnorm_rows: unsupported operands")
    y = torch.empty((rows.numel(), HIDDEN), dtype=torch.bfloat16, device=x.device)
    fn(x, weight, y, rows, x.stride(0), float(eps), torch.cuda.current_stream(x.device).cuda_stream)
    return y


_NO_ROWS: dict[torch.device, torch.Tensor] = {}


def _no_rows(device: torch.device) -> torch.Tensor:
    if device not in _NO_ROWS:
        _NO_ROWS[device] = torch.zeros((1,), dtype=torch.int64, device=device)
    return _NO_ROWS[device]


def add(fn: CompiledKernel, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """New compact ``[M, 6144]`` BF16 ``a + b``."""

    if not supports(a, b) or a.device != b.device or a.shape != b.shape:
        raise ValueError("add: unsupported operands")
    y = torch.empty((a.shape[0], HIDDEN), dtype=torch.bfloat16, device=a.device)
    fn(a, b, y, a.stride(0), b.stride(0), torch.cuda.current_stream(a.device).cuda_stream)
    return y


def add_rmsnorm(
    fn: CompiledKernel, a: torch.Tensor, b: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """New compact ``[M, 6144]`` BF16 RMSNorm of ``a + b`` (``fn`` from
    ``compile_add_rmsnorm``): bit for bit ``rmsnorm(add(a, b), weight, eps)``."""

    if (
        not supports(a, b)
        or a.device != b.device
        or a.shape != b.shape
        or not _on_device_of(a, weight)
        or weight.shape != (HIDDEN,)
        or weight.dtype != torch.bfloat16
        or not weight.is_contiguous()
    ):
        raise ValueError("add_rmsnorm: unsupported operands")
    y = torch.empty((a.shape[0], HIDDEN), dtype=torch.bfloat16, device=a.device)
    fn(
        a,
        b,
        weight,
        y,
        a.stride(0),
        b.stride(0),
        float(eps),
        torch.cuda.current_stream(a.device).cuda_stream,
    )
    return y
