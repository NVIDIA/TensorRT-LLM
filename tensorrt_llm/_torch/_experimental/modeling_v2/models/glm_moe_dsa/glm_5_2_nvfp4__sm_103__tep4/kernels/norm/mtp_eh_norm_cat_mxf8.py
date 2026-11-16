# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 MTP front end on context rows: ``concat(enorm(e), hnorm(h))`` written directly
as the MXF8 A operand of the ``eh_proj`` block-scaled GEMM.

The MTP module projects ``[T, 12288] = [rmsnorm(e) | rmsnorm(h)]`` with ``eh_proj``
(``[12288] -> [6144]``, replicated on every TP4 rank at context row counts).  The served
BF16 path runs two hidden-width RMSNorms (``hidden_rmsnorm_add_h6144_bf16``), a two-copy
concat and the BF16 GEMM; at T ~ 7.9k the norms and the concat alone take ~280 us (the
copies move 4 x 97 MB at ~1.9 TB/s on a served step).  Here the two norms quantize their
BF16 results in registers and store them straight into the two column halves of one MXF8
operand ``(q [T, 12288] e4m3, sf blocked e8m0)``: no BF16 concat, no activation-quant
launch.

Bytes: each half's BF16 values are those of the served RMSNorm kernel, instruction for
instruction (one 256-thread CTA per row, three 8-wide vectors per thread, the same FP32
sum of squares / warp butterfly / smem reduction order, ``rsqrt.approx``, the same two
roundings), and the quantization is ``self_attn_prefill_act_quant_mxf8``'s fused-producer
helper (``mxf8_quant_store_x8``: same block max, power-of-two scale and e4m3 cast).  So
``(q, sf)`` is bitwise ``act_quant(concat(rmsnorm(e), rmsnorm(h)))`` -- the operand the
served act quant would write from the served BF16 concat -- including code 127 in the
scale rows past T (the entry's test checks the whole buffer).

Thread / block layout: a lane holds 8 consecutive BF16 of one row; vector ``v`` of thread
``t`` covers columns ``(v * 256 + t) * 8 .. + 7`` of its half, so the 4 lanes of a
32-element scale block are consecutive lanes of one warp (the helper's contract).  One
jit call launches the kernel twice (``e`` -> columns 0..6143, ``h`` -> 6144..12287); the
first launch also writes the pad scale rows.  Row count and row pitches are runtime
values: one compiled function serves every T.  Pointer arguments only: CUDA-graph safe.

Pattern: glm-mtp-eh-norm-cat-mxf8
Device: SM100+ (Blackwell)
Shape: runtime T x 6144 BF16 (two sources) -> e4m3 [T, 12288] + e8m0 blocked SF
"""

from __future__ import annotations

from collections.abc import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream, make_fake_tensor

from ..common import self_attn_prefill_act_quant_mxf8 as _quant

HIDDEN = 6144
K_TOTAL = 2 * HIDDEN  # the eh_proj input width
K_TILES = K_TOTAL // 128  # 128-column k-groups of the blocked scale layout
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
def _warp_sum(value):
    for offset in (16, 8, 4, 2, 1):
        value = value + cute.arch.shuffle_sync_bfly(value, offset)
    return value


@cute.kernel
def glm52_mtp_eh_norm_q8_kernel(
    x: cute.Tensor,
    weight: cute.Tensor,
    q: cute.Tensor,
    sf: cute.Tensor,
    x_row_stride: cutlass.Int32,
    col0: cutlass.Int32,
    rows: cutlass.Int32,
    fill_pad: cutlass.Int32,
    eps: cutlass.Float32,
) -> None:
    """Row ``blockIdx.x`` of ``x``: RMSNorm (the served hidden-width kernel's arithmetic)
    quantized into columns ``col0 .. col0 + 6143`` of the ``[rows, 12288]`` MXF8 operand."""
    row, _, _ = cute.arch.block_idx()
    grid_x, _, _ = cute.arch.grid_dim()
    tidx, _, _ = cute.arch.thread_idx()
    lane = tidx % _WARP_SIZE
    warp = tidx // _WARP_SIZE
    warp_sums = cutlass.Array(
        cutlass.Float32,
        _WARPS,
        space=cutlass.AddressSpace.smem,
        alignment=_ALIGNMENT_BYTES,
    )

    weights = []
    for v in cutlass.range_constexpr(_VECS_PER_THREAD):
        weights.append(_load8(weight.iterator, (v * _THREADS + tidx) * _VEC))

    src_base = cutlass.Int64(row) * cutlass.Int64(x_row_stride)
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

    m = cutlass.Int64(row)
    for v in cutlass.range_constexpr(_VECS_PER_THREAD):
        col = cutlass.Int64(col0) + cutlass.Int64((v * _THREADS + tidx) * _VEC)
        vals = cute.make_rmem_tensor((_VEC,), cutlass.Float32)
        for k in cutlass.range_constexpr(_VEC):
            f = values[v][k].to(cutlass.Float32) * inv_rms
            # the served kernel's BF16 output value, as the act quant reads it back
            vals[k] = (
                (f * weights[v][k].to(cutlass.Float32)).to(cutlass.BFloat16).to(cutlass.Float32)
            )
        _quant.mxf8_quant_store_x8(
            vals,
            q.iterator,
            m * cutlass.Int64(K_TOTAL) + col,
            sf.iterator,
            m,
            col,
            cutlass.Int64(K_TILES),
            cutlass.Boolean(True),
        )
    if fill_pad != cutlass.Int32(0):
        _quant.mxf8_fill_pad_scales(
            sf.iterator,
            rows,
            cutlass.Int32(K_TILES),
            row * _THREADS + tidx,
            grid_x * _THREADS,
        )


@cute.jit
def _launch(
    x0: cute.Tensor,
    w0: cute.Tensor,
    x1: cute.Tensor,
    w1: cute.Tensor,
    q: cute.Tensor,
    sf: cute.Tensor,
    x0_row_stride: cutlass.Int32,
    x1_row_stride: cutlass.Int32,
    rows: cutlass.Int32,
    eps: cutlass.Float32,
    stream: cuda.CUstream,
) -> None:
    glm52_mtp_eh_norm_q8_kernel(
        x0, w0, q, sf, x0_row_stride, cutlass.Int32(0), rows, cutlass.Int32(1), eps
    ).launch(grid=(rows, 1, 1), block=(_THREADS, 1, 1), stream=stream)
    glm52_mtp_eh_norm_q8_kernel(
        x1, w1, q, sf, x1_row_stride, cutlass.Int32(HIDDEN), rows, cutlass.Int32(0), eps
    ).launch(grid=(rows, 1, 1), block=(_THREADS, 1, 1), stream=stream)


def _fake_rows():
    """[M, 6144] BF16 with a runtime row pitch, 16-byte-aligned."""
    return make_fake_tensor(
        cutlass.BFloat16,
        (cute.sym_int32(), HIDDEN),
        stride=(cute.sym_int32(divisibility=_VEC), 1),
        assumed_align=_ALIGNMENT_BYTES,
    )


def compile_norm_cat_q8(device_index: int = 0) -> CompiledKernel:
    """``fn(x0[T,6144], w0[6144], x1[T,6144], w1[6144], q[T,12288] u8, sf u8, x0_pitch,
    x1_pitch, T, eps, stream)`` (``norm_cat_q8`` builds the call)."""
    key = ("norm_cat_q8", device_index)
    if key not in _COMPILED:
        weight = make_fake_compact_tensor(
            cutlass.BFloat16, (HIDDEN,), assumed_align=_ALIGNMENT_BYTES
        )
        _COMPILED[key] = cute.compile(
            _launch,
            _fake_rows(),
            weight,
            _fake_rows(),
            weight,
            make_fake_tensor(
                cutlass.Uint8,
                (cute.sym_int32(), K_TOTAL),
                stride=(K_TOTAL, 1),
                assumed_align=_ALIGNMENT_BYTES,
            ),
            make_fake_compact_tensor(
                cutlass.Uint8, (cute.sym_int32(),), assumed_align=_ALIGNMENT_BYTES
            ),
            cutlass.Int32(HIDDEN),
            cutlass.Int32(HIDDEN),
            cutlass.Int32(1),
            cutlass.Float32(1e-5),
            make_fake_stream(),
            options="--enable-tvm-ffi",
        )
    return _COMPILED[key]


def _rows_ok(t: torch.Tensor) -> bool:
    return (
        t.is_cuda
        and t.dtype == torch.bfloat16
        and t.ndim == 2
        and t.shape[1] == HIDDEN
        and t.shape[0] >= 1
        and t.stride(1) == 1
        and t.stride(0) % _VEC == 0
        and t.data_ptr() % _ALIGNMENT_BYTES == 0
    )


def supports(x0: torch.Tensor, x1: torch.Tensor) -> bool:
    """Whether ``e`` / ``h`` are row blocks this kernel takes: CUDA BF16 ``[T, 6144]`` on one
    device, the same T, inner dimension contiguous, 16-byte-aligned rows."""
    return _rows_ok(x0) and _rows_ok(x1) and x0.shape[0] == x1.shape[0] and x0.device == x1.device


def norm_cat_q8(
    fn: CompiledKernel,
    x0: torch.Tensor,
    w0: torch.Tensor,
    x1: torch.Tensor,
    w1: torch.Tensor,
    eps: float,
    q: torch.Tensor | None = None,
    sf: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``act_quant(concat(rmsnorm(x0, w0), rmsnorm(x1, w1)))`` as ``(q [T, 12288] uint8
    e4m3 bytes, sf uint8 blocked e8m0 scales)``; allocates the outputs when not given."""
    if not supports(x0, x1):
        raise ValueError(
            "mtp eh norm cat q8: x0 / x1 must be BF16 [T, 6144] (same T, one device), unit "
            f"column stride, row pitch % 8 == 0, 16-byte aligned; got {tuple(x0.shape)} "
            f"{x0.stride()} / {tuple(x1.shape)} {x1.stride()}"
        )
    for w in (w0, w1):
        if (
            w.dtype != torch.bfloat16
            or tuple(w.shape) != (HIDDEN,)
            or not w.is_contiguous()
            or w.device != x0.device
            or w.data_ptr() % _ALIGNMENT_BYTES != 0
        ):
            raise ValueError("mtp eh norm cat q8: weights must be BF16 [6144] contiguous")
    rows = int(x0.shape[0])
    if q is None:
        q = torch.empty(rows, K_TOTAL, dtype=torch.uint8, device=x0.device)
    if sf is None:
        sf = torch.empty(_quant.sf_bytes(rows, K_TOTAL), dtype=torch.uint8, device=x0.device)
    if (
        q.dtype != torch.uint8
        or tuple(q.shape) != (rows, K_TOTAL)
        or not q.is_contiguous()
        or q.device != x0.device
        or q.data_ptr() % _ALIGNMENT_BYTES != 0
        or sf.dtype != torch.uint8
        or sf.numel() < _quant.sf_bytes(rows, K_TOTAL)
        or not sf.is_contiguous()
        or sf.device != x0.device
        or sf.data_ptr() % _ALIGNMENT_BYTES != 0
    ):
        raise ValueError(
            "mtp eh norm cat q8: q must be uint8 [T, 12288] contiguous, sf uint8 >= sf_bytes"
        )
    fn(
        x0,
        w0,
        x1,
        w1,
        q,
        sf,
        x0.stride(0),
        x1.stride(0),
        rows,
        float(eps),
        torch.cuda.current_stream(x0.device).cuda_stream,
    )
    return q, sf


def reference(x0, w0, x1, w1, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch twin: FP32 RMSNorm of each half rounded to BF16, concat, the act quant's torch
    reference.  (The kernel's sum of squares runs in a different order, so the BF16 norm
    values may differ from this twin by one ulp; the entry's test compares bytes against the
    served kernels instead and only bounds the distance to this twin.)"""

    def norm(x, w):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()).to(
            torch.bfloat16
        )

    return _quant.reference(torch.cat([norm(x0, w0), norm(x1, w1)], -1))
