# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 context-step SwiGLU that writes its result as an MXF8 operand (e4m3 values +
e8m0 1x32 block scales), the A operand of the MXF8 down projection.

On a context step the dense MLP (layers 0-2, per-rank width 3072) and the MoE shared expert
(width 512) run their gate|up GEMM into BF16 ``src [T, 2W]`` (gate columns first, up
second), then ``silu(gate) * up``.  With the down projection as an MXF8 block-scaled GEMM
(``self_attn_prefill_projections_mxf8``: ``dense_down_mxf8``, ``shared_down_mxf8``) the
SwiGLU result would otherwise go through a BF16 store and a separate activation-quant
launch; this kernel computes it and quantizes it in one pass:

``a = bf16(silu(g) * u)`` in FP32 math, rounded to BF16, then per 32-element block of a row
the act-quant rule of ``self_attn_prefill_act_quant_mxf8`` (``amax / 448`` rounded up to a
power of two, e4m3 values, blocked scale layout, code 127 in the padded scale rows).  The
SwiGLU expression is the served BF16 SwiGLU's that the MXF8 path replaces, chosen at compile
time: ``fast=False`` -- ``g / (1 + exp(-g)) * u`` (precise exp / divide), the MoE shared
expert's ``moe_sigmoid_topk_swiglu_rows.swiglu``; ``fast=True`` -- ``u * g * rcp(1 +
exp2(-g log2 e))`` (MUFU ex2 / rcp, approximate, FTZ), the dense MLP's
``mlp_dense_swiglu_n3072_bf16``.  The bytes are therefore exactly ``quantize(swiglu(src))``
for the matching served SwiGLU (the entry's test checks the dense one byte for byte).  At
W = 3072 the precise math makes the kernel instruction-bound (36 vs 22 us for the BF16
SwiGLU at T = 7168); the fast variant is the dense MLP's.

One 256-thread CTA covers ``16 * passes`` rows x 128 output columns (16 threads per row, 8
outputs per thread, 4 threads per scale block): every pass issues its two 16-byte loads
(gate and up) unconditionally (rows past R re-read row R - 1), so all of a thread's loads
are in flight before the first exponential.  The grid is ``(padded_rows(R) / (16 *
passes), W / 128)``.  Row count and source pitch are runtime values; one compiled instance
serves every W that is a multiple of 128.  Pointer arguments only: CUDA-graph safe.

Pattern: glm-ctx-swiglu-mxf8
Device: SM100+ (Blackwell)
Shape: runtime rows R, runtime W (W % 128 == 0); BF16 [R, 2W] in, e4m3 [R, W] + e8m0 blocked SF out
"""

from collections.abc import Callable
from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream, make_fake_tensor

from ..common import self_attn_prefill_act_quant_mxf8 as _quant

_VEC = 8  # outputs per thread (8 gate + 8 up BF16 loaded)
_THREADS_PER_ROW = 16  # 128 output columns per CTA tile
_ROWS_PER_PASS = 256 // _THREADS_PER_ROW  # 16
# 16-row passes per CTA (1, 2, 4 or 8: a CTA never straddles a 128-row scale block); the act
# quant's best choice at the context row counts (DEFAULT_PASSES there) is kept.
PASSES_CHOICES = (1, 2, 4, 8)
DEFAULT_PASSES = 4
_THREADS = 256
_SF_VEC = 32
_COLS_PER_CTA = _VEC * _THREADS_PER_ROW  # 128
_MIN_AMAX = 1.0e-12
_INV_E4M3_MAX = 1.0 / 448.0
_ALIGNMENT_BYTES = 16
_LOG2E = 1.4426950408889634  # mlp_dense_swiglu_n3072_bf16's constant

CompiledKernel = Callable[..., object]


@cute.jit
def _load_bf16x8(ptr, offset):
    values = cute.make_rmem_tensor((_VEC,), cutlass.BFloat16)
    src = cute.make_tensor(ptr + cute.assume(offset, _VEC), cute.make_layout((_VEC,)))
    cute.autovec_copy(src, values)
    return values


@cute.kernel
def glm52_prefill_swiglu_mxf8_kernel(
    src: cute.Tensor,
    q: cute.Tensor,
    sf: cute.Tensor,
    rows: cutlass.Int32,
    k_tiles: cutlass.Int32,
    src_row_stride: cutlass.Int64,
    passes: cutlass.Constexpr[int],
    fast: cutlass.Constexpr[bool],
) -> None:
    row_tile, k_tile, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    chunk = tidx % _THREADS_PER_ROW
    row_in_pass = tidx // _THREADS_PER_ROW
    col = cutlass.Int64(k_tile) * _COLS_PER_CTA + cutlass.Int64(chunk) * _VEC
    width = cutlass.Int64(k_tiles) * _COLS_PER_CTA  # W: the output (and q) row length
    # Every pass loads unconditionally (rows past R re-read row R - 1 and store nothing),
    # so all of a thread's loads are in flight before the first use.
    last = rows - 1
    gate = cute.make_rmem_tensor((passes * _VEC,), cutlass.BFloat16)
    up = cute.make_rmem_tensor((passes * _VEC,), cutlass.BFloat16)
    for p in cutlass.range_constexpr(passes):
        m_ld = row_tile * (passes * _ROWS_PER_PASS) + p * _ROWS_PER_PASS + row_in_pass
        if m_ld > last:
            m_ld = last
        base = cutlass.Int64(m_ld) * src_row_stride + col
        g8 = _load_bf16x8(src.iterator, base)
        u8 = _load_bf16x8(src.iterator, base + width)
        for i in cutlass.range_constexpr(_VEC):
            gate[p * _VEC + i] = g8[i]
            up[p * _VEC + i] = u8[i]
    for p in cutlass.range_constexpr(passes):
        m = row_tile * (passes * _ROWS_PER_PASS) + p * _ROWS_PER_PASS + row_in_pass
        vals = cute.make_rmem_tensor((_VEC,), cutlass.Float32)
        local_max = cutlass.Float32(0.0)
        for i in cutlass.range_constexpr(_VEC):
            g = gate[p * _VEC + i].to(cutlass.Float32)
            u = up[p * _VEC + i].to(cutlass.Float32)
            # the BF16 SwiGLU value, as the served SwiGLU it replaces stores it
            if cutlass.const_expr(fast):
                exp_neg = cute.math.exp2(-g * cutlass.Float32(_LOG2E), fastmath=True)
                sig = cute.math.rcp(cutlass.Float32(1.0) + exp_neg, approx=True, ftz=True)
                vals[i] = (u * g * sig).to(cutlass.BFloat16).to(cutlass.Float32)
            else:
                s = g / (cutlass.Float32(1.0) + cute.math.exp(-g))
                vals[i] = (s * u).to(cutlass.BFloat16).to(cutlass.Float32)
            a = cute.math.absf(vals[i])
            if a > local_max:
                local_max = a
        # From here on the act quant kernel's arithmetic, instruction for instruction:
        # 4 consecutive lanes hold one 32-element scale block (one row); every lane of
        # the warp reaches the shuffles.
        peer = cute.arch.shuffle_sync_bfly(local_max, 1)
        if peer > local_max:
            local_max = peer
        peer = cute.arch.shuffle_sync_bfly(local_max, 2)
        if peer > local_max:
            local_max = peer
        safe_amax = local_max
        if safe_amax < cutlass.Float32(_MIN_AMAX):
            safe_amax = cutlass.Float32(_MIN_AMAX)
        scale = safe_amax * cutlass.Float32(_INV_E4M3_MAX)
        scale_bits = scale.bitcast(cutlass.Int32)
        exp_bits = scale_bits & cutlass.Int32(0x7F800000)
        if (scale_bits & cutlass.Int32(0x007FFFFF)) != cutlass.Int32(0):
            exp_bits += cutlass.Int32(0x00800000)
        code = exp_bits >> cutlass.Int32(23)
        inv_scale = cutlass.Float32(1.0) / exp_bits.bitcast(cutlass.Float32)
        if m < rows:
            # 8 values -> 8 e4m3 bytes in one vector conversion and one 8-byte store
            q8 = cute.make_rmem_tensor((_VEC,), cutlass.Float8E4M3FN)
            q8.store((vals.load() * inv_scale).to(cutlass.Float8E4M3FN))
            dst = cute.make_tensor(
                cute.recast_ptr(
                    q.iterator + cute.assume(cutlass.Int64(m) * width + col, _VEC),
                    dtype=cutlass.Float8E4M3FN,
                ),
                cute.make_layout((_VEC,)),
            )
            cute.autovec_copy(q8, dst)
        if chunk % 4 == 0:
            # padded scale rows (m >= R) get code 127 (scale 1), as the act quant writes them
            sf_code = code
            if m >= rows:
                sf_code = cutlass.Int32(127)
            mm = cutlass.Int64(m)
            off = (
                (mm % 32) * 16
                + ((mm // 32) % 4) * 4
                + cutlass.Int64(chunk // 4)
                + cutlass.Int64(k_tile) * 512
                + (mm // 128) * 512 * cutlass.Int64(k_tiles)
            )
            (sf.iterator + off).store(cutlass.Uint8(sf_code))


@cute.jit
def _launch(
    src: cute.Tensor,
    q: cute.Tensor,
    sf: cute.Tensor,
    rows: cutlass.Int32,
    k_tiles: cutlass.Int32,
    src_row_stride: cutlass.Int64,
    row_tiles: cutlass.Int32,
    stream: cuda.CUstream,
    passes: cutlass.Constexpr[int],
    fast: cutlass.Constexpr[bool],
) -> None:
    glm52_prefill_swiglu_mxf8_kernel(
        src, q, sf, rows, k_tiles, src_row_stride, passes, fast
    ).launch(grid=(row_tiles, k_tiles, 1), block=(_THREADS, 1, 1), stream=stream)


@dataclass(frozen=True)
class SwigluFn:
    """A compiled instance and its rows per CTA (``16 * passes``)."""

    fn: CompiledKernel
    rows_per_cta: int


_COMPILED: dict[tuple[int, int, bool], SwigluFn] = {}


def compile_swiglu_mxf8(
    device_index: int = 0, passes: int = DEFAULT_PASSES, fast: bool = False
) -> SwigluFn:
    """``SwigluFn``; ``fn(src[R,2W] bf16 (pitch % 8 == 0), q[R,W] u8, sf u8, R, W // 128,
    src_pitch, padded_rows(R) // rows_per_cta, stream)`` (``swiglu_mxf8`` builds the call).
    ``fast``: the dense MLP SwiGLU's MUFU math instead of the rows SwiGLU's precise one."""
    if passes not in PASSES_CHOICES:
        raise ValueError(f"swiglu mxf8: passes must be one of {PASSES_CHOICES}, got {passes}")
    key = (device_index, passes, bool(fast))
    if key not in _COMPILED:
        z32, z64 = cutlass.Int32(0), cutlass.Int64(0)
        fn = cute.compile(
            _launch,
            make_fake_tensor(
                cutlass.BFloat16,
                (cute.sym_int32(), cute.sym_int32()),
                stride=(cute.sym_int64(divisibility=_VEC), 1),
                assumed_align=_ALIGNMENT_BYTES,
            ),
            make_fake_tensor(
                cutlass.Uint8,
                (cute.sym_int32(), cute.sym_int32()),
                stride=(cute.sym_int64(divisibility=_ALIGNMENT_BYTES), 1),
                assumed_align=_ALIGNMENT_BYTES,
            ),
            make_fake_compact_tensor(
                cutlass.Uint8, (cute.sym_int32(),), assumed_align=_ALIGNMENT_BYTES
            ),
            z32,
            z32,
            z64,
            z32,
            make_fake_stream(),
            passes,
            bool(fast),
            options="--enable-tvm-ffi",
        )
        _COMPILED[key] = SwigluFn(fn, passes * _ROWS_PER_PASS)
    return _COMPILED[key]


def swiglu_mxf8(
    fn: SwigluFn,
    src: torch.Tensor,
    q: torch.Tensor | None = None,
    sf: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``silu(src[:, :W]) * src[:, W:]`` of BF16 ``src [R, 2W]`` as the MXF8 operand ``(q
    [R, W] uint8 e4m3 bytes, sf uint8 blocked e8m0 scales)``; allocates the outputs when
    not given.  ``q`` may be the first R rows of a larger contiguous buffer and ``sf`` a
    prefix of one (the blocked layout of row m does not depend on R)."""
    if (
        not src.is_cuda
        or src.dtype != torch.bfloat16
        or src.ndim != 2
        or src.stride(1) != 1
        or src.stride(0) % _VEC != 0
        or src.data_ptr() % _ALIGNMENT_BYTES != 0
        or src.shape[1] % (2 * _COLS_PER_CTA) != 0
        or src.shape[0] < 1
    ):
        raise ValueError(
            f"swiglu mxf8: src must be BF16 [R, 2W] with W % 128 == 0, unit column stride, "
            f"row pitch % 8 == 0 and 16-byte alignment; got {tuple(src.shape)} "
            f"{src.stride()} {src.dtype}"
        )
    rows, width = int(src.shape[0]), int(src.shape[1]) // 2
    if q is None:
        q = torch.empty(rows, width, dtype=torch.uint8, device=src.device)
    if sf is None:
        sf = torch.empty(_quant.sf_bytes(rows, width), dtype=torch.uint8, device=src.device)
    if (
        q.dtype != torch.uint8
        or tuple(q.shape) != (rows, width)
        or not q.is_contiguous()
        or q.device != src.device
        or q.data_ptr() % _ALIGNMENT_BYTES != 0
        or sf.dtype != torch.uint8
        or sf.numel() < _quant.sf_bytes(rows, width)
        or not sf.is_contiguous()
        or sf.device != src.device
        or sf.data_ptr() % _ALIGNMENT_BYTES != 0
    ):
        raise ValueError(
            f"swiglu mxf8: q must be uint8 [{rows}, {width}] contiguous, sf uint8 >= "
            f"sf_bytes({rows}, {width}); got q {tuple(q.shape)} {q.dtype}, sf {sf.numel()}"
        )
    fn.fn(
        src,
        q,
        sf,
        rows,
        width // _COLS_PER_CTA,
        src.stride(0),
        _quant.padded_rows(rows) // fn.rows_per_cta,
        torch.cuda.current_stream(src.device).cuda_stream,
    )
    return q, sf


def reference(src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch twin: the BF16 SwiGLU (FP32 math, the rows SwiGLU's expression) quantized by
    the act quant's torch twin.  Agrees with the kernel up to the BF16 SwiGLU's last-ulp
    freedom (exp / divide rounding)."""
    width = src.shape[1] // 2
    g, u = src[:, :width].float(), src[:, width:].float()
    a = (g / (1 + torch.exp(-g)) * u).to(torch.bfloat16)
    return _quant.reference(a.contiguous())
