# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BF16 rows -> MXF8 (e4m3 values + e8m0 1x32 block scales) for the GLM-5.2 context
projection GEMMs, in the scale-factor layout the block-scaled tcgen05 MMA reads.

The context step's kv_a / q_b / o_proj run as e4m3 x e4m3 block-scaled GEMMs
(``self_attn_prefill_projections_mxf8.py``).  Their A operand is the step's BF16
activation; this kernel quantizes it in one pass at DRAM rate.  The same kernel
quantizes the weights once at load.

Per 32-element block of a row (along K): ``amax = max |x|`` (at least
``_MIN_AMAX``); ``s = amax * fp32(1/448)`` rounded UP to a power of two (the
UE8M0 rule of ``self_attn_fused_cat_fp8``); the e8m0 code is the biased
exponent of ``s`` and the values are ``e4m3(x / s)``.  ``|x / s| <= 448`` by
construction, so nothing saturates.  ``x / s`` is exact (``s`` is a power of
two), so the only rounding is the e4m3 cast.

Scale-factor layout (``M32x4xrm_K4xrk``, as ``dense_blockscaled_gemm_persistent``
reorders it and cuBLAS calls "blocked"): rows are padded to 128 and the
``K / 32`` scale columns to 4; byte ``(m, kf)`` lives at
``(m % 32) * 16 + ((m // 32) % 4) * 4 + kf % 4 + (kf // 4) * 512
+ (m // 128) * 512 * (K / 128)``.  Rows past ``R`` in the last 128-row block
get code 127 (scale 1) so the MMA never reads an uninitialised factor.

One 256-thread CTA covers ``16 * passes`` rows x 128 columns (16 threads per row, 8
BF16 per thread, 4 threads per scale block): every pass issues its 16-byte load
unconditionally (rows past R re-read row R - 1), so all of a thread's loads are in
flight before the first conversion; then per pass the block amax, the scale, one
8-wide e4m3 convert and an 8-byte store.  The grid is ``(padded_rows(R) / (16 *
passes), K / 128)``.  The row count and pitches are runtime values:
one compiled instance serves every T.  Pointer arguments only: CUDA-graph safe.

Pattern: glm-ctx-proj-act-quant-mxf8
Device: SM100+ (Blackwell)
Shape: runtime rows R x K (K % 128 == 0), BF16 in, e4m3 [R, K] + e8m0 blocked SF out
"""

from collections.abc import Callable
from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream, make_fake_tensor

from .cuda_stream import raw_stream as _raw_stream

_VEC = 8  # BF16 elements per thread
_THREADS_PER_ROW = 16  # 128 columns per CTA tile
_ROWS_PER_PASS = 256 // _THREADS_PER_ROW  # 16
# 16-row passes per CTA (a compile-time choice: 1, 2, 4 or 8, so a CTA never straddles a
# 128-row scale-factor block).  Power-capped B300 graph-replay sweep (kv_a
# input K=6144 T=7255): 4 passes with the vector convert 28.2 us (DRAM floor 17.8), 2 / 8
# passes 30.1 / 30.8, scalar converts +15-18 %; small T 2.2 us.
PASSES_CHOICES = (1, 2, 4, 8)
DEFAULT_PASSES = 4
# e4m3 conversion as one 8-wide vector convert + 8-byte store (True) or per-element
# converts packed into two 4-byte words (False); bitwise the same bytes.
DEFAULT_VEC_CVT = True
_THREADS = 256
_SF_VEC = 32
_COLS_PER_CTA = _VEC * _THREADS_PER_ROW  # 128
_MIN_AMAX = 1.0e-12
_INV_E4M3_MAX = 1.0 / 448.0
_ALIGNMENT_BYTES = 16

CompiledKernel = Callable[..., object]
_COMPILED: dict[tuple[int, int, bool, bool], "QuantFn"] = {}


@cute.jit
def _load_bf16x8(ptr, offset):
    values = cute.make_rmem_tensor((_VEC,), cutlass.BFloat16)
    src = cute.make_tensor(ptr + cute.assume(offset, _VEC), cute.make_layout((_VEC,)))
    cute.autovec_copy(src, values)
    return values


# ----------------------------------------------------------------------------- fused producers
# Producers that already hold a BF16 row in registers (the two-shot AR+residual+RMSNorm for
# kv_a, the q_a RMSNorm for q_b) emit the MXF8 operand themselves with these helpers instead
# of a separate launch of the kernel below.  Their contract is this kernel's: a lane holds 8
# consecutive BF16 of one row, the 4 lanes of one 32-element scale block are consecutive lanes
# of one warp (xor 1 / xor 2 apart) and all 32 lanes reach the shuffles; the arithmetic is the
# kernel's instruction for instruction, so the bytes are identical (``verify_fused`` in the
# producers' modules checks them against ``reference``).


@cute.jit
def mxf8_x8_code_and_inv_scale(local_max: cutlass.Float32):
    """``(e8m0 code, 1 / scale)`` of the 32-element block whose 4 lanes each hold the
    ``local_max = max |x|`` of their 8 values (butterfly over lanes xor 1, xor 2)."""
    peer = cute.arch.shuffle_sync_bfly(local_max, 1)
    if peer > local_max:
        local_max = peer
    peer = cute.arch.shuffle_sync_bfly(local_max, 2)
    if peer > local_max:
        local_max = peer
    return mxf8_code_and_inv_scale(local_max)


@cute.jit
def mxf8_code_and_inv_scale(block_max: cutlass.Float32):
    """``(e8m0 code, 1 / scale)`` of a 32-element block with ``block_max = max |x|`` (the
    kernel's rule: at least ``_MIN_AMAX``, ``amax / 448`` rounded up to a power of two)."""
    safe_amax = block_max
    if safe_amax < cutlass.Float32(_MIN_AMAX):
        safe_amax = cutlass.Float32(_MIN_AMAX)
    scale = safe_amax * cutlass.Float32(_INV_E4M3_MAX)
    scale_bits = scale.bitcast(cutlass.Int32)
    exp_bits = scale_bits & cutlass.Int32(0x7F800000)
    if (scale_bits & cutlass.Int32(0x007FFFFF)) != cutlass.Int32(0):
        exp_bits += cutlass.Int32(0x00800000)
    code = exp_bits >> cutlass.Int32(23)
    inv_scale = cutlass.Float32(1.0) / exp_bits.bitcast(cutlass.Float32)
    return code, inv_scale


@cute.jit
def mxf8_sf_offset(m: cutlass.Int64, kf: cutlass.Int64, k_tiles: cutlass.Int64):
    """Byte offset of row ``m``, scale column ``kf`` (= column // 32) in the blocked layout."""
    return (
        (m % 32) * 16 + ((m // 32) % 4) * 4 + kf % 4 + (kf // 4) * 512 + (m // 128) * 512 * k_tiles
    )


@cute.jit
def mxf8_quant_store_x8(
    vals: cute.Tensor,
    q_base,
    q_off: cutlass.Int64,
    sf_base,
    m: cutlass.Int64,
    col: cutlass.Int64,
    k_tiles: cutlass.Int64,
    store: cutlass.Boolean,
) -> None:
    """Quantize the 8 FP32 values ``vals`` (rmem [8], exact BF16 values) of row ``m`` at
    columns ``col .. col + 7`` (``col % 8 == 0``): every lane of the warp must call this (the
    block max is a shuffle); lanes with ``store`` write the 8 e4m3 bytes at ``q_base +
    q_off`` (8-byte aligned) and, on the block's first lane, the scale byte."""
    local_max = cutlass.Float32(0.0)
    for i in cutlass.range_constexpr(_VEC):
        a = cute.math.absf(vals[i])
        if a > local_max:
            local_max = a
    code, inv_scale = mxf8_x8_code_and_inv_scale(local_max)
    if store:
        q8 = cute.make_rmem_tensor((_VEC,), cutlass.Float8E4M3FN)
        q8.store((vals.load() * inv_scale).to(cutlass.Float8E4M3FN))
        dst = cute.make_tensor(
            cute.recast_ptr(q_base + cute.assume(q_off, _VEC), dtype=cutlass.Float8E4M3FN),
            cute.make_layout((_VEC,)),
        )
        cute.autovec_copy(q8, dst)
        if (col % _SF_VEC) == cutlass.Int64(0):
            (sf_base + mxf8_sf_offset(m, col // _SF_VEC, k_tiles)).store(cutlass.Uint8(code))


@cute.jit
def mxf8_fill_pad_scales(
    sf_base,
    rows: cutlass.Int32,
    k_tiles: cutlass.Int32,
    start: cutlass.Int32,
    stride: cutlass.Int32,
) -> None:
    """Code 127 (scale 1) for rows ``rows .. padded_rows(rows) - 1`` of the blocked scale
    tensor, as the kernel below writes them (grid-stride over those bytes from ``start``)."""
    kfs = k_tiles * 4
    count = ((rows + 127) // 128 * 128 - rows) * kfs
    i = start
    while i < count:
        m = cutlass.Int64(rows + i // kfs)
        kf = cutlass.Int64(i % kfs)
        (sf_base + mxf8_sf_offset(m, kf, cutlass.Int64(k_tiles))).store(cutlass.Uint8(127))
        i = i + stride


@cute.kernel
def glm52_prefill_act_quant_mxf8_kernel(
    x: cute.Tensor,
    q: cute.Tensor,
    sf: cute.Tensor,
    rows: cutlass.Int32,
    k_tiles: cutlass.Int32,
    x_row_stride: cutlass.Int64,
    passes: cutlass.Constexpr[int],
    vec_cvt: cutlass.Constexpr[bool],
    pdl: cutlass.Constexpr[bool],
) -> None:
    if cutlass.const_expr(pdl):
        # Decode chains (``self_attn_decode_projections_mxf8``): release the dependent
        # GEMM at once (it streams its weights meanwhile), then wait for the producer.
        cute.arch.griddepcontrol_launch_dependents()
        cute.arch.griddepcontrol_wait()
    row_tile, k_tile, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    chunk = tidx % _THREADS_PER_ROW
    row_in_pass = tidx // _THREADS_PER_ROW
    col = cutlass.Int64(k_tile) * _COLS_PER_CTA + cutlass.Int64(chunk) * _VEC
    q_row_stride = cutlass.Int64(k_tiles) * _COLS_PER_CTA
    # Every pass loads unconditionally (rows past R re-read row R - 1 and store
    # nothing), so the passes' loads are all in flight before the first use.
    last = rows - 1
    raw = cute.make_rmem_tensor((passes * _VEC,), cutlass.BFloat16)
    for p in cutlass.range_constexpr(passes):
        m_ld = row_tile * (passes * _ROWS_PER_PASS) + p * _ROWS_PER_PASS + row_in_pass
        if m_ld > last:
            m_ld = last
        src = _load_bf16x8(x.iterator, cutlass.Int64(m_ld) * x_row_stride + col)
        for i in cutlass.range_constexpr(_VEC):
            raw[p * _VEC + i] = src[i]
    for p in cutlass.range_constexpr(passes):
        m = row_tile * (passes * _ROWS_PER_PASS) + p * _ROWS_PER_PASS + row_in_pass
        vals = cute.make_rmem_tensor((_VEC,), cutlass.Float32)
        local_max = cutlass.Float32(0.0)
        for i in cutlass.range_constexpr(_VEC):
            vals[i] = raw[p * _VEC + i].to(cutlass.Float32)
            a = cute.math.absf(vals[i])
            if a > local_max:
                local_max = a
        # 4 consecutive lanes hold one 32-element scale block (one row); every lane
        # of the warp reaches the shuffles.
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
        if m < rows and cutlass.const_expr(vec_cvt):
            # 8 values -> 8 e4m3 bytes in one vector conversion and one 8-byte store
            q8 = cute.make_rmem_tensor((_VEC,), cutlass.Float8E4M3FN)
            q8.store((vals.load() * inv_scale).to(cutlass.Float8E4M3FN))
            dst = cute.make_tensor(
                cute.recast_ptr(
                    q.iterator + cute.assume(cutlass.Int64(m) * q_row_stride + col, _VEC),
                    dtype=cutlass.Float8E4M3FN,
                ),
                cute.make_layout((_VEC,)),
            )
            cute.autovec_copy(q8, dst)
        if m < rows and not cutlass.const_expr(vec_cvt):
            words = cute.make_rmem_tensor((2,), cutlass.Int32)
            for w in cutlass.range_constexpr(2):
                packed = cutlass.Int32(0)
                for j in cutlass.range_constexpr(4):
                    byte = (
                        (vals[w * 4 + j] * inv_scale).to(cutlass.Float8E4M3FN).bitcast(cutlass.Int8)
                    )
                    packed = packed | ((cutlass.Int32(byte) & cutlass.Int32(0xFF)) << (8 * j))
                words[w] = packed
            base = cutlass.Int64(m) * q_row_stride + col
            for w in cutlass.range_constexpr(2):
                ptr = cute.recast_ptr((q.iterator + base + w * 4).align(4), dtype=cutlass.Int32)
                ptr.store(words[w])
        if chunk % 4 == 0:
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


@cute.kernel
def glm52_prefill_act_quant_mxf8_rows_kernel(
    x: cute.Tensor,
    q: cute.Tensor,
    sf: cute.Tensor,
    row0: cutlass.Int32,
    rows: cutlass.Int32,
    k_tiles: cutlass.Int32,
    x_row_stride: cutlass.Int64,
) -> None:
    """Rows ``row0 .. row0 + rows - 1`` of an MXF8 operand whose rows ``< row0`` another
    producer wrote (the FP8 context MHA's epilogue writes the context rows of the o_proj
    operand; this writes a mixed step's generation rows): ``x`` holds those ``rows`` BF16
    rows; ``q`` / ``sf`` are the whole operand's buffers.  Also writes code 127 into the
    scale rows ``row0 + rows .. padded_rows(row0 + rows) - 1``.  One 16-row pass per CTA
    (``rows`` is small), the per-row bytes are the kernel above's."""
    row_tile, k_tile, _ = cute.arch.block_idx()
    grid_x, grid_y, _ = cute.arch.grid_dim()
    tidx, _, _ = cute.arch.thread_idx()
    if rows > cutlass.Int32(0):
        chunk = tidx % _THREADS_PER_ROW
        m_local = row_tile * _ROWS_PER_PASS + tidx // _THREADS_PER_ROW
        m_ld = m_local
        if m_ld > rows - 1:
            m_ld = rows - 1
        col = cutlass.Int64(k_tile) * _COLS_PER_CTA + cutlass.Int64(chunk) * _VEC
        src = _load_bf16x8(x.iterator, cutlass.Int64(m_ld) * x_row_stride + col)
        vals = cute.make_rmem_tensor((_VEC,), cutlass.Float32)
        for i in cutlass.range_constexpr(_VEC):
            vals[i] = src[i].to(cutlass.Float32)
        m = cutlass.Int64(row0 + m_local)
        mxf8_quant_store_x8(
            vals,
            q.iterator,
            m * cutlass.Int64(k_tiles) * _COLS_PER_CTA + col,
            sf.iterator,
            m,
            col,
            cutlass.Int64(k_tiles),
            m_local < rows,
        )
    mxf8_fill_pad_scales(
        sf.iterator,
        row0 + rows,
        k_tiles,
        (k_tile * grid_x + row_tile) * _THREADS + tidx,
        grid_x * grid_y * _THREADS,
    )


@cute.jit
def _launch_rows(
    x: cute.Tensor,
    q: cute.Tensor,
    sf: cute.Tensor,
    row0: cutlass.Int32,
    rows: cutlass.Int32,
    k_tiles: cutlass.Int32,
    x_row_stride: cutlass.Int64,
    row_tiles: cutlass.Int32,
    stream: cuda.CUstream,
) -> None:
    glm52_prefill_act_quant_mxf8_rows_kernel(x, q, sf, row0, rows, k_tiles, x_row_stride).launch(
        grid=(row_tiles, k_tiles, 1), block=(_THREADS, 1, 1), stream=stream
    )


@cute.jit
def _launch(
    x: cute.Tensor,
    q: cute.Tensor,
    sf: cute.Tensor,
    rows: cutlass.Int32,
    k_tiles: cutlass.Int32,
    x_row_stride: cutlass.Int64,
    row_tiles: cutlass.Int32,
    stream: cuda.CUstream,
    passes: cutlass.Constexpr[int],
    vec_cvt: cutlass.Constexpr[bool],
    pdl: cutlass.Constexpr[bool],
) -> None:
    glm52_prefill_act_quant_mxf8_kernel(
        x, q, sf, rows, k_tiles, x_row_stride, passes, vec_cvt, pdl
    ).launch(grid=(row_tiles, k_tiles, 1), block=(_THREADS, 1, 1), stream=stream, use_pdl=pdl)


@dataclass(frozen=True)
class QuantFn:
    """A compiled instance and its rows per CTA (``16 * passes``)."""

    fn: CompiledKernel
    rows_per_cta: int


def compile_quant(
    device_index: int = 0,
    passes: int = DEFAULT_PASSES,
    vec_cvt: bool = DEFAULT_VEC_CVT,
    pdl: bool = False,
) -> QuantFn:
    """``QuantFn``; ``fn(x[R,K] bf16 (pitch % 8 == 0), q[R,K] u8, sf[sf_bytes(R,K)] u8, R,
    K // 128, x_pitch, padded_rows(R) // rows_per_cta, stream)``.  ``pdl``: launched as a
    programmatic dependent that releases its own dependent at entry (decode chains)."""
    if passes not in PASSES_CHOICES:
        raise ValueError(f"act quant mxf8: passes must be one of {PASSES_CHOICES}, got {passes}")
    key = (device_index, passes, vec_cvt, pdl)
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
            vec_cvt,
            pdl,
            options="--enable-tvm-ffi",
        )
        _COMPILED[key] = QuantFn(fn, passes * _ROWS_PER_PASS)
    return _COMPILED[key]


_COMPILED_ROWS: dict[int, CompiledKernel] = {}


def compile_quant_rows(device_index: int = 0) -> CompiledKernel:
    """``glm52_prefill_act_quant_mxf8_rows_kernel``: ``fn(x[R,K] bf16, q[T,K] u8, sf u8,
    row0, R, K // 128, x_pitch, row_tiles, stream)`` (``quantize_rows`` builds the call)."""
    if device_index not in _COMPILED_ROWS:
        z32, z64 = cutlass.Int32(0), cutlass.Int64(0)
        _COMPILED_ROWS[device_index] = cute.compile(
            _launch_rows,
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
            z32,
            z64,
            z32,
            make_fake_stream(),
            options="--enable-tvm-ffi",
        )
    return _COMPILED_ROWS[device_index]


def quantize_rows(
    fn: CompiledKernel,
    x: torch.Tensor,
    q: torch.Tensor,
    sf: torch.Tensor,
    row0: int,
) -> None:
    """Rows ``row0 .. row0 + R - 1`` of the MXF8 operand ``(q [T, K], sf)`` from BF16
    ``x [R, K]`` (``R`` may be 0: only the padded scale rows past ``row0`` are written), and
    code 127 for the scale rows from ``row0 + R`` to ``padded_rows(row0 + R)``."""
    rows, k = x.shape
    if (
        not x.is_cuda
        or x.dtype != torch.bfloat16
        or x.ndim != 2
        or x.stride(1) != 1
        or (rows > 0 and (x.stride(0) % _VEC != 0 or x.data_ptr() % _ALIGNMENT_BYTES != 0))
        or k % _COLS_PER_CTA != 0
        or q.dtype != torch.uint8
        or q.ndim != 2
        or q.shape[1] != k
        or not q.is_contiguous()
        or q.shape[0] < row0 + rows
        or sf.dtype != torch.uint8
        or not sf.is_contiguous()
        or sf.numel() < sf_bytes(row0 + rows, k)
        or row0 < 0
    ):
        raise ValueError(
            f"act quant mxf8 rows: x BF16 [R, K] (K % 128 == 0) into q [>= row0 + R, K] u8 + "
            f"sf >= sf_bytes(row0 + R, K); got x {tuple(x.shape)} q {tuple(q.shape)} row0 {row0}"
        )
    row_tiles = max(1, (rows + _ROWS_PER_PASS - 1) // _ROWS_PER_PASS)
    fn(
        x,
        q,
        sf,
        row0,
        rows,
        k // _COLS_PER_CTA,
        x.stride(0) if rows > 0 else _VEC,
        row_tiles,
        _raw_stream(x.device),
    )


def padded_rows(rows: int) -> int:
    return (rows + 127) // 128 * 128


def sf_bytes(rows: int, k: int) -> int:
    """Bytes of the blocked e8m0 scale tensor for ``rows x k`` (k % 128 == 0)."""
    return padded_rows(rows) * (k // _SF_VEC)


def quantize(
    fn: QuantFn,
    x: torch.Tensor,
    q: torch.Tensor | None = None,
    sf: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize BF16 ``x [R, K]`` into ``q [R, K]`` (uint8 e4m3 bytes) and ``sf``
    (uint8 e8m0 bytes, blocked layout).  Allocates the outputs when not given."""
    if (
        not x.is_cuda
        or x.dtype != torch.bfloat16
        or x.ndim != 2
        or x.stride(1) != 1
        or x.stride(0) % _VEC != 0
        or x.data_ptr() % _ALIGNMENT_BYTES != 0
        or x.shape[1] % _COLS_PER_CTA != 0
        or x.shape[0] < 1
    ):
        raise ValueError(
            f"act quant mxf8: x must be BF16 [R, K] with K % 128 == 0, unit column stride, "
            f"row pitch % 8 == 0 and 16-byte alignment; got {tuple(x.shape)} {x.stride()} {x.dtype}"
        )
    rows, k = x.shape
    if q is None:
        q = torch.empty(rows, k, dtype=torch.uint8, device=x.device)
    if sf is None:
        sf = torch.empty(sf_bytes(rows, k), dtype=torch.uint8, device=x.device)
    if (
        q.dtype != torch.uint8
        or tuple(q.shape) != (rows, k)
        or not q.is_contiguous()
        or q.device != x.device
        or sf.dtype != torch.uint8
        or sf.numel() < sf_bytes(rows, k)
        or not sf.is_contiguous()
        or sf.device != x.device
        or sf.data_ptr() % _ALIGNMENT_BYTES != 0
        or q.data_ptr() % _ALIGNMENT_BYTES != 0
    ):
        raise ValueError("act quant mxf8: q must be uint8 [R, K] contiguous; sf uint8 >= sf_bytes")
    fn.fn(
        x,
        q,
        sf,
        rows,
        k // _COLS_PER_CTA,
        x.stride(0),
        padded_rows(rows) // fn.rows_per_cta,
        _raw_stream(x.device),
    )
    return q, sf


# ----------------------------------------------------------------------------- reference
def reference(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch twin of the kernel (same fp32 operations): ``(q [R, K] uint8, sf uint8)``."""
    rows, k = x.shape
    xf = x.float().reshape(rows, k // _SF_VEC, _SF_VEC)
    amax = xf.abs().amax(-1).clamp_min(_MIN_AMAX)
    scale = amax * torch.tensor(_INV_E4M3_MAX, dtype=torch.float32, device=x.device)
    bits = scale.view(torch.int32)
    exp_bits = (bits & 0x7F800000) + ((bits & 0x007FFFFF) != 0).to(torch.int32) * 0x00800000
    code = (exp_bits >> 23).to(torch.int32)
    inv = 1.0 / exp_bits.view(torch.float32)
    q = (xf * inv[..., None]).to(torch.float8_e4m3fn).view(torch.uint8).reshape(rows, k)
    rp, kf = padded_rows(rows), k // _SF_VEC
    full = torch.full((rp, kf), 127, dtype=torch.int32, device=x.device)
    full[:rows] = code
    m = torch.arange(rp, device=x.device)[:, None]
    f = torch.arange(kf, device=x.device)[None, :]
    off = (
        (m % 32) * 16 + ((m // 32) % 4) * 4 + f % 4 + (f // 4) * 512 + (m // 128) * 512 * (k // 128)
    )
    sf = torch.empty(rp * kf, dtype=torch.uint8, device=x.device)
    sf[off.flatten()] = full.flatten().to(torch.uint8)
    return q, sf


def dequantize(q: torch.Tensor, sf: torch.Tensor, rows: int, k: int) -> torch.Tensor:
    """FP32 values of ``(q, sf)`` (for numerics checks)."""
    kf = k // _SF_VEC
    m = torch.arange(rows, device=q.device)[:, None]
    f = torch.arange(kf, device=q.device)[None, :]
    off = (
        (m % 32) * 16 + ((m // 32) % 4) * 4 + f % 4 + (f // 4) * 512 + (m // 128) * 512 * (k // 128)
    )
    code = sf[off].to(torch.int32)
    s = torch.exp2((code - 127).float())
    v = q.view(torch.float8_e4m3fn).float().reshape(rows, kf, _SF_VEC)
    return (v * s[..., None]).reshape(rows, k)
