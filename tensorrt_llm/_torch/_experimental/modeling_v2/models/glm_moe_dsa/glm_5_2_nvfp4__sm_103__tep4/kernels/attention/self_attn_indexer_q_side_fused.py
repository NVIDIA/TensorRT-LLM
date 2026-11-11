# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 DSA indexer, q-side post-projection fusion (decode and prefill rows).

One launch replaces the four launches that follow the indexer ``q_b`` GEMM on a
full-indexer layer once the split-output pre-indexer projection has produced the
packed head weights::

    q_pe, q_nope = split(q, [64, 64], -1)  # q bf16 [T, 32, 128]
    q_pe = rope_inplace(q_pe, pos, cos_sin, is_neox)  # (1) flashinfer rope
    q_fp8, q_scale = fused_cat_fp8(q_pe, q_nope, use_ue8m0)  # (2) [pe|nope] e4m3 ue8m0
    weights = (wq * q_scale.view(T, 32)) * c  # (3)(4) two fp32 muls

ABI (all tensors CUDA, on the same device; ``T`` and the table length are runtime):

    inputs
      q          bf16  [T, 32, 128]  contiguous (strides (4096, 128, 1)), 16 B aligned;
                                     read only -- the rotated PE half is NOT written
                                     back (nothing downstream reads it)
      pos        int32 [T]           contiguous; caller contract
                                     0 <= pos < cos_sin.shape[0], not validated on
                                     the device path
      cos_sin    fp32  [max_pos, 64] contiguous, 16 B aligned: ``[cos(32) | sin(32)]``
                                     per position (the indexer rope table, theta
                                     already applied)
      wq         fp32  [T, 32]       contiguous (the packed raw head weights)
      c          float               ``idx_weight_scale = Di**-0.5 * Hi**-0.5``
    outputs (caller-owned, ``out=`` style, must not alias any input)
      q_fp8      e4m3  [T, 32, 128]  contiguous, 16 B aligned: ``[pe | nope]`` codes
      weights    fp32  [T, 32]       contiguous -> stride (32, 1),
                                     ``(wq * q_scale) * c``

    ``q_scale`` (one ue8m0 power-of-two fp32 scale per (t, h)) is consumed inside
    ``weights`` and not emitted, exactly as the forward does today.

Mapping: one (token, head) row of 128 bf16 = 256 B per eight lanes, four rows per
warp, ``rows_per_block`` rows per CTA (default 16 -> 128 threads).  Lane ``i`` of a row
owns PE dims ``8i..8i+7`` and NOPE dims ``64+8i..64+8i+7`` as two 16 B loads; the NeoX
partner 32 dims away is an xor-4 lane shuffle inside the row, the interleaved partner
is the neighbouring register.  Per-row amax is a three-step butterfly over the eight
lanes; lane 0 of the row writes ``weights``.

Precision boundaries reproduced from the reference chain, in order: bf16 inputs are
flushed to signed zero when subnormal (flashinfer's fast-math cast), rope is computed as
``fma(x, cos, (+/- partner) * sin)`` in fp32 and **rounded to bf16** (the materialised
in-place write of the reference), amax / scale / codes are then computed from those
bf16 values exactly as the TRT-LLM ``fused_cat_fp8`` op does (``scale = max(amax, 1e-12)
/ 448`` rounded up to a power of two, ``code = e4m3(x * (1/scale))``), and ``weights``
is two separately rounded fp32 products, left-associated.  ``q_fp8`` and ``weights``
are bit-identical to the torch/flashinfer/TRT-LLM chain.

``is_neox``: this checkpoint's forward passes ``indexer_is_neox = not
indexer_rope_interleave = False`` (adjacent pairs rotate together); both modes are compiled
and verified because a wrong mode is a silent accuracy defect (max abs ~7.8 vs 2.4e-4).

PDL: ``compile_indexer_q_side_fused(use_pdl=True)`` issues ``griddepcontrol.wait``
before the first global read and launches with the programmatic-stream-serialisation
attribute; like the fused k entry it relies on kernel completion to release the next
dependent (no explicit ``launch_dependents``).  The default builds the plain kernel.

Kernel name:
``glm_self_attn_indexer_q_side_fused_<neox|interleave>_rpb<N>_pdl<0|1>``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

_WARP_SIZE = 32
_NUM_HEADS = 32
_HEAD_DIM = 128
_PE_DIM = 64
_ROPE_HALF = _PE_DIM // 2
_LANES_PER_ROW = 8
_ROWS_PER_WARP = _WARP_SIZE // _LANES_PER_ROW
_VALUES_PER_LANE = _PE_DIM // _LANES_PER_ROW  # 8 PE + 8 NOPE values per lane
_BF16_VEC_BYTES = _VALUES_PER_LANE * 2
_FP32_VEC_VALUES = 4
_FP32_VEC_BYTES = _FP32_VEC_VALUES * 4
_FP8_MAX = 448.0
_INV_FP8_MAX = 1.0 / _FP8_MAX
_MIN_AMAX = 1.0e-12
_DEFAULT_ROWS_PER_BLOCK = 16
_ALLOWED_ROWS_PER_BLOCK = (4, 8, 16, 32, 64)
# What the GLM-5.2 forward passes (``modeling.py``: ``indexer_is_neox = not
# indexer_rope_interleave``; the checkpoint sets ``indexer_rope_interleave: true``).
CHECKPOINT_IS_NEOX = False
KERNEL_NAME_PREFIX = "glm_self_attn_indexer_q_side_fused"


@dataclass(frozen=True)
class QSideFusedConfig:
    """Compile-time specialisation of the fused q-side kernel."""

    is_neox: bool = CHECKPOINT_IS_NEOX
    use_ue8m0: bool = True
    rows_per_block: int = _DEFAULT_ROWS_PER_BLOCK
    use_pdl: bool = False

    def __post_init__(self) -> None:
        if self.rows_per_block not in _ALLOWED_ROWS_PER_BLOCK:
            raise ValueError(
                f"rows_per_block must be one of {_ALLOWED_ROWS_PER_BLOCK}, "
                f"got {self.rows_per_block}"
            )

    @property
    def threads(self) -> int:
        return self.rows_per_block // _ROWS_PER_WARP * _WARP_SIZE

    @property
    def kernel_name(self) -> str:
        mode = "neox" if self.is_neox else "interleave"
        return f"{KERNEL_NAME_PREFIX}_{mode}_rpb{self.rows_per_block}_pdl{int(self.use_pdl)}"


_COMPILE_CACHE: dict[tuple[int, QSideFusedConfig], Callable] = {}


@cute.jit
def _bf16_to_fp32_flashinfer_ftz(value: cutlass.BFloat16) -> cutlass.Float32:
    """Match flashinfer's fast-math BF16 -> FP32 boundary: subnormals become +/-0."""
    bits = value.bitcast(cutlass.Uint16)
    converted = value.to(cutlass.Float32)
    if (bits & cutlass.Uint16(0x7F80)) == cutlass.Uint16(0):
        signed_zero_bits = bits & cutlass.Uint16(0x8000)
        converted = signed_zero_bits.bitcast(cutlass.BFloat16).to(cutlass.Float32)
    return converted


@cute.jit
def _round_bf16(value: cutlass.Float32) -> cutlass.Float32:
    """The materialised BF16 boundary of the in-place rope write."""
    return value.to(cutlass.BFloat16).to(cutlass.Float32)


@cute.jit
def _row8_reduction_max(value: cutlass.Float32) -> cutlass.Float32:
    """Max over the eight lanes of one row (butterfly stays inside the row)."""
    active_mask = cute.arch.activemask()
    value = cutlass.max(value, cute.arch.shuffle_sync_bfly(value, 4, mask=active_mask))
    value = cutlass.max(value, cute.arch.shuffle_sync_bfly(value, 2, mask=active_mask))
    return cutlass.max(value, cute.arch.shuffle_sync_bfly(value, 1, mask=active_mask))


@cute.jit
def _quantize_pack_fp8x4(
    x0: cutlass.Float32,
    x1: cutlass.Float32,
    x2: cutlass.Float32,
    x3: cutlass.Float32,
    inv_scale: cutlass.Float32,
) -> cutlass.Int32:
    """Quantise four values to E4M3 and pack the bytes little-endian into one word."""
    q0 = (x0 * inv_scale).to(cutlass.Float8E4M3FN).bitcast(cutlass.Int8)
    q1 = (x1 * inv_scale).to(cutlass.Float8E4M3FN).bitcast(cutlass.Int8)
    q2 = (x2 * inv_scale).to(cutlass.Float8E4M3FN).bitcast(cutlass.Int8)
    q3 = (x3 * inv_scale).to(cutlass.Float8E4M3FN).bitcast(cutlass.Int8)
    return (
        (cutlass.Int32(q0) & cutlass.Int32(0xFF))
        | ((cutlass.Int32(q1) & cutlass.Int32(0xFF)) << 8)
        | ((cutlass.Int32(q2) & cutlass.Int32(0xFF)) << 16)
        | ((cutlass.Int32(q3) & cutlass.Int32(0xFF)) << 24)
    )


@cute.jit
def _pack_words_i64(lo: cutlass.Int32, hi: cutlass.Int32) -> cutlass.Int64:
    """Two little-endian 32-bit words -> one 64-bit word (``lo`` at the low address)."""
    low = cutlass.Int64(lo) & cutlass.Int64(0xFFFFFFFF)
    return (cutlass.Int64(hi) << 32) | low


@cute.kernel
def indexer_q_side_fused_kernel(
    q: cute.Tensor,
    position_ids: cute.Tensor,
    cos_sin: cute.Tensor,
    wq: cute.Tensor,
    q_fp8: cute.Tensor,
    weights: cute.Tensor,
    rows: cutlass.Int32,
    weight_scale_factor: cutlass.Float32,
    IS_NEOX: cutlass.Constexpr[bool],
    USE_UE8M0: cutlass.Constexpr[bool],
    ROWS_PER_BLOCK: cutlass.Constexpr[int],
    USE_PDL: cutlass.Constexpr[bool],
) -> None:
    """Rope the PE half, quantise ``[pe | nope]`` to E4M3, finalise the head weight.

    ``rows = T * 32``; row ``r`` is token ``r // 32``, head ``r % 32``.  All tensors are
    addressed through their raw pointers as flat row-major storage.
    """
    block, _, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    warp_in_block = tidx // cutlass.Int32(_WARP_SIZE)
    lane = tidx % cutlass.Int32(_WARP_SIZE)
    row_in_warp = lane // cutlass.Int32(_LANES_PER_ROW)
    lane_in_row = lane % cutlass.Int32(_LANES_PER_ROW)
    row = (
        block * cutlass.Int32(ROWS_PER_BLOCK)
        + warp_in_block * cutlass.Int32(_ROWS_PER_WARP)
        + row_in_warp
    )

    if cutlass.const_expr(USE_PDL):
        # Nothing above touches global memory; every thread waits before its first
        # load of the predecessor GEMM's output (no CTA barrier on this path).
        cute.arch.griddepcontrol_wait()

    if row < rows:
        pe_col = lane_in_row * cutlass.Int32(_VALUES_PER_LANE)
        nope_col = pe_col + cutlass.Int32(_PE_DIM)
        q_base = row * cutlass.Int32(_HEAD_DIM)
        q_ptr = q.iterator.raw_ptr()
        pe_vals = (q_ptr + q_base + pe_col).load(count=_VALUES_PER_LANE, alignment=_BF16_VEC_BYTES)
        nope_vals = (q_ptr + q_base + nope_col).load(
            count=_VALUES_PER_LANE, alignment=_BF16_VEC_BYTES
        )
        x0 = _bf16_to_fp32_flashinfer_ftz(pe_vals[0])
        x1 = _bf16_to_fp32_flashinfer_ftz(pe_vals[1])
        x2 = _bf16_to_fp32_flashinfer_ftz(pe_vals[2])
        x3 = _bf16_to_fp32_flashinfer_ftz(pe_vals[3])
        x4 = _bf16_to_fp32_flashinfer_ftz(pe_vals[4])
        x5 = _bf16_to_fp32_flashinfer_ftz(pe_vals[5])
        x6 = _bf16_to_fp32_flashinfer_ftz(pe_vals[6])
        x7 = _bf16_to_fp32_flashinfer_ftz(pe_vals[7])
        n0 = nope_vals[0].to(cutlass.Float32)
        n1 = nope_vals[1].to(cutlass.Float32)
        n2 = nope_vals[2].to(cutlass.Float32)
        n3 = nope_vals[3].to(cutlass.Float32)
        n4 = nope_vals[4].to(cutlass.Float32)
        n5 = nope_vals[5].to(cutlass.Float32)
        n6 = nope_vals[6].to(cutlass.Float32)
        n7 = nope_vals[7].to(cutlass.Float32)

        token = row // cutlass.Int32(_NUM_HEADS)
        position = (position_ids.iterator + token).load()
        table_base = cutlass.Int64(position) * cutlass.Int64(_PE_DIM)
        table_ptr = cos_sin.iterator.raw_ptr()

        if cutlass.const_expr(IS_NEOX):
            # Partner is 32 dims = 4 lanes away; every lane of the row shuffles before
            # any lane branches on its half.
            active_mask = cute.arch.activemask()
            source_lane = row_in_warp * cutlass.Int32(_LANES_PER_ROW) + (
                lane_in_row ^ cutlass.Int32(4)
            )
            p0 = cute.arch.shuffle_sync(x0, source_lane, mask=active_mask)
            p1 = cute.arch.shuffle_sync(x1, source_lane, mask=active_mask)
            p2 = cute.arch.shuffle_sync(x2, source_lane, mask=active_mask)
            p3 = cute.arch.shuffle_sync(x3, source_lane, mask=active_mask)
            p4 = cute.arch.shuffle_sync(x4, source_lane, mask=active_mask)
            p5 = cute.arch.shuffle_sync(x5, source_lane, mask=active_mask)
            p6 = cute.arch.shuffle_sync(x6, source_lane, mask=active_mask)
            p7 = cute.arch.shuffle_sync(x7, source_lane, mask=active_mask)

            table_dim = pe_col
            if table_dim >= cutlass.Int32(_ROPE_HALF):
                table_dim = table_dim - cutlass.Int32(_ROPE_HALF)
            cos_base = table_base + cutlass.Int64(table_dim)
            sin_base = cos_base + cutlass.Int64(_ROPE_HALF)
            cos_lo = (table_ptr + cos_base).load(count=_FP32_VEC_VALUES, alignment=_FP32_VEC_BYTES)
            cos_hi = (table_ptr + cos_base + cutlass.Int64(_FP32_VEC_VALUES)).load(
                count=_FP32_VEC_VALUES, alignment=_FP32_VEC_BYTES
            )
            sin_lo = (table_ptr + sin_base).load(count=_FP32_VEC_VALUES, alignment=_FP32_VEC_BYTES)
            sin_hi = (table_ptr + sin_base + cutlass.Int64(_FP32_VEC_VALUES)).load(
                count=_FP32_VEC_VALUES, alignment=_FP32_VEC_BYTES
            )
            # flashinfer: vec[i] * cos[i] + (+/- partner[i]) * sin[i], contracted to
            # one FMA with the (signed) partner product rounded first.
            s0 = p0 * sin_lo[0]
            s1 = p1 * sin_lo[1]
            s2 = p2 * sin_lo[2]
            s3 = p3 * sin_lo[3]
            s4 = p4 * sin_hi[0]
            s5 = p5 * sin_hi[1]
            s6 = p6 * sin_hi[2]
            s7 = p7 * sin_hi[3]
            y0 = cute.math.fma(x0, cos_lo[0], s0)
            y1 = cute.math.fma(x1, cos_lo[1], s1)
            y2 = cute.math.fma(x2, cos_lo[2], s2)
            y3 = cute.math.fma(x3, cos_lo[3], s3)
            y4 = cute.math.fma(x4, cos_hi[0], s4)
            y5 = cute.math.fma(x5, cos_hi[1], s5)
            y6 = cute.math.fma(x6, cos_hi[2], s6)
            y7 = cute.math.fma(x7, cos_hi[3], s7)
            if pe_col < cutlass.Int32(_ROPE_HALF):
                y0 = cute.math.fma(x0, cos_lo[0], -s0)
                y1 = cute.math.fma(x1, cos_lo[1], -s1)
                y2 = cute.math.fma(x2, cos_lo[2], -s2)
                y3 = cute.math.fma(x3, cos_lo[3], -s3)
                y4 = cute.math.fma(x4, cos_hi[0], -s4)
                y5 = cute.math.fma(x5, cos_hi[1], -s5)
                y6 = cute.math.fma(x6, cos_hi[2], -s6)
                y7 = cute.math.fma(x7, cos_hi[3], -s7)
        else:
            # Interleaved: element e pairs with e ^ 1 and uses cos[e // 2], sin[e // 2];
            # this lane's four table values start at pe_col // 2 = 4 * lane_in_row.
            cos_base = table_base + cutlass.Int64(pe_col // cutlass.Int32(2))
            sin_base = cos_base + cutlass.Int64(_ROPE_HALF)
            cos_v = (table_ptr + cos_base).load(count=_FP32_VEC_VALUES, alignment=_FP32_VEC_BYTES)
            sin_v = (table_ptr + sin_base).load(count=_FP32_VEC_VALUES, alignment=_FP32_VEC_BYTES)
            y0 = cute.math.fma(x0, cos_v[0], -(x1 * sin_v[0]))
            y1 = cute.math.fma(x1, cos_v[0], x0 * sin_v[0])
            y2 = cute.math.fma(x2, cos_v[1], -(x3 * sin_v[1]))
            y3 = cute.math.fma(x3, cos_v[1], x2 * sin_v[1])
            y4 = cute.math.fma(x4, cos_v[2], -(x5 * sin_v[2]))
            y5 = cute.math.fma(x5, cos_v[2], x4 * sin_v[2])
            y6 = cute.math.fma(x6, cos_v[3], -(x7 * sin_v[3]))
            y7 = cute.math.fma(x7, cos_v[3], x6 * sin_v[3])

        # The reference writes the rotated PE half back as bf16 and re-reads it.
        r0 = _round_bf16(y0)
        r1 = _round_bf16(y1)
        r2 = _round_bf16(y2)
        r3 = _round_bf16(y3)
        r4 = _round_bf16(y4)
        r5 = _round_bf16(y5)
        r6 = _round_bf16(y6)
        r7 = _round_bf16(y7)

        local_max = cute.math.absf(r0)
        local_max = cutlass.max(local_max, cute.math.absf(r1))
        local_max = cutlass.max(local_max, cute.math.absf(r2))
        local_max = cutlass.max(local_max, cute.math.absf(r3))
        local_max = cutlass.max(local_max, cute.math.absf(r4))
        local_max = cutlass.max(local_max, cute.math.absf(r5))
        local_max = cutlass.max(local_max, cute.math.absf(r6))
        local_max = cutlass.max(local_max, cute.math.absf(r7))
        local_max = cutlass.max(local_max, cute.math.absf(n0))
        local_max = cutlass.max(local_max, cute.math.absf(n1))
        local_max = cutlass.max(local_max, cute.math.absf(n2))
        local_max = cutlass.max(local_max, cute.math.absf(n3))
        local_max = cutlass.max(local_max, cute.math.absf(n4))
        local_max = cutlass.max(local_max, cute.math.absf(n5))
        local_max = cutlass.max(local_max, cute.math.absf(n6))
        local_max = cutlass.max(local_max, cute.math.absf(n7))
        amax = _row8_reduction_max(local_max)
        safe_amax = amax
        if safe_amax < cutlass.Float32(_MIN_AMAX):
            safe_amax = cutlass.Float32(_MIN_AMAX)

        scale = safe_amax * cutlass.Float32(_INV_FP8_MAX)
        if cutlass.const_expr(USE_UE8M0):
            scale_bits = scale.bitcast(cutlass.Int32)
            mantissa = scale_bits & cutlass.Int32(0x007FFFFF)
            exp_bits = scale_bits & cutlass.Int32(0x7F800000)
            if mantissa != cutlass.Int32(0):
                exp_bits += cutlass.Int32(0x00800000)
            scale = exp_bits.bitcast(cutlass.Float32)
            # 1 / 2^k is exact: negate the biased exponent.
            inv_scale_bits = cutlass.Int32(0x7F000000) - exp_bits
            inv_scale = inv_scale_bits.bitcast(cutlass.Float32)
        else:
            inv_scale = cutlass.Float32(1.0) / scale

        pe_word = _pack_words_i64(
            _quantize_pack_fp8x4(r0, r1, r2, r3, inv_scale),
            _quantize_pack_fp8x4(r4, r5, r6, r7, inv_scale),
        )
        nope_word = _pack_words_i64(
            _quantize_pack_fp8x4(n0, n1, n2, n3, inv_scale),
            _quantize_pack_fp8x4(n4, n5, n6, n7, inv_scale),
        )
        out_base = row * cutlass.Int32(_HEAD_DIM)
        pe_out_ptr = cute.recast_ptr(
            (q_fp8.iterator + out_base + pe_col).align(8), dtype=cutlass.Int64
        )
        pe_out_ptr.store(pe_word)
        nope_out_ptr = cute.recast_ptr(
            (q_fp8.iterator + out_base + nope_col).align(8), dtype=cutlass.Int64
        )
        nope_out_ptr.store(nope_word)

        if lane_in_row == cutlass.Int32(0):
            # Two separately rounded fp32 products, left-associated, as the forward's
            # ``mul(mul(wq, q_scale), c)``.
            raw_weight = (wq.iterator + row).load()
            product = raw_weight * scale
            (weights.iterator + row).store(product * weight_scale_factor)


@cute.jit
def _launch_indexer_q_side_fused(
    q: cute.Tensor,
    position_ids: cute.Tensor,
    cos_sin: cute.Tensor,
    wq: cute.Tensor,
    q_fp8: cute.Tensor,
    weights: cute.Tensor,
    num_tokens: cutlass.Int32,
    weight_scale_factor: cutlass.Float32,
    stream: cuda.CUstream,
    IS_NEOX: cutlass.Constexpr[bool],
    USE_UE8M0: cutlass.Constexpr[bool],
    ROWS_PER_BLOCK: cutlass.Constexpr[int],
    USE_PDL: cutlass.Constexpr[bool],
) -> None:
    rows = num_tokens * cutlass.Int32(_NUM_HEADS)
    threads = ROWS_PER_BLOCK // _ROWS_PER_WARP * _WARP_SIZE
    indexer_q_side_fused_kernel(
        q,
        position_ids,
        cos_sin,
        wq,
        q_fp8,
        weights,
        rows,
        weight_scale_factor,
        IS_NEOX,
        USE_UE8M0,
        ROWS_PER_BLOCK,
        USE_PDL,
    ).launch(
        grid=(cute.ceil_div(rows, ROWS_PER_BLOCK), 1, 1),
        block=(threads, 1, 1),
        stream=stream,
        use_pdl=USE_PDL,
    )


def compile_indexer_q_side_fused(
    device_id: int = 0,
    is_neox: bool = CHECKPOINT_IS_NEOX,
    use_ue8m0: bool = True,
    rows_per_block: int = _DEFAULT_ROWS_PER_BLOCK,
    *,
    use_pdl: bool = False,
) -> Callable:
    """Compile (or fetch) one specialisation; allocates no GPU tensors.

    The compiled callable takes ``(q, position_ids, cos_sin, wq, q_fp8_out,
    weights_out, num_tokens: int, weight_scale_factor: float, stream: cuda.CUstream)``
    and is what :func:`indexer_q_side_fused` launches.
    """
    config = QSideFusedConfig(
        is_neox=bool(is_neox),
        use_ue8m0=bool(use_ue8m0),
        rows_per_block=int(rows_per_block),
        use_pdl=bool(use_pdl),
    )
    key = (int(device_id), config)
    cached = _COMPILE_CACHE.get(key)
    if cached is not None:
        return cached

    cutlass.cuda.initialize_cuda_context(device_id)
    sym_tokens = cute.sym_int32()
    sym_positions = cute.sym_int32()
    fake_q = make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_tokens, _NUM_HEADS, _HEAD_DIM),
        stride_order=(2, 1, 0),
        assumed_align=_BF16_VEC_BYTES,
    )
    fake_pos = make_fake_compact_tensor(cutlass.Int32, (sym_tokens,), assumed_align=4)
    fake_cos_sin = make_fake_compact_tensor(
        cutlass.Float32,
        (sym_positions, _PE_DIM),
        stride_order=(1, 0),
        assumed_align=_FP32_VEC_BYTES,
    )
    fake_wq = make_fake_compact_tensor(
        cutlass.Float32, (sym_tokens, _NUM_HEADS), stride_order=(1, 0), assumed_align=4
    )
    fake_q_fp8 = make_fake_compact_tensor(
        cutlass.Float8E4M3FN,
        (sym_tokens, _NUM_HEADS, _HEAD_DIM),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    fake_weights = make_fake_compact_tensor(
        cutlass.Float32, (sym_tokens, _NUM_HEADS), stride_order=(1, 0), assumed_align=4
    )
    indexer_q_side_fused_kernel.set_name_prefix(config.kernel_name, remove_cutlass_symbol=True)
    compiled = cute.compile(
        _launch_indexer_q_side_fused,
        fake_q,
        fake_pos,
        fake_cos_sin,
        fake_wq,
        fake_q_fp8,
        fake_weights,
        cutlass.Int32(0),
        cutlass.Float32(1.0),
        make_fake_stream(),
        config.is_neox,
        config.use_ue8m0,
        config.rows_per_block,
        config.use_pdl,
        options="--enable-tvm-ffi",
    )
    _COMPILE_CACHE[key] = compiled
    return compiled


def _byte_span(tensor: torch.Tensor) -> tuple[int, int]:
    begin = tensor.data_ptr()
    return begin, begin + tensor.numel() * tensor.element_size()


def _overlaps(left: torch.Tensor, right: torch.Tensor) -> bool:
    lb, le = _byte_span(left)
    rb, re = _byte_span(right)
    return left.device == right.device and max(lb, rb) < min(le, re)


def _require(tensor: torch.Tensor, name: str, dtype, shape: tuple[int, ...], align: int):
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor, got {tensor.device}")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must be {dtype}, got {tensor.dtype}")
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got stride {tuple(tensor.stride())}")
    if tensor.data_ptr() % align:
        raise ValueError(f"{name} must be {align}-byte aligned")


def validate_abi(
    q: torch.Tensor,
    position_ids: torch.Tensor,
    cos_sin: torch.Tensor,
    wq: torch.Tensor,
    q_fp8_out: torch.Tensor,
    weights_out: torch.Tensor,
) -> int:
    """Fail closed on every host-checkable property of the ABI; returns ``T``.

    Metadata only -- no device read, no synchronisation -- so it is safe inside a
    CUDA-graph capture.  The ``position_ids`` range is a caller contract.
    """
    if q.ndim != 3:
        raise ValueError(f"q must be [T, 32, 128], got {tuple(q.shape)}")
    num_tokens = int(q.shape[0])
    if num_tokens < 1:
        raise ValueError("q must hold at least one token")
    if cos_sin.ndim != 2 or cos_sin.shape[1] != _PE_DIM:
        raise ValueError(f"cos_sin must be [max_pos, {_PE_DIM}], got {tuple(cos_sin.shape)}")
    _require(q, "q", torch.bfloat16, (num_tokens, _NUM_HEADS, _HEAD_DIM), _BF16_VEC_BYTES)
    _require(position_ids, "position_ids", torch.int32, (num_tokens,), 4)
    _require(cos_sin, "cos_sin", torch.float32, tuple(cos_sin.shape), _FP32_VEC_BYTES)
    _require(wq, "wq", torch.float32, (num_tokens, _NUM_HEADS), 4)
    _require(
        q_fp8_out,
        "q_fp8_out",
        torch.float8_e4m3fn,
        (num_tokens, _NUM_HEADS, _HEAD_DIM),
        16,
    )
    _require(weights_out, "weights_out", torch.float32, (num_tokens, _NUM_HEADS), 4)
    device = q.device
    for name, tensor in (
        ("position_ids", position_ids),
        ("cos_sin", cos_sin),
        ("wq", wq),
        ("q_fp8_out", q_fp8_out),
        ("weights_out", weights_out),
    ):
        if tensor.device != device:
            raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    for out_name, out in (("q_fp8_out", q_fp8_out), ("weights_out", weights_out)):
        for in_name, inp in (
            ("q", q),
            ("position_ids", position_ids),
            ("cos_sin", cos_sin),
            ("wq", wq),
        ):
            if _overlaps(out, inp):
                raise ValueError(f"{out_name} must not alias {in_name}")
    if _overlaps(q_fp8_out, weights_out):
        raise ValueError("q_fp8_out and weights_out must not alias")
    return num_tokens


def indexer_q_side_fused(
    q: torch.Tensor,
    position_ids: torch.Tensor,
    cos_sin: torch.Tensor,
    wq: torch.Tensor,
    weight_scale_factor: float,
    q_fp8_out: torch.Tensor,
    weights_out: torch.Tensor,
    *,
    is_neox: bool = CHECKPOINT_IS_NEOX,
    use_ue8m0: bool = True,
    compiled_fn: Callable | None = None,
    rows_per_block: int = _DEFAULT_ROWS_PER_BLOCK,
    use_pdl: bool = False,
    validate: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Enqueue the fused q-side kernel on the current stream of ``q``'s device.

    Allocation-free and sync-free; writes ``q_fp8_out`` and ``weights_out`` and
    returns them.  ``compiled_fn`` (from :func:`compile_indexer_q_side_fused`) skips
    the cache lookup; when given, ``is_neox`` / ``use_ue8m0`` / ``rows_per_block`` /
    ``use_pdl`` must describe it.
    """
    if validate:
        num_tokens = validate_abi(q, position_ids, cos_sin, wq, q_fp8_out, weights_out)
    else:
        num_tokens = int(q.shape[0])
    if compiled_fn is None:
        compiled_fn = compile_indexer_q_side_fused(
            device_id=q.device.index if q.device.index is not None else 0,
            is_neox=is_neox,
            use_ue8m0=use_ue8m0,
            rows_per_block=rows_per_block,
            use_pdl=use_pdl,
        )
    stream = cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
    compiled_fn(
        q,
        position_ids,
        cos_sin,
        wq,
        q_fp8_out,
        weights_out,
        num_tokens,
        float(weight_scale_factor),
        stream,
    )
    return q_fp8_out, weights_out
