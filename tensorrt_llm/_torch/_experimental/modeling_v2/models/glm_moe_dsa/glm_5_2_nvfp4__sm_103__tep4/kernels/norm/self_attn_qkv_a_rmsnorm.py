# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CuTe DSL GLM self-attention q_a/kv_a RMSNorm.

GLM MLA produces q and compressed-kv latent slices from a wider projection:

    q_input  bf16 [T, 2048], usually row stride 2624
    kv_input bf16 [T,  512], usually row stride 2624

This kernel replaces the two independent FlashInfer RMSNorm launches with a
single fused launch. Each CTA normalizes one q or kv row. Accumulation is FP32
and outputs are compact BF16 tensors matching TRT-LLM's non-residual RMSNorm
path used by GLM self attention.

Option ``inplace_kv_view`` (default off): the outputs may be strided views
instead of compact tensors, and ``kv_output`` may be the exact ``[rows, 512]`` kv
view of the ``q|kv|k_pe`` projection itself (rows = the call's padded decode row
count, row pitch 2624), so the normalized KV is written in
place and the latent concat/pack launch is not needed. Same kernel; the
option selects the ``strided_dst`` constexpr, which takes the destination row
pitch as a runtime argument instead of the compact width. Numerics identical:
the per-row FP32 accumulation and BF16 rounding are the same instructions, only
the store address differs. Contract shared with the compact mode, plus the
exact-view alias validation (``_validate_output_aliases``) for the in-place
destination.

Two kernels serve ``qkv_a_rmsnorm_run``, selected by row count:

* rows <= 64 (decode): ``_qkv_a_rmsnorm_kernel`` -- one 256-thread CTA
  per row, CTA-wide reduction through shared memory.
* rows >= ``_LARGE_T_MIN_ROWS`` (prefill, T ~ 1k..8k):
  ``_qkv_a_rmsnorm_kernel_warp_row`` -- one warp per row, eight rows per CTA,
  warp-shuffle reduction only (no shared memory, no CTA barrier in the row
  path). Bit-identical to the CTA kernel: lane ``l`` of pass ``p`` owns the
  same eight columns as thread ``p * 32 + l`` of the CTA kernel, the per-pass
  butterfly sum equals the CTA kernel's ``warp_sums[p]``, and the passes are
  summed in the same order as thread 0 sums the eight warp partials.
"""

from typing import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir as _mlir_ir
from cutlass._mlir.dialects import llvm as _llvm
from cutlass._mlir.dialects import nvvm as _nvvm
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream, make_fake_tensor
from cutlass.cutlass_dsl import T as _T
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.experimental import primitives as _prims
from cutlass.experimental.primitives.nvvm_wrapper import FPRoundingMode as _FPRnd
from cutlass.memory import SmemAllocator

from ..common import self_attn_prefill_act_quant_mxf8 as _mxf8

_Q_DIM = 2048
_KV_DIM = 512
_PROJECTION_DIM = _Q_DIM + _KV_DIM + 64
_THREADS_PER_BLOCK = 256
_WARPS_PER_BLOCK = _THREADS_PER_BLOCK // 32
_WARP_SIZE = 32
_BF16_VEC_SIZE = 8
_ALIGNMENT_BYTES = 16
_DEFAULT_EPS = 1.0e-6
_COMPILE_CACHE: dict[tuple[int, float, bool, bool], Callable] = {}

# Large-T (prefill) path: decode cells are <= 64 rows and keep the CTA kernel.
_LARGE_T_MIN_ROWS = 65
_WARP_ROW_COMPILE_CACHE: dict[tuple[int, float, bool, bool], Callable] = {}


@cute.jit
def _warp_sum_f32(value):
    value = value + cute.arch.shuffle_sync_bfly(value, 16)
    value = value + cute.arch.shuffle_sync_bfly(value, 8)
    value = value + cute.arch.shuffle_sync_bfly(value, 4)
    value = value + cute.arch.shuffle_sync_bfly(value, 2)
    return value + cute.arch.shuffle_sync_bfly(value, 1)


@cute.jit
def _load_bf16x8(ptr, offset):
    values = cute.make_rmem_tensor((_BF16_VEC_SIZE,), cutlass.BFloat16)
    src = cute.make_tensor(ptr + offset, cute.make_layout((_BF16_VEC_SIZE,)))
    cute.autovec_copy(src, values)
    return values


@cute.jit
def _store_bf16x8(ptr, offset, values) -> None:
    dst = cute.make_tensor(ptr + offset, cute.make_layout((_BF16_VEC_SIZE,)))
    cute.autovec_copy(values, dst)


# Aligned variants for the warp-per-row kernel.  ``autovec_copy`` picks the
# vector width from the pointer alignment; a plain ``ptr + offset`` with a
# runtime offset only proves 2-byte alignment (the CTA kernel above therefore
# copies with 16-bit global loads and stores).  Every offset here is a multiple
# of 8 elements on 16-byte-aligned storage, so asserting it yields single 128-bit
# accesses.
@cute.jit
def _load_bf16x8_aligned(ptr, offset):
    values = cute.make_rmem_tensor((_BF16_VEC_SIZE,), cutlass.BFloat16)
    src = cute.make_tensor(
        ptr + cute.assume(offset, _BF16_VEC_SIZE),
        cute.make_layout((_BF16_VEC_SIZE,)),
    )
    cute.autovec_copy(src, values)
    return values


@cute.jit
def _store_bf16x8_aligned(ptr, offset, values) -> None:
    dst = cute.make_tensor(
        ptr + cute.assume(offset, _BF16_VEC_SIZE),
        cute.make_layout((_BF16_VEC_SIZE,)),
    )
    cute.autovec_copy(values, dst)


@cute.jit
def _rmsnorm_row(
    src_ptr,
    weight_ptr,
    dst_ptr,
    row: cutlass.Int32,
    src_row_stride: cutlass.Int32,
    dst_row_stride: cutlass.Int32,
    warp_sums: cute.Tensor,
    tidx: cutlass.Int32,
    eps: cutlass.Constexpr[cutlass.Float32],
    use_pdl: cutlass.Constexpr[bool],
    strided_dst: cutlass.Constexpr[bool],
    WIDTH: cutlass.Constexpr[int],
) -> None:
    """Normalize one contiguous logical row with a CTA-wide FP32 reduction."""

    lane_id = tidx % _WARP_SIZE
    warp_id = tidx // _WARP_SIZE
    col = tidx * _BF16_VEC_SIZE
    src_offset = row * src_row_stride + col
    if cutlass.const_expr(strided_dst):
        # The destination is a strided view (in-place KV into the
        # projection row), so the row pitch is a runtime argument.
        dst_offset = row * dst_row_stride + col
    else:
        dst_offset = row * cutlass.Int32(WIDTH) + col

    values = cute.make_rmem_tensor((_BF16_VEC_SIZE,), cutlass.BFloat16)
    weights = cute.make_rmem_tensor((_BF16_VEC_SIZE,), cutlass.BFloat16)
    values.fill(0.0)
    weights.fill(0.0)

    local_sum_sq = cutlass.Float32(0.0)
    if col + _BF16_VEC_SIZE <= WIDTH:
        loaded_weights = _load_bf16x8(weight_ptr, col)
        for vec_lane in cutlass.range_constexpr(_BF16_VEC_SIZE):
            weights[vec_lane] = loaded_weights[vec_lane]

    # The norm weights are immutable and can be fetched while the projection
    # GEMM before this kernel finishes.
    # The source row is predecessor-owned, so no source load may cross this
    # wait when this grid was programmatically early-launched.
    if cutlass.const_expr(use_pdl):
        cute.arch.griddepcontrol_wait()

    if col + _BF16_VEC_SIZE <= WIDTH:
        loaded_values = _load_bf16x8(src_ptr, src_offset)
        for vec_lane in cutlass.range_constexpr(_BF16_VEC_SIZE):
            values[vec_lane] = loaded_values[vec_lane]
            value_f32 = loaded_values[vec_lane].to(cutlass.Float32)
            local_sum_sq = local_sum_sq + value_f32 * value_f32

    warp_sum = _warp_sum_f32(local_sum_sq)
    if lane_id == 0:
        warp_sums[warp_id] = warp_sum

    cute.arch.barrier(barrier_id=0, number_of_threads=_THREADS_PER_BLOCK)

    if tidx == 0:
        total_sum = cutlass.Float32(0.0)
        for warp_idx in cutlass.range_constexpr(_WARPS_PER_BLOCK):
            total_sum = total_sum + warp_sums[warp_idx]
        warp_sums[0] = cute.math.rsqrt(
            total_sum * cutlass.Float32(1.0 / WIDTH) + eps,
            fastmath=True,
        )

    cute.arch.barrier(barrier_id=0, number_of_threads=_THREADS_PER_BLOCK)
    inv_rms = warp_sums[0]

    if col + _BF16_VEC_SIZE <= WIDTH:
        normalized = cute.make_rmem_tensor((_BF16_VEC_SIZE,), cutlass.BFloat16)
        for vec_lane in cutlass.range_constexpr(_BF16_VEC_SIZE):
            norm_f32 = values[vec_lane].to(cutlass.Float32) * inv_rms
            norm_f32 = norm_f32 * weights[vec_lane].to(cutlass.Float32)
            normalized[vec_lane] = norm_f32.to(cutlass.BFloat16)
        _store_bf16x8(dst_ptr, dst_offset, normalized)


@cute.kernel
def _qkv_a_rmsnorm_kernel(
    q_input: cute.Tensor,
    q_weight: cute.Tensor,
    kv_input: cute.Tensor,
    kv_weight: cute.Tensor,
    q_output: cute.Tensor,
    kv_output: cute.Tensor,
    q_row_stride: cutlass.Int32,
    kv_row_stride: cutlass.Int32,
    q_dst_row_stride: cutlass.Int32,
    kv_dst_row_stride: cutlass.Int32,
    eps: cutlass.Constexpr[cutlass.Float32],
    use_pdl: cutlass.Constexpr[bool],
    strided_dst: cutlass.Constexpr[bool],
) -> None:
    row, segment, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()

    smem = SmemAllocator()
    warp_sums = smem.allocate_tensor(
        cutlass.Float32,
        cute.make_layout((_WARPS_PER_BLOCK,)),
        byte_alignment=_ALIGNMENT_BYTES,
    )

    if segment == 0:
        _rmsnorm_row(
            q_input.iterator,
            q_weight.iterator,
            q_output.iterator,
            row,
            q_row_stride,
            q_dst_row_stride,
            warp_sums,
            tidx,
            eps,
            use_pdl,
            strided_dst,
            _Q_DIM,
        )
    else:
        _rmsnorm_row(
            kv_input.iterator,
            kv_weight.iterator,
            kv_output.iterator,
            row,
            kv_row_stride,
            kv_dst_row_stride,
            warp_sums,
            tidx,
            eps,
            use_pdl,
            strided_dst,
            _KV_DIM,
        )

    cute.arch.barrier(barrier_id=0, number_of_threads=_THREADS_PER_BLOCK)
    if cutlass.const_expr(use_pdl):
        if tidx == 0:
            cute.arch.griddepcontrol_launch_dependents()


@cute.jit
def _launch_qkv_a_rmsnorm(
    q_input: cute.Tensor,
    q_weight: cute.Tensor,
    kv_input: cute.Tensor,
    kv_weight: cute.Tensor,
    q_output: cute.Tensor,
    kv_output: cute.Tensor,
    q_row_stride: cutlass.Int32,
    kv_row_stride: cutlass.Int32,
    q_dst_row_stride: cutlass.Int32,
    kv_dst_row_stride: cutlass.Int32,
    stream: cuda.CUstream,
    eps: cutlass.Constexpr[cutlass.Float32],
    use_pdl: cutlass.Constexpr[bool],
    strided_dst: cutlass.Constexpr[bool],
) -> None:
    rows = cute.size(q_input, mode=[0])
    _qkv_a_rmsnorm_kernel(
        q_input,
        q_weight,
        kv_input,
        kv_weight,
        q_output,
        kv_output,
        q_row_stride,
        kv_row_stride,
        q_dst_row_stride,
        kv_dst_row_stride,
        eps,
        use_pdl,
        strided_dst,
    ).launch(
        grid=(rows, 2, 1),
        block=(_THREADS_PER_BLOCK, 1, 1),
        stream=stream,
        use_pdl=use_pdl,
    )


@cute.jit
def _rmsnorm_row_q8(
    src_ptr,
    weight_ptr,
    dst_ptr,
    row: cutlass.Int32,
    src_row_stride: cutlass.Int32,
    dst_row_stride: cutlass.Int32,
    warp_sums: cute.Tensor,
    tidx: cutlass.Int32,
    eps: cutlass.Constexpr[cutlass.Float32],
    use_pdl: cutlass.Constexpr[bool],
    strided_dst: cutlass.Constexpr[bool],
    q8_base,
    sf_base,
    q8_rows: cutlass.Int32,
) -> None:
    """``_rmsnorm_row`` for the q segment (``_Q_DIM`` = 256 threads x 8 columns, so every
    thread owns a chunk) that also writes the row as MXF8: e4m3 bytes at ``q8_base``
    (compact ``[q8_rows, _Q_DIM]``) and the blocked e8m0 scales at ``sf_base``, from the
    BF16 values just stored -- the bytes ``self_attn_prefill_act_quant_mxf8`` writes.
    Thread ``t`` holds columns ``8t .. 8t + 7``: a 32-column scale block is lanes
    ``4j .. 4j + 3`` of one warp, all of which reach the block-max shuffles.  The BF16
    arithmetic is ``_rmsnorm_row``'s instruction for instruction."""
    WIDTH = _Q_DIM
    lane_id = tidx % _WARP_SIZE
    warp_id = tidx // _WARP_SIZE
    col = tidx * _BF16_VEC_SIZE
    src_offset = row * src_row_stride + col
    if cutlass.const_expr(strided_dst):
        dst_offset = row * dst_row_stride + col
    else:
        dst_offset = row * cutlass.Int32(WIDTH) + col

    values = cute.make_rmem_tensor((_BF16_VEC_SIZE,), cutlass.BFloat16)
    weights = cute.make_rmem_tensor((_BF16_VEC_SIZE,), cutlass.BFloat16)
    loaded_weights = _load_bf16x8(weight_ptr, col)
    for vec_lane in cutlass.range_constexpr(_BF16_VEC_SIZE):
        weights[vec_lane] = loaded_weights[vec_lane]

    if cutlass.const_expr(use_pdl):
        cute.arch.griddepcontrol_wait()

    local_sum_sq = cutlass.Float32(0.0)
    loaded_values = _load_bf16x8(src_ptr, src_offset)
    for vec_lane in cutlass.range_constexpr(_BF16_VEC_SIZE):
        values[vec_lane] = loaded_values[vec_lane]
        value_f32 = loaded_values[vec_lane].to(cutlass.Float32)
        local_sum_sq = local_sum_sq + value_f32 * value_f32

    warp_sum = _warp_sum_f32(local_sum_sq)
    if lane_id == 0:
        warp_sums[warp_id] = warp_sum

    cute.arch.barrier(barrier_id=0, number_of_threads=_THREADS_PER_BLOCK)

    if tidx == 0:
        total_sum = cutlass.Float32(0.0)
        for warp_idx in cutlass.range_constexpr(_WARPS_PER_BLOCK):
            total_sum = total_sum + warp_sums[warp_idx]
        warp_sums[0] = cute.math.rsqrt(
            total_sum * cutlass.Float32(1.0 / WIDTH) + eps,
            fastmath=True,
        )

    cute.arch.barrier(barrier_id=0, number_of_threads=_THREADS_PER_BLOCK)
    inv_rms = warp_sums[0]

    normalized = cute.make_rmem_tensor((_BF16_VEC_SIZE,), cutlass.BFloat16)
    vals = cute.make_rmem_tensor((_BF16_VEC_SIZE,), cutlass.Float32)
    for vec_lane in cutlass.range_constexpr(_BF16_VEC_SIZE):
        norm_f32 = values[vec_lane].to(cutlass.Float32) * inv_rms
        norm_f32 = norm_f32 * weights[vec_lane].to(cutlass.Float32)
        normalized[vec_lane] = norm_f32.to(cutlass.BFloat16)
        vals[vec_lane] = normalized[vec_lane].to(cutlass.Float32)
    _store_bf16x8(dst_ptr, dst_offset, normalized)
    _mxf8.mxf8_quant_store_x8(
        vals,
        q8_base,
        cutlass.Int64(row) * WIDTH + cutlass.Int64(col),
        sf_base,
        cutlass.Int64(row),
        cutlass.Int64(col),
        cutlass.Int64(WIDTH // 128),
        row < q8_rows,
    )


@cute.kernel
def _qkv_a_rmsnorm_kernel_q8(
    q_input: cute.Tensor,
    q_weight: cute.Tensor,
    kv_input: cute.Tensor,
    kv_weight: cute.Tensor,
    q_output: cute.Tensor,
    kv_output: cute.Tensor,
    q8_output: cute.Tensor,
    q8_scales: cute.Tensor,
    q_row_stride: cutlass.Int32,
    kv_row_stride: cutlass.Int32,
    q_dst_row_stride: cutlass.Int32,
    kv_dst_row_stride: cutlass.Int32,
    eps: cutlass.Constexpr[cutlass.Float32],
    use_pdl: cutlass.Constexpr[bool],
    strided_dst: cutlass.Constexpr[bool],
    early_signal: cutlass.Constexpr[bool],
) -> None:
    """``_qkv_a_rmsnorm_kernel`` (one CTA per row and segment, decode rows) that also writes
    the first ``q8_rows = q8_output.shape[0]`` normalized q rows as MXF8 (decode act-quant fusion: the
    decode q_b launcher's scratch).  The kv-segment CTAs also write code 127 into the scale
    rows ``q8_rows .. padded_rows(q8_rows) - 1`` after the PDL wait.  ``early_signal``
    releases the dependent (q_b's decode GEMM) at entry; otherwise at the end, as the CTA
    kernel does.  BF16 outputs bitwise ``_qkv_a_rmsnorm_kernel``'s."""
    if cutlass.const_expr(use_pdl and early_signal):
        cute.arch.griddepcontrol_launch_dependents()
    row, segment, _ = cute.arch.block_idx()
    grid_x, _, _ = cute.arch.grid_dim()
    tidx, _, _ = cute.arch.thread_idx()
    q8_rows = cute.size(q8_output, mode=[0])

    smem = SmemAllocator()
    warp_sums = smem.allocate_tensor(
        cutlass.Float32,
        cute.make_layout((_WARPS_PER_BLOCK,)),
        byte_alignment=_ALIGNMENT_BYTES,
    )

    if segment == 0:
        _rmsnorm_row_q8(
            q_input.iterator,
            q_weight.iterator,
            q_output.iterator,
            row,
            q_row_stride,
            q_dst_row_stride,
            warp_sums,
            tidx,
            eps,
            use_pdl,
            strided_dst,
            q8_output.iterator,
            q8_scales.iterator,
            q8_rows,
        )
    else:
        _rmsnorm_row(
            kv_input.iterator,
            kv_weight.iterator,
            kv_output.iterator,
            row,
            kv_row_stride,
            kv_dst_row_stride,
            warp_sums,
            tidx,
            eps,
            use_pdl,
            strided_dst,
            _KV_DIM,
        )
        _mxf8.mxf8_fill_pad_scales(
            q8_scales.iterator,
            q8_rows,
            cutlass.Int32(_Q_DIM // 128),
            row * _THREADS_PER_BLOCK + tidx,
            grid_x * _THREADS_PER_BLOCK,
        )

    cute.arch.barrier(barrier_id=0, number_of_threads=_THREADS_PER_BLOCK)
    if cutlass.const_expr(use_pdl and not early_signal):
        if tidx == 0:
            cute.arch.griddepcontrol_launch_dependents()


@cute.jit
def _launch_qkv_a_rmsnorm_q8(
    q_input: cute.Tensor,
    q_weight: cute.Tensor,
    kv_input: cute.Tensor,
    kv_weight: cute.Tensor,
    q_output: cute.Tensor,
    kv_output: cute.Tensor,
    q8_output: cute.Tensor,
    q8_scales: cute.Tensor,
    q_row_stride: cutlass.Int32,
    kv_row_stride: cutlass.Int32,
    q_dst_row_stride: cutlass.Int32,
    kv_dst_row_stride: cutlass.Int32,
    stream: cuda.CUstream,
    eps: cutlass.Constexpr[cutlass.Float32],
    use_pdl: cutlass.Constexpr[bool],
    strided_dst: cutlass.Constexpr[bool],
    early_signal: cutlass.Constexpr[bool],
) -> None:
    rows = cute.size(q_input, mode=[0])
    _qkv_a_rmsnorm_kernel_q8(
        q_input,
        q_weight,
        kv_input,
        kv_weight,
        q_output,
        kv_output,
        q8_output,
        q8_scales,
        q_row_stride,
        kv_row_stride,
        q_dst_row_stride,
        kv_dst_row_stride,
        eps,
        use_pdl,
        strided_dst,
        early_signal,
    ).launch(
        grid=(rows, 2, 1),
        block=(_THREADS_PER_BLOCK, 1, 1),
        stream=stream,
        use_pdl=use_pdl,
    )


@cute.jit
def _rmsnorm_row_warp(
    src_ptr,
    weight_ptr,
    dst_ptr,
    row: cutlass.Int32,
    src_row_stride: cutlass.Int32,
    dst_row_stride: cutlass.Int32,
    lane_id: cutlass.Int32,
    eps: cutlass.Constexpr[cutlass.Float32],
    use_pdl: cutlass.Constexpr[bool],
    strided_dst: cutlass.Constexpr[bool],
    WIDTH: cutlass.Constexpr[int],
    q8_base,
    sf_base,
    q8_rows: cutlass.Int32,
    emit_q8: cutlass.Constexpr[bool],
) -> None:
    """Normalize one row with a single warp; same FP32 reduction order as
    ``_rmsnorm_row`` (see the module docstring), so the bytes are identical.

    ``emit_q8``: rows ``< q8_rows`` are also written as MXF8 (e4m3 bytes at ``q8_base``,
    compact ``[q8_rows, WIDTH]``, and the blocked e8m0 scales at ``sf_base``) from the BF16
    values just stored -- the bytes ``self_attn_prefill_act_quant_mxf8`` writes for them.
    Lane ``l`` of pass ``p`` holds columns ``(p * 32 + l) * 8``, so a 32-column scale block
    is lanes ``4j .. 4j + 3`` of one pass; the whole warp shares the row, so every lane
    reaches the block-max shuffles."""

    PASSES = WIDTH // (_WARP_SIZE * _BF16_VEC_SIZE)
    src_row = row * src_row_stride
    if cutlass.const_expr(strided_dst):
        dst_row = row * dst_row_stride
    else:
        dst_row = row * cutlass.Int32(WIDTH)

    if cutlass.const_expr(use_pdl):
        cute.arch.griddepcontrol_wait()

    # 1) Issue every 16-byte load of the row before any reduction so all
    #    PASSES loads are in flight together (one 8-element register chunk per
    #    pass, like the CTA kernel's per-thread chunk).
    chunks = []
    for pass_idx in cutlass.range_constexpr(PASSES):
        col = (pass_idx * _WARP_SIZE + lane_id) * _BF16_VEC_SIZE
        chunks.append(_load_bf16x8_aligned(src_ptr, src_row + col))

    # 2) Pass p reproduces CTA-kernel warp p: per-thread 8-term sum, butterfly
    #    over 32 lanes, then the passes are accumulated in order (thread 0's loop).
    total_sum = cutlass.Float32(0.0)
    for pass_idx in cutlass.range_constexpr(PASSES):
        local_sum_sq = cutlass.Float32(0.0)
        for vec_lane in cutlass.range_constexpr(_BF16_VEC_SIZE):
            value_f32 = chunks[pass_idx][vec_lane].to(cutlass.Float32)
            local_sum_sq = local_sum_sq + value_f32 * value_f32
        total_sum = total_sum + _warp_sum_f32(local_sum_sq)

    inv_rms = cute.math.rsqrt(
        total_sum * cutlass.Float32(1.0 / WIDTH) + eps,
        fastmath=True,
    )

    # 3) Normalize; the (L1/L2-hot) weights are fetched here rather than held
    #    in registers across the reduction, keeping occupancy high.
    for pass_idx in cutlass.range_constexpr(PASSES):
        col = (pass_idx * _WARP_SIZE + lane_id) * _BF16_VEC_SIZE
        loaded_weights = _load_bf16x8_aligned(weight_ptr, col)
        normalized = cute.make_rmem_tensor((_BF16_VEC_SIZE,), cutlass.BFloat16)
        for vec_lane in cutlass.range_constexpr(_BF16_VEC_SIZE):
            norm_f32 = chunks[pass_idx][vec_lane].to(cutlass.Float32) * inv_rms
            norm_f32 = norm_f32 * loaded_weights[vec_lane].to(cutlass.Float32)
            normalized[vec_lane] = norm_f32.to(cutlass.BFloat16)
        _store_bf16x8_aligned(dst_ptr, dst_row + col, normalized)
        if cutlass.const_expr(emit_q8):
            vals = cute.make_rmem_tensor((_BF16_VEC_SIZE,), cutlass.Float32)
            for vec_lane in cutlass.range_constexpr(_BF16_VEC_SIZE):
                vals[vec_lane] = normalized[vec_lane].to(cutlass.Float32)
            _mxf8.mxf8_quant_store_x8(
                vals,
                q8_base,
                cutlass.Int64(row) * WIDTH + cutlass.Int64(col),
                sf_base,
                cutlass.Int64(row),
                cutlass.Int64(col),
                cutlass.Int64(WIDTH // 128),
                row < q8_rows,
            )


@cute.kernel
def _qkv_a_rmsnorm_kernel_warp_row(
    q_input: cute.Tensor,
    q_weight: cute.Tensor,
    kv_input: cute.Tensor,
    kv_weight: cute.Tensor,
    q_output: cute.Tensor,
    kv_output: cute.Tensor,
    q_row_stride: cutlass.Int32,
    kv_row_stride: cutlass.Int32,
    q_dst_row_stride: cutlass.Int32,
    kv_dst_row_stride: cutlass.Int32,
    eps: cutlass.Constexpr[cutlass.Float32],
    use_pdl: cutlass.Constexpr[bool],
    strided_dst: cutlass.Constexpr[bool],
) -> None:
    """Large-T kernel: grid ``(ceil(rows / 8), 2)``, one warp per row."""
    row_block, segment, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    rows = cute.size(q_input, mode=[0])
    lane_id = tidx % _WARP_SIZE
    row = row_block * _WARPS_PER_BLOCK + tidx // _WARP_SIZE

    if row < rows:
        if segment == 0:
            _rmsnorm_row_warp(
                q_input.iterator,
                q_weight.iterator,
                q_output.iterator,
                row,
                q_row_stride,
                q_dst_row_stride,
                lane_id,
                eps,
                use_pdl,
                strided_dst,
                _Q_DIM,
                q_output.iterator,
                q_output.iterator,
                cutlass.Int32(0),
                False,
            )
        else:
            _rmsnorm_row_warp(
                kv_input.iterator,
                kv_weight.iterator,
                kv_output.iterator,
                row,
                kv_row_stride,
                kv_dst_row_stride,
                lane_id,
                eps,
                use_pdl,
                strided_dst,
                _KV_DIM,
                kv_output.iterator,
                kv_output.iterator,
                cutlass.Int32(0),
                False,
            )

    cute.arch.barrier(barrier_id=0, number_of_threads=_THREADS_PER_BLOCK)
    if cutlass.const_expr(use_pdl):
        if tidx == 0:
            cute.arch.griddepcontrol_launch_dependents()


@cute.jit
def _launch_qkv_a_rmsnorm_warp_row(
    q_input: cute.Tensor,
    q_weight: cute.Tensor,
    kv_input: cute.Tensor,
    kv_weight: cute.Tensor,
    q_output: cute.Tensor,
    kv_output: cute.Tensor,
    q_row_stride: cutlass.Int32,
    kv_row_stride: cutlass.Int32,
    q_dst_row_stride: cutlass.Int32,
    kv_dst_row_stride: cutlass.Int32,
    stream: cuda.CUstream,
    eps: cutlass.Constexpr[cutlass.Float32],
    use_pdl: cutlass.Constexpr[bool],
    strided_dst: cutlass.Constexpr[bool],
) -> None:
    rows = cute.size(q_input, mode=[0])
    _qkv_a_rmsnorm_kernel_warp_row(
        q_input,
        q_weight,
        kv_input,
        kv_weight,
        q_output,
        kv_output,
        q_row_stride,
        kv_row_stride,
        q_dst_row_stride,
        kv_dst_row_stride,
        eps,
        use_pdl,
        strided_dst,
    ).launch(
        grid=(cute.ceil_div(rows, _WARPS_PER_BLOCK), 2, 1),
        block=(_THREADS_PER_BLOCK, 1, 1),
        stream=stream,
        use_pdl=use_pdl,
    )


@cute.kernel
def _qkv_a_rmsnorm_kernel_warp_row_q8(
    q_input: cute.Tensor,
    q_weight: cute.Tensor,
    kv_input: cute.Tensor,
    kv_weight: cute.Tensor,
    q_output: cute.Tensor,
    kv_output: cute.Tensor,
    q8_output: cute.Tensor,
    q8_scales: cute.Tensor,
    q_row_stride: cutlass.Int32,
    kv_row_stride: cutlass.Int32,
    q_dst_row_stride: cutlass.Int32,
    kv_dst_row_stride: cutlass.Int32,
    eps: cutlass.Constexpr[cutlass.Float32],
    use_pdl: cutlass.Constexpr[bool],
    strided_dst: cutlass.Constexpr[bool],
    early_signal: cutlass.Constexpr[bool],
) -> None:
    """``_qkv_a_rmsnorm_kernel_warp_row`` that also writes the first ``q8_rows =
    q8_output.shape[0]`` normalized q rows as MXF8 (the FP8 context route's q_b operand).
    The kv-segment CTAs (light: 512 columns) also write code 127 into the scale rows
    ``q8_rows .. padded_rows(q8_rows) - 1``, after the PDL wait like every other store.

    ``early_signal`` (decode act-quant fusion decode rows): release the dependent (q_b's decode GEMM, which
    streams its weights before its own wait) at entry, as the standalone PDL act quant did,
    instead of after the last store."""
    if cutlass.const_expr(use_pdl and early_signal):
        cute.arch.griddepcontrol_launch_dependents()
    row_block, segment, _ = cute.arch.block_idx()
    grid_x, _, _ = cute.arch.grid_dim()
    tidx, _, _ = cute.arch.thread_idx()
    rows = cute.size(q_input, mode=[0])
    q8_rows = cute.size(q8_output, mode=[0])
    lane_id = tidx % _WARP_SIZE
    row = row_block * _WARPS_PER_BLOCK + tidx // _WARP_SIZE

    if row < rows:
        if segment == 0:
            _rmsnorm_row_warp(
                q_input.iterator,
                q_weight.iterator,
                q_output.iterator,
                row,
                q_row_stride,
                q_dst_row_stride,
                lane_id,
                eps,
                use_pdl,
                strided_dst,
                _Q_DIM,
                q8_output.iterator,
                q8_scales.iterator,
                q8_rows,
                True,
            )
        else:
            _rmsnorm_row_warp(
                kv_input.iterator,
                kv_weight.iterator,
                kv_output.iterator,
                row,
                kv_row_stride,
                kv_dst_row_stride,
                lane_id,
                eps,
                use_pdl,
                strided_dst,
                _KV_DIM,
                kv_output.iterator,
                kv_output.iterator,
                cutlass.Int32(0),
                False,
            )
    if segment == 1:
        if cutlass.const_expr(use_pdl):
            cute.arch.griddepcontrol_wait()
        _mxf8.mxf8_fill_pad_scales(
            q8_scales.iterator,
            q8_rows,
            cutlass.Int32(_Q_DIM // 128),
            row_block * _THREADS_PER_BLOCK + tidx,
            grid_x * _THREADS_PER_BLOCK,
        )

    cute.arch.barrier(barrier_id=0, number_of_threads=_THREADS_PER_BLOCK)
    if cutlass.const_expr(use_pdl):
        if tidx == 0:
            cute.arch.griddepcontrol_launch_dependents()


@cute.jit
def _launch_qkv_a_rmsnorm_warp_row_q8(
    q_input: cute.Tensor,
    q_weight: cute.Tensor,
    kv_input: cute.Tensor,
    kv_weight: cute.Tensor,
    q_output: cute.Tensor,
    kv_output: cute.Tensor,
    q8_output: cute.Tensor,
    q8_scales: cute.Tensor,
    q_row_stride: cutlass.Int32,
    kv_row_stride: cutlass.Int32,
    q_dst_row_stride: cutlass.Int32,
    kv_dst_row_stride: cutlass.Int32,
    stream: cuda.CUstream,
    eps: cutlass.Constexpr[cutlass.Float32],
    use_pdl: cutlass.Constexpr[bool],
    strided_dst: cutlass.Constexpr[bool],
    early_signal: cutlass.Constexpr[bool] = False,
) -> None:
    rows = cute.size(q_input, mode=[0])
    _qkv_a_rmsnorm_kernel_warp_row_q8(
        q_input,
        q_weight,
        kv_input,
        kv_weight,
        q_output,
        kv_output,
        q8_output,
        q8_scales,
        q_row_stride,
        kv_row_stride,
        q_dst_row_stride,
        kv_dst_row_stride,
        eps,
        use_pdl,
        strided_dst,
        early_signal,
    ).launch(
        grid=(cute.ceil_div(rows, _WARPS_PER_BLOCK), 2, 1),
        block=(_THREADS_PER_BLOCK, 1, 1),
        stream=stream,
        use_pdl=use_pdl,
    )


def _make_fake_rmsnorm_arguments(inplace_kv_view: bool, assumed_align: int):
    """Fake tensors of the serving ABI (dynamic rows and row strides)."""
    sym_rows = cute.sym_int32()
    sym_q_stride = cute.sym_int32(divisibility=_BF16_VEC_SIZE)
    sym_kv_stride = cute.sym_int32(divisibility=_BF16_VEC_SIZE)
    fake_q = make_fake_tensor(
        cutlass.BFloat16,
        (sym_rows, _Q_DIM),
        (sym_q_stride, 1),
        assumed_align=assumed_align,
    )
    fake_kv = make_fake_tensor(
        cutlass.BFloat16,
        (sym_rows, _KV_DIM),
        (sym_kv_stride, 1),
        assumed_align=assumed_align,
    )
    fake_q_weight = make_fake_compact_tensor(
        cutlass.BFloat16, (_Q_DIM,), assumed_align=assumed_align
    )
    fake_kv_weight = make_fake_compact_tensor(
        cutlass.BFloat16, (_KV_DIM,), assumed_align=assumed_align
    )
    if inplace_kv_view:
        sym_q_dst_stride = cute.sym_int32(divisibility=_BF16_VEC_SIZE)
        sym_kv_dst_stride = cute.sym_int32(divisibility=_BF16_VEC_SIZE)
        fake_q_output = make_fake_tensor(
            cutlass.BFloat16,
            (sym_rows, _Q_DIM),
            (sym_q_dst_stride, 1),
            assumed_align=assumed_align,
        )
        fake_kv_output = make_fake_tensor(
            cutlass.BFloat16,
            (sym_rows, _KV_DIM),
            (sym_kv_dst_stride, 1),
            assumed_align=assumed_align,
        )
    else:
        fake_q_output = make_fake_compact_tensor(
            cutlass.BFloat16,
            (sym_rows, _Q_DIM),
            stride_order=(1, 0),
            assumed_align=assumed_align,
        )
        fake_kv_output = make_fake_compact_tensor(
            cutlass.BFloat16,
            (sym_rows, _KV_DIM),
            stride_order=(1, 0),
            assumed_align=assumed_align,
        )
    return fake_q, fake_q_weight, fake_kv, fake_kv_weight, fake_q_output, fake_kv_output


def compile_qkv_a_rmsnorm_warp_row(
    device_id: int = 0,
    eps: float = _DEFAULT_EPS,
    *,
    use_pdl: bool = True,
    inplace_kv_view: bool = False,
) -> Callable:
    """Compile the large-T warp-per-row kernel (same ABI as ``compile_qkv_a_rmsnorm``).

    Unlike the CTA kernel it is compiled with 16-byte ``assumed_align`` on every
    tensor so the row chunks are single 128-bit loads/stores; the host only
    selects it when all data pointers are 16-byte aligned.
    """

    key = (device_id, float(eps), bool(use_pdl), bool(inplace_kv_view))
    if key in _WARP_ROW_COMPILE_CACHE:
        return _WARP_ROW_COMPILE_CACHE[key]

    cutlass.cuda.initialize_cuda_context(device_id)
    fakes = _make_fake_rmsnorm_arguments(bool(inplace_kv_view), _ALIGNMENT_BYTES)
    compiled = cute.compile(
        _launch_qkv_a_rmsnorm_warp_row,
        *fakes,
        cutlass.Int32(0),
        cutlass.Int32(0),
        cutlass.Int32(0),
        cutlass.Int32(0),
        make_fake_stream(),
        cutlass.Float32(eps),
        bool(use_pdl),
        bool(inplace_kv_view),
        options="--enable-tvm-ffi",
    )
    _WARP_ROW_COMPILE_CACHE[key] = compiled
    return compiled


_WARP_ROW_Q8_COMPILE_CACHE: dict[tuple[int, float, bool, bool, bool], Callable] = {}


def compile_qkv_a_rmsnorm_warp_row_q8(
    device_id: int = 0,
    eps: float = _DEFAULT_EPS,
    *,
    use_pdl: bool = True,
    inplace_kv_view: bool = False,
    early_signal: bool = False,
) -> Callable:
    """The warp-row kernel with the MXF8 q copy (``_qkv_a_rmsnorm_kernel_warp_row_q8``).

    ``fn(q_in, q_w, kv_in, kv_w, q_out, kv_out, q8 [R8, 2048] uint8 (compact, R8 <= rows),
    sf uint8 (>= act_quant sf_bytes(R8, 2048), 32-byte aligned), q_pitch, kv_pitch,
    q_dst_pitch, kv_dst_pitch, stream)``; ``R8 >= 1``.  The host selects it only where it
    would select the warp-row kernel (``_use_warp_row_path``) -- and (decode act-quant fusion) on decode
    rows with ``q8`` the decode q_b launcher's 64-row scratch and ``early_signal=True``;
    its BF16 outputs are bitwise the CTA kernel's (module docstring)."""
    key = (
        device_id,
        float(eps),
        bool(use_pdl),
        bool(inplace_kv_view),
        bool(early_signal),
    )
    if key in _WARP_ROW_Q8_COMPILE_CACHE:
        return _WARP_ROW_Q8_COMPILE_CACHE[key]
    cutlass.cuda.initialize_cuda_context(device_id)
    fakes = _make_fake_rmsnorm_arguments(bool(inplace_kv_view), _ALIGNMENT_BYTES)
    fake_q8 = make_fake_compact_tensor(
        cutlass.Uint8,
        (cute.sym_int32(), _Q_DIM),
        stride_order=(1, 0),
        assumed_align=_ALIGNMENT_BYTES,
    )
    fake_sf = make_fake_compact_tensor(cutlass.Uint8, (cute.sym_int32(),), assumed_align=32)
    compiled = cute.compile(
        _launch_qkv_a_rmsnorm_warp_row_q8,
        *fakes,
        fake_q8,
        fake_sf,
        cutlass.Int32(0),
        cutlass.Int32(0),
        cutlass.Int32(0),
        cutlass.Int32(0),
        make_fake_stream(),
        cutlass.Float32(eps),
        bool(use_pdl),
        bool(inplace_kv_view),
        bool(early_signal),
        options="--enable-tvm-ffi",
    )
    _WARP_ROW_Q8_COMPILE_CACHE[key] = compiled
    return compiled


# --------------------------------------------------------------------------- context-step stitch
# Context-step variant of ``_qkv_a_rmsnorm_kernel_warp_row_q8`` (the served kernels above are
# untouched; this is a separate symbol, ``..._warp_row_q8_ctx``). Same row work, same FP32
# reduction order, same BF16 / MXF8 bytes, plus two context-only options:
#
# * ``q_store_from`` (runtime): the BF16 q row is stored only for rows >= ``q_store_from``.
#   On a context step whose indexer takes the engine's substituted selection (``need_q``
#   False) the MXF8 copy is q_b's whole operand and the only reader of the BF16 q_a is the
#   generation rows' q_b cell, which reads at most the last 64 rows (``_Projection.tail``);
#   the caller passes ``max(0, ctx_rows - 64)`` there and 0 everywhere else. At T ~ 7.3k that
#   is 30 MB of the kernel's ~90 MB of DRAM traffic not written.
# * ``kv_append`` (constexpr): the kv segment also writes each context row's normalized
#   c_kv (rows < ``q8_rows`` = the step's context rows) as E4M3 into the layer's paged
#   latent pool -- the bytes and the KVBlockArray address the prefill rope/append kernel's
#   c_kv copy groups write (``cvt.rn.satfinite`` of the stored BF16 values, row ``pos`` of
#   request ``b`` at ``block_offsets[b, 0, pos // tpb] * tpb * 576 + (pos % tpb) * 576``,
#   ``pos = row - cu_seqlens[b] + cached_lens[b]``) -- so that kernel runs with
#   ``skip_kv_copy`` and never re-reads the 7.5 MB of latent rows to copy them.
_LATENT_ROW_BYTES = _KV_DIM + 64  # one E4M3 pool row: [c_kv 512 | k_pe 64]
_CTX_MAX_REQUESTS = 64  # binary-search depth 6, as the rope/append kernel


@cute.jit
def _ctx_locate_request(cu_seqlens: cute.Tensor, num_requests, token) -> cutlass.Int32:
    """The largest b < num_requests with cu_seqlens[b] <= token (the rope/append kernel's
    ``_locate_request``: a fixed six-probe binary search over non-decreasing prefix sums)."""
    req = cutlass.Int32(0)
    for i in cutlass.range_constexpr(6):
        cand = req + cutlass.Int32(_CTX_MAX_REQUESTS // 2 >> i)
        if cand < num_requests:
            if token >= cu_seqlens[cand]:
                req = cand
    return req


@cute.jit
def _rmsnorm_row_warp_ctx(
    src_ptr,
    weight_ptr,
    dst_ptr,
    row: cutlass.Int32,
    src_row_stride: cutlass.Int32,
    dst_row_stride: cutlass.Int32,
    lane_id: cutlass.Int32,
    eps: cutlass.Constexpr[cutlass.Float32],
    use_pdl: cutlass.Constexpr[bool],
    WIDTH: cutlass.Constexpr[int],
    q8_base,
    sf_base,
    q8_rows: cutlass.Int32,
    emit_q8: cutlass.Constexpr[bool],
    store_row: cutlass.Boolean,
    pool_base,
    cache_base: cutlass.Int64,
    append: cutlass.Boolean,
    kv_append: cutlass.Constexpr[bool],
) -> None:
    """``_rmsnorm_row_warp`` (strided destination) with the two context-step stitch options: the BF16
    store only when ``store_row``, and, under ``kv_append``, the E4M3 copy of the stored row
    at ``pool_base + cache_base`` when ``append``. The loads, the reduction, the normalize
    and the MXF8 copy are the same instructions in the same order."""

    PASSES = WIDTH // (_WARP_SIZE * _BF16_VEC_SIZE)
    src_row = row * src_row_stride
    dst_row = row * dst_row_stride

    if cutlass.const_expr(use_pdl):
        cute.arch.griddepcontrol_wait()

    chunks = []
    for pass_idx in cutlass.range_constexpr(PASSES):
        col = (pass_idx * _WARP_SIZE + lane_id) * _BF16_VEC_SIZE
        chunks.append(_load_bf16x8_aligned(src_ptr, src_row + col))

    total_sum = cutlass.Float32(0.0)
    for pass_idx in cutlass.range_constexpr(PASSES):
        local_sum_sq = cutlass.Float32(0.0)
        for vec_lane in cutlass.range_constexpr(_BF16_VEC_SIZE):
            value_f32 = chunks[pass_idx][vec_lane].to(cutlass.Float32)
            local_sum_sq = local_sum_sq + value_f32 * value_f32
        total_sum = total_sum + _warp_sum_f32(local_sum_sq)

    inv_rms = cute.math.rsqrt(
        total_sum * cutlass.Float32(1.0 / WIDTH) + eps,
        fastmath=True,
    )

    for pass_idx in cutlass.range_constexpr(PASSES):
        col = (pass_idx * _WARP_SIZE + lane_id) * _BF16_VEC_SIZE
        loaded_weights = _load_bf16x8_aligned(weight_ptr, col)
        normalized = cute.make_rmem_tensor((_BF16_VEC_SIZE,), cutlass.BFloat16)
        for vec_lane in cutlass.range_constexpr(_BF16_VEC_SIZE):
            norm_f32 = chunks[pass_idx][vec_lane].to(cutlass.Float32) * inv_rms
            norm_f32 = norm_f32 * loaded_weights[vec_lane].to(cutlass.Float32)
            normalized[vec_lane] = norm_f32.to(cutlass.BFloat16)
        if store_row:
            _store_bf16x8_aligned(dst_ptr, dst_row + col, normalized)
        if cutlass.const_expr(emit_q8):
            vals = cute.make_rmem_tensor((_BF16_VEC_SIZE,), cutlass.Float32)
            for vec_lane in cutlass.range_constexpr(_BF16_VEC_SIZE):
                vals[vec_lane] = normalized[vec_lane].to(cutlass.Float32)
            _mxf8.mxf8_quant_store_x8(
                vals,
                q8_base,
                cutlass.Int64(row) * WIDTH + cutlass.Int64(col),
                sf_base,
                cutlass.Int64(row),
                cutlass.Int64(col),
                cutlass.Int64(WIDTH // 128),
                row < q8_rows,
            )
        if cutlass.const_expr(kv_append):
            if append:
                kv_vals = cute.make_rmem_tensor((_BF16_VEC_SIZE,), cutlass.Float32)
                for vec_lane in cutlass.range_constexpr(_BF16_VEC_SIZE):
                    kv_vals[vec_lane] = normalized[vec_lane].to(cutlass.Float32)
                kv8 = cute.make_rmem_tensor((_BF16_VEC_SIZE,), cutlass.Float8E4M3FN)
                kv8.store(kv_vals.load().to(cutlass.Float8E4M3FN))
                pool_dst = cute.make_tensor(
                    cute.recast_ptr(
                        pool_base + cute.assume(cache_base + cutlass.Int64(col), _BF16_VEC_SIZE),
                        dtype=cutlass.Float8E4M3FN,
                    ),
                    cute.make_layout((_BF16_VEC_SIZE,)),
                )
                cute.autovec_copy(kv8, pool_dst)


@cute.kernel
def _qkv_a_rmsnorm_kernel_warp_row_q8_ctx(
    q_input: cute.Tensor,
    q_weight: cute.Tensor,
    kv_input: cute.Tensor,
    kv_weight: cute.Tensor,
    q_output: cute.Tensor,
    kv_output: cute.Tensor,
    q8_output: cute.Tensor,
    q8_scales: cute.Tensor,
    pool: cute.Tensor,
    block_offsets: cute.Tensor,
    cu_seqlens: cute.Tensor,
    cached_lens: cute.Tensor,
    q_row_stride: cutlass.Int32,
    kv_row_stride: cutlass.Int32,
    q_dst_row_stride: cutlass.Int32,
    kv_dst_row_stride: cutlass.Int32,
    q_store_from: cutlass.Int32,
    eps: cutlass.Constexpr[cutlass.Float32],
    use_pdl: cutlass.Constexpr[bool],
    kv_append: cutlass.Constexpr[bool],
    tokens_per_block: cutlass.Constexpr[int],
) -> None:
    """``_qkv_a_rmsnorm_kernel_warp_row_q8`` (strided destinations, no early signal) with the
    context-step stitch options (block comment above). Grid ``(ceil(rows / 8), 2)``."""
    row_block, segment, _ = cute.arch.block_idx()
    grid_x, _, _ = cute.arch.grid_dim()
    tidx, _, _ = cute.arch.thread_idx()
    rows = cute.size(q_input, mode=[0])
    q8_rows = cute.size(q8_output, mode=[0])
    lane_id = tidx % _WARP_SIZE
    row = row_block * _WARPS_PER_BLOCK + tidx // _WARP_SIZE

    if row < rows:
        if segment == 0:
            _rmsnorm_row_warp_ctx(
                q_input.iterator,
                q_weight.iterator,
                q_output.iterator,
                row,
                q_row_stride,
                q_dst_row_stride,
                lane_id,
                eps,
                use_pdl,
                _Q_DIM,
                q8_output.iterator,
                q8_scales.iterator,
                q8_rows,
                True,
                row >= q_store_from,
                pool.iterator,
                cutlass.Int64(0),
                row < 0,
                False,
            )
        else:
            cache_base = cutlass.Int64(0)
            append = cutlass.Int32(0)
            if cutlass.const_expr(kv_append):
                # the request tables are host uploads ahead of this step; read after the
                # wait like every other operand
                if cutlass.const_expr(use_pdl):
                    cute.arch.griddepcontrol_wait()
                if row < q8_rows:
                    num_requests = cute.size(cached_lens, mode=[0])
                    max_blocks = cute.size(block_offsets, mode=[2])
                    req = _ctx_locate_request(cu_seqlens, num_requests, row)
                    position = row - cu_seqlens[req] + cached_lens[req]
                    block_in_seq = position // cutlass.Int32(tokens_per_block)
                    local_token = position & cutlass.Int32(tokens_per_block - 1)
                    if block_in_seq < max_blocks:
                        block = block_offsets[req, 0, block_in_seq]
                        cache_base = cutlass.Int64(block) * cutlass.Int64(
                            tokens_per_block * _LATENT_ROW_BYTES
                        ) + cutlass.Int64(local_token * _LATENT_ROW_BYTES)
                        append = cutlass.Int32(1)
            _rmsnorm_row_warp_ctx(
                kv_input.iterator,
                kv_weight.iterator,
                kv_output.iterator,
                row,
                kv_row_stride,
                kv_dst_row_stride,
                lane_id,
                eps,
                use_pdl,
                _KV_DIM,
                kv_output.iterator,
                kv_output.iterator,
                cutlass.Int32(0),
                False,
                row >= 0,
                pool.iterator,
                cache_base,
                append != 0,
                kv_append,
            )
    if segment == 1:
        if cutlass.const_expr(use_pdl):
            cute.arch.griddepcontrol_wait()
        _mxf8.mxf8_fill_pad_scales(
            q8_scales.iterator,
            q8_rows,
            cutlass.Int32(_Q_DIM // 128),
            row_block * _THREADS_PER_BLOCK + tidx,
            grid_x * _THREADS_PER_BLOCK,
        )

    cute.arch.barrier(barrier_id=0, number_of_threads=_THREADS_PER_BLOCK)
    if cutlass.const_expr(use_pdl):
        if tidx == 0:
            cute.arch.griddepcontrol_launch_dependents()


@cute.jit
def _launch_qkv_a_rmsnorm_warp_row_q8_ctx(
    q_input: cute.Tensor,
    q_weight: cute.Tensor,
    kv_input: cute.Tensor,
    kv_weight: cute.Tensor,
    q_output: cute.Tensor,
    kv_output: cute.Tensor,
    q8_output: cute.Tensor,
    q8_scales: cute.Tensor,
    pool: cute.Pointer,
    block_offsets: cute.Tensor,
    cu_seqlens: cute.Tensor,
    cached_lens: cute.Tensor,
    q_row_stride: cutlass.Int32,
    kv_row_stride: cutlass.Int32,
    q_dst_row_stride: cutlass.Int32,
    kv_dst_row_stride: cutlass.Int32,
    q_store_from: cutlass.Int32,
    stream: cuda.CUstream,
    eps: cutlass.Constexpr[cutlass.Float32],
    use_pdl: cutlass.Constexpr[bool],
    kv_append: cutlass.Constexpr[bool],
    tokens_per_block: cutlass.Constexpr[int],
) -> None:
    rows = cute.size(q_input, mode=[0])
    storage = cute.make_tensor(pool, cute.make_layout((1,), stride=(1,)))
    _qkv_a_rmsnorm_kernel_warp_row_q8_ctx(
        q_input,
        q_weight,
        kv_input,
        kv_weight,
        q_output,
        kv_output,
        q8_output,
        q8_scales,
        storage,
        block_offsets,
        cu_seqlens,
        cached_lens,
        q_row_stride,
        kv_row_stride,
        q_dst_row_stride,
        kv_dst_row_stride,
        q_store_from,
        eps,
        use_pdl,
        kv_append,
        tokens_per_block,
    ).launch(
        grid=(cute.ceil_div(rows, _WARPS_PER_BLOCK), 2, 1),
        block=(_THREADS_PER_BLOCK, 1, 1),
        stream=stream,
        use_pdl=use_pdl,
    )


_WARP_ROW_Q8_CTX_COMPILE_CACHE: dict[tuple[int, float, bool, bool, int, bool], Callable] = {}


def compile_qkv_a_rmsnorm_warp_row_q8_ctx(
    device_id: int = 0,
    eps: float = _DEFAULT_EPS,
    *,
    use_pdl: bool = True,
    kv_append: bool = False,
    tokens_per_block: int = 32,
    ilp: bool = False,
) -> Callable:
    """The context-step warp-row q8 kernel with the context-step stitch options (in-place kv view).

    ``fn(q_in, q_w, kv_in, kv_w, q_out, kv_out, q8 [R8, 2048] uint8, sf uint8, pool (int
    layer base pointer, 16-byte aligned), block_offsets int32 [B, 2, max_blocks] contiguous,
    cu_seqlens int32 [>= B + 1], cached_lens int32 [B], q_pitch, kv_pitch, q_dst_pitch,
    kv_dst_pitch, q_store_from, stream)``. ``R8`` = the step's context rows (rows below it
    get the MXF8 copy and, under ``kv_append``, the pool append); ``q_store_from = 0``
    reproduces ``compile_qkv_a_rmsnorm_warp_row_q8``'s bytes exactly. Without ``kv_append``
    the pool / table operands are not read (any valid tensors of the right dtypes).
    ``ilp=True``: the ILP build (kernel ``..._warp_row_q8_ctx_ilp``; same ABI and output
    bytes, fewer instructions per element -- block comment above ``_cvt_bf16x2_rn``)."""
    if tokens_per_block <= 0 or tokens_per_block & (tokens_per_block - 1):
        raise ValueError(f"tokens_per_block must be a power of two, got {tokens_per_block}")
    if ilp and kv_append:
        raise ValueError("ilp=True has no kv_append build")
    key = (
        device_id,
        float(eps),
        bool(use_pdl),
        bool(kv_append),
        int(tokens_per_block),
        bool(ilp),
    )
    if key in _WARP_ROW_Q8_CTX_COMPILE_CACHE:
        return _WARP_ROW_Q8_CTX_COMPILE_CACHE[key]
    cutlass.cuda.initialize_cuda_context(device_id)
    fakes = _make_fake_rmsnorm_arguments(True, _ALIGNMENT_BYTES)
    fake_q8 = make_fake_compact_tensor(
        cutlass.Uint8,
        (cute.sym_int32(), _Q_DIM),
        stride_order=(1, 0),
        assumed_align=_ALIGNMENT_BYTES,
    )
    fake_sf = make_fake_compact_tensor(cutlass.Uint8, (cute.sym_int32(),), assumed_align=32)
    fake_pool = cute.runtime.make_ptr(
        cutlass.Uint8, 0, cutlass.AddressSpace.gmem, assumed_align=_ALIGNMENT_BYTES
    )
    sym_requests = cute.sym_int32()
    fake_offsets = make_fake_compact_tensor(
        cutlass.Int32,
        (sym_requests, 2, cute.sym_int32()),
        stride_order=(2, 1, 0),
        assumed_align=4,
    )
    fake_cu = make_fake_compact_tensor(cutlass.Int32, (cute.sym_int32(),), assumed_align=4)
    fake_cached = make_fake_compact_tensor(cutlass.Int32, (sym_requests,), assumed_align=4)
    compiled = cute.compile(
        _launch_qkv_a_rmsnorm_warp_row_q8_ctx_ilp if ilp else _launch_qkv_a_rmsnorm_warp_row_q8_ctx,
        *fakes,
        fake_q8,
        fake_sf,
        fake_pool,
        fake_offsets,
        fake_cu,
        fake_cached,
        cutlass.Int32(0),
        cutlass.Int32(0),
        cutlass.Int32(0),
        cutlass.Int32(0),
        cutlass.Int32(0),
        make_fake_stream(),
        cutlass.Float32(eps),
        bool(use_pdl),
        bool(kv_append),
        int(tokens_per_block),
        options="--enable-tvm-ffi",
    )
    _WARP_ROW_Q8_CTX_COMPILE_CACHE[key] = compiled
    return compiled


# ---------------------------------------------------------------------------------------------
# The ILP build (option ``ilp=True`` of ``compile_qkv_a_rmsnorm_warp_row_q8_ctx``; separate
# kernel symbol ``..._warp_row_q8_ctx_ilp``, the served build is untouched). The context norm is
# issue-bound (issue slots 53 %, DRAM 27 %, ~21 SASS instructions per element per lane,
# 12.1M warp instructions at T 7310). Same loads, reduction order, rsqrt and per-element
# IEEE operations as ``_rmsnorm_row_warp_ctx``; fewer instructions for them:
#   * normalize: ``x * inv_rms`` and ``* w`` as packed FMUL2 pairs (two FMUL each before);
#   * BF16 rounding: one packed ``cvt.rn.bf16x2.f32`` (F2FP, ALU pipe) per pair instead of a scalar
#     ``F2F.BF16.F32`` per value on the XU pipe; the quantizer's FP32 inputs are the two halves of
#     the packed word (shift / mask), exactly the BF16 values the served path widens;
#   * MXF8 block max: an FMNMX tree over |v| and FMNMX after each of the two lane shuffles (a
#     compare + select per value before; the same value: |v| >= 0, maxNumber ignores NaN as
#     ``a > m`` does);
#   * the block scale's reciprocal from its exponent: the scale is a power of two 2^(c - 127) with
#     c = code in [79, 247] (amax >= 1e-12, BF16 finite), so 1 / scale == bits (254 - c) << 23
#     exactly (an FP32 division -- MUFU.RCP + Newton + special-case branch -- before).
# Output bytes (BF16 q / kv rows, E4M3 q, every scale byte) are identical to the served build.
# No ``kv_append``: appending the c_kv rows from this kernel measured slower.


@dsl_user_op
def _cvt_bf16x2_rn(hi, lo, *, loc=None, ip=None) -> cutlass.Int32:
    """Round two FP32 values to BF16 (RN-even) with one packed ``cvt.rn.bf16x2.f32``: ``hi`` in
    bits 31..16, ``lo`` in bits 15..0 (``F2FP.BF16.F32.PACK_AB``)."""
    v = _nvvm.convert_f32x2_to_bf16x2(
        _mlir_ir.VectorType.get([2], cutlass.BFloat16.mlir_type, loc=loc),
        cutlass.Float32(hi).ir_value(loc=loc, ip=ip),
        cutlass.Float32(lo).ir_value(loc=loc, ip=ip),
        rnd=_nvvm.FPRoundingMode.RN,
        loc=loc,
        ip=ip,
    )
    return cutlass.Int32(_llvm.bitcast(_T.i32(), v, loc=loc, ip=ip))


@cute.jit
def _mxf8_quant_store_x8_ilp(
    vals: cute.Tensor,
    local_max: cutlass.Float32,
    q_base,
    q_off: cutlass.Int64,
    sf_base,
    m: cutlass.Int64,
    col: cutlass.Int64,
    k_tiles: cutlass.Int64,
    store: cutlass.Boolean,
) -> None:
    """``_mxf8.mxf8_quant_store_x8`` with the lane's ``local_max = max |vals|`` given: the block
    max over lanes xor 1 / xor 2 by FMNMX, the scale rule of ``mxf8_code_and_inv_scale`` with the
    power-of-two reciprocal taken from the exponent (block comment above). Same bytes."""
    local_max = cute.arch.fmax(local_max, cute.arch.shuffle_sync_bfly(local_max, 1))
    local_max = cute.arch.fmax(local_max, cute.arch.shuffle_sync_bfly(local_max, 2))
    safe_amax = cute.arch.fmax(local_max, cutlass.Float32(_mxf8._MIN_AMAX))
    scale = safe_amax * cutlass.Float32(_mxf8._INV_E4M3_MAX)
    scale_bits = scale.bitcast(cutlass.Int32)
    exp_bits = scale_bits & cutlass.Int32(0x7F800000)
    if (scale_bits & cutlass.Int32(0x007FFFFF)) != cutlass.Int32(0):
        exp_bits += cutlass.Int32(0x00800000)
    code = exp_bits >> cutlass.Int32(23)
    inv_scale = ((cutlass.Int32(254) - code) << cutlass.Int32(23)).bitcast(cutlass.Float32)
    if store:
        q8 = cute.make_rmem_tensor((_mxf8._VEC,), cutlass.Float8E4M3FN)
        q8.store((vals.load() * inv_scale).to(cutlass.Float8E4M3FN))
        dst = cute.make_tensor(
            cute.recast_ptr(q_base + cute.assume(q_off, _mxf8._VEC), dtype=cutlass.Float8E4M3FN),
            cute.make_layout((_mxf8._VEC,)),
        )
        cute.autovec_copy(q8, dst)
        if (col % _mxf8._SF_VEC) == cutlass.Int64(0):
            (sf_base + _mxf8.mxf8_sf_offset(m, col // _mxf8._SF_VEC, k_tiles)).store(
                cutlass.Uint8(code)
            )


@cute.jit
def _rmsnorm_row_warp_ctx_ilp(
    src_ptr,
    weight_ptr,
    dst_ptr,
    row: cutlass.Int32,
    src_row_stride: cutlass.Int32,
    dst_row_stride: cutlass.Int32,
    lane_id: cutlass.Int32,
    eps: cutlass.Constexpr[cutlass.Float32],
    use_pdl: cutlass.Constexpr[bool],
    WIDTH: cutlass.Constexpr[int],
    q8_base,
    sf_base,
    q8_rows: cutlass.Int32,
    emit_q8: cutlass.Constexpr[bool],
    store_row: cutlass.Boolean,
) -> None:
    """``_rmsnorm_row_warp_ctx`` (no kv_append) with the ILP build's instruction diet."""

    PASSES = WIDTH // (_WARP_SIZE * _BF16_VEC_SIZE)
    src_row = row * src_row_stride
    dst_row = row * dst_row_stride

    if cutlass.const_expr(use_pdl):
        cute.arch.griddepcontrol_wait()

    chunks = []
    for pass_idx in cutlass.range_constexpr(PASSES):
        col = (pass_idx * _WARP_SIZE + lane_id) * _BF16_VEC_SIZE
        chunks.append(_load_bf16x8_aligned(src_ptr, src_row + col))

    # the served reduction, instruction for instruction
    total_sum = cutlass.Float32(0.0)
    for pass_idx in cutlass.range_constexpr(PASSES):
        local_sum_sq = cutlass.Float32(0.0)
        for vec_lane in cutlass.range_constexpr(_BF16_VEC_SIZE):
            value_f32 = chunks[pass_idx][vec_lane].to(cutlass.Float32)
            local_sum_sq = local_sum_sq + value_f32 * value_f32
        total_sum = total_sum + _warp_sum_f32(local_sum_sq)

    inv_rms = cute.math.rsqrt(
        total_sum * cutlass.Float32(1.0 / WIDTH) + eps,
        fastmath=True,
    )

    for pass_idx in cutlass.range_constexpr(PASSES):
        col = (pass_idx * _WARP_SIZE + lane_id) * _BF16_VEC_SIZE
        loaded_weights = _load_bf16x8_aligned(weight_ptr, col)
        words = cute.make_rmem_tensor((_BF16_VEC_SIZE // 2,), cutlass.Int32)
        vals = cute.make_rmem_tensor((_BF16_VEC_SIZE,), cutlass.Float32)
        for pair in cutlass.range_constexpr(_BF16_VEC_SIZE // 2):
            x0 = chunks[pass_idx][2 * pair].to(cutlass.Float32)
            x1 = chunks[pass_idx][2 * pair + 1].to(cutlass.Float32)
            # rnd explicit: some cutlass-dsl builds default to RN, public cutlass-dsl 4.8 to none
            n0, n1 = _prims.mul_packed_f32x2((x0, x1), (inv_rms, inv_rms), rnd=_FPRnd.RN)
            n0, n1 = _prims.mul_packed_f32x2(
                (n0, n1),
                (
                    loaded_weights[2 * pair].to(cutlass.Float32),
                    loaded_weights[2 * pair + 1].to(cutlass.Float32),
                ),
                rnd=_FPRnd.RN,
            )
            packed = _cvt_bf16x2_rn(n1, n0)
            words[pair] = packed
            vals[2 * pair] = (packed << cutlass.Int32(16)).bitcast(cutlass.Float32)
            vals[2 * pair + 1] = (packed & cutlass.Int32(-65536)).bitcast(cutlass.Float32)
        if store_row:
            dst = cute.make_tensor(
                cute.recast_ptr(
                    dst_ptr + cute.assume(dst_row + col, _BF16_VEC_SIZE),
                    dtype=cutlass.Int32,
                ),
                cute.make_layout((_BF16_VEC_SIZE // 2,)),
            )
            cute.autovec_copy(words, dst)
        if cutlass.const_expr(emit_q8):
            m01 = cute.arch.fmax(cute.math.absf(vals[0]), cute.math.absf(vals[1]))
            m23 = cute.arch.fmax(cute.math.absf(vals[2]), cute.math.absf(vals[3]))
            m45 = cute.arch.fmax(cute.math.absf(vals[4]), cute.math.absf(vals[5]))
            m67 = cute.arch.fmax(cute.math.absf(vals[6]), cute.math.absf(vals[7]))
            local_max = cute.arch.fmax(cute.arch.fmax(m01, m23), cute.arch.fmax(m45, m67))
            _mxf8_quant_store_x8_ilp(
                vals,
                local_max,
                q8_base,
                cutlass.Int64(row) * WIDTH + cutlass.Int64(col),
                sf_base,
                cutlass.Int64(row),
                cutlass.Int64(col),
                cutlass.Int64(WIDTH // 128),
                row < q8_rows,
            )


@cute.kernel
def _qkv_a_rmsnorm_kernel_warp_row_q8_ctx_ilp(
    q_input: cute.Tensor,
    q_weight: cute.Tensor,
    kv_input: cute.Tensor,
    kv_weight: cute.Tensor,
    q_output: cute.Tensor,
    kv_output: cute.Tensor,
    q8_output: cute.Tensor,
    q8_scales: cute.Tensor,
    q_row_stride: cutlass.Int32,
    kv_row_stride: cutlass.Int32,
    q_dst_row_stride: cutlass.Int32,
    kv_dst_row_stride: cutlass.Int32,
    q_store_from: cutlass.Int32,
    eps: cutlass.Constexpr[cutlass.Float32],
    use_pdl: cutlass.Constexpr[bool],
) -> None:
    """``_qkv_a_rmsnorm_kernel_warp_row_q8_ctx`` (kv_append off) on the ILP build's row
    function. Grid ``(ceil(rows / 8), 2)``."""
    row_block, segment, _ = cute.arch.block_idx()
    grid_x, _, _ = cute.arch.grid_dim()
    tidx, _, _ = cute.arch.thread_idx()
    rows = cute.size(q_input, mode=[0])
    q8_rows = cute.size(q8_output, mode=[0])
    lane_id = tidx % _WARP_SIZE
    row = row_block * _WARPS_PER_BLOCK + tidx // _WARP_SIZE

    if row < rows:
        if segment == 0:
            _rmsnorm_row_warp_ctx_ilp(
                q_input.iterator,
                q_weight.iterator,
                q_output.iterator,
                row,
                q_row_stride,
                q_dst_row_stride,
                lane_id,
                eps,
                use_pdl,
                _Q_DIM,
                q8_output.iterator,
                q8_scales.iterator,
                q8_rows,
                True,
                row >= q_store_from,
            )
        else:
            _rmsnorm_row_warp_ctx_ilp(
                kv_input.iterator,
                kv_weight.iterator,
                kv_output.iterator,
                row,
                kv_row_stride,
                kv_dst_row_stride,
                lane_id,
                eps,
                use_pdl,
                _KV_DIM,
                kv_output.iterator,
                kv_output.iterator,
                cutlass.Int32(0),
                False,
                row >= 0,
            )
    if segment == 1:
        if cutlass.const_expr(use_pdl):
            cute.arch.griddepcontrol_wait()
        _mxf8.mxf8_fill_pad_scales(
            q8_scales.iterator,
            q8_rows,
            cutlass.Int32(_Q_DIM // 128),
            row_block * _THREADS_PER_BLOCK + tidx,
            grid_x * _THREADS_PER_BLOCK,
        )

    cute.arch.barrier(barrier_id=0, number_of_threads=_THREADS_PER_BLOCK)
    if cutlass.const_expr(use_pdl):
        if tidx == 0:
            cute.arch.griddepcontrol_launch_dependents()


@cute.jit
def _launch_qkv_a_rmsnorm_warp_row_q8_ctx_ilp(
    q_input: cute.Tensor,
    q_weight: cute.Tensor,
    kv_input: cute.Tensor,
    kv_weight: cute.Tensor,
    q_output: cute.Tensor,
    kv_output: cute.Tensor,
    q8_output: cute.Tensor,
    q8_scales: cute.Tensor,
    pool: cute.Pointer,
    block_offsets: cute.Tensor,
    cu_seqlens: cute.Tensor,
    cached_lens: cute.Tensor,
    q_row_stride: cutlass.Int32,
    kv_row_stride: cutlass.Int32,
    q_dst_row_stride: cutlass.Int32,
    kv_dst_row_stride: cutlass.Int32,
    q_store_from: cutlass.Int32,
    stream: cuda.CUstream,
    eps: cutlass.Constexpr[cutlass.Float32],
    use_pdl: cutlass.Constexpr[bool],
    kv_append: cutlass.Constexpr[bool],
    tokens_per_block: cutlass.Constexpr[int],
) -> None:
    # the ABI of ``_launch_qkv_a_rmsnorm_warp_row_q8_ctx``; pool / request tables are not read
    rows = cute.size(q_input, mode=[0])
    _qkv_a_rmsnorm_kernel_warp_row_q8_ctx_ilp(
        q_input,
        q_weight,
        kv_input,
        kv_weight,
        q_output,
        kv_output,
        q8_output,
        q8_scales,
        q_row_stride,
        kv_row_stride,
        q_dst_row_stride,
        kv_dst_row_stride,
        q_store_from,
        eps,
        use_pdl,
    ).launch(
        grid=(cute.ceil_div(rows, _WARPS_PER_BLOCK), 2, 1),
        block=(_THREADS_PER_BLOCK, 1, 1),
        stream=stream,
        use_pdl=use_pdl,
    )


def compile_qkv_a_rmsnorm(
    device_id: int = 0,
    eps: float = _DEFAULT_EPS,
    *,
    use_pdl: bool = True,
    inplace_kv_view: bool = False,
) -> Callable:
    """Compile the fused GLM q_a/kv_a RMSNorm candidate.

    :param inplace_kv_view: Compile for strided output views (row pitch
        passed at run time) so ``kv_output`` can be the projection's own kv
        view. Off: compact outputs (same kernel body, compact addressing).
    """

    key = (device_id, float(eps), bool(use_pdl), bool(inplace_kv_view))
    if key in _COMPILE_CACHE:
        return _COMPILE_CACHE[key]

    cutlass.cuda.initialize_cuda_context(device_id)
    sym_rows = cute.sym_int32()
    sym_q_stride = cute.sym_int32(divisibility=_BF16_VEC_SIZE)
    sym_kv_stride = cute.sym_int32(divisibility=_BF16_VEC_SIZE)
    fake_q = make_fake_tensor(
        cutlass.BFloat16,
        (sym_rows, _Q_DIM),
        (sym_q_stride, 1),
        assumed_align=2,
    )
    fake_kv = make_fake_tensor(
        cutlass.BFloat16,
        (sym_rows, _KV_DIM),
        (sym_kv_stride, 1),
        assumed_align=2,
    )
    fake_q_weight = make_fake_compact_tensor(
        cutlass.BFloat16,
        (_Q_DIM,),
        assumed_align=2,
    )
    fake_kv_weight = make_fake_compact_tensor(
        cutlass.BFloat16,
        (_KV_DIM,),
        assumed_align=2,
    )
    if inplace_kv_view:
        sym_q_dst_stride = cute.sym_int32(divisibility=_BF16_VEC_SIZE)
        sym_kv_dst_stride = cute.sym_int32(divisibility=_BF16_VEC_SIZE)
        fake_q_output = make_fake_tensor(
            cutlass.BFloat16,
            (sym_rows, _Q_DIM),
            (sym_q_dst_stride, 1),
            assumed_align=2,
        )
        fake_kv_output = make_fake_tensor(
            cutlass.BFloat16,
            (sym_rows, _KV_DIM),
            (sym_kv_dst_stride, 1),
            assumed_align=2,
        )
    else:
        fake_q_output = make_fake_compact_tensor(
            cutlass.BFloat16,
            (sym_rows, _Q_DIM),
            stride_order=(1, 0),
            assumed_align=2,
        )
        fake_kv_output = make_fake_compact_tensor(
            cutlass.BFloat16,
            (sym_rows, _KV_DIM),
            stride_order=(1, 0),
            assumed_align=2,
        )

    compiled = cute.compile(
        _launch_qkv_a_rmsnorm,
        fake_q,
        fake_q_weight,
        fake_kv,
        fake_kv_weight,
        fake_q_output,
        fake_kv_output,
        cutlass.Int32(0),
        cutlass.Int32(0),
        cutlass.Int32(0),
        cutlass.Int32(0),
        make_fake_stream(),
        cutlass.Float32(eps),
        bool(use_pdl),
        bool(inplace_kv_view),
        options="--enable-tvm-ffi",
    )
    _COMPILE_CACHE[key] = compiled
    return compiled


_Q8_COMPILE_CACHE: dict[tuple[int, float, bool, bool, bool], Callable] = {}


def compile_qkv_a_rmsnorm_q8(
    device_id: int = 0,
    eps: float = _DEFAULT_EPS,
    *,
    use_pdl: bool = True,
    inplace_kv_view: bool = False,
    early_signal: bool = False,
) -> Callable:
    """The CTA kernel with the MXF8 q copy (``_qkv_a_rmsnorm_kernel_q8``, decode rows).

    ``fn(q_in, q_w, kv_in, kv_w, q_out, kv_out, q8 [R8, 2048] uint8 (compact, R8 <= rows),
    sf uint8 (>= act_quant sf_bytes(R8, 2048), 32-byte aligned), q_pitch, kv_pitch,
    q_dst_pitch, kv_dst_pitch, stream)``; operands as ``compile_qkv_a_rmsnorm`` (2-byte
    alignment) plus the q8 buffers (16 / 32 bytes)."""
    key = (
        device_id,
        float(eps),
        bool(use_pdl),
        bool(inplace_kv_view),
        bool(early_signal),
    )
    if key in _Q8_COMPILE_CACHE:
        return _Q8_COMPILE_CACHE[key]
    cutlass.cuda.initialize_cuda_context(device_id)
    fakes = _make_fake_rmsnorm_arguments(bool(inplace_kv_view), 2)
    fake_q8 = make_fake_compact_tensor(
        cutlass.Uint8,
        (cute.sym_int32(), _Q_DIM),
        stride_order=(1, 0),
        assumed_align=_ALIGNMENT_BYTES,
    )
    fake_sf = make_fake_compact_tensor(cutlass.Uint8, (cute.sym_int32(),), assumed_align=32)
    compiled = cute.compile(
        _launch_qkv_a_rmsnorm_q8,
        *fakes,
        fake_q8,
        fake_sf,
        cutlass.Int32(0),
        cutlass.Int32(0),
        cutlass.Int32(0),
        cutlass.Int32(0),
        make_fake_stream(),
        cutlass.Float32(eps),
        bool(use_pdl),
        bool(inplace_kv_view),
        bool(early_signal),
        options="--enable-tvm-ffi",
    )
    _Q8_COMPILE_CACHE[key] = compiled
    return compiled


def _current_cuda_stream(torch, device=None):
    """Stream of ``device`` (the tensors' device), not of the current device."""
    return cuda.CUstream(torch.cuda.current_stream(device=device).cuda_stream)


def _require_bf16_cuda_inner_contiguous(torch, tensor, name: str) -> None:
    if tensor.dtype != torch.bfloat16:
        raise TypeError(f"{name} must be torch.bfloat16, got {tensor.dtype}")
    if not tensor.is_cuda:
        raise TypeError(f"{name} must be a CUDA tensor")
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {tuple(tensor.shape)}")
    if tensor.stride(-1) != 1:
        raise ValueError(f"{name} must have innermost stride 1, got {tensor.stride(-1)}")


def _require_bf16_cuda_contiguous(torch, tensor, name: str) -> None:
    _require_bf16_cuda_inner_contiguous(torch, tensor, name)
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _storage_identity(tensor) -> int:
    """Return the identity of the complete backing storage, not a view pointer."""

    return int(tensor.untyped_storage()._cdata)


def _same_exact_view(lhs, rhs) -> bool:
    return (
        _storage_identity(lhs) == _storage_identity(rhs)
        and lhs.storage_offset() == rhs.storage_offset()
        and tuple(lhs.shape) == tuple(rhs.shape)
        and tuple(lhs.stride()) == tuple(rhs.stride())
        and lhs.data_ptr() == rhs.data_ptr()
    )


def _require_vector_row_layout(tensor, name: str, width: int) -> None:
    rows = int(tensor.shape[0])
    if rows <= 0:
        raise ValueError(f"{name} must have at least one row")
    row_stride = int(tensor.stride(0))
    if row_stride < width:
        raise ValueError(f"{name} row stride must be at least {width}, got {row_stride}")
    if row_stride % _BF16_VEC_SIZE != 0:
        raise ValueError(
            f"{name} row stride must be divisible by {_BF16_VEC_SIZE}, got {row_stride}"
        )
    if tensor.data_ptr() % _ALIGNMENT_BYTES != 0:
        raise ValueError(f"{name} data pointer must be {_ALIGNMENT_BYTES}-byte aligned")
    storage_elements = tensor.untyped_storage().nbytes() // tensor.element_size()
    last_element = int(tensor.storage_offset()) + (rows - 1) * row_stride + width - 1
    if last_element >= storage_elements:
        raise ValueError(
            f"{name} layout exceeds backing storage: last={last_element}, "
            f"storage_elements={storage_elements}"
        )


def _validate_output_aliases(
    q_input,
    q_weight,
    kv_input,
    kv_weight,
    q_output,
    kv_output,
) -> bool:
    """Reject every output alias except the exact in-place KV input/output view."""

    q_output_storage = _storage_identity(q_output)
    forbidden_q_storages = {
        _storage_identity(q_input),
        _storage_identity(kv_input),
        _storage_identity(q_weight),
        _storage_identity(kv_weight),
        _storage_identity(kv_output),
    }
    if q_output_storage in forbidden_q_storages:
        raise ValueError("q_output must have distinct storage from all inputs and kv_output")

    kv_output_storage = _storage_identity(kv_output)
    kv_input_storage = _storage_identity(kv_input)
    if kv_output_storage != kv_input_storage:
        forbidden_kv_storages = {
            _storage_identity(q_input),
            _storage_identity(q_weight),
            _storage_identity(kv_weight),
        }
        if kv_output_storage in forbidden_kv_storages:
            raise ValueError("out-of-place kv_output must have distinct storage")
        return False

    if not _same_exact_view(kv_output, kv_input):
        raise ValueError(
            "kv_output sharing kv_input storage must be the exact same view; "
            "partial overlap is forbidden"
        )
    # The parent is the [rows, 2624] q|kv|k_pe projection of the call; rows is
    # the padded decode row count, and the row pitch is the same for every row
    # count.  The kernel itself is row-wise (sym rows).
    rows = int(q_input.shape[0])
    if (
        rows <= 0
        or tuple(kv_input.shape) != (rows, _KV_DIM)
        or tuple(kv_input.stride()) != (_PROJECTION_DIM, 1)
        or tuple(q_input.shape) != (rows, _Q_DIM)
        or tuple(q_input.stride()) != (_PROJECTION_DIM, 1)
        or _storage_identity(q_input) != kv_input_storage
        or int(kv_input.storage_offset()) != int(q_input.storage_offset()) + _Q_DIM
    ):
        raise ValueError(
            "in-place kv_output is allowed only for the exact shared "
            "q|kv|k_pe parent layout ([rows, 2624], row pitch 2624)"
        )
    parent_last = int(q_input.storage_offset()) + rows * _PROJECTION_DIM - 1
    storage_elements = q_input.untyped_storage().nbytes() // q_input.element_size()
    if parent_last >= storage_elements:
        raise ValueError("q|kv|k_pe parent storage is too small for the untouched K-PE tail")
    return True


def _require_bf16_cuda_vector(torch, tensor, name: str, length: int) -> None:
    if tensor.dtype != torch.bfloat16:
        raise TypeError(f"{name} must be torch.bfloat16, got {tensor.dtype}")
    if not tensor.is_cuda:
        raise TypeError(f"{name} must be a CUDA tensor")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if tensor.shape != (length,):
        raise ValueError(f"{name} must have shape [{length}], got {tuple(tensor.shape)}")


def _use_warp_row_path(*tensors) -> bool:
    """Large-T selection: prefill row count, 16-byte-aligned data pointers AND 16-byte-aligned row
    pitches (the warp-row kernel asserts ``cute.assume(row * stride + col, 8)`` on every 2-D operand, so a
    view such as ``torch.empty((rows, 2049))[:, :2048]`` must take the CTA kernel instead)."""
    return int(tensors[0].shape[0]) >= _LARGE_T_MIN_ROWS and all(
        t.data_ptr() % _ALIGNMENT_BYTES == 0
        and (t.ndim < 2 or int(t.stride(0)) % _BF16_VEC_SIZE == 0)
        for t in tensors
    )


def qkv_a_rmsnorm_run(
    q_input,
    q_weight,
    kv_input,
    kv_weight,
    eps: float = _DEFAULT_EPS,
    compiled_fn: Callable | None = None,
    q_output=None,
    kv_output=None,
    *,
    inplace_kv_view: bool = False,
):
    """Run fused GLM q_a/kv_a RMSNorm and return the q/kv outputs.

    Default: compact outputs (allocated when not given). With
    ``inplace_kv_view`` the outputs may be inner-contiguous strided
    views; ``kv_output`` may alias ``kv_input`` only as the exact same view of
    the shared-M64 ``q|kv|k_pe`` projection (validated fail-closed).
    """

    import torch

    require_output = (
        _require_bf16_cuda_inner_contiguous if inplace_kv_view else _require_bf16_cuda_contiguous
    )

    _require_bf16_cuda_inner_contiguous(torch, q_input, "q_input")
    _require_bf16_cuda_inner_contiguous(torch, kv_input, "kv_input")
    _require_bf16_cuda_vector(torch, q_weight, "q_weight", _Q_DIM)
    _require_bf16_cuda_vector(torch, kv_weight, "kv_weight", _KV_DIM)
    if q_input.shape[1] != _Q_DIM:
        raise ValueError(f"q_input must have shape [T,{_Q_DIM}], got {tuple(q_input.shape)}")
    if kv_input.shape[1] != _KV_DIM:
        raise ValueError(f"kv_input must have shape [T,{_KV_DIM}], got {tuple(kv_input.shape)}")
    if q_input.shape[0] != kv_input.shape[0]:
        raise ValueError("q_input and kv_input must agree on T")
    if q_input.device != kv_input.device:
        raise ValueError("q_input and kv_input must be on the same CUDA device")
    if q_output is None:
        q_output = torch.empty(
            (q_input.shape[0], _Q_DIM),
            dtype=torch.bfloat16,
            device=q_input.device,
        )
    else:
        require_output(torch, q_output, "q_output")
        if q_output.shape != (q_input.shape[0], _Q_DIM):
            raise ValueError(f"q_output shape mismatch: {tuple(q_output.shape)}")

    if kv_output is None:
        kv_output = torch.empty(
            (kv_input.shape[0], _KV_DIM),
            dtype=torch.bfloat16,
            device=kv_input.device,
        )
    else:
        require_output(torch, kv_output, "kv_output")
        if kv_output.shape != (kv_input.shape[0], _KV_DIM):
            raise ValueError(f"kv_output shape mismatch: {tuple(kv_output.shape)}")

    if inplace_kv_view:
        _require_vector_row_layout(q_input, "q_input", _Q_DIM)
        _require_vector_row_layout(kv_input, "kv_input", _KV_DIM)
        _require_vector_row_layout(q_output, "q_output", _Q_DIM)
        _require_vector_row_layout(kv_output, "kv_output", _KV_DIM)
        _validate_output_aliases(q_input, q_weight, kv_input, kv_weight, q_output, kv_output)

    if compiled_fn is None:
        # Compile for the tensors' device, not the current device: callers may
        # hold views on a non-current GPU.  Prefill row counts take the
        # warp-per-row kernel (same bytes, see module docstring); decode
        # cells (<= 64 rows) keep the CTA kernel.
        compile_fn = (
            compile_qkv_a_rmsnorm_warp_row
            if _use_warp_row_path(q_input, q_weight, kv_input, kv_weight, q_output, kv_output)
            else compile_qkv_a_rmsnorm
        )
        compiled_fn = compile_fn(
            device_id=q_input.device.index,
            eps=eps,
            inplace_kv_view=inplace_kv_view,
        )

    compiled_fn(
        q_input,
        q_weight,
        kv_input,
        kv_weight,
        q_output,
        kv_output,
        int(q_input.stride(0)),
        int(kv_input.stride(0)),
        int(q_output.stride(0)),
        int(kv_output.stride(0)),
        _current_cuda_stream(torch, q_input.device),
    )
    return q_output, kv_output
