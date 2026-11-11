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
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream, make_fake_tensor
from cutlass.memory import SmemAllocator

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
) -> None:
    """Normalize one row with a single warp; same FP32 reduction order as
    ``_rmsnorm_row`` (see the module docstring), so the bytes are identical."""

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
