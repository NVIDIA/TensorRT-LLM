# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The end of a MegaMoE layer whose top-k weights apply in the reduce, in one kernel.

A MegaMoE launch with ``defer_topk_reduce=True`` and ``apply_topk_in_fc1=False`` leaves one
unweighted FC2 row per (token, top-k slot) in its pre-reduced plane. The stock layer then ends
with TopkReduce (weights applied), ``shared.add_(routed)`` and the next layer's input norm. Per
token row ``t``, with scores ``s``:

    acc         = plane[t, 0] * s[t, 0]                    (FP32 mul)
    acc         = fma(plane[t, k], s[t, k], acc)           (FP32, k = 1 .. K - 1 in slot order)
    routed      = bf16(acc)
    moe         = bf16(shared[t] + routed)                 (FP32 add)

``megamoe_topk_reduce_shared_add`` stops there and writes ``moe`` into ``shared``; the caller's
norm then runs unchanged. ``megamoe_topk_reduce_add_rmsnorm`` continues as FlashInfer's CuTe DSL
``FusedAddRMSNormKernel`` does:

    h           = moe + residual[t]                        (FP32)
    residual[t] = bf16(h)
    shared[t]   = bf16(h * rsqrt(mean(h * h) + eps) * weight)

The reduce uses TopkReduce's packed FP32 multiply and fused multiply-add in its slot order, so
both forms are bit-identical to the stock kernels. The norm form is bit-identical only where
FlashInfer runs that kernel with one CTA per row (see
``megamoe_topk_reduce_add_rmsnorm_matches_flashinfer``): the sum of squares runs FlashInfer's
``row_reduce_sum_multirow`` over that kernel's per-thread column sets, and the remaining
statements are FlashInfer's.

Given MXFP8 output buffers, the norm form also writes the normalized rows' MXFP8 form
(``mxfp8_rows.store_mxfp8_vector``), bit for bit ``trtllm::mxfp8_quantize(shared, True)``.

One CTA owns one token row; each thread loads whole 8-element vectors of every input, so the K
plane rows of a vector are in flight together. With PDL, the kernels read the plane, the scores,
the shared-expert output and the residual only after ``griddepcontrol.wait``, and release their
dependents right after it: a dependent reads the outputs only after its own wait, which returns
once this grid has completed.
"""

import functools
from typing import Optional

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64

from .mxfp8_rows import (
    check_mxfp8_rows,
    mxfp8_fakes,
    rcp_approx_ftz,
    store_mxfp8_vector,
    zero_mxfp8_pad_rows,
)

_VEC = 8  # BF16 elements per 16-byte vector
# Above this many rows the shared-add form caps its CTA at _PREFILL_MAX_THREADS threads: with one H / 8-thread CTA
# per row, too few CTAs stay resident per SM to keep the plane loads in flight (H = 6144, K = 4;
# up to 1024 rows the geometry is unchanged).
_PREFILL_MIN_ROWS = 1025
_PREFILL_MAX_THREADS = 384


def _aligned(tensor: cute.Tensor, byte_alignment: int) -> cute.Tensor:
    pointer = tensor.iterator
    return cute.make_tensor(
        cute.make_ptr(
            pointer.dtype, pointer.toint(), pointer.memspace, assumed_align=byte_alignment
        ),
        tensor.layout,
    )


def _num_threads(hidden_size: int, min_threads: int, max_threads: int = 1024) -> int:
    """The largest CTA (at most ``max_threads`` threads, whole warps) giving each thread whole vectors."""
    num_vectors = hidden_size // _VEC
    for warps in range(max_threads // 32, 0, -1):
        threads = warps * 32
        if num_vectors % threads == 0 and threads >= min_threads:
            return threads
    return 0


class _TopkReduceSharedAdd:
    """Weighted top-k reduce of the plane plus the shared-expert output, per 8-element vector."""

    def __init__(self, hidden_size: int, num_topk: int, num_threads: int):
        num_vectors = hidden_size // _VEC
        if hidden_size % _VEC != 0 or num_threads % 32 != 0 or num_vectors % num_threads != 0:
            raise ValueError(
                f"{num_threads} threads do not tile the {num_vectors} vectors of a "
                f"{hidden_size}-wide row"
            )
        if num_topk <= 0:
            raise ValueError(f"num_topk must be positive, got {num_topk}")
        self.H = hidden_size
        self.num_topk = num_topk
        self.num_threads = num_threads
        self.vectors_per_thread = num_vectors // num_threads

    @cute.jit
    def _load_row(self, mPlane, mScores, mShared, row, tidx, terms, score_reg, shared_vec):
        num_topk: cutlass.Constexpr[int] = self.num_topk
        num_threads: cutlass.Constexpr[int] = self.num_threads
        vectors_per_thread: cutlass.Constexpr[int] = self.vectors_per_thread
        bf16_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=128
        )
        vec_bytes = _VEC * cutlass.BFloat16.width // 8
        gShared = cute.zipped_divide(mShared[row, None], (_VEC,))
        gPlane = cute.zipped_divide(mPlane[row, None, None], (num_topk, _VEC))
        cute.autovec_copy(mScores[row, None], score_reg)
        for i in cutlass.range_constexpr(vectors_per_thread):
            v = tidx + i * num_threads
            plane_vec = gPlane[(None, None), (0, v)]
            for k in cutlass.range_constexpr(num_topk):
                cute.copy(bf16_atom, _aligned(plane_vec[k, None], vec_bytes), terms[None, k, i])
            cute.copy(bf16_atom, _aligned(gShared[(None,), v], vec_bytes), shared_vec[None, i])

    @cute.jit
    def _reduce_add(self, terms, score_reg, shared_vec, i: cutlass.Constexpr[int], moe):
        num_topk: cutlass.Constexpr[int] = self.num_topk
        # TopkReduce: FP32 slot-order mul / fma with the top-k score, rounded to BF16.
        acc = cute.make_rmem_tensor((_VEC,), Float32)
        for k in cutlass.range_constexpr(num_topk):
            score_pair = (Float32(score_reg[k]), Float32(score_reg[k]))
            for j in cutlass.range_constexpr(0, _VEC, 2):
                value_pair = (Float32(terms[j, k, i]), Float32(terms[j + 1, k, i]))
                if cutlass.const_expr(k == 0):
                    acc[j], acc[j + 1] = cute.arch.mul_packed_f32x2(value_pair, score_pair)
                else:
                    acc[j], acc[j + 1] = cute.arch.fma_packed_f32x2(
                        value_pair, score_pair, (acc[j], acc[j + 1])
                    )
        routed = cute.make_rmem_tensor((_VEC,), cutlass.BFloat16)
        routed.store(acc.load().to(cutlass.BFloat16))
        # shared + routed in FP32, rounded to BF16 (PyTorch's BF16 add).
        moe_f32 = cute.make_rmem_tensor((_VEC,), Float32)
        for j in cutlass.range_constexpr(_VEC):
            moe_f32[j] = Float32(shared_vec[j, i]) + Float32(routed[j])
        moe.store(moe_f32.load().to(cutlass.BFloat16))


class TopkReduceSharedAddKernel(_TopkReduceSharedAdd):
    """Weighted top-k reduce plus the shared-expert output, written into the shared rows.

    Args:
        hidden_size: Row length H.
        num_topk: Top-k slots K per token in the plane.
        num_threads: Threads per CTA; H / 8 must be a multiple of it.
    """

    @cute.jit
    def __call__(
        self,
        mPlane: cute.Tensor,
        mScores: cute.Tensor,
        mShared: cute.Tensor,
        M: Int64,
        enable_pdl: cutlass.Constexpr[bool],
        stream,
    ):
        self.kernel(mPlane, mScores, mShared, enable_pdl).launch(
            grid=[M, 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
            use_pdl=enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        mPlane: cute.Tensor,
        mScores: cute.Tensor,
        mShared: cute.Tensor,
        enable_pdl: cutlass.Constexpr[bool],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        num_topk: cutlass.Constexpr[int] = self.num_topk
        num_threads: cutlass.Constexpr[int] = self.num_threads
        vectors_per_thread: cutlass.Constexpr[int] = self.vectors_per_thread
        bf16_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=128
        )
        vec_bytes = _VEC * cutlass.BFloat16.width // 8
        gShared = cute.zipped_divide(mShared[row, None], (_VEC,))

        if cutlass.const_expr(enable_pdl):
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()

        terms = cute.make_rmem_tensor((_VEC, num_topk, vectors_per_thread), cutlass.BFloat16)
        score_reg = cute.make_rmem_tensor((num_topk,), Float32)
        shared_vec = cute.make_rmem_tensor((_VEC, vectors_per_thread), cutlass.BFloat16)
        self._load_row(mPlane, mScores, mShared, row, tidx, terms, score_reg, shared_vec)

        for i in cutlass.range_constexpr(vectors_per_thread):
            v = tidx + i * num_threads
            moe = cute.make_rmem_tensor((_VEC,), cutlass.BFloat16)
            self._reduce_add(terms, score_reg, shared_vec, i, moe)
            cute.copy(bf16_atom, moe, _aligned(gShared[(None,), v], vec_bytes))


class TopkReduceAddRMSNormKernel(_TopkReduceSharedAdd):
    """Weighted top-k reduce, shared add and residual add + RMSNorm of BF16 token rows.

    Args:
        hidden_size: Row length H.
        num_topk: Top-k slots K per token in the plane.
        num_threads: Threads per CTA; H / 8 must be a multiple of it.
        emit_mxfp8: Also write the normalized rows' MXFP8 form into the q / sf arguments (H a multiple of 128);
            without it the kernel never touches them.
    """

    def __init__(self, hidden_size: int, num_topk: int, num_threads: int, emit_mxfp8: bool = False):
        from flashinfer.norm.kernels.fused_add_rmsnorm import FusedAddRMSNormKernel
        from flashinfer.norm.utils import row_reduce_sum_multirow

        super().__init__(hidden_size, num_topk, num_threads)
        if emit_mxfp8 and hidden_size % 128 != 0:
            raise ValueError(
                f"MXFP8 rows need a hidden size that is a multiple of 128, got {hidden_size}"
            )
        self.emit_mxfp8 = emit_mxfp8
        self._row_reduce_sum_multirow = row_reduce_sum_multirow
        reference = FusedAddRMSNormKernel(cutlass.BFloat16, hidden_size, weight_bias=0.0)
        if reference.cluster_n != 1 or reference.vec_size != _VEC:
            raise ValueError(
                f"hidden size {hidden_size} does not map to one CTA per row with 8-element "
                "vectors in FusedAddRMSNormKernel"
            )
        if reference.threads_per_row > num_threads:
            raise ValueError("the row reduction needs more threads than the CTA has")
        if reference.threads_per_row * reference.num_vec_blocks * _VEC != hidden_size:
            raise ValueError("FusedAddRMSNormKernel does not cover the row exactly")
        self.weight_bias = reference.weight_bias
        # FlashInfer's per-thread geometry for this H: thread t of the row holds the vectors
        # t, t + threads_per_row, ... (num_vec_blocks of them).
        self.ref_threads_per_row = reference.threads_per_row
        self.ref_warps_per_row = reference.warps_per_row
        self.ref_vec_blocks = reference.num_vec_blocks

    def _smem_size_in_bytes(self) -> int:
        return self.H * 4 + self.ref_warps_per_row * 4 + 16

    @cute.jit
    def __call__(
        self,
        mPlane: cute.Tensor,
        mScores: cute.Tensor,
        mShared: cute.Tensor,
        mResidual: cute.Tensor,
        mW: cute.Tensor,
        mQ: cute.Tensor,
        mSF: cute.Tensor,
        M: Int64,
        eps: Float32,
        enable_pdl: cutlass.Constexpr[bool],
        stream,
    ):
        self.kernel(
            mPlane, mScores, mShared, mResidual, mW, mQ, mSF, Int32(M), eps, enable_pdl
        ).launch(
            grid=[M, 1, 1],
            block=[self.num_threads, 1, 1],
            smem=self._smem_size_in_bytes(),
            stream=stream,
            use_pdl=enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        mPlane: cute.Tensor,
        mScores: cute.Tensor,
        mShared: cute.Tensor,
        mResidual: cute.Tensor,
        mW: cute.Tensor,
        mQ: cute.Tensor,
        mSF: cute.Tensor,
        num_rows: Int32,
        eps: Float32,
        enable_pdl: cutlass.Constexpr[bool],
    ):
        row_reduce_sum_multirow = self._row_reduce_sum_multirow
        tidx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        warp_idx = cute.arch.warp_idx()

        H = self.H
        num_topk: cutlass.Constexpr[int] = self.num_topk
        num_threads: cutlass.Constexpr[int] = self.num_threads
        vectors_per_thread: cutlass.Constexpr[int] = self.vectors_per_thread
        ref_threads_per_row: cutlass.Constexpr[int] = self.ref_threads_per_row
        ref_warps_per_row: cutlass.Constexpr[int] = self.ref_warps_per_row
        ref_vec_blocks: cutlass.Constexpr[int] = self.ref_vec_blocks
        weight_bias = self.weight_bias

        smem = cutlass.utils.SmemAllocator()
        sH = smem.allocate_tensor(Float32, cute.make_layout((H,)), byte_alignment=16)
        reduction_buffer = smem.allocate_tensor(
            Float32, cute.make_layout((1, ref_warps_per_row)), byte_alignment=4
        )
        sRstd = smem.allocate_tensor(Float32, cute.make_layout((1,)), byte_alignment=4)
        vH = cute.zipped_divide(sH, (_VEC,))

        bf16_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=128
        )
        vec_bytes = _VEC * cutlass.BFloat16.width // 8

        gW = cute.zipped_divide(mW, (_VEC,))
        gShared = cute.zipped_divide(mShared[row, None], (_VEC,))
        gResidual = cute.zipped_divide(mResidual[row, None], (_VEC,))

        # The norm weight is a parameter, written before any forward: load it before the wait.
        w_vec = cute.make_rmem_tensor((_VEC, vectors_per_thread), cutlass.BFloat16)
        for i in cutlass.range_constexpr(vectors_per_thread):
            v = tidx + i * num_threads
            cute.copy(bf16_atom, _aligned(gW[(None,), v], vec_bytes), w_vec[None, i])

        if cutlass.const_expr(enable_pdl):
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()

        terms = cute.make_rmem_tensor((_VEC, num_topk, vectors_per_thread), cutlass.BFloat16)
        score_reg = cute.make_rmem_tensor((num_topk,), Float32)
        shared_vec = cute.make_rmem_tensor((_VEC, vectors_per_thread), cutlass.BFloat16)
        self._load_row(mPlane, mScores, mShared, row, tidx, terms, score_reg, shared_vec)
        residual_vec = cute.make_rmem_tensor((_VEC, vectors_per_thread), cutlass.BFloat16)
        for i in cutlass.range_constexpr(vectors_per_thread):
            v = tidx + i * num_threads
            cute.copy(bf16_atom, _aligned(gResidual[(None,), v], vec_bytes), residual_vec[None, i])

        h_vec = cute.make_rmem_tensor((_VEC, vectors_per_thread), Float32)
        for i in cutlass.range_constexpr(vectors_per_thread):
            v = tidx + i * num_threads
            moe = cute.make_rmem_tensor((_VEC,), cutlass.BFloat16)
            self._reduce_add(terms, score_reg, shared_vec, i, moe)
            # FusedAddRMSNormKernel: h = input + residual in FP32; the residual keeps bf16(h).
            h = moe.load().to(Float32) + residual_vec[None, i].load().to(Float32)
            h_vec[None, i].store(h)
            residual_out = cute.make_rmem_tensor((_VEC,), cutlass.BFloat16)
            residual_out.store(h.to(cutlass.BFloat16))
            cute.copy(bf16_atom, residual_out, _aligned(gResidual[(None,), v], vec_bytes))
            cute.autovec_copy(h_vec[None, i], vH[(None,), v])

        cute.arch.barrier()

        # Sum of squares as FusedAddRMSNormKernel computes it: thread t of the first
        # ref_threads_per_row threads holds FlashInfer thread t's columns, vector blocks in order.
        if warp_idx < ref_warps_per_row:
            fragment = cute.make_rmem_tensor((_VEC, ref_vec_blocks), Float32)
            for b in cutlass.range_constexpr(ref_vec_blocks):
                cute.autovec_copy(vH[(None,), tidx + b * ref_threads_per_row], fragment[None, b])
            h_rows = fragment.load()
            h_sq = h_rows * h_rows
            sum_sq = row_reduce_sum_multirow(h_sq, ref_threads_per_row, reduction_buffer, None, 1)
            mean_sq = sum_sq / Float32(H)
            rstd = cute.math.rsqrt(mean_sq + eps, fastmath=True)
            if tidx == 0:
                sRstd[0] = rstd
        else:
            # The reduction's block barrier.
            cute.arch.barrier()

        cute.arch.barrier()
        rstd = sRstd[0]

        emit_mxfp8: cutlass.Constexpr[bool] = self.emit_mxfp8
        num_k_tiles: cutlass.Constexpr[int] = (H // 32 + 3) // 4
        for i in cutlass.range_constexpr(vectors_per_thread):
            v = tidx + i * num_threads
            w = w_vec[None, i].load().to(Float32)
            y = h_vec[None, i].load() * rstd * (w + Float32(weight_bias))
            out = cute.make_rmem_tensor((_VEC,), cutlass.BFloat16)
            out.store(y.to(cutlass.BFloat16))
            cute.copy(bf16_atom, out, _aligned(gShared[(None,), v], vec_bytes))
            if cutlass.const_expr(emit_mxfp8):
                store_mxfp8_vector(
                    out, mQ, mSF, row, v, num_k_tiles, rcp_approx_ftz(Float32(448.0))
                )

        if cutlass.const_expr(emit_mxfp8):
            zero_mxfp8_pad_rows(mSF, row, num_rows, tidx, num_threads, num_k_tiles)


def megamoe_topk_reduce_add_rmsnorm_matches_flashinfer(
    device: torch.device, hidden_size: int
) -> bool:
    """Whether ``flashinfer.norm.fused_add_rmsnorm`` runs the kernel the norm form copies.

    FlashInfer's CUDA JIT norm and the cluster variant of ``FusedAddRMSNormKernel`` (very large
    hidden sizes) are not covered, nor is a FlashInfer without that kernel.
    """
    try:
        import flashinfer.norm
        from flashinfer.norm.kernels.fused_add_rmsnorm import FusedAddRMSNormKernel
        from flashinfer.norm.utils import row_reduce_sum_multirow  # noqa: F401
    except ImportError:
        return False
    use_cuda_norm = getattr(flashinfer.norm, "_use_cuda_norm", None)
    if use_cuda_norm is None or use_cuda_norm(device) or hidden_size % _VEC != 0:
        return False
    with torch.cuda.device(device):
        reference = FusedAddRMSNormKernel(cutlass.BFloat16, hidden_size, weight_bias=0.0)
    return (
        reference.cluster_n == 1
        and reference.vec_size == _VEC
        and reference.threads_per_row * reference.num_vec_blocks * _VEC == hidden_size
        and _num_threads(hidden_size, reference.threads_per_row) > 0
    )


def _fake_inputs(hidden_size: int, num_topk: int):
    sym_m = cute.sym_int(64)
    plane = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (sym_m, num_topk, hidden_size), stride_order=(2, 1, 0), assumed_align=16
    )
    scores = cute.runtime.make_fake_compact_tensor(
        Float32, (sym_m, num_topk), stride_order=(1, 0), assumed_align=4
    )
    rows = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (sym_m, hidden_size), stride_order=(1, 0), assumed_align=16
    )
    return sym_m, plane, scores, rows


@functools.cache
def _compile_topk_reduce_shared_add(
    hidden_size: int, num_topk: int, enable_pdl: bool, max_threads: int = 1024
):
    num_threads = _num_threads(hidden_size, 32, max_threads)
    if num_threads == 0:
        raise ValueError(f"hidden size {hidden_size} does not tile into 8-element vectors")
    kernel_obj = TopkReduceSharedAddKernel(hidden_size, num_topk, num_threads)
    _, plane_fake, scores_fake, shared_fake = _fake_inputs(hidden_size, num_topk)
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        kernel_obj,
        plane_fake,
        scores_fake,
        shared_fake,
        Int64(1),
        enable_pdl,
        stream_fake,
        options="--enable-tvm-ffi",
    )


@functools.cache
def _compile_topk_reduce_add_rmsnorm(
    hidden_size: int, num_topk: int, enable_pdl: bool, emit_mxfp8: bool = False
):
    from flashinfer.norm.kernels.fused_add_rmsnorm import FusedAddRMSNormKernel

    reference = FusedAddRMSNormKernel(cutlass.BFloat16, hidden_size, weight_bias=0.0)
    num_threads = _num_threads(hidden_size, reference.threads_per_row)
    kernel_obj = TopkReduceAddRMSNormKernel(hidden_size, num_topk, num_threads, emit_mxfp8)
    sym_m, plane_fake, scores_fake, shared_fake = _fake_inputs(hidden_size, num_topk)
    residual_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (sym_m, hidden_size), stride_order=(1, 0), assumed_align=16
    )
    w_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (hidden_size,), assumed_align=16
    )
    # Own row count: without the epilogue the kernel takes a one-row placeholder.
    q_fake, sf_fake = mxfp8_fakes(hidden_size, cute.sym_int())
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    # Same compile options as FlashInfer's norm kernels.
    return cute.compile(
        kernel_obj,
        plane_fake,
        scores_fake,
        shared_fake,
        residual_fake,
        w_fake,
        q_fake,
        sf_fake,
        Int64(1),
        Float32(1e-6),
        enable_pdl,
        stream_fake,
        options="--enable-tvm-ffi",
    )


def _check_rows(plane: torch.Tensor, scores: torch.Tensor, shared: torch.Tensor) -> None:
    if plane.dim() != 3 or scores.dim() != 2 or shared.dim() != 2:
        raise ValueError("plane is [M, K, H], scores [M, K] and shared [M, H]")
    num_rows, num_topk, hidden_size = plane.shape
    if tuple(scores.shape) != (num_rows, num_topk) or tuple(shared.shape) != (
        num_rows,
        hidden_size,
    ):
        raise ValueError(
            f"shapes disagree: plane {tuple(plane.shape)}, scores {tuple(scores.shape)}, "
            f"shared {tuple(shared.shape)}"
        )
    if plane.dtype != torch.bfloat16 or shared.dtype != torch.bfloat16:
        raise ValueError("plane and shared must be BF16")
    if scores.dtype != torch.float32:
        raise ValueError("scores must be FP32")
    for name, tensor in (("plane", plane), ("scores", scores), ("shared", shared)):
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if plane.data_ptr() % 16 != 0 or shared.data_ptr() % 16 != 0:
        raise ValueError("plane and shared must be 16-byte aligned")


def megamoe_topk_reduce_shared_add(
    plane: torch.Tensor,
    scores: torch.Tensor,
    shared: torch.Tensor,
    *,
    enable_pdl: bool,
) -> None:
    """Weighted top-k reduce of the plane plus the shared-expert output, into ``shared``.

    Same values as ``shared.add_(TopkReduce(plane, scores))`` with the BF16 combine format.

    Args:
        plane: Pre-reduced plane [M, K, H], BF16, contiguous: one unweighted FC2 row per
            (token, top-k slot).
        scores: Top-k scores [M, K], FP32, contiguous, in the plane's slot order.
        shared: Shared-expert output [M, H], BF16, contiguous. Receives the sum.
        enable_pdl: Launch with programmatic dependent launch.
    """
    _check_rows(plane, scores, shared)
    num_rows, num_topk, hidden_size = plane.shape
    if num_rows == 0:
        return
    max_threads = _PREFILL_MAX_THREADS if num_rows >= _PREFILL_MIN_ROWS else 1024
    kernel = _compile_topk_reduce_shared_add(hidden_size, num_topk, enable_pdl, max_threads)
    kernel(plane, scores, shared, num_rows)


def megamoe_topk_reduce_add_rmsnorm(
    plane: torch.Tensor,
    scores: torch.Tensor,
    shared: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    enable_pdl: bool,
    q: Optional[torch.Tensor] = None,
    sf: Optional[torch.Tensor] = None,
) -> None:
    """Weighted top-k reduce, shared add, residual add and RMSNorm, in place.

    Same values as ``shared.add_(TopkReduce(plane, scores))`` followed by
    ``flashinfer.norm.fused_add_rmsnorm(shared, residual, weight, eps)`` where
    ``megamoe_topk_reduce_add_rmsnorm_matches_flashinfer`` holds; given ``q`` and ``sf``, also
    ``trtllm::mxfp8_quantize(shared, True)`` of the normalized rows.

    Args:
        plane: Pre-reduced plane [M, K, H], BF16, contiguous: one unweighted FC2 row per
            (token, top-k slot).
        scores: Top-k scores [M, K], FP32, contiguous, in the plane's slot order.
        shared: Shared-expert output [M, H], BF16, contiguous. Receives the normalized rows.
        residual: Residual [M, H], BF16, contiguous. Receives the new residual.
        weight: RMSNorm weight [H], BF16, contiguous.
        eps: RMSNorm epsilon.
        enable_pdl: Launch with programmatic dependent launch.
        q: E4M3 [M, H], contiguous, or None. Receives the normalized rows' MXFP8 form.
        sf: UINT8, at least ``mxfp8_rows.scale_bytes(M, H)``, given with ``q``. Receives the
            swizzled UE8M0 scales.
    """
    if (q is None) != (sf is None):
        raise ValueError("q and sf go together")
    _check_rows(plane, scores, shared)
    num_rows, num_topk, hidden_size = plane.shape
    if tuple(residual.shape) != tuple(shared.shape) or tuple(weight.shape) != (hidden_size,):
        raise ValueError(
            f"residual {tuple(residual.shape)} and weight {tuple(weight.shape)} do not match "
            f"shared {tuple(shared.shape)}"
        )
    if residual.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("residual and weight must be BF16")
    if not residual.is_contiguous() or not weight.is_contiguous():
        raise ValueError("residual and weight must be contiguous")
    if residual.data_ptr() % 16 != 0 or weight.data_ptr() % 16 != 0:
        raise ValueError("residual and weight must be 16-byte aligned")
    if q is not None:
        check_mxfp8_rows(q, sf, num_rows, hidden_size)
    if num_rows == 0:
        return
    emit_mxfp8 = q is not None
    if emit_mxfp8:
        q_words = q.view(torch.int64)
    else:
        q_words, sf = _no_mxfp8_rows(hidden_size, shared.device)
    kernel = _compile_topk_reduce_add_rmsnorm(hidden_size, num_topk, enable_pdl, emit_mxfp8)
    kernel(plane, scores, shared, residual, weight, q_words, sf, num_rows, eps)


@functools.cache
def _no_mxfp8_rows(hidden_size: int, device: torch.device):
    """One-row placeholders for the MXFP8 arguments of a kernel compiled without the epilogue."""
    return (
        torch.empty((1, hidden_size // 8), dtype=torch.int64, device=device),
        torch.empty(16, dtype=torch.uint8, device=device),
    )


__all__ = [
    "TopkReduceAddRMSNormKernel",
    "TopkReduceSharedAddKernel",
    "megamoe_topk_reduce_add_rmsnorm",
    "megamoe_topk_reduce_add_rmsnorm_matches_flashinfer",
    "megamoe_topk_reduce_shared_add",
]
